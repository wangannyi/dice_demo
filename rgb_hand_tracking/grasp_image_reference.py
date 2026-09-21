"""Offline audit of a stopped pregrasp image and a newly visible cup reference.

This pairs saved images in one fixed view. It does not establish a palm contact
transform, select an IPPE branch, extrapolate to a moved cup or enable motion.
No camera, CAN, SDK or robot access is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from board_rgb import BoardRgbObserver


SCHEMA = 'usb_grasp_image_teach_v1'
THRESHOLDS = {
    'required_RGB_frames_per_dataset': 12,
    'minimum_valid_cup_frames': 8,
    'maximum_cup_center_deviation_m': .002,
    'maximum_cup_radius_deviation_m': .002,
    'minimum_board_frames_per_corner_per_dataset': 8,
    'median_board_corner_drift_px': .75,
    'maximum_board_corner_drift_px': 1.5,
    'maximum_recorded_tag_corner_deviation_px': 1.5,
    'maximum_win2_tag_replay_difference_px': .5,
}
FALSE_FLAGS = {
    'physical_registration_valid': False,
    'physical_contact_verified': False,
    'physical_branch_verified': False,
    'physical_palm_transform_valid': False,
    'motion_target_valid': False,
    'execution_enabled': False,
    'reference_activation_valid': False,
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _finite(value, shape, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid '+label) from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Invalid '+label)
    return result


def _number(value, label):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not np.isfinite(value)):
        raise ValueError('Invalid '+label)
    return float(value)


def _stats(values):
    a = np.asarray(values, dtype=float)
    return {'count': int(a.size), 'minimum': float(a.min()),
            'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)),
            'maximum': float(a.max())}


def _raw_tag_observation(frame):
    """Fresh ID40 corners with the collector's pinned Win2; no filter/state."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    factory = getattr(cv2.aruco, 'DetectorParameters_create', None)
    params = factory() if factory else cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 2
    if hasattr(params, 'relativeCornerRefinmentWinSize'):
        params.relativeCornerRefinmentWinSize = 1.0
    if hasattr(cv2.aruco, 'ArucoDetector'):
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=params)
    selected = ([] if ids is None else
                [c for c, marker_id in zip(corners, np.asarray(ids).ravel()) if marker_id == 40])
    if len(selected) != 1:
        raise ValueError('Fresh unique ID40 required in each source PNG')
    return _finite(np.asarray(selected[0]).reshape(4, 2), (4, 2), 'replayed raw tag corners')


def _load_dataset(path, purpose):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    data = json.loads(raw)
    if (data.get('schema') != SCHEMA or data.get('kind') != SCHEMA
            or data.get('purpose') != purpose):
        raise ValueError('Wrong dataset kind or purpose: '+str(path))
    for key in ('execution_enabled', 'motion_target_valid', 'physical_palm_transform_valid',
                'reference_activation_valid'):
        if data.get(key) is not False:
            raise ValueError('Source dataset must remain inactive: '+key)
    for key in ('physical_registration_valid', 'physical_contact_verified', 'physical_branch_verified'):
        if data.get(key, False) is not False:
            raise ValueError('Image-only pairing cannot accept '+key)
    if data.get('T_marker_contact') is not None:
        raise ValueError('Image-only pairing cannot accept a contact transform')
    if data.get('selected_marker_pose_index') is not None:
        raise ValueError('Image-only pairing cannot accept a selected physical branch')
    if purpose == 'grasp_pose_record' and (data.get('pose_record_valid') is not True
                                          or data.get('operator_pregrasp_pose_confirmed') is not True):
        raise ValueError('An operator-confirmed stopped pregrasp pose is required')
    samples = data.get('samples')
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError('Exactly one stopped sample per dataset required')
    sample = samples[0]
    if (sample.get('stationary') is not True or sample.get('physical_branch_verified') is not False
            or sample.get('selected_marker_pose_index') is not None
            or sample.get('T_marker_contact') is not None
            or sample.get('motion_target_valid') is not False):
        raise ValueError('Sample must be stopped with no physical branch or motion target')
    return path, raw, data, sample


def _replay(path, data, sample, board):
    images, rows = sample.get('images'), sample.get('repeated_rgb_observations')
    required = THRESHOLDS['required_RGB_frames_per_dataset']
    if not isinstance(images, list) or not isinstance(rows, list) or len(images) != required or len(rows) != required:
        raise ValueError('Need all twelve saved RGB images and observations')
    root, sources, board_points, tag_points, tag_differences = path.parent, [], {}, [], []
    previous_time = previous_serial = generation = camera_epoch = None
    paths = set()
    sizes = set()
    for index, (image, row) in enumerate(zip(images, rows)):
        relative = Path(image['path'])
        png = (root/relative).resolve(strict=True)
        if relative.is_absolute() or not png.is_relative_to(root) or png.suffix.lower() != '.png' or png in paths:
            raise ValueError('Source PNG path must be unique and contained in its dataset')
        paths.add(png)
        raw = png.read_bytes()
        digest = _sha(raw)
        if digest != image.get('sha256'):
            raise ValueError('Source PNG SHA256 mismatch: '+str(png))
        source = row.get('source', {})
        if source.get('image_path') != image['path'] or source.get('image_sha256') != digest:
            raise ValueError('Observation source image identity mismatch')
        if row.get('hand_marker_pose_candidates', {}).get('source') != source:
            raise ValueError('IPPE source identity differs from its own RGB image')
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or [frame.shape[1], frame.shape[0]] != [image.get('width'), image.get('height')]:
            raise ValueError('Source PNG dimensions mismatch')
        size = (int(frame.shape[1]), int(frame.shape[0]))
        sizes.add(size)
        timestamp = _number(source.get('timestamp_s'), 'source timestamp')
        serial, epoch, current_generation = source.get('frame_serial'), source.get('camera_epoch'), row.get('tracking_generation')
        if (type(serial) is not int or serial < 1 or type(current_generation) is not int
                or current_generation < 0 or not isinstance(epoch, str)
                or not epoch.startswith(data['camera_configuration_epoch']+':')
                or (previous_time is not None and (timestamp <= previous_time or serial <= previous_serial))
                or (generation is not None and current_generation != generation)
                or (camera_epoch is not None and epoch != camera_epoch)):
            raise ValueError('Source timestamps, serials or capture continuity mismatch')
        previous_time, previous_serial, generation, camera_epoch = timestamp, serial, current_generation, epoch
        raw_tag = _finite(row.get('raw_marker_corners_px'), (4, 2), 'recorded raw tag corners')
        candidate_tag = _finite(row['hand_marker_pose_candidates'].get('raw_corners_px'), (4, 2), 'IPPE raw tag corners')
        if not np.array_equal(raw_tag, candidate_tag):
            raise ValueError('Recorded raw tag differs from IPPE source corners')
        replay_tag = _raw_tag_observation(frame)
        differences = np.linalg.norm(replay_tag-raw_tag, axis=1)
        if differences.max() > THRESHOLDS['maximum_win2_tag_replay_difference_px']:
            raise ValueError('Win2 replay disagrees with captured raw tag corners')
        tag_points.append(raw_tag)
        tag_differences.extend(differences.tolist())
        observation = board.observe(frame)
        if observation.get('valid') is not True or observation.get('corner_convention') != 'opencv_4_6':
            raise ValueError('Every RGB image requires valid opencv_4_6 board replay')
        ids = observation['charuco_corner_ids']
        points = _finite(observation['charuco_corners_px'], (len(ids), 2), 'board replay corners')
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate replayed board corner IDs')
        for corner_id, point in zip(ids, points):
            board_points.setdefault(int(corner_id), []).append(point)
        sources.append({'rgb_index': index, 'source': dict(source), 'image_size': list(size),
                        'PNG_SHA256_verified': True, 'board_corner_ids': list(ids),
                        'board_corners_px': points.tolist(),
                        'replayed_raw_tag_corners_win2_px': replay_tag.tolist(),
                        'maximum_tag_replay_difference_px': float(differences.max())})
    if len(sizes) != 1:
        raise ValueError('Image size changed within a dataset')
    tags = np.asarray(tag_points)
    median = np.median(tags, axis=0)
    deviations = np.linalg.norm(tags-median, axis=2)
    if deviations.max() > THRESHOLDS['maximum_recorded_tag_corner_deviation_px']:
        raise ValueError('Stopped raw tag corners are unstable')
    if not np.allclose(median, _finite(sample.get('marker_corner_statistics', {}).get('median_px'),
                                     (4, 2), 'stored tag median'), rtol=0, atol=1e-8):
        raise ValueError('Stored raw tag median disagrees with original observations')
    return {'source_rows': sources, 'board_points': board_points, 'tag_median': median,
            'image_size': list(next(iter(sizes))), 'tag_scatter_px': _stats(deviations),
            'tag_replay_difference_px': _stats(tag_differences), 'rows': rows}


def _cup_reference(rows):
    valid = [row for row in rows if row.get('cup_metric', {}).get('valid') is True]
    if len(valid) < THRESHOLDS['minimum_valid_cup_frames']:
        raise ValueError('Need at least eight fresh valid cup-top metric observations')
    centers, radii, pixels, source_indices = [], [], [], []
    for row in valid:
        top, metric = row.get('cup_top', {}), row['cup_metric']
        if (top.get('valid') is not True or top.get('source') != 'green_top_geometry'
                or metric.get('source') != 'undistorted_edge_rays_height_plane_circle_fit'):
            raise ValueError('Cup must use supported internal top edge, not green body diagnostics')
        center = _finite(metric.get('center_board_m'), (3,), 'cup center')
        radius = _number(metric.get('radius_m'), 'cup radius')
        if not .005 < radius < .1:
            raise ValueError('Invalid fitted cup radius')
        centers.append(center)
        radii.append(radius)
        pixels.append(_finite(top.get('center_px'), (2,), 'cup top center pixels'))
        source_indices.append(next(i for i, candidate in enumerate(rows) if candidate is row))
    centers, radii, pixels = np.asarray(centers), np.asarray(radii), np.asarray(pixels)
    center_median, radius_median, pixel_median = np.median(centers, axis=0), float(np.median(radii)), np.median(pixels, axis=0)
    center_scatter = np.linalg.norm(centers-center_median, axis=1)
    radius_scatter = np.abs(radii-radius_median)
    if (center_scatter.max() > THRESHOLDS['maximum_cup_center_deviation_m']
            or radius_scatter.max() > THRESHOLDS['maximum_cup_radius_deviation_m']):
        raise ValueError('Current cup center or radius is unstable')
    return {'valid_current_cup_frames': len(valid), 'valid_source_rgb_indices': source_indices,
            'median_center_px': pixel_median.tolist(), 'median_center_board_candidate_m': center_median.tolist(),
            'median_fitted_radius_m': radius_median,
            'center_deviation_m': _stats(center_scatter), 'radius_deviation_m': _stats(radius_scatter),
            'pixel_center_deviation_px': _stats(np.linalg.norm(pixels-pixel_median, axis=1)),
            'metric_accuracy_independently_validated': False,
            'physical_contact_verified': False, 'motion_target_valid': False}


def pair_image_reference(grasp_dataset, cup_dataset, board_config):
    """Verify both original image sets; return an inactive fixed-view image pair."""
    grasp_path, grasp_raw, grasp, grasp_sample = _load_dataset(grasp_dataset, 'grasp_pose_record')
    cup_path, cup_raw, cup, cup_sample = _load_dataset(cup_dataset, 'before_contact')
    identities = ('intrinsics_sha256', 'marker_attachment_epoch', 'camera_configuration_epoch')
    for key in identities:
        if not isinstance(grasp.get(key), str) or not grasp[key].strip() or grasp[key] != cup.get(key):
            raise ValueError('Dataset identity mismatch: '+key)
    digest = grasp['intrinsics_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid intrinsics SHA256')
    # A single observer and normalized ID convention are used for BOTH sets.
    board = BoardRgbObserver(board_config, corner_convention='opencv_4_6')
    a, b = _replay(grasp_path, grasp, grasp_sample, board), _replay(cup_path, cup, cup_sample, board)
    if a['image_size'] != b['image_size']:
        raise ValueError('Image size changed between datasets')
    minimum = THRESHOLDS['minimum_board_frames_per_corner_per_dataset']
    common = sorted(i for i in a['board_points']
                    if len(a['board_points'][i]) >= minimum and len(b['board_points'].get(i, [])) >= minimum)
    if len(common) < board.minimum:
        raise ValueError('Insufficient repeated common board corner IDs')
    corners, drifts = [], []
    for corner_id in common:
        points_a, points_b = np.asarray(a['board_points'][corner_id]), np.asarray(b['board_points'][corner_id])
        median_a, median_b = np.median(points_a, axis=0), np.median(points_b, axis=0)
        drift = float(np.linalg.norm(median_a-median_b))
        drifts.append(drift)
        corners.append({'corner_id': corner_id, 'grasp_frames': len(points_a), 'cup_frames': len(points_b),
                        'grasp_median_px': median_a.tolist(), 'cup_median_px': median_b.tolist(), 'median_drift_px': drift,
                        'grasp_deviation_px': _stats(np.linalg.norm(points_a-median_a, axis=1)),
                        'cup_deviation_px': _stats(np.linalg.norm(points_b-median_b, axis=1))})
    if (np.median(drifts) > THRESHOLDS['median_board_corner_drift_px']
            or max(drifts) > THRESHOLDS['maximum_board_corner_drift_px']):
        raise ValueError('Camera/tabletop board view moved between datasets')
    current_cup = _cup_reference(b['rows'])
    tag = a['tag_median']
    if not np.allclose(tag, _finite(grasp.get('reference_marker_corners_px'), (4, 2), 'grasp reference corners'),
                       rtol=0, atol=1e-8):
        raise ValueError('Grasp reference differs from captured raw tag median')
    source_records = []
    for path, raw, data, replay in ((grasp_path, grasp_raw, grasp, a), (cup_path, cup_raw, cup, b)):
        source_records.append({'dataset_path': str(path), 'dataset_SHA256': _sha(raw),
                               'purpose': data['purpose'], 'original_capture_source_SHA256': data.get('source_sha256', {}),
                               'all_original_RGB_sources': replay['source_rows'],
                               'tag_scatter_px': replay['tag_scatter_px'], 'win2_tag_replay_difference_px': replay['tag_replay_difference_px']})
    return {'schema': 'fixed_view_grasp_image_pair_v1', 'kind': 'fixed_view_grasp_image_pair_v1',
            'image_pair_valid': True, 'fixed_view_valid': True, **FALSE_FLAGS,
            'T_marker_contact': None, 'selected_marker_pose_index': None,
            'identities': {key: grasp[key] for key in identities}, 'image_size': a['image_size'],
            'thresholds': dict(THRESHOLDS), 'sources': source_records,
            'board_replay': {'corner_convention': 'opencv_4_6', 'board_configuration': board.cfg,
                             'common_repeated_corner_count': len(common), 'minimum_common_corner_count': board.minimum,
                             'corner_median_drift_px': _stats(drifts), 'corners': corners},
            'current_cup_reference': current_cup,
            'goal': {'definition': 'captured_grasp_raw_tag_median_minus_new_visible_cup_top_pixel_median',
                     'raw_tag_corners_px': tag.tolist(), 'cup_top_center_px': current_cup['median_center_px'],
                     'relative_corners_8D_px': (tag-np.asarray(current_cup['median_center_px'])).reshape(8).tolist(),
                     'feature_order': 'corner0_x,corner0_y,corner1_x,corner1_y,corner2_x,corner2_y,corner3_x,corner3_y',
                     'corner_order': 'decoded_top_left_top_right_bottom_right_bottom_left',
                     'detector_contract': {'dictionary': 'DICT_4X4_50', 'marker_id': 40,
                                           'cornerRefinementMethod': 'CORNER_REFINE_SUBPIX',
                                           'cornerRefinementWinSize': 2, 'relativeCornerRefinmentWinSize_when_available': 1.0,
                                           'temporal_filter': False, 'Win5_comparison_allowed': False},
                     'reference_q_rad': grasp.get('reference_q_rad'), 'motion_target_valid': False,
                     'operator_pregrasp_image_only': True, 'future_cup_translation_extrapolation_valid': False},
            'software': {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__,
                         'audit_source_SHA256': _sha(Path(__file__).read_bytes()),
                         'board_observer_source_SHA256': _sha(Path(__file__).with_name('board_rgb.py').read_bytes()),
                         'capture_opencv_version': 'not recorded in these source datasets'},
            'limitations': [
                'Fixed-view local image association, not a physical palm point or verified cup contact.',
                'Board pixel invariance verifies relative camera/board view; coordinated camera and board motion is not excluded.',
                'Matching attachment epochs are operator assertions; tag rigidity and cup immobility between captures are not independently proven.',
                'Raw tag goal requires the pinned Win2 detector; a default Win5 running detector cannot be compared directly.',
                'No IPPE branch is selected and original 3D registration gates remain unchanged.',
                'Do not translate this target to a future cup position; marker and cup heights can differ.',
                'This audit replays board/tag pixels and validates saved cup metrics; cup segmentation/metric fit is not independently rerun.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grasp-dataset', type=Path, required=True)
    parser.add_argument('--cup-dataset', type=Path, required=True)
    parser.add_argument('--board-config', type=Path,
                        default=Path(__file__).resolve().parent/'config/usb7_relative_grasp.json')
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; never overwrite existing files')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new file')
    config_raw = args.board_config.read_bytes()
    result = pair_image_reference(args.grasp_dataset, args.cup_dataset, json.loads(config_raw))
    result['board_configuration_source'] = {'path': str(args.board_config.resolve()), 'SHA256': _sha(config_raw)}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output.resolve()), 'image_pair_valid': True,
                      'fixed_view_valid': True, **FALSE_FLAGS}))


if __name__ == '__main__':
    main()
