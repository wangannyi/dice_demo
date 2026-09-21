"""Stopped-pose image reference for manual cup-rim teaching; zero CAN TX.

This separate dataset does not relax the rigid-registration collector's pose
repeatability gates. Unstable IPPE poses remain hypotheses. No configuration,
physical palm transform, handeye calibration or motion target is published.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from board_rgb import BoardRgbObserver
from camera_profile import CameraBrightnessProfile
from cup_top import CupTopDetector
from marker_frontend import MarkerRgbCamera
import marker_registration as registration
from rgb_geometry import CalibratedRgbGeometry
from usb_marker_registration_capture import (
    PassivePoseClient, checked_snapshot, commit_sample, fixed_window_tracker,
    stationarity_evidence)


def cup_rim_target(cup_geometry, T_camera_board, K, D):
    """A unique fitted-rim point: smallest projected image Y, not body centroid."""
    if cup_geometry.get('valid') is not True:
        raise ValueError('Fresh valid cup-top geometry required')
    center = np.asarray(cup_geometry.get('center_board_m'), float)
    radius = cup_geometry.get('radius_m')
    T = np.asarray(T_camera_board, float)
    if (center.shape != (3,) or not np.isfinite(center).all()
            or isinstance(radius, bool) or not isinstance(radius, (int, float))
            or not np.isfinite(radius) or not .005 < radius < .1
            or T.shape != (4, 4) or not np.isfinite(T).all()):
        raise ValueError('Invalid fitted cup geometry')
    theta = np.linspace(0, 2*np.pi, 1440, endpoint=False)
    points = center+np.column_stack((radius*np.cos(theta), radius*np.sin(theta),
                                    np.zeros(len(theta))))
    camera_points = points@T[:3, :3].T+T[:3, 3]
    if (camera_points[:, 2] <= 0).any():
        raise ValueError('Cup rim behind camera')
    pixels = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    index = int(np.argmin(pixels[:, 1]))
    return {'definition': 'cup_top_fitted_rim_minimum_image_y',
            'contact_point_board_candidate_m': points[index].tolist(),
            'contact_point_px': pixels[index].tolist(),
            'cup_center_board_m': center.tolist(), 'fitted_radius_m': float(radius),
            'independent_metric_accuracy_validated': False,
            'motion_target_valid': False}


def same_source(frame, timestamp, serial, marker, generation, epoch):
    expected = {'capture_timestamp_s': timestamp, 'timestamp_s': timestamp,
                'frame_serial': serial, 'camera_epoch': epoch,
                'tracking_generation': generation, 'marker_id': 40,
                'width': frame.shape[1], 'height': frame.shape[0]}
    if (any(marker.get(key) != value for key, value in expected.items())
            or marker.get('observation_valid') is not True
            or marker.get('raw_corners_px') is None):
        raise ValueError('Confirmed same-frame current hand marker required')


def cup_fixed_verification(rows, target):
    valid = [row['cup_metric'] for row in rows if row['cup_metric'].get('valid') is True]
    result = {'verified': False, 'valid_current_cup_frames': len(valid),
              'minimum_frames': 6, 'center_tolerance_m': .002, 'radius_tolerance_m': .002,
              'reason': 'cup_occluded_or_insufficient_fresh_geometry',
              'maximum_center_change_m': None, 'maximum_radius_change_m': None}
    if not valid:
        return result
    centers = np.asarray([row['center_board_m'] for row in valid], float)
    radii = np.asarray([row['radius_m'] for row in valid], float)
    if not np.isfinite(centers).all() or not np.isfinite(radii).all():
        raise ValueError('Invalid current cup geometry')
    center_change = float(np.linalg.norm(centers-np.asarray(target['cup_center_board_m']), axis=1).max())
    radius_change = float(np.abs(radii-target['fitted_radius_m']).max())
    result.update(maximum_center_change_m=center_change, maximum_radius_change_m=radius_change)
    if center_change > .002 or radius_change > .002:
        raise ValueError('Cup moved or current cup geometry disagrees with reference')
    if len(valid) >= 6:
        result.update(verified=True, reason='fresh_geometry_agrees_with_baseline')
    return result


def current_cup_reference(rows, K, D):
    """A new visible cup reference; occlusion preserves pose without old cup reuse."""
    valid = [row for row in rows if row['cup_metric'].get('valid') is True]
    result = {'valid': False, 'valid_current_cup_frames': len(valid), 'minimum_frames': 8,
              'reason': 'insufficient_current_cup_top_geometry', 'rim_target': None,
              'physical_contact_verified': False, 'motion_target_valid': False}
    if len(valid) < 8:
        return result
    centers = np.asarray([row['cup_metric']['center_board_m'] for row in valid], float)
    radii = np.asarray([row['cup_metric']['radius_m'] for row in valid], float)
    if not np.isfinite(centers).all() or not np.isfinite(radii).all():
        raise ValueError('Invalid current cup geometry')
    center_scatter = float(np.linalg.norm(centers-np.median(centers, axis=0), axis=1).max())
    radius_scatter = float(np.abs(radii-np.median(radii)).max())
    result.update(maximum_center_deviation_m=center_scatter,
                  maximum_radius_deviation_m=radius_scatter)
    if center_scatter > .002 or radius_scatter > .002:
        result['reason'] = 'unstable_current_cup_top_geometry'
        return result
    row = min(valid, key=lambda row: row['table_board_pose']['reprojection_rms_px'])
    target = cup_rim_target(row['cup_metric'], row['T_camera_table_board'], K, D)
    target.update(source=deepcopy(row['source']), reference_rgb_index=rows.index(row))
    result.update(valid=True, reason='fresh_current_cup_top_reference', rim_target=target)
    return result


def grasp_pose_fields(sample, K, D):
    cup = current_cup_reference(sample['repeated_rgb_observations'], K, D)
    return {'pose_record_valid': True, 'operator_pregrasp_pose_confirmed': True,
            'physical_contact_verified': False, 'physical_branch_verified': False,
            'T_marker_contact': None, 'reference_activation_valid': False,
            'motion_target_valid': False, 'execution_enabled': False,
            'cup_reference_valid': cup['valid'], 'current_cup_reference': cup,
            'rim_target': cup['rim_target'], 'rim_target_is_physical_contact_goal': False,
            'previous_cup_target_reused': False,
            'reference_marker_corners_px': sample['marker_corner_statistics']['median_px'],
            'reference_q_rad': sample['software_bracket']['before']['q_rad'],
            'reference_is_fixed_camera_image_goal_only': True,
            'independent_contact_reference_validation_required': True}


def capture_reference(camera, client, board, geometry, cup, epoch, *, clock=time.monotonic,
                      sleep=time.sleep, timeout_s=15):
    deadline = clock()+timeout_s
    snapshots, rows, frames = [], [], []

    def snapshot():
        if clock() > deadline:
            raise RuntimeError('Image teaching capture timed out')
        row = client.snapshot()
        checked_snapshot(row, snapshots[-1] if snapshots else None)
        snapshots.append(row)
        return row

    for index in range(8):
        if index:
            sleep(.05)
        snapshot()
    stationarity_evidence(snapshots)
    serial, previous_time, generation = 0, None, None
    for _ in range(12):
        before = snapshot()
        while True:
            if clock() > deadline:
                raise RuntimeError('New current RGB frame timed out')
            frame, timestamp, next_serial, marker, current_generation = camera.next(serial)
            if (type(next_serial) is not int or next_serial <= serial
                    or not np.isfinite(timestamp)
                    or (previous_time is not None and timestamp <= previous_time)):
                raise ValueError('RGB source identity did not advance')
            serial, previous_time = next_serial, timestamp
            if timestamp > before['request_end_monotonic_s']:
                break
        after = snapshot()
        if not timestamp < after['request_start_monotonic_s']:
            raise ValueError('RGB outside software FK bracket')
        same_source(frame, timestamp, serial, marker, current_generation, epoch)
        if generation is not None and generation != current_generation:
            raise ValueError('Hand-marker continuity changed')
        generation = current_generation
        pair = registration._core().distance(checked_snapshot(before), checked_snapshot(after))
        if pair[0] > .0005 or pair[1] > .2:
            raise ValueError('Flange moved across the RGB capture')
        board_observation = board.observe(frame)
        board_pose, _, _ = geometry._board_pose(board_observation)
        if board_pose.get('valid') is not True:
            raise ValueError('Current valid tabletop board pose required')
        cup_observation = cup.process(frame)
        metric = geometry.observe(board_observation, cup_observation)
        source = {'frame_serial': serial, 'timestamp_s': timestamp, 'camera_epoch': epoch}
        rows.append({'source': source, 'tracking_generation': generation,
                     'raw_marker_corners_px': marker['raw_corners_px'],
                     'T_camera_table_board': board_pose['T_camera_board'],
                     'table_board_pose': board_pose, 'cup_top': cup_observation,
                     'cup_metric': metric['cup_top'],
                     'hand_marker_pose_candidates': registration.ippe_square_candidates(
                         marker['raw_corners_px'], geometry.camera_matrix, geometry.distortion,
                         marker_length_m=.03, marker_id=40, source=source),
                     'software_bracket': {'before': deepcopy(before), 'after': deepcopy(after),
                                          'hardware_synchronized': False}})
        frames.append(frame.copy())
    evidence = stationarity_evidence(snapshots)
    stats = camera.stats_snapshot()
    if (stats.get('failure_reason') is not None or stats.get('tracking_generation') != generation
            or stats.get('latest_marker_observation_valid') is not True):
        raise ValueError('Marker invalidated before capture completed')
    corners = np.asarray([row['raw_marker_corners_px'] for row in rows])
    median = np.median(corners, axis=0)
    scatter = np.linalg.norm(corners-median, axis=2)
    if not np.isfinite(corners).all() or scatter.max() > 1.5:
        raise ValueError('Stationary image corners insufficiently stable')
    chosen = min(range(len(rows)), key=lambda i: float(
        np.linalg.norm(corners[i]-median)))
    sample = deepcopy(rows[chosen])
    pose_scatter = []
    for branch in range(2):
        transforms = [registration._core().inverse(row['T_camera_table_board']) @ np.asarray(
            row['hand_marker_pose_candidates']['candidates'][branch]['T_camera_hand_marker'])
            for row in rows]
        differences = [registration._core().distance(a, b) for i, a in enumerate(transforms)
                       for b in transforms[i+1:]]
        pose_scatter.append({'candidate_index': branch,
                             'maximum_pairwise_position_m': max(v[0] for v in differences),
                             'maximum_pairwise_angle_deg': max(v[1] for v in differences),
                             'candidate_is_physical_calibration': False})
    sample.update(repeated_rgb_observations=rows, representative_rgb_index=chosen,
                  stationarity=evidence, marker_frontend=stats,
                  T_base_flange=checked_snapshot(rows[chosen]['software_bracket']['before']).tolist(),
                  marker_corner_statistics={'median_px': median.tolist(),
                                            'maximum_deviation_from_median_px': float(scatter.max())},
                  raw_ippe_repeatability=pose_scatter,
                  stationary=True, physical_branch_verified=False,
                  selected_marker_pose_index=None, T_marker_contact=None,
                  motion_target_valid=False, hardware_synchronized=False)
    return sample, frames


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--purpose', choices=('before_contact', 'contact_teach', 'grasp_pose_record'),
                        required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--marker-attachment-epoch', required=True)
    parser.add_argument('--operator-contact-confirmed', action='store_true')
    parser.add_argument('--operator-pregrasp-confirmed', action='store_true')
    parser.add_argument('--config', type=Path, default=root/'config/usb7_relative_grasp.json')
    parser.add_argument('--rgb-profile', type=Path, default=root/'config/usb7_marker_rgb_profile.json')
    parser.add_argument('--sdk-python', type=Path,
                        default=Path('/home/test2/agilex-api-test/venv/bin/python'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new directory')
    if not args.marker_attachment_epoch.strip():
        parser.error('Attachment epoch must be nonempty')
    if args.purpose == 'contact_teach' and (args.reference is None
                                          or not args.operator_contact_confirmed):
        parser.error('Contact teaching requires an existing reference and operator placement')
    if args.purpose == 'grasp_pose_record' and not args.operator_pregrasp_confirmed:
        parser.error('Grasp pose recording requires operator placement confirmation')
    config = json.loads(args.config.read_text())
    calibration_path = args.config.parent/config['camera']['intrinsics_file']
    calibration = json.loads(calibration_path.read_text())
    geometry = CalibratedRgbGeometry(config, calibration)
    profile = CameraBrightnessProfile(config['camera']['device'],
                                     json.loads(args.rgb_profile.read_text()), config['camera'])
    reference = None if args.reference is None else json.loads(args.reference.read_text())
    if reference is not None and (reference.get('purpose') != 'before_contact'
                                  or reference.get('kind') != 'usb_grasp_image_teach_v1'
                                  or reference.get('marker_attachment_epoch') != args.marker_attachment_epoch
                                  or reference.get('intrinsics_sha256') != hashlib.sha256(
                                      calibration_path.read_bytes()).hexdigest()):
        raise ValueError('Wrong baseline image reference or intrinsics')
    camera = client = None
    args.output.mkdir(parents=True)
    epoch = config['camera']['camera_epoch']+':image_teach:'+uuid.uuid4().hex
    try:
        profile.apply()
        client = PassivePoseClient([str(args.sdk_python), str(root/'passive_pose_bridge.py')])
        camera = MarkerRgbCamera(config['camera']['device'], config['camera'], camera_epoch=epoch,
                                 tracker_factory=fixed_window_tracker)
        profile.check_active()
        sample, frames = capture_reference(camera, client, BoardRgbObserver(
            config['board'], corner_convention='opencv_4_6'), geometry,
            CupTopDetector(backend='green_top'), epoch)
        sample['pose_helper_ready'] = deepcopy(client.ready)
        profile.check_active()
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                try:
                    profile.restore()
                finally:
                    (args.output/'camera_profile.json').write_text(json.dumps(
                        profile.record, indent=2, allow_nan=False)+'\n')
    sample['pose_helper_closed'] = deepcopy(client.closed_report)
    dataset = {'schema': 'usb_grasp_image_teach_v1', 'kind': 'usb_grasp_image_teach_v1',
               'purpose': args.purpose, 'samples': [], 'execution_enabled': False,
               'motion_target_valid': False, 'physical_palm_transform_valid': False,
               'reference_activation_valid': False,
               'marker_attachment_epoch': args.marker_attachment_epoch,
               'camera_configuration_epoch': config['camera']['camera_epoch'],
               'intrinsics_sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
               'scene_contract': 'cup, camera and tabletop board fixed during manual teaching',
               'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), root/'usb_marker_registration_capture.py']}}
    if args.purpose == 'before_contact':
        valid = [row for row in sample['repeated_rgb_observations'] if row['cup_metric']['valid']]
        if len(valid) < 8:
            raise ValueError('Need at least eight fresh valid cup-top observations')
        row = min(valid, key=lambda row: row['table_board_pose']['reprojection_rms_px'])
        dataset['rim_target'] = cup_rim_target(row['cup_metric'], row['T_camera_table_board'],
                                             geometry.camera_matrix, geometry.distortion)
        dataset['rim_target']['source'] = deepcopy(row['source'])
        dataset['rim_target']['reference_rgb_index'] = sample['repeated_rgb_observations'].index(row)
    elif args.purpose == 'contact_teach':
        if reference['camera_configuration_epoch'] != config['camera']['camera_epoch']:
            raise ValueError('Camera epoch changed during teaching')
        baseline = reference['samples'][0]
        delta = registration._core().distance(baseline['T_camera_table_board'],
                                             sample['T_camera_table_board'])
        if delta[0] > .005 or delta[1] > 2:
            raise ValueError('Camera/tabletop board changed during teaching')
        dataset['reference_sha256'] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
        dataset['rim_target'] = deepcopy(reference['rim_target'])
        dataset['operator_contact_placement_confirmed'] = True
        dataset['cup_fixed_verification'] = cup_fixed_verification(
            sample['repeated_rgb_observations'], reference['rim_target'])
        dataset['independent_contact_reference_validation_required'] = True
        dataset['reference_marker_corners_px'] = sample['marker_corner_statistics']['median_px']
        dataset['reference_q_rad'] = sample['software_bracket']['before']['q_rad']
        dataset['reference_is_fixed_camera_image_goal_only'] = True
    else:
        dataset.update(grasp_pose_fields(sample, geometry.camera_matrix, geometry.distortion))
        dataset['scene_contract'] = 'operator holds current pregrasp pose/cup/camera/tabletop board fixed'
        if reference is not None:
            dataset['previous_reference_sha256'] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
            dataset['previous_cup_reference_status'] = 'invalidated_after_operator_reported_relocation'
    commit_sample(args.output, dataset, sample, frames)
    print(json.dumps({'dataset': str((args.output/'dataset.json').resolve()),
                      'purpose': args.purpose, 'actual_tx_count': 0,
                      'motion_target_valid': False}))


if __name__ == '__main__':
    main()
