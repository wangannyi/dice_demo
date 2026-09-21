"""Offline registration for a hand marker and a fixed tabletop board.

T_A_B maps coordinates from frame B into frame A. The hand marker is never
called a board here. Camera motion cancels in each same-frame pair before the
existing eye-to-hand solver is called. No device, SDK, configuration writing,
physical palm estimate, or motion command is provided by this module.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path

import cv2
import numpy as np


POSE_SCHEMA = 'rgb_hand_marker_ippe_square_candidates_v1'
DATASET_SCHEMA = 'rgb_marker_registration_dataset_v1'
RESULT_SCHEMA = 'rgb_marker_registration_result_v1'
RECOVERY_SCHEMA = 'rgb_table_board_recovery_candidate_v1'


@lru_cache(maxsize=1)
def _core():
    """Reuse the unchanged solver in a repository or isolated RGB deployment."""
    root = Path(__file__).resolve().parent
    for path in (root/'nero_calibration/core.py', root.parent/'nero_calibration/core.py'):
        if path.is_file():
            spec = importlib.util.spec_from_file_location('_rgb_marker_nero_core', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError('Missing unchanged nero_calibration/core.py dependency')


def _number(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Invalid '+name)
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid '+name) from exc
    if not np.isfinite(value):
        raise ValueError('Invalid '+name)
    return value


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing '+name)
    return value


def _source(value):
    if not isinstance(value, dict):
        raise ValueError('Missing same-frame source identity')
    serial = value.get('frame_serial')
    if not isinstance(serial, int) or isinstance(serial, bool) or serial < 1:
        raise ValueError('Invalid source frame_serial')
    timestamp = _number(value.get('timestamp_s'), 'source timestamp_s')
    if timestamp < 0:
        raise ValueError('Invalid source timestamp_s')
    epoch = _label(value.get('camera_epoch'), 'source camera_epoch')
    return {'frame_serial': serial, 'timestamp_s': timestamp, 'camera_epoch': epoch}


def ippe_square_candidates(raw_corners_px, camera_matrix, distortion_coefficients, *,
                           marker_length_m=.03, marker_id=40, source=None):
    """Return both IPPE branches from raw, unsmoothed ArUco detector corners.

    Corner order is top-left, top-right, bottom-right, bottom-left in the
    decoded marker. Marker +X points right and +Y up on its printed face;
    +Z points out of that face. Candidate indices preserve OpenCV ordering.
    Lower reprojection error is recorded, never treated as branch verification.
    The caller supplies the actual black-border side length and matching K/D.
    """
    corners = np.asarray(raw_corners_px, dtype=float)
    K = np.asarray(camera_matrix, dtype=float)
    D = np.asarray(distortion_coefficients, dtype=float)
    length = _number(marker_length_m, 'marker_length_m')
    if not 0 < length < 1:
        raise ValueError('Invalid marker_length_m')
    if not isinstance(marker_id, int) or isinstance(marker_id, bool) or not 0 <= marker_id < 50:
        raise ValueError('Invalid DICT_4X4_50 marker_id')
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError('Expected four finite raw marker corners')
    edges = np.roll(corners, -1, axis=0)-corners
    cross = edges[:, 0]*np.roll(edges[:, 1], -1)-edges[:, 1]*np.roll(edges[:, 0], -1)
    if not ((cross > 1e-7).all() or (cross < -1e-7).all()):
        raise ValueError('Raw marker corners must form a nondegenerate convex quadrilateral')
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-12, rtol=0)
            or not np.allclose([K[0, 1], K[1, 0]], 0, atol=1e-12, rtol=0)
            or D.shape != (5,) or not np.isfinite(D).all()):
        raise ValueError('Expected finite RGB K and five OpenCV Brown distortion coefficients')
    half = length/2
    objects = np.array([[-half, half, 0], [half, half, 0],
                        [half, -half, 0], [-half, -half, 0]], dtype=float)
    result = cv2.solvePnPGeneric(objects, corners, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not result[0] or len(result[1]) != 2:
        raise ValueError('IPPE_SQUARE did not provide both pose candidates')
    candidates = []
    for index, (rvec, tvec) in enumerate(zip(result[1], result[2])):
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            raise ValueError('Nonfinite IPPE pose candidate')
        transform = np.eye(4)
        transform[:3, :3] = cv2.Rodrigues(rvec)[0]
        transform[:3, 3] = np.asarray(tvec).reshape(3)
        depths = (objects @ transform[:3, :3].T+transform[:3, 3])[:, 2]
        reprojection = cv2.projectPoints(objects, rvec, tvec, K, D)[0].reshape(4, 2)
        if not np.isfinite(reprojection).all():
            raise ValueError('Nonfinite IPPE candidate reprojection')
        errors = np.linalg.norm(reprojection-corners, axis=1)
        candidates.append({
            'candidate_index': index, 'T_camera_hand_marker': transform.tolist(),
            'all_corners_positive_depth': bool((depths > 0).all()),
            'corner_depths_m': depths.tolist(),
            'printed_face_toward_camera': bool(
                np.dot(transform[:3, 2], -transform[:3, 3]) > 0),
            'reprojected_corners_px': reprojection.tolist(),
            'corner_reprojection_errors_px': errors.tolist(),
            'reprojection_rms_px': float(np.sqrt(np.mean(errors**2))),
            'maximum_reprojection_error_px': float(errors.max())})
    return {
        'schema': POSE_SCHEMA, 'dictionary': 'DICT_4X4_50', 'marker_id': marker_id,
        'marker_length_m': length, 'raw_corners_px': corners.tolist(),
        'corner_order': 'top_left_top_right_bottom_right_bottom_left',
        'marker_axes': '+X right, +Y up, +Z out of printed face',
        'camera_matrix': K.tolist(), 'distortion_coefficients': D.tolist(),
        'source': None if source is None else _source(source), 'candidates': candidates,
        'selected_candidate_index': None, 'physical_branch_verified': False,
        'motion_target_valid': False}


def _selected_pose(report, index, source, marker):
    if not isinstance(report, dict) or report.get('schema') != POSE_SCHEMA:
        raise ValueError('Expected explicit hand-marker candidate schema')
    if (not isinstance(index, int) or isinstance(index, bool) or index not in (0, 1)):
        raise ValueError('Explicit selected_marker_pose_index 0 or 1 is required')
    candidates = report.get('candidates')
    if (not isinstance(candidates, list) or len(candidates) != 2
            or not all(isinstance(c, dict) for c in candidates)
            or [c.get('candidate_index') for c in candidates] != [0, 1]):
        raise ValueError('Both indexed hand-marker pose candidates are required')
    if _source(report.get('source')) != _source(source):
        raise ValueError('Table board and hand marker must use the same-frame source')
    if (report.get('dictionary') != marker['dictionary']
            or report.get('marker_id') != marker['marker_id']
            or not np.isclose(_number(report.get('marker_length_m'), 'marker_length_m'),
                              marker['marker_length_m'], atol=1e-12, rtol=0)):
        raise ValueError('Hand-marker specification mismatch')
    selected = candidates[index]
    if selected.get('all_corners_positive_depth') is not True:
        raise ValueError('Selected hand-marker branch is not positive-depth')
    if _number(selected.get('reprojection_rms_px'), 'selected reprojection error') < 0:
        raise ValueError('Invalid selected reprojection error')
    return _core().matrix(selected.get('T_camera_hand_marker'))


def _marker(value):
    if not isinstance(value, dict) or value.get('dictionary') != 'DICT_4X4_50':
        raise ValueError('Expected explicit DICT_4X4_50 hand-marker specification')
    marker_id = value.get('marker_id')
    if not isinstance(marker_id, int) or isinstance(marker_id, bool) or not 0 <= marker_id < 50:
        raise ValueError('Invalid hand marker_id')
    length = _number(value.get('marker_length_m'), 'marker_length_m')
    if not 0 < length < 1:
        raise ValueError('Invalid marker_length_m')
    return {'dictionary': 'DICT_4X4_50', 'marker_id': marker_id, 'marker_length_m': length}


def solve_registration(dataset, *, max_position_m=.005, max_angle_deg=2.):
    """Solve fixed tabletop-board and flange-marker transforms offline.

    Dataset schema is DATASET_SCHEMA. It requires a fixed table_board_frame_id,
    a marker_attachment_epoch, a hand_marker specification, and samples with
    T_base_flange, T_camera_table_board, hand_marker_pose_candidates, an explicit
    selected_marker_pose_index, source identity, and stationary=True. Stationary
    FK and same-frame acquisition are caller responsibilities; this module only
    checks their recorded contract. Tableboard motion during collection is not
    permitted, although camera motion between frames is permitted.

    The unchanged solver receives T_board_marker as its T_camera_board input
    and identity TCP. Its T_base_camera becomes T_base_board; its T_tcp_board
    becomes T_flange_marker. Its 12-sample, two-axis and residual gates remain.
    """
    if not isinstance(dataset, dict) or dataset.get('schema') != DATASET_SCHEMA:
        raise ValueError('Expected explicit RGB hand-marker registration dataset schema')
    board_frame = _label(dataset.get('table_board_frame_id'), 'table_board_frame_id')
    attachment = _label(dataset.get('marker_attachment_epoch'), 'marker_attachment_epoch')
    if dataset.get('table_board_fixed_during_collection') is not True:
        raise ValueError('Table board must stay fixed during registration collection')
    marker = _marker(dataset.get('hand_marker'))
    samples = dataset.get('samples')
    if not isinstance(samples, list) or len(samples) < 12:
        raise ValueError('Need at least 12 distinct stationary poses, including held-out poses')
    core = _core()
    adapted, poses, identities, selected_indices = [], [], set(), []
    last_timestamp = None
    for sample in samples:
        if not isinstance(sample, dict) or sample.get('stationary') is not True:
            raise ValueError('Every registration sample must be recorded stationary')
        source = _source(sample.get('source'))
        identity = (source['camera_epoch'], source['frame_serial'])
        if identity in identities:
            raise ValueError('Duplicate source frame in registration dataset')
        if last_timestamp is not None and source['timestamp_s'] <= last_timestamp:
            raise ValueError('Registration source timestamps must be strictly forward')
        identities.add(identity)
        last_timestamp = source['timestamp_s']
        bf = core.matrix(sample.get('T_base_flange'))
        if any(p < .005 and a < 3 for p, a in (core.distance(bf, previous) for previous in poses)):
            raise ValueError('Duplicate stationary pose: samples must be 5 mm or 3 deg apart')
        cb = core.matrix(sample.get('T_camera_table_board'))
        index = sample.get('selected_marker_pose_index')
        cm = _selected_pose(sample.get('hand_marker_pose_candidates'), index, source, marker)
        poses.append(bf)
        selected_indices.append(index)
        adapted.append({'T_base_flange': bf.tolist(),
                        'T_camera_board': (core.inverse(cb) @ cm).tolist()})
    result = core.solve(adapted, np.eye(4), max_position_m, max_angle_deg)
    return {
        'schema': RESULT_SCHEMA, 'mode': 'fixed_table_board_with_hand_marker',
        'quality_passed': result['quality_passed'],
        'table_board_frame_id': board_frame, 'marker_attachment_epoch': attachment,
        'hand_marker': marker, 'T_base_board': result['T_base_camera'],
        'T_flange_marker': result['T_tcp_board'],
        'transform_directions': {
            'T_base_board': 'tabletop printed-board coordinates into arm base',
            'T_flange_marker': 'hand-marker coordinates into flange'},
        'selected_marker_pose_indices': selected_indices,
        'rotation_diversity_singular_values': result['rotation_diversity_singular_values'],
        'holdout_indices': result['holdout_indices'],
        'holdout_errors_m_deg': result['holdout_errors_m_deg'],
        'all_sample_errors_m_deg': result['all_sample_errors_m_deg'],
        'thresholds': result['thresholds'], 'distinct_pose_count': len(poses),
        'solver_adapter': 'unchanged nero_calibration.core.solve; identity TCP; T_board_marker',
        'physical_branch_verified': False, 'independent_validation_passed': False,
        'T_marker_contact': None, 'physical_palm_m': None,
        'motion_target_valid': False, 'execution_enabled': False}


def recover_table_board(registration, *, T_base_flange, T_camera_table_board,
                        hand_marker_pose_candidates, selected_marker_pose_index, source,
                        marker_attachment_epoch, table_board_frame_id, stationary):
    """Compute a moved table-board candidate from one stationary same-frame pair.

    B_board = B_flange @ flange_marker @ inv(camera_marker) @ camera_board.
    Existing flange-marker registration and its marker attachment must hold.
    The result always requires independent verification and is not a target.
    """
    if (not isinstance(registration, dict) or registration.get('schema') != RESULT_SCHEMA
            or registration.get('quality_passed') is not True):
        raise ValueError('A quality-passed hand-marker registration is required')
    attachment = _label(marker_attachment_epoch, 'marker_attachment_epoch')
    if attachment != registration.get('marker_attachment_epoch'):
        raise ValueError('Hand marker attachment changed; flange-marker registration is stale')
    if stationary is not True:
        raise ValueError('Recovery requires a recorded stationary flange pose')
    frame_id = _label(table_board_frame_id, 'table_board_frame_id')
    marker = _marker(registration.get('hand_marker'))
    cm = _selected_pose(hand_marker_pose_candidates, selected_marker_pose_index, source, marker)
    core = _core()
    bf = core.matrix(T_base_flange)
    fm = core.matrix(registration.get('T_flange_marker'))
    cb = core.matrix(T_camera_table_board)
    recovered = core.matrix(bf @ fm @ core.inverse(cm) @ cb)
    return {
        'schema': RECOVERY_SCHEMA, 'status': 'requires_independent_validation',
        'T_base_board': recovered.tolist(), 'table_board_frame_id': frame_id,
        'previous_table_board_frame_id': registration.get('table_board_frame_id'),
        'marker_attachment_epoch': attachment, 'hand_marker': marker,
        'selected_marker_pose_index': selected_marker_pose_index, 'source': _source(source),
        'physical_branch_verified': False, 'independent_validation_passed': False,
        'T_marker_contact': None, 'physical_palm_m': None,
        'motion_target_valid': False, 'execution_enabled': False}
