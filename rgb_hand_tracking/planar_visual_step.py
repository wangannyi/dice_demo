"""Pure offline local tag Jacobian and virtual-goal axis proposals; no SDK.

Columns are raw tag-centre pixel responses per actual SDK-FK base X/Y mm.
Cup and board medians diagnose scene stability; cup noise is never subtracted
from tag responses. This locally measured mapping is not contact registration.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from cartesian_microstep import _numbers, _rotation, _rotation_vector


SCHEMA = 'local_planar_raw_tag_jacobian_v1'
DETECTOR = 'fixed_subpixel_win_2_same_as_teach'
MIN_FRAMES = 8
WINDOW_FRAMES = 10
LOCAL_RADIUS_MM = 3.
MAX_CONDITION = 10.


def _base_epoch(value):
    if not isinstance(value, str) or not value:
        raise ValueError('Missing camera epoch')
    return value.split(':cartesian_probe:', 1)[0]


def _pose_distance(a, b):
    return float(np.linalg.norm(a[:3]-b[:3])), float(np.linalg.norm(
        _rotation_vector(_rotation(a) @ _rotation(b).T)))


def _bootstrap_delta(before, after, rng, count):
    a = np.median(before[rng.integers(len(before), size=(count, len(before)))], axis=1)
    b = np.median(after[rng.integers(len(after), size=(count, len(after)))], axis=1)
    return b-a


def _window(rows):
    tags = np.array([_numbers(r['marker']['raw_corners_px'], (4, 2), 'raw tag corners').mean(0)
                     for r in rows])
    cups = np.array([_numbers(r['cup_top']['center_px'], (2,), 'cup centre') for r in rows])
    ids = set.intersection(*(set(r['board']['charuco_corner_ids']) for r in rows))
    if len(ids) < 8:
        raise ValueError('Fewer than eight shared board corners')
    ids = sorted(ids)
    boards = []
    for row in rows:
        corners = _numbers(row['board']['charuco_corners_px'],
                           (len(row['board']['charuco_corner_ids']), 2), 'board corners')
        mapping = dict(zip(row['board']['charuco_corner_ids'], corners))
        boards.append([mapping[index] for index in ids])
    boards = np.array(boards)
    tag_median, cup_median, board_median = (np.median(v, axis=0) for v in (tags, cups, boards))
    tag_scatter = float(np.quantile(np.linalg.norm(tags-tag_median, axis=1), .9))
    cup_scatter = float(np.quantile(np.linalg.norm(cups-cup_median, axis=1), .9))
    board_scatter = float(np.quantile(np.linalg.norm(boards-board_median, axis=2), .9))
    if tag_scatter > .35 or cup_scatter > 3. or board_scatter > .25:
        raise ValueError('Window scatter is too large for a confident local response')
    diagnostic = {'count': len(rows), 'frame_serials': [r['frame_serial'] for r in rows],
                  'first_timestamp_s': rows[0]['timestamp_s'], 'last_timestamp_s': rows[-1]['timestamp_s'],
                  'tag_median_px': tag_median.tolist(), 'tag_scatter_p90_px': tag_scatter,
                  'cup_median_px': cup_median.tolist(), 'cup_scatter_p90_px': cup_scatter,
                  'board_scatter_p90_px': board_scatter}
    return tags, cups, dict(zip(ids, board_median)), diagnostic


def _trial(scene, sdk, axis, rng, bootstrap_samples):
    if (scene.get('success') is not True or sdk.get('success') is not True
            or sdk.get('event') != 'cartesian_microprobe_reached'
            or scene.get('axis') != axis or sdk['requested_probe']['axis'] != axis
            or scene.get('marker_corner_detector') != DETECTOR):
        raise ValueError('Successful fixed-Win2 axis experiment is required')
    verification = sdk['target_verification']
    if verification.get('fresh_stable_samples', 0) < 10 or len(verification['samples']) < 10:
        raise ValueError('Ten strictly settled SDK samples are required')
    baseline = _numbers(sdk['baseline_feedback'][-1]['fk_flange_pose_m_rad'], (6,), 'SDK before FK')
    goal = _numbers(sdk['proposal']['target_position_m'], (3,), 'SDK position goal')
    poses = []
    for sample in verification['samples'][-10:]:
        feedback, checks = sample['feedback'], sample['verification']
        pose = _numbers(feedback['fk_flange_pose_m_rad'], (6,), 'SDK settled FK')
        state = feedback['status']
        q = _numbers(feedback['q_rad'], (7,), 'SDK settled joints')
        target = _numbers(sdk['q_target_rad'], (7,), 'SDK target joints')
        if (checks.get('reached') is not True or not all(checks['checks'].values())
                or state['arm_status'] != 0 or state['ctrl_mode'] != 1 or state['motion_status'] != 0
                or feedback['enabled'] != [True]*7 or np.max(np.abs(q-target)) > math.radians(.02)
                or abs(pose[2]-baseline[2]) > .0001 or np.linalg.norm(pose[:3]-goal) > .00025
                or _pose_distance(pose, baseline)[1] > math.radians(.05)):
            raise ValueError('Settled SDK pose violates original microprobe gates')
        poses.append(pose)
    applied = (poses[-1][0 if axis == 'x' else 1]-baseline[0 if axis == 'x' else 1])*1000
    request_mm = float(sdk['requested_probe']['distance_m'])*1000
    if not math.isfinite(request_mm) or not 0 < abs(request_mm) <= 1.:
        raise ValueError('Invalid axis request')
    if applied*math.copysign(1., request_mm) < abs(request_mm)*.5:
        raise ValueError('Actual SDK-FK axis response is insufficient')
    rows, epochs, generations = [], set(), set()
    last_timestamp, last_serial = -math.inf, -1
    for row in scene['rows']:
        stamp = float(row['timestamp_s'])
        if not math.isfinite(stamp) or stamp <= last_timestamp or row['frame_serial'] <= last_serial:
            raise ValueError('Scene frame clocks/serials are not strictly advancing')
        last_timestamp, last_serial = stamp, row['frame_serial']
        epochs.add(row['camera_epoch'])
        generations.add(row['tracking_generation'])
        marker = row['marker']
        if (marker['camera_epoch'] != row['camera_epoch']
                or marker['tracking_generation'] != row['tracking_generation']
                or marker['frame_serial'] != row['frame_serial'] or marker['timestamp_s'] != stamp):
            raise ValueError('Tag does not belong to the same current RGB frame')
        if row.get('same_frame_current_tag_board_cup_valid') is True:
            age = float(row['read_age_s'])
            if (not 0 <= age <= .15 or marker.get('observation_valid') is not True
                    or row['board'].get('valid') is not True or row['cup_top'].get('valid') is not True):
                raise ValueError('Valid row contains stale or invalid observations')
            rows.append(row)
    if len(epochs) != 1 or len(generations) != 1:
        raise ValueError('Camera epoch or tracking generation changed during experiment')
    pre = [r for r in rows if r['timestamp_s'] < sdk['move_j_sent_monotonic_s']][-WINDOW_FRAMES:]
    post = [r for r in rows if r['timestamp_s'] > sdk['strict_settled_monotonic_s']+.05][-WINDOW_FRAMES:]
    if min(len(pre), len(post)) < MIN_FRAMES:
        raise ValueError('At least eight current pre-TX and post-settle frames are required')
    a, cup_a, board_a, pre_info = _window(pre)
    b, cup_b, board_b, post_info = _window(post)
    delta = np.median(b, axis=0)-np.median(a, axis=0)
    samples = _bootstrap_delta(a, b, rng, bootstrap_samples)
    uncertainty = max(.1, float(np.quantile(np.linalg.norm(samples-delta, axis=1), .95)))
    signal = float(np.linalg.norm(delta))
    if signal < .5 or signal/uncertainty < 5. or uncertainty/signal > .2:
        raise ValueError('Raw tag response is too weak or uncertain')
    diagnostic = {'axis': axis, 'actual_sdk_fk_axis_mm': float(applied),
                  'raw_tag_delta_px': delta.tolist(), 'response_uncertainty_p95_px': uncertainty,
                  'response_snr': signal/uncertainty, 'pre': pre_info, 'post': post_info,
                  'run_camera_epoch': next(iter(epochs)), 'tracking_generation': next(iter(generations))}
    return {'column': delta/applied, 'bootstrap': samples/applied, 'diagnostic': diagnostic,
            'cup_centres': [np.median(cup_a, 0), np.median(cup_b, 0)],
            'board_centres': [board_a, board_b], 'before_pose': baseline, 'after_pose': poses[-1],
            'epoch': _base_epoch(next(iter(epochs))), 'intrinsics': scene['intrinsics_sha256']}


def build_calibration(scene_x, sdk_x, scene_y, sdk_y, *, bootstrap_samples=512, seed=20260916):
    """Return a validated local mapping or valid=false with an explicit error."""
    result = {'schema': SCHEMA, 'valid': False, 'physical_registration': False,
              'scope': 'local current pose; virtual pixel goals only', 'local_radius_mm': LOCAL_RADIUS_MM,
              'marker_corner_detector': DETECTOR, 'columns_subtract_cup_noise': False}
    try:
        if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or not 128 <= bootstrap_samples <= 4096:
            raise ValueError('Bootstrap count must be between 128 and 4096')
        rng = np.random.default_rng(seed)
        x = _trial(scene_x, sdk_x, 'x', rng, bootstrap_samples)
        y = _trial(scene_y, sdk_y, 'y', rng, bootstrap_samples)
        if (x['epoch'] != y['epoch'] or x['intrinsics'] != y['intrinsics']
                or not isinstance(x['intrinsics'], str) or len(x['intrinsics']) != 64):
            raise ValueError('Optical camera epoch or intrinsics mismatch')
        if _pose_distance(x['after_pose'], y['before_pose'])[0] > .00025 or _pose_distance(
                x['after_pose'], y['before_pose'])[1] > math.radians(.05):
            raise ValueError('Starting pose changed between local axis experiments')
        cups = np.asarray(x['cup_centres']+y['cup_centres'])
        cup_drift = max(float(np.linalg.norm(a-b)) for a in cups for b in cups)
        boards = x['board_centres']+y['board_centres']
        shared = set.intersection(*(set(board) for board in boards))
        if len(shared) < 8:
            raise ValueError('Fewer than eight common board corners across experiments')
        board_drift = max(float(np.linalg.norm(a[index]-b[index]))
                          for a in boards for b in boards for index in shared)
        if cup_drift > 1. or board_drift > .25:
            raise ValueError('Cup or board robust pixel medians moved across response windows')
        matrix = np.column_stack((x['column'], y['column']))
        singular = np.linalg.svd(matrix, compute_uv=False)
        condition = float(singular[0]/singular[-1]) if singular[-1] > 1e-9 else math.inf
        boot = np.stack((x['bootstrap'], y['bootstrap']), axis=2)
        boot_conditions = np.linalg.cond(boot)
        if singular[-1] < .25 or condition >= MAX_CONDITION or np.quantile(boot_conditions, .95) >= MAX_CONDITION:
            raise ValueError('Local tag Jacobian lacks reliable rank or conditioning')
        result.update({'valid': True, 'J_px_per_mm': matrix.tolist(), 'condition_number': condition,
                       'bootstrap_condition_p95': float(np.quantile(boot_conditions, .95)),
                       'singular_values_px_per_mm': singular.tolist(), 'camera_epoch': x['epoch'],
                       'intrinsics_sha256': x['intrinsics'], 'trials': [x['diagnostic'], y['diagnostic']],
                       'anchor_tag_center_px': x['diagnostic']['pre']['tag_median_px'],
                       'virtual_goal_px': x['diagnostic']['pre']['tag_median_px'],
                       'local_center_tag_px': y['diagnostic']['post']['tag_median_px'],
                       'anchor_sdk_fk_pose_m_rad': x['before_pose'].tolist(),
                       'latest_sdk_fk_pose_m_rad': y['after_pose'].tolist(),
                       'scene_stability': {'maximum_cup_median_drift_px': cup_drift,
                                           'maximum_shared_board_corner_drift_px': board_drift},
                       'error': None})
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError, np.linalg.LinAlgError) as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
    return result


def propose_axis_step(calibration, current_tag_center_px, target_virtual_px, *, timestamp_s,
                      now_s, camera_epoch, intrinsics_sha256, tracking_generation,
                      stable_generation, max_age_s=.15, max_step_mm=1.):
    """Propose one base axis step; absolute local radius <=3 mm, no commands."""
    result = {'schema': 'local_virtual_tag_axis_proposal_v1', 'valid': False, 'stop': False,
              'axis': None, 'distance_mm': None, 'physical_registration': False,
              'scope': 'virtual tag pixel goal; not a cup/contact target'}
    try:
        if calibration.get('valid') is not True or calibration.get('schema') != SCHEMA:
            raise ValueError('Validated local calibration is required')
        if camera_epoch != calibration['camera_epoch'] or intrinsics_sha256 != calibration['intrinsics_sha256']:
            raise ValueError('Current optical epoch or intrinsics mismatch')
        if (isinstance(tracking_generation, bool) or not isinstance(tracking_generation, int)
                or isinstance(stable_generation, bool) or not isinstance(stable_generation, int)
                or tracking_generation < 0 or tracking_generation != stable_generation):
            raise ValueError('Tracking generation is not stable')
        stamp, now, age_limit, cap = _numbers([timestamp_s, now_s, max_age_s, max_step_mm], (4,), 'proposal timing')
        if not 0 < age_limit <= .15 or not 0 <= now-stamp <= age_limit or not 0 < cap <= 1.:
            raise ValueError('Stale/future frame, invalid freshness bound or step cap')
        current = _numbers(current_tag_center_px, (2,), 'current raw tag centre')
        goal = _numbers(target_virtual_px, (2,), 'virtual goal')
        anchor = _numbers(calibration['local_center_tag_px'], (2,), 'local centre at Y post')
        matrix = _numbers(calibration['J_px_per_mm'], (2, 2), 'local Jacobian')
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular[-1] < .25 or singular[0]/singular[-1] >= MAX_CONDITION:
            raise ValueError('Current Jacobian is rank deficient or ill-conditioned')
        offsets = [np.linalg.solve(matrix, point-anchor) for point in (current, goal)]
        if max(np.linalg.norm(offset) for offset in offsets) > LOCAL_RADIUS_MM:
            raise ValueError('Current pose or virtual target exceeds local 3 mm radius')
        error = goal-current
        correction = np.linalg.solve(matrix, error)
        result.update({'pixel_error_px': error.tolist(), 'pixel_error_norm_px': float(np.linalg.norm(error)),
                       'base_xy_correction_mm': correction.tolist(),
                       'current_base_xy_offset_mm': offsets[0].tolist(), 'frame_age_s': float(now-stamp)})
        if np.linalg.norm(error) <= .25:
            result.update({'valid': True, 'stop': True, 'distance_mm': 0., 'error': None})
        else:
            axis = int(np.argmax(np.abs(correction)))
            step = float(np.clip(correction[axis], -cap, cap))
            next_offset = offsets[0].copy()
            next_offset[axis] += step
            if np.linalg.norm(next_offset) > LOCAL_RADIUS_MM:
                raise ValueError('Selected axis would leave the local 3 mm radius')
            predicted_error = error-matrix[:, axis]*step
            if np.linalg.norm(predicted_error) >= np.linalg.norm(error):
                raise ValueError('Selected axis would not reduce predicted virtual pixel error')
            result.update({'valid': True, 'axis': ('x', 'y')[axis], 'distance_mm': step,
                           'predicted_remaining_error_px': predicted_error.tolist(), 'error': None})
    except (ValueError, TypeError, KeyError, ArithmeticError, np.linalg.LinAlgError) as exc:
        result.update({'valid': False, 'axis': None, 'distance_mm': None,
                       'error': type(exc).__name__+': '+str(exc)})
    return result


def create_calibration(x_scene_path, x_sdk_path, y_scene_path, y_sdk_path):
    """Read four local evidence JSONs and attach their hashes; never access hardware."""
    paths = [Path(value) for value in (x_scene_path, x_sdk_path, y_scene_path, y_sdk_path)]
    try:
        blobs = [path.read_bytes() for path in paths]
        result = build_calibration(*(json.loads(blob) for blob in blobs))
        result['source_files_sha256'] = {str(path.resolve()): hashlib.sha256(blob).hexdigest()
                                        for path, blob in zip(paths, blobs)}
    except (OSError, ValueError, TypeError) as exc:
        result = {'schema': SCHEMA, 'valid': False, 'physical_registration': False,
                  'error': type(exc).__name__+': '+str(exc)}
    result['tool_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--x-scene', type=Path, required=True)
    parser.add_argument('--x-sdk', type=Path, required=True)
    parser.add_argument('--y-scene', type=Path, required=True)
    parser.add_argument('--y-sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = create_calibration(args.x_scene, args.x_sdk, args.y_scene, args.y_sdk)
    output = json.dumps(result, indent=2, allow_nan=False)+'\n'
    temporary = args.output.with_name(args.output.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(output)
    temporary.replace(args.output)
    print(output, end='')
    return 0 if result['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
