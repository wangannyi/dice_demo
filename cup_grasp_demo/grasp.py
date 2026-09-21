"""Bounded RGB-D snapshot and read-only palm side-grasp proposal for NERO.

This module never sends arm or hand commands. Positions are metres in arm base;
poses use XYZ plus ZYX Euler angles (radians), matching nero_calibration.core.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cup_grasp_demo import planning, vision
from dice_cup_localization.geometry import localize
from nero_calibration.core import matrix, matrix_pose, pose_matrix


DEFAULT_SERIAL = '346222071954'
FRESH_SNAPSHOT_MAX_AGE_S = 10.0
MAX_COLOR_DEPTH_TIMESTAMP_GAP_MS = 40.0
# Keep this rig-specific non-singular ready pose in step with
# nero_revo2_control.nero_revo2_demo.READY_HOME_DEG.  Importing that demo here
# would require pyAgxArm even for offline snapshot planning.
READY_HOME_DEG = (55.0, -78.0, 80.0, -45.0, 130.0, -30.0, 40.0)
READY_HOME_TOLERANCE_DEG = 1.0


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(_json(data))


def _check_frame_timestamps(color_ms, depth_ms, color_domain, depth_domain):
    color_ms, depth_ms = float(color_ms), float(depth_ms)
    if not (math.isfinite(color_ms) and math.isfinite(depth_ms)):
        raise ValueError('RGB-D frame timestamps must be finite')
    if not isinstance(color_domain, str) or color_domain != depth_domain:
        raise ValueError('RGB-D frame timestamp domains differ')
    gap_ms = abs(color_ms - depth_ms)
    if gap_ms > MAX_COLOR_DEPTH_TIMESTAMP_GAP_MS:
        raise ValueError(f'RGB-D frame timestamp gap {gap_ms:.1f} ms exceeds sync limit')
    return gap_ms


def save_snapshot(directory, frame):
    """Write one registered color/Z16 pair with its capture identity and scale."""
    directory = Path(directory)
    color = np.asarray(frame['color_bgr'])
    depth = np.asarray(frame['depth_raw'])
    if (color.ndim != 3 or color.shape[2] != 3 or color.dtype != np.uint8
            or depth.shape != color.shape[:2] or depth.dtype != np.uint16):
        raise ValueError('Expected one aligned BGR uint8/Z16 uint16 pair')
    intr = frame['intrinsics']
    if (intr.get('width'), intr.get('height')) != (color.shape[1], color.shape[0]):
        raise ValueError('Color intrinsics do not match aligned RGB-D resolution')
    coeffs = np.asarray(intr.get('dist_coeffs'), dtype=float)
    if coeffs.shape != (5,) or not np.isfinite(coeffs).all():
        raise ValueError('RealSense capture must report five distortion coefficients')
    if np.any(coeffs != 0):
        raise ValueError('Nonzero color distortion requires rectification before pinhole deprojection')
    scale = float(frame['depth_scale_m'])
    if not (math.isfinite(scale) and 0 < scale < 0.1):
        raise ValueError('Invalid RealSense depth scale')
    if frame.get('frame') != 'color_optical':
        raise ValueError('Only color_optical aligned capture is supported')
    timestamp_gap_ms = _check_frame_timestamps(
        frame['timestamps_ms']['color'], frame['timestamps_ms']['depth_aligned'],
        frame['timestamp_domains']['color'], frame['timestamp_domains']['depth_aligned'])
    directory.mkdir(parents=True, exist_ok=False)
    color_path, depth_path = directory / 'color.png', directory / 'depth.npz'
    if not cv2.imwrite(str(color_path), color):
        raise OSError(f'Could not write {color_path}')
    np.savez_compressed(depth_path, depth_raw=depth)
    metadata = {
        'schema': 1, 'camera_backend': 'realsense', 'serial': str(frame['serial']),
        'frame': 'color_optical', 'depth_registered_to': 'color_optical',
        'alignment': 'librealsense rs.align(color)', 'depth_format': 'Z16_raw',
        'depth_scale_m': scale, 'intrinsics': {
            'width': int(intr['width']), 'height': int(intr['height']),
            'fx': float(intr['fx']), 'fy': float(intr['fy']),
            'cx': float(intr['ppx']), 'cy': float(intr['ppy']),
            'dist_coeffs': coeffs.tolist(),
            'distortion_model': str(intr.get('distortion_model', 'unknown')),
            'frame': 'color_optical',
        },
        'frame_id': f"{frame['serial']}:{frame['frame_id']}",
        'timestamp_ms': float(frame['timestamps_ms']['color']),
        'timestamp_domain': str(frame['timestamp_domains']['color']),
        'depth_timestamp_ms': float(frame['timestamps_ms']['depth_aligned']),
        'depth_timestamp_domain': str(frame['timestamp_domains']['depth_aligned']),
        'color_depth_timestamp_gap_ms': timestamp_gap_ms,
        'host_capture_time_ns': int(frame['host_capture_time_ns']),
        'sha256_color': _sha256(color_path), 'sha256_depth': _sha256(depth_path),
    }
    _write_new(directory / 'metadata.json', metadata)
    return metadata


def load_snapshot(directory):
    """Read saved same-frame assets; never open a camera."""
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('schema') != 1 or metadata.get('depth_registered_to') != 'color_optical':
        raise ValueError('Snapshot must be registered to color_optical')
    _check_frame_timestamps(metadata['timestamp_ms'], metadata['depth_timestamp_ms'],
                            metadata['timestamp_domain'], metadata['depth_timestamp_domain'])
    color_path, depth_path = directory / 'color.png', directory / 'depth.npz'
    if _sha256(color_path) != metadata.get('sha256_color') or _sha256(depth_path) != metadata.get('sha256_depth'):
        raise ValueError('Saved color/depth hash differs from capture metadata')
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise OSError(f'Could not read {color_path}')
    with np.load(depth_path, allow_pickle=False) as bundle:
        depth = bundle['depth_raw'].copy()
    if depth.dtype != np.uint16 or depth.shape != color.shape[:2]:
        raise ValueError('Saved depth does not match color frame')
    return color, depth, metadata


def _adjacent_table_bbox(bbox, shape):
    """Use the near-side strip below the upright cup, avoiding hand/blue mat."""
    x0, y0, x1, y1 = (int(v) for v in bbox)
    height, width = shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Cup bbox must be inside color image')
    top, bottom = max(0, y1-10), min(height, y1+60)
    if bottom-top < 20:
        raise ValueError('Cup is too close to frame edge for adjacent tabletop strip')
    return (x0, top, x1, bottom)


def localize_snapshot(color, depth_raw, metadata, *, object_mask_path=None,
                      table_mask_path=None, bbox=None, instance_id='cup-1',
                      target_color='dark'):
    """Localize the observed side geometry in the saved color optical frame."""
    shape = color.shape[:2]
    mask_refinement = None
    table_refinement = None
    if target_color not in ('dark', 'green'):
        raise ValueError('Target color must be dark or green')
    if object_mask_path is not None:
        object_mask = vision.load_mask_png(object_mask_path, shape)
        object_source = f'manual_png:{Path(object_mask_path).name}'
    else:
        if bbox is None:
            raise ValueError('Provide --object-mask or a cup --bbox ROI')
        if target_color == 'dark':
            object_mask = vision.segment_dark_cup(color, bbox)
            object_source = 'dark_cup_hsv_roi'
        else:
            object_mask = vision.segment_green_cup(color, bbox)
            object_source = 'green_cup_hsv_roi'
    if table_mask_path is not None:
        table_mask = vision.load_mask_png(table_mask_path, shape)
        table_source = f'manual_png:{Path(table_mask_path).name}'
    else:
        if bbox is None:
            raise ValueError('Provide --table-mask or --bbox for adjacent tabletop')
        table_bbox = _adjacent_table_bbox(bbox, shape)
        depth_m = depth_raw.astype(np.float32) * float(metadata['depth_scale_m'])
        if target_color == 'green' and object_mask_path is None:
            table_mask, table_refinement = vision.segment_red_table_plane_supported(
                color, object_mask, table_bbox, depth_m, metadata['intrinsics'])
            table_source = table_refinement['mask_source']
        else:
            table_mask = vision.segment_table(
                color, object_mask, bbox=table_bbox, depth_m=depth_m)
            table_source = 'adjacent_table_hsv_depth'
    if object_mask_path is None and target_color == 'dark':
        object_mask, mask_refinement = vision.segment_dark_cup_depth_refined(
            color, bbox,
            depth_raw.astype(np.float32) * float(metadata['depth_scale_m']),
            table_mask, metadata['intrinsics'])
        object_source = mask_refinement['mask_source']
    observation = dict(metadata)
    observation.update(instance_id=instance_id, mask_source=f'{object_source};{table_source}')
    result = localize(depth_raw, object_mask, table_mask, observation)
    result['camera_serial'] = metadata['serial']
    result['mask_pixels'] = {'object': int(np.count_nonzero(object_mask)),
                             'table': int(np.count_nonzero(table_mask))}
    result['target_color'] = target_color
    if mask_refinement is not None:
        result['mask_refinement'] = mask_refinement
    if table_refinement is not None:
        result['table_refinement'] = table_refinement
    return result, object_mask, table_mask


def _check_capture_calibration(metadata, calibration):
    camera = calibration['camera']
    if (metadata['serial'] != camera['serial']
            or metadata['intrinsics']['width'] != camera['width']
            or metadata['intrinsics']['height'] != camera['height']):
        raise ValueError('Snapshot camera serial/resolution differs from calibration')
    intr = metadata['intrinsics']
    expected = camera['camera_matrix']
    actual = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    if not np.allclose(actual, expected, atol=1e-3, rtol=0):
        raise ValueError('Snapshot color intrinsics differ from calibration')
    if not np.allclose(intr['dist_coeffs'], camera['dist_coeffs'], atol=1e-8, rtol=0):
        raise ValueError('Snapshot distortion differs from calibration')


def read_flange_feedback(channel='can0'):
    """Use the existing NERO calibration feedback command; it sends no motion."""
    script = Path(__file__).resolve().parents[1] / 'nero_calibration' / 'run_k3.sh'
    command = [str(script), 'feedback-check', '--channel', channel, '--tcp', 'palm']
    completed = subprocess.run(command, check=True, text=True, capture_output=True, timeout=15)
    messages = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith('{')]
    if not messages or 'flange_pose_m_rad' not in messages[-1]:
        raise RuntimeError('NERO feedback did not return a flange pose')
    return pose_matrix(messages[-1]['flange_pose_m_rad']), {
        'kind': 'can_feedback_read_only', 'channel': channel, 'host_read_time_ns': time.time_ns(),
    }


def read_ready_home_feedback(channel='can0', python_executable=sys.executable):
    """Read seven CAN joints through the existing demo; never send a target."""
    demo = Path(__file__).resolve().parents[1] / 'nero_revo2_control' / 'nero_revo2_demo.py'
    command = [str(python_executable), str(demo), '--format', 'json',
               '--channel', str(channel), 'read-joints']
    completed = subprocess.run(command, check=True, text=True, capture_output=True, timeout=15)
    received_ns = time.time_ns()
    messages = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith('{')]
    matching = [message for message in messages if message.get('event') == 'read_joints']
    if not matching:
        raise RuntimeError('Read-only NERO joint command returned no seven-axis feedback')
    feedback = matching[-1]
    joints = np.asarray(feedback['joints_deg'], dtype=float)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError('Ready-home check requires seven finite live joint angles')
    enabled = feedback['joints_enabled']
    if not isinstance(enabled, list) or len(enabled) != 7 or any(type(v) is not bool for v in enabled):
        raise ValueError('Ready-home check requires seven joint enable flags')
    arm_status, ctrl_mode = int(feedback['arm_status']), int(feedback['ctrl_mode'])
    errors = np.abs(joints - np.asarray(READY_HOME_DEG))
    max_error = float(np.max(errors))
    arrived = max_error <= READY_HOME_TOLERANCE_DEG
    ready = arrived and arm_status == 0 and ctrl_mode == 1 and all(enabled)
    return {
        'required': True, 'confirmed': bool(ready),
        'joint_target_deg': list(READY_HOME_DEG),
        'joint_feedback_deg': joints.tolist(),
        'joint_errors_deg': errors.tolist(),
        'max_joint_error_deg': max_error,
        'joint_tolerance_deg': READY_HOME_TOLERANCE_DEG,
        'joint_angles_arrived': bool(arrived),
        'arm_status': arm_status, 'ctrl_mode': ctrl_mode,
        'joints_enabled': enabled,
        'provenance': {
            'kind': 'can_joint_feedback_read_only', 'channel': str(channel),
            'command': 'read-joints', 'python_executable': str(python_executable),
            'host_feedback_received_time_ns': received_ns,
        },
    }


def _unit(values, name):
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f'{name} must be a finite 3-vector')
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError(f'{name} has zero length')
    return vector / norm


def _yaw_align_palm_x(current_rotation, inward):
    """Align horizontal palm +X heading by a base +Z yaw; retain its tilt."""
    current_xy = np.array(current_rotation[:2, 0], dtype=float, copy=True)
    inward_xy = np.array(inward[:2], dtype=float, copy=True)
    if min(np.linalg.norm(current_xy), np.linalg.norm(inward_xy)) < 1e-8:
        raise ValueError('Palm/cup horizontal heading is undefined')
    current_xy /= np.linalg.norm(current_xy)
    inward_xy /= np.linalg.norm(inward_xy)
    correction = math.atan2(current_xy[0]*inward_xy[1]-current_xy[1]*inward_xy[0],
                            float(np.dot(current_xy, inward_xy)))
    if abs(correction) > math.radians(70):
        raise ValueError('Palm heading correction exceeds 70°; reposition and review')
    cosine, sine = math.cos(correction), math.sin(correction)
    base_yaw = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
    return base_yaw @ current_rotation, correction


def _explicit_home_palm_rotation(current_rotation, base_yaw_deg, palm_x_roll_deg):
    """Apply a base +Z yaw, then a local palm +X roll to live HOME palm."""
    yaw, roll = math.radians(base_yaw_deg), math.radians(palm_x_roll_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    base_yaw = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
    local_roll = np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])
    return base_yaw @ current_rotation @ local_roll


def compute_side_grasp(localization, calibration, current_T_base_flange,
                       snapshot_metadata, *, allow_provisional=False,
                       pose_source=None, now_ns=None, min_facing_dot=.7,
                       align_palm_to_cup=False, alignment_reviewed=False,
                       nominal_cup_height_m=None, assume_base_z_upright=False,
                       axis_reviewed=False, require_home=False, home_check=None,
                       base_yaw_deg=None, palm_x_roll_deg=None):
    """Propose targets; yaw, nominal height, and base-Z axis are each opt-in."""
    if localization.get('valid') is not True:
        raise ValueError(f"Cup localization invalid: {localization.get('reason')}")
    _check_capture_calibration(snapshot_metadata, calibration)
    current_flange = matrix(current_T_base_flange)
    flange_tcp = matrix(calibration['T_flange_tcp'])
    current_palm = matrix(current_flange @ flange_tcp)
    cup = planning.map_localization(localization, calibration,
                                    camera_serial=snapshot_metadata['serial'],
                                    allow_provisional=allow_provisional)
    center = np.asarray(cup['center_m'], dtype=float)
    axis = _unit(cup['axis'], 'measured cup axis')
    if axis_reviewed and not assume_base_z_upright:
        raise ValueError('--axis-reviewed requires --assume-base-z-upright')
    if assume_base_z_upright and nominal_cup_height_m is None:
        raise ValueError('--assume-base-z-upright requires --nominal-cup-height-m')
    measured_axis_base_z_dot = float(np.dot(axis, [0., 0., 1.]))
    if assume_base_z_upright and measured_axis_base_z_dot < math.cos(math.radians(45)):
        raise ValueError('Measured cup axis differs from base +Z by more than 45°; review mount/table')
    planning_axis = np.array([0., 0., 1.]) if assume_base_z_upright else axis
    support = np.asarray(cup['support_center_m'], dtype=float)
    observed_height_value = cup['dimensions'].get('observed_height_m')
    observed_height_m = None if observed_height_value is None else float(observed_height_value)
    if observed_height_m is not None and not (math.isfinite(observed_height_m) and observed_height_m > 0):
        raise ValueError('Observed RGB-D cup height must be positive and finite')
    if nominal_cup_height_m is not None:
        nominal_cup_height_m = float(nominal_cup_height_m)
        if not (math.isfinite(nominal_cup_height_m) and .03 <= nominal_cup_height_m <= .30):
            raise ValueError('Nominal cup height must be finite in 0.03..0.30 m')
        planning_center = support + planning_axis * (nominal_cup_height_m / 2.)
        center_source = ('support_plus_explicit_nominal_height_half_base_z_assumption'
                         if assume_base_z_upright else 'support_plus_explicit_nominal_height_half')
    else:
        planning_center = center
        center_source = 'observed_rgbd_height_midpoint'
    candidate_orientation = base_yaw_deg is not None or palm_x_roll_deg is not None
    if candidate_orientation:
        if base_yaw_deg is None or palm_x_roll_deg is None:
            raise ValueError('--base-yaw-deg and --palm-x-roll-deg must be provided together')
        if align_palm_to_cup:
            raise ValueError('Explicit HOME orientation is mutually exclusive with --align-palm-to-cup')
        if not require_home:
            raise ValueError('Explicit HOME orientation requires --require-home')
        if (not math.isfinite(base_yaw_deg) or abs(base_yaw_deg) > 45.
                or not math.isfinite(palm_x_roll_deg) or abs(palm_x_roll_deg) > 35.):
            raise ValueError('Explicit HOME yaw/roll must be finite within ±45°/±35°')
    orientation_changed = bool(align_palm_to_cup or candidate_orientation)
    if alignment_reviewed and not orientation_changed:
        raise ValueError('--alignment-reviewed requires an explicit palm orientation change')
    diameter = float(cup['dimensions']['observed_side_diameter_m'])
    if not (math.isfinite(diameter) and .02 <= diameter <= .20):
        raise ValueError('Observed cup side diameter is implausible')
    delta = current_palm[:3, 3] - planning_center
    outward = _unit(delta - planning_axis * np.dot(delta, planning_axis),
                    'horizontal cup-to-palm approach')
    current_facing_dot = float(np.dot(current_palm[:3, 0], -outward))
    if align_palm_to_cup:
        target_rotation, heading_correction_rad = _yaw_align_palm_x(current_palm[:3, :3], -outward)
        orientation_source = 'explicit_base_z_yaw_aligned_palm'
    elif candidate_orientation:
        target_rotation = _explicit_home_palm_rotation(
            current_palm[:3, :3], base_yaw_deg, palm_x_roll_deg)
        heading_correction_rad = math.radians(base_yaw_deg)
        orientation_source = 'explicit_home_base_yaw_local_palm_x_roll'
    else:
        target_rotation, heading_correction_rad = current_palm[:3, :3], 0.
        orientation_source = 'current_palm_orientation'
    target_facing_dot = float(np.dot(target_rotation[:, 0], -outward))
    radius = diameter / 2
    contact = planning_center + outward * (radius + .07) + planning_axis * .025
    pregrasp = contact + outward * .04 + planning_axis * .03
    if candidate_orientation:
        pregrasp = pregrasp + np.array([0., 0., .04])
    lift = contact + planning_axis * .05
    waypoints = {}
    stages = ([('clearance', current_palm[:3, 3] + np.array([0., 0., .08]))]
              if orientation_changed else [])
    stages += [('pregrasp', pregrasp), ('contact', contact), ('lift', lift)]
    for name, location in stages:
        orientation = ({'palm_orientation_base': target_rotation}
                       if orientation_changed and name != 'clearance'
                       else {'current_T_base_palm': current_palm})
        target = planning.plan_palm_target(
            localization, calibration, camera_serial=snapshot_metadata['serial'],
            palm_position_base_m=location, allow_provisional=allow_provisional,
            **orientation)
        palm_altitude = float(np.dot(location-support, axis))
        palm_planning_altitude = float(np.dot(location-support, planning_axis))
        flange_position = np.asarray(target['T_base_flange_target'])[:3, 3]
        flange_altitude = float(np.dot(flange_position-support, axis))
        flange_planning_altitude = float(np.dot(flange_position-support, planning_axis))
        if palm_altitude < -1e-6:
            raise ValueError(f'{name} palm target lies below observed tabletop plane')
        waypoints[name] = {
            'palm_pose_base_m_rad': matrix_pose(target['T_base_palm_target']),
            'flange_pose_base_m_rad': matrix_pose(target['T_base_flange_target']),
            'T_base_palm': target['T_base_palm_target'],
            'T_base_flange': target['T_base_flange_target'],
            'palm_base_z_m': float(location[2]),
            'flange_base_z_m': float(flange_position[2]),
            'palm_altitude_from_table_m': palm_altitude,
            'flange_altitude_from_table_m': flange_altitude,
            'palm_altitude_from_planning_axis_m': palm_planning_altitude,
            'flange_altitude_from_planning_axis_m': flange_planning_altitude,
        }
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    acquired_ns = int(snapshot_metadata.get('host_capture_time_ns', 0))
    age_s = max(0.0, (now_ns-acquired_ns)/1e9) if acquired_ns > 0 else None
    freshness_passed = age_s is not None and 0 <= age_s <= FRESH_SNAPSHOT_MAX_AGE_S
    planning_axis_base_z_dot = float(np.dot(planning_axis, [0., 0., 1.]))
    if not (math.isfinite(min_facing_dot) and .5 <= min_facing_dot <= 1.):
        raise ValueError('Minimum palm facing dot must be in 0.5..1.0')
    measured_upright_passed = measured_axis_base_z_dot >= .9
    planning_upright_passed = planning_axis_base_z_dot >= .9
    facing_passed = target_facing_dot >= min_facing_dot
    pose_live = (pose_source or {}).get('kind') == 'can_feedback_read_only'
    home_confirmed = bool(
        isinstance(home_check, dict)
        and home_check.get('confirmed') is True
        and home_check.get('provenance', {}).get('kind') == 'can_joint_feedback_read_only'
    )
    home_gate_passed = not require_home or home_confirmed
    return {
        'schema': 1, 'kind': 'read_only_side_grasp_proposal', 'units': 'm_rad',
        'localization': localization, 'cup_base': cup,
        'current_flange_pose_base_m_rad': matrix_pose(current_flange),
        'current_palm_pose_base_m_rad': matrix_pose(current_palm),
        'approach_outward_base': outward.tolist(),
        'planning_cup_axis_base': planning_axis.tolist(),
        'axis_provenance': {
            'source': ('explicit_base_z_upright_assumption' if assume_base_z_upright
                       else 'measured_rgbd_support_plane_normal'),
            'measured_axis_base': axis.tolist(),
            'measured_vs_base_z_angle_deg': math.degrees(math.acos(
                float(np.clip(measured_axis_base_z_dot, -1., 1.)))),
            'mount_and_table_orientation_verified': False,
            'axis_hardware_validated': False,
        },
        'orientation_source': orientation_source,
        'orientation_provenance': {
            'source': orientation_source,
            'rotation_order': ('Rz_base(yaw) @ R_HOME_palm @ Rx_palm(roll)'
                               if candidate_orientation else None),
            'base_yaw_deg': float(base_yaw_deg) if candidate_orientation else None,
            'local_palm_x_roll_deg': float(palm_x_roll_deg) if candidate_orientation else None,
            'pregrasp_extra_base_z_m': .04 if candidate_orientation else 0.,
            'home_reference_verified': home_confirmed if candidate_orientation else None,
            'hardware_path_validated': False,
        },
        'heading_correction_rad': heading_correction_rad,
        'heading_correction_deg': math.degrees(heading_correction_rad),
        'planning_cup_center_base_m': planning_center.tolist(),
        'center_provenance': {
            'source': center_source,
            'observed_height_m': observed_height_m,
            'nominal_height_m': nominal_cup_height_m,
            'nominal_minus_observed_height_m': (
                nominal_cup_height_m-observed_height_m
                if nominal_cup_height_m is not None and observed_height_m is not None else None),
            'geometry_hardware_validated': False,
        },
        'waypoints': waypoints,
        'checks': {
            'axis_base_z_dot': measured_axis_base_z_dot,
            'axis_upright': measured_upright_passed,
            'measured_axis_base_z_dot': measured_axis_base_z_dot,
            'measured_axis_upright': measured_upright_passed,
            'planning_axis_base_z_dot': planning_axis_base_z_dot,
            'planning_axis_upright': planning_upright_passed,
            'assume_base_z_upright': bool(assume_base_z_upright),
            'axis_reviewed': bool(axis_reviewed),
            'axis_override_requires_review': bool(assume_base_z_upright and not axis_reviewed),
            'axis_override_requires_alignment_review': bool(
                assume_base_z_upright and not (orientation_changed and alignment_reviewed)),
            'palm_x_toward_cup_dot': current_facing_dot,
            'target_palm_x_toward_cup_dot': target_facing_dot,
            'minimum_palm_facing_dot': min_facing_dot,
            'palm_facing_cup': facing_passed,
            'align_palm_to_cup': bool(align_palm_to_cup),
            'explicit_home_palm_orientation': bool(candidate_orientation),
            'alignment_reviewed': bool(alignment_reviewed),
            'alignment_requires_ik_scene_review': bool(orientation_changed and not alignment_reviewed),
            'clearance_required': bool(orientation_changed),
            'clearance_offset_base_z_m': .08 if orientation_changed else None,
            'snapshot_age_s': age_s, 'snapshot_fresh': freshness_passed,
            'pose_source_live_can': pose_live,
            'require_home': bool(require_home),
            'home_confirmed': home_confirmed,
            'home_gate_passed': home_gate_passed,
            'calibration_quality_passed': calibration['quality_passed'],
            'provisional_opt_in': bool(allow_provisional),
            'execute_ready': bool(planning_upright_passed and facing_passed and freshness_passed
                                  and pose_live and (calibration['quality_passed'] or allow_provisional)
                                  and (not orientation_changed or alignment_reviewed)
                                  and (not assume_base_z_upright or (
                                      axis_reviewed and orientation_changed and alignment_reviewed))
                                  and home_gate_passed),
        },
        'home_check': home_check,
        'pose_source': pose_source,
        'calibration_sha256': calibration.get('_source', {}).get('sha256'),
        'calibration_holdout_errors_m_deg': calibration.get('holdout_errors_m_deg'),
        'calibration_all_sample_errors_m_deg': calibration.get('all_sample_errors_m_deg'),
        'snapshot_frame_id': snapshot_metadata['frame_id'],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    capture = sub.add_parser('snapshot', help='Save one bounded aligned D435i RGB-D frame')
    capture.add_argument('--dataset', type=Path, required=True)
    capture.add_argument('--serial', default=DEFAULT_SERIAL)
    capture.add_argument('--timeout-ms', type=int, default=3000)
    capture.add_argument('--warmup-frames', type=int, default=3,
                         help='Discard startup frames; use 30 for K3 D435i depth warmup')
    plan = sub.add_parser('plan', help='Offline localization + read-only CAN pose + waypoints')
    plan.add_argument('--dataset', type=Path, required=True)
    plan.add_argument('--calibration', type=Path, required=True)
    plan.add_argument('--allow-provisional', action='store_true',
                      help='Explicitly acknowledge calibration failed residual thresholds')
    plan.add_argument('--object-mask', type=Path)
    plan.add_argument('--table-mask', type=Path)
    plan.add_argument('--bbox', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'))
    plan.add_argument('--target-color', choices=('dark', 'green'), default='dark')
    plan.add_argument('--min-facing-dot', type=float, default=.7,
                      help='Facing check; initial default 0.7, lower only for supervised trial')
    orientation_choice = plan.add_mutually_exclusive_group()
    orientation_choice.add_argument('--align-palm-to-cup', action='store_true',
                                    help='Opt in to base +Z yaw alignment of palm +X toward cup')
    orientation_choice.add_argument('--base-yaw-deg', type=float,
                                    help='Explicit HOME palm candidate: base +Z yaw within ±45°')
    plan.add_argument('--palm-x-roll-deg', type=float,
                      help='Pair with --base-yaw-deg: local palm +X roll within ±35°')
    plan.add_argument('--alignment-reviewed', action='store_true',
                      help='Assert separate URDF IK and live scene review for the aligned orientation')
    plan.add_argument('--nominal-cup-height-m', type=float,
                      help='Measured cup height for waypoint center; axis depends on explicit assumption')
    plan.add_argument('--assume-base-z-upright', action='store_true',
                      help='Explicit provisional assumption: use base +Z for waypoint cup axis')
    plan.add_argument('--axis-reviewed', action='store_true',
                      help='Assert separate mount/table and live scene review of base +Z axis assumption')
    plan.add_argument('--flange-pose', type=float, nargs=6, metavar=('X', 'Y', 'Z', 'RX', 'RY', 'RZ'),
                      help='Manual pose for offline analysis; prevents execute_ready')
    plan.add_argument('--channel', default='can0', help='Read-only NERO CAN feedback channel')
    plan.add_argument('--require-home', action='store_true',
                      help='Gate execution readiness on live seven-joint READY_HOME feedback')
    plan.add_argument('--python', default=sys.executable,
                      help='Python with pyAgxArm installed for --require-home read-joints')
    plan.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'snapshot':
            if not 100 <= args.timeout_ms <= 10000:
                raise ValueError('Snapshot timeout must be within 100..10000 ms')
            if not 0 <= args.warmup_frames <= 60:
                raise ValueError('Snapshot warmup must be within 0..60 frames')
            if args.warmup_frames == 3:
                frame = vision.capture_frame(args.serial, timeout_ms=args.timeout_ms)
            else:
                frame = vision.capture_frame(
                    args.serial, timeout_ms=args.timeout_ms,
                    warmup_frames=args.warmup_frames)
            output = save_snapshot(args.dataset, frame)
        else:
            color, depth, metadata = load_snapshot(args.dataset)
            calibration = planning.load_calibration(
                args.calibration, expected_camera_serial=metadata['serial'],
                allow_provisional=args.allow_provisional)
            result, _, _ = localize_snapshot(
                color, depth, metadata, object_mask_path=args.object_mask,
                table_mask_path=args.table_mask, bbox=args.bbox,
                target_color=args.target_color)
            if result.get('valid') is not True:
                raise ValueError(f"Cup localization invalid: {result.get('reason')}")
            if args.flange_pose is not None:
                flange = pose_matrix(args.flange_pose)
                source = {'kind': 'manual_offline', 'host_read_time_ns': None}
            else:
                flange, source = read_flange_feedback(args.channel)
            home_check = (read_ready_home_feedback(args.channel, args.python)
                          if args.require_home else None)
            output = compute_side_grasp(
                result, calibration, flange, metadata,
                allow_provisional=args.allow_provisional, pose_source=source,
                min_facing_dot=args.min_facing_dot,
                align_palm_to_cup=args.align_palm_to_cup,
                alignment_reviewed=args.alignment_reviewed,
                nominal_cup_height_m=args.nominal_cup_height_m,
                assume_base_z_upright=args.assume_base_z_upright,
                axis_reviewed=args.axis_reviewed,
                require_home=args.require_home, home_check=home_check,
                base_yaw_deg=args.base_yaw_deg, palm_x_roll_deg=args.palm_x_roll_deg)
            if args.output is not None:
                _write_new(args.output, output)
        print(_json(output), end='')
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        print(_json({'ok': False, 'reason': str(exc)}), end='', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
