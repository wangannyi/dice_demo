"""Opt-in, read-only top-of-cup proposal for NERO with a right Revo2 hand.

The official Revo2 fingers extend along palm +X and curl toward palm +Z.
Consequently a top approach places palm +Z toward the *negative* upright cup
axis.  ``PALM`` is a kinematic frame, not a tactile point: the palm-base STL
extends roughly 12 mm in front of its origin.  None of the nominal positions
below prove contact, a held cup, or the actual controller path.

This module reads files/URDFs only.  It does not open a camera, CAN, or SDK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from cup_grasp_demo.hand_geometry import RightRevo2Model, check_hand_pose
from cup_grasp_demo.ik_review import DOCUMENTED_LIMITS_DEG, _solve_bounded
from nero_calibration.core import inverse, matrix, matrix_pose, pose_matrix
from nero_revo2_control.kinematics import load_model


HAND_CHANNELS = ('thumb_tip', 'thumb_base', 'index_finger', 'middle_finger',
                 'ring_finger', 'pinky_finger')
TOP_STAGES = ('clearance', 'pretop', 'guarded_hover', 'probe_1', 'probe_2',
              'contact_candidate', 'tiny_shake_a', 'tiny_shake_b')
REVEAL_STAGES = ('reveal_lift', 'reveal_exit')
PROVISIONAL_POSITION_ERROR_M = .013
APPROACH_MARGIN_M = .005
MAX_SNAPSHOT_AGE_S = 10.
MAX_PROBE_STEP_M = .005


def _unit(values, name):
    value = np.asarray(values, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a finite three-vector')
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ValueError(f'{name} is zero')
    return value / norm


def _finite_joints(values, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError(f'{name} requires seven finite radians')
    return values


def _pose_error(a, b):
    a, b = matrix(a), matrix(b)
    position = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    cosine = float((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2)
    return position, math.acos(max(-1., min(1., cosine)))


def _rotation(values):
    out = np.asarray(values, dtype=float)
    if out.shape != (3, 3) or not np.isfinite(out).all():
        raise ValueError('palm_orientation_base must be a finite 3x3 matrix')
    if not np.allclose(out.T @ out, np.eye(3), atol=1e-5, rtol=0) or np.linalg.det(out) < .99999:
        raise ValueError('palm_orientation_base must be a right-handed rotation')
    return out


def _sdk_ranges(limits):
    if not isinstance(limits, (list, tuple)) or len(limits) != 7:
        raise ValueError('Live SDK limits must contain seven [min,max] degree pairs')
    result = []
    for pair in limits:
        if len(pair) != 2:
            raise ValueError('Each SDK range must contain two degrees')
        lo, hi = (float(value) for value in pair)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError('SDK limit pair must be ordered and finite')
        result.append((lo, hi))
    return result


def _within_all_limits(model, q, sdk_ranges, *, margin_deg=0.):
    if not model.within_limits(q, math.radians(margin_deg)):
        return False
    deg = np.degrees(q)
    return all(lo+margin_deg <= angle <= hi-margin_deg
               for angle, (lo, hi) in zip(deg, sdk_ranges)) and all(
                   lo+margin_deg <= angle <= hi-margin_deg
                   for angle, (lo, hi) in zip(deg, DOCUMENTED_LIMITS_DEG))


def _hand_angles(positions):
    if not isinstance(positions, dict) or set(positions) != set(HAND_CHANNELS):
        raise ValueError('hand_open_positions must have all six normalized Revo2 channels')
    values = {name: float(positions[name]) for name in HAND_CHANNELS}
    if any(not math.isfinite(v) or v < 0 or v > 100
           for k, v in values.items()):
        raise ValueError('Revo2 normalized open-hand feedback is outside official mapping')
    # agx_arm_ctrl_single_node.py REVO2_FINGER_CONFIG and position maxima.
    return {
        'right_thumb_metacarpal_joint': values['thumb_base'] * 1.57 / 100.,
        'right_thumb_proximal_joint': min(values['thumb_tip'], 79.8) * 1.03 / 79.8,
        **{f'right_{finger}_proximal_joint': values[f'{finger}_finger'] * 1.41 / 100.
           for finger in ('index', 'middle', 'ring', 'pinky')},
    }


def _source_age_s(source_checks, pose_source, now_ns):
    """Upper bound from the side planner's snapshot age and pose read time."""
    side_age = source_checks.get('snapshot_age_s')
    read_ns = pose_source.get('host_read_time_ns')
    if (side_age is None or read_ns is None or not math.isfinite(float(side_age))
            or float(side_age) < 0 or int(read_ns) <= 0):
        return None
    return float(side_age) + max(0., (int(now_ns)-int(read_ns))/1e9) + 1.


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError(f'{name} must be a lowercase SHA-256 digest')
    return value


def _observed_top(side_plan, cup, support, axis, nominal_height, uncertainty):
    """Use YOLO same-frame visible plane/side-axis intersection when present."""
    nominal = support + axis * nominal_height
    measured = cup.get('measured_top_surface')
    if measured is None:
        return nominal, {
            'source': 'legacy_observed_support_plus_side_height',
            'visible_top_measured': False, 'center_definition': 'support_plus_height',
            'nominal_minus_measured_m': None, 'source_frame_id': None,
            'quality': None,
        }
    if not isinstance(measured, dict) or measured.get('valid') is not True:
        raise ValueError('Present measured_top_surface must be valid; recapture visible cup top')
    if measured.get('source_frame_id') != side_plan.get('snapshot_frame_id'):
        raise ValueError('Visible top and side source frame IDs differ')
    for kind in ('color', 'depth'):
        key = f'snapshot_sha256_{kind}'
        _digest(side_plan.get(key), f'side {key}')
        if measured.get(key) != side_plan[key]:
            raise ValueError(f'Visible top {key} differs from side RGB-D source')
    _digest(measured.get('model_sha256'), 'YOLO model SHA-256')
    quality = measured.get('quality')
    if (not isinstance(quality, dict)
            or quality.get('rim_center_independently_measured') is not False
            or quality.get('coaxial_cup_assumption') is not True
            or measured.get('center_definition')
            != 'side_axis_intersection_with_visible_top_plane'):
        raise ValueError('Measured top point must retain side-axis/visible-plane provenance')
    top = np.asarray(measured.get('center_base_m'), dtype=float)
    if top.shape != (3,) or not np.isfinite(top).all():
        raise ValueError('Measured visible cup top must be three finite base metres')
    normal = _unit(measured.get('normal_base'), 'measured visible top normal')
    if float(normal @ axis) < math.cos(math.radians(15)):
        raise ValueError('Measured visible top plane normal differs from upright cup axis')
    difference = top-nominal
    axial = float(difference @ axis)
    lateral = float(np.linalg.norm(difference-axial*axis))
    if abs(axial) > uncertainty+.005 or lateral > .008:
        raise ValueError('Visible top disagrees with nominal cup geometry beyond uncertainty')
    return top, {
        'source': 'same_frame_yolo_rgbd_visible_top_plane_side_axis_intersection',
        'visible_top_measured': True,
        'center_definition': measured['center_definition'],
        'nominal_minus_measured_m': (nominal-top).tolist(),
        'axial_deviation_m': axial, 'lateral_deviation_m': lateral,
        'source_frame_id': measured['source_frame_id'],
        'snapshot_sha256_color': measured['snapshot_sha256_color'],
        'snapshot_sha256_depth': measured['snapshot_sha256_depth'],
        'model_sha256': measured['model_sha256'],
        'quality': quality,
        'rim_center_independently_measured': False,
    }


def _red_target_workspace(side_plan, top_provenance):
    """Bind the chosen green cup to the red-mat instance and exact color frame."""
    recognition = side_plan.get('recognition')
    if recognition is None:
        return {'verified': False, 'source': 'no_formal_red_workspace_recognition'}
    if not isinstance(recognition, dict):
        raise ValueError('Top source recognition must be an object')
    red = recognition.get('red_workspace')
    if red is None:
        return {'verified': False, 'source': 'formal_red_workspace_evidence_missing'}
    if (not isinstance(red, dict)
            or red.get('selected_green_cap_in_red_mat') is not True
            or red.get('table_support_center_inside_red_mat') is not True
            or red.get('selected_instance') != recognition.get('selected_instance')
            or red.get('source_frame_id') != side_plan.get('snapshot_frame_id')
            or red.get('snapshot_sha256_color') != side_plan.get('snapshot_sha256_color')):
        raise ValueError('Selected green cap/red mat workspace evidence disagrees')
    _digest(red.get('config_sha256'), 'Red mat workspace config SHA-256')
    if (top_provenance.get('visible_top_measured')
            and recognition.get('model_sha256') != top_provenance.get('model_sha256')):
        raise ValueError('Red-mat selected YOLO model differs from measured top model')
    return {'verified': True,
            'source': 'same_frame_selected_green_cap_inside_red_workspace',
            **red}


def _reveal_lane(side_plan, lane, top, support, axis, cup_radius, cup_height,
                 uncertainty, contact_standoff):
    """Reject invented dice ROI/exit geometry and check the held-cup volume."""
    if lane is None:
        return None
    source = lane.get('source') if isinstance(lane, dict) else None
    if source not in ('measured_dice_roi_plus_reviewed_exit_lane',
                      'measured_cup_footprint_conservative_dice_roi'):
        raise ValueError('Reveal exit lane needs a direct measured dice ROI or conservative cup-footprint proxy')
    if (not isinstance(lane, dict) or lane.get('frame') != 'base'
            or lane.get('reviewed') is not True
            or lane.get('source_frame_id') != side_plan.get('snapshot_frame_id')):
        raise ValueError('Reveal needs a reviewed ROI/proxy in this RGB-D frame')
    for kind in ('color', 'depth'):
        key = f'snapshot_sha256_{kind}'
        _digest(side_plan.get(key), f'side {key}')
        if lane.get(key) != side_plan[key]:
            raise ValueError(f'Reveal exit lane {key} differs from source RGB-D frame')
    roi = np.asarray(lane.get('dice_roi_center_base_m'), dtype=float)
    destination = np.asarray(lane.get('destination_palm_base_m'), dtype=float)
    if (roi.shape != (3,) or destination.shape != (3,)
            or not np.isfinite(roi).all() or not np.isfinite(destination).all()):
        raise ValueError('Reveal ROI center/destination must be finite base-frame 3-vectors')
    radius = float(lane.get('dice_roi_radius_m'))
    margin = float(lane.get('minimum_roi_clearance_m'))
    if (not math.isfinite(radius) or not .02 <= radius <= .30
            or not math.isfinite(margin) or not .01 <= margin <= .10):
        raise ValueError('Reveal dice ROI radius/margin outside plausible metres')
    if source == 'measured_cup_footprint_conservative_dice_roi':
        if (lane.get('dice_visibility') != 'hidden_under_inverted_cup'
                or lane.get('dice_directly_observed') is not False):
            raise ValueError('Cup-footprint ROI must identify hidden, unobserved dice')
        delta = roi-support
        axial = float(delta @ axis)
        lateral = float(np.linalg.norm(delta-axial*axis))
        if abs(axial) > .008 or lateral > uncertainty:
            raise ValueError('Proxy dice ROI center must match measured cup mouth footprint')
        if radius < cup_radius+uncertainty+.02:
            raise ValueError('Proxy dice ROI must cover cup radius, hand-eye error, and dice spread')
    # Cup is rigidly held with the mouth at the tabletop at contact.  The
    # destination mouth is this far below PALM, regardless of an arm TCP.
    mouth = destination-axis*(cup_height+contact_standoff)
    mouth_altitude = float((mouth-support) @ axis)
    lateral = destination-roi
    lateral -= axis*float(lateral @ axis)
    projected_distance = float(np.linalg.norm(lateral))
    required_distance = radius+cup_radius+uncertainty+margin
    if mouth_altitude < .075:
        raise ValueError('Reveal exit needs inverted cup mouth at least 75 mm above tabletop')
    if projected_distance < required_distance:
        raise ValueError('Held cup cylinder still overlaps measured dice visual ROI')
    return {
        **lane,
        'dice_roi_source_kind': source,
        'dice_directly_observed': source == 'measured_dice_roi_plus_reviewed_exit_lane',
        'cup_footprint_proxy': source == 'measured_cup_footprint_conservative_dice_roi',
        'held_cup_mouth_target_base_m': mouth.tolist(),
        'held_cup_mouth_altitude_from_table_m': mouth_altitude,
        'held_cup_projected_distance_from_dice_roi_m': projected_distance,
        'held_cup_required_projected_distance_m': required_distance,
        'cup_radius_m': cup_radius, 'uncertainty_envelope_m': uncertainty,
        'held_cup_volume_model': 'same_orientation_inverted_cylinder_with_mouth_below_palm',
    }


def _stage_record(name, palm, flange, q, ik, path, hand, path_hand,
                  *, contact_policy, required_clearance, margin_m):
    stage_table_passed = hand['table_margin_passed']
    path_table_passed = all(item['table_margin_passed'] for item in path_hand)
    stage_cup_passed = hand['cup_margin_passed']
    path_cup_passed = all(item['cup_margin_passed'] for item in path_hand)
    geometry_passed = (stage_table_passed and path_table_passed and
                       (stage_cup_passed and path_cup_passed if required_clearance else True))
    return {
        'name': name, 'T_base_palm': palm.tolist(), 'T_base_flange': flange.tolist(),
        'palm_pose_base_m_rad': matrix_pose(palm),
        'flange_pose_base_m_rad': matrix_pose(flange),
        'target_joints_rad': [float(v) for v in q],
        'target_joints_deg': [float(v) for v in np.degrees(q)],
        'ik': ik,
        'joint_path': {
            'kinematic_checks_passed': path.kinematic_checks_passed,
            'joint_limits_passed': path.joint_limits_passed,
            'continuity_passed': path.continuity_passed,
            'singularity_warning': path.singularity_warning,
            'sample_count': path.sample_count,
            'collision_verified': False,
            'scene_collision_verified': False,
        },
        'hand_geometry': {
            'contact_policy': contact_policy, 'model_margin_m': margin_m,
            'stage_exact_mesh_table_min_m': hand['exact_collision_mesh_table_min_m'],
            'path_exact_mesh_table_min_m': min(
                item['exact_collision_mesh_table_min_m'] for item in path_hand),
            'stage_table_margin_passed': stage_table_passed,
            'path_table_margin_passed': path_table_passed,
            'stage_cup_margin_passed': stage_cup_passed,
            'path_cup_margin_passed': path_cup_passed,
            'stage_possible_collision': hand['possible_collision'],
            'path_possible_collision': any(item['possible_collision'] for item in path_hand),
            'screen_passed': geometry_passed,
            'actual_hand_posture_verified': False,
            'controller_trajectory_verified': False,
            'cup_contact_verified': False,
        },
    }


def _held_cup_path_screen(arm, path, flange_palm, contact_palm, support, axis,
                          cup_radius, name, lane=None):
    """Joint-linear screen of the held inverted cup mouth/table/ROI.

    Real controller sweep, hand grip and tabletop load remain unverified.
    """
    local_mouth = contact_palm[:3, :3].T @ (support-contact_palm[:3, 3])
    local_axis = contact_palm[:3, :3].T @ axis
    reports = []
    for q in path.samples_rad:
        palm = matrix(arm.fk(q)) @ flange_palm
        mouth = palm[:3, :3] @ local_mouth+palm[:3, 3]
        mouth_axis = palm[:3, :3] @ local_axis
        altitude = float((mouth-support) @ axis)
        tilt = math.acos(max(-1., min(1., float(mouth_axis @ axis))))
        rim_low = altitude-cup_radius*math.sin(tilt)
        rim_high = altitude+cup_radius*math.sin(tilt)
        horizontal = mouth-support-axis*altitude
        reports.append((altitude, tilt, rim_low, rim_high,
                        float(np.linalg.norm(horizontal))))
    if not reports:
        return {'screen_passed': False, 'reason': 'no_joint_samples',
                'actual_mouth_support_verified': False}
    altitudes = [v[0] for v in reports]
    rim_lows = [v[2] for v in reports]
    rim_highs = [v[3] for v in reports]
    if name in ('tiny_shake_a', 'tiny_shake_b'):
        passed = (max(abs(v) for v in altitudes) <= .003
                  and min(rim_lows) >= -.003 and max(rim_highs) <= .005
                  and max(v[1] for v in reports) <= math.radians(3))
        policy = 'inverted_mouth_table_supported_rim_screen'
    elif name == 'reveal_lift':
        low_motion = [v[4] for v in reports if v[0] < .025]
        passed = (min(rim_lows) >= -.003 and altitudes[-1] >= .075
                  and (max(low_motion) if low_motion else 0.) <= .005
                  and reports[-1][4] <= .005)
        policy = 'raise_mouth_75mm_before_lateral_exit'
    elif name == 'reveal_exit':
        passed = min(rim_lows) >= .070 and lane is not None
        policy = 'already_lifted_mouth_outside_dice_roi'
    else:
        raise ValueError(f'Unexpected held cup stage {name}')
    return {
        'screen_passed': bool(passed), 'policy': policy,
        'joint_linear_sample_count': len(reports),
        'mouth_center_table_altitude_min_m': min(altitudes),
        'mouth_center_table_altitude_max_m': max(altitudes),
        'rim_table_altitude_min_m': min(rim_lows),
        'rim_table_altitude_max_m': max(rim_highs),
        'mouth_axis_tilt_max_deg': max(math.degrees(v[1]) for v in reports),
        'mouth_horizontal_displacement_max_m': max(v[4] for v in reports),
        'actual_mouth_support_verified': False,
        'actual_held_cup_verified': False,
        'controller_path_verified': False,
    }


def compute_top_grasp(
    side_plan, *, source_side_plan_sha256, live_joint_feedback,
    hand_open_positions, hand_status_source=None, palm_orientation_base=None,
    orientation_reviewed=False, prep_joints_rad=None,
    provisional_opt_in=False, provisional_position_error_m=PROVISIONAL_POSITION_ERROR_M,
    pretop_height_m=.065, clearance_height_m=.08, reveal_lift_height_m=.08,
    tiny_shake_amplitude_m=.003, reveal_exit_lane=None,
    snapshot_max_age_s=MAX_SNAPSHOT_AGE_S,
    require_retracted_thumb=False,
    now_ns=None,
):
    """Derive a separate top proposal from a fresh side observation and live q.

    The side plan supplies the measured cup and contemporaneous flange pose;
    `live_joint_feedback` supplies seven current joints and controller state.
    Inputs may be offline JSON, but only live provenance can set execute_ready.
    An optional prep joint vector is an explicit candidate, never inherited
    from READY_HOME or any older arm state.
    """
    compute_started_ns = time.monotonic_ns()
    if (side_plan.get('schema') != 1 or side_plan.get('kind') != 'read_only_side_grasp_proposal'
            or side_plan.get('units') != 'm_rad'
            or side_plan.get('localization', {}).get('valid') is not True):
        raise ValueError('Require a valid schema-1 measured side-grasp source plan')
    if (not isinstance(source_side_plan_sha256, str) or len(source_side_plan_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_side_plan_sha256)):
        raise ValueError('source_side_plan_sha256 must be a lowercase SHA-256 digest')
    source_checks = side_plan.get('checks', {})
    pose_source = side_plan.get('pose_source', {})
    if not isinstance(source_checks, dict) or not isinstance(pose_source, dict):
        raise ValueError('Source checks and pose provenance must be objects')
    cup = side_plan['cup_base']
    support = np.asarray(cup['support_center_m'], dtype=float)
    axis = _unit(cup['axis'], 'measured cup/table axis')
    height = float(cup['dimensions']['observed_height_m'])
    diameter = float(cup['dimensions']['observed_side_diameter_m'])
    if (support.shape != (3,) or not np.isfinite(support).all()
            or not math.isfinite(height) or not .03 <= height <= .30
            or not math.isfinite(diameter) or not .02 <= diameter <= .20):
        raise ValueError('Measured cup support/height/diameter is implausible')
    if axis[2] < .9:
        raise ValueError('Top proposal requires a measured upright cup axis')
    for value, lo, hi, name in (
        (provisional_position_error_m, .005, .015, 'provisional position error'),
        (pretop_height_m, .04, .12, 'pretop height'),
        (clearance_height_m, .025, .08, 'clearance height'),
        (reveal_lift_height_m, .075, .12, 'reveal lift height'),
        (tiny_shake_amplitude_m, 0., .01, 'tiny shake amplitude'),
        (snapshot_max_age_s, MAX_SNAPSHOT_AGE_S, 120., 'source snapshot review window'),
    ):
        if not math.isfinite(float(value)) or not lo <= float(value) <= hi:
            raise ValueError(f'{name} must be finite within {lo}..{hi} m')
    if pretop_height_m < provisional_position_error_m + .03:
        raise ValueError('Pre-top must leave at least 30 mm beyond uncertainty')
    if provisional_position_error_m < .013 and not source_checks.get('calibration_quality_passed'):
        raise ValueError('Failed hand-eye calibration requires at least 13 mm provisional envelope')
    q_current = _finite_joints(live_joint_feedback['joints_rad'], 'live current joints')
    sdk = _sdk_ranges(live_joint_feedback['sdk_limits_deg'])
    enabled = live_joint_feedback['joints_enabled']
    if not isinstance(enabled, list) or len(enabled) != 7 or any(type(x) is not bool for x in enabled):
        raise ValueError('Live feedback must contain seven boolean enable flags')
    arm_status, ctrl_mode = int(live_joint_feedback['arm_status']), int(live_joint_feedback['ctrl_mode'])
    arm = load_model()
    current_flange = pose_matrix(side_plan['current_flange_pose_base_m_rad'])
    current_palm = pose_matrix(side_plan['current_palm_pose_base_m_rad'])
    flange_palm = matrix(inverse(current_flange) @ current_palm)
    current_fk = matrix(arm.fk(q_current))
    current_fk_error_m, current_fk_error_rad = _pose_error(current_fk, current_flange)
    if current_fk_error_m > .005 or current_fk_error_rad > math.radians(2):
        raise ValueError('Live joint FK differs from source flange; recapture cup and flange')
    prep_q = (_finite_joints(prep_joints_rad, 'explicit prep joints')
              if prep_joints_rad is not None else None)
    prep_palm = (matrix(arm.fk(prep_q)) @ flange_palm
                 if prep_q is not None else None)
    orientation = (_rotation(palm_orientation_base) if palm_orientation_base is not None
                   else (prep_palm[:3, :3].copy() if prep_palm is not None
                         else current_palm[:3, :3].copy()))
    palm_z_toward_top_dot = float(orientation[:, 2] @ -axis)
    if palm_z_toward_top_dot < .85:
        raise ValueError('Palm +Z must face down toward the observed cup top (dot >= 0.85)')
    top, top_provenance = _observed_top(
        side_plan, cup, support, axis, height, float(provisional_position_error_m))
    target_workspace = _red_target_workspace(side_plan, top_provenance)
    hand_angles = _hand_angles(hand_open_positions)
    thumb_retracted = (float(hand_open_positions['thumb_tip']) <= 3.
                       and float(hand_open_positions['thumb_base']) <= 3.)
    hand_model = RightRevo2Model()
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    source_age_s = _source_age_s(source_checks, pose_source, now_ns)
    # PALM's rigid mesh projects up to +12 mm in its local palmar direction.
    # Keep the final origin target 13 mm above the measured top; that leaves
    # roughly 1 mm nominal front-surface separation.  The provisional hand-
    # eye error can erase or enlarge it, so only an observed touch may close.
    # No target presses the model palm origin into the nominal cup top.
    h0 = float(provisional_position_error_m)
    probe_heights = (h0+.015, h0+.010, h0+.005, h0)
    if any(a-b > MAX_PROBE_STEP_M+1e-12 for a, b in zip(probe_heights, probe_heights[1:])):
        raise ValueError('Probe sequence exceeds 5 mm per target step')
    start_palm = current_palm if prep_palm is None else prep_palm
    shake_dir = _unit(orientation[:, 0] - axis * float(orientation[:, 0] @ axis),
                      'horizontal palm finger direction')
    reveal_lane = _reveal_lane(
        side_plan, reveal_exit_lane, top, support, axis, diameter/2,
        float((top-support) @ axis), float(provisional_position_error_m), h0)
    positions = {
        'clearance': start_palm[:3, 3] + axis*clearance_height_m,
        'pretop': top + axis*pretop_height_m,
        'guarded_hover': top + axis*probe_heights[0],
        'probe_1': top + axis*probe_heights[1],
        'probe_2': top + axis*probe_heights[2],
        'contact_candidate': top + axis*probe_heights[3],
        'tiny_shake_a': top + axis*h0 + shake_dir*tiny_shake_amplitude_m,
        'tiny_shake_b': top + axis*h0 - shake_dir*tiny_shake_amplitude_m,
    }
    if reveal_lane is not None:
        positions['reveal_lift'] = top + axis*(h0+reveal_lift_height_m)
        positions['reveal_exit'] = np.asarray(reveal_lane['destination_palm_base_m'], dtype=float)
    stage_sequence = ((('prep',) if prep_q is not None else ()) + TOP_STAGES
                      + (REVEAL_STAGES if reveal_lane is not None else ()))
    waypoints = {}
    prior = q_current
    all_ik = all_paths = all_limits = all_approach_geometry = all_hand_geometry = True
    supported_shake_screen = reveal_path_screen = True
    approach_thumb_tip_clearance_min_m = math.inf
    contact_palm = None
    for name in stage_sequence:
        if name == 'prep':
            target_q = prep_q
            flange = matrix(arm.fk(target_q))
            palm = matrix(flange @ flange_palm)
            ik_record = {'success': True, 'source': 'caller_supplied_explicit_joint_pose',
                         'position_error_m': 0., 'orientation_error_rad': 0.,
                         'iterations': 0, 'reason': 'FK_of_explicit_prep'}
        else:
            palm = matrix(np.block([[start_palm[:3, :3] if name == 'clearance' else orientation,
                                      positions[name].reshape(3, 1)], [np.array([[0., 0., 0., 1.]])]]))
            flange = matrix(palm @ inverse(flange_palm))
            seed_name, solved, _ = _solve_bounded(arm, flange, prior)
            target_q = np.asarray(solved.joints_rad, dtype=float)
            fk_pos, fk_ori = _pose_error(arm.fk(target_q), flange)
            ik_record = {'success': bool(solved.success and fk_pos <= .001 and fk_ori <= .01),
                         'seed_name': seed_name, 'position_error_m': fk_pos,
                         'orientation_error_rad': fk_ori,
                         'iterations': solved.iterations, 'reason': solved.reason}
        path = arm.check_joint_path(prior, target_q, max_sample_step_rad=math.radians(5))
        # A failed IK's best iterate is an inspection candidate, never the target.
        # Geometry still screens it so the failure does not hide a table threat.
        conservative_margin = APPROACH_MARGIN_M + (
            provisional_position_error_m if name in ('prep', 'clearance', 'pretop') else 0.)
        conservative_margin = min(.03, conservative_margin)
        hand = check_hand_pose(
            flange, support_center_m=support, axis=axis,
            cup_diameter_m=diameter, cup_height_m=height,
            joint_positions_rad=hand_angles, margin_m=conservative_margin,
            model=hand_model)
        # Endpoints are present in the joint path; bounded 5-degree samples
        # are only a screen and cannot certify a controller/Cartesian sweep.
        path_hand = [check_hand_pose(
            arm.fk(sample), support_center_m=support, axis=axis,
            cup_diameter_m=diameter, cup_height_m=height,
            joint_positions_rad=hand_angles, margin_m=conservative_margin,
            model=hand_model) for sample in path.samples_rad]
        if not path_hand:
            path_hand = [hand]
        required_clearance = name in ('prep', 'clearance', 'pretop')
        contact_policy = ('required_clearance' if required_clearance else
                          'guarded_probe_contact_unverified' if name in (
                              'guarded_hover', 'probe_1', 'probe_2', 'contact_candidate') else
                          'tabletop_supported_inverted_cup_requires_observed_hold' if name in (
                              'tiny_shake_a', 'tiny_shake_b') else
                          'reveal_requires_observed_held_cup')
        waypoints[name] = _stage_record(
            name, palm, flange, target_q, ik_record, path, hand, path_hand,
            contact_policy=contact_policy, required_clearance=required_clearance,
            margin_m=conservative_margin)
        if name == 'contact_candidate':
            contact_palm = palm
        if name in (*REVEAL_STAGES, 'tiny_shake_a', 'tiny_shake_b'):
            mouth = _held_cup_path_screen(
                arm, path, flange_palm, contact_palm, support, axis, diameter/2,
                name, lane=reveal_lane)
            waypoints[name]['held_inverted_cup'] = mouth
            if name in ('tiny_shake_a', 'tiny_shake_b'):
                supported_shake_screen &= mouth['screen_passed']
            else:
                reveal_path_screen &= mouth['screen_passed']
        if name in ('prep', 'clearance', 'pretop'):
            all_approach_geometry &= waypoints[name]['hand_geometry']['screen_passed']
        if name == 'guarded_hover':
            approach_thumb_tip_clearance_min_m = min(
                item['tips']['thumb']['cup_signed_clearance_m']
                for item in path_hand)
        all_hand_geometry &= waypoints[name]['hand_geometry']['screen_passed']
        all_ik &= ik_record['success']
        all_paths &= path.kinematic_checks_passed
        all_limits &= _within_all_limits(arm, target_q, sdk, margin_deg=1.)
        prior = target_q
    current_limits_passed = _within_all_limits(arm, q_current, sdk)
    status_safe = arm_status == 0 and ctrl_mode == 1 and all(enabled)
    source_live = pose_source.get('kind') == 'can_feedback_read_only'
    source_fresh = source_age_s is not None and source_age_s <= snapshot_max_age_s
    hand_read_ns = (hand_status_source.get('host_read_time_ns')
                    if isinstance(hand_status_source, dict) else None)
    hand_age_s = (max(0., (now_ns-int(hand_read_ns))/1e9)
                  if hand_read_ns is not None and int(hand_read_ns) > 0
                  and now_ns >= int(hand_read_ns) else None)
    hand_live = (isinstance(hand_status_source, dict)
                 and hand_status_source.get('kind') == 'hand_feedback_read_only'
                 and hand_age_s is not None and hand_age_s <= MAX_SNAPSHOT_AGE_S)
    quality_passed = source_checks.get('calibration_quality_passed') is True
    source_provisional = source_checks.get('provisional_opt_in') is True
    provisional_allowed = quality_passed or (provisional_opt_in and source_provisional)
    thumb_clearance_threshold_m = float(provisional_position_error_m)+.005
    thumb_approach_passed = (thumb_retracted and
                             approach_thumb_tip_clearance_min_m >= thumb_clearance_threshold_m)
    live_ready = bool(status_safe and current_limits_passed and source_live and source_fresh
                      and hand_live and all_ik and all_paths and all_limits
                      and all_approach_geometry and palm_z_toward_top_dot >= .9
                      and orientation_reviewed and provisional_allowed
                      and top_provenance['visible_top_measured']
                      and target_workspace['verified'] and all_hand_geometry
                      and supported_shake_screen and reveal_lane is not None
                      and reveal_path_screen
                      and (not require_retracted_thumb or thumb_approach_passed))
    close_updates = {name: float(hand_open_positions[name]) for name in HAND_CHANNELS}
    # Keep the first grasp deliberately small, but command the thumb as well
    # as all four long fingers.  This target is not a force/hold proof.
    close_updates['thumb_base'] = min(100., close_updates['thumb_base']+3.)
    close_updates['thumb_tip'] = min(100., close_updates['thumb_tip']+5.)
    for finger in ('index', 'middle', 'ring', 'pinky'):
        close_updates[f'{finger}_finger'] = min(100., close_updates[f'{finger}_finger']+18.)
    compute_duration_s = (time.monotonic_ns()-compute_started_ns)/1e9
    return {
        'schema': 1, 'kind': 'read_only_top_grasp_proposal', 'units': 'm_rad',
        'source_side_plan_sha256': source_side_plan_sha256,
        'snapshot_frame_id': side_plan['snapshot_frame_id'],
        'calibration_sha256': side_plan.get('calibration_sha256'),
        'cup_base': cup, 'cup_top_center_base_m': top.tolist(),
        'cup_top_provenance': top_provenance,
        'target_workspace_provenance': target_workspace,
        'reveal_exit_lane': reveal_lane,
        'T_flange_palm': flange_palm.tolist(),
        'current_flange_pose_base_m_rad': matrix_pose(current_flange),
        'current_palm_pose_base_m_rad': matrix_pose(current_palm),
        'current_joints_rad': q_current.tolist(),
        'pose_source': {**pose_source, 'sdk_limits_deg': [list(pair) for pair in sdk]},
        'live_joint_feedback': live_joint_feedback,
        'hand_open_positions': {name: float(hand_open_positions[name]) for name in HAND_CHANNELS},
        'hand_status_source': hand_status_source,
        'hand_close_updates': close_updates,
        'stage_sequence': list(stage_sequence), 'waypoints': waypoints,
        'provenance': {
            'source_kind': 'new_measured_side_plan_plus_caller_live_joint_and_hand_feedback',
            'orientation_source': ('caller_supplied_palm_rotation' if palm_orientation_base is not None
                                   else 'explicit_prep_palm_rotation' if prep_q is not None
                                   else 'new_current_palm_rotation'),
            'prep_source': 'caller_supplied_seven_joint_pose' if prep_q is not None else None,
            'palm_contact_axis_local': '+Z',
            'palm_finger_extension_axis_local': '+X',
            'uncertainty_envelope_m': float(provisional_position_error_m),
            'clearance_height_m': float(clearance_height_m),
            'nominal_palm_STL_front_of_TCP_m': .012,
            'source_plan_age_upper_bound_s': source_age_s,
            'source_snapshot_age_limit_s': float(snapshot_max_age_s),
            'require_retracted_thumb': bool(require_retracted_thumb),
            'hand_feedback_age_s': hand_age_s,
            'planning_compute_duration_s': compute_duration_s,
            'hand_joint_mapping': 'official agx_arm_ctrl_single_node.py REVO2_FINGER_CONFIG',
            'hand_collision_model': hand_model.provenance,
            'actual_contact_verified': False, 'cup_held_verified': False,
            'full_arm_scene_collision_verified': False,
        },
        'checks': {
            'source_snapshot_fresh': source_fresh,
            'source_pose_live_can': source_live,
            'source_plan_age_upper_bound_s': source_age_s,
            'current_fk_position_error_m': current_fk_error_m,
            'current_fk_orientation_error_rad': current_fk_error_rad,
            'current_joints_within_sdk_urdf_documented_limits': current_limits_passed,
            'current_arm_status_normal': arm_status == 0,
            'current_control_mode_can': ctrl_mode == 1,
            'current_all_joints_enabled': all(enabled),
            'current_hand_status_live': hand_live,
            'current_hand_feedback_age_s': hand_age_s,
            'planning_compute_duration_s': compute_duration_s,
            'visible_top_measured_same_frame': top_provenance['visible_top_measured'],
            'green_cap_in_red_mat_verified': target_workspace['verified'],
            'all_stage_hand_geometry_screen_passed': bool(all_hand_geometry),
            'measured_dice_exit_lane_reviewed': reveal_lane is not None,
            'shake_mouth_support_geometry_passed': bool(supported_shake_screen),
            'reveal_held_cup_volume_path_screen_passed': bool(reveal_path_screen),
            'palm_z_toward_top_dot': palm_z_toward_top_dot,
            'palm_top_facing': palm_z_toward_top_dot >= .9,
            'orientation_reviewed': bool(orientation_reviewed),
            'calibration_quality_passed': quality_passed,
            'provisional_opt_in': bool(provisional_opt_in),
            'provisional_allowed': bool(provisional_allowed),
            'all_ik_passed': bool(all_ik),
            'all_joint_paths_kinematically_passed': bool(all_paths),
            'all_stage_joint_limits_margin_1deg_passed': bool(all_limits),
            'approach_hand_geometry_screen_passed': bool(all_approach_geometry),
            'top_approach_thumb_retracted': bool(thumb_retracted),
            'top_approach_thumb_tip_clearance_min_m': float(approach_thumb_tip_clearance_min_m),
            'top_approach_thumb_clearance_threshold_m': thumb_clearance_threshold_m,
            'top_approach_thumb_clearance_passed': bool(thumb_approach_passed),
            'probe_max_step_m': MAX_PROBE_STEP_M,
            'physical_contact_verified': False,
            'cup_held_verified': False,
            'scene_collision_verified': False,
            'execute_ready': live_ready,
        },
        'motion_sent': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side-plan', type=Path, required=True)
    parser.add_argument('--joint-feedback', type=Path, required=True,
                        help='Saved read-joints event; this command never queries CAN')
    parser.add_argument('--hand-feedback', type=Path, required=True,
                        help='Saved hand-status event with six normalized positions')
    parser.add_argument('--palm-orientation', type=Path,
                        help='JSON 3x3 base-frame rotation; otherwise use source live palm')
    parser.add_argument('--orientation-reviewed', action='store_true')
    parser.add_argument('--prep-joints-deg', type=float, nargs=7)
    parser.add_argument('--pretop-height-m', type=float, default=.065,
                        help='Measured top-axis distance for pretop, 0.04..0.12 m')
    parser.add_argument('--clearance-height-m', type=float, default=.08,
                        help='Reviewed initial axial retreat, 0.025..0.08 m')
    parser.add_argument('--snapshot-max-age-s', type=float, default=MAX_SNAPSHOT_AGE_S,
                        help='10..120 s opt-in review window; first move requires fresh RGB-D revalidation')
    parser.add_argument('--require-retracted-thumb', action='store_true',
                        help='Top approach requires open thumb channels and tip/cup clearance')
    parser.add_argument('--reveal-exit-lane', type=Path,
                        help='Reviewed same-frame dice ROI and cup exit JSON; never inferred here')
    parser.add_argument('--allow-provisional', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source_raw = args.side_plan.read_bytes()
        side = json.loads(source_raw)
        joints = json.loads(args.joint_feedback.read_text(encoding='utf-8'))
        hand_raw = args.hand_feedback.read_bytes()
        hand = json.loads(hand_raw)
        if joints.get('event') != 'read_joints' or hand.get('event') != 'hand_status':
            raise ValueError('Expected saved read_joints and hand_status event JSON')
        positions = hand.get('positions')
        if positions is None:
            positions = hand.get('finger_positions')
        if (hand.get('is_ok') is not True or positions is None
                or (hand.get('feedback_complete') is not True
                    and 'position' not in hand.get('available_feedback', []))):
            raise ValueError('Read-only Revo2 needs valid six-finger position feedback')
        read_ns = hand.get('host_read_time_ns')
        hand_source = {
            'kind': ('hand_feedback_read_only' if type(read_ns) is int and read_ns > 0
                     else 'saved_feedback_file'),
            'host_read_time_ns': read_ns if type(read_ns) is int and read_ns > 0 else None,
            'source_path': str(args.hand_feedback),
            'source_sha256': hashlib.sha256(hand_raw).hexdigest(),
        }
        rotation = (json.loads(args.palm_orientation.read_text(encoding='utf-8'))
                    if args.palm_orientation else None)
        exit_lane = (json.loads(args.reveal_exit_lane.read_text(encoding='utf-8'))
                     if args.reveal_exit_lane else None)
        result = compute_top_grasp(
            side, source_side_plan_sha256=hashlib.sha256(source_raw).hexdigest(),
            live_joint_feedback=joints, hand_open_positions=positions,
            hand_status_source=hand_source,
            palm_orientation_base=rotation,
            orientation_reviewed=args.orientation_reviewed,
            prep_joints_rad=(np.radians(args.prep_joints_deg) if args.prep_joints_deg else None),
            provisional_opt_in=args.allow_provisional,
            pretop_height_m=args.pretop_height_m,
            clearance_height_m=args.clearance_height_m,
            snapshot_max_age_s=args.snapshot_max_age_s,
            require_retracted_thumb=args.require_retracted_thumb,
            reveal_exit_lane=exit_lane)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'output': str(args.output), 'execute_ready': result['checks']['execute_ready'],
                          'stage_sequence': result['stage_sequence'],
                          'planning_compute_duration_s': round(
                              result['checks']['planning_compute_duration_s'], 3),
                          'all_ik_passed': result['checks']['all_ik_passed'],
                          'approach_geometry_passed': result['checks']['approach_hand_geometry_screen_passed'],
                          'motion_sent': False}, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(f'top grasp proposal: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
