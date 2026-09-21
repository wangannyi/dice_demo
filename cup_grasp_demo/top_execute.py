"""Opt-in, one-stage-at-a-time NERO/Revo2 top grasp executor.

The default is an offline preview. A real stage needs ``--execute`` plus the
stage-specific scene/contact/hold declarations. This module never enables,
disables, or stops the arm. It rechecks the exact RGB-D, calibration, source
plan, joint targets, URDF FK/path limits, and open/closed hand geometry before
calling the existing Nero/Revo2 low-speed ``move-j``/``hand`` demo.

Position arrival is not physical cup contact or grip. The nominal cup-top
candidate can be wrong by the provisional calibration error; the successive
probe waypoints are bounded to 5 mm along the observed cup axis. Every stage
has a separate receipt and failure after a command is reported as possibly
having moved hardware.
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

import numpy as np

from cup_grasp_demo import execute, hand_geometry, ik_review
from cup_grasp_demo.grasp import _check_frame_timestamps
from cup_grasp_demo.revalidate_target import check_stationary_cup
from nero_calibration.core import matrix, pose_matrix
from nero_revo2_control.kinematics import load_model


MOTION_STAGES = ('prep', 'clearance', 'pretop', 'guarded_hover',
                 'probe_1', 'probe_2', 'contact_candidate',
                 'tiny_shake_a', 'tiny_shake_b', 'reveal_lift', 'reveal_exit')
CONTACT_CANDIDATES = ('guarded_hover', 'probe_1', 'probe_2', 'contact_candidate')
SHAKE_STAGES = ('tiny_shake_a', 'tiny_shake_b')
ALL_STAGES = (*MOTION_STAGES, 'close')
HAND_KEYS = ('thumb_tip', 'thumb_base', 'index_finger', 'middle_finger',
             'ring_finger', 'pinky_finger')
MAX_PROBE_STEP_M = .0055
MAX_PROBE_LATERAL_M = .0015
MAX_PROBE_ROTATION_RAD = math.radians(.75)
MAX_SHAKE_STEP_M = .015
MAX_SHAKE_ROTATION_RAD = math.radians(1.)
MAX_SHAKE_JOINT_STEP_RAD = math.radians(5.)
MAX_PROBE_JOINT_STEP_RAD = math.radians(4.)
MIN_REVEAL_LIFT_M = .075
MAX_REVALIDATED_FIRST_STAGE_PLAN_AGE_S = 600.
MAX_HAND_TARGET_STEP = 30
MAX_HAND_FEEDBACK_ERROR = 3


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pose(value, label):
    return execute._pose(value, label)


def _joints(value, label):
    return execute._joints_rad(value, label)


def _matrix_error(actual, expected):
    return execute._matrix_error(actual, expected)


def _hand_positions(value, label):
    if not isinstance(value, dict) or set(value) != set(HAND_KEYS):
        raise ValueError(f'{label} requires the exact six right Revo2 channels')
    result = {}
    for key in HAND_KEYS:
        v = value[key]
        if (type(v) not in (int, float) or not math.isfinite(v)
                or abs(v-round(v)) > 1e-6 or not 0 <= v <= 100):
            raise ValueError(f'{label} {key} must be integral normalized 0..100')
        result[key] = int(round(v))
    return result


def _hand_near(actual, expected, label):
    actual = _hand_positions(actual, label)
    expected = _hand_positions(expected, 'reviewed hand positions')
    error = max(abs(actual[name] - expected[name]) for name in HAND_KEYS)
    if error > MAX_HAND_FEEDBACK_ERROR:
        raise RuntimeError(f'{label} differs from reviewed hand feedback by {error} points')
    return actual


def hand_urdf_angles(positions):
    """ROS bridge mapping, agx_arm_ctrl_single_node.py REVO2_FINGER_CONFIG.

    These are model angles, not a calibrated tactile/force or pad-contact map.
    The SDK feedback is normalized 0..100; thumb_tip saturates at 79.8.
    """
    p = _hand_positions(positions, 'Revo2 positions')
    scales = {'thumb_base': (100., 1.57), 'thumb_tip': (79.8, 1.03),
              'index_finger': (100., 1.41), 'middle_finger': (100., 1.41),
              'ring_finger': (100., 1.41), 'pinky_finger': (100., 1.41)}
    names = {'thumb_base': 'thumb_metacarpal_joint',
             'thumb_tip': 'thumb_proximal_joint',
             'index_finger': 'index_proximal_joint',
             'middle_finger': 'middle_proximal_joint',
             'ring_finger': 'ring_proximal_joint',
             'pinky_finger': 'pinky_proximal_joint'}
    return {f'right_{names[key]}': min(p[key], span) * maximum/span
            for key, (span, maximum) in scales.items()}


def _motion_sequence(plan):
    sequence = plan.get('stage_sequence')
    optional_prep = isinstance(sequence, list) and sequence[:1] == ['prep']
    optional_reveal = (isinstance(sequence, list)
                       and sequence[-2:] == ['reveal_lift', 'reveal_exit'])
    expected = list(MOTION_STAGES)
    if not optional_prep:
        expected.remove('prep')
    if not optional_reveal:
        expected.remove('reveal_lift')
        expected.remove('reveal_exit')
    if sequence != expected:
        raise ValueError('Top plan must have exact optional-prep/probe/support-shake/optional-reveal sequence')
    return tuple(sequence)


def _execution_sequence(plan):
    motion = _motion_sequence(plan)
    split = motion.index('tiny_shake_a')
    return (*motion[:split], 'close', *motion[split:])


def _waypoint(plan, name):
    item = (plan.get('waypoints') or {}).get(name)
    if not isinstance(item, dict):
        raise ValueError(f'Top plan lacks {name} waypoint')
    flange = matrix(item.get('T_base_flange'))
    palm = matrix(item.get('T_base_palm'))
    encoded = pose_matrix(_pose(item.get('flange_pose_base_m_rad'),
                                f'{name} flange pose'))
    d, r = _matrix_error(flange, encoded)
    if d > 1e-6 or r > 1e-5:
        raise ValueError(f'{name} flange matrix and pose disagree')
    d, r = _matrix_error(flange @ matrix(plan['T_flange_palm']), palm)
    if d > 1e-6 or r > 1e-5:
        raise ValueError(f'{name} palm and flange TCP transforms disagree')
    return item, flange, palm


def _cup_geometry(plan):
    cup = plan.get('cup_base')
    if not isinstance(cup, dict) or cup.get('frame') != 'base':
        raise ValueError('Top plan requires observed cup geometry in arm base')
    dims = cup.get('dimensions') or {}
    support = np.asarray(cup.get('support_center_m'), dtype=float)
    axis = np.asarray(cup.get('axis'), dtype=float)
    if (support.shape != (3,) or axis.shape != (3,)
            or not np.isfinite(support).all() or not np.isfinite(axis).all()
            or abs(float(np.linalg.norm(axis))-1) > 1e-3 or axis[2] < .9):
        raise ValueError('Top plan requires finite support center and unit cup axis')
    diameter = float(dims.get('observed_side_diameter_m', math.nan))
    height = float(dims.get('observed_height_m', math.nan))
    if not (.06 <= diameter <= .09 and .05 <= height <= .09):
        raise ValueError('Observed top-grasp cup size must match measured 6..9 cm diameter/5..9 cm height')
    return support, axis/np.linalg.norm(axis), diameter, height


def _check_probe_steps(plan):
    support, cup_axis, _, cup_height = _cup_geometry(plan)
    names = ('guarded_hover', 'probe_1', 'probe_2', 'contact_candidate')
    heights = []
    nominal_top = support + cup_axis*cup_height
    top = np.asarray(plan.get('cup_top_center_base_m'), dtype=float)
    if top.shape != (3,) or not np.isfinite(top).all():
        raise ValueError('Top plan must identify a finite base-frame cup-top point')
    for name in names:
        _, _, palm = _waypoint(plan, name)
        heights.append(palm[:3, 3])
    provenance = plan.get('provenance') or {}
    uncertainty = float(provenance.get('uncertainty_envelope_m', math.nan))
    mesh_front = float(provenance.get('nominal_palm_STL_front_of_TCP_m', math.nan))
    if (not math.isfinite(uncertainty) or not .005 <= uncertainty <= .015
            or not math.isfinite(mesh_front) or not .005 <= mesh_front <= .02):
        raise ValueError('Top plan needs measured provisional and palm STL geometry provenance')
    difference = top-nominal_top
    axial_error = float(difference @ cup_axis)
    lateral_error = float(np.linalg.norm(difference-axial_error*cup_axis))
    if abs(axial_error) > uncertainty+.005 or lateral_error > .008:
        raise ValueError('Planned cup-top point exceeds observed upright geometry/uncertainty envelope')
    nominal = [float((point-top) @ cup_axis) for point in heights]
    if (nominal[0]+1e-6 < mesh_front+uncertainty+.002
            or nominal[-1]+1e-6 < mesh_front+.001):
        raise ValueError('Top hover/contact candidate cannot drive palm mesh into nominal cup top')
    for prior, later, first, second in zip(heights[:-1], heights[1:], names[:-1], names[1:]):
        step = later-prior
        axial = -float(step @ cup_axis)
        lateral = float(np.linalg.norm(step + axial*cup_axis))
        if not (0 < axial <= MAX_PROBE_STEP_M and lateral <= MAX_PROBE_LATERAL_M):
            raise ValueError(f'{first}->{second} probe must advance 0..5 mm along cup axis')
        previous_flange, next_flange = _waypoint(plan, first)[1], _waypoint(plan, second)[1]
        _, rotation = _matrix_error(previous_flange, next_flange)
        if rotation > MAX_PROBE_ROTATION_RAD:
            raise ValueError(f'{first}->{second} probe rotates too far')


def _check_shake_steps(plan):
    axis = _cup_geometry(plan)[1]
    for first, second in (('contact_candidate', 'tiny_shake_a'),
                          ('tiny_shake_a', 'tiny_shake_b')):
        a, b = _waypoint(plan, first)[1], _waypoint(plan, second)[1]
        d, r = _matrix_error(a, b)
        if d > MAX_SHAKE_STEP_M or r > MAX_SHAKE_ROTATION_RAD:
            raise ValueError(f'{first}->{second} exceeds tiny-shake amplitude')
        if abs(float((b[:3, 3]-a[:3, 3]) @ axis)) > .002:
            raise ValueError(f'{first}->{second} cannot lift unsupported cup mouth while shaking')
        a_q = _joints(plan['waypoints'][first].get('target_joints_rad'), first)
        b_q = _joints(plan['waypoints'][second].get('target_joints_rad'), second)
        if max(abs(x-y) for x, y in zip(a_q, b_q)) > MAX_SHAKE_JOINT_STEP_RAD:
            raise ValueError(f'{first}->{second} exceeds tiny-shake joint amplitude')


def _check_reveal(plan, metadata):
    """Require staged lift then exit from measured dice ROI or explicit cup proxy."""
    if 'reveal_lift' not in _motion_sequence(plan):
        return
    lane = plan.get('reveal_exit_lane')
    source = lane.get('source') if isinstance(lane, dict) else None
    if (not isinstance(lane, dict) or lane.get('frame') != 'base'
            or source not in ('measured_dice_roi_plus_reviewed_exit_lane',
                              'measured_cup_footprint_conservative_dice_roi')
            or lane.get('reviewed') is not True):
        raise ValueError('Reveal requires direct measured dice ROI or explicit cup-footprint proxy')
    if (lane.get('source_frame_id') != plan.get('snapshot_frame_id')
            or lane.get('snapshot_sha256_color') != metadata.get('sha256_color')
            or lane.get('snapshot_sha256_depth') != metadata.get('sha256_depth')):
        raise ValueError('Reveal dice ROI is not bound to this exact RGB-D frame')
    center = np.asarray(lane.get('dice_roi_center_base_m'), dtype=float)
    destination = np.asarray(lane.get('destination_palm_base_m'), dtype=float)
    if (center.shape != (3,) or destination.shape != (3,)
            or not np.isfinite(center).all() or not np.isfinite(destination).all()):
        raise ValueError('Reveal ROI center/destination must be finite base-frame 3-points')
    radius = float(lane.get('dice_roi_radius_m', math.nan))
    minimum = float(lane.get('minimum_roi_clearance_m', math.nan))
    if (not math.isfinite(radius) or not .02 <= radius <= .30
            or not math.isfinite(minimum) or not .01 <= minimum <= .20):
        raise ValueError('Reveal measured dice ROI/clearance dimensions are implausible')
    support, axis, diameter, _ = _cup_geometry(plan)
    _, _, reveal_lift = _waypoint(plan, 'reveal_lift')
    _, _, reveal_exit = _waypoint(plan, 'reveal_exit')
    _, _, contact_palm = _waypoint(plan, 'contact_candidate')
    point = reveal_exit[:3, 3]
    if float(np.linalg.norm(point-destination)) > .005:
        raise ValueError('Reveal destination differs from reviewed exit lane')
    lift_delta = reveal_lift[:3, 3]-contact_palm[:3, 3]
    lift = float(lift_delta @ axis)
    lift_lateral = lift_delta-lift*axis
    if lift < MIN_REVEAL_LIFT_M:
        raise ValueError('Reveal does not lift inverted cup clear of dice/table region')
    if float(np.linalg.norm(lift_lateral)) > .005:
        raise ValueError('Reveal lift must rise along cup axis before lateral exit')
    move = point-reveal_lift[:3, 3]
    if float(move @ axis) < -.005:
        raise ValueError('Reveal exit cannot lower inverted cup back toward table')
    local_mouth = contact_palm[:3, :3].T @ (support-contact_palm[:3, 3])
    actual_mouth = reveal_exit[:3, :3] @ local_mouth+point
    mouth_altitude = float((actual_mouth-support) @ axis)
    if mouth_altitude < MIN_REVEAL_LIFT_M:
        raise ValueError('Reveal exit held-cup mouth lacks axial table clearance')
    difference = actual_mouth-center
    lateral = difference-(difference @ axis)*axis
    uncertainty = float((plan.get('provenance') or {}).get('uncertainty_envelope_m', math.nan))
    if source == 'measured_cup_footprint_conservative_dice_roi':
        if (lane.get('dice_visibility') != 'hidden_under_inverted_cup'
                or lane.get('dice_directly_observed') is not False
                or lane.get('cup_footprint_proxy') is not True
                or lane.get('dice_roi_source_kind') != source):
            raise ValueError('Proxy dice ROI must state hidden, unobserved dice and cup-footprint provenance')
        center_delta = center-support
        center_axial = float(center_delta @ axis)
        center_lateral = float(np.linalg.norm(center_delta-center_axial*axis))
        if (not math.isfinite(uncertainty) or abs(center_axial) > .008
                or center_lateral > uncertainty):
            raise ValueError('Proxy dice ROI center differs from measured cup mouth footprint')
        if radius < diameter/2+uncertainty+.02:
            raise ValueError('Proxy dice ROI must cover cup radius, calibration error, and dice spread')
    elif (lane.get('dice_roi_source_kind') != source
          or lane.get('dice_directly_observed') is not True
          or lane.get('cup_footprint_proxy') is not False):
        raise ValueError('Direct measured dice ROI must retain directly observed provenance')
    if (not math.isfinite(uncertainty)
            or float(np.linalg.norm(lateral)) < radius+diameter/2+uncertainty+minimum):
        raise ValueError('Revealed inverted cup body does not exit measured dice vision ROI')


def load_top_plan(args, *, now_ns):
    """Bind exact read-only plan and RGB-D/calibration assets before any action."""
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    if (plan.get('schema') != 1 or plan.get('kind') != 'read_only_top_grasp_proposal'
            or plan.get('units') != 'm_rad'):
        raise ValueError('Expected metres/radians top-grasp plan schema 1')
    sequence = _execution_sequence(plan)
    if args.stage not in sequence:
        raise ValueError(f'{args.stage} is not in this top-grasp plan')
    checks = plan.get('checks')
    if not isinstance(checks, dict) or checks.get('execute_ready') is not True:
        raise ValueError('Top plan is not execute_ready; no stage may be sent')
    if (checks.get('source_pose_live_can') is not True
            or (plan.get('pose_source') or {}).get('kind') != 'can_feedback_read_only'
            or plan['pose_source'].get('channel') != args.channel):
        raise ValueError('Top plan needs fresh read-only CAN feedback on this channel')
    for gate in ('source_snapshot_fresh', 'current_arm_status_normal',
                 'current_control_mode_can', 'current_all_joints_enabled',
                 'current_hand_status_live', 'visible_top_measured_same_frame',
                 'green_cap_in_red_mat_verified',
                 'all_stage_hand_geometry_screen_passed',
                 'measured_dice_exit_lane_reviewed',
                 'shake_mouth_support_geometry_passed',
                 'reveal_held_cup_volume_path_screen_passed',
                 'palm_top_facing',
                 'orientation_reviewed', 'all_ik_passed',
                 'all_joint_paths_kinematically_passed',
                 'all_stage_joint_limits_margin_1deg_passed',
                 'approach_hand_geometry_screen_passed'):
        if checks.get(gate) is not True:
            raise ValueError(f'Top plan prerequisite {gate} is not true')
    if (plan.get('provenance') or {}).get('require_retracted_thumb') is True:
        if (checks.get('top_approach_thumb_retracted') is not True
                or checks.get('top_approach_thumb_clearance_passed') is not True):
            raise ValueError('Top plan needs retracted thumb and independent cup clearance')
        baseline_thumb = plan.get('hand_open_positions') or {}
        if any(float(baseline_thumb.get(name, 100)) > 3.
               for name in ('thumb_tip', 'thumb_base')):
            raise ValueError('Top plan thumb baseline is not open 0..3')
    if checks.get('physical_contact_verified') is not False:
        raise ValueError('Top planner must not assert physical cup contact')
    calibration = json.loads(args.calibration.read_bytes())
    quality = calibration.get('quality_passed')
    if type(quality) is not bool or checks.get('calibration_quality_passed') is not quality:
        raise ValueError('Top plan/calibration quality provenance disagrees')
    if not quality and not (args.allow_provisional and checks.get('provisional_opt_in') is True):
        raise ValueError('Provisional hand-eye needs both plan and action --allow-provisional')
    if plan.get('calibration_sha256') != _sha256(args.calibration):
        raise ValueError('Calibration SHA256 differs from top plan')
    if calibration.get('tcp') != 'palm':
        raise ValueError('Top grasp requires palm TCP calibration')
    d, r = _matrix_error(plan.get('T_flange_palm'), calibration.get('T_flange_tcp'))
    if d > 1e-6 or r > 1e-5:
        raise ValueError('Top plan palm TCP differs from calibrated flange-to-palm')
    source_sha = plan.get('source_side_plan_sha256')
    if not isinstance(source_sha, str) or len(source_sha) != 64 or args.source_side_plan is None:
        raise ValueError('Top plan requires exact --source-side-plan SHA256 provenance')
    if _sha256(args.source_side_plan) != source_sha:
        raise ValueError('Source side plan SHA256 differs from top plan')
    source_plan = json.loads(args.source_side_plan.read_bytes())
    if (source_plan.get('schema') != 1 or
            source_plan.get('kind') != 'read_only_side_grasp_proposal'
            or source_plan.get('snapshot_frame_id') != plan.get('snapshot_frame_id')
            or source_plan.get('calibration_sha256') != plan.get('calibration_sha256')):
        raise ValueError('Top plan frame/calibration differs from source side plan')
    if source_plan.get('cup_base') != plan.get('cup_base'):
        raise ValueError('Top cup geometry differs from exact source side observation')
    workspace = (source_plan.get('recognition') or {}).get('red_workspace')
    top_workspace = plan.get('target_workspace_provenance')
    if (not isinstance(workspace, dict)
            or workspace.get('selected_green_cap_in_red_mat') is not True
            or workspace.get('table_support_center_inside_red_mat') is not True
            or workspace.get('selected_instance')
            != (source_plan.get('recognition') or {}).get('selected_instance')
            or not isinstance(top_workspace, dict)
            or top_workspace.get('verified') is not True
            or any(top_workspace.get(key) != value for key, value in workspace.items())):
        raise ValueError('Selected green cup/red mat workspace differs from source plan')
    config_sha = workspace.get('config_sha256')
    if (not isinstance(config_sha, str) or len(config_sha) != 64
            or any(c not in '0123456789abcdef' for c in config_sha)):
        raise ValueError('Selected red mat workspace config SHA256 is invalid')
    for kind in ('color', 'depth'):
        key = f'snapshot_sha256_{kind}'
        if source_plan.get(key) != plan['cup_base'].get('measured_top_surface', {}).get(key):
            raise ValueError(f'Measured visible cup top differs from source {kind} frame')
    source_flange = source_plan.get('current_flange_pose_base_m_rad')
    if source_flange is None or any(abs(a-b) > 1e-6 for a, b in zip(
            _pose(source_flange, 'source flange'),
            _pose(plan.get('current_flange_pose_base_m_rad'), 'top initial flange'))):
        raise ValueError('Top initial flange differs from exact source side observation')
    snap = Path(args.snapshot_dir)
    metadata = json.loads((snap/'metadata.json').read_bytes())
    if (metadata.get('schema') != 1 or metadata.get('camera_backend') != 'realsense'
            or metadata.get('frame') != 'color_optical'
            or metadata.get('depth_registered_to') != 'color_optical'):
        raise ValueError('Snapshot is not registered RealSense color_optical RGB-D')
    _check_frame_timestamps(metadata.get('timestamp_ms'), metadata.get('depth_timestamp_ms'),
                            metadata.get('timestamp_domain'), metadata.get('depth_timestamp_domain'))
    if (_sha256(snap/'color.png') != metadata.get('sha256_color')
            or _sha256(snap/'depth.npz') != metadata.get('sha256_depth')):
        raise ValueError('Top snapshot RGB-D hashes differ from capture metadata')
    source_localization = source_plan.get('localization') or {}
    if (metadata.get('frame_id') != plan.get('snapshot_frame_id')
            or metadata.get('serial') != source_localization.get('camera_serial')
            or metadata.get('frame_id') != source_localization.get('frame_id')
            or source_localization.get('valid') is not True
            or (plan.get('cup_base') or {}).get('source', {}).get('frame_id') != metadata.get('frame_id')):
        raise ValueError('Top plan, cup localization, and snapshot frame identities differ')
    if (workspace.get('source_frame_id') != metadata['frame_id']
            or workspace.get('snapshot_sha256_color') != metadata['sha256_color']):
        raise ValueError('Selected red mat workspace differs from RGB capture frame')
    measured = plan['cup_base'].get('measured_top_surface')
    top_source = plan.get('cup_top_provenance')
    if (not isinstance(measured, dict) or measured.get('valid') is not True
            or measured.get('source_frame_id') != metadata['frame_id']
            or measured.get('snapshot_sha256_color') != metadata['sha256_color']
            or measured.get('snapshot_sha256_depth') != metadata['sha256_depth']
            or measured.get('center_definition')
            != 'side_axis_intersection_with_visible_top_plane'
            or not isinstance(measured.get('quality'), dict)
            or measured['quality'].get('rim_center_independently_measured') is not False
            or measured['quality'].get('coaxial_cup_assumption') is not True
            or not isinstance(top_source, dict)
            or top_source.get('visible_top_measured') is not True
            or top_source.get('source_frame_id') != metadata['frame_id']):
        raise ValueError('Top execution requires same-frame measured visible RGB-D top provenance')
    model_sha = measured.get('model_sha256')
    if (not isinstance(model_sha, str) or len(model_sha) != 64
            or any(c not in '0123456789abcdef' for c in model_sha)
            or top_source.get('model_sha256') != model_sha
            or (source_plan.get('recognition') or {}).get('model_sha256') != model_sha):
        raise ValueError('Measured YOLO model SHA256 differs from top plan provenance')
    for kind in ('color', 'depth'):
        key = f'snapshot_sha256_{kind}'
        if top_source.get(key) != metadata[f'sha256_{kind}']:
            raise ValueError('Visible cup top plan provenance differs from RGB-D snapshot')
    top_point = np.asarray(plan.get('cup_top_center_base_m'), dtype=float)
    measured_point = np.asarray(measured.get('center_base_m'), dtype=float)
    if (top_point.shape != (3,) or measured_point.shape != (3,)
            or not np.isfinite(top_point).all() or not np.isfinite(measured_point).all()
            or float(np.linalg.norm(top_point-measured_point)) > 1e-6):
        raise ValueError('Planned cup top differs from measured visible top point')
    _cup_geometry(plan)
    _check_probe_steps(plan)
    _check_shake_steps(plan)
    _check_reveal(plan, metadata)
    _hand_positions(plan.get('hand_open_positions'), 'planned open hand')
    close_targets = _close_targets(plan.get('hand_close_updates'),
                                   _hand_positions(plan['hand_open_positions'],
                                                   'planned open hand'))
    if all(close_targets[k] == plan['hand_open_positions'][k] for k in HAND_KEYS):
        raise ValueError('Top plan has no close finger change')
    for name in _motion_sequence(plan):
        hand = (plan['waypoints'][name].get('hand_geometry') or {})
        if hand.get('screen_passed') is not True:
            raise ValueError(f'{name} top plan hand geometry screen is not passed')
        if hand.get('actual_hand_posture_verified') is not False:
            raise ValueError(f'{name} plan must not claim actual hand posture verified')
        if name in (*SHAKE_STAGES, 'reveal_lift', 'reveal_exit') and (
                (plan['waypoints'][name].get('held_inverted_cup') or {}).get('screen_passed')
                is not True):
            raise ValueError(f'{name} plan lacks held inverted cup path screen')
    observed_model = (plan.get('provenance') or {}).get('hand_collision_model')
    if observed_model != hand_geometry.RightRevo2Model().provenance:
        raise ValueError('Official right Revo2 hand geometry changed since top plan')
    if args.stage == sequence[0]:
        snapshot_age = execute._age_s(metadata.get('host_capture_time_ns'), now_ns)
        pose_age = execute._age_s(plan['pose_source'].get('host_read_time_ns'), now_ns)
        revalidate = getattr(args, 'revalidation_dir', None) is not None
        age_limit = (MAX_REVALIDATED_FIRST_STAGE_PLAN_AGE_S if revalidate
                     and (plan.get('provenance') or {}).get('source_snapshot_age_limit_s', 10)
                     > execute.MAX_SNAPSHOT_AGE_S else execute.MAX_SNAPSHOT_AGE_S)
        if snapshot_age > age_limit or pose_age > age_limit:
            raise ValueError('First top stage RGB-D or CAN pose is stale; recapture and replan')
    else:
        snapshot_age = execute._age_s(metadata.get('host_capture_time_ns'), now_ns)
    identity = {'plan_sha256': hashlib.sha256(raw).hexdigest(),
                'source_side_plan_sha256': source_sha,
                'calibration_sha256': plan['calibration_sha256'],
                'snapshot_frame_id': metadata['frame_id'],
                'snapshot_sha256_color': metadata['sha256_color'],
                'snapshot_sha256_depth': metadata['sha256_depth']}
    return plan, sequence, identity, snapshot_age, close_targets


def _sdk_within(q, sdk, label):
    for axis, (angle, pair) in enumerate(zip(q, sdk), 1):
        degrees = math.degrees(angle)
        if not pair[0] <= degrees <= pair[1]:
            raise ValueError(f'{label} J{axis} outside reviewed SDK soft limits')


def independent_kinematics(plan):
    """Recompute URDF IK/FK and joint-linear limits from plan numbers."""
    model = load_model()
    q0 = _joints(plan.get('current_joints_rad'), 'planned current joints')
    sdk = execute._sdk_ranges((plan.get('pose_source') or {}).get('sdk_limits_deg'))
    if not model.within_limits(q0):
        raise ValueError('Planned initial joints exceed conservative URDF limits')
    _sdk_within(q0, sdk, 'current')
    initial = pose_matrix(_pose(plan.get('current_flange_pose_base_m_rad'),
                                'planned current flange'))
    d, r = _matrix_error(model.fk(q0), initial)
    if d > ik_review.MAX_CURRENT_FK_POSITION_ERROR_M or r > ik_review.MAX_CURRENT_FK_ORIENTATION_ERROR_RAD:
        raise ValueError('Independent current URDF FK disagrees with planned flange feedback')
    targets = {}
    previous = q0
    for name in _motion_sequence(plan):
        item, flange, _ = _waypoint(plan, name)
        q = _joints(item.get('target_joints_rad'), f'{name} joint target')
        if (not model.within_limits(q)
                or not all(lower <= math.degrees(a) <= upper
                           for a, (lower, upper) in zip(q, ik_review.DOCUMENTED_LIMITS_DEG))):
            raise ValueError(f'{name} target exceeds URDF/documented joint limits')
        _sdk_within(q, sdk, name)
        d, r = _matrix_error(model.fk(q), flange)
        if d > ik_review.POSITION_IK_TOLERANCE_M or r > ik_review.ORIENTATION_IK_TOLERANCE_RAD:
            raise ValueError(f'{name} independent URDF FK fails target matrix')
        # Verify inverse feasibility from the chosen reviewed branch. The
        # separate FK/limits checks above prevent a self-reported IK success
        # field from being accepted as evidence.
        solved = model.ik(flange, q, position_tolerance_m=ik_review.POSITION_IK_TOLERANCE_M,
                          orientation_tolerance_rad=ik_review.ORIENTATION_IK_TOLERANCE_RAD)
        if not solved.success:
            raise ValueError(f'{name} independent URDF IK fails from reviewed branch')
        path = model.check_joint_path(previous, q)
        if not path.kinematic_checks_passed:
            raise ValueError(f'{name} independent joint interpolation fails limits/continuity')
        if (item.get('ik') or {}).get('success') is not True:
            raise ValueError(f'{name} plan did not report successful IK')
        if name in CONTACT_CANDIDATES[1:] and max(
                abs(a-b) for a, b in zip(previous, q)) > MAX_PROBE_JOINT_STEP_RAD:
            raise ValueError(f'{name} probe exceeds modest joint step')
        targets[name] = q
        previous = q
    return model, q0, sdk, targets


def _screen_cup_height(plan, support, axis, nominal_height):
    """Extend the cylinder to the visible top plane when supplied by vision."""
    visible = np.asarray(plan.get('cup_top_center_base_m'), dtype=float)
    if visible.shape != (3,) or not np.isfinite(visible).all():
        raise ValueError('Visible top must be a finite base-frame 3-point')
    height = float((visible-support) @ axis)
    uncertainty = float((plan.get('provenance') or {}).get('uncertainty_envelope_m', math.nan))
    if (not math.isfinite(uncertainty)
            or abs(height-nominal_height) > uncertainty+.005
            or not .05 <= height <= .09):
        raise ValueError('Visible cup-top height disagrees with upright source geometry')
    return height


def _geometry_screen(plan, model, start_q, end_q, hand_positions, stage):
    """Recheck official hand mesh/table and cup screen with current hand q."""
    support, axis, diameter, nominal_height = _cup_geometry(plan)
    height = _screen_cup_height(plan, support, axis, nominal_height)
    joints = hand_urdf_angles(hand_positions)
    revo = hand_geometry.RightRevo2Model()
    path = model.check_joint_path(start_q, end_q)
    if not path.kinematic_checks_passed:
        raise ValueError(f'{stage} path failed current independent URDF check')
    closest_table = math.inf
    closest_cup = math.inf
    closest_thumb = math.inf
    cup_mouth_altitudes = []
    cup_mouth_lateral = []
    contact_palm = _waypoint(plan, 'contact_candidate')[2]
    reference_palm = contact_palm[:3, 3]
    local_mouth = contact_palm[:3, :3].T @ (support-reference_palm)
    local_axis = contact_palm[:3, :3].T @ axis
    mouth_rim_lows = []
    mouth_rim_highs = []
    mouth_tilts = []
    low_laterals = []
    require_cup_clearance = stage in ('prep', 'clearance', 'pretop')
    uncertainty = float((plan.get('provenance') or {}).get('uncertainty_envelope_m', math.nan))
    if not math.isfinite(uncertainty) or not .005 <= uncertainty <= .015:
        raise ValueError('Top plan must retain 5..15 mm provisional calibration uncertainty')
    margin = min(.03, .005 + uncertainty) if require_cup_clearance else .005
    for q in path.samples_rad:
        palm = np.asarray(model.fk(q)) @ matrix(plan['T_flange_palm'])
        mouth = palm[:3, :3] @ local_mouth+palm[:3, 3]
        mouth_axis = palm[:3, :3] @ local_axis
        mouth_altitude = float((mouth-support) @ axis)
        tilt = math.acos(max(-1., min(1., float(mouth_axis @ axis))))
        rim_low = mouth_altitude-(diameter/2)*math.sin(tilt)
        rim_high = mouth_altitude+(diameter/2)*math.sin(tilt)
        horizontal = mouth-support-axis*mouth_altitude
        lateral = float(np.linalg.norm(horizontal))
        cup_mouth_altitudes.append(mouth_altitude)
        cup_mouth_lateral.append(lateral)
        mouth_rim_lows.append(rim_low)
        mouth_rim_highs.append(rim_high)
        mouth_tilts.append(tilt)
        if mouth_altitude < .025:
            low_laterals.append(lateral)
        screen = hand_geometry.check_hand_pose(
            model.fk(q), support_center_m=support, axis=axis,
            cup_diameter_m=diameter, cup_height_m=height,
            joint_positions_rad=joints, margin_m=margin, model=revo)
        closest_table = min(closest_table, screen['exact_collision_mesh_table_min_m'])
        closest_cup = min(closest_cup,
                          min(v['cup_signed_clearance_m'] for v in screen['tips'].values()))
        closest_thumb = min(closest_thumb,
                            screen['tips']['thumb']['cup_signed_clearance_m'])
        if not screen['table_margin_passed']:
            raise ValueError(f'{stage} independent hand mesh path reaches table margin')
        if require_cup_clearance and not screen['cup_margin_passed']:
            raise ValueError(f'{stage} independent open hand path reaches cup margin')
    if ((plan.get('provenance') or {}).get('require_retracted_thumb') is True
            and stage in ('pretop', 'guarded_hover')):
        if any(float(hand_positions[name]) > 3. for name in ('thumb_tip', 'thumb_base')):
            raise ValueError('Top approach live thumb is not retracted')
        if closest_thumb < uncertainty+.005:
            raise ValueError('Top approach thumb tip enters provisional cup envelope')
    if stage in SHAKE_STAGES and (
            max(abs(v) for v in cup_mouth_altitudes) > .003
            or min(mouth_rim_lows) < -.003 or max(mouth_rim_highs) > .005
            or max(mouth_tilts) > math.radians(3.)):
        raise ValueError(f'{stage} joint-screened cup mouth lifts or presses table')
    if stage == 'reveal_lift' and (
            min(mouth_rim_lows) < -.003
            or cup_mouth_altitudes[-1] < MIN_REVEAL_LIFT_M
            or (max(low_laterals) if low_laterals else 0.) > .005
            or cup_mouth_lateral[-1] > .005
            or max(mouth_tilts) > math.radians(3.)):
        raise ValueError('Reveal lift joint-screened cup drags or fails axial tabletop clearance')
    if stage == 'reveal_exit' and (min(mouth_rim_lows) < .070
                                    or max(mouth_tilts) > math.radians(3.)):
        raise ValueError('Reveal exit joint-screened cup returns toward dice/table')
    return {'sample_count': len(path.samples_rad),
            'min_exact_hand_mesh_table_m': closest_table,
            'min_fingertip_cup_clearance_m': closest_cup,
            'min_thumb_tip_cup_clearance_m': closest_thumb,
            'cup_clearance_required': require_cup_clearance,
            'mouth_support_projection': {
                'reference': 'nominal_rigid_grip_at_contact_candidate_palm',
                'min_mouth_altitude_m': min(cup_mouth_altitudes),
                'max_mouth_altitude_m': max(cup_mouth_altitudes),
                'max_mouth_lateral_m': max(cup_mouth_lateral),
                'min_rim_altitude_m': min(mouth_rim_lows),
                'max_rim_altitude_m': max(mouth_rim_highs),
                'max_mouth_axis_tilt_deg': max(math.degrees(v) for v in mouth_tilts),
                'controller_path_verified': False},
            'model_provenance': revo.provenance,
            'hand_posture_source': 'live_right_revo2_position_feedback_ros_bridge_map'}


def _close_geometry_screen(plan, flange_pose, start_hand, target_hand):
    """Screen the full six-channel curl from live pose to a modest target.

    The official URDF STL/table intersection is checked at several sampled
    hand positions. Cup overlap is intentional only after independent scene
    contact confirmation; it is never accepted as proof of grip.
    """
    start = _hand_positions(start_hand, 'live before-close hand')
    target = _hand_positions(target_hand, 'reviewed close target')
    support, axis, diameter, nominal_height = _cup_geometry(plan)
    height = _screen_cup_height(plan, support, axis, nominal_height)
    revo = hand_geometry.RightRevo2Model()
    minimum = math.inf
    samples = max(2, 1+math.ceil(max(abs(target[k]-start[k]) for k in HAND_KEYS)/4))
    for alpha in np.linspace(0., 1., samples):
        positions = {name: int(round(start[name]*(1-alpha)+target[name]*alpha))
                     for name in HAND_KEYS}
        screen = hand_geometry.check_hand_pose(
            pose_matrix(flange_pose), support_center_m=support, axis=axis,
            cup_diameter_m=diameter, cup_height_m=height,
            joint_positions_rad=hand_urdf_angles(positions),
            margin_m=.005, model=revo)
        minimum = min(minimum, screen['exact_collision_mesh_table_min_m'])
        if not screen['table_margin_passed']:
            raise ValueError('Close hand mesh/fingertip trajectory reaches tabletop margin')
    return {'sample_count': samples, 'min_exact_hand_mesh_table_m': minimum,
            'cup_overlap_policy': 'physical_contact_observed_by_scene_required',
            'model_provenance': revo.provenance,
            'hand_posture_source': 'live_right_revo2_position_feedback_ros_bridge_map'}


def _allowed_next(plan, receipt):
    """A verified observed contact can end the remaining tiny probes early."""
    sequence = _execution_sequence(plan)
    done = tuple(receipt.get('stages', {}))
    if not done:
        return {sequence[0]}
    if any(receipt['stages'][name].get('verified') is not True for name in done):
        raise ValueError('Top receipt includes an unverified stage')
    last = done[-1]
    if last == 'close':
        return {'tiny_shake_a'}
    if last in CONTACT_CANDIDATES:
        choices = {'close'}
        if last != 'contact_candidate':
            choices.add(sequence[sequence.index(last)+1])
        return choices
    if last == 'tiny_shake_b' and 'reveal_lift' not in sequence:
        return set()
    if last == 'reveal_exit':
        return set()
    return {sequence[sequence.index(last)+1]}


def load_receipt(args, plan, identity, *, now_ns):
    first = _execution_sequence(plan)[0]
    path = Path(args.receipt)
    if not path.exists():
        if args.stage != first:
            raise ValueError(f'{args.stage} requires a verified {first} top-grasp receipt')
        return {'schema': 1, 'kind': 'top_grasp_stage_receipt', **identity,
                'stages': {}, 'motion_mode': 'reviewed_move_j'}
    receipt = json.loads(path.read_bytes())
    if receipt.get('schema') != 1 or receipt.get('kind') != 'top_grasp_stage_receipt':
        raise ValueError('Invalid top-grasp receipt kind')
    if any(receipt.get(key) != value for key, value in identity.items()):
        raise ValueError('Top receipt belongs to a different plan/RGB-D/calibration')
    if receipt.get('motion_mode') != 'reviewed_move_j' or not isinstance(receipt.get('stages'), dict):
        raise ValueError('Top receipt arm mode/stages are invalid')
    # Replay the actual sequence through the same transition rule. One can
    # close after any genuinely observed probe and omit later probes; no
    # motion stage may be inserted, repeated, or reordered.
    validated = {**receipt, 'stages': {}}
    last_completed_ns = None
    for name, item in receipt['stages'].items():
        if name not in _allowed_next(plan, validated) or not isinstance(item, dict):
            raise ValueError('Top receipt has out-of-order or repeated stages')
        validated['stages'][name] = item
        if item.get('verified') is not True:
            raise ValueError('Top receipt contains an unverified stage')
        completed_ns = item.get('completed_at_ns')
        if (type(completed_ns) is not int or completed_ns <= 0
                or (last_completed_ns is not None and completed_ns < last_completed_ns)):
            raise ValueError('Top receipt stage completion timestamps are invalid/out of order')
        last_completed_ns = completed_ns
        expected_event = 'hand_target_reached' if name == 'close' else 'arm_target_reached'
        if item.get('feedback_event') != expected_event:
            raise ValueError('Top receipt lacks required fresh arrival feedback')
        if (name == first and (plan.get('provenance') or {}).get(
                'source_snapshot_age_limit_s', 10.) > execute.MAX_SNAPSHOT_AGE_S):
            refresh = item.get('target_revalidation')
            if (not isinstance(refresh, dict)
                    or refresh.get('stationary_cup_verified') is not True
                    or refresh.get('original_frame_id') != identity['snapshot_frame_id']
                    or refresh.get('original_color_sha256') != identity['snapshot_sha256_color']
                    or refresh.get('original_depth_sha256') != identity['snapshot_sha256_depth']
                    or not refresh.get('fresh_dir')):
                raise ValueError('Extended-age first top receipt lacks fresh cup proof')
            fresh_meta = json.loads((Path(refresh['fresh_dir'])/'metadata.json').read_bytes())
            if (fresh_meta.get('sha256_color') != refresh.get('fresh_color_sha256')
                    or fresh_meta.get('sha256_depth') != refresh.get('fresh_depth_sha256')
                    or fresh_meta.get('host_capture_time_ns') != refresh.get('fresh_capture_time_ns')
                    or fresh_meta['host_capture_time_ns'] > completed_ns):
                raise ValueError('First top receipt fresh RGB-D hashes/times changed')
        prior = tuple(validated['stages'])[-2] if len(validated['stages']) > 1 else None
        start_name = (receipt.get('contact_observed_at_stage') if prior == 'close'
                      else prior)
        start_q = (_joints(plan['current_joints_rad'], 'source planned joints')
                   if start_name is None else _joints(
                       plan['waypoints'][start_name]['target_joints_rad'],
                       f'{start_name} planned joints'))
        given_start = _joints(item.get('start_joints_rad'), f'{name} receipt start joints')
        if max(abs(a-b) for a, b in zip(given_start, start_q)) > 1e-6:
            raise ValueError(f'{name} receipt start joints differ from exact plan')
        target_q = (None if name == 'close' else _joints(
            plan['waypoints'][name]['target_joints_rad'], f'{name} planned target joints'))
        if target_q is None:
            if item.get('target_joints_rad') is not None:
                raise ValueError('Close receipt must not have an arm joint target')
        else:
            given_target = _joints(item.get('target_joints_rad'), f'{name} receipt target joints')
            if max(abs(a-b) for a, b in zip(given_target, target_q)) > 1e-6:
                raise ValueError(f'{name} receipt target joints differ from exact plan')
        start_flange = (plan['current_flange_pose_base_m_rad'] if start_name is None
                        else plan['waypoints'][start_name]['flange_pose_base_m_rad'])
        target_flange = (start_flange if name == 'close'
                         else plan['waypoints'][name]['flange_pose_base_m_rad'])
        if any(abs(a-b) > 1e-6 for a, b in zip(
                _pose(item.get('target_flange_m_rad'), f'{name} receipt target flange'),
                _pose(target_flange, f'{name} planned target flange'))):
            raise ValueError(f'{name} receipt target flange differs from exact plan')
        for field, expected in (('start_flange_m_rad', start_flange),
                                ('arrival_flange_m_rad', target_flange)):
            d, r = _matrix_error(pose_matrix(_pose(item.get(field), f'{name} receipt {field}')),
                                 pose_matrix(_pose(expected, f'{name} planned {field}')))
            if d > .005 or r > math.radians(1.5):
                raise ValueError(f'{name} receipt {field} differs from planned flange')
        if name == 'close' and (item.get('physical_contact_observed') is not True
                                or receipt.get('contact_observed_at_stage')
                                != tuple(validated['stages'])[-2]):
            raise ValueError('Top close receipt lacks independently observed prior contact')
        if name in (*SHAKE_STAGES, 'reveal_lift', 'reveal_exit') and item.get('hold_observed') is not True:
            raise ValueError(f'Top {name} receipt lacks independently observed cup hold')
        if name in SHAKE_STAGES and item.get('mouth_supported_observed') is not True:
            raise ValueError(f'Top {name} receipt lacks observed inverted-cup mouth support')
    if args.stage not in _allowed_next(plan, receipt):
        raise ValueError(f'{args.stage} is not the next reviewed top-grasp stage')
    if receipt['stages']:
        last = next(reversed(receipt['stages'].values()))
        age = execute._age_s(last.get('completed_at_ns'), now_ns)
        first_age = execute._age_s(next(iter(receipt['stages'].values())).get('completed_at_ns'), now_ns)
        if age > execute.MAX_STAGE_GAP_S or first_age > execute.MAX_SEQUENCE_AGE_S:
            raise ValueError('Top receipt is stale; refresh scene/plan')
    return receipt


def _close_targets(updates, baseline):
    targets = _hand_positions(updates, 'planned close positions')
    for name in HAND_KEYS:
        if targets[name] < baseline[name] or targets[name]-baseline[name] > MAX_HAND_TARGET_STEP:
            raise ValueError(f'Close {name} must be a modest monotone change <=30 points')
    return targets


def _command(prefix, stage, targets, close_positions, timeout, speed_percent=1):
    if stage == 'close':
        return [*prefix, 'hand', '--positions',
                *(str(close_positions[name]) for name in HAND_KEYS),
                '--duration', '1.5', '--timeout', '10', '--execute']
    q = targets[stage]
    return [*prefix, 'move-j', '--joints-deg',
            *(format(math.degrees(v), '.12g') for v in q),
            '--speed', str(speed_percent), '--timeout', format(timeout, '.12g'), '--execute']


def run_stage(args, *, runner=subprocess.run, now_ns=None):
    """Review/dry-run or dispatch exactly one authorized stage."""
    clock_injected = now_ns is not None
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    plan, sequence, identity, snapshot_age, close_targets = load_top_plan(args, now_ns=now_ns)
    receipt = load_receipt(args, plan, identity, now_ns=now_ns)
    model, q0, sdk, targets = independent_kinematics(plan)
    done = tuple(receipt['stages'])
    previous = done[-1] if done else None
    start_stage = previous if previous in targets else (
        receipt.get('contact_observed_at_stage') if previous == 'close' else None)
    if start_stage is None and previous == 'close':
        raise ValueError('Close receipt lacks contact_observed_at_stage')
    start_q = q0 if start_stage is None else targets[start_stage]
    start_flange = (plan['current_flange_pose_base_m_rad'] if start_stage is None
                    else plan['waypoints'][start_stage]['flange_pose_base_m_rad'])
    target_flange = (start_flange if args.stage == 'close'
                     else plan['waypoints'][args.stage]['flange_pose_base_m_rad'])
    baseline = _hand_positions(plan['hand_open_positions'], 'planned open hand')
    required_prior = {'tiny_shake_a': 'close', 'tiny_shake_b': 'tiny_shake_a',
                      'reveal_lift': 'tiny_shake_b', 'reveal_exit': 'reveal_lift'}
    if args.stage in required_prior and previous != required_prior[args.stage]:
        raise ValueError(f'{args.stage} requires prior verified {required_prior[args.stage]} stage')
    if not math.isfinite(args.move_timeout) or not 10 <= args.move_timeout <= 300:
        raise ValueError('Move timeout must be 10..300 seconds')
    speed_percent = getattr(args, 'speed_percent', 1)
    if (type(speed_percent) is not int or not 1 <= speed_percent <= 3 or
            (args.stage in (*CONTACT_CANDIDATES, 'close') and speed_percent != 1)):
        raise ValueError('Reviewed speed must be 1..3%; contact approach stays at 1%')
    if args.execute:
        if not args.scene_observed:
            raise ValueError(f'{args.stage} requires current scene --scene-observed')
        if args.stage == 'close' and not args.contact_observed:
            raise ValueError('close requires observed physical palm/cup contact')
        if args.stage in (*SHAKE_STAGES, 'reveal_lift', 'reveal_exit') and not args.hold_observed:
            raise ValueError(f'{args.stage} requires independent observed cup hold')
        if args.stage in SHAKE_STAGES and not args.mouth_supported_observed:
            raise ValueError(f'{args.stage} requires observed inverted-cup mouth support')
        if args.stage in SHAKE_STAGES and plan['checks'].get('shake_mouth_support_geometry_passed') is not True:
            raise ValueError(f'{args.stage} lacks checked table/cup-mouth support geometry')
    if args.stage == 'tiny_shake_a' and previous == 'close':
        prior_contact = receipt.get('contact_observed_at_stage')
        actual_contact_palm = _waypoint(plan, prior_contact)[2]
        shake_palm = _waypoint(plan, 'tiny_shake_a')[2]
        axis = _cup_geometry(plan)[1]
        if abs(float((shake_palm[:3, 3]-actual_contact_palm[:3, 3]) @ axis)) > .002:
            raise ValueError('Shake target is at a different height from observed contact stage')
    control_script = Path(__file__).resolve().parents[1]/'nero_revo2_control'/'nero_revo2_demo.py'
    prefix = execute._prefix(args.python, control_script, args.channel)
    command = _command(prefix, args.stage, targets, close_targets,
                       args.move_timeout, speed_percent)
    preview = {'event': 'top_stage_plan', 'stage': args.stage,
               'execute': bool(args.execute), 'plan_sha256': identity['plan_sha256'],
               'snapshot_frame_id': identity['snapshot_frame_id'],
               'snapshot_age_s': snapshot_age, 'previous_verified_stage': previous,
               'expected_start_joints_deg': [math.degrees(v) for v in start_q],
               'expected_target_joints_deg': (
                   [math.degrees(v) for v in targets[args.stage]] if args.stage != 'close' else None),
               'expected_start_flange_m_rad': start_flange,
               'expected_target_flange_m_rad': target_flange,
               'scene_observed': bool(args.scene_observed),
               'speed_percent': speed_percent,
               'contact_observed': bool(args.contact_observed),
               'hold_observed': bool(args.hold_observed),
               'mouth_supported_observed': bool(args.mouth_supported_observed),
               'hand_close_positions': close_targets if args.stage == 'close' else None,
               'grip_confirmed': False}
    lane = plan.get('reveal_exit_lane') or {}
    preview['dice_roi_source_kind'] = lane.get('source')
    preview['dice_directly_observed'] = (
        lane.get('source') == 'measured_dice_roi_plus_reviewed_exit_lane')
    if not args.execute:
        preview['message'] = 'dry run: no CAN command and no status connection'
        return preview
    before = execute.read_arm_status(prefix, runner=runner)
    live_q, live_sdk = execute.read_joint_limits(prefix, runner=runner)
    execute._joint_near(before.get('joints_rad'), start_q, 'top stage start q')
    execute._joint_near(live_q, start_q, 'top stage independent joint feedback')
    if live_sdk != sdk:
        raise ValueError('Live SDK limits differ from the top plan read-only limits')
    _sdk_within(live_q, sdk, 'live current')
    start_error = execute._near(before['flange_m_rad'], start_flange, 'top stage start flange')
    model_d, model_r = _matrix_error(model.fk(live_q), pose_matrix(before['flange_m_rad']))
    if model_d > .005 or model_r > math.radians(2.):
        raise ValueError('Live flange feedback disagrees with independent URDF FK')
    live_hand = execute.read_hand_status(prefix, runner=runner)
    expected_hand = (close_targets if ('close' in done and args.stage != 'close')
                     else baseline)
    _hand_near(live_hand, expected_hand, 'live hand positions')
    if args.stage == 'close' and any(live_hand[name] > 10 for name in
                                      ('index_finger', 'middle_finger',
                                       'ring_finger', 'pinky_finger')):
        raise ValueError('Close requires an open four-finger position baseline')
    screen = (_close_geometry_screen(plan, before['flange_m_rad'], live_hand, close_targets)
              if args.stage == 'close' else
              _geometry_screen(plan, model, start_q, targets[args.stage], live_hand, args.stage))
    # A hand geometry screen may take seconds. Read controller state again at
    # the dispatch boundary so a WEB switch, brake, fault, or manual joint
    # movement during that work never silently crosses into a CAN command.
    latest = execute.read_arm_status(prefix, runner=runner)
    latest_q, latest_sdk = execute.read_joint_limits(prefix, runner=runner)
    execute._joint_near(latest.get('joints_rad'), start_q, 'top dispatch joint feedback')
    execute._joint_near(latest_q, start_q, 'top dispatch independent joints')
    if latest_sdk != sdk:
        raise ValueError('SDK limits changed during top preflight')
    _sdk_within(latest_q, sdk, 'top dispatch')
    execute._near(latest['flange_m_rad'], start_flange, 'top dispatch flange')
    _hand_near(execute.read_hand_status(prefix, runner=runner), expected_hand,
               'top dispatch hand feedback')
    revalidation = None
    if args.stage == sequence[0] and (not clock_injected or
                                      getattr(args, 'revalidation_dir', None) is not None):
        current_ns = time.time_ns()
        revalidation_dir = getattr(args, 'revalidation_dir', None)
        if revalidation_dir is None:
            if (execute._age_s(json.loads((Path(args.snapshot_dir)/'metadata.json').read_bytes())
                               ['host_capture_time_ns'], current_ns) > execute.MAX_SNAPSHOT_AGE_S
                    or execute._age_s(plan['pose_source']['host_read_time_ns'], current_ns)
                    > execute.MAX_SNAPSHOT_AGE_S):
                raise ValueError('Top plan became stale during preflight; recapture and replan')
        else:
            if Path(revalidation_dir).exists():
                raise ValueError('Fresh-check RGB-D directory already exists; never reuse it')
            camera_python = getattr(args, 'camera_python', sys.executable)
            camera_cmd = [camera_python, '-m', 'cup_grasp_demo.grasp', 'snapshot',
                          '--dataset', str(revalidation_dir), '--serial',
                          json.loads((Path(args.snapshot_dir)/'metadata.json').read_bytes())['serial'],
                          '--warmup-frames', '60', '--timeout-ms', '9000']
            captured = runner(camera_cmd, capture_output=True, text=True,
                              timeout=45, check=False)
            if captured.returncode:
                raise ValueError('First-stage fresh RGB-D capture failed: '+captured.stderr.strip())
            recognition_dir = Path(args.source_side_plan).parent/'recognition'
            revalidation = check_stationary_cup(
                args.snapshot_dir, recognition_dir, revalidation_dir,
                now_ns=time.time_ns())
            revalidation['fresh_dir'] = str(revalidation_dir)
            latest = execute.read_arm_status(prefix, runner=runner)
            latest_q, latest_sdk = execute.read_joint_limits(prefix, runner=runner)
            execute._joint_near(latest.get('joints_rad'), start_q,
                                'post-refresh top dispatch joints')
            execute._joint_near(latest_q, start_q,
                                'post-refresh independent joints')
            if latest_sdk != sdk:
                raise ValueError('SDK limits changed after RGB-D fresh check')
            execute._near(latest['flange_m_rad'], start_flange,
                          'post-refresh dispatch flange')
            _hand_near(execute.read_hand_status(prefix, runner=runner), expected_hand,
                       'post-refresh dispatch hand')
            if execute._age_s(revalidation['fresh_capture_time_ns'], time.time_ns()) > 10.:
                raise ValueError('Fresh cup check expired before first CAN motion')
    try:
        events = execute._run_control(command, timeout=(20 if args.stage == 'close'
                                                         else args.move_timeout+20), runner=runner)
        wanted = 'hand_target_reached' if args.stage == 'close' else 'arm_target_reached'
        arrival = next((event for event in events if event['event'] == wanted), None)
        if arrival is None:
            raise RuntimeError(f'{args.stage} lacks fresh {wanted} feedback')
        if args.stage != 'close' and arrival.get('command') != 'move-j':
            raise RuntimeError(f'{args.stage} arrival was not a move-j command')
        after = execute.read_arm_status(prefix, runner=runner)
        arrival_error = execute._near(after['flange_m_rad'], target_flange,
                                      'top stage arrival flange')
        if args.stage != 'close':
            execute._joint_near(after.get('joints_rad'), targets[args.stage],
                                'top stage arrival joints')
        else:
            execute._joint_near(after.get('joints_rad'), start_q,
                                'top close must not move arm joints')
            settled_hand = execute.read_hand_status(prefix, runner=runner)
            _hand_near(settled_hand, close_targets, 'closed hand arrival')
            _close_geometry_screen(plan, after['flange_m_rad'], live_hand, settled_hand)
            receipt['contact_observed_at_stage'] = previous
            receipt['grip_confirmed'] = False
        completed_ns = time.time_ns()
        receipt['stages'][args.stage] = {
            'verified': True, 'completed_at_ns': completed_ns,
            'start_joints_rad': list(start_q),
            'target_joints_rad': (list(targets[args.stage]) if args.stage != 'close' else None),
            'start_flange_m_rad': before['flange_m_rad'],
            'target_flange_m_rad': target_flange,
            'arrival_flange_m_rad': after['flange_m_rad'],
            'start_error': start_error, 'arrival_error': arrival_error,
            'feedback_event': wanted, 'geometry_screen': screen,
            'speed_percent': speed_percent,
            'scene_observed': bool(args.scene_observed),
            'physical_contact_observed': bool(args.contact_observed) if args.stage == 'close' else False,
            'hold_observed': bool(args.hold_observed),
            'mouth_supported_observed': bool(args.mouth_supported_observed),
            'grip_confirmed': False,
        }
        if revalidation is not None:
            receipt['stages'][args.stage]['target_revalidation'] = revalidation
        execute._write_receipt(args.receipt, receipt, first=not done)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            subprocess.TimeoutExpired) as exc:
        raise execute.StageCommandError(str(exc)) from exc
    return {'event': 'top_stage_verified', 'stage': args.stage, 'verified': True,
            'receipt': str(args.receipt), 'plan_sha256': identity['plan_sha256'],
            'arrival_error': arrival_error, 'grip_confirmed': False,
            'dice_roi_source_kind': lane.get('source'),
            'dice_directly_observed': (
                lane.get('source') == 'measured_dice_roi_plus_reviewed_exit_lane')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=ALL_STAGES)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--source-side-plan', type=Path, required=True)
    parser.add_argument('--snapshot-dir', required=True, type=Path)
    parser.add_argument('--revalidation-dir', type=Path,
                        help='New camera dataset captured at first-stage dispatch')
    parser.add_argument('--camera-python', default=sys.executable)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--allow-provisional', action='store_true')
    parser.add_argument('--scene-observed', action='store_true')
    parser.add_argument('--contact-observed', action='store_true',
                        help='Physical palm/cup contact confirmed at preceding hover/probe; required for close')
    parser.add_argument('--hold-observed', action='store_true',
                        help='Cup independently observed held; required for shake and reveal')
    parser.add_argument('--mouth-supported-observed', action='store_true',
                        help='Inverted cup mouth visually confirmed against table; required for each shake')
    parser.add_argument('--move-timeout', type=float, default=90.)
    parser.add_argument('--speed-percent', type=int, default=1,
                        help='Reviewed 1..3%% move speed; contact stages stay 1%%')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = run_stage(args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(json.dumps({'event': 'top_stage_failed', 'stage': args.stage,
                          'error': str(exc),
                          'command_may_have_been_sent': bool(getattr(
                              exc, 'command_may_have_been_sent', False))},
                         ensure_ascii=False), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
