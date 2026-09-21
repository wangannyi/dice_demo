"""Supervised, one-stage-at-a-time cup grasp execution on K3.

The default is an offline preview. ``--execute`` is required for each physical
stage. The existing Nero/Revo2 demo owns CAN commands and arrival feedback.
This module never enables motors or treats a hand position as proof of grip.
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

from cup_grasp_demo.grasp import _check_frame_timestamps
from cup_grasp_demo import ik_review
from nero_calibration.core import matrix, pose_matrix
from nero_revo2_control.kinematics import load_model


STAGES = ('pregrasp', 'contact', 'close', 'lift')
ALL_STAGES = ('clearance', *STAGES)
TARGET_WAYPOINT = {'clearance': 'clearance', 'pregrasp': 'pregrasp', 'contact': 'contact',
                   'close': 'contact', 'lift': 'lift'}
START_WAYPOINT = {'clearance': None, 'pregrasp': None,
                  'contact': 'pregrasp', 'close': 'contact', 'lift': 'contact'}
MAX_SNAPSHOT_AGE_S = 10.0
MAX_STAGE_GAP_S = 180.0
MAX_SEQUENCE_AGE_S = 600.0
MAX_START_TRANSLATION_M = .005
MAX_START_ROTATION_RAD = math.radians(1.5)
DEFAULT_FINGER_UPDATES = ('index=18', 'middle=18', 'ring=18', 'pinky=18')
MAX_JOINT_FEEDBACK_ERROR_RAD = math.radians(.5)
MAX_IK_FK_POSITION_ERROR_M = ik_review.POSITION_IK_TOLERANCE_M
MAX_IK_FK_ORIENTATION_ERROR_RAD = ik_review.ORIENTATION_IK_TOLERANCE_RAD


class StageCommandError(RuntimeError):
    """An action was attempted; the hardware may have moved before failure."""

    command_may_have_been_sent = True


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _age_s(timestamp_ns, now_ns):
    if not isinstance(timestamp_ns, int) or timestamp_ns <= 0:
        raise ValueError('Missing positive capture/receipt time')
    age = (now_ns - timestamp_ns) / 1e9
    if not 0 <= age:
        raise ValueError('Capture/receipt time is in the future')
    return age


def _pose(value, name):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError(f'{name} must have six finite m/rad values')
    return pose.tolist()


def pose_error(actual, expected):
    """Return position and SO(3) angular error, independent of Euler wrap."""
    actual_t = pose_matrix(_pose(actual, 'actual flange pose'))
    expected_t = pose_matrix(_pose(expected, 'expected flange pose'))
    translation = float(np.linalg.norm(actual_t[:3, 3] - expected_t[:3, 3]))
    relative = actual_t[:3, :3] @ expected_t[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1) / 2, -1., 1.))
    return translation, math.acos(cosine)


def _joints_rad(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise ValueError(f'{name} must contain seven joint angles')
    joints = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in joints):
        raise ValueError(f'{name} must contain seven finite joint angles')
    return joints


def _joint_near(actual, expected, name):
    actual = _joints_rad(actual, name)
    difference = max(abs(a-b) for a, b in zip(actual, expected))
    if difference > MAX_JOINT_FEEDBACK_ERROR_RAD:
        raise RuntimeError(f'{name} differs from reviewed joint target by '
                           f'{math.degrees(difference):.2f} deg')
    return actual


def _sdk_ranges(value):
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError('Joint execution requires seven current SDK soft-limit pairs')
    ranges = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError('SDK soft-limit pairs must each have two values')
        lower, upper = (float(item) for item in pair)
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError('SDK soft-limit pairs must be finite and ordered')
        ranges.append((lower, upper))
    return tuple(ranges)


def _matrix_error(actual, expected):
    actual, expected = matrix(actual), matrix(expected)
    translation = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
    relative = actual[:3, :3] @ expected[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1) / 2, -1., 1.))
    return translation, math.acos(cosine)


def _validate_fk(model, joints, target, name, *, position_limit, orientation_limit):
    translation, rotation = _matrix_error(model.fk(joints), target)
    if translation > position_limit or rotation > orientation_limit:
        raise ValueError(f'{name} URDF FK differs from planned flange target: '
                         f'{translation * 1000:.2f} mm, {math.degrees(rotation):.2f} deg')


def load_joint_review(review_path, plan, identity, *, stage_order, channel):
    """Independently recheck a read-only IK report against this exact plan."""
    if review_path is None:
        raise ValueError('Joint arm mode requires --ik-review')
    report = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (report.get('schema') != 1 or report.get('kind') != 'read_only_urdf_kinematics_review'
            or report.get('source_plan_sha256') != identity['plan_sha256']
            or report.get('source_plan_frame_id') != identity['snapshot_frame_id']):
        raise ValueError('IK review is not bound to the exact plan SHA256 and RGB-D frame')
    motion_stages = tuple(name for name in stage_order if name != 'close')
    if (report.get('sequence') != ['current', *motion_stages]
            or report.get('clearance_required') is not (motion_stages[0] == 'clearance')
            or report.get('kinematic_checks_passed') is not True):
        raise ValueError('IK review stage sequence or kinematic checks failed')
    if report.get('joint_source') != f'nero_revo2_control_read_joints:{channel}':
        raise ValueError('Joint execution requires IK review from this CAN channel')
    if not isinstance(report.get('stages'), list) or len(report['stages']) != len(motion_stages):
        raise ValueError('IK review lacks the exact motion stages')
    model = load_model()
    if (report.get('model_base_frame') != model.base_frame
            or report.get('model_flange_frame') != model.flange_frame
            or report.get('urdf_source') != str(model.source)):
        raise ValueError('IK review uses a different URDF flange model')
    current = _joints_rad(report.get('current_joints_rad'), 'reviewed current joints')
    current_limits = report.get('current_joint_limits')
    if (not isinstance(current_limits, list) or len(current_limits) != 7
            or not all(isinstance(item, dict) for item in current_limits)):
        raise ValueError('IK review lacks seven current joint limit records')
    sdk = _sdk_ranges([item.get('sdk_range_deg') for item in current_limits])
    _validate_fk(model, current, pose_matrix(_pose(plan['current_flange_pose_base_m_rad'],
                                               'planned initial flange pose')),
                 'current', position_limit=ik_review.MAX_CURRENT_FK_POSITION_ERROR_M,
                 orientation_limit=ik_review.MAX_CURRENT_FK_ORIENTATION_ERROR_RAD)
    if not model.within_limits(current):
        raise ValueError('IK review current joints exceed URDF limits')
    for axis, (angle, record, bounds) in enumerate(zip(current, current_limits, sdk), 1):
        degrees = math.degrees(angle)
        if (record.get('joint') != f'J{axis}'
                or any(record.get(flag) is not True for flag in
                       ('within_documented', 'within_urdf', 'within_sdk'))
                or abs(float(record.get('angle_deg')) - degrees) > 1e-6
                or not bounds[0] <= degrees <= bounds[1]):
            raise ValueError(f'IK review current J{axis} limit record is inconsistent')
    by_stage = {}
    previous = current
    for name, item in zip(motion_stages, report['stages']):
        if (not isinstance(item, dict) or item.get('name') != name
                or item.get('target_frame') != 'base_link_to_link7'):
            raise ValueError(f'IK review stage name/frame differs at {name}')
        ik = item.get('ik') or {}
        if not isinstance(ik, dict) or ik.get('success') is not True:
            raise ValueError(f'{name} IK did not succeed')
        joints = _joints_rad(ik.get('joints_rad'), f'{name} IK joints')
        joint_degrees = _joints_rad(ik.get('joints_deg'), f'{name} IK joint degrees')
        if max(abs(given - math.degrees(angle))
               for given, angle in zip(joint_degrees, joints)) > 1e-6:
            raise ValueError(f'{name} IK radian/degree targets disagree')
        waypoint = plan['waypoints'][name]
        target = matrix(waypoint['T_base_flange'])
        encoded_pose = pose_matrix(_pose(waypoint['flange_pose_base_m_rad'],
                                         f'{name} flange pose'))
        pose_position, pose_rotation = _matrix_error(target, encoded_pose)
        if pose_position > 1e-6 or pose_rotation > 1e-5:
            raise ValueError(f'{name} flange pose and matrix disagree')
        _validate_fk(model, joints, target, name,
                     position_limit=MAX_IK_FK_POSITION_ERROR_M,
                     orientation_limit=MAX_IK_FK_ORIENTATION_ERROR_RAD)
        if (not math.isfinite(float(ik.get('position_error_m', math.inf)))
                or float(ik['position_error_m']) > MAX_IK_FK_POSITION_ERROR_M
                or not math.isfinite(float(ik.get('orientation_error_rad', math.inf)))
                or float(ik['orientation_error_rad']) > MAX_IK_FK_ORIENTATION_ERROR_RAD):
            raise ValueError(f'{name} IK report exceeds FK tolerance')
        records = item.get('goal_joint_limits')
        if (not isinstance(records, list) or len(records) != 7
                or not all(isinstance(record, dict) for record in records)):
            raise ValueError(f'{name} IK review lacks seven joint limit records')
        if not model.within_limits(joints):
            raise ValueError(f'{name} IK joints exceed URDF limits')
        for axis, (angle, record, bounds) in enumerate(zip(joints, records, sdk), 1):
            degrees = math.degrees(angle)
            documented = ik_review.DOCUMENTED_LIMITS_DEG[axis-1]
            record_sdk = _sdk_ranges([record.get('sdk_range_deg')] * 7)[0]
            if (record.get('joint') != f'J{axis}'
                    or any(record.get(flag) is not True for flag in
                           ('within_documented', 'within_urdf', 'within_sdk'))
                    or abs(float(record.get('angle_deg')) - degrees) > 1e-6
                    or record_sdk != bounds
                    or not (bounds[0] <= degrees <= bounds[1]
                            and documented[0] <= degrees <= documented[1])):
                raise ValueError(f'{name} IK J{axis} limit record is inconsistent')
        path = item.get('joint_path') or {}
        computed_path = model.check_joint_path(previous, joints)
        if (not isinstance(path, dict) or path.get('joint_limits_passed') is not True
                or path.get('continuity_passed') is not True
                or not computed_path.joint_limits_passed
                or not computed_path.continuity_passed):
            raise ValueError(f'{name} joint path failed URDF limits/continuity')
        by_stage[name] = joints
        previous = joints
    return report, _sha256(review_path), current, sdk, by_stage


def read_joint_limits(prefix, *, runner=subprocess.run):
    events = _run_control([*prefix, 'read-joints'], timeout=15, runner=runner)
    status = next((e for e in events if e['event'] == 'read_joints'), None)
    if status is None:
        raise RuntimeError('No fresh read-only joint/SDK feedback from control program')
    if (status.get('joints_enabled') != [True] * 7
            or status.get('arm_status') != 0 or status.get('ctrl_mode') != 1):
        raise RuntimeError('Joint/SDK feedback requires NORMAL, CAN_CTRL, seven joints enabled')
    return _joints_rad(status.get('joints_rad'), 'live joints'), _sdk_ranges(status.get('sdk_limits_deg'))


def _near(actual, expected, stage):
    translation, rotation = pose_error(actual, expected)
    if translation > MAX_START_TRANSLATION_M or rotation > MAX_START_ROTATION_RAD:
        raise RuntimeError(
            f'{stage} flange pose differs from expected: '
            f'{translation * 1000:.1f} mm, {math.degrees(rotation):.2f} deg')
    return {'translation_m': translation, 'rotation_rad': rotation}


def _stage_order(plan):
    required = plan['checks'].get('clearance_required', False)
    if type(required) is not bool:
        raise ValueError('Plan clearance_required must be boolean')
    return ALL_STAGES if required else STAGES


def _validate_clearance_waypoint(plan):
    """The first waypoint is a vertical 80 mm lift without flange rotation."""
    current = pose_matrix(_pose(plan['current_flange_pose_base_m_rad'], 'planned initial flange pose'))
    clearance = pose_matrix(_pose(plan['waypoints']['clearance']['flange_pose_base_m_rad'],
                                  'clearance flange pose'))
    delta = clearance[:3, 3] - current[:3, 3]
    rotation = current[:3, :3].T @ clearance[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1) / 2, -1., 1.))
    if (np.linalg.norm(delta[:2]) > .001 or abs(delta[2] - .080) > .001
            or math.acos(cosine) > math.radians(.5)):
        raise ValueError('Clearance waypoint must lift 80 mm in base Z with unchanged flange orientation')


def load_plan_and_snapshot(plan_path, snapshot_dir, calibration_path, *,
                           stage, allow_provisional, now_ns):
    """Tie the motion targets to unmodified camera assets and calibration."""
    plan_path, snapshot_dir = Path(plan_path), Path(snapshot_dir)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    metadata = json.loads((snapshot_dir / 'metadata.json').read_text(encoding='utf-8'))
    if (plan.get('schema') != 1
            or plan.get('kind') != 'read_only_side_grasp_proposal'
            or plan.get('units') != 'm_rad'):
        raise ValueError('Expected NERO side grasp plan schema 1 in m/rad')
    checks = plan.get('checks')
    if not isinstance(checks, dict) or checks.get('execute_ready') is not True:
        raise ValueError('Plan execute_ready is not true; create a fresh verified plan')
    if checks.get('pose_source_live_can') is not True or plan.get('pose_source', {}).get('kind') != 'can_feedback_read_only':
        raise ValueError('Plan must use live read-only CAN pose feedback')
    stage_order = _stage_order(plan)
    if stage not in stage_order:
        raise ValueError(f'{stage} is not a stage for this plan')
    quality = checks.get('calibration_quality_passed')
    if type(quality) is not bool:
        raise ValueError('Plan lacks calibration quality provenance')
    if not quality and not (allow_provisional and checks.get('provisional_opt_in') is True):
        raise ValueError('Failed calibration requires plan and action --allow-provisional')
    calibration_sha = plan.get('calibration_sha256')
    if not isinstance(calibration_sha, str) or len(calibration_sha) != 64:
        raise ValueError('Plan lacks calibration SHA256')
    if _sha256(calibration_path) != calibration_sha:
        raise ValueError('Calibration file differs from planned calibration')
    if (metadata.get('schema') != 1 or metadata.get('camera_backend') != 'realsense'
            or metadata.get('frame') != 'color_optical'
            or metadata.get('depth_registered_to') != 'color_optical'):
        raise ValueError('Snapshot is not an aligned RealSense color/depth pair')
    _check_frame_timestamps(metadata['timestamp_ms'], metadata['depth_timestamp_ms'],
                            metadata['timestamp_domain'], metadata['depth_timestamp_domain'])
    for name, expected_hash in [('color.png', 'sha256_color'),
                                ('depth.npz', 'sha256_depth')]:
        if _sha256(snapshot_dir / name) != metadata.get(expected_hash):
            raise ValueError(f'Snapshot {name} differs from capture metadata')
    frame_id = metadata.get('frame_id')
    localization = plan.get('localization')
    cup_base = plan.get('cup_base')
    if (not isinstance(localization, dict) or localization.get('valid') is not True
            or localization.get('frame') != 'color_optical'
            or not isinstance(cup_base, dict)):
        raise ValueError('Plan has no valid color-optical cup localization')
    source = cup_base.get('source') or {}
    if (frame_id != plan.get('snapshot_frame_id')
            or frame_id != localization.get('frame_id')
            or frame_id != source.get('frame_id')):
        raise ValueError('Plan and saved RGB-D snapshot frame IDs differ')
    if (metadata.get('serial') != plan.get('localization', {}).get('camera_serial')
            or metadata.get('serial') != source.get('camera_serial')):
        raise ValueError('Plan and saved RGB-D camera serials differ')
    for waypoint in ('pregrasp', 'contact', 'lift'):
        if waypoint not in (plan.get('waypoints') or {}):
            raise ValueError(f'Plan lacks {waypoint} waypoint')
        _pose(plan['waypoints'][waypoint]['flange_pose_base_m_rad'], waypoint)
    _pose(plan.get('current_flange_pose_base_m_rad'), 'planned initial flange pose')
    if checks.get('clearance_required', False):
        if 'clearance' not in plan['waypoints']:
            raise ValueError('Plan requires clearance waypoint before pregrasp')
        _validate_clearance_waypoint(plan)
    capture_age_s = _age_s(metadata.get('host_capture_time_ns'), now_ns)
    if stage == stage_order[0] and capture_age_s > MAX_SNAPSHOT_AGE_S:
        raise ValueError(f'{stage} snapshot is {capture_age_s:.1f}s old; recapture and replan')
    pose_read_ns = plan['pose_source'].get('host_read_time_ns')
    if stage == stage_order[0] and _age_s(pose_read_ns, now_ns) > MAX_SNAPSHOT_AGE_S:
        raise ValueError(f'{stage} planned arm feedback is stale; replan')
    identity = {'plan_sha256': _sha256(plan_path),
                'calibration_sha256': calibration_sha,
                'snapshot_frame_id': frame_id,
                'snapshot_sha256_color': metadata['sha256_color'],
                'snapshot_sha256_depth': metadata['sha256_depth']}
    return plan, identity, capture_age_s


def load_receipt(receipt_path, identity, stage, now_ns, *, stage_order):
    """Require the immediately preceding verified stage on this exact plan."""
    receipt_path = Path(receipt_path)
    if stage == stage_order[0]:
        if receipt_path.exists():
            raise ValueError(f'Receipt already exists; {stage} cannot be repeated')
        return {'schema': 1, 'kind': 'cup_grasp_stage_receipt',
                **identity, 'stages': {}}
    if not receipt_path.exists():
        raise ValueError(f'{stage} requires existing receipt from {stage_order[0]}')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('schema') != 1 or receipt.get('kind') != 'cup_grasp_stage_receipt':
        raise ValueError('Invalid stage receipt')
    if any(receipt.get(key) != value for key, value in identity.items()):
        raise ValueError('Stage receipt belongs to a different plan/snapshot/calibration')
    stages = receipt.get('stages')
    if not isinstance(stages, dict) or stage in stages:
        raise ValueError(f'{stage} already completed or receipt invalid')
    expected_prefix = stage_order[:stage_order.index(stage)]
    if tuple(stages.keys()) != expected_prefix or not all(
            stages[name].get('verified') is True for name in expected_prefix):
        raise ValueError('Stage receipt must contain exact verified prior-stage sequence')
    previous = stage_order[stage_order.index(stage) - 1]
    gap_s = _age_s(stages[previous].get('completed_at_ns'), now_ns)
    sequence_s = _age_s(stages[stage_order[0]].get('completed_at_ns'), now_ns)
    if gap_s > MAX_STAGE_GAP_S or sequence_s > MAX_SEQUENCE_AGE_S:
        raise ValueError('Stage receipt is stale; refresh scene and plan before continuing')
    return receipt


def _events(output):
    events = []
    for line in output.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get('event'), str):
            events.append(payload)
    return events


def _run_control(command, *, timeout, runner=subprocess.run):
    result = runner(command, text=True, capture_output=True, timeout=timeout)
    events = _events(result.stdout)
    if result.returncode != 0 or any(e['event'] == 'failed' for e in events):
        detail = next((e.get('error') for e in events if e['event'] == 'failed'),
                      (result.stderr or result.stdout).strip())
        raise RuntimeError(f'Existing Nero/Revo2 control command failed: {detail}')
    return events


def _prefix(python_executable, control_script, channel):
    return [str(python_executable), str(control_script), '--channel', channel,
            '--format', 'json']


def read_arm_status(prefix, *, runner=subprocess.run):
    events = _run_control([*prefix, 'status'], timeout=15, runner=runner)
    status = next((e for e in events if e['event'] == 'arm_status'), None)
    if status is None:
        raise RuntimeError('No fresh arm_status event from control program')
    rendered = status.get('status')
    if (not isinstance(rendered, str) or 'ctrl_mode: CAN_CTRL(' not in rendered
            or 'arm_status: NORMAL(' not in rendered
            or status.get('joints_enabled') != [True] * 7):
        raise RuntimeError('Arm must be NORMAL, CAN_CTRL, and seven joints enabled')
    status['flange_m_rad'] = _pose(status.get('flange_m_rad'), 'live flange pose')
    return status


def read_hand_status(prefix, *, runner=subprocess.run):
    events = _run_control([*prefix, 'hand-status'], timeout=15, runner=runner)
    status = next((e for e in events if e['event'] == 'hand_status'), None)
    if status is None or status.get('is_ok') is not True:
        raise RuntimeError('Right Revo2 hand feedback is not healthy')
    positions = status.get('positions')
    if not isinstance(positions, dict):
        raise RuntimeError('Right Revo2 hand has no position feedback')
    return positions


def _command(prefix, stage, plan, *, finger_updates, finger_positions,
             duration_s, move_timeout_s, arm_mode='cartesian', joint_target=None):
    if stage != 'close':
        if arm_mode == 'joint':
            if joint_target is None:
                raise ValueError(f'{stage} lacks reviewed seven-joint target')
            return [*prefix, 'move-j', '--joints-deg',
                    *(format(math.degrees(v), '.12g') for v in joint_target),
                    '--speed', '1', '--timeout', format(move_timeout_s, '.12g'),
                    '--execute']
        target = plan['waypoints'][TARGET_WAYPOINT[stage]]['flange_pose_base_m_rad']
        return [*prefix, 'move-p', '--pose', *(format(v, '.12g') for v in target),
                '--target-frame', 'flange', '--speed', '1',
                '--timeout', format(move_timeout_s, '.12g'), '--execute']
    command = [*prefix, 'hand']
    if finger_positions is not None:
        command.extend(['--positions', *(str(value) for value in finger_positions)])
    else:
        for update in finger_updates:
            command.extend(['--set', update])
    command.extend(['--duration', format(duration_s, '.12g'), '--timeout', '10',
                    '--execute'])
    return command


def _validate_hand_updates(updates, positions):
    if positions is not None:
        if len(positions) != 6 or any(not 0 <= v <= 100 for v in positions):
            raise ValueError('Six finger positions must be integers 0..100')
        return ()
    updates = DEFAULT_FINGER_UPDATES if updates is None else tuple(updates)
    if not updates:
        raise ValueError('At least one finger update is required')
    aliases = {'thumb_tip': 'thumb_tip', 'thumb_base': 'thumb_base',
               'index': 'index_finger', 'index_finger': 'index_finger',
               'middle': 'middle_finger', 'middle_finger': 'middle_finger',
               'ring': 'ring_finger', 'ring_finger': 'ring_finger',
               'pinky': 'pinky_finger', 'pinky_finger': 'pinky_finger'}
    seen = set()
    for update in updates:
        try:
            name, raw = update.split('=', 1)
            value = int(raw)
        except (ValueError, TypeError):
            raise ValueError(f'Invalid finger update {update!r}') from None
        canonical = aliases.get(name)
        if canonical is None or canonical in seen or not 0 <= value <= 100:
            raise ValueError(f'Invalid or repeated finger update {update!r}')
        seen.add(canonical)
    return updates


def _write_receipt(path, receipt, *, first):
    path = Path(path)
    data = json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if first:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(data)
        return
    temporary = path.with_name(path.name + '.new')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(data)
    temporary.replace(path)


def run_stage(args, *, runner=subprocess.run, now_ns=None):
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    plan, identity, snapshot_age_s = load_plan_and_snapshot(
        args.plan, args.snapshot_dir, args.calibration, stage=args.stage,
        allow_provisional=args.allow_provisional, now_ns=now_ns)
    stage_order = _stage_order(plan)
    first_stage = stage_order[0]
    stage_index = stage_order.index(args.stage)
    previous = stage_order[stage_index - 1] if stage_index else None
    receipt = load_receipt(args.receipt, identity, args.stage, now_ns,
                           stage_order=stage_order)
    arm_mode = getattr(args, 'arm_mode', 'cartesian')
    review_path = getattr(args, 'ik_review', None)
    if arm_mode not in ('cartesian', 'joint'):
        raise ValueError('Arm mode must be cartesian or joint')
    if arm_mode == 'cartesian' and review_path is not None:
        raise ValueError('--ik-review requires --arm-mode joint')
    if arm_mode == 'joint':
        _, review_sha, reviewed_start, reviewed_sdk, joint_targets = load_joint_review(
            review_path, plan, identity, stage_order=stage_order,
            channel=args.channel)
        if previous is None:
            receipt['arm_mode'] = 'joint'
            receipt['ik_review_sha256'] = review_sha
        elif (receipt.get('arm_mode') != 'joint'
              or receipt.get('ik_review_sha256') != review_sha
              or any(receipt['stages'][name].get('arm_mode') != 'joint'
                     or receipt['stages'][name].get('ik_review_sha256') != review_sha
                     for name in stage_order[:stage_index])):
            raise ValueError('Stage receipt belongs to a different arm mode or IK review')
    elif receipt.get('arm_mode') == 'joint':
        raise ValueError('Stage receipt belongs to a joint-mode grasp sequence')
    updates = _validate_hand_updates(args.set_values, args.positions)
    if (plan['checks'].get('clearance_required', False)
            and args.stage in ('clearance', 'pregrasp')
            and args.execute and not args.scene_observed):
        raise ValueError(f'{args.stage} requires --scene-observed for the clearance sequence')
    if args.stage in ('contact', 'lift') and args.execute and not args.scene_observed:
        raise ValueError(f'{args.stage} requires explicit --scene-observed after current camera/scene inspection')
    if args.stage == 'lift' and args.execute and not args.grip_observed:
        raise ValueError('lift requires explicit --grip-observed after visual/technician hold check')
    if not math.isfinite(args.duration) or not .01 <= args.duration <= 2.55:
        raise ValueError('Hand duration must be 0.01..2.55s')
    if not math.isfinite(args.move_timeout) or not 10 <= args.move_timeout <= 300:
        raise ValueError('Move timeout must be 10..300s')
    if args.stage != 'close' and (args.set_values is not None or args.positions is not None):
        raise ValueError('Finger options apply only to the close stage')
    start_waypoint = 'clearance' if args.stage == 'pregrasp' and previous == 'clearance' else START_WAYPOINT[args.stage]
    expected_start = (plan['current_flange_pose_base_m_rad'] if start_waypoint is None
                      else plan['waypoints'][start_waypoint]['flange_pose_base_m_rad'])
    expected_target = plan['waypoints'][TARGET_WAYPOINT[args.stage]]['flange_pose_base_m_rad']
    control_script = (Path(__file__).resolve().parents[1] / 'nero_revo2_control' /
                      'nero_revo2_demo.py')
    prefix = _prefix(args.python, control_script, args.channel)
    command = _command(prefix, args.stage, plan, finger_updates=updates,
                       finger_positions=args.positions, duration_s=args.duration,
                       move_timeout_s=args.move_timeout, arm_mode=arm_mode,
                       joint_target=(joint_targets.get(args.stage)
                                     if arm_mode == 'joint' else None))
    preview = {'event': 'stage_plan', 'stage': args.stage,
               'execute': bool(args.execute), 'arm_mode': arm_mode,
               'plan_sha256': identity['plan_sha256'],
               'snapshot_frame_id': identity['snapshot_frame_id'],
               'snapshot_age_s': snapshot_age_s,
               'expected_start_flange_m_rad': expected_start,
               'expected_target_flange_m_rad': expected_target,
               'hand_updates': list(updates) if args.stage == 'close' else None,
               'hand_positions': args.positions if args.stage == 'close' else None,
               'scene_observed': bool(args.scene_observed),
               'grip_observed': bool(args.grip_observed),
               'previous_verified_stage': previous}
    if arm_mode == 'joint':
        preview['ik_review_sha256'] = review_sha
        preview['target_joints_deg'] = (
            [math.degrees(v) for v in joint_targets[args.stage]]
            if args.stage != 'close' else None)
    if not args.execute:
        preview['message'] = 'dry run: no CAN command and no status connection'
        return preview
    before = read_arm_status(prefix, runner=runner)
    start_error = _near(before['flange_m_rad'], expected_start, 'start')
    if arm_mode == 'joint':
        expected_joints = (reviewed_start if previous is None
                           else joint_targets[TARGET_WAYPOINT[previous]])
        before_joints = _joint_near(before.get('joints_rad'), expected_joints,
                                    'start joint feedback')
        live_joints, live_sdk = read_joint_limits(prefix, runner=runner)
        _joint_near(live_joints, before_joints, 'joint/SDK feedback')
        if any(max(abs(a-b) for a, b in zip(live_pair, reviewed_pair)) > 1e-6
               for live_pair, reviewed_pair in zip(live_sdk, reviewed_sdk)):
            raise ValueError('Live SDK joint limits differ from IK review')
    if args.stage == 'close':
        finger_feedback = read_hand_status(prefix, runner=runner)
        if args.set_values is None and args.positions is None and any(
                not isinstance(finger_feedback.get(name), int)
                or not 0 <= finger_feedback[name] <= 10
                for name in ('index_finger', 'middle_finger',
                             'ring_finger', 'pinky_finger')):
            raise ValueError('Default light close expects open fingers 0..10; '
                             'review hand feedback and specify --set/--positions')
    timeout = args.move_timeout + 20 if args.stage != 'close' else 20
    try:
        events = _run_control(command, timeout=timeout, runner=runner)
        wanted = 'hand_target_reached' if args.stage == 'close' else 'arm_target_reached'
        arrival = next((e for e in events if e['event'] == wanted), None)
        if arrival is None:
            raise RuntimeError(f'{args.stage} command lacks fresh {wanted} feedback')
        expected_command = 'move-j' if arm_mode == 'joint' else 'move-p'
        if args.stage != 'close' and arrival.get('command') != expected_command:
            raise RuntimeError(f'Motion arrival event is not from {expected_command}')
        after = read_arm_status(prefix, runner=runner)
        target_error = _near(after['flange_m_rad'], expected_target, 'arrival')
        if arm_mode == 'joint':
            target_joints = joint_targets[TARGET_WAYPOINT[args.stage]]
            arrival_joints = _joint_near(after.get('joints_rad'), target_joints,
                                          'arrival joint feedback')
        completed_ns = time.time_ns()
        receipt['stages'][args.stage] = {
            'verified': True, 'completed_at_ns': completed_ns,
            'start_flange_m_rad': before['flange_m_rad'],
            'target_flange_m_rad': expected_target,
            'arrival_flange_m_rad': after['flange_m_rad'],
            'start_error': start_error, 'arrival_error': target_error,
            'feedback_event': wanted, 'scene_observed': bool(args.scene_observed),
            'grip_observed': bool(args.grip_observed),
            'hand_command': list(updates) if args.stage == 'close' and args.positions is None
                            else args.positions if args.stage == 'close' else None,
            'grip_confirmed': False if args.stage == 'close' else None,
        }
        if arm_mode == 'joint':
            receipt['stages'][args.stage].update({
                'arm_mode': 'joint', 'ik_review_sha256': review_sha,
                'target_joints_deg': [math.degrees(v) for v in target_joints],
                'arrival_joints_deg': [math.degrees(v) for v in arrival_joints],
            })
        _write_receipt(args.receipt, receipt, first=args.stage == first_stage)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            subprocess.TimeoutExpired) as exc:
        raise StageCommandError(str(exc)) from exc
    return {'event': 'stage_verified', 'stage': args.stage, 'verified': True,
            'receipt': str(args.receipt), 'plan_sha256': identity['plan_sha256'],
            'arrival_error': target_error,
            'grip_confirmed': False if args.stage == 'close' else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=ALL_STAGES)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--snapshot-dir', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--python', default=sys.executable,
                        help='Python with existing pyAgxArm/protocol dependencies')
    parser.add_argument('--allow-provisional', action='store_true')
    parser.add_argument('--scene-observed', action='store_true',
                        help='Technician and PC debug camera inspected current scene')
    parser.add_argument('--grip-observed', action='store_true',
                        help='Technician and PC debug camera observed cup held before lift')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--arm-mode', choices=('cartesian', 'joint'), default='cartesian',
                        help='Explicitly use reviewed move-j goals for arm stages')
    parser.add_argument('--ik-review', type=Path,
                        help='Read-only IK report bound to the exact plan; required for joint mode')
    hand = parser.add_mutually_exclusive_group()
    hand.add_argument('--set', dest='set_values', action='append',
                      metavar='FINGER=POSITION')
    hand.add_argument('--positions', nargs=6, type=int,
                      metavar=('THUMB_TIP', 'THUMB_BASE', 'INDEX', 'MIDDLE', 'RING', 'PINKY'))
    parser.add_argument('--duration', type=float, default=1.5)
    parser.add_argument('--move-timeout', type=float, default=90.0)
    args = parser.parse_args(argv)
    try:
        result = run_stage(args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError,
            subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(json.dumps({'event': 'stage_failed', 'stage': args.stage,
                          'error': str(exc),
                          'command_may_have_been_sent': bool(getattr(
                              exc, 'command_may_have_been_sent', False))},
                         ensure_ascii=False), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
