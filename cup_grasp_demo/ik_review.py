"""Read-only URDF kinematics review of a proposed Nero cup-grasp plan.

The model ends at ``link7`` (the flange). This command reads JSON or existing
joint feedback; it never sends a motion, enable, or stop command. A successful
local IK and joint-linear interpolation do not verify scene or self collision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from nero_calibration.core import matrix, pose_matrix
from nero_revo2_control.kinematics import load_model


DOCUMENTED_LIMITS_DEG = (
    (-157.0, 157.0), (-102.0, 102.0), (-160.0, 160.0),
    (-60.0, 125.0), (-160.0, 160.0), (-44.0, 57.0), (-97.0, 97.0),
)
DEFAULT_STAGES = ('pregrasp', 'contact', 'lift')
REFERENCE_READY_DEG = (55.0, -78.0, 80.0, -45.0, 130.0, -30.0, 40.0)
POSITION_IK_TOLERANCE_M = 0.001
ORIENTATION_IK_TOLERANCE_RAD = 0.01
MAX_IK_SEEDS_PER_STAGE = 7
MAX_CURRENT_FK_POSITION_ERROR_M = 0.005
MAX_CURRENT_FK_ORIENTATION_ERROR_RAD = math.radians(2.0)


def _finite_joint_vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise ValueError('Seven current joint angles in radians are required')
    joints = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in joints):
        raise ValueError('Current joint angles must be finite')
    return joints


def _pose_error(target, measured):
    target, measured = matrix(target), matrix(measured)
    position = math.sqrt(sum((target[i, 3]-measured[i, 3])**2 for i in range(3)))
    trace = sum(sum(target[i, k]*measured[i, k] for k in range(3)) for i in range(3))
    orientation = math.acos(max(-1.0, min(1.0, (trace-1.0)/2.0)))
    return position, orientation


def _limit_records(model, joints_rad, sdk_limits_deg=None):
    if sdk_limits_deg is not None:
        if not isinstance(sdk_limits_deg, list) or len(sdk_limits_deg) != 7:
            raise ValueError('SDK joint limits must contain seven pairs')
        sdk_limits_deg = [list(pair) for pair in sdk_limits_deg]
        if any(len(pair) != 2 or not all(math.isfinite(float(v)) for v in pair)
               for pair in sdk_limits_deg):
            raise ValueError('SDK joint limits must contain finite pairs')
    records = []
    for index, (angle, documented, urdf) in enumerate(zip(
            joints_rad, DOCUMENTED_LIMITS_DEG, model.limits_rad), 1):
        degrees = math.degrees(angle)
        urdf_deg = tuple(math.degrees(v) for v in urdf)
        sdk = sdk_limits_deg[index-1] if sdk_limits_deg is not None else None
        records.append({
            'joint': f'J{index}', 'angle_deg': degrees,
            'documented_range_deg': list(documented),
            'urdf_range_deg': list(urdf_deg),
            'sdk_range_deg': sdk,
            'within_documented': documented[0] <= degrees <= documented[1],
            'within_urdf': urdf[0] <= angle <= urdf[1],
            'within_sdk': (None if sdk is None else float(sdk[0]) <= degrees <= float(sdk[1])),
        })
    return records


def _clipped_seed(model, joints_rad):
    clipped = tuple(max(lower, min(upper, angle))
                    for angle, (lower, upper) in zip(joints_rad, model.limits_rad))
    return clipped, [f'J{i}' for i, (old, new) in enumerate(zip(joints_rad, clipped), 1)
                     if abs(old-new) > 1e-10]


def _candidate_seeds(model, initial):
    """Bounded deterministic starting branches; never promote them as a path."""
    initial = tuple(initial)
    ready = tuple(math.radians(value) for value in REFERENCE_READY_DEG)
    balanced = tuple((old+reference)/2.0 for old, reference in zip(initial, ready))
    definitions = [
        ('current_clipped_or_prior_solution', initial),
        ('balanced_midpoint', balanced),
        ('ready_home_reference', ready),
    ]
    for name, changes in (
            ('j3_plus_10deg', {2: 10.0}),
            ('j3_minus_10deg', {2: -10.0}),
            ('j4_plus_10deg', {3: 10.0}),
            ('j7_minus_10deg', {6: -10.0})):
        varied = list(initial)
        for index, degrees in changes.items():
            lower, upper = model.limits_rad[index]
            varied[index] = max(lower, min(upper, varied[index]+math.radians(degrees)))
        definitions.append((name, tuple(varied)))
    unique = []
    for name, values in definitions:
        if not model.within_limits(values):
            continue
        if not any(max(abs(a-b) for a, b in zip(values, old)) < 1e-10
                   for _, old in unique):
            unique.append((name, values))
    return unique[:MAX_IK_SEEDS_PER_STAGE]


def _solve_bounded(model, target, seed):
    """Select first converged branch, otherwise the smallest normalized residual."""
    attempts = []
    best = None
    best_score = math.inf
    for name, candidate in _candidate_seeds(model, seed):
        result = model.ik(
            target, candidate, position_tolerance_m=POSITION_IK_TOLERANCE_M,
            orientation_tolerance_rad=ORIENTATION_IK_TOLERANCE_RAD)
        attempts.append({
            'seed_name': name, 'seed_joints_rad': list(candidate),
            'success': result.success, 'position_error_m': result.position_error_m,
            'orientation_error_rad': result.orientation_error_rad,
            'iterations': result.iterations, 'reason': result.reason,
        })
        score = (result.position_error_m/POSITION_IK_TOLERANCE_M
                 + result.orientation_error_rad/ORIENTATION_IK_TOLERANCE_RAD)
        if score < best_score:
            best, best_score = (name, result), score
        if result.success:
            return name, result, attempts
    if best is None:
        raise RuntimeError('No valid URDF IK seed could be generated')
    return best[0], best[1], attempts


def _validate_plan(plan):
    if not isinstance(plan, dict) or plan.get('schema') != 1:
        raise ValueError('Expected schema-1 side-grasp plan JSON')
    if plan.get('kind') != 'read_only_side_grasp_proposal' or plan.get('units') != 'm_rad':
        raise ValueError('Expected a metres/radians side-grasp proposal')
    if not isinstance(plan.get('waypoints'), dict):
        raise ValueError('Plan has no waypoint mapping')
    checks = plan.get('checks', {})
    if not isinstance(checks, dict):
        raise ValueError('Plan checks must be an object')
    if checks.get('clearance_required') not in (None, False, True):
        raise ValueError('clearance_required must be a boolean')
    stage_names = (('clearance',) + DEFAULT_STAGES
                   if checks.get('clearance_required') is True else DEFAULT_STAGES)
    current = pose_matrix(plan['current_flange_pose_base_m_rad'])
    targets = {}
    for stage in stage_names:
        if stage not in plan['waypoints']:
            raise ValueError(f'{stage} waypoint required by plan sequence is missing')
        item = plan['waypoints'][stage]
        if not isinstance(item, dict):
            raise ValueError(f'{stage} waypoint must be an object')
        target = matrix(item['T_base_flange'])
        if 'flange_pose_base_m_rad' in item:
            pose_target = pose_matrix(item['flange_pose_base_m_rad'])
            position, orientation = _pose_error(target, pose_target)
            if position > 1e-6 or orientation > 1e-5:
                raise ValueError(f'{stage} flange pose and matrix disagree')
        targets[stage] = target
    return current, targets, stage_names


def _stage_review(model, name, target, seed, start, sdk_limits_deg):
    selected_seed_name, result, attempts = _solve_bounded(model, target, seed)
    candidate = _limit_records(model, result.joints_rad, sdk_limits_deg)
    stage = {
        'name': name, 'target_frame': 'base_link_to_link7',
        'ik': {
            'success': result.success,
            'selected_seed_name': selected_seed_name,
            'attempt_count': len(attempts),
            'max_seed_attempts': MAX_IK_SEEDS_PER_STAGE,
            'attempts': attempts,
            'joints_rad': list(result.joints_rad),
            'joints_deg': [math.degrees(value) for value in result.joints_rad],
            'position_error_m': result.position_error_m,
            'orientation_error_rad': result.orientation_error_rad,
            'iterations': result.iterations, 'reason': result.reason,
        },
        'goal_joint_limits': candidate, 'joint_path': None,
        'path_start_available': start is not None,
        'collision_verified': False, 'scene_collision_verified': False,
        'self_collision_verified': False,
    }
    if result.success and start is not None:
        path = model.check_joint_path(start, result.joints_rad)
        stage['joint_path'] = {
            'type': 'joint_linear_interpolation_only',
            'sample_count': path.sample_count,
            'joint_limits_passed': path.joint_limits_passed,
            'continuity_passed': path.continuity_passed,
            'singularity_warning': path.singularity_warning,
            'reason': path.reason,
            'collision_verified': False, 'scene_collision_verified': False,
            'self_collision_verified': False,
        }
    return stage


def review_plan(plan, current_joints_rad, *, sdk_limits_deg=None, joint_source='cli',
                source_plan_sha256=None):
    """Review one plan against supplied current feedback with no hardware side effects."""
    current_joints = _finite_joint_vector(current_joints_rad)
    planned_current, targets, stage_names = _validate_plan(plan)
    model = load_model()
    model_current = model.fk(current_joints)
    position_error, orientation_error = _pose_error(planned_current, model_current)
    current_limits = _limit_records(model, current_joints, sdk_limits_deg)
    seed, clipped_axes = _clipped_seed(model, current_joints)
    stages = []
    start = current_joints
    for name in stage_names:
        stage = _stage_review(model, name, targets[name], seed, start, sdk_limits_deg)
        stages.append(stage)
        if stage['ik']['success']:
            seed = tuple(stage['ik']['joints_rad'])
            if start is not None:
                start = seed
        else:
            start = None

    blockers = []
    if position_error > MAX_CURRENT_FK_POSITION_ERROR_M or orientation_error > MAX_CURRENT_FK_ORIENTATION_ERROR_RAD:
        blockers.append('URDF FK does not agree with plan current flange feedback')
    for record in current_limits:
        if not record['within_documented']:
            blockers.append(f"{record['joint']} current angle exceeds documented mechanical range")
        if not record['within_urdf']:
            blockers.append(f"{record['joint']} current angle exceeds conservative URDF range")
        if record['within_sdk'] is False:
            blockers.append(f"{record['joint']} current angle exceeds SDK soft range")
    for stage in stages:
        if not stage['ik']['success']:
            blockers.append(f"{stage['name']} local IK failed: {stage['ik']['reason']}")
        elif stage['joint_path'] is None:
            blockers.append(f"{stage['name']} joint interpolation unavailable after earlier IK failure")
        elif not (stage['joint_path']['joint_limits_passed']
                  and stage['joint_path']['continuity_passed']):
            blockers.append(f"{stage['name']} joint interpolation failed conservative limits")
        for record in stage['goal_joint_limits']:
            if record['within_sdk'] is False:
                blockers.append(f"{stage['name']} {record['joint']} IK candidate exceeds SDK soft range")
    blockers.append('Nero/Revo2/table/cup/cable collision geometry and controller trajectory are unverified')
    checks_passed = len(blockers) == 1
    return {
        'schema': 1, 'kind': 'read_only_urdf_kinematics_review',
        'source_plan_frame_id': plan.get('snapshot_frame_id'),
        'source_plan_sha256': source_plan_sha256,
        'joint_source': joint_source,
        'urdf_source': str(model.source),
        'model_base_frame': model.base_frame, 'model_flange_frame': model.flange_frame,
        'current_joints_rad': list(current_joints),
        'sequence': ['current', *stage_names],
        'clearance_required': stage_names[0] == 'clearance',
        'current_joint_limits': current_limits,
        'seed_clipped_to_urdf_axes': clipped_axes,
        'current_fk': {
            'model_position_m': [model_current[i][3] for i in range(3)],
            'plan_position_m': [planned_current[i, 3] for i in range(3)],
            'position_difference_m': position_error,
            'orientation_difference_rad': orientation_error,
            'agreement_passed': (position_error <= MAX_CURRENT_FK_POSITION_ERROR_M
                                 and orientation_error <= MAX_CURRENT_FK_ORIENTATION_ERROR_RAD),
        },
        'stages': stages, 'kinematic_checks_passed': checks_passed,
        'collision_verified': False, 'scene_collision_verified': False,
        'self_collision_verified': False,
        'controller_trajectory_verified': False,
        'executable': False, 'motion_sent': False,
        'blockers': blockers,
    }


def read_joints_existing_demo(channel, python_executable):
    """Call the existing read-joints subcommand; parse its JSON feedback only."""
    demo = Path(__file__).resolve().parents[1] / 'nero_revo2_control' / 'nero_revo2_demo.py'
    command = [str(python_executable), str(demo), '--format', 'json',
               '--channel', str(channel), 'read-joints']
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    messages = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith('{')]
    matching = [item for item in messages if item.get('event') == 'read_joints']
    if not matching:
        raise RuntimeError('Read-only joint command returned no seven-axis feedback')
    payload = matching[-1]
    return _finite_joint_vector(payload['joints_rad']), payload.get('sdk_limits_deg')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    joints = parser.add_mutually_exclusive_group(required=True)
    joints.add_argument('--joints-rad', type=float, nargs=7, metavar='RAD')
    joints.add_argument('--read-joints', action='store_true',
                        help='Run the existing read-only CAN joint feedback command')
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--python', default=sys.executable,
                        help='Python executable with pyAgxArm installed for --read-joints')
    parser.add_argument('--output', type=Path, help='Create a new JSON report file')
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text(encoding='utf-8'))
        if args.read_joints:
            current, sdk_limits = read_joints_existing_demo(args.channel, args.python)
            source = f'nero_revo2_control_read_joints:{args.channel}'
        else:
            current, sdk_limits = args.joints_rad, None
            source = 'cli_explicit_rad'
        report = review_plan(
            plan, current, sdk_limits_deg=sdk_limits, joint_source=source,
            source_plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest())
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output is not None:
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(serialized)
        print(serialized, end='')
        return 0 if report['kinematic_checks_passed'] else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({'ok': False, 'reason': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
