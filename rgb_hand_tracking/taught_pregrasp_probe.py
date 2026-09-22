"""One <=0.2 degree joint segment toward a lifted taught flange; default inspect.

The reference remains inactive and physically unregistered. The only taught
source is the named descriptor and its SHA-bound pose dataset. An endpoint FK
prediction and one reached segment do not validate the remaining trajectory,
whole hand collision clearance, finger posture, cup contact or a grasp.
Default lift remains 20 mm. Lifts above 30 mm require explicit extended opt-in
and are bounded by 120 mm. Being below the requested high-position floor is
an execution blocker; an entire prospective rise never counts as clearance.
Rotational merit length defaults to 100 mm; explicit 50..100 mm settings
change solver weighting only, preserving all endpoint and execution gates.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from rgb_hand_tracking.cartesian_microstep import _numbers, _pose, _rotation, _rotation_vector
from rgb_hand_tracking.taught_pregrasp import plan_taught_pregrasp
from rgb_hand_tracking.visual_servo_probe import (AuditedSendGuard, PassivePoseSession, TOLERANCE_RAD,
                                control_lock, evidence_blockers, fresh_feedback,
                                fresh_hold, host_control_evidence, joint_limits,
                                load_sdk_runtime, numbers, ready_blockers,
                                stopped_window, take_can_control, validate_target)


SCHEMA = 'taught_pregrasp_first_segment_probe_v1'
REFERENCE_NAME = 'usb7_middle_root_pregrasp_reference_20260916_19.json'
MAX_SEGMENT_RAD = math.radians(.2)
JOINT_TOLERANCE_RAD = math.radians(.02)
SOFT_MARGIN_RAD = math.radians(.1)
LIFT_M = .020
EXTENDED_JOINT_CAPS_RAD = tuple(math.radians(v) for v in (30, 30, 30, 30, 10, 30, 10))


def _lift_parameters(lift_m, allow_extended_lift):
    lift = numbers([lift_m], 1, 'requested lift')[0]
    if type(allow_extended_lift) is not bool:
        raise ValueError('Extended lift requires explicit boolean opt-in')
    if not .001 <= lift <= (.120 if allow_extended_lift else .030):
        raise ValueError('Lift requires 1..30 mm, or explicit extended opt-in for up to 120 mm')
    caps = None
    if allow_extended_lift:
        vector = _numbers(EXTENDED_JOINT_CAPS_RAD, (7,), 'fixed extended joint caps')
        ceilings = np.radians([30, 30, 30, 30, 10, 30, 10])
        if np.any(vector <= 0) or not np.array_equal(vector, ceilings):
            raise ValueError('Extended caps must remain fixed at [30,30,30,30,10,30,10] degrees')
        caps = vector.tolist()
    return lift, caps


def _rotation_length(value):
    length = numbers([value], 1, 'rotation merit length')[0]
    if not .05 <= length <= .1:
        raise ValueError('Rotation merit length must be finite and between 50 and 100 mm')
    return length


def load_reference(path, *, camera_epoch, intrinsics_sha256, marker_attachment_epoch):
    """Load the named inactive descriptor; validate SHA-bound pose provenance."""
    path = Path(path)
    if path.name != REFERENCE_NAME or path.parent.name != 'config':
        raise ValueError('Only the specified config taught descriptor is accepted')
    blob = path.read_bytes()
    descriptor = json.loads(blob)
    if (descriptor.get('schema') != 1
            or descriptor.get('kind') != 'usb_rgb_operator_middle_root_pregrasp_reference'):
        raise ValueError('Invalid taught descriptor schema/kind')
    if any(descriptor.get(key) is not False for key in (
            'execution_enabled', 'motion_target_valid', 'reference_activation_valid', 'physical_contact_verified')):
        raise ValueError('Taught descriptor must remain inactive and physically unverified')
    if (descriptor.get('camera_configuration_epoch') != camera_epoch
            or descriptor.get('marker_attachment_epoch') != marker_attachment_epoch):
        raise ValueError('Taught camera or attachment epoch mismatch')
    if not isinstance(intrinsics_sha256, str) or not re.fullmatch('[0-9a-f]{64}', intrinsics_sha256):
        raise ValueError('Invalid current intrinsics SHA256')
    if (descriptor['marker'].get('T_marker_contact') is not None
            or descriptor['marker'].get('physical_branch_verified') is not False
            or descriptor['contact_definition'].get('physical_point_hand_frame_m') is not None):
        raise ValueError('Reference must not supply a physical contact registration')
    root = path.parent.parent.resolve()
    source = descriptor['sources']['pose_dataset']
    source_path = (root/source['path']).resolve()
    if not source_path.is_relative_to(root):
        raise ValueError('Bound pose dataset must remain within isolated root')
    source_blob = source_path.read_bytes()
    source_hash = hashlib.sha256(source_blob).hexdigest()
    if source_hash != source['sha256']:
        raise ValueError('Bound pose dataset SHA256 mismatch')
    dataset = json.loads(source_blob)
    if (dataset.get('intrinsics_sha256') != intrinsics_sha256
            or dataset.get('camera_configuration_epoch') != camera_epoch
            or dataset.get('marker_attachment_epoch') != marker_attachment_epoch
            or dataset.get('pose_record_valid') is not True):
        raise ValueError('Bound pose dataset intrinsics/epoch/record mismatch')
    if any(dataset.get(key) is not False for key in (
            'execution_enabled', 'motion_target_valid', 'reference_activation_valid',
            'physical_contact_verified', 'physical_branch_verified', 'physical_palm_transform_valid')):
        raise ValueError('Bound pose dataset must remain inactive and physically unverified')
    taught = _numbers(descriptor['arm']['joints_rad'], (7,), 'taught joint radians')
    degrees = _numbers(descriptor['arm']['joints_deg'], (7,), 'taught joint degrees')
    recorded = _numbers(dataset['reference_q_rad'], (7,), 'bound taught joints')
    if (not np.allclose(taught, recorded, rtol=0, atol=1e-12)
            or not np.allclose(np.degrees(taught), degrees, rtol=0, atol=1e-8)):
        raise ValueError('Descriptor taught joints differ from bound record or degree units')
    return {'taught_q_rad': taught.tolist(), 'reference_sha256': hashlib.sha256(blob).hexdigest(),
            'bound_pose_dataset_sha256': source_hash, 'reference_path': str(path.resolve()),
            'camera_epoch': camera_epoch, 'marker_attachment_epoch': marker_attachment_epoch,
            'intrinsics_sha256': intrinsics_sha256, 'physical_registration_valid': False}


def first_segment(fk, current_q_rad, candidate_q_rad, taught_fk_m_rad, limits_rad, *,
                  lift_m=LIFT_M, allow_extended_lift=False):
    """Plan exactly one interpolated joint segment, never an unchecked macro."""
    current = _numbers(current_q_rad, (7,), 'current joints')
    candidate = _numbers(candidate_q_rad, (7,), 'candidate joints')
    limits = _numbers(limits_rad, (7, 2), 'joint limits')
    taught_pose = _numbers(taught_fk_m_rad, (6,), 'original taught FK')
    lift, _ = _lift_parameters(lift_m, allow_extended_lift)
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError('Invalid supplied joint limits')
    delta = candidate-current
    maximum = float(np.max(np.abs(delta)))
    count = max(1, math.ceil(maximum/MAX_SEGMENT_RAD))
    next_q = current+delta/count
    if (np.any(candidate < limits[:, 0]+SOFT_MARGIN_RAD)
            or np.any(candidate > limits[:, 1]-SOFT_MARGIN_RAD)
            or np.any(next_q < limits[:, 0]+SOFT_MARGIN_RAD)
            or np.any(next_q > limits[:, 1]-SOFT_MARGIN_RAD)):
        raise ValueError('Candidate or first segment violates 0.1 degree soft margin')
    before, next_pose = _pose(fk, current), _pose(fk, next_q)
    translation = float(np.linalg.norm(next_pose[:3]-before[:3]))
    rotation = float(np.linalg.norm(_rotation_vector(_rotation(next_pose) @ _rotation(before).T)))
    floor = float(taught_pose[2]+lift-.00002)
    if (np.max(np.abs(next_q-current)) > MAX_SEGMENT_RAD+1e-12 or translation > .002
            or rotation > math.radians(.35)
            or (not allow_extended_lift and next_pose[2] < floor)):
        raise ValueError('First segment violates joint/2mm/0.35degree/lifted-height gates')
    blockers = []
    if allow_extended_lift:
        if before[2] < floor:
            blockers.append('Current flange is below requested high-position floor; whole-hand clearance is unverified')
        if next_pose[2] < floor:
            blockers.append('First endpoint remains below requested high-position floor; full prospective lift is not clearance')
        if before[2] < floor and next_pose[2] < before[2]-1e-12:
            blockers.append('First extended segment predicts downward flange motion; upward path must be established separately')
    if count > 1 and np.allclose(next_q, candidate, rtol=0, atol=1e-12):
        raise ValueError('Unchecked full macro endpoint must not be transmitted')
    return {'current_q_rad': current.tolist(), 'candidate_q_rad': candidate.tolist(),
            'next_q_rad': next_q.tolist(), 'fk_current_m_rad': before.tolist(),
            'fk_next_m_rad': next_pose.tolist(), 'remaining_interpolated_segments': count,
            'requested_max_joint_delta_rad': float(np.max(np.abs(next_q-current))),
            'candidate_max_joint_delta_rad': maximum,
            'already_pregrasp': maximum < JOINT_TOLERANCE_RAD,
            'next_equals_candidate': bool(np.array_equal(next_q, candidate)),
            'predicted_translation_m': translation, 'predicted_rotation_change_rad': rotation,
            'requested_lift_m': lift, 'allow_extended_lift': allow_extended_lift,
            'minimum_flange_z_m': floor, 'blocking_reasons': blockers,
            'current_flange_floor_satisfied': bool(before[2] >= floor),
            'first_endpoint_floor_satisfied': bool(next_pose[2] >= floor),
            'first_model_z_change_m': float(next_pose[2]-before[2]),
            'whole_hand_clearance_validated': False, 'physical_contact_verified': False,
            'trajectory_collision_validated': False, 'trajectory_validated': False,
            'full_macro_goal_transmitted': False, 'physical_registration_valid': False}


def target_checks(feedback, baseline, segment):
    q = _numbers(feedback['q_rad'], (7,), 'fresh joints')
    current = _numbers(baseline['q_rad'], (7,), 'baseline joints')
    target = _numbers(segment['next_q_rad'], (7,), 'next joints')
    pose = _numbers(feedback['fk_flange_pose_m_rad'], (6,), 'fresh SDK FK')
    goal = _numbers(segment['fk_next_m_rad'], (6,), 'next SDK FK')
    rotation_error = float(np.linalg.norm(_rotation_vector(_rotation(pose) @ _rotation(goal).T)))
    change = float(np.max(np.abs(q-current)))
    checks = {'ready_stationary': not ready_blockers(feedback),
              'all_seven_joints_at_target': bool(np.max(np.abs(q-target)) <= JOINT_TOLERANCE_RAD),
              'minimum_applied_joint_change': change >= .5*segment['requested_max_joint_delta_rad'],
              'position_goal': bool(np.linalg.norm(pose[:3]-goal[:3]) <= .00025),
              'rotation_goal': rotation_error <= math.radians(.05)}
    return {'checks': checks, 'reached': all(checks.values()),
            'applied_max_joint_delta_rad': change,
            'sdk_fk_position_goal_error_m': float(np.linalg.norm(pose[:3]-goal[:3])),
            'sdk_fk_rotation_goal_error_rad': rotation_error,
            'sdk_fk_is_model_not_visual_displacement': True}


def wait_target(session, baseline, segment, *, timeout_s, previous=None,
                monotonic=time.monotonic, wallclock=time.time, sleep=time.sleep):
    deadline, consecutive, samples = monotonic()+timeout_s, 0, []
    while monotonic() < deadline:
        feedback = fresh_feedback(session, previous=previous, monotonic=monotonic,
                                  wallclock=wallclock, sleep=sleep)
        previous = feedback
        verification = target_checks(feedback, baseline, segment)
        samples.append({'feedback': feedback, 'verification': verification})
        if (feedback['status']['arm_status'] != 0 or feedback['status']['ctrl_mode'] != 1
                or feedback['enabled'] != [True]*7):
            raise RuntimeError('Arm readiness/control changed during taught segment')
        consecutive = consecutive+1 if verification['reached'] else 0
        if consecutive >= 10:
            return {'fresh_stable_samples': consecutive, 'samples': samples,
                    'joint_target_tolerance_deg': .02, 'position_tolerance_m': .00025,
                    'rotation_tolerance_deg': .05, 'minimum_applied_joint_fraction': .5}
        sleep(.05)
    raise TimeoutError('First taught segment did not satisfy strict all-axis/FK feedback')


def run_probe(bus_class, robot_factory, *, reference, execute=False, take_control=False,
              operator_same_scene_confirmed=False, channel='can0', timeout_s=5,
              lift_m=LIFT_M, allow_extended_lift=False, rotation_length_m=.1,
              evidence_provider=host_control_evidence, monotonic=time.monotonic,
              wallclock=time.time, sleep=time.sleep, session_factory=PassivePoseSession):
    """Caller holds common control lock; reference is from load_reference()."""
    report = {'schema': SCHEMA, 'execute_selected': execute, 'take_can_control_selected': take_control,
              'success': False, 'motion_target_valid': False, 'joint_motion_attempted': False,
              'physical_registration_valid': False, 'finger_command_sent': False,
              'whole_hand_clearance_validated': False, 'physical_contact_verified': False,
              'trajectory_validated': False,
              'trajectory_collision_validated': False, 'full_macro_goal_transmitted': False,
              'reference': reference, 'blockers': [], 'started_monotonic_s': monotonic()}
    guard = AuditedSendGuard(bus_class, monotonic=monotonic, wallclock=wallclock)
    session = session_factory(bus_class, robot_factory, deadline_s=2,
                              monotonic=monotonic, wallclock=wallclock, sleep=sleep)
    session.guard = guard
    control_attempted, limits = False, None
    try:
        timeout = numbers([timeout_s], 1, 'timeout')[0]
        if not 1 <= timeout <= 10 or operator_same_scene_confirmed is not True:
            raise ValueError('Timeout 1..10 seconds and operator same-scene confirmation required')
        lift, caps = _lift_parameters(lift_m, allow_extended_lift)
        rotation_length = _rotation_length(rotation_length_m)
        report['requested_lift_m'], report['allow_extended_lift'] = lift, allow_extended_lift
        report['per_joint_delta_caps_rad'] = caps
        report['rotation_length_m'] = rotation_length
        report['rotation_weight_is_merit_only'] = True
        before = evidence_provider(channel)
        report['host_control_before_connect'] = before
        report['sdk_ready'] = session.start()
        feedback, stationary = stopped_window(session, monotonic=monotonic,
                                              wallclock=wallclock, sleep=sleep)
        report['baseline_feedback'], report['stationarity'] = feedback, stationary
        baseline = feedback[-1]
        limits = joint_limits(session.robot)
        current = validate_target(baseline['q_rad'], limits)
        candidate = plan_taught_pregrasp(session.robot.fk, reference['taught_q_rad'], limits,
            lift_m=lift, current_q_rad=current, operator_same_cup_scene_confirmed=True,
            allow_extended_lift=allow_extended_lift, per_joint_delta_caps_rad=caps,
            rotation_length_m=rotation_length)
        report['model_candidate'] = candidate
        if candidate['valid']:
            segment = first_segment(session.robot.fk, current, candidate['candidate_q_rad'],
                                    candidate['taught_fk_flange_m_rad'], limits,
                                    lift_m=lift, allow_extended_lift=allow_extended_lift)
            report['proposal'] = segment
            report['blockers'].extend(segment['blocking_reasons'])
        else:
            report['blockers'].append('Taught model candidate rejected: '+str(candidate['error']))
        for row in feedback:
            report['blockers'].extend(ready_blockers(row, take_can_control=take_control))
        current_evidence = evidence_provider(channel)
        report['host_control_before_first_tx'] = current_evidence
        report['blockers'].extend(evidence_blockers(before, current_evidence))
        report['blockers'] = list(dict.fromkeys(report['blockers']))
        report['motion_target_valid'] = candidate['valid'] and not report['blockers']
        if not execute:
            report['event'], report['success'] = 'taught_pregrasp_inspected', True
        elif report['blockers']:
            report['event'] = 'taught_pregrasp_blocked'
        elif segment['already_pregrasp']:
            report['event'], report['success'] = 'already_pregrasp', True
        else:
            target = validate_target(segment['next_q_rad'], limits)
            guard.permit()
            if baseline['status']['ctrl_mode'] == 3:
                control_attempted = True
                report['can_handoff'] = take_can_control(session, baseline, timeout_s=timeout,
                    monotonic=monotonic, wallclock=wallclock, sleep=sleep)
            session.robot.set_joint_limits_enabled(True)
            if session.robot.get_joint_limits_enabled() is not True:
                raise RuntimeError('SDK soft limits did not remain enabled')
            control_attempted = True
            session.robot.set_speed_percent(1)
            previous = report['can_handoff']['samples'][-1] if 'can_handoff' in report else baseline
            latest = fresh_feedback(session, previous=previous, monotonic=monotonic,
                                    wallclock=wallclock, sleep=sleep)
            if (ready_blockers(latest) or max(abs(a-b) for a, b in zip(latest['q_rad'], current)) > TOLERANCE_RAD):
                raise RuntimeError('Baseline changed before first-segment transmission')
            report['feedback_before_joint_tx'] = latest
            report['q_before_rad'], report['q_target_rad'] = current, target
            report['joint_motion_attempted'] = True
            report['move_j_sent_monotonic_s'] = monotonic()
            session.robot.move_j(target)
            report['move_j_call_end_monotonic_s'] = monotonic()
            report['target_verification'] = wait_target(session, baseline, segment, timeout_s=timeout,
                previous=latest, monotonic=monotonic, wallclock=wallclock, sleep=sleep)
            settled = report['target_verification']['samples'][-1]['feedback']
            report['strict_settled_monotonic_s'] = settled['observed_monotonic_s']
            report['q_after_rad'] = settled['q_rad']
            report['sdk_limits_enabled_at_completion'] = session.robot.get_joint_limits_enabled()
            if report['sdk_limits_enabled_at_completion'] is not True:
                raise RuntimeError('SDK joint limits became disabled')
            report['event'], report['success'] = 'taught_pregrasp_first_segment_reached', True
    except BaseException as exc:
        report['event'], report['success'] = 'taught_pregrasp_failed', False
        report['error'] = type(exc).__name__+': '+str(exc)
        if execute and control_attempted and limits is not None:
            try:
                report['hold'] = fresh_hold(session, limits, monotonic=monotonic, wallclock=wallclock, sleep=sleep)
            except BaseException as hold_error:
                report['hold'] = {'requested': False, 'hold_verified': False,
                                  'error': type(hold_error).__name__+': '+str(hold_error)}
    finally:
        guard.allowed = False
        try:
            session.close()
            report['sdk_disconnected'] = True
        except BaseException as exc:
            report['sdk_disconnected'], report['success'] = False, False
            report['disconnect_error'] = type(exc).__name__+': '+str(exc)
            report['event'] = 'taught_pregrasp_failed'
        finally:
            if report.get('sdk_disconnected'):
                guard.restore()
        report['tx'] = guard.report()
        report['ended_monotonic_s'], report['ended_epoch_s'] = monotonic(), wallclock()
        report['hardware_synchronized'] = False
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--inspect', action='store_true')
    action.add_argument('--execute', action='store_true')
    parser.add_argument('--camera-epoch', required=True)
    parser.add_argument('--intrinsics-sha256', required=True)
    parser.add_argument('--marker-attachment-epoch', required=True)
    parser.add_argument('--operator-same-scene-confirmed', action='store_true')
    parser.add_argument('--lift-mm', type=float, default=20.)
    parser.add_argument('--allow-extended-lift', action='store_true')
    parser.add_argument('--rotation-length-mm', type=float, default=100.)
    parser.add_argument('--take-can-control', action='store_true')
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--timeout-s', type=float, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if (not args.operator_same_scene_confirmed or not re.fullmatch('[A-Za-z0-9_-]+', args.channel)
            or not math.isfinite(args.timeout_s) or not 1 <= args.timeout_s <= 10):
        parser.error('Require same-scene confirmation, valid interface and timeout 1..10 seconds')
    try:
        _lift_parameters(args.lift_mm/1000, args.allow_extended_lift)
        _rotation_length(args.rotation_length_mm/1000)
    except ValueError as exc:
        parser.error(str(exc))
    lock_path = Path('/tmp')/('nero_'+args.channel+'_control.lock')
    try:
        reference = load_reference(Path(__file__).parent/'config'/REFERENCE_NAME,
            camera_epoch=args.camera_epoch, intrinsics_sha256=args.intrinsics_sha256,
            marker_attachment_epoch=args.marker_attachment_epoch)
        with control_lock(lock_path), redirect_stdout(sys.stderr):
            bus_class, factory = load_sdk_runtime(args.channel)
            report = run_probe(bus_class, factory, reference=reference, execute=args.execute,
                take_control=args.take_can_control, operator_same_scene_confirmed=True,
                channel=args.channel, timeout_s=args.timeout_s,
                lift_m=args.lift_mm/1000, allow_extended_lift=args.allow_extended_lift,
                rotation_length_m=args.rotation_length_mm/1000)
    except BaseException as exc:
        report = {'schema': SCHEMA, 'event': 'taught_pregrasp_startup_failed',
                  'success': False, 'error': type(exc).__name__+': '+str(exc)}
    report['control_lock_path'] = str(lock_path)
    report['tool_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['planner_source_sha256'] = hashlib.sha256(Path(__file__).with_name('taught_pregrasp.py').read_bytes()).hexdigest()
    output = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        temporary = args.output.with_name(args.output.name+'.'+str(os.getpid())+'.tmp')
        temporary.write_text(output)
        temporary.replace(args.output)
    sys.stdout.write(output)
    return 0 if report['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
