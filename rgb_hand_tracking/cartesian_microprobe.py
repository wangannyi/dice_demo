"""Isolated <=1 mm base-frame X/Y flange probe; inspection is the default.

Use the SDK Python runtime with NumPy. Importing opens no devices. This tool
never sends finger commands. FK verification is a robot-model calculation,
not a measurement of actual visual/contact-point displacement or trajectory.
The shared lock coordinates cooperating local controllers only.
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
from rgb_hand_tracking.cartesian_microstep import (_rotation, _rotation_vector, plan_microstep,
                                MAX_TRANSLATION_M)
from rgb_hand_tracking.visual_servo_probe import (AuditedSendGuard, PassivePoseSession, TOLERANCE_RAD,
                                control_lock, evidence_blockers, fresh_feedback,
                                fresh_hold, host_control_evidence, joint_limits,
                                load_sdk_runtime, numbers, ready_blockers,
                                stopped_window, take_can_control, validate_target)


SCHEMA = 'cartesian_flange_microprobe_v1'
JOINT_TARGET_TOLERANCE_RAD = math.radians(.02)
POSITION_TARGET_TOLERANCE_M = .00025
Z_TOLERANCE_M = .0001
ORIENTATION_TOLERANCE_RAD = math.radians(.05)


def _request(axis, distance_m, timeout_s):
    distance, timeout = numbers([distance_m, timeout_s], 2, 'probe parameters')
    if axis not in ('x', 'y') or not 0 < abs(distance) <= MAX_TRANSLATION_M:
        raise ValueError('Require base axis x/y and nonzero translation <=1 mm')
    if not 1 <= timeout <= 10:
        raise ValueError('Timeout must be between 1 and 10 seconds')
    return distance, timeout


def target_checks(feedback, baseline, target, plan):
    """All-axis feedback and endpoint SDK FK checks; no visual-point claim."""
    q = numbers(feedback['q_rad'], 7, 'fresh joints')
    pose = numbers(feedback['fk_flange_pose_m_rad'], 6, 'fresh SDK flange FK')
    before = numbers(baseline['fk_flange_pose_m_rad'], 6, 'baseline SDK flange FK')
    goal = numbers(plan['target_position_m'], 3, 'planned position')
    requested = plan['requested_distance_m']
    axis = 0 if plan['axis'] == 'x' else 1
    applied = pose[axis]-before[axis]
    rotation_error = float(np.linalg.norm(_rotation_vector(_rotation(pose) @ _rotation(before).T)))
    position_error = math.dist(pose[:3], goal)
    checks = {'normal_can_enabled_stationary': not ready_blockers(feedback),
              'all_seven_joints_at_target': all(abs(a-b) <= JOINT_TARGET_TOLERANCE_RAD
                                                for a, b in zip(q, target)),
              'applied_axis_change': math.copysign(1., requested)*applied >= abs(requested)*.5,
              'position_goal': position_error <= POSITION_TARGET_TOLERANCE_M,
              'z_retained': abs(pose[2]-before[2]) <= Z_TOLERANCE_M,
              'orientation_retained': rotation_error <= ORIENTATION_TOLERANCE_RAD}
    return {'checks': checks, 'reached': all(checks.values()),
            'sdk_fk_applied_axis_displacement_m': applied,
            'sdk_fk_position_goal_error_m': position_error,
            'sdk_fk_z_change_m': pose[2]-before[2],
            'sdk_fk_orientation_error_rad': rotation_error,
            'sdk_fk_is_model_not_visual_displacement': True}


def wait_target(session, baseline, target, plan, *, timeout_s, previous=None,
                monotonic=time.monotonic, wallclock=time.time, sleep=time.sleep):
    deadline, consecutive, samples = monotonic()+timeout_s, 0, []
    while monotonic() < deadline:
        feedback = fresh_feedback(session, previous=previous, monotonic=monotonic,
                                  wallclock=wallclock, sleep=sleep)
        previous = feedback
        verification = target_checks(feedback, baseline, target, plan)
        samples.append({'feedback': feedback, 'verification': verification})
        if (feedback['status']['arm_status'] != 0 or feedback['status']['ctrl_mode'] != 1
                or feedback['enabled'] != [True]*7):
            raise RuntimeError('Arm readiness/control changed during Cartesian microprobe')
        consecutive = consecutive+1 if verification['reached'] else 0
        if consecutive >= 10:
            return {'fresh_stable_samples': consecutive, 'samples': samples,
                    'joint_target_tolerance_deg': .02, 'position_goal_tolerance_m': .00025,
                    'z_tolerance_m': .0001, 'orientation_tolerance_deg': .05,
                    'minimum_requested_axis_fraction': .5,
                    'sdk_fk_is_model_not_visual_displacement': True}
        sleep(.05)
    raise TimeoutError('Cartesian probe did not reach strict all-axis/FK target')


def run_probe(bus_class, robot_factory, *, axis, distance_m, execute=False,
              take_control=False, channel='can0', timeout_s=5,
              evidence_provider=host_control_evidence, monotonic=time.monotonic,
              wallclock=time.time, sleep=time.sleep, session_factory=PassivePoseSession):
    """Caller must hold common control lock; injected dependencies enable CPU tests."""
    report = {'schema': SCHEMA, 'execute_selected': execute,
              'take_can_control_selected': take_control, 'motion_target_valid': False,
              'physical_palm_registered': False, 'finger_command_sent': False,
              'sdk_fk_is_model_not_visual_displacement': True,
              'success': False, 'joint_motion_attempted': False, 'blockers': [],
              'started_monotonic_s': monotonic(), 'started_epoch_s': wallclock()}
    guard = AuditedSendGuard(bus_class, monotonic=monotonic, wallclock=wallclock)
    session = session_factory(bus_class, robot_factory, deadline_s=2,
                              monotonic=monotonic, wallclock=wallclock, sleep=sleep)
    session.guard = guard
    control_attempted, limits = False, None
    try:
        distance_m, timeout_s = _request(axis, distance_m, timeout_s)
        report['requested_probe'] = {'axis': axis, 'distance_m': distance_m, 'speed_percent': 1}
        before = evidence_provider(channel)
        report['host_control_before_connect'] = before
        report['sdk_ready'] = session.start()  # Denying hook precedes factory/connect.
        feedback, stationary = stopped_window(session, monotonic=monotonic,
                                              wallclock=wallclock, sleep=sleep)
        report['baseline_feedback'], report['stationarity'] = feedback, stationary
        baseline = feedback[-1]
        limits = joint_limits(session.robot)
        current = validate_target(baseline['q_rad'], limits)
        plan = plan_microstep(session.robot.fk, current, limits, axis=axis, distance_m=distance_m)
        report['proposal'] = plan
        if not plan['valid']:
            report['blockers'].append('Offline Cartesian plan rejected: '+str(plan['error']))
        for row in feedback:
            report['blockers'].extend(ready_blockers(row, take_can_control=take_control))
        current_evidence = evidence_provider(channel)
        report['host_control_before_first_tx'] = current_evidence
        report['blockers'].extend(evidence_blockers(before, current_evidence))
        report['blockers'] = list(dict.fromkeys(report['blockers']))
        report['motion_target_valid'] = plan['valid'] and not report['blockers']
        if not execute:
            report['event'], report['success'] = 'cartesian_microprobe_inspected', True
        elif report['blockers']:
            report['event'] = 'cartesian_microprobe_blocked'
        else:
            target = validate_target(plan['target_q_rad'], limits)
            guard.permit()
            if baseline['status']['ctrl_mode'] == 3:
                control_attempted = True
                report['can_handoff'] = take_can_control(
                    session, baseline, timeout_s=timeout_s, monotonic=monotonic,
                    wallclock=wallclock, sleep=sleep)
            session.robot.set_joint_limits_enabled(True)
            if session.robot.get_joint_limits_enabled() is not True:
                raise RuntimeError('SDK soft limits did not remain enabled')
            control_attempted = True
            session.robot.set_speed_percent(1)
            previous = (report['can_handoff']['samples'][-1]
                        if 'can_handoff' in report else baseline)
            latest = fresh_feedback(session, previous=previous, monotonic=monotonic,
                                    wallclock=wallclock, sleep=sleep)
            if (ready_blockers(latest) or max(abs(a-b) for a, b in zip(latest['q_rad'], current))
                    > TOLERANCE_RAD):
                raise RuntimeError('Baseline changed before joint target transmission')
            report['feedback_before_joint_tx'] = latest
            report['q_before_rad'], report['q_target_rad'] = current, target
            report['joint_motion_attempted'] = True
            report['move_j_sent_monotonic_s'] = monotonic()
            report['move_j_clock_definition'] = 'host_monotonic immediately before SDK call'
            session.robot.move_j(target)
            report['move_j_call_end_monotonic_s'] = monotonic()
            report['target_verification'] = wait_target(
                session, baseline, target, plan, timeout_s=timeout_s, previous=latest,
                monotonic=monotonic, wallclock=wallclock, sleep=sleep)
            settled = report['target_verification']['samples'][-1]['feedback']
            report['strict_settled_monotonic_s'] = settled['observed_monotonic_s']
            report['q_after_rad'] = settled['q_rad']
            report['actual_joint_delta_deg'] = [math.degrees(a-b)
                                                for a, b in zip(settled['q_rad'], current)]
            report['sdk_limits_enabled_at_completion'] = session.robot.get_joint_limits_enabled()
            if report['sdk_limits_enabled_at_completion'] is not True:
                raise RuntimeError('SDK joint limits became disabled')
            report['event'], report['success'] = 'cartesian_microprobe_reached', True
    except BaseException as exc:
        report['event'], report['success'] = 'cartesian_microprobe_failed', False
        report['error'] = type(exc).__name__+': '+str(exc)
        if execute and control_attempted and limits is not None:
            try:
                report['hold'] = fresh_hold(session, limits, monotonic=monotonic,
                                            wallclock=wallclock, sleep=sleep)
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
            report['event'] = 'cartesian_microprobe_failed'
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
    parser.add_argument('--axis', choices=('x', 'y'), required=True)
    parser.add_argument('--distance-mm', type=float, required=True)
    parser.add_argument('--take-can-control', action='store_true')
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--timeout-s', type=float, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        _request(args.axis, args.distance_mm/1000, args.timeout_s)
        if not re.fullmatch('[A-Za-z0-9_-]+', args.channel):
            raise ValueError('Invalid interface name')
    except ValueError as exc:
        parser.error(str(exc))
    lock_path = Path('/tmp')/('nero_'+args.channel+'_control.lock')
    try:
        with control_lock(lock_path), redirect_stdout(sys.stderr):
            bus_class, factory = load_sdk_runtime(args.channel)
            report = run_probe(bus_class, factory, axis=args.axis,
                               distance_m=args.distance_mm/1000, execute=args.execute,
                               take_control=args.take_can_control, channel=args.channel,
                               timeout_s=args.timeout_s)
    except BaseException as exc:
        report = {'schema': SCHEMA, 'event': 'cartesian_microprobe_startup_failed',
                  'success': False, 'error': type(exc).__name__+': '+str(exc)}
    report['control_lock_path'] = str(lock_path)
    report['tool_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['planner_source_sha256'] = hashlib.sha256(
        Path(__file__).with_name('cartesian_microstep.py').read_bytes()).hexdigest()
    output = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        temporary = args.output.with_name(args.output.name+'.'+str(os.getpid())+'.tmp')
        temporary.write_text(output)
        temporary.replace(args.output)
    sys.stdout.write(output)
    return 0 if report['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
