"""Isolated current-pose +Z 1 mm flange probe; default inspection, no fingers.

SDK FK and sampled joint interpolation are model predictions. Current lowest
fingertip extent is unknown; whole-hand/table/cup clearance, contact registration
and actual controller trajectory are unverified. Execution requires the caller's
visual free-space check. The shared lock coordinates cooperating local senders.
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

from cartesian_microstep import _numbers, _pose, _rotation, _rotation_vector
from taught_pregrasp import plan_taught_pregrasp
from visual_servo_probe import (AuditedSendGuard, PassivePoseSession, TOLERANCE_RAD,
                                control_lock, evidence_blockers, fresh_feedback,
                                fresh_hold, host_control_evidence, joint_limits,
                                load_sdk_runtime, numbers, ready_blockers,
                                stopped_window, take_can_control, validate_target)


SCHEMA = 'isolated_positive_z_microprobe_v1'
LIFT_M = .001
MAX_JOINT_DELTA_RAD = math.radians(.35)
SOFT_MARGIN_RAD = math.radians(.1)
JOINT_TOLERANCE_RAD = math.radians(.02)
ROTATION_TOLERANCE_RAD = math.radians(.05)


def _lift(value):
    lift = numbers([value], 1, 'positive Z lift')[0]
    if lift != LIFT_M:
        raise ValueError('This bounded strategy accepts exactly +1 mm (solver minimum)')
    return lift


def _rotation_error(a, b):
    return float(np.linalg.norm(_rotation_vector(_rotation(a) @ _rotation(b).T)))


def plan_vertical_step(fk, current_q_rad, limits_rad, *, lift_m=LIFT_M):
    """Pure plan from current flange; check quarterpoints before publishing q."""
    report = {'valid': False, 'target_q_rad': None, 'error': None,
              'whole_hand_clearance_validated': False, 'physical_contact_verified': False,
              'physical_registration_valid': False, 'trajectory_collision_validated': False,
              'actual_trajectory_validated': False, 'current_fingertip_extent_m': None}
    try:
        lift = _lift(lift_m)
        current = _numbers(current_q_rad, (7,), 'seven current joints')
        limits = _numbers(limits_rad, (7, 2), 'seven joint limits')
        if (np.any(limits[:, 0] >= limits[:, 1])
                or np.any(current < limits[:, 0]+SOFT_MARGIN_RAD)
                or np.any(current > limits[:, 1]-SOFT_MARGIN_RAD)):
            raise ValueError('Current joints must satisfy all soft limits plus 0.1 degree margin')
        model = plan_taught_pregrasp(fk, current, limits, lift_m=lift,
            current_q_rad=current, max_joint_delta_rad=MAX_JOINT_DELTA_RAD,
            operator_same_cup_scene_confirmed=True)
        report['model_candidate'] = model
        if not model['valid']:
            raise ValueError('Bounded current-pose vertical model rejected: '+str(model['error']))
        target = _numbers(model['candidate_q_rad'], (7,), 'vertical candidate')
        before, goal = _pose(fk, current), _pose(fk, target)
        delta = target-current
        if np.max(np.abs(delta)) > MAX_JOINT_DELTA_RAD+1e-12:
            raise ValueError('Vertical target exceeds fixed 0.35 degree joint cap')
        samples, previous_z = [], before[2]
        for fraction in (0., .25, .5, .75, 1.):
            q = current+delta*fraction
            pose = _pose(fk, q)
            if (np.any(q < limits[:, 0]+SOFT_MARGIN_RAD)
                    or np.any(q > limits[:, 1]-SOFT_MARGIN_RAD)
                    or pose[2] < previous_z-1e-10
                    or np.max(np.abs(pose[:2]-before[:2])) > .0001
                    or _rotation_error(pose, before) > ROTATION_TOLERANCE_RAD):
                raise ValueError('Sampled vertical path violates monotonic Z, XY, rotation or soft limits')
            previous_z = pose[2]
            samples.append({'fraction': fraction, 'q_rad': q.tolist(), 'fk_m_rad': pose.tolist()})
        if goal[2]-before[2] < .0005 or np.linalg.norm(goal[:3]-(before[:3]+[0., 0., lift])) > .00002:
            raise ValueError('Vertical endpoint does not reach bounded positive Z target')
        report.update({'valid': True, 'target_q_rad': target.tolist(), 'current_q_rad': current.tolist(),
                       'fk_before_m_rad': before.tolist(), 'fk_goal_m_rad': goal.tolist(),
                       'delta_q_rad': delta.tolist(), 'requested_lift_m': lift,
                       'predicted_displacement_m': (goal[:3]-before[:3]).tolist(),
                       'sampled_model_path': samples, 'sampled_model_z_monotonic': True,
                       'sampled_model_path_is_actual_trajectory': False})
    except (ValueError, TypeError, KeyError, ArithmeticError, RuntimeError) as exc:
        report['error'] = type(exc).__name__+': '+str(exc)
    return report


def target_checks(feedback, baseline, plan):
    q = _numbers(feedback['q_rad'], (7,), 'fresh joints')
    target = _numbers(plan['target_q_rad'], (7,), 'vertical target joints')
    pose = _numbers(feedback['fk_flange_pose_m_rad'], (6,), 'fresh SDK FK')
    before = _numbers(baseline['fk_flange_pose_m_rad'], (6,), 'baseline SDK FK')
    goal = _numbers(plan['fk_goal_m_rad'], (6,), 'planned SDK FK')
    displacement = pose[:3]-before[:3]
    checks = {'ready_stationary': not ready_blockers(feedback),
              'all_seven_joints_at_target': bool(np.max(np.abs(q-target)) <= JOINT_TOLERANCE_RAD),
              'applied_positive_z': bool(displacement[2] >= .0005),
              'xy_drift': bool(np.max(np.abs(displacement[:2])) <= .0001),
              'rotation_retained': _rotation_error(pose, before) <= ROTATION_TOLERANCE_RAD,
              'position_goal': bool(np.linalg.norm(pose[:3]-goal[:3]) <= .00025)}
    return {'checks': checks, 'reached': all(checks.values()),
            'sdk_fk_displacement_m': displacement.tolist(),
            'sdk_fk_position_goal_error_m': float(np.linalg.norm(pose[:3]-goal[:3])),
            'sdk_fk_rotation_error_rad': _rotation_error(pose, before),
            'sdk_fk_is_model_not_visual_displacement': True}


def wait_target(session, baseline, plan, *, timeout_s, previous=None,
                monotonic=time.monotonic, wallclock=time.time, sleep=time.sleep):
    deadline, stable, samples = monotonic()+timeout_s, 0, []
    while monotonic() < deadline:
        row = fresh_feedback(session, previous=previous, monotonic=monotonic,
                             wallclock=wallclock, sleep=sleep)
        previous = row
        verification = target_checks(row, baseline, plan)
        samples.append({'feedback': row, 'verification': verification})
        if (row['status']['arm_status'] != 0 or row['status']['ctrl_mode'] != 1
                or row['enabled'] != [True]*7):
            raise RuntimeError('Arm readiness/control changed during positive Z probe')
        stable = stable+1 if verification['reached'] else 0
        if stable >= 10:
            return {'fresh_stable_samples': stable, 'samples': samples,
                    'joint_target_tolerance_deg': .02, 'minimum_applied_z_m': .0005,
                    'maximum_xy_drift_each_axis_m': .0001, 'rotation_tolerance_deg': .05,
                    'position_goal_tolerance_m': .00025}
        sleep(.05)
    raise TimeoutError('Positive Z probe did not satisfy strict all-axis/FK feedback')


def run_probe(bus_class, robot_factory, *, lift_m=LIFT_M, execute=False, take_control=False,
              noncontact_free_space_confirmed=False, channel='can0', timeout_s=5,
              evidence_provider=host_control_evidence, monotonic=time.monotonic,
              wallclock=time.time, sleep=time.sleep, session_factory=PassivePoseSession):
    """Caller holds shared lock and checks current free space visually."""
    report = {'schema': SCHEMA, 'success': False, 'execute_selected': execute,
              'take_can_control_selected': take_control, 'motion_target_valid': False,
              'joint_motion_attempted': False, 'finger_command_sent': False,
              'current_fingertip_extent_m': None, 'whole_hand_clearance_validated': False,
              'physical_contact_verified': False, 'physical_registration_valid': False,
              'trajectory_collision_validated': False, 'actual_trajectory_validated': False,
              'noncontact_free_space_confirmed': noncontact_free_space_confirmed,
              'free_space_confirmation_is_external_assertion': True,
              'blockers': [], 'started_monotonic_s': monotonic()}
    guard = AuditedSendGuard(bus_class, monotonic=monotonic, wallclock=wallclock)
    session = session_factory(bus_class, robot_factory, deadline_s=2,
                              monotonic=monotonic, wallclock=wallclock, sleep=sleep)
    session.guard = guard
    control_attempted, limits = False, None
    try:
        lift, timeout = _lift(lift_m), numbers([timeout_s], 1, 'timeout')[0]
        if not 1 <= timeout <= 10 or type(noncontact_free_space_confirmed) is not bool:
            raise ValueError('Timeout 1..10 seconds and boolean free-space assertion required')
        before = evidence_provider(channel)
        report['host_control_before_connect'] = before
        report['sdk_ready'] = session.start()
        feedback, stationary = stopped_window(session, monotonic=monotonic,
                                              wallclock=wallclock, sleep=sleep)
        report['baseline_feedback'], report['stationarity'] = feedback, stationary
        baseline = feedback[-1]
        limits = joint_limits(session.robot)
        current = validate_target(baseline['q_rad'], limits)
        plan = plan_vertical_step(session.robot.fk, current, limits, lift_m=lift)
        report['proposal'] = plan
        if not plan['valid']:
            report['blockers'].append('Vertical model plan rejected: '+str(plan['error']))
        if not noncontact_free_space_confirmed:
            report['blockers'].append('Current hand must be visually confirmed in noncontact free space')
        for row in feedback:
            report['blockers'].extend(ready_blockers(row, take_can_control=take_control))
        current_evidence = evidence_provider(channel)
        report['host_control_before_first_tx'] = current_evidence
        report['blockers'].extend(evidence_blockers(before, current_evidence))
        report['blockers'] = list(dict.fromkeys(report['blockers']))
        report['motion_target_valid'] = plan['valid'] and not report['blockers']
        if not execute:
            report['event'], report['success'] = 'vertical_microprobe_inspected', True
        elif report['blockers']:
            report['event'] = 'vertical_microprobe_blocked'
        else:
            target = validate_target(plan['target_q_rad'], limits)
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
                raise RuntimeError('Baseline changed before positive Z transmission')
            report['feedback_before_joint_tx'] = latest
            report['q_before_rad'], report['q_target_rad'] = current, target
            report['joint_motion_attempted'] = True
            report['move_j_sent_monotonic_s'] = monotonic()
            session.robot.move_j(target)
            report['move_j_call_end_monotonic_s'] = monotonic()
            report['target_verification'] = wait_target(session, baseline, plan, timeout_s=timeout,
                previous=latest, monotonic=monotonic, wallclock=wallclock, sleep=sleep)
            settled = report['target_verification']['samples'][-1]['feedback']
            report['strict_settled_monotonic_s'], report['q_after_rad'] = settled['observed_monotonic_s'], settled['q_rad']
            report['sdk_limits_enabled_at_completion'] = session.robot.get_joint_limits_enabled()
            if report['sdk_limits_enabled_at_completion'] is not True:
                raise RuntimeError('SDK joint limits became disabled')
            report['event'], report['success'] = 'vertical_microprobe_reached', True
    except BaseException as exc:
        report['event'], report['success'] = 'vertical_microprobe_failed', False
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
            report['event'] = 'vertical_microprobe_failed'
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
    parser.add_argument('--lift-mm', type=float, required=True)
    parser.add_argument('--noncontact-free-space-confirmed', action='store_true')
    parser.add_argument('--take-can-control', action='store_true')
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--timeout-s', type=float, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        _lift(args.lift_mm/1000)
        if not re.fullmatch('[A-Za-z0-9_-]+', args.channel) or not math.isfinite(args.timeout_s) or not 1 <= args.timeout_s <= 10:
            raise ValueError('Require valid interface and timeout 1..10 seconds')
    except ValueError as exc:
        parser.error(str(exc))
    lock_path = Path('/tmp')/('nero_'+args.channel+'_control.lock')
    try:
        with control_lock(lock_path), redirect_stdout(sys.stderr):
            bus_class, factory = load_sdk_runtime(args.channel)
            report = run_probe(bus_class, factory, lift_m=args.lift_mm/1000, execute=args.execute,
                take_control=args.take_can_control, noncontact_free_space_confirmed=args.noncontact_free_space_confirmed,
                channel=args.channel, timeout_s=args.timeout_s)
    except BaseException as exc:
        report = {'schema': SCHEMA, 'event': 'vertical_microprobe_startup_failed',
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
