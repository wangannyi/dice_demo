"""Isolated pregrasp opening, with no arm target and passive default inspection.

The opening is an actual hand command. It is never presented as a position
query, and its target is the user-confirmed all-zero silver-cup approach posture.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import time

from nero_revo2_control.bridges.finger_feedback_probe import BroadcastCache, FINGERS, _copy_getter
from nero_revo2_control.bridges.passive_pose_bridge import PassivePoseSession, load_sdk_runtime
from nero_revo2_control.bridges.visual_servo_probe import (AuditedSendGuard, control_lock, evidence_blockers,
                                host_control_evidence, ready_blockers, stopped_window, take_can_control)


OPEN_TARGET = dict(zip(FINGERS, [0, 0, 0, 0, 0, 0]))


def send_open(hand, *, monotonic=time.monotonic, command_mode='timed'):
    if command_mode == 'position':
        hand.position_ctrl(**OPEN_TARGET)
        return None
    if command_mode != 'timed':
        raise ValueError('Unsupported hand command mode')
    start = monotonic()
    hand.position_time_ctrl(mode='pos', **OPEN_TARGET)
    hand.position_time_ctrl(mode='time', **dict.fromkeys(FINGERS, 100))
    interval = monotonic()-start
    if interval > .05:
        raise RuntimeError('Position/time command interval exceeded 50 ms')
    return interval


def run(bus_class, factory, *, execute=False, take_can=False, command_mode='timed',
        evidence_provider=host_control_evidence,
        monotonic=time.monotonic, wallclock=time.time, sleep=time.sleep):
    report = {'kind': 'isolated_pregrasp_open', 'execute_selected': execute,
              'target_0_100': OPEN_TARGET, 'target_user_confirmed': True, 'take_can_control_selected': take_can,
              'command_mode': command_mode,
              'duration_s': 1. if command_mode == 'timed' else None,
              'arm_target_sent': False, 'finger_close_sent': False,
              'position_feedback_valid': False, 'open_target_reached': False,
              'initial_position_feedback': None, 'grasp_ready': False,
              'success': False, 'blockers': []}
    guard = AuditedSendGuard(bus_class, monotonic=monotonic, wallclock=wallclock)
    cache = BroadcastCache(monotonic)
    holder = {}
    def with_effector():
        robot = factory()
        holder['hand'] = robot.init_effector('revo2')
        robot.get_context().register_parser_packet_fun(cache.receive)
        return robot
    session = PassivePoseSession(bus_class, with_effector, monotonic=monotonic,
                                 wallclock=wallclock, sleep=sleep)
    session.guard = guard
    try:
        before = evidence_provider('can0')
        report['sdk_ready'] = session.start()
        baseline, stationary = stopped_window(session, monotonic=monotonic,
                                              wallclock=wallclock, sleep=sleep)
        report['baseline_feedback'], report['stationarity'] = baseline, stationary
        for row in baseline:
            report['blockers'].extend(ready_blockers(row, take_can_control=take_can))
        current = evidence_provider('can0')
        report['host_before_connect'], report['host_before_first_tx'] = before, current
        report['blockers'].extend(evidence_blockers(before, current))
        hand = holder['hand']
        report['initial_position_feedback'] = _copy_getter(hand, 'position', cache, wallclock)
        active_current = _copy_getter(hand, 'current', cache, wallclock)
        report['initial_current_feedback'] = active_current
        if not active_current['fresh']:
            report['blockers'].append('No fresh effector current broadcast to establish communication')
        report['blockers'] = list(dict.fromkeys(report['blockers']))
        if not execute:
            report['event'], report['success'] = 'opening_inspected', True
        elif report['blockers']:
            report['event'] = 'opening_blocked'
        else:
            guard.permit()
            if baseline[-1]['status']['ctrl_mode'] == 3:
                report['can_handoff'] = take_can_control(
                    session, baseline[-1], timeout_s=3., monotonic=monotonic,
                    wallclock=wallclock, sleep=sleep)
            report['hand_command_epoch_s'] = wallclock()
            report['hand_command_monotonic_s'] = monotonic()
            report['position_time_interval_s'] = send_open(
                hand, monotonic=monotonic, command_mode=command_mode)
            deadline, previous, stable, samples = monotonic()+3., None, 0, []
            while monotonic() < deadline:
                row = _copy_getter(hand, 'position', cache, wallclock)
                stamp = row['timestamp_getter_epoch_s']
                if (row['fresh'] and stamp > report['hand_command_epoch_s']
                        and (previous is None or stamp > previous)):
                    previous = stamp
                    normalized = {name: round(value*100/255) for name, value in row['values'].items()}
                    error = max(abs(normalized[name]-OPEN_TARGET[name]) for name in FINGERS)
                    samples.append({'timestamp_epoch_s': stamp, 'raw_0_255': row['values'],
                                    'normalized_0_100': normalized, 'maximum_error': error})
                    stable = stable+1 if error <= 3 else 0
                    report['open_target_reached'] = stable >= 3
                sleep(.02)
            report['position_samples'] = samples
            report['position_feedback_valid'] = len(samples) >= 3
            report['latest_position_feedback'] = _copy_getter(hand, 'position', cache, wallclock)
            report['latest_current_feedback'] = _copy_getter(hand, 'current', cache, wallclock)
            if not report['latest_position_feedback']['fresh']:
                report['position_feedback_valid'] = False
            elif max(abs(round(value*100/255)-OPEN_TARGET[name])
                     for name, value in report['latest_position_feedback']['values'].items()) > 3:
                report['open_target_reached'] = False
            after, stopped = stopped_window(session, monotonic=monotonic,
                                             wallclock=wallclock, sleep=sleep)
            report['after_arm_feedback'], report['after_stationarity'] = after, stopped
            if any(ready_blockers(row) for row in after):
                raise RuntimeError('Arm readiness changed during hand-only preparation')
            span = max(abs(x-y) for r in after for x, y in zip(r['q_rad'], baseline[-1]['q_rad']))
            if span > .000872664626:
                raise RuntimeError('Arm moved during hand-only preparation')
            report['success'] = (report['position_feedback_valid'] and report['open_target_reached'])
            report['event'] = 'opening_verified' if report['success'] else 'opening_feedback_incomplete'
            if not report['success']:
                report['blockers'].append('Post-opening position broadcast or target feedback requirement failed')
    except BaseException as exc:
        report['event'], report['success'] = 'opening_failed', False
        report['error'] = type(exc).__name__+': '+str(exc)
    finally:
        guard.allowed = False
        try:
            session.close()
            report['sdk_disconnected'] = True
        except BaseException as exc:
            report['sdk_disconnected'], report['success'] = False, False
            report['disconnect_error'] = type(exc).__name__+': '+str(exc)
        report['tx'], report['broadcasts'] = guard.report(), cache.stats()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--take-can-control', action='store_true')
    parser.add_argument('--command-mode', choices=('timed', 'position'), default='timed')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with control_lock('/tmp/nero_can0_control.lock'), redirect_stdout(sys.stderr):
        bus, factory = load_sdk_runtime('can0')
        report = run(bus, factory, execute=args.execute, take_can=args.take_can_control,
                     command_mode=args.command_mode)
    report['tool_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ['event', 'success', 'blockers']}))
    return 0 if report['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
