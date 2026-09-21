"""Isolated fixed J4 +0.2 degree probe, with passive inspection by default.

Use the SDK Python runtime. Importing uses only the standard library and opens
no devices. A process lock and same-host CAN/process observations coordinate
cooperating controllers; they cannot enforce exclusion of WEB or remote users.
"""
import argparse
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from passive_pose_bridge import PassivePoseSession, PassiveTransmitForbidden, load_sdk_runtime


SCHEMA = 'visual_servo_fixed_j4_probe_v1'
DELTA_RAD = math.radians(.2)
TOLERANCE_RAD = math.radians(.05)
PACKETS = ('joint_12', 'joint_34', 'joint_56', 'joint_7')


def numbers(values, length, label):
    if (not isinstance(values, (list, tuple)) or len(values) != length
            or any(isinstance(value, bool) for value in values)):
        raise ValueError('Invalid '+label)
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError('Nonfinite '+label)
    return result


class AuditedSendGuard:
    """One send hook from factory to disconnect: deny first, audit when allowed."""

    def __init__(self, bus_class, *, monotonic=time.monotonic, wallclock=time.time):
        self.bus_class, self.monotonic, self.wallclock = bus_class, monotonic, wallclock
        self.installed = self.allowed = False
        self.original_send = None
        self.lock = threading.Lock()
        self.history = []
        self.denied = 0

    @property
    def attempts(self):
        # PassivePoseSession's check contract counts forbidden attempts only.
        with self.lock:
            return self.denied

    def install(self):
        if self.installed:
            return
        self.original_send = self.bus_class.send

        def send(bus, message, *args, **kwargs):
            row = {'monotonic_s': self.monotonic(), 'epoch_s': self.wallclock(),
                   'arbitration_id': getattr(message, 'arbitration_id', None),
                   'data_hex': bytes(getattr(message, 'data', b'')).hex(),
                   'is_extended_id': bool(getattr(message, 'is_extended_id', False)),
                   'allowed': self.allowed, 'send_returned_successfully': False}
            with self.lock:
                self.history.append(row)
                if not self.allowed:
                    self.denied += 1
            if not row['allowed']:
                raise PassiveTransmitForbidden('Inspection forbids CAN transmit')
            try:
                result = self.original_send(bus, message, *args, **kwargs)
            except BaseException as exc:
                with self.lock:
                    row['error'] = type(exc).__name__+': '+str(exc)
                    row['transmission_outcome_uncertain'] = True
                raise
            with self.lock:
                row['send_returned_successfully'] = True
                row['send_end_monotonic_s'] = self.monotonic()
            return result

        self.bus_class.send = send
        self.installed = True

    def check(self):
        if self.attempts:
            raise PassiveTransmitForbidden('Forbidden CAN transmit was attempted')

    def permit(self):
        self.check()
        if not self.installed:
            raise RuntimeError('Send guard is not installed')
        self.allowed = True

    def restore(self):
        if self.installed:
            self.bus_class.send = self.original_send
            self.installed = False

    def report(self):
        with self.lock:
            history = deepcopy(self.history)
        return {'tx_attempts': len(history), 'denied_tx_attempts': self.denied,
                'actual_tx_count': sum(row['send_returned_successfully'] for row in history),
                'actual_tx_count_definition': 'native bus.send returned; not wire acknowledgement',
                'transmission_outcome_uncertain': any(row.get('transmission_outcome_uncertain')
                                                       for row in history),
                'history': history, 'guard_installed': self.installed,
                'transmit_permitted': self.allowed}


def ancestors(pid, proc_root=Path('/proc')):
    result = {pid}
    while pid > 1:
        try:
            fields = (proc_root/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            break
        if pid in result:
            break
        result.add(pid)
    return result


def host_control_evidence(channel, *, proc_root=Path('/proc'), own_pid=None,
                          monotonic=time.monotonic, wallclock=time.time):
    """Read snapshots only; never stop a process or send/query a CAN frame."""
    rows, errors = [], []
    for kind in ('all', 'fil', 'sff', 'eff', 'err', 'inv'):
        try:
            content = (proc_root/'net/can'/('rcvlist_'+kind)).read_text()
        except OSError as exc:
            errors.append(type(exc).__name__+': '+str(exc))
            continue
        for line in content.splitlines():
            fields = line.split()
            if (len(fields) >= 3 and fields[0] in (channel, 'any')
                    and re.fullmatch('[0-9a-fA-F]+', fields[1])):
                rows.append({'list': kind, 'line': line.strip()})
    candidates = []
    excluded = ancestors(os.getpid() if own_pid is None else own_pid, proc_root)
    try:
        processes = subprocess.check_output(['ps', '-eo', 'pid,ppid,args'], text=True, timeout=3)
        pattern = re.compile(r'(nero|revo2|agilex|pyAgx|passive_pose_bridge|visual_servo|'
                             r'cup_grasp|ros2|moveit|cansend|canplayer|can-utils)', re.IGNORECASE)
        for line in processes.splitlines()[1:]:
            fields = line.split(None, 2)
            if len(fields) == 3 and int(fields[0]) not in excluded and pattern.search(fields[2]):
                candidates.append({'pid': int(fields[0]), 'ppid': int(fields[1]), 'args': fields[2]})
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        errors.append(type(exc).__name__+': '+str(exc))
    return {'observed_monotonic_s': monotonic(), 'observed_epoch_s': wallclock(),
            'channel': channel, 'receiver_rows': rows, 'candidate_control_processes': candidates,
            'errors': errors,
            'scope': 'same-host snapshot; lock only coordinates cooperating processes; '
                     'WEB/remote exclusion relies on authorized operator coordination'}


def evidence_blockers(before, current):
    result = []
    if before.get('errors') or current.get('errors'):
        result.append('Host control evidence is unavailable or incomplete')
    if before.get('receiver_rows'):
        result.append('CAN receivers existed before this SDK connection')
    # This audited SocketcanBus runtime registers one catch-all plus one
    # 0x1fffffff error filter. They are two registrations of one socket.
    # Require the observed empty baseline and exact default layout; this
    # remains same-host snapshot evidence, not enforcement of WEB exclusion.
    rows = current.get('receiver_rows', [])
    layout = []
    for row in rows:
        fields = row.get('line', '').split()
        if len(fields) >= 3:
            layout.append((row.get('list'), fields[0], fields[1], fields[2]))
    channel = current.get('channel')
    expected = [('all', channel, '000', '00000000'), ('err', channel, '000', '1fffffff')]
    if len(rows) != 2 or sorted(layout) != sorted(expected):
        result.append('CAN receiver registrations differ from the isolated SDK default layout')
    if before.get('candidate_control_processes') or current.get('candidate_control_processes'):
        result.append('Other candidate controller processes are present')
    return result


def joint_limits(robot):
    config = robot.get_config().get('joint_limits', {})
    result = [numbers(config.get('joint'+str(index)), 2, 'SDK joint limits')
              for index in range(1, 8)]
    if any(lower >= upper for lower, upper in result):
        raise ValueError('Invalid SDK joint range')
    return result


def validate_target(values, limits):
    values = numbers(values, 7, 'seven joints')
    if any(not lower <= value <= upper for value, (lower, upper) in zip(values, limits)):
        raise ValueError('Seven-axis target exceeds SDK soft limits')
    return values


def fresh_feedback(session, *, previous=None, monotonic=time.monotonic,
                   wallclock=time.time, sleep=time.sleep, _refresh_remaining=2):
    snapshot = session.snapshot()
    q = numbers(snapshot['q_rad'], 7, 'joint feedback')
    fk = numbers(snapshot['fk_flange_pose_m_rad'], 6, 'flange FK')
    if max(snapshot['packet_ages_s'].values()) > .1:
        raise RuntimeError('Joint feedback exceeds 100 ms freshness limit')
    deadline = monotonic()+2
    while monotonic() < deadline:
        message = session.robot.get_arm_status()
        stamp = getattr(message, 'timestamp', None)
        if (stamp is not None and isinstance(stamp, (int, float)) and math.isfinite(stamp)
                and 0 <= wallclock()-stamp <= .25
                and (previous is None or stamp > previous['status_timestamp_epoch_s'])):
            break
        sleep(.01)
    else:
        raise TimeoutError('No fresh arm status feedback')
    status = message.msg
    enable_stamps = []
    deadline = monotonic()+2
    missing = []
    while monotonic() < deadline:
        enable_stamps, missing = [], []
        for index in range(1, 8):
            driver = session.robot.get_driver_states(joint_index=index)
            stamp = getattr(driver, 'timestamp', None)
            if (stamp is None or not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
                    or not 0 <= wallclock()-stamp <= .25):
                missing.append(index)
            enable_stamps.append(stamp)
        if not missing:
            break
        sleep(.01)
    else:
        raise RuntimeError('No fresh joint enable feedback for axes '+str(missing))
    enabled = session.robot.get_joints_enable_status_list()
    if not isinstance(enabled, (list, tuple)) or len(enabled) != 7:
        raise RuntimeError('Seven joint enable states unavailable')
    # Waiting for the first enable packets must not publish an old joint/FK
    # snapshot. Refresh all four joint packets, then recheck all timestamps.
    if (max(wallclock()-stamp for stamp in snapshot['packet_timestamps_after_epoch_s'].values()) > .1
            or wallclock()-message.timestamp > .25):
        if _refresh_remaining <= 0:
            raise RuntimeError('Could not obtain simultaneous fresh joint/status/enable feedback')
        return fresh_feedback(session, previous=previous, monotonic=monotonic,
                              wallclock=wallclock, sleep=sleep,
                              _refresh_remaining=_refresh_remaining-1)
    snapshot.update(q_rad=q, fk_flange_pose_m_rad=fk, tx_attempts=session.guard.report()['tx_attempts'],
                    actual_tx_count=session.guard.report()['actual_tx_count'],
                    execution_enabled=session.guard.allowed, read_only=not session.guard.allowed)
    return {'sdk_snapshot': snapshot, 'q_rad': q, 'q_deg': [math.degrees(v) for v in q],
            'fk_flange_pose_m_rad': fk, 'status_timestamp_epoch_s': message.timestamp,
            'status': {'arm_status': int(status.arm_status), 'ctrl_mode': int(status.ctrl_mode),
                       'motion_status': int(status.motion_status), 'repr': str(status)},
            'enabled': list(enabled), 'enable_feedback_timestamps_epoch_s': enable_stamps,
            'observed_monotonic_s': monotonic(), 'observed_epoch_s': wallclock()}


def ready_blockers(feedback, *, take_can_control=False):
    state = feedback['status']
    result = []
    if state['arm_status'] != 0:
        result.append('Arm is not NORMAL')
    if state['motion_status'] != 0:
        result.append('Arm has not reached a stationary target')
    if feedback['enabled'] != [True]*7:
        result.append('All seven joints must already be enabled')
    if state['ctrl_mode'] != 1 and not (state['ctrl_mode'] == 3 and take_can_control):
        result.append('CAN mode is required; WEB mode requires explicit --take-can-control')
    return result


def stopped_window(session, *, monotonic=time.monotonic, wallclock=time.time,
                   sleep=time.sleep):
    feedback = []
    for index in range(10):
        if index:
            sleep(.05)
        feedback.append(fresh_feedback(session, previous=feedback[-1] if feedback else None,
                                       monotonic=monotonic, wallclock=wallclock, sleep=sleep))
    spans = [max(row['q_rad'][axis] for row in feedback)-min(row['q_rad'][axis] for row in feedback)
             for axis in range(7)]
    if max(spans) > TOLERANCE_RAD:
        raise RuntimeError('Arm changed by more than .05 degree during stopped window')
    if len({row['status']['ctrl_mode'] for row in feedback}) != 1:
        raise RuntimeError('Control mode changed during stopped window')
    return feedback, {'fresh_samples': 10, 'joint_spans_deg': [math.degrees(v) for v in spans],
                      'duration_s': feedback[-1]['observed_monotonic_s']-
                                    feedback[0]['sdk_snapshot']['request_start_monotonic_s']}


def strict_target(feedback, current, target):
    q = feedback['q_rad']
    return (not ready_blockers(feedback)
            and abs(q[3]-target[3]) <= TOLERANCE_RAD
            and q[3]-current[3] >= math.radians(.1)
            and all(abs(q[axis]-current[axis]) <= TOLERANCE_RAD for axis in range(7) if axis != 3))


def wait_target(session, current, target, *, timeout_s, previous=None, monotonic=time.monotonic,
                wallclock=time.time, sleep=time.sleep):
    deadline, stable, samples = monotonic()+timeout_s, 0, []
    while monotonic() < deadline:
        feedback = fresh_feedback(session, previous=previous, monotonic=monotonic,
                                  wallclock=wallclock, sleep=sleep)
        samples.append(feedback)
        previous = feedback
        if (feedback['status']['arm_status'] != 0 or feedback['status']['ctrl_mode'] != 1
                or feedback['enabled'] != [True]*7):
            raise RuntimeError('Arm readiness/control changed during micro-move')
        stable = stable+1 if strict_target(feedback, current, target) else 0
        if stable >= 10:
            return {'fresh_stable_samples': stable, 'samples': samples,
                    'joint_target_tolerance_deg': .05, 'minimum_applied_j4_change_deg': .1}
        sleep(.05)
    raise TimeoutError('Fixed J4 probe did not satisfy strict target feedback')


def take_can_control(session, baseline, *, timeout_s, monotonic=time.monotonic,
                     wallclock=time.time, sleep=time.sleep):
    session.robot.set_motion_mode('j')
    deadline, previous, samples = monotonic()+timeout_s, baseline, []
    while monotonic() < deadline:
        feedback = fresh_feedback(session, previous=previous, monotonic=monotonic,
                                  wallclock=wallclock, sleep=sleep)
        previous = feedback
        samples.append(feedback)
        if (feedback['status']['arm_status'] != 0 or feedback['enabled'] != [True]*7
                or max(abs(a-b) for a, b in zip(feedback['q_rad'], baseline['q_rad'])) > TOLERANCE_RAD):
            raise RuntimeError('CAN handoff changed readiness or joint position')
        if feedback['status']['ctrl_mode'] == 1 and feedback['status']['motion_status'] == 0:
            return {'mode_command_sent': True, 'joint_target_sent': False, 'samples': samples}
        sleep(.05)
    raise TimeoutError('Fresh CAN handoff was not confirmed')


def fresh_hold(session, limits, *, monotonic=time.monotonic, wallclock=time.time,
               sleep=time.sleep):
    """Hold only fresh valid feedback; do not disable limits or hide failures."""
    feedback = fresh_feedback(session, monotonic=monotonic, wallclock=wallclock, sleep=sleep)
    state = feedback['status']
    if state['arm_status'] != 0 or state['ctrl_mode'] != 1 or feedback['enabled'] != [True]*7:
        raise RuntimeError('Cannot hold without fresh NORMAL/CAN/7-enabled feedback')
    target = validate_target(feedback['q_rad'], limits)
    session.robot.set_joint_limits_enabled(True)
    session.robot.move_j(target)
    return {'requested': True, 'feedback': feedback, 'target_rad': target,
            'sdk_limits_enabled': session.robot.get_joint_limits_enabled(),
            'hold_verified': False}


def run_probe(bus_class, robot_factory, *, execute=False, take_control=False,
              channel='can0', timeout_s=5, evidence_provider=host_control_evidence,
              monotonic=time.monotonic, wallclock=time.time, sleep=time.sleep,
              session_factory=PassivePoseSession):
    """Caller must hold the control lock. Dependency injection is CPU-test only."""
    report = {'schema': SCHEMA, 'execute_selected': execute, 'take_can_control_selected': take_control,
              'fixed_command': {'joint': 4, 'delta_deg': .2, 'speed_percent': 1},
              'started_monotonic_s': monotonic(), 'started_epoch_s': wallclock(),
              'motion_target_valid': False, 'physical_palm_registered': False,
              'blockers': [], 'success': False, 'joint_motion_attempted': False}
    guard = AuditedSendGuard(bus_class, monotonic=monotonic, wallclock=wallclock)
    session = session_factory(bus_class, robot_factory, deadline_s=2,
                              monotonic=monotonic, wallclock=wallclock, sleep=sleep)
    session.guard = guard
    control_attempted, limits = False, None
    try:
        before = evidence_provider(channel)
        report['host_control_before_connect'] = before
        report['sdk_ready'] = session.start()  # guard precedes SDK factory and connect.
        feedback, stationary = stopped_window(session, monotonic=monotonic,
                                              wallclock=wallclock, sleep=sleep)
        report['baseline_feedback'], report['stationarity'] = feedback, stationary
        baseline = feedback[-1]
        limits = joint_limits(session.robot)
        current = validate_target(baseline['q_rad'], limits)
        target = list(current)
        target[3] += DELTA_RAD
        target = validate_target(target, limits)
        fk_target = numbers(session.robot.fk(target), 6, 'target flange FK')
        translation_m = math.dist(fk_target[:3], baseline['fk_flange_pose_m_rad'][:3])
        report['proposal'] = {'current_rad': current, 'target_rad': target,
                              'current_deg': [math.degrees(v) for v in current],
                              'target_deg': [math.degrees(v) for v in target],
                              'sdk_limits_rad': limits, 'fk_current_m_rad': baseline['fk_flange_pose_m_rad'],
                              'fk_target_m_rad': fk_target, 'fk_predicted_translation_m': translation_m,
                              'fk_prediction_is_model_not_visual_displacement': True}
        if translation_m > .005:
            report['blockers'].append('Predicted flange translation exceeds 5 mm')
        for row in feedback:
            report['blockers'].extend(ready_blockers(row, take_can_control=take_control))
        current_evidence = evidence_provider(channel)
        report['host_control_before_first_tx'] = current_evidence
        report['blockers'].extend(evidence_blockers(before, current_evidence))
        report['blockers'] = list(dict.fromkeys(report['blockers']))
        if not execute:
            report['event'] = 'visual_servo_inspected'
            report['success'] = True
        elif report['blockers']:
            report['event'] = 'visual_servo_blocked'
        else:
            guard.permit()  # Stable hook remains installed throughout mode/speed/move/hold.
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
            # Refresh the four packet/status baseline immediately before target TX.
            last_feedback = (report['can_handoff']['samples'][-1]
                             if 'can_handoff' in report else baseline)
            latest = fresh_feedback(session, previous=last_feedback, monotonic=monotonic,
                                    wallclock=wallclock, sleep=sleep)
            if (ready_blockers(latest) or max(abs(a-b) for a, b in zip(latest['q_rad'], current))
                    > TOLERANCE_RAD):
                raise RuntimeError('Baseline changed before joint target transmission')
            report['feedback_before_joint_tx'] = latest
            report['joint_motion_attempted'] = True
            report['q_before_rad'], report['q_target_rad'] = current, target
            report['move_j_sent_monotonic_s'] = monotonic()
            report['move_j_clock_definition'] = 'host_monotonic immediately before SDK call'
            session.robot.move_j(target)
            report['move_j_call_end_monotonic_s'] = monotonic()
            report['target_verification'] = wait_target(
                session, current, target, timeout_s=timeout_s, previous=latest, monotonic=monotonic,
                wallclock=wallclock, sleep=sleep)
            settled = report['target_verification']['samples'][-1]
            report['strict_settled_monotonic_s'] = settled['observed_monotonic_s']
            report['q_after_rad'] = settled['q_rad']
            report['j4_actual_delta_deg'] = math.degrees(settled['q_rad'][3]-current[3])
            report['sdk_limits_enabled_at_completion'] = session.robot.get_joint_limits_enabled()
            if report['sdk_limits_enabled_at_completion'] is not True:
                raise RuntimeError('SDK joint limits became disabled')
            report['event'], report['success'] = 'visual_servo_probe_reached', True
    except BaseException as exc:
        report['event'] = 'visual_servo_probe_failed'
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
            report['event'] = 'visual_servo_probe_failed'
        finally:
            if report.get('sdk_disconnected'):
                guard.restore()
        report['tx'] = guard.report()
        report['ended_monotonic_s'], report['ended_epoch_s'] = monotonic(), wallclock()
        report['hardware_synchronized'] = False
    return report


@contextmanager
def control_lock(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--inspect', action='store_true')
    action.add_argument('--execute', action='store_true')
    parser.add_argument('--take-can-control', action='store_true')
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--timeout-s', type=float, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if (not re.fullmatch('[A-Za-z0-9_-]+', args.channel)
            or not math.isfinite(args.timeout_s) or not 1 <= args.timeout_s <= 10):
        parser.error('Require a valid interface name and timeout between 1 and 10 seconds')
    lock_path = Path('/tmp')/('nero_'+args.channel+'_control.lock')
    try:
        with control_lock(lock_path), redirect_stdout(sys.stderr):
            bus_class, factory = load_sdk_runtime(args.channel)
            report = run_probe(bus_class, factory, execute=args.execute,
                               take_control=args.take_can_control, channel=args.channel,
                               timeout_s=args.timeout_s)
    except BaseException as exc:
        report = {'schema': SCHEMA, 'event': 'visual_servo_startup_failed', 'success': False,
                  'error': type(exc).__name__+': '+str(exc)}
    report['control_lock_path'] = str(lock_path)
    report['tool_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        temporary = args.output.with_name(args.output.name+'.'+str(os.getpid())+'.tmp')
        temporary.write_text(output)
        temporary.replace(args.output)
    sys.stdout.write(output)
    return 0 if report['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
