"""Short, zero-transmit Revo2 broadcast diagnosis; no hand-control commands.

The SDK's Revo2 getters read caches and have no request frame. A missing 0x1C1
broadcast therefore remains a blocker. Six channels represent five fingers
with two thumb motors. This module uses only stdlib on import and opens no bus
until main() is explicitly invoked in the SDK runtime.
"""
import argparse
from collections import deque
from contextlib import redirect_stdout
import hashlib
import inspect
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

from nero_revo2_control.bridges.passive_pose_bridge import TxGuard, load_sdk_runtime, source_runtime_info


FINGERS = ('thumb_tip', 'thumb_base', 'index_finger', 'middle_finger', 'ring_finger', 'pinky_finger')
PACKETS = {
    'position': (0x1C1, 'finger_pos', 'get_finger_pos'),
    'speed': (0x1C2, 'finger_spd', 'get_finger_spd'),
    'current': (0x1C3, 'finger_current', 'get_finger_current'),
    'status': (0x1C0, 'hand_status', 'get_hand_status')}
CONTROL_IDS = (0x1B1, 0x1B2, 0x1B3, 0x1B5)


def _timestamp(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


class BroadcastCache:
    """Immutable raw copies independently verify the SDK's mutable cache."""

    def __init__(self, monotonic=time.monotonic):
        self.monotonic = monotonic
        self.lock = threading.Lock()
        self.packets = {key: deque(maxlen=64) for key in PACKETS}
        self.counts = {key: 0 for key in PACKETS}
        self.invalid_frames = 0
        self.controls = {hex(key): {'count': 0, 'local_count': 0, 'remote_count': 0,
                                    'last_timestamp_epoch_s': None, 'last_payload_bytes': None}
                         for key in CONTROL_IDS}

    def receive(self, frame):
        can_id = getattr(frame, 'arbitration_id', None)
        if (can_id not in CONTROL_IDS and can_id not in [value[0] for value in PACKETS.values()]):
            return
        try:
            timestamp = _timestamp(getattr(frame, 'timestamp', None))
            data = bytes(frame.data)
            remote = getattr(frame, 'is_rx', None) is True
            if (timestamp is None or len(data) != 8 or getattr(frame, 'is_error_frame', False)
                    or getattr(frame, 'is_remote_frame', False)
                    or getattr(frame, 'is_extended_id', False)):
                raise ValueError('Malformed basic Revo2 frame')
            with self.lock:
                if can_id in CONTROL_IDS:
                    row = self.controls[hex(can_id)]
                    row['count'] += 1
                    row['remote_count' if remote else 'local_count'] += 1
                    row.update(last_timestamp_epoch_s=timestamp, last_payload_bytes=list(data))
                    return
                kind = next(key for key, spec in PACKETS.items() if spec[0] == can_id)
                values = dict(zip(FINGERS, data[2:8]))
                if kind in ('speed', 'current'):
                    values = {key: value-256 if value >= 128 else value for key, value in values.items()}
                self.packets[kind].append({
                    'timestamp_epoch_s': timestamp, 'callback_monotonic_s': self.monotonic(),
                    'is_remote_rx': remote, 'values': values,
                    'left_or_right': data[0] if kind == 'status' else None})
                self.counts[kind] += 1
        except (TypeError, ValueError, OverflowError, AttributeError):
            with self.lock:
                self.invalid_frames += 1

    def match(self, kind, timestamp, values, side=None):
        with self.lock:
            for row in reversed(self.packets[kind]):
                if (row['timestamp_epoch_s'] == timestamp and row['values'] == values
                        and (kind != 'status' or row['left_or_right'] == side)):
                    return {'timestamp_epoch_s': row['timestamp_epoch_s'],
                            'callback_monotonic_s': row['callback_monotonic_s'],
                            'is_remote_rx': row['is_remote_rx']}
        return None

    def stats(self):
        with self.lock:
            return {'received_frame_counts': self.counts.copy(), 'invalid_frames': self.invalid_frames,
                    'observed_control_frames': {key: dict(value) for key, value in self.controls.items()}}


def _copy_getter(hand, kind, cache, wallclock, max_age_s=.25):
    can_id, attribute, getter = PACKETS[kind]
    row = {'can_id': hex(can_id), 'available': False, 'fresh': False,
           'timestamp_before_epoch_s': None, 'timestamp_getter_epoch_s': None,
           'timestamp_after_epoch_s': None, 'age_s': None, 'values': None,
           'left_or_right': None, 'raw_broadcast': None, 'reason': 'packet_missing'}
    parser = getattr(hand, '_parser', None)
    before = _timestamp(getattr(getattr(parser, attribute, None), 'timestamp', None))
    message = getattr(hand, getter)()
    row['timestamp_before_epoch_s'] = before
    if message is None:
        return row
    row['available'] = True
    stamp = _timestamp(getattr(message, 'timestamp', None))
    row['timestamp_getter_epoch_s'] = stamp
    try:
        values = {name: getattr(message.msg, name) for name in FINGERS}
        if not all(type(value) is int for value in values.values()):
            raise ValueError('Noninteger finger value')
        low, high = (-128, 127) if kind in ('speed', 'current') else (0, 255)
        if not all(low <= value <= high for value in values.values()):
            raise ValueError('Finger value outside raw wire range')
        side = getattr(message.msg, 'left_or_right', None) if kind == 'status' else None
        if kind == 'status' and (type(side) is not int or not 0 <= side <= 255):
            raise ValueError('Invalid hand-side byte')
        row['values'], row['left_or_right'] = values, side
    except (AttributeError, TypeError, ValueError) as exc:
        row['reason'] = 'invalid_payload: '+str(exc)
        return row
    after = _timestamp(getattr(getattr(parser, attribute, None), 'timestamp', None))
    row['timestamp_after_epoch_s'] = after
    if stamp is None or before != stamp or after != stamp:
        row['reason'] = 'timestamp_changed_during_copy'
        return row
    age = wallclock()-stamp
    row['age_s'] = age
    raw = cache.match(kind, stamp, values, side)
    row['raw_broadcast'] = raw
    if raw is None:
        row['reason'] = 'no_matching_raw_broadcast_in_this_process'
    elif not raw['is_remote_rx']:
        row['reason'] = 'local_injected_frame_not_hardware_rx'
    elif not math.isfinite(age) or not 0 <= age <= max_age_s:
        row['reason'] = 'packet_stale_or_future'
    else:
        row['fresh'], row['reason'] = True, 'ok'
    return row


def _extra_sources(hand):
    paths = {str(Path(__file__).resolve())}
    for obj in (hand, getattr(hand, '_parser', None), getattr(hand, '_effector_ctx', None)):
        if obj is not None:
            for cls in type(obj).__mro__:
                try:
                    path = inspect.getsourcefile(cls)
                except (TypeError, OSError):
                    path = None
                if path and Path(path).is_file():
                    paths.add(str(Path(path).resolve()))
    return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sorted(paths)}


def run_probe(bus_class, robot_factory, *, seconds=2., monotonic=time.monotonic,
              wallclock=time.time, sleep=time.sleep):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds <= 3:
        raise ValueError('Probe duration must be greater than zero and at most 3 seconds')
    guard = TxGuard(bus_class)
    cache = BroadcastCache(monotonic)
    robot, hand = None, None
    accepted = []
    latest = {}
    regression = False
    report = {
        'schema': 1, 'kind': 'passive_revo2_finger_feedback_probe', 'valid': False,
        'blocker': None, 'requested_duration_s': seconds, 'freshness_limit_s': .25,
        'minimum_new_position_frames': 3, 'finger_names': list(FINGERS),
        'effector': 'revo2', 'read_only': True, 'actual_tx_count': 0, 'tx_attempts': 0,
        'grasp_ready': False, 'motion_target_valid': False, 'execution_enabled': False,
        'cleanup': {'disconnect_attempted': False, 'disconnect_succeeded': False,
                    'tx_guard_active': False}, 'errors': [],
        'source_units': {
            'position': 'raw unsigned bytes 0..255; rig display round(raw*100/255), not joint radians',
            'speed': 'raw signed bytes -128..127; physical speed scale not verified',
            'current': 'raw signed bytes -128..127; ampere/force scale not verified',
            'status': 'raw motor bytes; 0 idle, 1 running, 2 stalled; side 1 left/2 right',
            'epoch': 'python-can SocketCAN kernel RX wall-clock timestamp; not measurement timestamp'},
        'query_frame_available': False, 'getter_path_sends_frames': False}
    guard.install()
    try:
        guard.check()
        robot = robot_factory()
        guard.check()
        hand = robot.init_effector('revo2')
        guard.check()
        robot.get_context().register_parser_packet_fun(cache.receive)
        robot.connect()
        guard.check()
        report['source_runtime_info'] = source_runtime_info(robot, bus_class)
        report['effector_source_sha256'] = _extra_sources(hand)
        start = monotonic()
        end = start+seconds
        report['capture_start_monotonic_s'] = start
        last_clock = start
        while True:
            now = monotonic()
            if not math.isfinite(now) or now < last_clock:
                raise RuntimeError('Monotonic diagnostic clock moved backward')
            last_clock = now
            guard.check()
            latest = {kind: _copy_getter(hand, kind, cache, wallclock) for kind in PACKETS}
            position = latest['position']
            if position['fresh']:
                stamp = position['timestamp_getter_epoch_s']
                if accepted and stamp < accepted[-1]['timestamp_epoch_s']:
                    regression = True
                elif not accepted or stamp > accepted[-1]['timestamp_epoch_s']:
                    raw = position['values'].copy()
                    accepted.append({'timestamp_epoch_s': stamp, 'age_s_at_copy': position['age_s'],
                                     'raw_0_255': raw,
                                     'position_0_100': {key: round(value*100/255) for key, value in raw.items()}})
            guard.check()
            if now >= end:
                break
            sleep(min(.02, end-now))
        report['capture_end_monotonic_s'] = monotonic()
        report['position_timestamps_strictly_forward'] = not regression
        report['valid'] = len(accepted) >= 3 and latest['position']['fresh'] and not regression
        if not report['valid']:
            if cache.stats()['received_frame_counts']['position'] == 0:
                report['blocker'] = 'Missing 0x1C1 position broadcast; no read-only SDK query frame exists'
            else:
                report['blocker'] = ('Fresh six-channel 0x1C1 position requirement not met: '
                                     +latest['position']['reason'])
        report['latest_getters'] = latest
        report['accepted_position_frames'] = accepted
        report['position_frame_count'] = len(accepted)
        report['position_span_raw_0_255'] = {
            name: max(sample['raw_0_255'][name] for sample in accepted)
            -min(sample['raw_0_255'][name] for sample in accepted) for name in FINGERS} if accepted else None
        report['hand_side'] = (latest['status']['left_or_right']
                               if latest['status']['fresh'] else None)
    except BaseException as exc:
        report.update(valid=False, blocker=type(exc).__name__+': '+str(exc))
        report['errors'].append(type(exc).__name__+': '+str(exc))
    finally:
        try:
            if robot is not None:
                report['cleanup']['disconnect_attempted'] = True
                robot.disconnect()
                report['cleanup']['disconnect_succeeded'] = True
                robot = hand = None
            guard.restore()
            guard.check()
        except BaseException as exc:
            report.update(valid=False, blocker=type(exc).__name__+': '+str(exc))
            report['errors'].append(type(exc).__name__+': '+str(exc))
        report.update(cache.stats())
        report['tx_attempts'] = guard.attempts
        report['cleanup']['tx_guard_active'] = guard.installed
    return report


def _terminate(signum, frame):
    raise RuntimeError('Signal '+str(signum)+' interrupted finger feedback probe')


def main(argv=None, *, runtime_loader=load_sdk_runtime, probe=run_probe):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--seconds', type=float, default=2.)
    args = parser.parse_args(argv)
    if not args.channel.strip() or not 0 < args.seconds <= 3:
        parser.error('channel must be nonempty; seconds must be greater than zero and at most 3')
    output = sys.stdout
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        for number in handlers:
            signal.signal(number, _terminate)
        with redirect_stdout(sys.stderr):
            bus, factory = runtime_loader(args.channel)
            report = probe(bus, factory, seconds=args.seconds)
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
    output.write(json.dumps(report, allow_nan=False)+'\n')
    output.flush()
    return 0 if report['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
