"""Independent passive NERO SDK JSONL process, with CAN transmit refused.

Importing this module uses only the standard library and opens no devices.
Only explicit main() startup constructs/connects the SDK robot. No effector,
request, mode, enable, or motion API is called. SDK output goes to stderr;
stdout is reserved for ready/snapshot/error/closed protocol events.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import select
import signal
import sys
import threading
import time


SCHEMA = 'passive_nero_pose_bridge_v1'
PACKETS = ('joint_12', 'joint_34', 'joint_56', 'joint_7')


class PassiveTransmitForbidden(RuntimeError):
    pass


class TerminationRequested(RuntimeError):
    pass


class TxGuard:
    """Install before SDK factory/connect; never invoke the original send."""

    def __init__(self, bus_class):
        self.bus_class = bus_class
        self.original_send = None
        self._attempts = 0
        self._lock = threading.Lock()
        self.installed = False

    @property
    def attempts(self):
        with self._lock:
            return self._attempts

    def install(self):
        if self.installed:
            return
        self.original_send = self.bus_class.send

        def refuse_send(bus, *args, **kwargs):
            with self._lock:
                self._attempts += 1
            raise PassiveTransmitForbidden('Passive pose bridge forbids CAN transmit')

        self.bus_class.send = refuse_send
        self.installed = True

    def check(self):
        if self.attempts:
            raise PassiveTransmitForbidden(f'CAN transmit attempted {self.attempts} time(s)')

    def restore(self):
        if self.installed:
            self.bus_class.send = self.original_send
            self.installed = False


def _finite_number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a pose or timestamp value')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite pose or timestamp value')
    return number


def _finite_vector(values, length, label):
    try:
        result = [_finite_number(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError('Invalid '+label) from exc
    if len(result) != length:
        raise RuntimeError('Invalid '+label)
    return result


def _packet_timestamps(robot):
    """Copy scalar stamps; reread latest parser references after joint copying."""
    parser = getattr(robot, '_parser', None)
    result = {}
    for name in PACKETS:
        message = getattr(parser, name, None)
        try:
            timestamp = _finite_number(getattr(message, 'timestamp', None))
        except (TypeError, ValueError, OverflowError):
            return None
        if timestamp <= 0:
            return None
        result[name] = timestamp
    return result


def source_runtime_info(robot, bus_class):
    """Record actual runtime/source bytes rather than a hardcoded SDK version."""
    paths = {str(Path(__file__).resolve())}
    context = getattr(robot, '_ctx', None)
    objects = (robot, getattr(robot, '_parser', None), context, getattr(context, 'comm', None))
    classes = [bus_class]
    for obj in objects:
        if obj is not None:
            classes.extend(type(obj).__mro__)
    for cls in classes:
        try:
            path = inspect.getsourcefile(cls)
        except (TypeError, OSError):
            path = None
        if path and Path(path).is_file():
            paths.add(str(Path(path).resolve()))
    for name in ('pyAgxArm.api.agx_arm_factory', 'pyAgxArm.utiles.mdh_kinematics'):
        module = sys.modules.get(name)
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file():
            paths.add(str(Path(path).resolve()))
    return {'python_executable': sys.executable, 'python_version': sys.version,
            'robot_class': type(robot).__module__+'.'+type(robot).__qualname__,
            'bus_class': bus_class.__module__+'.'+bus_class.__qualname__,
            'source_sha256': {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                              for path in sorted(paths)},
            'io_scope': 'connect/get_joint_angles/fk/disconnect; no effector or requests'}


def load_sdk_runtime(channel):
    """Load python-can only; defer all SDK imports until after guard install."""
    from can.interfaces.socketcan.socketcan import SocketcanBus

    def factory():
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
        return AgxArmFactory.create_arm(create_agx_arm_config(
            robot=ArmModel.NERO, firmeware_version=NeroFW.V120,
            interface='socketcan', channel=channel))

    return SocketcanBus, factory


class PassivePoseSession:
    """Testable passive lifecycle and bounded four-packet snapshot reader."""

    def __init__(self, bus_class, robot_factory, *, deadline_s=2., max_age_s=.25,
                 max_span_s=.02, monotonic=time.monotonic, wallclock=time.time,
                 sleep=time.sleep):
        deadline_s = _finite_number(deadline_s)
        if not 1 <= deadline_s <= 5:
            raise ValueError('deadline_s must be between 1 and 5 seconds')
        max_age_s = _finite_number(max_age_s)
        max_span_s = _finite_number(max_span_s)
        if not .05 <= max_age_s <= 5:
            raise ValueError('max_age_s must be between 0.05 and 5 seconds')
        if not .01 <= max_span_s <= 1:
            raise ValueError('max_span_s must be between 0.01 and 1 seconds')
        self.guard = TxGuard(bus_class)
        self.robot_factory = robot_factory
        self.deadline_s = deadline_s
        self.max_age_s = max_age_s
        self.max_span_s = max_span_s
        self.monotonic = monotonic
        self.wallclock = wallclock
        self.sleep = sleep
        self.robot = None
        self.started = False
        self.closed = False
        self.snapshot_count = 0
        self.previous_packets = None
        self.previous_request_end = None
        self.last_clock = None

    def _now(self):
        value = _finite_number(self.monotonic())
        if self.last_clock is not None and value < self.last_clock:
            raise RuntimeError('Monotonic clock moved backward')
        self.last_clock = value
        return value

    def stats(self):
        return {'tx_attempts': self.guard.attempts, 'actual_tx_count': 0,
                'snapshot_count': self.snapshot_count,
                'tx_guard_active': self.guard.installed, 'read_only': True,
                'motion_target_valid': False, 'execution_enabled': False}

    def start(self):
        if self.started or self.closed:
            raise RuntimeError('Passive session cannot be restarted')
        self.guard.install()
        self.guard.check()
        self.robot = self.robot_factory()
        self.guard.check()
        self.robot.connect()
        self.guard.check()
        self.previous_packets = _packet_timestamps(self.robot)
        self.started = True
        return {'schema': SCHEMA, 'event': 'ready', 'request_id': None,
                'deadline_s': self.deadline_s, 'freshness_limit_s': self.max_age_s,
                'packet_span_limit_s': self.max_span_s, 'tx_guard_installed_before_factory': True,
                'source_runtime_info': source_runtime_info(self.robot, self.guard.bus_class),
                **self.stats()}

    def snapshot(self, request_id=None):
        if not self.started or self.closed:
            raise RuntimeError('Passive session is not connected')
        self.guard.check()
        start = self._now()
        if self.previous_request_end is not None and start <= self.previous_request_end:
            raise RuntimeError('Snapshot request monotonic timestamps did not advance')
        deadline = start+self.deadline_s
        last_rejection = 'joint packets unavailable'
        while self._now() < deadline:
            self.guard.check()
            copy_start = self._now()
            before = _packet_timestamps(self.robot)
            if before is not None:
                message = self.robot.get_joint_angles()
                if message is not None:
                    q = _finite_vector(message.msg, 7, 'seven-axis joint feedback')
                    after = _packet_timestamps(self.robot)
                    copy_end = self._now()
                    if before != after:
                        last_rejection = 'packet timestamps changed during copying'
                    else:
                        wall = _finite_number(self.wallclock())
                        ages = {name: wall-before[name] for name in PACKETS}
                        span = max(before.values())-min(before.values())
                        if not all(0 <= age <= self.max_age_s for age in ages.values()):
                            last_rejection = 'joint packets are stale or from the future'
                        elif span > self.max_span_s:
                            last_rejection = 'joint packet span exceeds %.0f ms' % (
                                self.max_span_s * 1000,)
                        elif self.previous_packets is not None and not all(
                                before[name] > self.previous_packets[name] for name in PACKETS):
                            last_rejection = 'not all four joint packets advanced'
                        else:
                            fk = _finite_vector(self.robot.fk(q.copy()), 6, 'flange FK m_rad')
                            self.guard.check()
                            end = self._now()
                            if not start <= copy_start <= copy_end <= end or end <= start:
                                raise RuntimeError('Invalid snapshot monotonic time bracket')
                            if end >= deadline:
                                raise TimeoutError('Snapshot computation exceeded deadline')
                            self.previous_packets = before.copy()
                            self.previous_request_end = end
                            index = self.snapshot_count
                            self.snapshot_count += 1
                            return {'schema': SCHEMA, 'event': 'snapshot',
                                    'request_id': request_id, 'snapshot_index': index,
                                    'q_rad': q, 'fk_flange_pose_m_rad': fk,
                                    'packet_timestamps_before_epoch_s': before,
                                    'packet_timestamps_after_epoch_s': after,
                                    'packet_ages_s': ages, 'packet_span_s': span,
                                    'copy_wallclock_epoch_s': wall,
                                    'request_start_monotonic_s': start,
                                    'request_end_monotonic_s': end,
                                    'copy_start_monotonic_s': copy_start,
                                    'copy_end_monotonic_s': copy_end,
                                    **self.stats()}
                else:
                    last_rejection = 'seven-axis joint feedback unavailable'
            self.guard.check()
            remaining = deadline-self._now()
            if remaining > 0:
                self.sleep(min(.01, remaining))
        raise TimeoutError('No fresh stable four-packet snapshot: '+last_rejection)

    def close(self):
        if self.closed:
            self.guard.check()
            return
        self.closed = True
        try:
            if self.robot is not None:
                self.robot.disconnect()
                # Release this process's SDK reference while send is still
                # guarded and SDK stdout still points at stderr.
                self.robot = None
        except BaseException:
            # A failed disconnect may leave reader threads alive. Keep send
            # refused until this isolated process exits.
            raise
        else:
            self.guard.restore()
        self.guard.check()


def _requests(stream, guard):
    """Wait for input while checking background transmit attempts, without spin."""
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError, ValueError):
        for line in stream:
            guard.check()
            yield line
        return
    buffer = b''
    while True:
        guard.check()
        readable, _, _ = select.select([descriptor], [], [], .05)
        if not readable:
            continue
        chunk = os.read(descriptor, 4096)
        if not chunk:
            if buffer:
                yield buffer.decode('utf-8')
            return
        buffer += chunk
        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            yield line.decode('utf-8')


def _emit(stream, value):
    stream.write(json.dumps(value, allow_nan=False)+'\n')
    stream.flush()


def _terminate(signum, frame):
    raise TerminationRequested('Signal '+str(signum)+' requested shutdown')


def main(argv=None, *, runtime_loader=load_sdk_runtime, stdin=None, stdout=None,
         session_factory=PassivePoseSession):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--deadline-s', type=float, default=2.)
    args = parser.parse_args(argv)
    if not args.channel.strip() or not 1 <= args.deadline_s <= 5:
        parser.error('channel must be nonempty and deadline-s must be between 1 and 5')
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    session = None
    request_id = None
    errors = []
    previous_handlers = {}
    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[number] = signal.signal(number, _terminate)
        with redirect_stdout(sys.stderr):
            bus_class, factory = runtime_loader(args.channel)
            session = session_factory(bus_class, factory, deadline_s=args.deadline_s)
            try:
                _emit(output_stream, session.start())
                for line in _requests(input_stream, session.guard):
                    request_id = None
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError('Expected a JSON request object')
                    request_id = request.get('request_id')
                    if request_id is not None and (isinstance(request_id, bool)
                                                   or not isinstance(request_id, (str, int))):
                        raise ValueError('request_id must be a string, integer or null')
                    if request.get('op') == 'snapshot':
                        _emit(output_stream, session.snapshot(request_id))
                    elif request.get('op') == 'close':
                        break
                    else:
                        raise ValueError('Unsupported request op')
                else:
                    request_id = None
            except BaseException as exc:
                errors.append(type(exc).__name__+': '+str(exc))
            finally:
                # Do not let another termination signal interrupt disconnect.
                for number in previous_handlers:
                    signal.signal(number, signal.SIG_IGN)
                try:
                    session.close()
                except BaseException as exc:
                    errors.append(type(exc).__name__+': '+str(exc))
    except BaseException as exc:
        errors.append(type(exc).__name__+': '+str(exc))
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)
        stats = (session.stats() if session is not None else
                 {'tx_attempts': 0, 'actual_tx_count': 0, 'snapshot_count': 0,
                  'tx_guard_active': False, 'read_only': True,
                  'motion_target_valid': False, 'execution_enabled': False})
        if errors:
            _emit(output_stream, {'schema': SCHEMA, 'event': 'error',
                                  'request_id': request_id, 'errors': errors, **stats})
        _emit(output_stream, {'schema': SCHEMA, 'event': 'closed',
                              'request_id': request_id, 'success': not errors, **stats})
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
