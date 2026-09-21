import importlib
import io
import json
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import passive_pose_bridge as bridge


class FakeBus:
    actual_sends = 0

    def send(self, *args, **kwargs):
        type(self).actual_sends += 1


class Clock:
    def __init__(self):
        self.value = 0.
        self.sleep_calls = []
        self.on_sleep = None

    def monotonic(self):
        self.value += .0001
        return self.value

    def wallclock(self):
        return 100.+self.value

    def sleep(self, duration):
        self.sleep_calls.append(duration)
        self.value += duration
        if self.on_sleep is not None:
            self.on_sleep()


class Robot:
    def __init__(self):
        self._parser = SimpleNamespace()
        self.set_packets(99.5)
        self.q = [.01*i for i in range(7)]
        self.fk_result = [.1, .2, .3, .4, .5, .6]
        self.connected = 0
        self.disconnected = 0
        self.getter_calls = 0
        self.before_get = None
        self.on_connect = None
        self.on_disconnect = None

    def set_packets(self, timestamp):
        for i, name in enumerate(bridge.PACKETS):
            setattr(self._parser, name, SimpleNamespace(timestamp=timestamp+i*.0001))

    def connect(self):
        self.connected += 1
        if self.on_connect:
            self.on_connect()
        self.set_packets(99.9)

    def disconnect(self):
        self.disconnected += 1
        if self.on_disconnect:
            self.on_disconnect()

    def get_joint_angles(self):
        self.getter_calls += 1
        if self.before_get:
            self.before_get()
        return SimpleNamespace(msg=self.q)

    def fk(self, q):
        self.last_fk_input = q
        return self.fk_result


class PassivePoseBridgeTests(unittest.TestCase):
    def setUp(self):
        FakeBus.actual_sends = 0
        self.original_send = FakeBus.send
        self.clock = Clock()
        self.robot = Robot()
        self.sessions = []

    def tearDown(self):
        for session in self.sessions:
            try:
                session.close()
            except Exception:
                pass
        FakeBus.send = self.original_send

    def session(self, factory=None, **options):
        session = bridge.PassivePoseSession(
            FakeBus, factory or (lambda: self.robot), deadline_s=1.,
            monotonic=self.clock.monotonic, wallclock=self.clock.wallclock,
            sleep=self.clock.sleep, **options)
        self.sessions.append(session)
        return session

    def started(self):
        session = self.session()
        session.start()
        self.robot.set_packets(self.clock.wallclock()-.01)
        return session

    def protocol(self, text, *, robot=None):
        if robot is not None:
            self.robot = robot
        self.clock.on_sleep = lambda: self.robot.set_packets(self.clock.wallclock()-.01)

        def runtime_loader(channel):
            self.assertEqual(channel, 'can0')
            print('sdk loader log')
            return FakeBus, lambda: self.robot

        def session_factory(bus_class, factory, deadline_s):
            session = bridge.PassivePoseSession(
                bus_class, factory, deadline_s=deadline_s,
                monotonic=self.clock.monotonic, wallclock=self.clock.wallclock,
                sleep=self.clock.sleep)
            self.sessions.append(session)
            return session

        output = io.StringIO()
        logs = io.StringIO()
        previous = signal.getsignal(signal.SIGTERM)
        with patch('sys.stderr', logs):
            status = bridge.main(['--channel', 'can0', '--deadline-s', '1'],
                                 runtime_loader=runtime_loader, session_factory=session_factory,
                                 stdin=io.StringIO(text), stdout=output)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertIn('sdk loader log', logs.getvalue())
        return status, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_import_has_no_sdk_or_device_side_effects(self):
        with patch('builtins.__import__', wraps=__import__) as imported:
            importlib.reload(bridge)
        self.assertFalse(any(call.args[0].startswith(('pyAgxArm', 'can.'))
                             for call in imported.call_args_list))
        self.assertIs(FakeBus.send, self.original_send)

    def test_guard_is_installed_before_factory_and_connect(self):
        order = []

        def factory():
            self.assertIsNot(FakeBus.send, self.original_send)
            order.append('factory')
            return self.robot

        def connect():
            self.assertIsNot(FakeBus.send, self.original_send)
            order.append('connect')

        self.robot.on_connect = connect
        session = self.session(factory)
        ready = session.start()
        self.assertEqual(order, ['factory', 'connect'])
        self.assertEqual(ready['event'], 'ready')
        self.assertTrue(ready['tx_guard_installed_before_factory'])
        self.assertEqual(ready['actual_tx_count'], 0)
        self.assertTrue(ready['source_runtime_info']['source_sha256'])
        session.close()
        self.assertIs(FakeBus.send, self.original_send)

    def test_snapshot_copies_q_stamps_fk_and_monotonic_bracket(self):
        session = self.started()
        first = session.snapshot('first')
        self.assertEqual(first['request_id'], 'first')
        self.assertEqual(first['q_rad'], self.robot.q)
        self.assertEqual(first['fk_flange_pose_m_rad'], self.robot.fk_result)
        self.assertIsNot(first['q_rad'], self.robot.q)
        self.assertIsNot(self.robot.last_fk_input, first['q_rad'])
        self.assertEqual(first['packet_timestamps_before_epoch_s'],
                         first['packet_timestamps_after_epoch_s'])
        self.assertTrue(all(0 <= age <= .25 for age in first['packet_ages_s'].values()))
        self.assertLessEqual(first['packet_span_s'], .02)
        self.assertLess(first['request_start_monotonic_s'], first['request_end_monotonic_s'])
        self.robot.q[0] = 9
        self.assertNotEqual(first['q_rad'][0], 9)
        self.clock.sleep(.02)
        self.robot.set_packets(self.clock.wallclock()-.01)
        second = session.snapshot(2)
        self.assertGreater(second['request_start_monotonic_s'], first['request_end_monotonic_s'])
        self.assertTrue(all(second['packet_timestamps_after_epoch_s'][name]
                            > first['packet_timestamps_after_epoch_s'][name]
                            for name in bridge.PACKETS))

    def test_latest_parser_timestamp_change_during_copy_is_retried(self):
        session = self.started()

        def change_once():
            if self.robot.getter_calls == 1:
                # Replace parser message references, rather than mutating the
                # old messages, to prove the after-copy read is fresh.
                self.robot.set_packets(self.clock.wallclock()-.005)

        self.robot.before_get = change_once
        self.clock.on_sleep = lambda: self.robot.set_packets(self.clock.wallclock()-.01)
        response = session.snapshot()
        self.assertGreaterEqual(self.robot.getter_calls, 2)
        self.assertGreaterEqual(len(self.clock.sleep_calls), 1)
        self.assertEqual(response['packet_timestamps_before_epoch_s'],
                         response['packet_timestamps_after_epoch_s'])

    def test_old_or_nonadvancing_packets_timeout_without_busy_wait(self):
        session = self.started()
        first = session.snapshot()
        with self.assertRaisesRegex(TimeoutError, 'No fresh stable four-packet snapshot'):
            session.snapshot()
        self.assertEqual(session.snapshot_count, 1)
        self.assertEqual(session.previous_packets, first['packet_timestamps_after_epoch_s'])
        self.assertTrue(self.clock.sleep_calls)
        self.assertTrue(all(0 < delay <= .01 for delay in self.clock.sleep_calls))
        self.assertLess(self.clock.value, 1.1)

    def test_one_packet_not_advanced_is_never_accepted(self):
        session = self.started()
        first = session.snapshot()
        old_seventh = self.robot._parser.joint_7.timestamp

        def update_three():
            self.robot.set_packets(self.clock.wallclock()-.01)
            self.robot._parser.joint_7.timestamp = old_seventh

        self.clock.on_sleep = update_three
        with self.assertRaises(TimeoutError):
            session.snapshot()
        self.assertEqual(session.previous_packets, first['packet_timestamps_after_epoch_s'])

    def test_packet_span_and_future_timestamp_are_rejected(self):
        session = self.started()
        self.robot._parser.joint_7.timestamp -= .1
        with self.assertRaises(TimeoutError):
            session.snapshot()
        self.robot.set_packets(self.clock.wallclock()+10.)
        with self.assertRaisesRegex(TimeoutError, 'stale or from the future'):
            session.snapshot()

    def test_missing_packet_is_not_replaced_by_sdk_joint_defaults(self):
        session = self.started()
        del self.robot._parser.joint_34
        with self.assertRaisesRegex(TimeoutError, 'joint packets unavailable'):
            session.snapshot()
        self.assertEqual(self.robot.getter_calls, 0)

    def test_bad_fk_structure_fails_instead_of_reusing_previous_snapshot(self):
        session = self.started()
        self.robot.fk_result = [1, 2, 3]
        with self.assertRaisesRegex(RuntimeError, 'Invalid flange FK m_rad'):
            session.snapshot()
        self.assertEqual(session.snapshot_count, 0)
        self.robot.fk_result = [.1, .2, .3, .4, float('nan'), .6]
        with self.assertRaisesRegex(RuntimeError, 'Invalid flange FK m_rad'):
            session.snapshot()

    def test_bad_seven_axis_joint_feedback_is_rejected(self):
        session = self.started()
        self.robot.q = [0]*6
        with self.assertRaisesRegex(RuntimeError, 'seven-axis joint feedback'):
            session.snapshot()

    def test_monotonic_regression_is_rejected(self):
        session = self.started()
        session.snapshot()
        self.clock.value = 0.
        with self.assertRaisesRegex(RuntimeError, 'Monotonic clock moved backward'):
            session.snapshot()

    def test_any_factory_connect_or_background_send_attempt_never_reaches_bus(self):
        session = self.session()
        self.robot.on_connect = lambda: FakeBus().send('forbidden')
        with self.assertRaises(bridge.PassiveTransmitForbidden):
            session.start()
        self.assertEqual(session.guard.attempts, 1)
        self.assertEqual(FakeBus.actual_sends, 0)
        with self.assertRaises(bridge.PassiveTransmitForbidden):
            session.close()
        self.assertEqual(self.robot.disconnected, 1)

    def test_guard_catches_attempt_even_when_sdk_swallows_exception(self):
        session = self.started()
        try:
            FakeBus().send('forbidden')
        except bridge.PassiveTransmitForbidden:
            pass
        with self.assertRaises(bridge.PassiveTransmitForbidden):
            session.snapshot()
        self.assertEqual(FakeBus.actual_sends, 0)

    def test_protocol_close_eof_ids_and_sdk_stdout_redirection(self):
        status, rows = self.protocol('{'+'"op":"snapshot","request_id":"a"}\n'
                                     '{"op":"close","request_id":"done"}\n')
        self.assertEqual(status, 0)
        self.assertEqual([row['event'] for row in rows], ['ready', 'snapshot', 'closed'])
        self.assertIsNone(rows[0]['request_id'])
        self.assertEqual(rows[1]['request_id'], 'a')
        self.assertEqual(rows[2]['request_id'], 'done')
        self.assertTrue(rows[2]['success'])
        self.assertEqual(self.robot.disconnected, 1)
        self.assertTrue(all(row['actual_tx_count'] == row['tx_attempts'] == 0 for row in rows))

    def test_eof_clears_last_snapshot_request_id(self):
        status, rows = self.protocol('{"op":"snapshot","request_id":3}\n')
        self.assertEqual(status, 0)
        self.assertEqual(rows[-1]['event'], 'closed')
        self.assertIsNone(rows[-1]['request_id'])

    def test_protocol_error_cleans_up_and_reports_nonzero(self):
        status, rows = self.protocol('{"op":"move","request_id":"bad"}\n')
        self.assertEqual(status, 2)
        self.assertEqual([row['event'] for row in rows], ['ready', 'error', 'closed'])
        self.assertEqual(rows[-1]['request_id'], 'bad')
        self.assertFalse(rows[-1]['success'])
        self.assertEqual(self.robot.disconnected, 1)

    def test_protocol_transmit_attempt_fails_entire_helper(self):
        self.robot.on_connect = lambda: FakeBus().send('forbidden')
        status, rows = self.protocol('')
        self.assertEqual(status, 2)
        self.assertEqual([row['event'] for row in rows], ['error', 'closed'])
        self.assertEqual(rows[-1]['tx_attempts'], 1)
        self.assertEqual(rows[-1]['actual_tx_count'], 0)
        self.assertEqual(FakeBus.actual_sends, 0)
        self.assertEqual(self.robot.disconnected, 1)

    def test_sigterm_and_failed_disconnect_keep_cleanup_explicit(self):
        def terminate():
            signal.raise_signal(signal.SIGTERM)

        self.robot.before_get = terminate
        status, rows = self.protocol('{"op":"snapshot"}\n')
        self.assertEqual(status, 2)
        self.assertEqual(rows[-1]['event'], 'closed')
        self.assertTrue(any('Signal '+str(int(signal.SIGTERM)) in error for error in rows[-2]['errors']))
        self.assertEqual(self.robot.disconnected, 1)
        self.robot = Robot()
        self.robot.on_disconnect = lambda: (_ for _ in ()).throw(RuntimeError('shutdown failed'))
        status, rows = self.protocol('{"op":"close"}\n')
        self.assertEqual(status, 2)
        self.assertTrue(rows[-1]['tx_guard_active'])
        self.assertTrue(any('shutdown failed' in error for error in rows[-2]['errors']))

    def test_deadline_bounds_and_lifecycle_reuse_are_rejected(self):
        for deadline in (0, .99, 5.01, float('nan')):
            with self.assertRaises(ValueError):
                bridge.PassivePoseSession(FakeBus, lambda: self.robot, deadline_s=deadline)
        session = self.started()
        session.close()
        session.close()
        self.assertEqual(self.robot.disconnected, 1)
        with self.assertRaises(RuntimeError):
            session.start()
        with self.assertRaises(RuntimeError):
            session.snapshot()


if __name__ == '__main__':
    unittest.main()
