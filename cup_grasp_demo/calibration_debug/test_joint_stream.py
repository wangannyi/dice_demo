import threading
import time
import unittest
from unittest.mock import patch

from cup_grasp_demo.calibration_debug.joint_stream import DeadlineClock, FeedbackReader
from cup_grasp_demo.calibration_debug.joint_profile import options


def row(stamp):
    return dict(sdk_snapshot=dict(packet_timestamps_after_epoch_s=dict(joint_12=stamp)),
                status_timestamp_epoch_s=stamp, enable_feedback_timestamps_epoch_s=[stamp]*7)


class StreamTest(unittest.TestCase):
    def test_200_hz_deadlines_without_accumulated_send_cost(self):
        now = [0.0]
        clock = DeadlineClock(200, 0, monotonic=lambda: now[0],
                              sleep=lambda dt: now.__setitem__(0, now[0]+dt))
        stamps = []
        for _ in range(201):
            stamps.append(clock.wait())
            now[0] += .001  # SDK send duration must not extend every period.
            clock.advance()
        self.assertAlmostEqual(stamps[-1], 1.0)
        self.assertEqual(clock.skipped, 0)

    def test_overrun_skips_slots_instead_of_burst(self):
        now = [0.0]
        clock = DeadlineClock(200, 0, monotonic=lambda: now[0],
                              sleep=lambda dt: now.__setitem__(0, now[0]+dt))
        clock.wait()
        now[0] = .021
        clock.advance()
        self.assertEqual(clock.skipped, 4)
        self.assertAlmostEqual(clock.wait(), .025)

    def test_feedback_block_does_not_block_sender_and_is_joined(self):
        entered = threading.Event()
        def read(**kwargs):
            entered.set()
            kwargs['sleep'](10)
        reader = FeedbackReader(read, row(time.time())).start()
        try:
            self.assertTrue(entered.wait(1))
            start = time.monotonic()
            for _ in range(20):
                reader.latest(time.time())
            self.assertLess(time.monotonic()-start, .1)
        finally:
            reader.close()
        self.assertFalse(reader.thread.is_alive())

    def test_stale_feedback_and_reader_failure_stop_stream(self):
        reader = FeedbackReader(None, row(1000))
        with self.assertRaisesRegex(RuntimeError, '100 ms'):
            reader.latest(1000.101)
        reader.error = TimeoutError('lost CAN')
        with self.assertRaisesRegex(RuntimeError, 'lost CAN'):
            reader.latest(1000)

    def test_legacy_default_and_invalid_rate(self):
        self.assertIsNone(options({})['command_rate_hz'])
        self.assertEqual(options({'command_rate_hz':200})['command_rate_hz'],200)
        for value in (0, 201, True, 199.5, float('nan')):
            with self.assertRaises(ValueError):
                options({'command_rate_hz':value})

    def test_fixed_rate_executor_completes_and_holds_on_reader_error(self):
        from cup_grasp_demo.calibration_debug import joint_execution as sdk
        from cup_grasp_demo.calibration_debug.test_joint_lab import ExecutionTest
        closed = []
        class Reader:
            def __init__(self, read, initial):
                self.read = read
            def start(self):
                return self
            def latest(self, stamp):
                result = self.read()
                result['observed_monotonic_s'] = sdk.time.monotonic()
                return result
            def close(self):
                closed.append(True)
        with patch('cup_grasp_demo.calibration_debug.joint_stream.FeedbackReader',Reader):
            result, _, _ = ExecutionTest().simulate(config=dict(command_rate_hz=200))
            self.assertTrue(result['success'], result.get('error'))
            self.assertEqual(result['command_stream']['requested_hz'],200)
            failed, _, _ = ExecutionTest().simulate(fail_at=.5, config=dict(command_rate_hz=200))
            self.assertFalse(failed['success'])
            self.assertIn('failure_hold', failed)
        self.assertEqual(len(closed),2)
