from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rgb_hand_tracking import finger_feedback_probe as probe
from rgb_hand_tracking.passive_pose_bridge import PassiveTransmitForbidden


def frame(can_id=0x1C1, stamp=100., remote=True, payload=None):
    return SimpleNamespace(arbitration_id=can_id, timestamp=stamp, is_rx=remote,
                           data=payload or bytes([0, 0, 10, 20, 30, 40, 50, 60]))


class Hand:
    def __init__(self):
        self._parser = SimpleNamespace()
        for kind, (_, attribute, getter) in probe.PACKETS.items():
            setattr(self._parser, attribute, None)
            setattr(self, getter, lambda attr=attribute: getattr(self._parser, attr))

    def update(self, kind, stamp, values):
        attribute = probe.PACKETS[kind][1]
        setattr(self._parser, attribute, SimpleNamespace(
            timestamp=stamp, msg=SimpleNamespace(**values)))


class Bus:
    def send(self, message):
        pass


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.original_send = Bus.send
        self.cache = probe.BroadcastCache()
        self.hand = Hand()
        self.raw = frame()
        self.cache.receive(self.raw)
        self.values = dict(zip(probe.FINGERS, self.raw.data[2:]))
        self.hand.update('position', 100., self.values)

    def tearDown(self):
        Bus.send = self.original_send

    def copy(self, wallclock=lambda: 100.01):
        return probe._copy_getter(self.hand, 'position', self.cache, wallclock)

    def test_fresh_six_channel_position_matches_raw(self):
        row = self.copy()
        self.assertTrue(row['fresh'])
        self.assertEqual(row['values'], self.values)

    def test_other_packet_does_not_make_position_fresh(self):
        self.hand.update('position', 99., self.values)
        self.cache.receive(frame(0x1C0))
        self.assertFalse(self.copy()['fresh'])

    def test_stale_and_future_position_are_rejected(self):
        for now in (101., 99.):
            self.assertFalse(self.copy(lambda: now)['fresh'])

    def test_mutated_sdk_values_without_matching_raw_are_rejected(self):
        wrong = self.values.copy()
        wrong['thumb_tip'] += 1
        self.hand.update('position', 100., wrong)
        self.assertEqual(self.copy()['reason'], 'no_matching_raw_broadcast_in_this_process')

    def test_local_frame_does_not_prove_hardware_feedback(self):
        cache = probe.BroadcastCache()
        cache.receive(frame(remote=False))
        row = probe._copy_getter(self.hand, 'position', cache, lambda: 100.01)
        self.assertEqual(row['reason'], 'local_injected_frame_not_hardware_rx')

    def test_timestamp_race_is_rejected(self):
        def changed():
            self.hand.update('position', 100.02, self.values)
            return self.hand._parser.finger_pos
        self.hand.get_finger_pos = changed
        self.assertEqual(self.copy()['reason'], 'timestamp_changed_during_copy')

    def test_signed_raw_current_is_not_claimed_as_force(self):
        self.cache.receive(frame(0x1C3, payload=bytes([0, 0, 255, 128, 127, 0, 1, 2])))
        self.assertEqual(self.cache.packets['current'][-1]['values']['thumb_tip'], -1)
        self.assertEqual(self.cache.packets['current'][-1]['values']['thumb_base'], -128)

    def test_factory_send_is_refused(self):
        def factory():
            Bus().send(SimpleNamespace())
        report = probe.run_probe(Bus, factory, seconds=.1)
        self.assertFalse(report['valid'])
        self.assertEqual(report['tx_attempts'], 1)
        self.assertEqual(report['actual_tx_count'], 0)

    def run_fake(self, *, broadcast=True, disconnect_error=False):
        clock = SimpleNamespace(now=200.)
        hand = Hand()
        robot = SimpleNamespace()
        def register(callback):
            robot.callback = callback
        def emit():
            if broadcast:
                raw = frame(stamp=clock.now)
                hand.update('position', clock.now, dict(zip(probe.FINGERS, raw.data[2:])))
                robot.callback(raw)
        def sleep(seconds):
            clock.now += seconds
            emit()
        def disconnect():
            if disconnect_error:
                raise RuntimeError('disconnect failed')
        robot.init_effector = lambda name: hand
        robot.get_context = lambda: SimpleNamespace(register_parser_packet_fun=register)
        robot.connect = emit
        robot.disconnect = disconnect
        with patch.object(probe, 'source_runtime_info', return_value={}), \
                patch.object(probe, '_extra_sources', return_value={}):
            return probe.run_probe(Bus, lambda: robot, seconds=.08,
                                   monotonic=lambda: clock.now, wallclock=lambda: clock.now,
                                   sleep=sleep)

    def test_full_probe_requires_advancing_broadcast_and_disconnects(self):
        report = self.run_fake()
        self.assertTrue(report['valid'])
        self.assertGreaterEqual(report['position_frame_count'], 3)
        self.assertEqual(report['tx_attempts'], 0)
        self.assertTrue(report['cleanup']['disconnect_succeeded'])
        self.assertFalse(report['grasp_ready'])

    def test_missing_broadcast_remains_explicit_blocker(self):
        report = self.run_fake(broadcast=False)
        self.assertFalse(report['valid'])
        self.assertIn('Missing 0x1C1', report['blocker'])
        self.assertEqual(report['tx_attempts'], 0)

    def test_disconnect_failure_keeps_transmit_refused(self):
        report = self.run_fake(disconnect_error=True)
        self.assertFalse(report['valid'])
        self.assertTrue(report['cleanup']['tx_guard_active'])
        with self.assertRaises(PassiveTransmitForbidden):
            Bus().send(SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
