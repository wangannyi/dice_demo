from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rgb_hand_tracking import prepare_open_hand as opening
from test_visual_servo_probe import FakeBus, FakeRobot, FakeSession, feedback, receiver_evidence


class OpeningTests(unittest.TestCase):
    def setUp(self):
        self.original = FakeBus.send

    def tearDown(self):
        FakeBus.send = self.original

    def test_position_time_order_target_and_time(self):
        commands = []
        hand = SimpleNamespace(position_time_ctrl=lambda **kw: commands.append(kw))
        opening.send_open(hand, monotonic=lambda: 1.)
        self.assertEqual(commands[0], dict(mode='pos', **opening.OPEN_TARGET))
        self.assertEqual(commands[1], dict(mode='time', **dict.fromkeys(opening.FINGERS, 100)))

    def test_excessive_pair_interval_reports_failure(self):
        times = iter([1., 1.051])
        hand = SimpleNamespace(position_time_ctrl=lambda **kw: None)
        with self.assertRaisesRegex(RuntimeError, '50 ms'):
            opening.send_open(hand, monotonic=lambda: next(times))

    def run_fake(self, *, execute=False, position=True, current=True, mode=1, drift=False):
        clock = SimpleNamespace(now=10.)
        robot = FakeRobot()
        commands = []
        def control(**kwargs):
            commands.append(kwargs)
            robot.bus.send(SimpleNamespace(arbitration_id=0x1B5, data=b'control'))
        hand = SimpleNamespace(position_time_ctrl=control)
        robot.init_effector = lambda name: hand
        robot.get_context = lambda: SimpleNamespace(register_parser_packet_fun=lambda cb: None)
        def getter(hand, kind, cache, wallclock):
            values = dict(zip(opening.FINGERS, [46, 8, 0, 0, 0, 0]))
            if drift and clock.now > 12:
                values['thumb_tip'] = 130
            return {'fresh': current if kind == 'current' else position,
                    'timestamp_getter_epoch_s': clock.now, 'values': values}
        def sleep(t):
            clock.now += t
        evidences = iter([{'errors': [], 'receiver_rows': [], 'candidate_control_processes': []},
                          receiver_evidence()])
        with patch.object(opening, 'PassivePoseSession', FakeSession), \
                patch.object(opening, 'stopped_window', return_value=([feedback(mode=mode)]*10, {})), \
                patch.object(opening, '_copy_getter', side_effect=getter):
            report = opening.run(FakeBus, lambda: robot, execute=execute,
                                 evidence_provider=lambda channel: next(evidences),
                                 monotonic=lambda: clock.now, wallclock=lambda: clock.now, sleep=sleep)
        return commands, robot, report

    def test_inspection_has_zero_transmit(self):
        commands, _, report = self.run_fake()
        self.assertEqual(commands, [])
        self.assertEqual(report['tx']['actual_tx_count'], 0)

    def test_opening_has_two_hand_frames_and_no_arm_target(self):
        commands, robot, report = self.run_fake(execute=True)
        self.assertEqual(len(commands), 2)
        self.assertEqual(robot.commands, [])
        self.assertEqual(report['tx']['actual_tx_count'], 2)
        self.assertTrue(report['open_target_reached'])
        self.assertFalse(report['grasp_ready'])

    def test_missing_position_after_command_is_not_success(self):
        commands, _, report = self.run_fake(execute=True, position=False)
        self.assertEqual(len(commands), 2)
        self.assertFalse(report['success'])
        self.assertFalse(report['position_feedback_valid'])

    def test_current_feedback_missing_blocks_before_command(self):
        commands, _, report = self.run_fake(execute=True, current=False)
        self.assertEqual(commands, [])
        self.assertEqual(report['tx']['actual_tx_count'], 0)

    def test_non_can_mode_blocks_before_command(self):
        commands, _, report = self.run_fake(execute=True, mode=3)
        self.assertEqual(commands, [])
        self.assertFalse(report['success'])

    def test_initial_target_then_rebound_is_not_success(self):
        _, _, report = self.run_fake(execute=True, drift=True)
        self.assertFalse(report['success'])
        self.assertFalse(report['open_target_reached'])


if __name__ == '__main__':
    unittest.main()
