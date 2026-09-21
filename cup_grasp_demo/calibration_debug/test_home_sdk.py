"""Installed SDK packet test. Never connects to the physical CAN interface."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cup_grasp_demo.calibration_debug.home_execution import execute_home


class HomeSDKTest(unittest.TestCase):
    def exercise(self, move, terminal=False, missing=False):
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
        from can.interfaces.socketcan import SocketcanBus
        frames, clock = [], SimpleNamespace(now=1000., joints=[0.] * 7)
        names = ('thumb_tip', 'thumb_base', 'index_finger', 'middle_finger', 'ring_finger', 'pinky_finger')
        def feedback():
            return SimpleNamespace(timestamp=clock.now)
        def sleep(seconds):
            clock.now += seconds
        with patch.object(SocketcanBus, 'send', side_effect=AssertionError('Physical CAN forbidden')) as physical:
            robot = AgxArmFactory.create_arm(create_agx_arm_config(robot='nero', firmeware_version='v120', channel='can0'))
            robot._ctx.get_comm = lambda: SimpleNamespace(send=frames.append, get_channel=lambda: 'memory')
            robot.get_joints_enable_status_list = lambda: [True] * 7
            hand = robot.init_effector(robot.OPTIONS.EFFECTOR.REVO2)
            hand.get_finger_pos = hand.get_finger_current = feedback
            if terminal:
                hand.get_finger_pos = lambda: (SimpleNamespace(timestamp=1001.05)
                                               if clock.now >= 1001.05 else None)
            if missing:
                hand.get_finger_pos = lambda: None
            demo = SimpleNamespace(
                FINGER_NAMES=names, require_right_hand=lambda *_: None,
                arm_snapshot=lambda _: (clock.joints, None, SimpleNamespace(arm_status=0, motion_status=0, ctrl_mode=1)),
                read_fresh=lambda getter, *_: getter(), feedback_stamp=lambda f: None if f is None else f.timestamp,
                finger_values=lambda _: dict.fromkeys(names, 0), position_values=lambda _: dict.fromkeys(names, 0))
            result = {}
            target = [.1] * 7 if move else [0.] * 7
            plan = dict(start_q_rad=[0.] * 7, home_target_q_rad=target,
                        stages=[{'target_q_rad': target}] if move else [])
            def arm_step(stage, speed):
                self.assertTrue(move)
                self.assertGreaterEqual(clock.now, 1002.)
                self.assertEqual([f.arbitration_id for f in frames], [0x1B5, 0x1B5])
                robot.set_auto_set_motion_mode_enabled(False)
                robot.move_js(stage['target_q_rad'])
                clock.joints = list(stage['target_q_rad'])
            execute_home(plan, {'speed_percent': 5}, robot, hand, demo,
                         arm_step, result,
                         monotonic=lambda: clock.now, wallclock=lambda: clock.now, sleep=sleep)
            self.assertEqual([(f.arbitration_id, bytes(f.data).hex()) for f in frames[:2]],
                             [(0x1B5, '1200000000000000'), (0x1B5, '2200646464646464')])
            self.assertEqual([f.arbitration_id for f in frames[2:]], [0x155, 0x156, 0x157, 0x170] if move else [])
            self.assertEqual(result['home_ready_verified'], not missing)
            self.assertTrue(result['home_joint_target_reached'])
            self.assertTrue(result['home_hand_open']['command_wait_completed'])
            self.assertEqual(result['home_hand_open']['completion_basis'],
                             'command_duration_only_unverified' if missing else
                             'terminal_position_event' if terminal else 'position_feedback')
            physical.assert_not_called()

    def test_home_open_encodes_two_zero_open_hand_frames_and_no_arm_target(self):
        self.exercise(move=False)

    def test_open_packets_and_wait_precede_home_joint_packets(self):
        self.exercise(move=True)

    def test_one_terminal_event_allows_home_without_resending_hand_packets(self):
        self.exercise(move=True, terminal=True)

    def test_missing_position_keeps_hand_unverified_and_confirms_joint_home(self):
        self.exercise(move=True, missing=True)


if __name__ == '__main__':
    unittest.main()
