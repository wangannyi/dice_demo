"""MoveJS delivery freshness: configurable limit with a bounded fresh-read fallback."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from cup_grasp_demo.flow.joint_delivery import ServoJointRobot, delivery_options

NOW = 100.0


def feedback(stamp, msg):
    return SimpleNamespace(timestamp=stamp, msg=msg)


def status_msg():
    return SimpleNamespace(arm_status=0, ctrl_mode=1, mode_feedback=1, motion_status=0)


def joint_msg():
    return [0.1] * 7


def make_robot(demo, status, joints):
    robot = Mock()
    robot.get_arm_status.return_value = status
    robot.get_joint_angles.return_value = joints
    robot.get_joints_enable_status_list.return_value = [True] * 7
    robot.set_speed_percent = Mock()
    robot.get_auto_set_motion_mode_enabled.return_value = True
    return robot


def make_delivery(options=None, status_age=0.2, joints_age=0.05,
                  arm_snapshot=None):
    status = feedback(NOW - status_age, status_msg())
    joints = feedback(NOW - joints_age, joint_msg())
    demo = Mock()
    demo.feedback_stamp.side_effect = lambda message: getattr(message, "timestamp", None)
    demo.check_comm.return_value = None
    if arm_snapshot is not None:
        demo.arm_snapshot.side_effect = arm_snapshot
    else:
        # Default fallback: a consistent fresh snapshot (joints, pose, status).
        demo.arm_snapshot.return_value = (joint_msg(), [0.0] * 6, status_msg())
    robot = make_robot(demo, status, joints)
    servo = ServoJointRobot(robot, demo, [], options=options,
                            wallclock=lambda: NOW)
    return servo, demo


class DeliveryFreshnessTests(unittest.TestCase):
    def test_options_default_and_bounds(self):
        self.assertEqual(delivery_options()['feedback_freshness_limit_s'], .1)
        self.assertEqual(delivery_options(
            {'feedback_freshness_limit_s': .3})['feedback_freshness_limit_s'], .3)
        for bad in (0, .05, .6, True, 'fast'):
            with self.assertRaises(ValueError):
                delivery_options({'feedback_freshness_limit_s': bad})

    def test_configured_limit_accepts_cached_age_inside_window(self):
        servo, demo = make_delivery(
            options={'feedback_freshness_limit_s': .3},
            status_age=.2, joints_age=.2)
        # 0.2 s aged feedback passes the 0.3 s budget without any fallback.
        self.assertEqual(servo.check_feedback(), joint_msg())
        demo.arm_snapshot.assert_not_called()

    def test_default_limit_falls_back_to_one_fresh_read(self):
        # 0.2 s age exceeds the legacy 0.1 s default, but the bounded fresh
        # read rescues delivery instead of killing the run.
        servo, demo = make_delivery(options=None, status_age=.2)
        self.assertEqual(servo.check_feedback(), joint_msg())
        demo.arm_snapshot.assert_called_once_with(servo.robot)

    def test_stale_joints_label_also_uses_the_configured_limit(self):
        servo, demo = make_delivery(
            options={'feedback_freshness_limit_s': .3},
            status_age=.02, joints_age=.2)
        self.assertEqual(servo.check_feedback(), joint_msg())
        demo.arm_snapshot.assert_not_called()
        servo, demo = make_delivery(
            options={'feedback_freshness_limit_s': .1},
            status_age=.02, joints_age=.2)
        self.assertEqual(servo.check_feedback(), joint_msg())
        demo.arm_snapshot.assert_called_once_with(servo.robot)

    def test_fresh_read_failure_surfaces_as_delivery_error(self):
        def broken(_robot):
            raise RuntimeError('no fresh arm status feedback')
        servo, _demo = make_delivery(options=None, status_age=.2,
                                     arm_snapshot=broken)
        with self.assertRaisesRegex(RuntimeError, 'fresh'):
            servo.check_feedback()


if __name__ == '__main__':
    unittest.main()
