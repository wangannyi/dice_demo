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
        self.assertEqual(delivery_options()['scheduling_gap_limit_s'], .08)
        self.assertEqual(delivery_options()['limits_read_retries'], 1)
        self.assertIs(delivery_options()['cache_live_limits'], False)
        self.assertEqual(delivery_options(
            {'feedback_freshness_limit_s': .3})['feedback_freshness_limit_s'], .3)
        self.assertEqual(delivery_options(
            {'scheduling_gap_limit_s': .3})['scheduling_gap_limit_s'], .3)
        for bad in (0, .05, .6, True, 'fast'):
            with self.assertRaises(ValueError):
                delivery_options({'feedback_freshness_limit_s': bad})
            with self.assertRaises(ValueError):
                delivery_options({'scheduling_gap_limit_s': bad})

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

class ControllerEndpointTests(unittest.TestCase):
    def make_endpoint(self):
        servo, demo = make_delivery(options={'command_mode': 'controller_endpoint'})
        servo.robot.get_joint_angle_vel_limits.return_value = feedback(NOW, SimpleNamespace(
            min_angle_limit=-2., max_angle_limit=2., max_joint_spd=2.))
        servo.robot.get_joint_acc_limits.return_value = feedback(NOW, SimpleNamespace(max_joint_acc=5.))
        clock = [0.]
        servo.monotonic = lambda: clock[0]
        servo.sleep = lambda seconds: clock.__setitem__(0, clock[0]+seconds)
        servo.check_feedback = Mock(side_effect=lambda *args: [0.4 if servo.robot.move_j.called else 0.1]*7)
        return servo

    def test_fixed_target_uses_move_j_once_and_ticks_hand_until_arrival(self):
        servo = self.make_endpoint()
        servo.on_motion_tick = Mock()
        servo.move_js([.4]*7)
        servo.robot.move_j.assert_called_once_with([.4]*7)
        servo.robot.move_js.assert_not_called()
        servo.robot.set_motion_mode.assert_called_once_with('j')
        self.assertGreaterEqual(servo.on_motion_tick.call_count, 2)
        self.assertTrue(servo.events[-1]['delivery_completed'])
        self.assertEqual(servo.events[-1]['sdk_method'], 'move_j')

    def test_no_arrival_times_out_instead_of_claiming_success(self):
        servo = self.make_endpoint()
        servo.check_feedback = Mock(return_value=[.1]*7)
        with self.assertRaises(TimeoutError):
            servo.move_js([.4]*7)
        self.assertFalse(servo.events[-1]['delivery_completed'])

    def test_target_outside_live_limits_never_sent(self):
        servo = self.make_endpoint()
        with self.assertRaises(ValueError):
            servo.move_js([3.]*7)
        servo.robot.move_j.assert_not_called()

    def test_legacy_default_remains_smooth_profile(self):
        self.assertEqual(delivery_options()['command_mode'], 'smooth_profile')

    def test_partial_first_packet_is_completed_by_identical_endpoint_resend(self):
        servo = self.make_endpoint()
        servo.options['endpoint_resend_interval_s'] = .05
        def observed(*args):
            count = servo.robot.move_j.call_count
            if count == 0:
                return [.1]*7
            return [.4]*7 if count >= 2 else [.4, .4, .1, .1, .1, .1, .1]
        servo.check_feedback = Mock(side_effect=observed)
        servo.move_js([.4]*7)
        self.assertEqual(servo.robot.move_j.call_count, 2)
        for call in servo.robot.move_j.call_args_list:
            self.assertEqual(call.args, ([.4]*7,))
        self.assertTrue(servo.events[-1]['delivery_completed'])
        self.assertEqual(servo.events[-1]['final_error_deg'], 0)
        servo.robot.move_js.assert_not_called()

    def test_resending_does_not_hide_stuck_joint_or_disable_timeout(self):
        servo = self.make_endpoint()
        servo.options['endpoint_resend_interval_s'] = .05
        servo.check_feedback = Mock(return_value=[.1]*7)
        with self.assertRaises(TimeoutError):
            servo.move_js([.4]*7)
        event = servo.events[-1]
        self.assertFalse(event['delivery_completed'])
        self.assertGreater(event['sent_count'], 1)
        timeout = min(120., max(3., event['duration_s']*3+1))
        self.assertLessEqual(event['sent_count'], int(timeout/.05)+1)
        self.assertLessEqual(event['actual_delivery_s'], timeout+.02)
        self.assertGreater(event['final_error_deg'], .5)

    def test_resend_interval_validation(self):
        for value in (-1, .001, .3, float('nan'), True):
            with self.assertRaises(ValueError):
                delivery_options({'endpoint_resend_interval_s': value})
        self.assertEqual(delivery_options()['endpoint_resend_interval_s'], 0)

    def test_small_start_drift_rebases_profile_before_command(self):
        import math
        servo = self.make_endpoint()
        servo.options['start_drift_tolerance_deg'] = .25
        observed = [.1 + math.radians(.2)]*7
        servo.check_feedback = Mock(side_effect=[[.1]*7, observed, [.4]*7, [.4]*7])
        servo.move_js([.4]*7)
        self.assertEqual(servo.events[-1]['start_q_rad'], observed)
        self.assertAlmostEqual(servo.events[-1]['start_drift_deg'], .2)
        self.assertTrue(servo.events[-1]['delivery_completed'])

    def test_large_start_drift_still_prevents_motion(self):
        import math
        servo = self.make_endpoint()
        servo.options['start_drift_tolerance_deg'] = .25
        servo.check_feedback = Mock(side_effect=[[.1]*7, [.1+math.radians(.3)]*7])
        with self.assertRaisesRegex(RuntimeError, 'starting posture'):
            servo.move_js([.4]*7)
        servo.robot.move_j.assert_not_called()

    def test_start_drift_bounds_and_legacy_default(self):
        for value in (.09, .51, True, float('nan')):
            with self.assertRaises(ValueError):
                delivery_options({'start_drift_tolerance_deg': value})
        self.assertEqual(delivery_options()['start_drift_tolerance_deg'], .1)
