"""Fake-clock overlap scheduling and executor failure isolation."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from cup_grasp_demo.flow.feedback_execution import HandSchedule, execute


class ScheduleTests(unittest.TestCase):
    def test_zero_and_delayed_dispatch_once(self):
        for delay in (0, .2, 3):
            now = [10.0]
            send = Mock()
            schedule = HandSchedule(delay, .5, send, clock=lambda:now[0])
            schedule.tick()
            if delay:
                send.assert_not_called()
            now[0] += delay + 1e-6
            schedule.tick()
            schedule.tick()
            send.assert_called_once()
            self.assertFalse(schedule.complete())
            now[0] += .51
            self.assertTrue(schedule.complete())

    def run_action(self, failure=False, maximum=False, duration=.5):
        hand, robot, demo = Mock(), Mock(), Mock()
        demo.FINGER_NAMES = ['a','b','c','d','e','f']
        demo.arm_snapshot.return_value = (None, None, SimpleNamespace(arm_status=0,ctrl_mode=1))
        robot.get_joints_enable_status_list.return_value = [True]*7
        motion = SimpleNamespace(on_motion_tick=None)
        result = {}
        now = [0.0]
        def step(*args):
            if failure:
                raise RuntimeError('arm failed before streaming')
            motion.on_motion_tick()
            now[0] += 1
        with patch('cup_grasp_demo.flow.feedback_execution.HandSchedule',
                   side_effect=lambda d,t,s:HandSchedule(d,t,s,clock=lambda:now[0])):
            try:
                execute(dict(stages=[dict(current_q_rad=[0]*7,
                                          target_q_rad=[.1]+[0]*6)],
                             target_0_100=[0]*6,hand_delay_s=0),
                        dict(green_cup=dict(finger_duration_s=.65 if maximum else duration, feedback_hand_max_speed=maximum),speed_percent=30),
                        robot,hand,demo,result,step,motion)
            except RuntimeError:
                if not failure:
                    raise
        self.assertIsNone(motion.on_motion_tick)
        return hand,result

    def test_overlap_executes_both_frames(self):
        hand,result = self.run_action()
        self.assertEqual(hand.position_time_ctrl.call_count, 2)
        self.assertEqual(result['hand_command']['actual_delay_s'],0)

    def test_fast_home_hand_dispatches_during_arm_stream(self):
        hand, result = self.run_action(duration=.25)
        self.assertEqual(hand.position_time_ctrl.call_count, 2)
        self.assertEqual(result['hand_command']['actual_delay_s'], 0)
        self.assertEqual(result['hand_command']['duration_s'], .25)

    def test_overlap_max_speed_uses_zero_time(self):
        hand,result = self.run_action(maximum=True)
        self.assertEqual(hand.position_time_ctrl.call_args.kwargs,
                         dict(mode='time', **dict.fromkeys(['a','b','c','d','e','f'], 0)))
        self.assertEqual(result['hand_command']['speed_mode'], 'max')

    def test_arm_failure_does_not_start_hand(self):
        hand,_ = self.run_action(failure=True)
        hand.position_time_ctrl.assert_not_called()

    def test_same_arm_pose_still_dispatches_hand(self):
        hand, robot, demo = Mock(), Mock(), Mock()
        demo.FINGER_NAMES = ['a','b','c','d','e','f']
        demo.arm_snapshot.return_value = (None, None, SimpleNamespace(arm_status=0,ctrl_mode=1))
        robot.get_joints_enable_status_list.return_value = [True]*7
        motion = SimpleNamespace(on_motion_tick=None)
        result = {}
        stage = dict(current_q_rad=[0]*7, target_q_rad=[0]*7)
        execute(dict(stages=[stage], target_0_100=[0]*6, hand_delay_s=0),
                dict(green_cup=dict(finger_duration_s=.25,
                     feedback_hand_max_speed=False), speed_percent=100),
                robot, hand, demo, result, Mock(), motion)
        self.assertTrue(result['arm_already_at_target'])
        self.assertEqual(hand.position_time_ctrl.call_count, 2)


class SequenceTests(unittest.TestCase):
    def test_three_cycles_finish_open_without_catchup_burst(self):
        from cup_grasp_demo.flow.feedback_sequence import sequence_values
        from cup_grasp_demo.flow.feedback_execution import SequenceSchedule
        a,b = [0]*6,[0,0,40,40,40,40]
        targets,interval = sequence_values(dict(poses=[a,b],cycles=3,interval_s=.65),a,.65)
        self.assertEqual(targets,[a,b,a,b,a,b,a])
        now,sent = [0.0],[]
        schedule=SequenceSchedule(0,.65,targets,interval,lambda p:sent.append(p),clock=lambda:now[0])
        schedule.tick()
        now[0]=10
        schedule.tick()
        schedule.tick()
        self.assertEqual(sent,[a,b])
        while len(sent)<7:
            now[0]+=.651
            schedule.tick()
        self.assertEqual(sent,targets)
        self.assertFalse(schedule.complete())
        now[0]+=.651
        self.assertTrue(schedule.complete())

    def test_reject_invalid_sequence_before_motion(self):
        from cup_grasp_demo.flow.feedback_sequence import sequence_values
        for change in [dict(cycles=0),dict(cycles=1.5),dict(interval_s=.01),dict(poses=[[0]*6,[101]*6])]:
            raw=dict(poses=[[0]*6,[0,0,40,40,40,40]],cycles=3,interval_s=.65)
            raw.update(change)
            with self.assertRaises(ValueError):
                sequence_values(raw,[0]*6,.65)

    def test_one_way_sequence_finishes_at_second_pose(self):
        from cup_grasp_demo.flow.feedback_sequence import sequence_values
        a, b = [0, 0, 100, 100, 100, 100], [100] * 6
        targets, interval = sequence_values(
            dict(poses=[a, b], cycles=1, interval_s=.65,
                 return_to_initial=False), a, .65)
        self.assertEqual(targets, [a, b])
        self.assertEqual(interval, .65)

    def test_one_way_sequence_can_overlap_finger_motion(self):
        from cup_grasp_demo.flow.feedback_sequence import sequence_values
        a, b = [0, 0, 100, 100, 100, 100], [100] * 6
        targets, interval = sequence_values(
            dict(poses=[a, b], cycles=1, interval_s=.1,
                 return_to_initial=False), a, .65)
        self.assertEqual(targets, [a, b])
        self.assertEqual(interval, .1)

    def test_single_action_remains_single(self):
        from cup_grasp_demo.flow.feedback_sequence import sequence_values
        self.assertEqual(sequence_values(None,[100]*6,.5),([[100]*6],.5))
