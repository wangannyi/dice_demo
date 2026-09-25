"""Feedback age policy stays strict by default and is configurable for held-cup shake."""

import time
import unittest

from cup_grasp_demo.flow import joint_execution
from cup_grasp_demo.flow.joint_stream import FeedbackReader


def feedback(stamp):
    return {
        'sdk_snapshot': {
            'packet_timestamps_after_epoch_s': {
                'joint_12': stamp,
                'joint_34': stamp,
                'joint_56': stamp,
                'joint_7': stamp,
            },
        },
        'status_timestamp_epoch_s': stamp,
        'enable_feedback_timestamps_epoch_s': [stamp] * 7,
    }


class FeedbackFreshnessTests(unittest.TestCase):
    def test_expanded_limit_is_restricted_to_held_cup_pipeline(self):
        self.assertEqual(
            joint_execution.feedback_freshness_limit({
                'load_context': 'green_cup_held',
                'feedback_freshness_limit_s': .3,
            }),
            .3,
        )
        with self.assertRaisesRegex(ValueError, 'only for green held-cup'):
            joint_execution.feedback_freshness_limit({
                'feedback_freshness_limit_s': .3,
            })

    def test_default_still_rejects_feedback_older_than_100_ms(self):
        reader = FeedbackReader(lambda **_: None, feedback(9.85))
        with self.assertRaisesRegex(RuntimeError, '100 ms'):
            reader.latest(10.0)

    def test_held_cup_limit_accepts_short_scheduling_gap(self):
        reader = FeedbackReader(
            lambda **_: None,
            feedback(9.85),
            joint_max_age_s=.3,
            state_max_age_s=.3,
        )
        self.assertIs(reader.latest(10.0), reader.row)

    def test_configured_limit_still_rejects_a_real_feedback_outage(self):
        reader = FeedbackReader(
            lambda **_: None,
            feedback(9.6),
            joint_max_age_s=.3,
            state_max_age_s=.3,
        )
        with self.assertRaisesRegex(RuntimeError, '300 ms'):
            reader.latest(10.0)

    def test_configured_limit_reaches_the_inner_sdk_read(self):
        # Regression: the inner fresh_feedback used to re-impose a hard-coded
        # 100 ms gate, voiding the configured budget before latest() ever ran.
        # fresh_feedback binds monotonic/wallclock defaults at import time, so
        # the mock must stamp with the real clock; only the packet ages (0.2 s)
        # simulate the aged feedback stream seen after a first shake round.
        from types import SimpleNamespace
        from nero_revo2_control.bridges import visual_servo_probe as probe

        def snapshot():
            return {'q_rad': [0.]*7, 'fk_flange_pose_m_rad': [0.]*6,
                    'packet_ages_s': {k: .2 for k in probe.PACKETS},
                    'packet_timestamps_after_epoch_s': {k: time.time() for k in probe.PACKETS}}

        robot = SimpleNamespace(
            get_arm_status=lambda: SimpleNamespace(timestamp=time.time(), msg=SimpleNamespace(
                arm_status=0, ctrl_mode=1, motion_status=0)),
            get_driver_states=lambda joint_index: SimpleNamespace(timestamp=time.time()),
            get_joints_enable_status_list=lambda: [True]*7)
        session = SimpleNamespace(snapshot=snapshot, robot=robot,
                                  guard=SimpleNamespace(
                                      report=lambda: {'tx_attempts': 0, 'actual_tx_count': 0},
                                      allowed=False))
        reader = FeedbackReader(
            lambda **kwargs: probe.fresh_feedback(session, **kwargs), None,
            joint_max_age_s=.3, state_max_age_s=.3).start()
        try:
            deadline = time.monotonic() + 2
            while reader.row is None and reader.error is None:
                self.assertLess(time.monotonic(), deadline, 'inner read never produced a row')
                time.sleep(.01)
            self.assertIsNone(reader.error, reader.error)
            row = reader.latest(time.time())
            self.assertEqual(len(row['q_rad']), 7)
        finally:
            reader.close()


if __name__ == '__main__':
    unittest.main()


class VerifyStopFreshnessTests(unittest.TestCase):
    """The stop verification after a shake must honour the same budget."""

    def _stopped_session(self, max_age):
        """A fake session whose every read returns max_age-old packets."""
        from types import SimpleNamespace
        from unittest.mock import Mock
        from nero_revo2_control.bridges import visual_servo_probe as probe
        age = max_age

        def snapshot():
            return {'q_rad': [0.]*7, 'fk_flange_pose_m_rad': [0.]*6,
                    'packet_ages_s': {k: age for k in probe.PACKETS},
                    'packet_timestamps_after_epoch_s': {k: time.time() for k in probe.PACKETS}}

        robot = SimpleNamespace(
            get_arm_status=lambda: SimpleNamespace(timestamp=time.time(), msg=SimpleNamespace(
                arm_status=0, ctrl_mode=1, motion_status=0)),
            get_driver_states=lambda joint_index: SimpleNamespace(timestamp=time.time()),
            get_joints_enable_status_list=lambda: [True]*7)
        session = SimpleNamespace(snapshot=snapshot, robot=robot,
                                  guard=SimpleNamespace(
                                      report=lambda: {'tx_attempts': 0, 'actual_tx_count': 0},
                                      allowed=False))
        return session

    def test_configured_limit_lets_verify_stop_accept_aged_feedback(self):
        from cup_grasp_demo.flow.shake_execution import verify_stop
        session = self._stopped_session(.2)
        report = []
        verify_stop(session, [0.]*7, report, stable_samples=3, poll_s=0.0,
                    joint_max_age_s=.3)
        self.assertEqual(len(report), 3)

    def test_default_stays_strict_in_verify_stop(self):
        from cup_grasp_demo.flow.shake_execution import verify_stop
        session = self._stopped_session(.2)
        with self.assertRaisesRegex(RuntimeError, 'freshness'):
            verify_stop(session, [0.]*7, [], stable_samples=3, poll_s=0.0)


class FreshJsHoldFreshnessTests(unittest.TestCase):
    """The failure-path hold must honour the same budget as the motion loop.

    Field evidence (game_20260924_163812, run ..._green_joint_shake_d47d58):
    the shake loop ran with 0.3s, then a busy CAN bus made the hold's default
    0.1s gate fail, so hold_verified never landed and the speed was not
    restored (restored_speed_percent = None).
    """

    def _hold_session(self, max_age):
        """A fake session whose every read returns max_age-old packets."""
        from types import SimpleNamespace
        from unittest.mock import Mock
        from nero_revo2_control.bridges import visual_servo_probe as probe
        age = max_age

        def snapshot():
            return {'q_rad': [0.]*7, 'fk_flange_pose_m_rad': [0.]*6,
                    'packet_ages_s': {k: age for k in probe.PACKETS},
                    'packet_timestamps_after_epoch_s': {k: time.time() for k in probe.PACKETS}}

        robot = SimpleNamespace(
            get_arm_status=lambda: SimpleNamespace(timestamp=time.time(), msg=SimpleNamespace(
                arm_status=0, ctrl_mode=1, motion_status=0)),
            get_driver_states=lambda joint_index: SimpleNamespace(timestamp=time.time()),
            get_joints_enable_status_list=lambda: [True]*7,
            set_joint_limits_enabled=Mock(),
            move_js=Mock())
        session = SimpleNamespace(snapshot=snapshot, robot=robot,
                                  guard=SimpleNamespace(
                                      report=lambda: {'tx_attempts': 0, 'actual_tx_count': 0},
                                      allowed=False))
        return session

    def _limits(self):
        return [(-1., 1.)] * 7

    def test_configured_limit_lets_failure_hold_accept_aged_feedback(self):
        from cup_grasp_demo.flow.joint_delivery import fresh_js_hold
        from nero_revo2_control.bridges import visual_servo_probe as probe
        session = self._hold_session(.2)
        result = fresh_js_hold(probe, session, self._limits(),
                               joint_max_age_s=.3)
        self.assertTrue(result['requested'])
        session.robot.move_js.assert_called_once()

    def test_default_stays_strict_in_failure_hold(self):
        from cup_grasp_demo.flow.joint_delivery import fresh_js_hold
        from nero_revo2_control.bridges import visual_servo_probe as probe
        session = self._hold_session(.2)
        with self.assertRaisesRegex(RuntimeError, 'freshness'):
            fresh_js_hold(probe, session, self._limits())
        session.robot.move_js.assert_not_called()


class RemainingFreshnessWiringTests(unittest.TestCase):
    """P2-5/6/7/8/9（2026-09-25）：同一配置预算贯穿五条剩余通道。"""

    def _aged_session(self, age):
        """Fake session whose state feeds (status/enable) are `age` old."""
        from types import SimpleNamespace
        from unittest.mock import Mock
        from nero_revo2_control.bridges import visual_servo_probe as probe
        now = time.time()

        def snapshot():
            return {'q_rad': [0.]*7, 'fk_flange_pose_m_rad': [0.]*6,
                    'packet_ages_s': {k: .05 for k in probe.PACKETS},
                    'packet_timestamps_after_epoch_s': {k: now - .05 for k in probe.PACKETS}}

        robot = SimpleNamespace(
            get_arm_status=lambda: SimpleNamespace(timestamp=now - age, msg=SimpleNamespace(
                arm_status=0, ctrl_mode=1, motion_status=0)),
            get_driver_states=lambda joint_index: SimpleNamespace(timestamp=now - age),
            get_joints_enable_status_list=lambda: [True]*7)
        return SimpleNamespace(snapshot=snapshot, robot=robot,
                               guard=SimpleNamespace(
                                   report=lambda: {'tx_attempts': 0, 'actual_tx_count': 0},
                                   allowed=False))

    def test_state_feed_honours_state_max_age_s(self):
        """P2-5：状态/使能流不再被内层 0.25 架空——0.2s 龄在 0.3 预算下放行。"""
        from nero_revo2_control.bridges import visual_servo_probe as probe
        session = self._aged_session(.2)
        row = probe.fresh_feedback(session, joint_max_age_s=.3,
                                   state_max_age_s=.3, sleep=lambda s: None)
        self.assertEqual(row['enabled'], [True]*7)

    def test_state_feed_default_stays_quarter_second(self):
        from nero_revo2_control.bridges import visual_servo_probe as probe
        session = self._aged_session(.26)
        with self.assertRaises((TimeoutError, RuntimeError)):
            probe.fresh_feedback(session, joint_max_age_s=.3,
                                 sleep=lambda s: None)

    def test_feedback_reader_passes_state_budget(self):
        """P2-5 接线：FeedbackReader 把 state_max_age_s 传给底层读取。"""
        seen = {}

        def read(**kwargs):
            seen.update(kwargs)
            reader.stop.set()   # one iteration is enough; stop after capture
            return feedback(time.time())

        reader = FeedbackReader(read, feedback(time.time()),
                                joint_max_age_s=.3, state_max_age_s=.4)
        reader._run()
        self.assertEqual(seen.get('state_max_age_s'), .4)

    def test_hand_feedback_honours_max_age_s(self):
        """P2-6：手指位置反馈门限可配，与臂侧同一预算。"""
        from types import SimpleNamespace
        from unittest.mock import Mock
        from cup_grasp_demo.flow.shake_execution import hand_feedback
        now = time.time()
        hand = SimpleNamespace(_parser=SimpleNamespace(
            finger_pos=SimpleNamespace(timestamp=now - .2),
            finger_current=None,
            hand_status=SimpleNamespace(timestamp=now - .2)))
        fresh_pos = SimpleNamespace(
            timestamp=now - .2,
            msg=SimpleNamespace(**{name: 100 for name in (
                'thumb_tip', 'thumb_base', 'index_finger', 'middle_finger',
                'ring_finger', 'pinky_finger')},
                left_or_right=2))
        fresh_status = SimpleNamespace(
            timestamp=now - .2,
            msg=SimpleNamespace(**{name: 0 for name in (
                'thumb_tip', 'thumb_base', 'index_finger', 'middle_finger',
                'ring_finger', 'pinky_finger')},
                left_or_right=2))
        hand.get_finger_pos = lambda: fresh_pos
        hand.get_finger_current = lambda: None
        hand.get_hand_status = lambda: fresh_status
        cache = Mock()
        cache.match.return_value = {'is_remote_rx': True}
        # 0.2s 龄：默认 0.25 放行、收紧到 0.1 拒绝。
        result = hand_feedback(hand, cache)
        self.assertTrue(result['position']['fresh'])
        with self.assertRaisesRegex(RuntimeError, '不新鲜'):
            hand_feedback(hand, cache, max_age_s=.1)

    def test_snapshot_max_age_s_configurable(self):
        """P2-7：snapshot 的包龄上限不再写死 0.25。"""
        from nero_revo2_control.bridges import passive_pose_bridge as bridge

        class FakeBus:
            pass

        session = bridge.PassivePoseSession(FakeBus, lambda: None,
                                            deadline_s=1, max_age_s=.4)
        self.assertEqual(session.max_age_s, .4)
        ready = {'deadline_s': 1}
        self.assertEqual(session.max_age_s, .4)  # 构造校验通过即接线生效
        with self.assertRaisesRegex(ValueError, 'max_age_s'):
            bridge.PassivePoseSession(FakeBus, lambda: None, deadline_s=1,
                                      max_age_s=.01)
        del ready

    def test_measurement_accepts_configured_sample_ages(self):
        """P2-8：验收测量用同一预算过滤——0.2s 龄样本不再被 0.1 静默丢弃。"""
        from cup_grasp_demo.flow.joint_profile import packet_samples
        stamp = time.time()
        rows = [
            dict(feedback={
                'q_rad': [0.1] * 7,
                'sdk_snapshot': {'packet_timestamps_after_epoch_s': {
                    'joint_12': stamp - .2, 'joint_34': stamp, 'joint_56': stamp,
                    'joint_7': stamp}},
                'observed_epoch_s': stamp},
                elapsed_s=float(i) * .02)
            for i in range(4)
        ]
        strict, _ = packet_samples(rows, 1)
        widened, _ = packet_samples(rows, 1, max_age_s=.3)
        self.assertEqual(len(strict), 0)
        self.assertGreater(len(widened), 0)

    def test_start_check_failure_maps_to_replan_code(self):
        """P2-9②：起点静止检查的反馈读取失败映射 start_position_changed。"""
        from types import SimpleNamespace

        class BoomSession:
            def snapshot(self):
                raise TimeoutError('No fresh stable four-packet snapshot')

        report = {}
        request = {'load_context': 'green_cup_held',
                   'feedback_freshness_limit_s': .3,
                   'start_tolerance_deg': .5}
        with self.assertRaises(TimeoutError):
            joint_execution.persistent_stopped_window(
                request, BoomSession(), report, max_windows=2)
        self.assertEqual(report.get('failure_code'), 'start_position_changed')
