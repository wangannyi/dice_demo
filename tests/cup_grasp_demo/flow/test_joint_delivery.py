"""Bounded MoveJS streaming, legacy handler routing, and real SDK wire encoding."""

import math
import struct
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from cup_grasp_demo.flow.joint_delivery import (
    ServoJointRobot,
    delivery_options,
    fresh_js_hold,
    take_js_control,
    trapezoid_profile,
    trapezoid_position,
)
from cup_grasp_demo.flow.shake_execution import ShakeGuard


class DeliveryTest(unittest.TestCase):
    def test_trapezoid_profile_respects_all_axis_budgets(self):
        import numpy as np
        rng = np.random.default_rng(42)
        for _ in range(30):
            target = rng.uniform(-1, 1, 7)
            velocity = rng.uniform(.1, 2, 7)
            acceleration = rng.uniform(.5, 5, 7)
            duration, ramp, peak = trapezoid_profile([0]*7, target, velocity, acceleration)
            ts = np.linspace(0, duration, 2001)
            us = np.array([trapezoid_position(t, duration, ramp, peak) for t in ts])
            self.assertAlmostEqual(us[0], 0)
            self.assertAlmostEqual(us[-1], 1)
            self.assertTrue(np.all(np.diff(us) >= -1e-12))
            self.assertTrue(np.all(abs(target) * peak <= velocity + 1e-12))
            self.assertTrue(np.all(abs(target) * peak / ramp <= acceleration + 1e-12))
            q = us[:, None] * target
            v = np.diff(q, axis=0)/(ts[1]-ts[0])
            a = np.diff(v, axis=0)/(ts[1]-ts[0])
            self.assertTrue(np.all(abs(v) <= velocity + 1e-8))
            self.assertTrue(np.all(abs(a) <= acceleration + 1e-7))

    def test_trapezoid_stream_retains_step_guard_and_endpoint(self):
        self.proxy.options['profile'] = 'trapezoid'
        self.proxy.set_speed_percent(60)
        self.proxy.move_js([.4, -.2, 0, 0, 0, 0, 0])
        self.assertEqual(self.q, [.4, -.2, 0, 0, 0, 0, 0])
        self.assertEqual(self.events[-1]['profile'], 'trapezoid')
        for (_, a), (_, b) in zip(self.sent, self.sent[1:]):
            self.assertLessEqual(max(abs(x-y) for x, y in zip(a, b)), math.radians(1))

    def setUp(self):
        self.clock = 100.0
        self.auto = True
        self.q = [0.0] * 7
        self.sent, self.modes = [], []
        self.state = NS(arm_status=0, ctrl_mode=1, mode_feedback=1, motion_status=0)
        self.robot = NS(
            get_auto_set_motion_mode_enabled=lambda: self.auto,
            set_auto_set_motion_mode_enabled=lambda v: setattr(self, "auto", v),
            set_motion_mode=lambda mode: self.modes.append(mode),
            set_speed_percent=lambda _: None,
            get_arm_status=lambda: NS(timestamp=self.clock, msg=self.state),
            get_joint_angles=lambda: NS(timestamp=self.clock, msg=self.q.copy()),
            get_joints_enable_status_list=lambda: [True] * 7,
            get_joint_angle_vel_limits=lambda *a, **k: NS(
                msg=NS(min_angle_limit=-3.0, max_angle_limit=3.0, max_joint_spd=3.0)
            ),
            get_joint_acc_limits=lambda *a, **k: NS(msg=NS(max_joint_acc=2.0)),
            move_js=self.send,
            move_j=lambda _: self.fail("Old SDK move_j called"),
        )
        self.demo = NS(
            feedback_stamp=lambda x: x.timestamp,
            check_comm=lambda _: None,
            read_fresh=lambda getter, *_: getter(),
        )
        self.events = []
        self.proxy = ServoJointRobot(
            self.robot,
            self.demo,
            self.events,
            sleep=self.sleep,
            wallclock=lambda: self.clock,
            monotonic=lambda: self.clock,
        )

    def sleep(self, dt):
        self.clock += dt

    def send(self, q):
        self.sent.append((self.clock, list(q)))
        self.q = list(q)

    def test_legacy_handler_streams_smooth_js_and_preserves_inactive_axes(self):
        self.proxy.move_j([0.4] + [0.0] * 6)
        self.assertEqual(self.modes, ["js"])
        self.assertGreater(len(self.sent), 10)
        self.assertAlmostEqual(self.q[0], 0.4)
        self.assertTrue(all(q[1:] == [0.0] * 6 for _, q in self.sent))
        velocities = [
            (b[1][0] - a[1][0]) / (b[0] - a[0])
            for a, b in zip(self.sent, self.sent[1:])
        ]
        self.assertLessEqual(max(velocities), 3 * 0.05 * 0.97 + 1e-6)
        self.assertLessEqual(
            max(abs(b - a) / 0.02 for a, b in zip(velocities, velocities[1:])),
            2 * 0.97 + 1e-5,
        )
        self.assertTrue(self.auto)
        self.assertTrue(self.events[0]["delivery_completed"])
        self.assertNotIn("target_reached", self.events[0])

    def test_optional_gesture_tick_runs_only_after_arm_transmission(self):
        seen = []
        self.proxy.on_motion_tick = lambda: seen.append(len(self.sent))
        self.proxy.move_js([.1] + [0.]*6)
        self.assertGreater(len(seen), 1)
        self.assertEqual(seen[0], 1)
        self.assertEqual(len(seen), len(self.sent)-2)

    def test_batched_limits_preserve_trajectory_and_fail_before_motion(self):
        self.proxy.move_j([.1] + [0.] * 6)
        duration = self.events[-1]['duration_s']
        self.setUp()
        self.proxy.batch_limits = True
        self.proxy.move_j([.1] + [0.] * 6)
        self.assertEqual(self.events[-1]['duration_s'], duration)
        self.assertTrue(self.events[-1]['limits_batched'])
        self.setUp()
        self.proxy.batch_limits = True
        self.robot.get_joint_acc_limits = lambda *a, **k: None
        with self.assertRaises(TimeoutError):
            self.proxy.move_j([.1] + [0.] * 6)
        self.assertEqual(self.sent, [])

    def test_high_speed_sampling_and_jitter_remain_bounded(self):
        for speed in (30,100):
            for jitter in (0,.035):
                self.setUp()
                self.robot.get_joint_acc_limits=lambda *a,**k:NS(msg=NS(max_joint_acc=5.))
                self.proxy.options=delivery_options(dict(limit_utilization=1.,velocity_cap_deg_s=None))
                self.proxy.set_speed_percent(speed)
                self.proxy.sleep=lambda dt:self.sleep(dt+jitter)
                self.proxy.move_js([.802]+[0.]*6)
                self.assertTrue(self.events[0]['delivery_completed'])
                self.assertLess(self.events[0]['sample_period_s'],.02)
                steps=[abs(b[1][0]-a[1][0]) for a,b in zip(self.sent,self.sent[1:])]
                self.assertLessEqual(max(steps),math.radians(.9)+1e-9)
                self.assertEqual(self.sent[-1][1],[.802]+[0.]*6)
                self.assertTrue(all(t2>t1 for (t1,_),(t2,_) in zip(self.sent,self.sent[1:])))
                if jitter:self.assertGreater(self.events[0]['phase_delay_s'],0)

    def test_feedback_failure_between_targets_stops_stream(self):
        def fault(q):
            self.send(q)
            self.state.arm_status = 7

        self.robot.move_js = fault
        with self.assertRaises(RuntimeError):
            self.proxy.move_js([0.1] * 7)
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(self.auto)

    def test_missing_limits_stale_feedback_or_invalid_target_sends_no_targets(self):
        for case in ("limits", "stale", "range", "nan"):
            self.setUp()
            target = [0.1] * 7
            if case == "limits":
                self.robot.get_joint_acc_limits = lambda *a, **k: None
            if case == "stale":
                self.robot.get_joint_angles = lambda: NS(timestamp=99.0, msg=self.q)
            if case == "range":
                target[0] = 4.0
            if case == "nan":
                target[0] = math.nan
            with self.subTest(case=case), self.assertRaises((ValueError, RuntimeError)):
                self.proxy.move_js(target)
            self.assertEqual(self.sent, [])

    def test_scheduler_stall_does_not_jump_ahead(self):
        self.proxy.sleep = lambda _: self.sleep(0.2)
        with self.assertRaisesRegex(RuntimeError, "80 ms"):
            self.proxy.move_js([0.1] * 7)
        self.assertEqual(len(self.sent), 1)

    def test_ignored_feedback_stops_at_tracking_bound(self):
        self.robot.move_js = lambda q: self.sent.append((self.clock, list(q)))
        with self.assertRaisesRegex(RuntimeError, "tracking error"):
            self.proxy.move_js([0.4] * 7)
        self.assertFalse(self.events[0]["delivery_completed"])

    def test_release_budget_uses_five_and_records_lag_until_end(self):
        self.robot.get_joint_acc_limits = lambda *a, **k: NS(msg=NS(max_joint_acc=5.))
        self.robot.move_js = lambda q: self.sent.append((self.clock, list(q)))
        self.proxy.options = delivery_options(dict(limit_utilization=1., tracking_error_action='record'))
        self.proxy.move_js([.4] * 7)
        event = self.events[0]
        self.assertEqual(event['acceleration_budget_rad_s2'], [5.] * 7)
        self.assertTrue(event['delivery_completed'])
        self.assertGreater(event['tracking_exceeded_samples'], 0)
        self.assertGreater(event['tracking_max_error_deg'], 5.)
        self.assertEqual(self.sent[-1][1], [.4] * 7)
        self.assertGreaterEqual(self.sent[-1][0]-self.sent[0][0], event['duration_s'])

    def test_record_mode_still_rejects_actual_angle_outside_envelope(self):
        self.proxy.options = delivery_options(dict(tracking_error_action='record'))
        def unexpected(q):
            self.send(q)
            self.q[6] = -.2
        self.robot.move_js = unexpected
        with self.assertRaisesRegex(RuntimeError, 'envelope'):
            self.proxy.move_js([.4] * 7)

    def test_handoff_and_failure_hold_use_js_without_changing_shared_core(self):
        session = NS(robot=self.robot)

        def handoff(s, *a, **k):
            s.robot.set_motion_mode("j")
            return {"ok": True}

        core = NS(take_can_control=handoff)
        take_js_control(core, session, {}, timeout_s=1)
        self.assertIs(session.robot, self.robot)
        self.assertEqual(self.modes, ["js"])
        self.robot.set_joint_limits_enabled = lambda _: None
        core.fresh_feedback = lambda _: dict(
            q_rad=[0.0] * 7, status=dict(arm_status=0, ctrl_mode=1), enabled=[True] * 7
        )
        core.validate_target = lambda q, limits: q
        fresh_js_hold(core, session, [])
        self.assertEqual(len(self.sent), 1)


class SDKPacketTest(unittest.TestCase):
    def test_real_sdk_js_mode_and_all_seven_angles_pass_guard_without_can_io(self):
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
        from can.interfaces.socketcan import SocketcanBus

        frames = []

        class MemoryBus:
            def send(self, frame):
                frames.append(frame)

            def get_channel(self):
                return "memory"

        with patch.object(
            SocketcanBus, "send", side_effect=AssertionError("Real CAN forbidden")
        ) as physical:
            robot = AgxArmFactory.create_arm(
                create_agx_arm_config(
                    robot="nero", firmeware_version="v120", channel="can0"
                )
            )
            robot._ctx.get_comm = lambda: MemoryBus()
            guard = ShakeGuard(MemoryBus)
            guard.install()
            guard.permit()
            guard.motion_allowed = True
            try:
                robot.set_speed_percent(5)
                q = [0.1, -0.2, 0.3, 0.4, -0.5, 0.1, -0.1]
                robot.move_js(q)
                modes = [f for f in frames if f.arbitration_id == 0x151]
                self.assertEqual(list(modes[-1].data[:4]), [1, 1, 5, 0xAD])
                joints = [
                    f
                    for f in frames
                    if f.arbitration_id in (0x155, 0x156, 0x157, 0x170)
                ]
                self.assertEqual(
                    [f.arbitration_id for f in joints], [0x155, 0x156, 0x157, 0x170]
                )
                values = [
                    v for f in joints for v in struct.unpack(">ii", bytes(f.data))
                ][:7]
                for a, b in zip(values, q):
                    self.assertAlmostEqual(a, math.degrees(b) * 1000, delta=1)
            finally:
                guard.restore()
            physical.assert_not_called()


if __name__ == "__main__":
    unittest.main()
