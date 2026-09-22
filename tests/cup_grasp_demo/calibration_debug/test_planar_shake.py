"""Offline geometry, duration, transport and fault-isolation tests; no hardware."""

from contextlib import ExitStack, redirect_stdout
import io
import tempfile
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cup_grasp_demo.calibration_debug import planar_shake as planner
from cup_grasp_demo.calibration_debug import planar_shake_execution as sdk
from cup_grasp_demo.calibration_debug import planar_shake_cli as cli
from cup_grasp_demo.calibration_debug.shake import Kinematics
from scipy.spatial.transform import Rotation

DATA = Path(__file__).resolve().parents[3] / "datasets/move_js_assessment_20260920"


class Clock:
    value = 0.

    def now(self):
        return self.value

    def sleep(self, amount):
        self.value += max(amount, .000001)


class PlanarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feedback = json.loads((DATA / "controller_limits_live.json").read_text())
        cls.feedback["q_after_rad"] = json.loads((DATA / "grasp_actual.json").read_text())["final_joints_rad"]
        cls.tcp = json.loads((DATA / "ready_plan.json").read_text())["T_flange_tcp"]
        cls.opts = json.loads(Path(planner.__file__).with_name("planar_shake.json").read_text())
        cls.plan = planner.make_plan(cls.feedback, cls.opts, cls.tcp)
        cls.kin = Kinematics()

    def row(self, q):
        q = list(q)
        t, _ = self.kin.forward(q)
        return dict(q_rad=q, enabled=[True]*7,
                    status=dict(arm_status=0, ctrl_mode=1, motion_status=1),
                    fk_flange_pose_m_rad=[*t[:3, 3], *Rotation.from_matrix(t[:3, :3]).as_euler("xyz")])

    def test_geometry_dynamics_and_closed_timing(self):
        p = self.plan
        self.assertTrue(p["planning_passed"], p["blockers"])
        self.assertEqual(p["samples"][-1]["t_s"], 20)
        self.assertEqual(len(p["samples"]), 2001)
        self.assertLess(max(p["joint_peak_acceleration_rad_s2"]), 5.)
        self.assertGreater(p["peak_deviation"]["rz_deg"], 5)
        self.assertGreater(p["peak_deviation"]["lateral_mm"], 7)
        for key in ("z_mm", "rx_deg", "ry_deg"):
            self.assertLess(p["peak_deviation"][key], .02)
        for sample in p["samples"][::100]:
            m = sdk.tcp_metrics(self.row(sample["q_rad"]), p)
            self.assertAlmostEqual(m["along_m"], sample["displacement_m"], places=6)

    def test_high_acceleration_still_blocks_without_silent_retiming(self):
        p = planner.make_plan(self.feedback, dict(self.opts, frequency_hz=3, auto_reduce_frequency=False), self.tcp)
        self.assertFalse(p["planning_passed"])
        self.assertEqual(p["parameters"]["frequency_hz"], 3)
        self.assertTrue(any("加速度" in b for b in p["blockers"]))

    def test_twenty_seconds_with_ordinary_tracking_error_and_no_motion_not_success(self):
        clock = Clock()
        robot = SimpleNamespace(get_joint_limits_enabled=lambda: True, move_js=lambda q: None)
        p = self.plan
        report = {}
        # Arm stays at center: very large phase error must not truncate the clock,
        # but the receipt must not say the requested physical motion happened.
        def poll(robot, stamps):
            return self.row(p["start_q_rad"]), {"test": clock.now()}
        sdk.stream(robot, p, report, clock=clock.now, sleep=clock.sleep, poll=poll)
        self.assertTrue(report["duration_completed"])
        self.assertGreaterEqual(report["motion_elapsed_s"], 20)
        self.assertLess(report["motion_elapsed_s"], 20.02)
        self.assertGreater(len(report["commands"]), 1990)
        self.assertFalse(report["measured_wave"]["tracking_verified"])
        self.assertIsNone(report["measured_wave"]["feedback_frequency_hz"])

    def test_watchdog_still_stops_persistent_feedback_loss(self):
        clock = Clock()
        robot = SimpleNamespace(get_joint_limits_enabled=lambda: True, move_js=lambda q: None)
        report = {}
        with self.assertRaisesRegex(RuntimeError, "250 ms"):
            sdk.stream(robot, self.plan, report, clock=clock.now, sleep=clock.sleep,
                       poll=lambda robot, stamps: (None, stamps))
        self.assertFalse(report["duration_completed"])
        self.assertLess(report["motion_elapsed_s"], .3)

    def test_simulated_following_completes_full_wave_with_free_yaw(self):
        clock = Clock()
        q = list(self.plan["start_q_rad"])
        def send(target):
            q[:] = target
        robot = SimpleNamespace(get_joint_limits_enabled=lambda: True, move_js=send)
        report = {}
        sdk.stream(robot, self.plan, report, clock=clock.now, sleep=clock.sleep,
                   poll=lambda robot, stamps: (self.row(q), {"test": clock.now()}))
        self.assertTrue(report["duration_completed"])
        self.assertTrue(report["measured_wave"]["tracking_verified"])
        self.assertAlmostEqual(report["measured_wave"]["feedback_frequency_hz"], self.opts["frequency_hz"], places=2)
        self.assertGreater(report["measured_wave"]["feedback_total_stroke_mm"], 99)

    def test_js_settle_uses_stationarity_even_without_movej_arrival_flag(self):
        clock = Clock()
        row = self.row(self.plan["start_q_rad"])
        session = SimpleNamespace(robot=SimpleNamespace(move_js=lambda q: None))
        recorded = []
        with patch.object(sdk, "poll_feedback", return_value=(row, {})), \
             patch.object(sdk.time, "monotonic", side_effect=clock.now), \
             patch.object(sdk.time, "sleep", side_effect=clock.sleep):
            sdk.settle(session, self.plan["start_q_rad"], recorded)
        self.assertGreaterEqual(len(recorded), 10)

    def test_planned_yaw_lateral_freedom_allowed_plane_loss_rejected(self):
        p = self.plan
        bounds = [(min(s["q_rad"][i] for s in p["samples"]), max(s["q_rad"][i] for s in p["samples"])) for i in range(7)]
        for sample in p["samples"][::20]:
            sdk.check_geometry(self.row(sample["q_rad"]), p, bounds)
        row = self.row(p["start_q_rad"])
        row["fk_flange_pose_m_rad"][2] -= .02
        with self.assertRaisesRegex(RuntimeError, "运动平面"):
            sdk.check_geometry(row, p, bounds)
        row = self.row(p["start_q_rad"])
        row["q_rad"][0] = 10
        with self.assertRaisesRegex(RuntimeError, "J1"):
            sdk.check_geometry(row, p, bounds)

    def test_guard_allows_js_but_never_torque_or_hand_writes(self):
        class Bus:
            def send(self, message):
                return True
        guard = sdk.JSGuard(Bus)
        guard.install()
        guard.permit()
        def send(i, data):
            Bus().send(SimpleNamespace(arbitration_id=i, data=bytes(data), is_extended_id=False))
        try:
            with self.assertRaises(RuntimeError):
                send(0x151, [1, 1, 97, 0xAD, 0, 0, 0, 0])
            guard.motion_allowed = True
            send(0x151, [1, 1, 97, 0xAD, 0, 0, 0, 0])
            send(0x155, [0]*8)
            for ident, data in [(0x151, [1, 1, 97, 0, 0, 0, 0, 0]),
                                (0x151, [1, 4, 97, 0xAD, 0, 0, 0, 0]),
                                (0x1B1, [0]*8), (0x471, [0]*8), (0x151, [1, 1, 100, 0, 0, 0, 0, 0])]:
                with self.assertRaises(RuntimeError):
                    send(ident, data)
        finally:
            guard.restore()

    def test_preview_never_launches_hardware(self):
        with patch.object(cli, "read_json", return_value=self.plan), patch.object(cli.subprocess, "Popen") as p:
            cli.run_command(SimpleNamespace(plan=Path("unused"), execute=False))
            p.assert_not_called()
        offline = dict(self.plan, offline_only=True)
        with patch.object(cli, "read_json", return_value=offline), patch.object(cli.subprocess, "Popen") as p:
            with self.assertRaisesRegex(ValueError, "不可执行"):
                cli.run_command(SimpleNamespace(plan=Path("unused"), execute=True))
            p.assert_not_called()

    def test_execute_dispatches_without_keyboard_confirmation(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            run = Path(temp)
            plan = dict(self.plan, offline_only=False, created_epoch_s=cli.time.time(),
                        input_hashes={}, session_path=temp, config_path='config',
                        feedback_path='feedback', trial_config_path='trial')
            data = {'plan': plan, 'feedback': self.feedback, 'trial': self.opts,
                    'actual.json': {'success': True}}
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(cli, 'read_json', side_effect=lambda p: data[Path(p).name]))
            stack.enter_context(patch.object(cli.common, 'verify_session', return_value=(
                {'T_flange_tcp': self.tcp, 'scene': {}}, None)))
            stack.enter_context(patch.object(cli, 'load_config', return_value={'channel': 'memory', 'speed_percent': 5}))
            stack.enter_context(patch.object(cli, 'make_plan', return_value=self.plan))
            screen = stack.enter_context(patch.object(cli, 'Screen'))
            screen.return_value.check_table_batch.return_value = {'blockers': []}
            stack.enter_context(patch.object(cli.common, 'new_run', return_value=run))
            stack.enter_context(patch.object(cli, 'digest', return_value='test'))
            keyboard = stack.enter_context(patch('builtins.input', side_effect=AssertionError('unexpected prompt')))
            def launch(*args, **kwargs):
                (run / 'actual.json').write_text('{}')
                return SimpleNamespace(wait=lambda **kwargs: 0)
            child = stack.enter_context(patch.object(cli.subprocess, 'Popen', side_effect=launch))
            result = cli.run_command(SimpleNamespace(plan=Path('plan'), execute=True, load='empty'))
            self.assertEqual(result, 0)
            child.assert_called_once()
            keyboard.assert_not_called()
            request = json.loads((run / 'request.json').read_text())
            self.assertTrue(request['execution_authorized'])

    def test_bad_units_and_missing_limits_are_rejected(self):
        for invalid in (float("nan"), True, 6, "5"):
            with self.assertRaises(ValueError):
                planner.options(dict(self.opts, joint_acceleration_cap_rad_s2=invalid))
        bad = deepcopy(self.feedback)
        bad["limits"] = []
        with self.assertRaises(ValueError):
            planner.make_plan(bad, self.opts, self.tcp)

    def test_full_budget_is_scoped_to_planar_and_still_enforces_five(self):
        from cup_grasp_demo.calibration_debug.parameters import shake_options

        full = dict(self.opts, frequency_hz=1.44, limit_utilization=1., auto_reduce_frequency=False)
        self.assertTrue(planner.make_plan(self.feedback, full, self.tcp)["planning_passed"])
        reduced = planner.make_plan(self.feedback, dict(full, limit_utilization=.97), self.tcp)
        self.assertFalse(reduced["planning_passed"])
        over = planner.make_plan(self.feedback, dict(full, frequency_hz=1.45), self.tcp)
        self.assertFalse(over["planning_passed"])
        self.assertGreater(over["joint_peak_acceleration_rad_s2"][0], 5.)
        shared = {"frequency_hz": 1.44, "amplitude_mm": 50, "duration_s": 20,
                  "azimuth_deg": 10, "limit_utilization": 1.}
        with self.assertRaises(ValueError):
            shake_options({"shake": shared})
        self.assertEqual(shake_options({"shake": dict(shared, limit_utilization=.97)})["limit_utilization"], .97)
        for invalid in (0., 1.001, True, "1", float("nan")):
            with self.assertRaises(ValueError):
                planner.options(dict(full, limit_utilization=invalid))
            with self.assertRaises(ValueError):
                sdk.validate({"plan": {"parameters": {"limit_utilization": invalid}}})

    def test_cached_feedback_does_not_wait_for_new_status_and_fault_is_immediate(self):
        stamps = {k: 9.88 for k in ("joint_12", "joint_34", "joint_56", "joint_7")}
        status = SimpleNamespace(timestamp=9.9, msg=SimpleNamespace(arm_status=0, ctrl_mode=1, motion_status=1))
        robot = SimpleNamespace(
            get_arm_status=lambda: status,
            get_driver_states=lambda **kwargs: SimpleNamespace(timestamp=9.9),
            get_joints_enable_status_list=lambda: [True]*7,
            get_joint_angles=lambda: SimpleNamespace(msg=list(self.plan["start_q_rad"])),
            fk=lambda q: self.row(q)["fk_flange_pose_m_rad"],
        )
        with patch.object(sdk, "_packet_timestamps", return_value=stamps):
            row, seen = sdk.poll_feedback(robot, wallclock=lambda: 10)
            self.assertIsNotNone(row)  # A single 120 ms joint interval is tolerated.
            row, _ = sdk.poll_feedback(robot, seen, wallclock=lambda: 10)
            self.assertIsNone(row)  # No blocking wait for another packet.
            with self.assertRaisesRegex(RuntimeError, "250 ms"):
                sdk.poll_feedback(robot, seen, wallclock=lambda: 10.15)
            status.msg.arm_status = 1
            with self.assertRaisesRegex(RuntimeError, "故障"):
                sdk.poll_feedback(robot, seen, wallclock=lambda: 10)


if __name__ == "__main__":
    unittest.main()
