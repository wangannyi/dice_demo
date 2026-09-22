"""Joint profiles, constraint Jacobians, live guards, and old shake defaults."""

from contextlib import ExitStack
from copy import deepcopy
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from cup_grasp_demo.flow.core import ROOT, read_json
from cup_grasp_demo.flow import joint_profile as profile
from cup_grasp_demo.flow import shake, shake_study
from cup_grasp_demo.flow.parameters import shake_options

FEEDBACK = (
    ROOT / "tests/cup_grasp_demo/flow/fixtures/controller_limits.json"
)


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.feedback = read_json(FEEDBACK)
        self.model = shake.Kinematics().model
        self.plan = profile.make_plan(self.feedback, {}, self.model.limits_rad)

    def test_full_limit_utilization_retains_controller_limits(self):
        self.assertEqual(profile.options({})['limit_utilization'], .97)
        self.assertEqual(profile.options({'limit_utilization': 1})['limit_utilization'], 1)
        for value in (1.01, 0, True):
            with self.assertRaises(ValueError):
                profile.options({'limit_utilization': value})
        p = profile.make_plan(self.feedback, dict(limit_utilization=1, acceleration_deg_s2=[150]), self.model.limits_rad)
        self.assertFalse(p['planning_passed'])
        self.assertTrue(any('加速度' in b for b in p['blockers']))

    def test_duration_angles_and_inactive_joints(self):
        plan = self.plan
        self.assertTrue(plan["planning_passed"], plan["blockers"])
        self.assertEqual(plan["samples"][0]["q_rad"], plan["start_q_rad"])
        self.assertEqual(plan["samples"][-1]["q_rad"], plan["start_q_rad"])
        self.assertEqual(plan["segments"][-1]["end"], 0)
        for row in plan["samples"]:
            self.assertEqual(row["q_rad"][:6], plan["start_q_rad"][:6])
        self.assertAlmostEqual(plan["cycle_period_s"], 5 / 3)
        self.assertAlmostEqual(plan["reference_frequency_hz"], 0.6)

    def test_triangular_profile_cannot_reach_requested_cruise_speed(self):
        plan = profile.make_plan(
            self.feedback, dict(velocity_deg_s=[170]), self.model.limits_rad
        )
        self.assertTrue(plan["triangular_profile"])
        self.assertLess(math.degrees(plan["joint_peak_velocity_rad_s"][6]), 170)
        for segment in plan["segments"]:
            for t in (0, segment["duration_s"]):
                self.assertAlmostEqual(profile.value(segment, t)[1], 0)

    def test_live_limit_budget_and_limits_are_enforced(self):
        for cfg in (
            dict(acceleration_deg_s2=[150]),
            dict(joints=[2], amplitude_deg=[15]),
        ):
            p = profile.make_plan(self.feedback, cfg, self.model.limits_rad)
            self.assertFalse(p["planning_passed"])
        p = profile.make_plan(
            self.feedback, dict(acceleration_deg_s2=[138]), self.model.limits_rad
        )
        self.assertTrue(p["planning_passed"], p["blockers"])
        self.assertFalse(
            profile.make_plan(
                self.feedback, dict(acceleration_deg_s2=[140]), self.model.limits_rad
            )["planning_passed"]
        )

    def test_invalid_parameters_fail(self):
        for cfg in (
            dict(joints=[1, 4]),
            dict(joints=[]),
            dict(joints=[1, 1], amplitude_deg=[2, 2], velocity_deg_s=[10, 10],
                 acceleration_deg_s2=[20, 20]),
            dict(joints=[True]),
            dict(joints=[8]),
            dict(cycles=1.5),
            dict(limit_utilization=1.001),
            dict(amplitude_deg=[float("nan")]),
            dict(velocity_deg_s=[0]),
            dict(controller_speed_percent=101),
            dict(foo=1),
        ):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                profile.options(cfg)

    def test_arbitrary_joint_sets_have_individual_signed_amplitudes(self):
        for joints in ([4, 1], [7, 3, 1, 5], list(range(1, 8))):
            amplitudes = [(-1 if i % 2 else 1) * (i + 1) * .2 for i in range(len(joints))]
            cfg = dict(joints=joints, amplitude_deg=amplitudes,
                       velocity_deg_s=[10] * len(joints),
                       acceleration_deg_s2=[40] * len(joints))
            plan = profile.make_plan(self.feedback, cfg, self.model.limits_rad)
            self.assertTrue(plan["planning_passed"], plan["blockers"])
            for sample in plan["samples"][::17]:
                u, _, _ = profile.at(plan["segments"], sample["t_s"])
                for j in range(1, 8):
                    expected = math.radians(amplitudes[joints.index(j)]) * u if j in joints else 0
                    self.assertAlmostEqual(sample["q_rad"][j-1] - plan["start_q_rad"][j-1], expected)
            self.assertEqual(plan["samples"][-1]["q_rad"], plan["start_q_rad"])

    def test_j147_keeps_other_axes_fixed_and_synchronizes_phase(self):
        cfg = dict(
            joints=[1, 4, 7],
            amplitude_deg=[2, -3, 5],
            velocity_deg_s=[10] * 3,
            acceleration_deg_s2=[40] * 3,
        )
        p = profile.make_plan(self.feedback, cfg, self.model.limits_rad)
        self.assertTrue(p["planning_passed"])
        for row in p["samples"][::17]:
            delta = np.asarray(row["q_rad"]) - p["start_q_rad"]
            np.testing.assert_allclose(delta[[1, 2, 4, 5]], 0, atol=1e-15)
            np.testing.assert_allclose(
                delta[[0, 3, 6]] / np.radians([2, -3, 5]),
                delta[0] / math.radians(2),
                atol=1e-14,
            )

    def test_feedback_fault_and_inactive_axis_motion_stop(self):
        row = dict(
            q_rad=self.plan["start_q_rad"].copy(),
            enabled=[True] * 7,
            status=dict(arm_status=0, ctrl_mode=1, motion_status=1),
        )
        profile.feedback_check(row, self.plan, 0)
        row["q_rad"][0] += math.radians(0.6)
        with self.assertRaisesRegex(RuntimeError, "J1"):
            profile.feedback_check(row, self.plan, 0)
        row["q_rad"] = row["q_rad"][:6]
        with self.assertRaises(RuntimeError):
            profile.feedback_check(row, self.plan, 0)

    def test_measurement_detects_reduced_actual_stroke(self):
        rows = [
            dict(elapsed_s=s["t_s"], feedback=dict(q_rad=s["q_rad"]))
            for s in self.plan["samples"]
        ]
        m = profile.measurements(rows, self.plan)
        self.assertTrue(m["tracking_verified"])
        for row in rows:
            q = row["feedback"]["q_rad"].copy()
            q[6] = self.plan["start_q_rad"][6] + 0.5 * (
                q[6] - self.plan["start_q_rad"][6]
            )
            row["feedback"]["q_rad"] = q
        self.assertFalse(profile.measurements(rows, self.plan)["tracking_verified"])

    def test_shake_80_percent_legacy_and_97_percent_option(self):
        cfg = {
            "shake": dict(
                frequency_hz=0.55, amplitude_mm=50, azimuth_deg=10, duration_s=20
            )
        }
        session = read_json(FEEDBACK.with_name("session.json"))
        a = shake.make_plan(session, self.feedback, cfg)
        cfg["shake"]["limit_utilization"] = 0.97
        b = shake.make_plan(session, self.feedback, cfg)
        self.assertEqual(a["configured_limit_utilization"], 0.8)
        self.assertEqual(b["configured_limit_utilization"], 0.97)
        self.assertEqual(a["samples"], b["samples"])
        self.assertFalse(a["planning_passed"])
        self.assertTrue(b["planning_passed"], b["blockers"])
        cfg["shake"]["limit_utilization"] = 1
        with self.assertRaises(ValueError):
            shake_options(cfg)


class CLITest(unittest.TestCase):
    def test_motor_plan_does_not_require_cup_capture_success(self):
        from cup_grasp_demo.flow import joint_test as cli

        scene = read_json(FEEDBACK.with_name("session.json"))["scene"]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / ".capture_pending").write_text("cup detection failed")
            # Regression fixture must not inherit the operator's live motor settings.
            fixture_config = directory / 'joint_config.json'
            fixture_config.write_text('{}')
            args = SimpleNamespace(
                session=directory,
                output=None,
                feedback_json=None,
                system_config=ROOT
                / "configs/green_cup.json",
                config=fixture_config,
            )
            with (
                patch.object(cli.common, "capture_rgbd") as capture,
                patch.object(
                    cli.common,
                    "verify_session",
                    side_effect=AssertionError("cup dependency"),
                ),
                patch(
                    "cup_grasp_demo.flow.pipeline_home.home_scene",
                    return_value=(scene, False),
                ),
                patch.object(cli, "readback", return_value=FEEDBACK),
            ):
                self.assertEqual(cli.design(args), 0)
                self.assertEqual(capture.call_count, 1)
            plan = read_json(directory / "joint_plan.json")
            self.assertFalse(plan["offline_only"])
            self.assertIn(plan["scene_path"], plan["input_hashes"])
            self.assertTrue((directory / ".capture_pending").exists())

    def test_combined_plan_run_stops_on_failed_plan(self):
        from cup_grasp_demo.flow import joint_test as cli
        for code in (0, 2):
            with tempfile.TemporaryDirectory() as temp, self.subTest(code=code):
                with patch.object(cli, 'design', return_value=code), \
                     patch.object(cli, 'execute', return_value=0) as execute:
                    result = cli.main(['plan', '--session', temp, '--execute'])
                    self.assertEqual(result, code)
                    self.assertEqual(execute.call_count, int(code == 0))

    def test_table_scene_reuse_avoids_camera_and_direct_run_never_prompts(self):
        from cup_grasp_demo.flow import joint_test as cli
        scene = read_json(FEEDBACK.with_name('session.json'))['scene']
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / 'joint.json'; config.write_text('{}')
            table = root / 'table.json'; table.write_text('{}')
            args = SimpleNamespace(session=root, output=None, feedback_json=None,
                system_config=ROOT / 'configs/green_cup.json',
                config=config, table_scene=table)
            record = dict(scene=scene, calibration_quality_passed=False, input_hashes={})
            with patch.object(cli.planar_scene, 'verify', return_value=(record, {})), \
                 patch.object(cli.common, 'capture_rgbd', side_effect=AssertionError('camera opened')), \
                 patch.object(cli, 'readback', return_value=FEEDBACK):
                self.assertEqual(cli.design(args), 0)
                def launch(argv, **kwargs):
                    Path(argv[argv.index('--output')+1]).write_text('{"success": true}')
                    return SimpleNamespace(wait=lambda **kwargs: 0)
                with patch.object(cli.subprocess, 'Popen', side_effect=launch) as child, \
                     patch('builtins.input', side_effect=AssertionError('unexpected prompt')):
                    self.assertEqual(cli.execute(SimpleNamespace(plan=root/'joint_plan.json', execute=True)), 0)
                    child.assert_called_once()

    def test_comparison_can_replace_same_output(self):
        from cup_grasp_demo.flow import joint_test as cli

        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(
                session=Path(temp),
                output=None,
                feedback_json=FEEDBACK,
                system_config=ROOT
                / "configs/green_cup.json",
            )
            with patch.object(
                cli,
                "compare",
                return_value={
                    "kind": "shake_constraint_study",
                    "execution_enabled": False,
                },
            ):
                self.assertEqual(cli.study(args), 0)
                self.assertEqual(cli.study(args), 0)
            self.assertTrue((args.session / "shake_study.json").exists())


class ConstraintTest(unittest.TestCase):
    def test_tcp_euler_jacobian_matches_finite_difference(self):
        kin = shake.Kinematics()
        q = read_json(FEEDBACK)["q_after_rad"]
        tcp = np.eye(4)
        tcp[:3, 3] = [0.17, 0.006, 0.011]
        p, jac, rpy = shake_study.tcp_state(kin, q, tcp)
        eps = 1e-7
        for i in range(7):
            changed = q.copy()
            changed[i] += eps
            p1, _, r1 = shake_study.tcp_state(kin, changed, tcp)
            np.testing.assert_allclose(
                np.r_[(p1[:3, 3] - p[:3, 3]) / eps, shake_study.wrap(r1 - rpy) / eps],
                jac[:, i],
                atol=2e-7,
            )

    def test_primary_j147_uses_auxiliary_joints_without_locking_them(self):
        kin = shake.Kinematics()
        q = read_json(FEEDBACK)["q_after_rad"]
        tcp = np.eye(4)
        tcp[:3, 3] = [0.17, 0.006, 0.011]
        pose, _, rpy = shake_study.tcp_state(kin, q, tcp)
        direction = np.array([math.cos(0.17), math.sin(0.17), 0.0])
        cost = shake_study.motion_cost({})
        free = shake_study.solve_planar(
            kin, q, tcp, pose, rpy, direction, 0.01, list(range(7))
        )
        weighted = shake_study.solve_planar(
            kin, q, tcp, pose, rpy, direction, 0.01, list(range(7)), cost
        )
        aux = np.asarray([1, 2, 4, 5])
        free_motion = np.linalg.norm((free - q)[aux])
        auxiliary_motion = np.linalg.norm((weighted - q)[aux])
        self.assertGreater(auxiliary_motion, 1e-6)
        self.assertLess(auxiliary_motion, free_motion)
        actual, _, angles = shake_study.tcp_state(kin, weighted, tcp)
        self.assertAlmostEqual(actual[2, 3], pose[2, 3], places=7)
        np.testing.assert_allclose(shake_study.wrap(angles[:2] - rpy[:2]), 0, atol=1e-7)
        self.assertAlmostEqual(
            np.dot(direction, actual[:3, 3] - pose[:3, 3]), 0.01, places=7
        )
        for bad in ([1] * 6, [0] * 7, [True] * 7, [float("nan")] * 7):
            with self.assertRaises(ValueError):
                shake_study.motion_cost({"shake_study": {"joint_motion_cost": bad}})

    def test_relaxed_constraints_are_met_and_j147_can_reduce_to_j1(self):
        kin = shake.Kinematics()
        q = read_json(FEEDBACK)["q_after_rad"]
        tcp = np.eye(4)
        tcp[:3, 3] = [0.17, 0.006, 0.011]
        p, _, rpy = shake_study.tcp_state(kin, q, tcp)
        direction = np.array([math.cos(0.17), math.sin(0.17), 0.0])
        for selected in (list(range(7)), [0, 3, 6]):
            solved = shake_study.solve_planar(
                kin, q, tcp, p, rpy, direction, 0.01, selected
            )
            p1, _, r1 = shake_study.tcp_state(kin, solved, tcp)
            np.testing.assert_allclose(p1[2, 3], p[2, 3], atol=1e-8)
            np.testing.assert_allclose(shake_study.wrap(r1[:2] - rpy[:2]), 0, atol=1e-7)
            self.assertAlmostEqual(
                np.dot(direction, p1[:3, 3] - p[:3, 3]), 0.01, places=7
            )
            if selected == [0, 3, 6]:
                np.testing.assert_allclose(solved[1:], q[1:], atol=1e-7)


class ExecutionTest(unittest.TestCase):
    def test_two_joint_executor_completes_and_keeps_other_axes_fixed(self):
        report, plan, _ = self.simulate(config=dict(
            joints=[4, 1], amplitude_deg=[-3, 2], velocity_deg_s=[15, 10],
            acceleration_deg_s2=[30, 20]))
        self.assertTrue(report["success"], report.get("error"))
        self.assertTrue(report["duration_completed"])
        for row in report["feedback"]:
            for i in (1, 2, 4, 5, 6):
                self.assertEqual(row["feedback"]["q_rad"][i], plan["start_q_rad"][i])

    def simulate(
        self, fail_at=None, config=None, lag_s=None, tracking_fault=False,
        host_evidence=None, host_conflicts=(), connected=False,
    ):
        from cup_grasp_demo.flow import joint_execution as sdk

        feedback = read_json(FEEDBACK)
        model = shake.Kinematics()
        plan = profile.make_plan(feedback, config or {}, model.model.limits_rad)
        plan.update(offline_only=False, model_flange_checks=[])
        request = dict(
            plan=plan,
            execution_authorized=True,
            authorized_epoch_s=1000,
            empty_hand_clear_path_confirmed=True,
            channel="can0",
            restore_speed_percent=5,
            table_screen=dict(blockers=[]),
            input_hashes={},
        )
        clock = [0.0]
        current = [plan["start_q_rad"].copy()]
        robot = MagicMock()
        robot.get_joint_limits_enabled.return_value = True
        history = [(0.0, current[0].copy())]

        def move(q):
            history.append((clock[0], list(q)))
            if lag_s is None:
                current[0] = list(q)

        robot.move_js.side_effect = move
        session = MagicMock(robot=robot)

        def row():
            t = model.forward(current[0])[0]
            return dict(
                q_rad=current[0],
                enabled=[True] * 7,
                status=dict(arm_status=0, ctrl_mode=1, motion_status=int(robot.move_js.call_count > 0)),
                fk_flange_pose_m_rad=[
                    *t[:3, 3],
                    *Rotation.from_matrix(t[:3, :3]).as_euler("xyz"),
                ],
            )

        def fresh(*args, **kwargs):
            clock[0] += 0.008 if lag_s is not None else 0.04
            if lag_s is not None:
                available = [q for t, q in history if t <= clock[0] - lag_s]
                current[0] = (available[-1] if available else history[0][1]).copy()
                if tracking_fault and clock[0] >= 0.5:
                    current[0][6] += math.radians(7.0)
            if fail_at and clock[0] >= fail_at:
                raise TimeoutError("simulated feedback loss")
            return row()

        timer = SimpleNamespace(
            monotonic=lambda: clock[0],
            time=lambda: 1000 + clock[0],
            sleep=lambda t: clock.__setitem__(0, clock[0] + t),
        )
        with ExitStack() as stack:
            for owner, name, value in [
                (sdk, "time", timer),
                (sdk, "JointGuard", MagicMock()),
                (sdk, "live_plan", MagicMock(return_value=feedback["limits"])),
                (sdk.shared, "verify_stop", MagicMock()),
                (
                    sdk.shared, "control_evidence",
                    MagicMock(return_value=host_evidence or {}),
                ),
                (
                    sdk.core,
                    "load_sdk_runtime",
                    MagicMock(return_value=(object(), object())),
                ),
                (sdk.core, "PassivePoseSession", MagicMock(return_value=session)),
                (sdk.core, "stopped_window", MagicMock(return_value=([row()], {}))),
                (
                    sdk.core,
                    "joint_limits",
                    MagicMock(return_value=model.model.limits_rad),
                ),
                (sdk.core, "ready_blockers", MagicMock(return_value=[])),
                (
                    sdk.core, "evidence_blockers",
                    MagicMock(return_value=list(host_conflicts)),
                ),
                (
                    sdk.core,
                    "take_can_control",
                    MagicMock(return_value={"samples": [row()]}),
                ),
                (sdk.core, "fresh_feedback", MagicMock(side_effect=fresh)),
                (
                    sdk,
                    "fresh_js_hold",
                    MagicMock(return_value={"target_rad": current[0]}),
                ),
            ]:
                stack.enter_context(patch.object(owner, name, value))
            if connected:
                request['load_context'] = 'green_cup_held'
                from cup_grasp_demo.flow import green_shake_validation
                stack.enter_context(patch.object(green_shake_validation, 'validate_held_request'))
            report = sdk.run(request, **({'connected': robot, 'connection_evidence': {}} if connected else {}))
            if connected:
                session.start.assert_not_called()
                session.close.assert_not_called()
        return report, plan, robot

    def test_full_duration_no_hand_or_controller_parameter_writes(self):
        report, plan, robot = self.simulate()
        self.assertTrue(report["success"], report.get("error"))
        self.assertTrue(report["duration_completed"])
        self.assertGreaterEqual(report["reference_send_elapsed_s"], plan["duration_s"])
        self.assertLess(report["reference_send_elapsed_s"], plan["duration_s"] + 0.06)
        self.assertTrue(report["measurement"]["tracking_verified"])
        robot.set_joint_acc_limits.assert_not_called()
        robot.set_joint_angle_vel_limits.assert_not_called()
        robot.init_effector.assert_not_called()
        self.assertEqual(robot.set_speed_percent.call_args_list[-1].args, (5,))

    def test_fault_stops_without_completing_or_retrying(self):
        report, _, _ = self.simulate(1.0)
        self.assertFalse(report["success"])
        self.assertFalse(report["duration_completed"])
        self.assertIn("feedback loss", report["error"])
        self.assertTrue(report["failure_hold"]["hold_verified"])

    def test_wire_budget_isolated_and_no_firmware_parameter_writes(self):
        from cup_grasp_demo.flow import joint_execution as sdk

        for cls, maximum in ((sdk.shared.ShakeGuard, 60), (sdk.JointGuard, 100)):

            class Bus:
                def send(self, message):
                    return True

            guard = cls(Bus)
            guard.install()
            guard.permit()
            guard.motion_allowed = True

            def send(ident, data):
                Bus().send(
                    SimpleNamespace(
                        arbitration_id=ident, data=bytes(data), is_extended_id=False
                    )
                )

            try:
                send(0x151, [1, 1, maximum, 0, 0, 0, 0, 0])
                for ident, data in [
                    (0x151, [1, 1, maximum + 1, 0, 0, 0, 0, 0]),
                    (0x471, [0] * 8),
                    (0x475, [0] * 8),
                    (0x1B1, [0] * 8),
                    (0x151, [1, 4, maximum, 0, 0, 0, 0, 0]),
                ]:
                    with self.assertRaises(RuntimeError):
                        send(ident, data)
            finally:
                guard.restore()

    def test_live_limits_and_changed_reference_rejected_before_motion(self):
        from cup_grasp_demo.flow import joint_execution as sdk

        feedback = read_json(FEEDBACK)
        plan = profile.make_plan(feedback, {}, shake.Kinematics().model.limits_rad)
        robot = MagicMock()
        robot.get_joint_angle_vel_limits.side_effect = lambda j, **kw: SimpleNamespace(
            msg=SimpleNamespace(
                min_angle_limit=feedback["limits"][j - 1]["min_angle_rad"],
                max_angle_limit=feedback["limits"][j - 1]["max_angle_rad"],
                max_joint_spd=feedback["limits"][j - 1]["max_velocity_rad_s"],
            )
        )
        robot.get_joint_acc_limits.side_effect = lambda j, **kw: SimpleNamespace(
            msg=SimpleNamespace(
                max_joint_acc=feedback["limits"][j - 1]["max_acceleration_rad_s2"]
            )
        )
        with patch.object(
            sdk.core, "joint_limits", return_value=shake.Kinematics().model.limits_rad
        ):
            sdk.live_plan(robot, plan)
            changed = deepcopy(plan)
            changed["samples"][50]["q_rad"][6] += 0.001
            with self.assertRaises(ValueError):
                sdk.live_plan(robot, changed)
            robot.get_joint_acc_limits.side_effect = lambda j, **kw: SimpleNamespace(
                msg=SimpleNamespace(max_joint_acc=0.1)
            )
            with self.assertRaises(ValueError):
                sdk.live_plan(robot, plan)
        robot.move_js.assert_not_called()

    def test_stale_or_offline_request_rejected(self):
        from cup_grasp_demo.flow import joint_execution as sdk

        p = profile.make_plan(
            read_json(FEEDBACK), {}, shake.Kinematics().model.limits_rad
        )
        request = dict(
            plan=p,
            execution_authorized=True,
            authorized_epoch_s=1000,
            empty_hand_clear_path_confirmed=True,
            table_screen=dict(blockers=[]),
            restore_speed_percent=5,
            input_hashes={},
        )
        sdk.validate_request(request, 1000)
        with self.assertRaises(ValueError):
            sdk.validate_request(request, 1061)
        p["offline_only"] = True
        with self.assertRaises(ValueError):
            sdk.validate_request(request, 1000)


if __name__ == "__main__":
    unittest.main()
