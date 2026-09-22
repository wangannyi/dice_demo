"""Independent feedback, wire allowlist, preview isolation and stop regressions."""

from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from cup_grasp_demo.flow import (
    shake,
    shake_cli,
    shake_execution as sdk,
    shake_tracking as track,
)
from cup_grasp_demo.flow.core import ROOT, read_json
from test_shake import EVIDENCE
from cup_grasp_demo.flow.parameters import shake_options


class ShakeExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = shake.make_plan(
            read_json(EVIDENCE / "session.json"),
            read_json(EVIDENCE / "controller_limits.json"),
            {
                "shake": dict(
                    frequency_hz=0.45,
                    amplitude_mm=50.0,
                    azimuth_deg=10.0,
                    duration_s=5.0,
                )
            },
        )
        cls.plan["table_normal_base"] = read_json(EVIDENCE / "session.json")["scene"][
            "cup_normal_base"
        ]
        cls.plan["offline_only"] = False
        cls.kin = shake.Kinematics()

    def row(self, sample):
        from scipy.spatial.transform import Rotation

        t, _ = self.kin.forward(sample["q_rad"])
        return dict(
            q_rad=sample["q_rad"],
            enabled=[True] * 7,
            status=dict(arm_status=0, ctrl_mode=1, motion_status=1),
            fk_flange_pose_m_rad=[
                *t[:3, 3],
                *Rotation.from_matrix(t[:3, :3]).as_euler("xyz"),
            ],
        )

    def test_exact_feedback_measures_requested_wave_without_command_frequency_substitution(
        self,
    ):
        rows = []
        for s in self.plan["samples"]:
            measured = track.check_feedback(
                self.row(s), self.plan, expected_s=s["displacement_m"]
            )
            rows.append(dict(elapsed_s=s["t_s"], tracking=measured))
        result = track.measured_wave(rows, self.plan)
        self.assertTrue(result["tracking_verified"], result)
        self.assertAlmostEqual(result["feedback_frequency_hz"], 0.45, places=4)
        self.assertGreater(result["feedback_total_stroke_mm"], 99)

    def test_motion_that_did_not_happen_cannot_pass(self):
        rows = [
            dict(
                elapsed_s=s["t_s"], tracking=dict(along_m=0, height_m=0, rotation_deg=0)
            )
            for s in self.plan["samples"]
        ]
        r = track.measured_wave(rows, self.plan)
        self.assertFalse(r["tracking_verified"])
        self.assertIsNone(r["feedback_frequency_hz"])

    def test_late_reader_never_sends_a_catch_up_jump(self):
        q = [0] * 7
        with self.assertRaisesRegex(RuntimeError, "80 ms"):
            track.check_delivery(q, q, 0.181, 0.10)
        with self.assertRaises(RuntimeError):
            track.check_delivery([math.radians(1.01)] + q[1:], q, 0.10, 0.10)
        track.check_delivery([math.radians(0.99)] + q[1:], q, 0.10, 0.10)

    def test_orientation_height_following_and_joint_envelopes(self):
        base = self.row(self.plan["samples"][0])
        for index, delta in [(2, 0.003), (3, math.radians(2))]:
            row = deepcopy(base)
            row["fk_flange_pose_m_rad"][index] += delta
            with self.assertRaises(RuntimeError):
                track.check_feedback(row, self.plan)
        with self.assertRaisesRegex(RuntimeError, "15 mm"):
            track.check_feedback(base, self.plan, expected_s=0.016)
        row = deepcopy(base)
        row["enabled"][0] = False
        with self.assertRaises(RuntimeError):
            track.check_feedback(row, self.plan)
        row = deepcopy(base)
        row["q_rad"][0] += 2
        with self.assertRaises(RuntimeError):
            track.check_feedback(row, self.plan)

    def test_disabled_orientation_records_error_and_preserves_other_guards(self):
        plan = deepcopy(self.plan)
        plan["parameters"]["orientation_error_limit_deg"] = None
        row = self.row(plan["samples"][0])
        row["fk_flange_pose_m_rad"][3] += math.radians(5)
        measured = track.check_feedback(row, plan)
        self.assertAlmostEqual(measured["rotation_deg"], 5, places=6)
        result = track.measured_wave([dict(elapsed_s=0, tracking=measured)], plan)
        self.assertIsNone(result["orientation_error_limit_deg"])
        self.assertFalse(result["orientation_limit_enabled"])
        self.assertAlmostEqual(result["max_orientation_error_deg"], 5, places=6)
        row["fk_flange_pose_m_rad"][2] += 0.003
        with self.assertRaisesRegex(RuntimeError, "离桌高度"):
            track.check_feedback(row, plan)

    def test_configured_orientation_limit_and_legacy_default(self):
        plan = deepcopy(self.plan)
        row = self.row(plan["samples"][0])
        row["fk_flange_pose_m_rad"][3] += math.radians(2)
        with self.assertRaisesRegex(RuntimeError, "朝向偏差"):
            track.check_feedback(row, plan)  # Omitted field retains 1 degree.
        plan["parameters"]["orientation_error_limit_deg"] = 3
        self.assertAlmostEqual(track.check_feedback(row, plan)["rotation_deg"], 2)
        plan["parameters"]["orientation_error_limit_deg"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "0.5°"):
            track.check_feedback(row, plan)

    def test_orientation_configuration_validation_and_plan_propagation(self):
        cfg = {"shake": dict(self.plan["parameters"], orientation_error_limit_deg=None)}
        result = shake.make_plan(
            read_json(EVIDENCE / "session.json"),
            read_json(EVIDENCE / "controller_limits.json"),
            cfg,
        )
        self.assertTrue(result["planning_passed"])
        self.assertIsNone(result["parameters"]["orientation_error_limit_deg"])
        self.assertEqual(result["samples"], self.plan["samples"])
        for invalid in (
            True,
            False,
            0,
            -1,
            181,
            "null",
            "2",
            float("nan"),
            float("inf"),
        ):
            with self.subTest(value=invalid):
                cfg["shake"]["orientation_error_limit_deg"] = invalid
                with self.assertRaises(ValueError):
                    shake_options(cfg)
        for valid in (None, 0.1, 1, 2.5, 180):
            cfg["shake"]["orientation_error_limit_deg"] = valid
            self.assertEqual(shake_options(cfg)["orientation_error_limit_deg"], valid)

    def test_request_auth_offline_and_nonfinite_rejected(self):
        request = dict(
            plan=self.plan,
            execution_authorized=True,
            authorized_epoch_s=10,
            table_screen=dict(blockers=[]),
            scene_observed=True,
            speed_percent=60,
            restore_speed_percent=5,
            input_hashes={},
        )
        sdk.validate_request(request, 20)
        for update in (
            dict(execution_authorized=False),
            dict(authorized_epoch_s=-100),
            dict(scene_observed=False),
        ):
            with self.assertRaises(ValueError):
                sdk.validate_request(dict(request, **update), 20)
        for field, value in [("offline_only", True), ("planning_passed", False)]:
            changed = deepcopy(request)
            changed["plan"][field] = value
            with self.assertRaises(ValueError):
                sdk.validate_request(changed, 20)
        changed = deepcopy(request)
        changed["plan"]["samples"][0]["q_rad"][0] = float("nan")
        with self.assertRaises(ValueError):
            sdk.validate_request(changed, 20)

    def test_configured_twenty_second_wave_is_accepted_without_truncation(self):
        cfg = {'shake': dict(self.plan['parameters'], duration_s=20)}
        plan = shake.make_plan(read_json(EVIDENCE / 'session.json'),
                               read_json(EVIDENCE / 'controller_limits.json'), cfg)
        self.assertTrue(plan['planning_passed'], plan['blockers'])
        plan['offline_only'] = False
        request = dict(plan=plan, execution_authorized=True, authorized_epoch_s=10,
                       table_screen=dict(blockers=[]), scene_observed=True,
                       speed_percent=60, restore_speed_percent=5, input_hashes={})
        sdk.validate_request(request, 20)
        self.assertEqual(len(plan['samples']), 2001)
        self.assertEqual(plan['samples'][-1]['t_s'], 20)
        plan['parameters']['duration_s'] = 61
        with self.assertRaises(ValueError):
            sdk.validate_request(request, 20)

    def test_preview_has_no_hardware_or_camera_access(self):
        with (
            patch.object(shake_cli, "read_json", return_value=self.plan),
            patch.object(shake_cli, "summarize"),
            patch.object(shake_cli.common, "capture_rgbd") as capture,
            patch.object(shake_cli.subprocess, "Popen") as spawn,
        ):
            self.assertEqual(
                shake_cli.execute(SimpleNamespace(plan=Path("unused"), execute=False)),
                0,
            )
            capture.assert_not_called()
            spawn.assert_not_called()

    def test_can_guard_forbids_enable_hand_and_wrong_motion_mode(self):
        class Bus:
            def send(self, message):
                return True

        original = Bus.send
        guard = sdk.ShakeGuard(Bus)
        guard.install()
        guard.permit()

        def send(i, data):
            Bus().send(
                SimpleNamespace(
                    arbitration_id=i, data=bytes(data), is_extended_id=False
                )
            )

        try:
            send(0x472, [1, 1, 0, 0, 0, 0, 0, 0])
            with self.assertRaises(RuntimeError):
                send(0x155, [0] * 8)
            guard.motion_allowed = True
            send(0x151, [1, 1, 60, 0, 0, 0, 0, 0])
            send(0x155, [0] * 8)
            for i, data in [
                (0x1B1, [0] * 8),
                (0x471, [0] * 8),
                (0x151, [1, 4, 60, 0, 0, 0, 0, 0]),
                (0x151, [1, 1, 100, 0, 0, 0, 0, 0]),
            ]:
                with self.assertRaises(RuntimeError):
                    send(i, data)
        finally:
            guard.restore()
        self.assertIs(Bus.send, original)

    def test_stale_fingers_or_large_hand_faults_block(self):
        fresh = dict(
            fresh=True, values={k: 100 for k in sdk.FINGERS}, left_or_right=None
        )
        status = dict(fresh=False)
        with patch.object(
            sdk, "_copy_getter", side_effect=[dict(fresh=False), {}, status]
        ):
            with self.assertRaisesRegex(RuntimeError, "不新鲜"):
                sdk.hand_feedback(None, None)
        with patch.object(
            sdk,
            "_copy_getter",
            side_effect=[
                fresh,
                {},
                dict(fresh=True, left_or_right=2, values={"motor": 3}),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "异常"):
                sdk.hand_feedback(None, None)

    def test_only_this_verified_camera_process_is_excluded(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "123").mkdir()
            argv = [
                "/python",
                str(Path(sdk.__file__).with_name("shake_camera.py")),
                "--serial",
                "camera",
                "--output",
                "/output",
            ]
            (root / "123/cmdline").write_bytes(
                b"\0".join(v.encode() for v in argv) + b"\0"
            )
            evidence = {
                "candidate_control_processes": [{"pid": 123}, {"pid": 456}],
                "receiver_rows": [{"keep": "unchanged"}],
            }
            with patch.object(
                sdk.core, "host_control_evidence", return_value=deepcopy(evidence)
            ):
                result = sdk.control_evidence(
                    "can0",
                    {"camera_process": {"pid": 123, "argv": argv}},
                    proc_root=root,
                )
            self.assertEqual(result["candidate_control_processes"], [{"pid": 456}])
            self.assertEqual(result["receiver_rows"], evidence["receiver_rows"])
            (root / "123/cmdline").write_bytes(b"/python\0other_controller.py\0")
            with patch.object(
                sdk.core, "host_control_evidence", return_value=deepcopy(evidence)
            ):
                with self.assertRaises(RuntimeError):
                    sdk.control_evidence(
                        "can0",
                        {"camera_process": {"pid": 123, "argv": argv}},
                        proc_root=root,
                    )

    def test_failure_after_control_attempts_fresh_hold_and_disconnect(self):
        # Fail on the first mode/speed operation, after read-only preflight succeeds.
        request = dict(
            plan=self.plan,
            channel="can0",
            execution_authorized=True,
            authorized_epoch_s=0,
            table_screen=dict(blockers=[]),
            scene_observed=True,
            speed_percent=60,
            restore_speed_percent=5,
            input_hashes={},
        )
        from unittest.mock import MagicMock

        session = MagicMock()
        robot = session.robot

        def fk(q):
            return self.row({"q_rad": q})["fk_flange_pose_m_rad"]

        robot.fk.side_effect = fk
        robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(
            msg=SimpleNamespace(max_joint_spd=10, min_angle_limit=-4, max_angle_limit=4)
        )
        robot.get_joint_acc_limits.return_value = SimpleNamespace(
            msg=SimpleNamespace(max_joint_acc=10)
        )
        robot.get_flange_vel_acc_limits.return_value = SimpleNamespace(
            msg=SimpleNamespace(end_max_linear_vel=10, end_max_linear_acc=10)
        )
        robot.set_speed_percent.side_effect = RuntimeError("injected speed error")
        row = self.row(self.plan["samples"][0])
        row["status"]["motion_status"] = 0
        with (
            patch.object(sdk, "validate_request"),
            patch.object(sdk.core, "load_sdk_runtime", return_value=(object, object)),
            patch.object(sdk, "ShakeGuard", return_value=MagicMock()),
            patch.object(sdk.core, "PassivePoseSession", return_value=session),
            patch.object(sdk.core, "stopped_window", return_value=([row], {})),
            patch.object(sdk.core, "joint_limits", return_value=[[-4, 4]] * 7),
            patch.object(sdk.core, "host_control_evidence"),
            patch.object(sdk.core, "evidence_blockers", return_value=[]),
            patch.object(
                sdk,
                "hand_feedback",
                return_value={"positions_0_100": [17, 99, 17, 27, 25, 17]},
            ),
            patch.object(
                sdk, "fresh_js_hold", return_value={"target_rad": row["q_rad"]}
            ) as hold,
            patch.object(sdk, "verify_stop") as stop,
        ):
            report = sdk.run(request)
        self.assertFalse(report["success"])
        hold.assert_called_once()
        stop.assert_called_once()
        session.close.assert_called_once()
        robot.move_js.assert_not_called()
        self.assertTrue(report["failure_hold"]["hold_verified"])


if __name__ == "__main__":
    unittest.main()
