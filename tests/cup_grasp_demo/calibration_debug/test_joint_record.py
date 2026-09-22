"""Optional tracking measurement without stopping the bounded motor experiment."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cup_grasp_demo.calibration_debug.core import ROOT, read_json
from cup_grasp_demo.calibration_debug import joint_profile as profile
from cup_grasp_demo.calibration_debug import joint_test as cli
import test_joint_lab as fixture

EVIDENCE = ROOT / "tests/cup_grasp_demo/calibration_debug/fixtures"


class JointRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = read_json(EVIDENCE / "failed_request.json")["plan"]
        cls.actual = read_json(EVIDENCE / "failed_actual.json")

    def test_omitted_option_keeps_legacy_stop_and_envelope(self):
        cfg = profile.options(dict(tracking_error_deg=8))
        self.assertEqual(cfg["tracking_error_action"], "stop")
        self.assertEqual(cfg["envelope_margin_deg"], 8)
        self.assertEqual(
            profile.options(dict(envelope_margin_deg=3))["envelope_margin_deg"], 3
        )
        for bad in [
            dict(tracking_error_action="ignore"),
            dict(envelope_margin_deg=11),
            dict(envelope_margin_deg=True),
        ]:
            with self.assertRaises(ValueError):
                profile.options(bad)

    def test_recorded_5_13_degree_reversal_continues_only_when_requested(self):
        last = self.actual["last_observation"]
        with self.assertRaisesRegex(RuntimeError, "5.13"):
            profile.feedback_check(last["feedback"], self.plan, last["elapsed_s"])
        p = deepcopy(self.plan)
        p["parameters"]["tracking_error_action"] = "record"
        p["parameters"]["envelope_margin_deg"] = 5
        errors = profile.feedback_check(last["feedback"], p, last["elapsed_s"])
        self.assertEqual(profile.tracking_exceedances(errors, p["parameters"]), [7])
        self.assertAlmostEqual(errors[6], 5.128795407081535)
        # Every recorded sample passes; there is no change to the path or timing.
        for row in self.actual["feedback"]:
            profile.feedback_check(row["feedback"], p, row["elapsed_s"])
        _, segments, duration = profile.trajectory(p["parameters"])
        self.assertEqual(segments, p["segments"])
        self.assertEqual(duration, p["duration_s"])

    def test_full_15_cycles_complete_and_excess_errors_remain_visible(self):
        cfg = dict(
            amplitude_deg=[20],
            velocity_deg_s=[120],
            acceleration_deg_s2=[100],
            cycles=15,
            feedback_reference_delay_s=0.1,
            tracking_error_action="record",
            envelope_margin_deg=5,
        )
        report, plan, _ = fixture.ExecutionTest().simulate(config=cfg, lag_s=0.22)
        self.assertTrue(report["success"], report.get("error"))
        self.assertTrue(report["duration_completed"])
        self.assertAlmostEqual(plan["duration_s"], 39.736186304020364)
        self.assertGreaterEqual(report["reference_send_elapsed_s"], plan["duration_s"])
        self.assertEqual(report["tracking_summary"]["action"], "record")
        self.assertGreater(report["tracking_summary"]["exceeded_samples"], 0)
        self.assertGreater(report["tracking_summary"]["max_abs_error_deg"][6], 5)
        self.assertTrue(any(not r["tracking_check_passed"] for r in report["feedback"]))

    def test_record_mode_retains_motion_envelope_inactive_axes_and_status(self):
        p = deepcopy(self.plan)
        p["parameters"].update(tracking_error_action="record", envelope_margin_deg=5)
        base = deepcopy(self.actual["last_observation"]["feedback"])
        for index, offset in [(6, 25.1), (0, 0.6)]:
            row = deepcopy(base)
            row["q_rad"][index] = p["start_q_rad"][index] + math.radians(offset)
            with self.assertRaisesRegex(RuntimeError, "包络|未选中"):
                profile.feedback_check(row, p, 1.58)
        base["status"]["arm_status"] = 1
        with self.assertRaisesRegex(RuntimeError, "状态异常"):
            profile.feedback_check(base, p, 1.58)

    def test_record_mode_retains_stale_feedback_stop_and_hold(self):
        report, _, _ = fixture.ExecutionTest().simulate(
            config=dict(tracking_error_action="record"), fail_at=0.25
        )
        self.assertFalse(report["duration_completed"])
        self.assertIn("feedback loss", report["error"])
        self.assertTrue(report["failure_hold"]["hold_verified"])

    def test_record_mode_does_not_raise_actuator_limits(self):
        feedback = read_json(fixture.FEEDBACK)
        limits = fixture.shake.Kinematics().model.limits_rad
        cfg = dict(tracking_error_action="record", acceleration_deg_s2=[150])
        p = profile.make_plan(feedback, cfg, limits)
        self.assertFalse(p["planning_passed"])
        self.assertTrue(any("加速度" in b for b in p["blockers"]))

    def test_missing_plan_has_actionable_message_and_never_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with patch.object(cli, "execute") as execute, redirect_stderr(stderr):
                code = cli.main(
                    ["run", "--plan", str(Path(tmp) / "missing.json"), "--execute"]
                )
            self.assertEqual(code, 2)
            execute.assert_not_called()
            self.assertIn("先修正配置并重新 plan", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
