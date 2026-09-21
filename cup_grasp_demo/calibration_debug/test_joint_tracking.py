"""Recorded fast-joint lag, clock jitter, and complete delayed executor regression."""

from copy import deepcopy
import math
import unittest

from cup_grasp_demo.calibration_debug.core import ROOT, read_json
from cup_grasp_demo.calibration_debug import joint_profile as profile
from cup_grasp_demo.calibration_debug import test_joint_lab as fixture

EVIDENCE = ROOT / "cup_grasp_demo/datasets/joint_tracking_20260918"


class TrackingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = read_json(EVIDENCE / "actual.json")
        cls.plan = read_json(EVIDENCE / "request.json")["plan"]

    def test_recorded_lag_alignment_without_changing_five_degree_threshold(self):
        plan = deepcopy(self.plan)
        plan["parameters"]["feedback_reference_delay_s"] = 0.1
        errors = [
            profile.feedback_check(r["feedback"], plan, r["elapsed_s"])[6]
            for r in self.report["feedback"]
        ]
        self.assertLess(max(map(abs, errors)), 0.72)
        row = self.report["feedback"][-1]
        self.assertGreater(
            abs(profile.reference_errors(row["feedback"], plan, row["elapsed_s"])[6]),
            4.7,
        )
        self.assertEqual(plan["parameters"]["tracking_error_deg"], 5.0)
        # Delay affects monitoring only, not targets, duration, v/a or amplitude.
        _, segments, duration = profile.trajectory(plan["parameters"])
        self.assertEqual(segments, plan["segments"])
        self.assertEqual(duration, plan["duration_s"])

    def test_delay_never_disables_envelope_or_inactive_axis_checks(self):
        plan = deepcopy(self.plan)
        plan["parameters"]["feedback_reference_delay_s"] = 0.1
        row = deepcopy(self.report["feedback"][-1]["feedback"])
        for index, delta in [(0, math.radians(0.6)), (6, math.radians(30))]:
            bad = deepcopy(row)
            bad["q_rad"][index] += delta
            with self.assertRaisesRegex(RuntimeError, "包络|未选中"):
                profile.feedback_check(bad, plan, 0.53)
        row["status"]["arm_status"] = 1
        with self.assertRaisesRegex(RuntimeError, "状态异常"):
            profile.feedback_check(row, plan, 0.53)

    def test_delay_is_bounded_and_legacy_zero(self):
        self.assertEqual(profile.options({})["feedback_reference_delay_s"], 0.0)
        for bad in [-1, 0.201, True, float("nan")]:
            with self.assertRaises(ValueError):
                profile.options(dict(feedback_reference_delay_s=bad))
        with self.assertRaisesRegex(ValueError, "10%"):
            profile.trajectory(
                dict(
                    amplitude_deg=[0.1],
                    acceleration_deg_s2=[100],
                    feedback_reference_delay_s=0.1,
                )
            )

    def test_derivatives_reject_duplicate_packets_and_handle_observer_jitter(self):
        rows = []
        for i in range(101):
            t = i * 0.01
            age = 0.001 + (i % 3) * 0.003
            q = [0.0] * 6 + [math.radians(round(0.5 * 100 * t * t, 3))]
            row = dict(
                q_rad=q,
                observed_epoch_s=1000 + t + age,
                sdk_snapshot=dict(
                    packet_timestamps_after_epoch_s={"joint_7": 1000 + t}
                ),
            )
            rows.append(dict(elapsed_s=t + age, feedback=row))
            duplicate = deepcopy(rows[-1])
            duplicate["elapsed_s"] += 0.0002
            duplicate["feedback"]["observed_epoch_s"] += 0.0002
            rows.append(duplicate)
        samples, sources = profile.packet_samples(rows, 7)
        self.assertEqual(len(samples), 101)
        self.assertEqual(sources, ["joint_packet_receipt"])
        for t, velocity, acceleration in profile.window_derivatives(samples, 0.12):
            self.assertAlmostEqual(velocity, 100 * t, places=5)
            self.assertAlmostEqual(acceleration, 100, places=4)
        self.assertEqual(profile.window_derivatives(samples[:4], 0.12), [])

    def test_short_cycle_record_preserves_delay_path_and_controller_limits(self):
        cfg = dict(joints=[1], amplitude_deg=[2], velocity_deg_s=[120],
                   acceleration_deg_s2=[120], cycles=15,
                   feedback_reference_delay_s=0.1, tracking_error_action="record")
        parsed, segments, duration = profile.trajectory(cfg)
        self.assertAlmostEqual(2 * segments[1]["duration_s"], 0.7302967433402214)
        self.assertEqual(parsed["feedback_reference_delay_s"], 0.1)
        _, original, original_duration = profile.trajectory(dict(cfg, feedback_reference_delay_s=0))
        self.assertEqual(segments, original)
        self.assertEqual(duration, original_duration)
        feedback = read_json(ROOT / "cup_grasp_demo/datasets/shake_assessment_20260918/controller_limits.json")
        from cup_grasp_demo.calibration_debug.shake import Kinematics
        plan = profile.make_plan(feedback, cfg, Kinematics().model.limits_rad)
        self.assertTrue(plan["warnings"])
        self.assertTrue(any("J1 加速度" in b for b in plan["blockers"]))
        with self.assertRaisesRegex(ValueError, "maximum 0.073030s"):
            profile.trajectory(dict(cfg, tracking_error_action="stop"))

    def test_recorded_acceleration_spike_is_not_reported_as_motor_capability(self):
        measurement = profile.measurements(self.report["feedback"], self.plan)
        axis = measurement["joints"][0]
        self.assertNotIn("finite_difference_peak_acceleration_deg_s2", axis)
        self.assertGreater(axis["derivative_estimate_count"], 10)
        self.assertLess(axis["estimated_peak_acceleration_deg_s2"], 500)
        self.assertIsNone(axis["frequency_hz"])
        self.assertFalse(measurement["tracking_verified"])

    def test_delayed_executor_runs_requested_cycles_and_preserves_parameters(self):
        cfg = dict(
            amplitude_deg=[20.0],
            velocity_deg_s=[120.0],
            acceleration_deg_s2=[100.0],
            feedback_reference_delay_s=0.1,
        )
        report, plan, _ = fixture.ExecutionTest().simulate(config=cfg, lag_s=0.1)
        self.assertTrue(report["success"], report.get("error"))
        self.assertTrue(report["duration_completed"])
        self.assertTrue(report["measurement"]["tracking_verified"])
        self.assertGreaterEqual(report["reference_send_elapsed_s"], 9.378)
        self.assertLess(report["reference_send_elapsed_s"], 9.40)
        self.assertEqual(plan["parameters"]["tracking_error_deg"], 5.0)
        self.assertGreater(
            max(abs(r["raw_error_deg"][6]) for r in report["feedback"]), 5.0
        )
        self.assertLess(max(abs(r["error_deg"][6]) for r in report["feedback"]), 1.0)
        old, _, _ = fixture.ExecutionTest().simulate(
            config=dict(cfg, feedback_reference_delay_s=0), lag_s=0.1
        )
        self.assertFalse(old["success"])
        self.assertIn("跟踪误差", old["error"])

    def test_true_tracking_error_still_stops_and_keeps_offending_sample(self):
        cfg = dict(
            amplitude_deg=[20.0],
            velocity_deg_s=[120.0],
            acceleration_deg_s2=[100.0],
            feedback_reference_delay_s=0.1,
        )
        report, _, _ = fixture.ExecutionTest().simulate(
            config=cfg, lag_s=0.1, tracking_fault=True
        )
        self.assertFalse(report["success"])
        self.assertFalse(report["duration_completed"])
        self.assertIn("跟踪误差", report["error"])
        self.assertFalse(report["last_observation"]["tracking_check_passed"])
        self.assertIs(report["last_observation"], report["feedback"][-1])
        self.assertTrue(report["failure_hold"]["hold_verified"])


if __name__ == "__main__":
    unittest.main()
