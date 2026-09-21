"""No-hardware regression checks for the independent planning entry."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import kinematics
import offline_plan


class OfflinePlanningTests(unittest.TestCase):
    def test_official_urdf_fk_matches_recorded_nero_flange(self):
        model = kinematics.load_model()
        historical_deg = (55, -78, 80, -45, 130, -30, 49)
        matrix = model.fk(tuple(math.radians(value) for value in historical_deg))
        recorded_flange_m = (0.114693, 0.500060, 0.217845)
        difference = math.sqrt(
            sum((matrix[i][3] - recorded_flange_m[i]) ** 2 for i in range(3))
        )
        self.assertLess(difference, 0.0003)

    def test_synthetic_two_stage_plan_proves_only_kinematic_checks(self):
        result = offline_plan.plan_synthetic()
        self.assertEqual(result["status"], "synthetic_kinematics_complete")
        self.assertEqual(
            [stage["name"] for stage in result["stages"]], ["pregrasp", "grasp"]
        )
        self.assertTrue(result["kinematics_ok"])
        for stage in result["stages"]:
            self.assertTrue(stage["ik"]["success"])
            self.assertLess(stage["ik"]["position_error_m"], 0.0001)
            self.assertLess(stage["ik"]["orientation_error_rad"], 0.001)
            self.assertGreaterEqual(stage["joint_path"]["sample_count"], 3)
            self.assertEqual(stage["joint_path"]["max_sample_step_deg"], 1.0)
            self.assertTrue(stage["joint_path"]["joint_limits_ok"])
            self.assertTrue(stage["joint_path"]["continuity_ok"])
            self.assertFalse(stage["joint_path"]["scene_collision_verified"])
        self.assertFalse(result["scene_collision_verified"])
        self.assertFalse(result["grasp_tcp_verified"])
        self.assertFalse(result["real_hand_eye_verified"])
        self.assertFalse(result["executable"])
        self.assertFalse(result["motion_sent"])

    def test_out_of_bounds_joint_interpolation_is_not_accepted(self):
        model = kinematics.load_model()
        goal = list(offline_plan.READY_HOME_RAD)
        goal[1] = model.limits_rad[1][0] - math.radians(1)
        check = model.check_joint_path(offline_plan.READY_HOME_RAD, goal)
        self.assertFalse(check.kinematic_checks_passed)
        self.assertFalse(check.scene_collision_verified)

    def test_unaccepted_cup_localization_cannot_supply_motion_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(
                json.dumps({"base_targets": None, "blockers": ["calibration failed"]}),
                encoding="utf-8",
            )
            result = offline_plan.plan_from_localization(path)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["stages"], [])
        self.assertFalse(result["base_targets_available"])
        self.assertFalse(result["executable"])
        self.assertFalse(result["motion_sent"])
        self.assertIn("calibration failed", result["input_blockers"])

    def test_planner_import_never_loads_hardware_sdk(self):
        isolated = subprocess.run(
            [
                sys.executable,
                "-c",
                "import offline_plan, sys; assert 'pyAgxArm' not in sys.modules; "
                "assert 'can' not in sys.modules",
            ],
            cwd=Path(__file__).resolve().parent,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(isolated.returncode, 0, isolated.stderr)


if __name__ == "__main__":
    unittest.main()
