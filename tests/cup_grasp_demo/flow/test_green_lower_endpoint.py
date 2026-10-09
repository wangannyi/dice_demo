"""Placement mode is request-local; other pipeline phases keep their transport."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from cup_grasp_demo.flow import green_pipeline as flow
from cup_grasp_demo.flow.core import load_config, ROOT


class LowerEndpointTests(unittest.TestCase):
    def test_endpoint_and_speed_do_not_leak_into_other_phases(self):
        with tempfile.TemporaryDirectory() as folder:
            w = object.__new__(flow.Workflow)
            w.root = Path(folder)
            w.cfg = {"speed_percent": 30, "joint_delivery": {"command_mode": "smooth_profile"}}
            w.g = {"lower_command_mode": "controller_endpoint",
                   "fast_phase_speed_percent": {"lower": 60}}
            before = copy.deepcopy(w.cfg)
            w.unchanged = Mock()
            w.receipts = {}
            requests = {}
            def bridge(_command, _out, _cfg, request, **kwargs):
                import json
                requests[current[0]] = json.loads(Path(request).read_text())
                return {"success": True}
            w.bridge = bridge
            current = [None]
            for phase in ("lower", "approach", "lift", "lower_correct", "return_home"):
                current[0] = phase
                w._issue_once({"blockers": [], "stages": []}, phase)
            self.assertEqual(requests["lower"]["config"]["speed_percent"], 60)
            self.assertEqual(requests["lower"]["config"]["joint_delivery"]["command_mode"], "controller_endpoint")
            self.assertEqual(requests["lower"]["config"]["joint_delivery"]["endpoint_resend_interval_s"], .05)
            for phase in ("approach", "lift", "lower_correct", "return_home"):
                self.assertEqual(requests[phase]["config"], before)
            self.assertEqual(w.cfg, before)
            w.g.pop("lower_command_mode")
            w._issue_once({"blockers": [], "stages": []}, "lower")
            self.assertEqual(requests[current[0]]["config"]["joint_delivery"]["command_mode"], "smooth_profile")

    def test_config_validation(self):
        cfg = load_config(ROOT / "configs/green_cup.json")
        flow.validate(cfg)
        for mode in ("smooth_profile", "controller_endpoint"):
            cfg["green_cup"]["lower_command_mode"] = mode
            flow.validate(cfg)
        cfg["green_cup"]["lower_command_mode"] = "jump"
        with self.assertRaisesRegex(ValueError, "lower_command_mode"):
            flow.validate(cfg)
        cfg["green_cup"]["lower_command_mode"] = "controller_endpoint"
        for speed in (0, 101, True):
            cfg["green_cup"]["fast_phase_speed_percent"]["lower"] = speed
            with self.assertRaisesRegex(ValueError, "fast_phase_speed_percent"):
                flow.validate(cfg)
