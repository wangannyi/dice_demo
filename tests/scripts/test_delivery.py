"""Delivery checks without camera/CAN access: python3 -m unittest scripts.test_delivery."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

class DeliveryTests(unittest.TestCase):
    def test_wrapper_rejects_unknown_mode(self):
        result = subprocess.run([str(ROOT/'run.sh'), 'invalid'], capture_output=True)
        self.assertEqual(result.returncode, 2)

    def test_wrapper_arguments_and_configuration_override(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory)/'python'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake.chmod(0o755)
            env = dict(os.environ, DICE_VISION_PYTHON=str(fake), DICE_CONFIG='/tmp/custom.json', DICE_RUN='/tmp/custom_run')
            result = subprocess.run([str(ROOT/'run.sh'), 'fast', '--until', 'grip'], env=env, capture_output=True, text=True, check=True)
            self.assertIn('/tmp/custom.json', result.stdout)
            self.assertIn('/tmp/custom_run', result.stdout)
            self.assertIn('fast', result.stdout)
            self.assertTrue(result.stdout.rstrip().endswith('grip'))
            self.assertNotIn('--execute', result.stdout)

    def test_shake_configuration_is_shared(self):
        cfg = json.loads((ROOT/'configs/green_cup.json').read_text())
        self.assertEqual(cfg['green_cup']['joint_test_config'], 'configs/joint_shake.json')
        self.assertTrue((ROOT/cfg['green_cup']['joint_test_config']).is_file())

    def test_shake_configurations_are_valid_independent_recipes(self):
        release = json.loads((ROOT/'configs/joint_shake.json').read_text())
        current = json.loads((ROOT/'cup_grasp_demo/calibration_debug/joint_test_config.json').read_text())
        from cup_grasp_demo.calibration_debug.joint_profile import options
        # Operators may tune the development and release recipes independently.
        for recipe in (release, current):
            self.assertEqual(options(recipe)["joints"], recipe["joints"])

    def test_feedback_uses_release_system_configuration(self):
        from scripts.result_feedback import DEFAULT_SYSTEM
        self.assertEqual(DEFAULT_SYSTEM, ROOT/'configs/green_cup.json')
