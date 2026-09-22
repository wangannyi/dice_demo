import json
import tempfile
import unittest
from pathlib import Path
from nero_revo2_control import nero_revo2_demo as demo
from cup_grasp_demo.calibration_debug.core import ROOT, load_config
from cup_grasp_demo.calibration_debug.hardware import arm_motion_args
from cup_grasp_demo.calibration_debug.joint_delivery import delivery_options
from cup_grasp_demo.calibration_debug.joint_profile import options

class GreenSpeedTest(unittest.TestCase):
    def test_config_and_executor_full_percentage(self):
        cfg = json.loads((ROOT/'cup_grasp_demo/calibration_debug/green_open_cup/stereo_config.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json'
            for speed in (20,100):
                cfg['speed_percent']=speed;p.write_text(json.dumps(cfg))
                checked=load_config(p)
                args=arm_motion_args(demo,dict(target_q_rad=[0]*7),speed,checked)
                self.assertEqual(args.speed,speed)
            cfg['pipeline_strategy']='legacy';p.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):load_config(p)
            with self.assertRaises(ValueError):arm_motion_args(demo,dict(target_q_rad=[0]*7),100,cfg)
    def test_speed_caps_optional_legacy_preserved(self):
        self.assertEqual(delivery_options()['velocity_cap_deg_s'],50)
        self.assertIsNone(delivery_options({'velocity_cap_deg_s':None})['velocity_cap_deg_s'])
        self.assertEqual(options({'controller_speed_percent':100})['controller_speed_percent'],100)
