import copy
import unittest
from unittest.mock import patch
from cup_grasp_demo.flow import core

class PlaneConfigTests(unittest.TestCase):
    def config(self, green, value):
        name='green_open_cup/stereo_config.json' if green else 'index_joint_center/config.json'
        cfg=core.read_json(core.ROOT/'cup_grasp_demo/flow'/name)
        cfg['plane_tolerance_mm']=value
        with patch.object(core,'read_json',return_value=copy.deepcopy(cfg)):
            return core.load_config('unused')
    def test_green_accepts_six(self):
        self.assertEqual(self.config(True,6.)['plane_tolerance_mm'],6.)
    def test_legacy_range_preserved(self):
        self.config(False,5.)
        with self.assertRaisesRegex(ValueError,'plane_tolerance_mm'):self.config(False,6.)
    def test_green_rejects_invalid(self):
        for value in [0,6.01,float('inf'),float('nan')]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError,'plane_tolerance_mm'):
                self.config(True,value)
