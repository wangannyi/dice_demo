"""Endpoint tolerance opt-out retains fresh idle/fault checks; no CAN access."""
import math
import unittest
from unittest.mock import patch
import nero_revo2_demo as demo
from test_nero_revo2_demo import FakeRobot, SimulatedClock
from cup_grasp_demo.calibration_debug.hardware import arm_motion_args

class CompletionTest(unittest.TestCase):
    def wait(self, robot, required):
        clock=SimulatedClock()
        with patch.object(demo.time,'monotonic',clock.monotonic), patch.object(demo.time,'sleep',clock.sleep), patch.object(demo,'emit') as emit:
            demo.wait_for_arm_target(robot,'move-j',[math.radians(.5003)]*7,1,
                                     require_joint_position=required)
        return emit

    def test_residual_is_record_only_when_opted_out(self):
        emit=self.wait(FakeRobot(),False)
        self.assertEqual(emit.call_args.args[0],'arm_motion_idle')
        self.assertFalse(emit.call_args.kwargs['position_verified'])
        with self.assertRaises(TimeoutError):self.wait(FakeRobot(),True)

    def test_fault_stale_and_moving_still_fail(self):
        for attr,value,error in [('arm_status',2,RuntimeError),('freeze_feedback',True,TimeoutError),('motion_status',1,TimeoutError)]:
            robot=FakeRobot();setattr(robot,attr,value)
            with self.subTest(attr=attr),self.assertRaises(error):self.wait(robot,False)

    def test_legacy_default_remains_enabled(self):
        cfg=dict(timeout_s=40,green_cup=dict(require_arm_position=False))
        stage=dict(target_q_rad=[0]*7)
        self.assertTrue(arm_motion_args(demo,stage,5,cfg).require_joint_position)
        cfg['pipeline_strategy']='green_open_cup'
        self.assertFalse(arm_motion_args(demo,stage,20,cfg).require_joint_position)
