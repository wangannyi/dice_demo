import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from cup_grasp_demo.flow import green_hand_execution as m

class HandCANTests(unittest.TestCase):
    def run_case(self, mode, switch=True, fault=0, read_feedback=True,
                 joints=None, hand_tolerance=None):
        state=SimpleNamespace(arm_status=fault,ctrl_mode=mode,motion_status=0)
        robot=Mock();robot.get_joints_enable_status_list.return_value=[True]*7
        def change(_):
            if switch: state.ctrl_mode=1
        robot.set_motion_mode.side_effect=change
        demo=Mock();demo.arm_snapshot.return_value=(joints or [0]*7,None,state)
        cfg={'green_cup':dict(open_targets_0_100=[0]*6,grip_targets_0_100=[100]*6,
             read_hand_feedback=read_feedback,require_hand_position=False,finger_duration_s=1.,finger_settle_s=.1)}
        if hand_tolerance is not None:
            cfg['green_cup']['hand_start_tolerance_deg'] = hand_tolerance
        result={}
        with patch.object(m,'send_closure',return_value={}) as send, patch.object(m.time,'monotonic',side_effect=[0,3]):
            try: m.execute(dict(target_0_100=[0]*6,start_q_rad=[0]*7),cfg,robot,Mock(),demo,result)
            except (RuntimeError,TimeoutError):
                send.assert_not_called()
                raise
            if not read_feedback:
                self.assertFalse(send.call_args.kwargs["read_feedback"])
        return robot,result
    def test_web_switches_before_hand(self):
        robot,result=self.run_case(3)
        robot.set_motion_mode.assert_called_once_with('js')
        self.assertTrue(result['can_mode_command_sent'])
    def test_can_does_not_switch(self):
        robot,_=self.run_case(1);robot.set_motion_mode.assert_not_called()
    def test_failed_switch_never_closes(self):
        with self.assertRaises(TimeoutError): self.run_case(3,False)
    def test_fault_never_closes(self):
        with self.assertRaises(RuntimeError): self.run_case(3,fault=1)

    def test_command_only_skips_hand_feedback(self):
        self.run_case(1, read_feedback=False)

    def test_configured_arm_tolerance_is_used_during_hand_command(self):
        import math
        drift = [math.radians(.75)] + [0] * 6
        self.run_case(1, joints=drift, hand_tolerance=1.0)
        with self.assertRaisesRegex(RuntimeError, 'Arm moved'):
            self.run_case(1, joints=drift, hand_tolerance=.5)
