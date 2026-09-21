import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from cup_grasp_demo.calibration_debug import green_pipeline as flow

class DirectApproachTest(unittest.TestCase):
    def test_zero_approach_does_not_remove_retreat(self):
        import copy
        cfg=flow.load_config(flow.ROOT/'cup_grasp_demo/calibration_debug/green_open_cup/stereo_config.json')
        cfg['green_cup'].update(approach_via_above=False,approach_clearance_mm=0)
        cfg['green_cup'].pop('retreat_clearance_mm',None)
        self.assertEqual(flow.validate(cfg)['retreat_clearance_mm'],50)
        legacy=copy.deepcopy(cfg)
        legacy['green_cup']['approach_via_above']=True
        with self.assertRaisesRegex(ValueError,'approach_clearance_mm'):
            flow.validate(legacy)


    def test_direct_and_legacy_routes(self):
        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as d:
                wf=object.__new__(flow.Workflow)
                wf.root=Path(d);wf.reference=dict(T_base_flange=np.eye(4).tolist(),joints_rad=[0]*7)
                wf.contact=np.array([.1,.2,.3]);wf.tcp=np.eye(4)
                wf.g=dict(approach_via_above=enabled,approach_clearance_mm=50,wrist_reference_deg=[0,-13,5])
                wf.cfg=dict(plan_max_age_s=600);wf.scene={};wf.capture_time=time.time()
                wf.snapshot=Mock(return_value=[0]*7);wf.move=Mock()
                with patch.object(flow,'solve',side_effect=[[.1]*7,[.2]*7]) as solve, patch.object(flow,'arm_plan',return_value=dict(blockers=[])) as plan:
                    wf.plan()
                    self.assertEqual(solve.call_count,2 if enabled else 1)
                    self.assertEqual(len(plan.call_args.args[1]),2 if enabled else 1)
                    self.assertEqual(plan.call_args.args[1][-1],[.1]*7)
                    wf.perform('APPROACH')
                    wf.move.assert_called_once_with(wf.approach_targets,'approach')


class DirectLiftTest(unittest.TestCase):
    def test_lift_sends_only_one_endpoint(self):
        wf=object.__new__(flow.Workflow)
        wf.place_q=[0]*7;wf.tcp=np.eye(4);wf.held={}
        wf.g=dict(lift_mm=50,wrist_reference_deg=[0,-13,5],place_tolerance_mm=3)
        wf.move=Mock();wf.snapshot=Mock(return_value=[.1]*7)
        with patch.object(flow,'vertical_targets',return_value=[[.1]*7]) as targets:
            wf.perform('LIFT')
        targets.assert_called_once_with(wf.place_q,wf.tcp,.05,[0,-13,5],single_target=True)
        wf.move.assert_called_once_with([[.1]*7],'lift',wf.held,-3)
        self.assertEqual(wf.lift_q,[.1]*7)

class PlacementTest(unittest.TestCase):
    def test_place_uses_saved_pose_and_checks_before_open(self):
        wf=object.__new__(flow.Workflow)
        wf.place_q=[.1]*7;wf.tcp=np.eye(4);wf.held={}
        wf.g=dict(place_tolerance_mm=3)
        wf.move=Mock();wf.snapshot=Mock(return_value=[.1]*7)
        wf.kin=Mock();wf.kin.forward.return_value=(np.eye(4),None)
        wf.perform('LOWER')
        wf.move.assert_called_once_with([wf.place_q],'lower',wf.held,-3)
        shifted=np.eye(4);shifted[2,3]=.01
        wf.kin.forward.side_effect=[(np.eye(4),None),(shifted,None),(shifted,None)]
        with self.assertRaisesRegex(RuntimeError,'保持闭手'):wf.perform('LOWER')

    def test_idle_completion_keeps_fault_checks_and_legacy_tolerance(self):
        from cup_grasp_demo.calibration_debug import shake_execution as execution
        from types import SimpleNamespace
        clock=[0.]
        timer=SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s))
        row=dict(q_rad=[.006]*7)
        with patch.object(execution,'time',timer), patch.object(execution.core,'fresh_feedback',return_value=row), patch.object(execution,'state_ok',return_value=True):
            self.assertIs(execution.verify_stop(None,[0]*7,[],timeout=1,require_position=False),row)
            with self.assertRaises(TimeoutError):execution.verify_stop(None,[0]*7,[],timeout=1)
        with patch.object(execution.core,'fresh_feedback',return_value=row), patch.object(execution,'state_ok',return_value=False):
            with self.assertRaises(RuntimeError):execution.verify_stop(None,[0]*7,[],require_position=False)


class RecoverPlacementTest(unittest.TestCase):
    def test_corrected_place_continues(self):
        wf=object.__new__(flow.Workflow);wf.place_q=[0]*7;wf.tcp=np.eye(4);wf.held={}
        wf.g=dict(place_tolerance_mm=3,recovery_attempts=1)
        wf.move=Mock();wf.snapshot=Mock(return_value=[0]*7);wf.kin=Mock()
        wrong=np.eye(4);wrong[2,3]=.01
        wf.kin.forward.side_effect=[(np.eye(4),None),(wrong,None),(np.eye(4),None)]
        wf.perform('LOWER')
        self.assertEqual(wf.move.call_count,2)
        self.assertEqual(wf.move.call_args.args[1],'lower_correct')
        self.assertEqual(len(wf.recovery_events),1)

    def test_small_place_error_continues_without_changing_collision_margin(self):
        wf=object.__new__(flow.Workflow);wf.place_q=[0]*7;wf.tcp=np.eye(4);wf.held={}
        wf.g=dict(place_tolerance_mm=3,place_arrival_tolerance_mm=5)
        wf.move=Mock();wf.snapshot=Mock(return_value=[0]*7);wf.kin=Mock()
        offset=np.eye(4);offset[2,3]=.00337
        wf.kin.forward.side_effect=[(np.eye(4),None),(offset,None)]
        wf.perform('LOWER')
        wf.move.assert_called_once_with([wf.place_q],'lower',wf.held,-3)
        self.assertAlmostEqual(wf.place_arrival_error_mm,3.37)

    def test_direct_return_home_has_no_vertical_retreat(self):
        wf=object.__new__(flow.Workflow);wf.g={'direct_return_home':True};wf.home=[0]*7
        wf.move=Mock();wf.vertical=Mock();wf.snapshot=Mock()
        wf.perform('RETURN_HOME')
        wf.move.assert_called_once_with([wf.home],'return_home')
        wf.vertical.assert_not_called()
