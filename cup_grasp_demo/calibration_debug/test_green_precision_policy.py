"""Offline recovery of precision mismatches; never retry a started action."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cup_grasp_demo.calibration_debug import green_pipeline as flow
from cup_grasp_demo.calibration_debug.hardware import validate_start, StartPositionChanged
from cup_grasp_demo.calibration_debug import test_joint_delivery as delivery_tests


class PrecisionTest(unittest.TestCase):
    def workflow(self):
        w = object.__new__(flow.Workflow)
        w.g = {'precision_error_action': 'record', 'recovery_attempts': 1}
        w.cfg = {}; w.scene = {}; w.snapshot = Mock(return_value=[.01] * 7)
        w.record_recovery = Mock()
        return w

    def test_held_replan_context_is_json_serializable(self):
        import json
        import numpy as np
        from cup_grasp_demo.calibration_debug import green_cup_planning as planning
        kin = Mock()
        kin.model.check_joint_path.return_value = SimpleNamespace(kinematic_checks_passed=True, samples_rad=[[0]*7])
        screen = Mock(); screen.check_table_batch.return_value = {'blockers': []}
        held = dict(T_flange_cup=np.eye(4), radius_m=.04, height_m=.065)
        with patch.object(planning, 'Kinematics', return_value=kin), patch.object(planning, 'Screen', return_value=screen), patch.object(planning, 'held_cup_clearance', return_value=10):
            plan = planning.arm_plan([0]*7, [[.01]*7], {}, {}, held=held, cup_margin_mm=-3)
        restored = json.loads(json.dumps(plan))
        self.assertEqual(restored['replan_context']['held']['T_flange_cup'], np.eye(4).tolist())
        self.assertEqual(restored['replan_context']['cup_margin_mm'], -3)

    def test_replan_preserves_held_cup_and_targets(self):
        w = self.workflow()
        w._issue_once = Mock(side_effect=[flow.ReplanStart(), {'success': True}])
        context = dict(held={'radius_m': .04}, cup_margin_mm=-3)
        plan = dict(kind='green_arm_plan', stages=[dict(target_q_rad=[.2]*7)], replan_context=context)
        rebuilt = dict(stages=[], blockers=[])
        with patch.object(flow, 'arm_plan', return_value=rebuilt) as planner:
            self.assertTrue(w.issue(plan, 'lower')['success'])
        self.assertEqual(planner.call_args.kwargs, context)
        self.assertEqual(planner.call_args.args[0], [.01]*7)
        self.assertEqual(planner.call_args.args[1], [[.2]*7])
        w.record_recovery.assert_called_once()

    def test_legacy_and_second_failure_stop(self):
        for policy, calls in [('stop', 1), ('record', 2)]:
            w = self.workflow(); w.g['precision_error_action'] = policy
            w._issue_once = Mock(side_effect=flow.ReplanStart())
            with self.assertRaises(flow.ReplanStart):
                w.issue(dict(stages=[], start_q_rad=[0]*7), 'grip')
            self.assertEqual(w._issue_once.call_count, calls)

    def test_replan_failure_does_not_send_second_request(self):
        w = self.workflow()
        w._issue_once = Mock(side_effect=flow.ReplanStart())
        plan = dict(kind='green_arm_plan', stages=[dict(target_q_rad=[.2]*7)],
                    replan_context=dict(held=None, cup_margin_mm=0))
        with patch.object(flow, 'arm_plan', side_effect=ValueError('关节路径检查未通过')):
            with self.assertRaises(ValueError):
                w.issue(plan, 'lift')
        w._issue_once.assert_called_once()

    def test_fault_is_never_classified_as_precision(self):
        status = SimpleNamespace(arm_status=1, motion_status=0, ctrl_mode=1)
        with self.assertRaises(RuntimeError) as error:
            validate_start({'start_q_rad':[0]*7}, [.1]*7, status, [True]*7, .25)
        self.assertNotIsInstance(error.exception, StartPositionChanged)
        status.arm_status = 0
        with self.assertRaises(StartPositionChanged):
            validate_start({'start_q_rad':[0]*7}, [.1]*7, status, [True]*7, .25)

    def test_only_pre_motion_receipt_can_trigger_replan(self):
        import json
        for attempted, fingers, code, recover in [(False,False,'start_position_changed',True),
                (True,False,'start_position_changed',False),
                (False,True,'start_position_changed',False), (False,False,'communication',False)]:
            with tempfile.TemporaryDirectory() as tmp:
                w = self.workflow(); w.root = Path(tmp); w.unchanged=Mock()
                w.args=SimpleNamespace(mode='fast'); w.close_sdk=Mock()
                def fail(*args):
                    Path(args[1]).write_text(json.dumps(dict(failure_code=code,
                        motion_attempted=attempted, finger_commands_sent=fingers)))
                    raise RuntimeError('SDK stopped')
                w.bridge=fail
                with patch.object(flow.common,'new_run',return_value=Path(tmp)):
                    with self.assertRaises(RuntimeError) as error:
                        w._issue_once({'blockers':[]},'lift')
                self.assertEqual(isinstance(error.exception,flow.ReplanStart),recover)
                self.assertEqual(w.close_sdk.called,recover)

    def test_no_final_repeat_keeps_endpoint_and_saves_40ms(self):
        fixture = delivery_tests.DeliveryTest(); fixture.setUp()
        fixture.proxy.move_js([.1]+[0]*6)
        original = fixture.events[0]['actual_delivery_s']; count=len(fixture.sent)
        fixture.setUp(); fixture.proxy.repeat_final=False
        fixture.proxy.move_js([.1]+[0]*6)
        self.assertAlmostEqual(original-fixture.events[0]['actual_delivery_s'], .04)
        self.assertEqual(len(fixture.sent),count-2)
        self.assertAlmostEqual(fixture.q[0], .1)
        self.assertTrue(fixture.events[0]['delivery_completed'])
