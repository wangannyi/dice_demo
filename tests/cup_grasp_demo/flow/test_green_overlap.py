"""Overlap only CPU/camera work; robot commands retain one sequential owner."""
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from cup_grasp_demo.flow import green_pipeline as f

class OverlapTests(unittest.TestCase):
    def test_at_home_capture_overlaps_open_without_arm_target(self):
        w=object.__new__(f.Workflow);w.args=SimpleNamespace(mode='fast');w.cfg={'home':'unused'}
        w.g={'persistent_runtime':False};w.prepare_vision=Mock();w.table=Mock();w.capture=Mock();w.snapshot=Mock(return_value=[0]*7);w.issue=Mock()
        try:
            with patch.object(f,'read_json',return_value={'joints_deg':[0]*7}):w.perform('HOME')
            w.perform('CAPTURE')
            w.capture.assert_called_once()
            self.assertEqual(w.issue.call_args.args[0]['stages'],[])
        finally:w._capture_pool.shutdown()

    def test_away_from_home_capture_is_not_started_early(self):
        w=object.__new__(f.Workflow);w.args=SimpleNamespace(mode='fast');w.cfg={'home':'unused'};w.scene={}
        w.g={'persistent_runtime':False};w.prepare_vision=Mock();w.table=Mock();w.capture=Mock();w.snapshot=Mock(return_value=[.1]*7);w.issue=Mock()
        with patch.object(f,'read_json',return_value={'joints_deg':[0]*7}),patch.object(f,'arm_plan',return_value={'stages':[{}]}):w.perform('HOME')
        w.capture.assert_not_called();self.assertFalse(hasattr(w,'_capture_future'))
        w.perform('CAPTURE');w.capture.assert_called_once()

    def test_capture_error_propagates_before_plan(self):
        w=object.__new__(f.Workflow);w._capture_future=Future();w._capture_future.set_exception(ValueError('no cup'))
        with self.assertRaisesRegex(ValueError,'no cup'):w.perform('CAPTURE')

    def test_shake_preparation_uses_real_limits_and_lift_endpoint(self):
        w=object.__new__(f.Workflow);w.args=SimpleNamespace(mode='fast',until='place');w.receipts={'approach':'unused'}
        w.build_shake=Mock(return_value=({'planning_passed':True},{}))
        limits=[{'joint':1,'max_acceleration_rad_s2':5}]
        try:
            with patch.object(f,'read_json',return_value={'joint_delivery_events':[{'live_limits':limits}]}):
                w.start_shake_prepare([.2]*7)
            self.assertTrue(w._shake_future.result(2)[0]['planning_passed'])
            self.assertEqual(w.build_shake.call_args.args[0],dict(success=True,limits=limits,q_after_rad=[.2]*7))
        finally:w._shake_pool.shutdown()

    def test_shake_background_inherits_geometry_cache(self):
        from cup_grasp_demo.flow.core import cached_screen_geometry, _screen_cache
        w=object.__new__(f.Workflow);w.args=SimpleNamespace(mode='fast',until='place');w.receipts={'approach':'unused'}
        w.build_shake=lambda feedback: _screen_cache.get()
        with cached_screen_geometry(), patch.object(f,'read_json',return_value={'joint_delivery_events':[{'live_limits':{'ok':True}}]}):
            expected=_screen_cache.get()
            try:
                w.start_shake_prepare([0]*7)
                self.assertIs(w._shake_future.result(2), expected)
            finally:w._shake_pool.shutdown()

    def test_fast_stop_requires_three_fresh_idle_reports(self):
        from cup_grasp_demo.flow import shake_execution as sh
        rows=[{'id':i,'q_rad':[0]*7} for i in range(3)]
        report=[]
        with patch.object(sh.core,'fresh_feedback',side_effect=rows) as fresh, patch.object(sh,'state_ok',return_value=True), patch.object(sh.time,'sleep'):
            sh.verify_stop(Mock(), [0]*7, report, stable_samples=3, poll_s=0)
        self.assertEqual(len(report),3)
        self.assertIs(fresh.call_args.kwargs['previous'],rows[1])
        with patch.object(sh.core,'fresh_feedback',return_value=rows[0]), patch.object(sh,'state_ok',return_value=False):
            with self.assertRaises(RuntimeError):
                sh.verify_stop(Mock(),[0]*7,[],stable_samples=3,poll_s=0)

    def test_legacy_limits_keep_readback_fallback(self):
        w=object.__new__(f.Workflow);w.args=SimpleNamespace(mode='fast',until='place');w.receipts={'approach':'unused'}
        with patch.object(f,'read_json',return_value={'joint_delivery_events':[]}):w.start_shake_prepare([0]*7)
        self.assertFalse(hasattr(w,'_shake_future'))

    def test_handoff_target_is_measured_not_assumed(self):
        w=object.__new__(f.Workflow)
        w.record_handoff('GRIP_TO_LIFT',100,100.2)
        w.record_handoff('LIFT_TO_SHAKE',101,102)
        self.assertTrue(w.handoff_timings['GRIP_TO_LIFT']['target_met'])
        self.assertFalse(w.handoff_timings['LIFT_TO_SHAKE']['target_met'])

    def test_connected_shake_fault_restores_guard_without_disconnect(self):
        from cup_grasp_demo.flow import joint_execution as j
        request=dict(channel='can0',load_context='green_cup_held',plan=dict(parameters={'tracking_error_deg':5},samples=[]))
        session=Mock();guard=Mock();robot=Mock()
        with patch.object(j,'validate_request'),patch.object(j.core,'load_sdk_runtime',return_value=(Mock(),Mock())),patch.object(j,'JointGuard',return_value=guard),patch.object(j.core,'PassivePoseSession',return_value=session),patch.object(j.shared,'control_evidence',return_value={}),patch.object(j.core,'fresh_feedback',side_effect=RuntimeError('fault')),patch.object(j,'measurements',return_value={}):
            result=j.run(request,connected=robot,connection_evidence={})
        self.assertFalse(result['success'])
        self.assertFalse(result['sdk_disconnected'])
        session.start.assert_not_called();session.close.assert_not_called()
        robot.disconnect.assert_not_called();guard.restore.assert_called_once()

    def test_persistent_connection_uses_original_receiver_baseline(self):
        from cup_grasp_demo.flow import joint_execution as j
        baseline = dict(channel='can0', receiver_rows=[], errors=[], candidate_control_processes=[])
        current = dict(baseline, receiver_rows=[
            dict(list='all', line='can0 000 00000000'),
            dict(list='err', line='can0 000 1fffffff')])
        self.assertEqual(j.shared.control_conflicts(baseline, current), [])
        self.assertIn('CAN receivers existed before this SDK connection',
                      j.shared.control_conflicts(current, current))
        request=dict(channel='can0',load_context='green_cup_held',plan=dict(parameters={'tracking_error_deg':5},samples=[]))
        with patch.object(j,'validate_request'),patch.object(j.core,'load_sdk_runtime',return_value=(Mock(),Mock())),patch.object(j,'JointGuard'),patch.object(j.core,'PassivePoseSession'),patch.object(j.core,'fresh_feedback',side_effect=RuntimeError('fault')),patch.object(j,'measurements',return_value={}):
            result=j.run(request,connected=Mock(),connection_evidence=baseline)
        self.assertEqual(result['host_control_before_connect'], baseline)

    def test_persistent_shake_stream_completes_and_feedback_loss_stops(self):
        from test_joint_lab import ExecutionTest
        report, _, robot = ExecutionTest().simulate(connected=True)
        self.assertTrue(report['success'], report.get('error'))
        self.assertTrue(report['duration_completed'])
        self.assertFalse(report['sdk_disconnected'])
        robot.disconnect.assert_not_called()
        failed, _, _ = ExecutionTest().simulate(connected=True, fail_at=.3)
        self.assertFalse(failed['success'])
        self.assertIn('feedback loss', failed['error'])
        self.assertTrue(failed['failure_hold']['hold_verified'])
