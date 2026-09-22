"""Resource lifecycle and conservative cache invalidation regression tests."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from cup_grasp_demo.calibration_debug.green_prepared import PreparedRoutes
from cup_grasp_demo.calibration_debug import green_runtime as runtime
from cup_grasp_demo.calibration_debug import green_pipeline as flow


class PreparedTest(unittest.TestCase):
    def test_only_identical_path_can_skip_full_screen(self):
        plan = dict(start_q_rad=[0]*7, stages=[dict(target_q_rad=[.1]*7)], blockers=[])
        for q, scene, targets, expected in (
            ([0]*7, {'height': 1}, [[.1]*7], True),
            ([.000001]*7, {'height': 1}, [[.1]*7], False),
            ([0]*7, {'height': 2}, [[.1]*7], False),
            ([0]*7, {'height': 1}, [[.2]*7], False),
        ):
            c = PreparedRoutes(); c.put('approach', plan, {'height': 1})
            self.assertEqual(c.take('approach', q, targets, scene) is not None, expected)
            self.assertIsNone(c.take('approach', q, targets, scene))

    def test_reuse_uses_existing_executor_tolerance_without_expanding_it(self):
        plan=dict(start_q_rad=[0]*7,stages=[dict(target_q_rad=[.1]*7)],blockers=[])
        for delta, expected in ((.009, True), (.011, False), (float('nan'),False)):
            c=PreparedRoutes();c.put('approach',plan,{})
            result=c.take('approach',np.radians([delta]*7),[[.1]*7],{},start_tolerance_deg=.01)
            self.assertEqual(result is not None,expected)

    def test_held_geometry_change_invalidates(self):
        p = dict(start_q_rad=[0]*7, stages=[dict(target_q_rad=[.1]*7)], blockers=[])
        c = PreparedRoutes(); c.put('lift', p, {}, {'radius': .0375}, -3)
        self.assertIsNone(c.take('lift', [0]*7, [[.1]*7], {}, {'radius': .04}, -3))

    def test_ik_cache_rejects_branch_or_translation_change(self):
        for delta, xyz, accepted in ((0, 0, True), (.001, 0, False), (0, .001, False)):
            c=PreparedRoutes();c.put_vertical('lift', [0]*7, [[.1]*7])
            a=np.eye(4);b=np.eye(4);b[0,3]=xyz
            kin=Mock();kin.forward.side_effect=[(a,None),(b,None)]
            self.assertEqual(c.take_vertical('lift', [delta]*7, kin) is not None, accepted)


class LifecycleTest(unittest.TestCase):
    def test_workflow_reuses_sdk_and_closes_before_shake_boundary(self):
        w=object.__new__(flow.Workflow);w.args=SimpleNamespace(mode='fast')
        w.g={};w.cfg={};w.root=Path('/tmp');w._sdk=None
        with patch.object(runtime,'SDKClient') as factory, patch.object(flow.common,'new_run',return_value=Path('/tmp')):
            w.bridge('snapshot', Path('/tmp/a'), {})
            w.bridge('snapshot', Path('/tmp/b'), {})
            factory.assert_called_once()
            self.assertEqual(factory.return_value.call.call_count,2)
            w.close_sdk();factory.return_value.close.assert_called_once()
            self.assertIsNone(w._sdk)

    def test_step_reuses_sdk_and_camera_during_prompts(self):
        w=object.__new__(flow.Workflow);w.args=SimpleNamespace(mode='step')
        w.g={'persistent_runtime':True,'perception':{'frame_count':5}}
        w.cfg={};w.root=Path('/tmp');w._sdk=None
        w._vision=Mock()
        with patch.object(runtime,'SDKClient') as factory, patch.object(flow.common,'new_run',return_value=Path('/tmp')):
            w.bridge('snapshot',Path('/tmp/a'),{})
            w.bridge('snapshot',Path('/tmp/b'),{})
            factory.assert_called_once()
            self.assertEqual(factory.return_value.call.call_count,2)
        with patch.object(flow.common,'capture_with_feedback',return_value={'joints_rad':[0]*7}) as capture:
            w.capture_with_feedback(Path('/tmp/rgbd'))
            self.assertEqual(capture.call_args.kwargs['bridge_fn'],w.bridge)
            capture.call_args.kwargs['capture_fn'](Path('/tmp/new'),{})
        w._vision.capture.assert_called_once_with(Path('/tmp/new'),5)

    def test_failed_rpc_closes_and_surfaces_receipt_error(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'receipt.json';out.write_text(json.dumps({'success':False,'error':'fault'}))
            client=object.__new__(runtime.SDKClient);client.cfg={'timeout_s':40}
            client.process=Mock();client.receive=Mock(return_value={'output':str(out),'returncode':2})
            client.close=Mock()
            with self.assertRaisesRegex(RuntimeError,'fault'):
                client.call('snapshot',out)
            client.close.assert_called_once()

    def test_shake_start_change_returns_to_bounded_workflow_recovery(self):
        for attempted in (False, True, None):
            with self.subTest(attempted=attempted), tempfile.TemporaryDirectory() as d:
                out = Path(d) / 'receipt.json'
                req = Path(d) / 'request.json'
                req.write_text(json.dumps({'plan': {'duration_s': 5}}))
                report = dict(success=False, error='start changed',
                              failure_code='start_position_changed', motion_attempted=attempted)
                out.write_text(json.dumps(report))
                client = object.__new__(runtime.SDKClient)
                client.cfg = {'timeout_s': 40}
                client.process = Mock()
                client.receive = Mock(return_value={'output': str(out), 'returncode': 2})
                client.close = Mock()
                if attempted is False:
                    self.assertEqual(client.call('shake', out, req), report)
                    client.close.assert_not_called()
                else:
                    with self.assertRaisesRegex(RuntimeError, 'start changed'):
                        client.call('shake', out, req)
                    client.close.assert_called_once()

    def test_capture_waits_for_resources_and_requests_fresh_frames(self):
        resource=object.__new__(runtime.VisionResources)
        resource.camera_future=Mock();resource.model_future=Mock();resource.camera=Mock()
        resource.capture(Path('/tmp/new'),1)
        resource.camera.capture.assert_called_once_with(Path('/tmp/new'),1,fresh=True)

    def test_failed_camera_start_never_captures(self):
        resource=object.__new__(runtime.VisionResources)
        resource.camera_future=Mock();resource.model_future=Mock();resource.camera=Mock()
        resource.camera_future.result.side_effect=RuntimeError('camera failed')
        with self.assertRaises(RuntimeError):resource.capture(Path('/tmp/new'),1)
        resource.camera.capture.assert_not_called()
