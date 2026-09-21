"""STEP visualization reads current feedback without issuing robot commands."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from cup_grasp_demo.calibration_debug import green_pipeline as flow


class StepTcpTest(unittest.TestCase):
    def test_fast_and_home_do_not_capture(self):
        wf = object.__new__(flow.Workflow)
        with patch.object(flow.common, 'capture_with_feedback') as capture:
            for mode, phase in [*[("fast", p) for p in flow.PHASES], ('step', 'HOME'), ('step', 'RETURN_HOME')]:
                wf.args = SimpleNamespace(mode=mode)
                wf.step_tcp_view(phase)
            capture.assert_not_called()

    def test_step_saves_current_feedback_and_frozen_target(self):
        with tempfile.TemporaryDirectory() as d:
            wf = object.__new__(flow.Workflow)
            wf.args = SimpleNamespace(mode='step', show=False)
            wf.root = Path(d)
            wf.cfg = {}
            wf.tcp = np.eye(4)
            wf.contact = np.array([.1, .2, .3])
            feedback = dict(joints_rad=[.1] * 7)
            image = np.zeros((480, 640, 3), np.uint8)
            with patch.object(flow.common, 'capture_with_feedback', return_value=feedback), \
                 patch.object(flow, 'load_batch', return_value=({'intrinsics': {}}, None, image, None)), \
                 patch.object(flow.common, 'camera_transform', return_value=(np.eye(4), True)), \
                 patch.object(flow.common, 'draw_tcp', return_value=(image, {})) as draw, \
                 patch.object(flow.common, 'show'), \
                 patch.object(flow.common, 'bridge', side_effect=AssertionError('No motion')):
                wf.step_tcp_view('APPROACH')
                self.assertIs(draw.call_args.args[2], feedback)
                np.testing.assert_equal(draw.call_args.args[3], wf.contact)
                self.assertTrue((wf.root / 'green_tcp_approach.png').is_file())
                self.assertTrue((wf.root / 'green_tcp_current.json').is_file())
                wf.step_tcp_view('GRIP')
                self.assertEqual(draw.call_args.kwargs['hand_state'], 'after_close_command')
