"""Feedback projection, display boundaries and read-only camera integration."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from cup_grasp_demo.calibration_debug import debug
from cup_grasp_demo.calibration_debug.core import ROOT, read_json
from cup_grasp_demo.calibration_debug.tcp_overlay import draw_tcp, project
from nero_revo2_control.kinematics import load_model


class TcpOverlayTest(unittest.TestCase):
    def setUp(self):
        self.session = dict(T_base_camera=np.eye(4).tolist(), T_flange_tcp=np.eye(4).tolist(),
                            intrinsics=dict(width=640, height=480, fx=500, fy=500, cx=320, cy=240))
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.snapshot = dict(joints_rad=[0.] * 7, observed_epoch_s=123.)

    def test_known_projection_uses_inverse_extrinsic(self):
        self.session['T_base_camera'][0][3] = .1
        result = project([.3, .1, 1.], self.session)
        np.testing.assert_allclose(result['pixel'], [420, 290])
        self.assertEqual(result['visibility'], 'in_frame')

    def test_out_of_frame_and_behind_camera_are_not_clamped(self):
        outside = project([1, 0, 1], self.session)
        self.assertEqual(outside['visibility'], 'outside_image')
        self.assertGreater(outside['pixel'][0], 640)
        behind = project([0, 0, -1], self.session)
        self.assertEqual(behind['visibility'], 'behind_camera')
        self.assertIsNone(behind['pixel'])

    def test_feedback_not_target_defines_tcp_and_j7_rotates_it(self):
        self.session['T_flange_tcp'][0][3] = .17
        arm = load_model()
        points = []
        for j7 in (0, .5):
            self.snapshot['joints_rad'][6] = j7
            _, report = draw_tcp(self.image, self.session, self.snapshot, target=[0, 0, 1])
            expected = np.asarray(arm.fk(self.snapshot['joints_rad'])) @ self.session['T_flange_tcp']
            np.testing.assert_allclose(report['tcp']['base_m'], expected[:3, 3])
            self.assertFalse(report['physical_tcp_measured'])
            self.assertEqual(report['observed_epoch_s'], 123)
            points.append(report['tcp']['base_m'])
        self.assertGreater(np.linalg.norm(np.subtract(*points)), .05)

    def test_separate_marker_colors_and_closed_hand_reference(self):
        fk = np.eye(4)
        fk[2, 3] = 1
        with patch('cup_grasp_demo.calibration_debug.tcp_overlay.load_model') as model:
            model.return_value.fk.return_value = fk
            image, report = draw_tcp(self.image, self.session, self.snapshot, [.1, 0, 1], 'after_close_command')
        np.testing.assert_array_equal(image[240, 320], [255, 0, 255])
        np.testing.assert_array_equal(image[240, 370], [0, 0, 255])
        self.assertEqual(report['hand_state'], 'after_close_command')
        self.assertTrue(report['fixed_open_hand_tcp'])
        self.assertFalse(report['physical_tcp_measured'])
        self.assertFalse(np.any(self.image))

    def test_home_marker_at_top_is_not_obscured_by_legend(self):
        fk = np.eye(4)
        fk[1, 3] = (17 - 240) / 500
        fk[2, 3] = 1
        with patch('cup_grasp_demo.calibration_debug.tcp_overlay.load_model') as model:
            model.return_value.fk.return_value = fk
            image, _ = draw_tcp(self.image, self.session, self.snapshot)
        np.testing.assert_array_equal(image[17, 320], [255, 0, 255])
        self.assertTrue(np.any(image[390:460]))

    def test_invalid_feedback_dimensions_and_distortion_rejected(self):
        for q in ([0] * 6, [float('nan')] * 7):
            with self.assertRaisesRegex(ValueError, 'feedback'):
                draw_tcp(self.image, self.session, dict(joints_rad=q))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            draw_tcp(self.image[:100], self.session, self.snapshot)
        self.session['intrinsics']['dist_coeffs'] = [.1, 0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, 'distortion'):
            project([0, 0, 1], self.session)

    def test_camera_bracket_rejects_motion_before_overlay(self):
        before = dict(self.snapshot, arm_status=0, motion_status=0, joints_enabled=[True] * 7, ctrl_mode=1)
        after = deepcopy(before)
        after['joints_rad'][0] += .1
        with patch.object(debug, 'bridge', side_effect=[before, after]) as sdk, \
             patch.object(debug, 'capture_rgbd'):
            with self.assertRaisesRegex(ValueError, '发生运动'):
                debug.capture_with_feedback(Path('/unused/rgbd'), {})
        self.assertEqual([call.args[0] for call in sdk.call_args_list], ['snapshot', 'snapshot'])

    def test_tcp_view_works_at_non_home_and_never_sends_run(self):
        fixture = ROOT / 'cup_grasp_demo/datasets/contact_yaw20_20260917_01/current'
        saved = fixture / 'runs/20260917_234925_execute_grasp_7161d7'
        actual = read_json(saved / 'actual.json')
        snapshot = dict(joints_rad=actual['final_joints_rad'], arm_status=0, motion_status=0,
                        joints_enabled=[True] * 7, ctrl_mode=1, observed_epoch_s=123.)
        cfg = ROOT / 'cup_grasp_demo/calibration_debug/index_joint_center/config.json'
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            def capture(output, _cfg):
                shutil.copytree(saved / 'after_rgbd', output)
            with patch.object(debug, 'new_run', return_value=run), \
                 patch.object(debug, 'bridge', return_value=snapshot) as sdk, \
                 patch.object(debug, 'capture_rgbd', side_effect=capture), \
                 patch.object(debug, 'show'):
                debug.tcp_view(SimpleNamespace(session=None, config=cfg, output=run, show=False))
            self.assertEqual([call.args[0] for call in sdk.call_args_list], ['snapshot', 'snapshot'])
            self.assertTrue((run / 'tcp_view.png').exists())
            self.assertEqual(read_json(run / 'tcp_view.json')['joints_rad'], actual['final_joints_rad'])
            self.assertFalse((run / 'request.json').exists())
            self.assertIsNotNone(cv2.imread(str(run / 'tcp_view.png')))


if __name__ == '__main__':
    unittest.main()
