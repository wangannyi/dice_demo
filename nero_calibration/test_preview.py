"""Collection UI tests; all camera, CAN and X11 access is mocked."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

from calibrate import main, write_new
from core import pose_matrix
from preview import CollectionPreview


ROOT = Path(__file__).resolve().parent


class PreviewTests(unittest.TestCase):
    def test_preview_preserves_rejection_resume_and_saved_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            board = json.loads((ROOT/'config/board_perceptive.json').read_text())
            camera_info = {'serial': 'mock-camera'}
            write_new(path/'manifest.json', {'schema': 1, 'mode': 'eye_to_hand',
                      'camera': camera_info, 'board': board, 'tcp': 'flange',
                      'T_flange_tcp': np.eye(4).tolist()})
            quality = {'corners': 12, 'reprojection_rms_px': .1}
            sample = {'T_base_flange': np.eye(4).tolist(),
                      'T_base_tcp': np.eye(4).tolist(),
                      'T_camera_board': pose_matrix([0, 0, .5, 0, 0, 0]).tolist(),
                      'quality': quality}
            write_new(path/'sample_0000.json', sample)
            (path/'sample_0000.png').write_bytes(b'old raw')
            (path/'sample_0000_detected.png').write_bytes(b'old overlay')
            original = {p.name: p.read_bytes() for p in path.iterdir()}
            fresh = {**sample, 'T_base_flange': pose_matrix([.03, 0, 0, 0, 0, 0]).tolist()}
            image = np.zeros((480, 640, 3), np.uint8)
            cam, arm, ui = Mock(), Mock(), Mock()
            cam.info = camera_info
            ui.read_command.side_effect = ['', '', 'q']
            with patch('sensors.RealSenseCamera', return_value=cam), \
                 patch('sensors.NeroFeedback', return_value=arm), \
                 patch('preview.CollectionPreview', return_value=ui), \
                 patch('cv2.imread', return_value=image), \
                 patch('calibrate.capture_sample', side_effect=[(sample, image, image),
                                                               (fresh, image, image)]), \
                 patch('builtins.input') as terminal_input, patch('builtins.print'):
                self.assertEqual(main(['collect', '--resume', '--preview',
                                       '--dataset', str(path)]), 0)
            self.assertEqual([c.args[0] for c in ui.result.call_args_list], [False, True])
            self.assertEqual([c.args[1] for c in ui.result.call_args_list], [1, 2])
            self.assertEqual(ui.restore.call_args.args[1], 1)
            self.assertEqual(len(list(path.glob('sample_*.json'))), 2)
            for name, data in original.items():
                self.assertEqual((path/name).read_bytes(), data)
            terminal_input.assert_not_called()
            ui.close.assert_called_once()
            cam.close.assert_called_once()
            arm.close.assert_called_once()

    def test_missing_x11_fails_before_camera_and_arm_open(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), \
             patch('sensors.RealSenseCamera') as camera, patch('sensors.NeroFeedback') as arm:
            with self.assertRaisesRegex(RuntimeError, 'ssh -X'):
                main(['collect', '--preview', '--dataset', str(Path(tmp)/'new')])
            camera.assert_not_called()
            arm.assert_not_called()

    def make_ui(self):
        with patch.dict(os.environ, {'DISPLAY': ':test'}), \
             patch('cv2.namedWindow'), patch('cv2.resizeWindow'):
            return CollectionPreview()

    def test_enter_can_attempt_capture_when_live_detection_fails(self):
        ui = self.make_ui()
        camera, detector = Mock(), Mock()
        frame = np.full((480, 640, 3), 70, np.uint8)
        camera.capture.return_value = frame
        detector.detect.side_effect = ValueError('Too few ChArUco corners')
        detector.roi = [0, 0, 440, 480]
        with patch('cv2.getWindowProperty', return_value=1), \
             patch('cv2.imshow') as show, patch('cv2.waitKey', side_effect=[13, -1]):
            self.assertEqual(ui.read_command(camera, detector, 3), '')
        self.assertEqual(show.call_count, 2)
        self.assertEqual(show.call_args.args[1].shape, (640, 1280, 3))
        np.testing.assert_array_equal(frame, np.full_like(frame, 70))

    def test_window_close_exits_without_capturing(self):
        ui = self.make_ui()
        camera = Mock()
        with patch('cv2.getWindowProperty', return_value=0):
            self.assertEqual(ui.read_command(camera, Mock(), 2), 'q')
        camera.capture.assert_not_called()

    def test_qt_receiver_removed_on_close_exits_without_capturing(self):
        ui = self.make_ui()
        camera = Mock()
        with patch('cv2.getWindowProperty', side_effect=cv2.error('NULL guiReceiver')):
            self.assertEqual(ui.read_command(camera, Mock(), 2), 'q')
        camera.capture.assert_not_called()

    def test_ui_failure_still_closes_arm_and_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            camera, arm, ui = Mock(), Mock(), Mock()
            camera.info = {'serial': 'mock-camera'}
            ui.read_command.side_effect = RuntimeError('Display disconnected')
            with patch('sensors.RealSenseCamera', return_value=camera), \
                 patch('sensors.NeroFeedback', return_value=arm), \
                 patch('preview.CollectionPreview', return_value=ui), patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError, 'Display disconnected'):
                    main(['collect', '--preview', '--dataset', str(Path(tmp)/'new')])
            ui.close.assert_called_once()
            camera.close.assert_called_once()
            arm.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
