"""Capture configuration reads vision/camera.json; camera.json is the only source."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from vision.capture.realsense_session import argument_parser
from cup_grasp_demo.flow.green_capture import capture_arguments
from cup_grasp_demo.flow import debug


class CaptureProfileTest(unittest.TestCase):
    def _camera(self, **overrides):
        base = dict(serial="test-serial", color_resolution=[1280, 720],
                    depth_resolution=[1280, 720], fps=6,
                    crop_xywh=[220, 0, 960, 720], warmup_frames=5,
                    fresh_discard_frames=0,
                    calibration_file="configs/calibration/handeye_result.json")
        base.update(overrides)
        return base

    def test_non_green_strategy_returns_empty(self):
        self.assertEqual(capture_arguments({'green_cup': {'camera': {'fps': 30}}}), [])

    def test_camera_json_drives_capture_arguments(self):
        cfg = {'serial': 'test', 'pipeline_strategy': 'green_open_cup',
               'green_cup': {'perception': {}}}
        with patch('vision.capture.config.load_camera_config',
                   return_value=self._camera()):
            args = argument_parser().parse_args(
                ['--output', '/tmp/x', '--serial', 's', *capture_arguments(cfg)])
        self.assertEqual((args.depth_width, args.depth_height, args.fps), (1280, 720, 6))
        self.assertEqual(args.crop_xywh, [220, 0, 960, 720])
        self.assertEqual(args.color_resolution, [1280, 720])

    def test_stereo_flag_when_geometry_method_is_stereo_rim(self):
        cfg = {'pipeline_strategy': 'green_open_cup',
               'green_cup': {'perception': {'geometry_method': 'stereo_rim'}}}
        with patch('vision.capture.config.load_camera_config',
                   return_value=self._camera()):
            argv = capture_arguments(cfg)
        self.assertIn('--stereo', argv)

    def test_camera_config_rejects_bad_values(self):
        from vision.capture.config import load_camera_config
        with patch('vision.capture.config.CAMERA_CONFIG') as mock_path:
            for bad in (self._camera(fps=25), self._camera(depth_resolution=[100, 200]),
                        self._camera(warmup_frames=100),
                        self._camera(serial=""), self._camera(unknown_key=1)):
                mock_path.read_text.return_value = json.dumps(bad)
                with self.assertRaises(ValueError):
                    load_camera_config()
