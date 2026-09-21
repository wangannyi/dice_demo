"""Capture configuration isolation and preservation of legacy stream defaults."""
import unittest
from pathlib import Path
from unittest.mock import patch
from dice_cup_localization.capture_rgbd import argument_parser
from cup_grasp_demo.calibration_debug.green_capture import capture_arguments
from cup_grasp_demo.calibration_debug import debug


class CaptureProfileTest(unittest.TestCase):
    def test_legacy_defaults(self):
        args = argument_parser().parse_args(['--output', '/tmp/not-used', '--serial', 'test'])
        self.assertEqual((args.depth_width, args.depth_height, args.fps), (640, 480, 15))
        self.assertEqual(capture_arguments({'green_cup': {'camera': {'fps': 30}}}), [])

    def test_green_only_command_and_calibrated_color_preserved(self):
        cfg = {'serial': 'test', 'pipeline_strategy': 'green_open_cup',
               'green_cup': {'camera': {'depth_resolution': [1280, 720], 'fps': 15}}}
        with patch.object(debug.subprocess, 'run') as run:
            debug.capture_rgbd(Path('/tmp/not-used'), cfg)
        args = argument_parser().parse_args(run.call_args.args[0][2:])
        self.assertEqual((args.depth_width, args.depth_height, args.fps), (1280, 720, 15))
        self.assertEqual(args.frames, 5)
        self.assertFalse(hasattr(args, 'color_width'))

    def test_redcloth_profile_reaches_capture_parser(self):
        cfg={'pipeline_strategy':'green_open_cup','green_cup':{'camera':{
            'color_resolution':[1280,720],'depth_resolution':[1280,720],
            'fps':15,'crop_xywh':[220,0,960,720]}}}
        args=argument_parser().parse_args(['--output','/tmp/not-used','--serial','test',*capture_arguments(cfg)])
        self.assertEqual(args.color_resolution,[1280,720])
        self.assertEqual(args.crop_xywh,[220,0,960,720])

    def test_ir_streams_are_opt_in(self):
        cfg = {'pipeline_strategy': 'green_open_cup', 'green_cup': {
            'perception': {'geometry_method': 'stereo_rim'}}}
        self.assertIn('--stereo', capture_arguments(cfg))
        cfg['green_cup']['perception']['geometry_method'] = 'image_rim_depth'
        self.assertNotIn('--stereo', capture_arguments(cfg))

    def test_invalid_profile_does_not_start_camera(self):
        for profile in ({'fps': True}, {'depth_resolution': [1280, 480]},
                        {'depth_resolution': [640.0, 480]}, {'color_resolution': [1920, 1080]}):
            with self.subTest(profile=profile), patch.object(debug.subprocess, 'run') as run:
                with self.assertRaises(ValueError):
                    debug.capture_rgbd(Path('/tmp/not-used'), {
                        'serial': 'test', 'pipeline_strategy': 'green_open_cup',
                        'green_cup': {'camera': profile}})
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
