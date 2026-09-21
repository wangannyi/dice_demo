import unittest
from unittest.mock import Mock, patch
from dice_cup_localization.capture_rgbd import argument_parser, next_frames

def frame(c, d):
    f=Mock()
    f.get_color_frame.return_value.get_frame_number.return_value=c
    f.get_depth_frame.return_value.get_frame_number.return_value=d
    return f

class FreshnessTests(unittest.TestCase):
    def test_legacy_defaults(self):
        a=argument_parser().parse_args(['--serial','test','--output','/tmp/test'])
        self.assertEqual((a.warmup_frames,a.fresh_discard_frames),(20,2))

    def test_next_new_frame_not_discarded(self):
        pipeline=Mock();wanted=frame(11,11)
        pipeline.wait_for_frames.return_value=wanted
        self.assertIs(next_frames(pipeline,(10,10)),wanted)
        self.assertEqual(pipeline.wait_for_frames.call_count,1)

    def test_both_streams_must_advance(self):
        pipeline=Mock();wanted=frame(13,12)
        pipeline.wait_for_frames.side_effect=[frame(10,10),frame(12,10),wanted]
        self.assertIs(next_frames(pipeline,(10,10)),wanted)
        self.assertEqual(pipeline.wait_for_frames.call_count,3)

    def test_stale_frames_timeout(self):
        pipeline=Mock();pipeline.wait_for_frames.return_value=frame(10,10)
        with patch('dice_cup_localization.capture_rgbd.time.monotonic',side_effect=[0,1,4]):
            with self.assertRaises(TimeoutError):next_frames(pipeline,(10,10))

    def test_fast_warmup_counts_all_streams_not_api_returns(self):
        from dice_cup_localization.capture_rgbd import CaptureSession
        args=argument_parser().parse_args(['--serial','test','--output','/tmp/test','--stereo','--warmup-frames','5'])
        args.unique_warmup=True
        def pair(c,d,ir):
            f=frame(c,d)
            f.get_infrared_frame.return_value.get_frame_number.return_value=ir
            return f
        rs=Mock()
        rs.pipeline.return_value.wait_for_frames.side_effect=[pair(1,1,1),pair(2,1,1),pair(3,2,1),pair(4,3,2),pair(5,4,3),pair(6,5,4),pair(7,6,5)]
        with patch.dict('sys.modules',pyrealsense2=rs):
            camera=CaptureSession(args);camera.start();camera.close()
        self.assertEqual(rs.pipeline.return_value.wait_for_frames.call_count,7)

    def test_legacy_warmup_keeps_twenty_calls(self):
        from dice_cup_localization.capture_rgbd import CaptureSession
        args=argument_parser().parse_args(['--serial','test','--output','/tmp/test'])
        rs=Mock()
        with patch.dict('sys.modules',pyrealsense2=rs):
            camera=CaptureSession(args);camera.start();camera.close()
        self.assertEqual(rs.pipeline.return_value.wait_for_frames.call_count,20)


    def test_capture_retries_only_once_on_transient_fast_failure(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from cup_grasp_demo.calibration_debug.green_pipeline import Workflow
        with tempfile.TemporaryDirectory() as d:
            w=object.__new__(Workflow);w.root=Path(d);w.args=SimpleNamespace(mode='fast');w._vision=Mock()
            w._capture_once=Mock(side_effect=[ValueError('table_plane_not_supported'), 'ok'])
            self.assertEqual(w.capture(),'ok')
            self.assertEqual(w._capture_once.call_count,2)
            w._capture_once=Mock(side_effect=ValueError('table_plane_not_supported'))
            with self.assertRaises(ValueError):w.capture()
            self.assertEqual(w._capture_once.call_count,2)
            w.args.mode='step';w._capture_once.reset_mock()
            with self.assertRaises(ValueError):w.capture()
            self.assertEqual(w._capture_once.call_count,1)
