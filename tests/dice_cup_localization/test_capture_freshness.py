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
        from cup_grasp_demo.flow.green_pipeline import Workflow
        with tempfile.TemporaryDirectory() as d:
            w=object.__new__(Workflow);w.root=Path(d);w.args=SimpleNamespace(mode='fast');w._vision=Mock();w.g={}
            w._capture_once=Mock(side_effect=[ValueError('table_plane_not_supported'), 'ok'])
            self.assertEqual(w.capture(),'ok')
            self.assertEqual(w._capture_once.call_count,2)
            w._capture_once=Mock(side_effect=ValueError('table_plane_not_supported'))
            with self.assertRaises(ValueError):w.capture()
            self.assertEqual(w._capture_once.call_count,2)


def make_intr(w=64,h=48):
    from types import SimpleNamespace
    return SimpleNamespace(width=w,height=h,fx=100.0,fy=100.0,ppx=32.0,ppy=24.0,coeffs=[],model='none')

def rich_frame(data,number):
    f=Mock()
    f.get_data.return_value=data
    f.get_frame_number.return_value=number
    f.get_timestamp.return_value=float(number)
    f.get_frame_timestamp_domain.return_value='hardware'
    return f

def rich_frameset(color,depth,cn,dn):
    from types import SimpleNamespace
    fs=Mock()
    raw_c=rich_frame(color,cn);raw_d=rich_frame(depth,dn)
    fs.get_color_frame.return_value=raw_c;fs.get_depth_frame.return_value=raw_d
    aligned=Mock()
    aligned.get_color_frame.return_value=rich_frame(color,cn)
    aligned.get_depth_frame.return_value=rich_frame(depth,dn)
    fs._aligned=aligned
    aligned.get_color_frame.return_value.profile.as_video_stream_profile.return_value.get_intrinsics.return_value=make_intr()
    aligned.get_color_frame.return_value.profile.as_video_stream_profile.return_value.fps.return_value=6
    raw_d.profile.as_video_stream_profile.return_value.width.return_value=64
    raw_d.profile.as_video_stream_profile.return_value.height.return_value=48
    raw_d.profile.as_video_stream_profile.return_value.fps.return_value=6
    raw_d.profile.as_video_stream_profile.return_value.get_intrinsics.return_value=make_intr()
    raw_d.profile.as_video_stream_profile.return_value.get_extrinsics_to.return_value=SimpleNamespace(
        rotation=[1,0,0,0,1,0,0,0,1],translation=[0,0,0])
    return fs

def fake_rs(wait):
    rs=Mock()
    pipeline=rs.pipeline.return_value
    pipeline.wait_for_frames.side_effect=wait
    pipeline.poll_for_frames.return_value=None
    device=pipeline.start.return_value.get_device.return_value
    device.first_depth_sensor.return_value.get_depth_scale.return_value=0.001
    device.get_info.return_value='2.0'
    rs.align.return_value.process.side_effect=lambda fs: fs._aligned
    return rs


class StreamingReaderTests(unittest.TestCase):
    def make_args(self):
        return argument_parser().parse_args(
            ['--serial','t','--output','/tmp/unused','--fps','6',
             '--warmup-frames','2','--fresh-discard-frames','0'])

    def test_reader_streams_idle_frames(self):
        import queue as pyq, time
        import numpy as np
        from dice_cup_localization.capture_rgbd import CaptureSession
        color=np.zeros((48,64,3),np.uint8);depth=np.zeros((48,64),np.uint16)
        q=pyq.Queue()
        rs=fake_rs(lambda t=3000:q.get(timeout=2))
        received=[]
        with patch.dict('sys.modules',pyrealsense2=rs):
            camera=CaptureSession(self.make_args())
            camera.set_stream_sink(received.append)
            q.put(rich_frameset(color,depth,1,1));q.put(rich_frameset(color,depth,2,2))
            camera.start()
            q.put(rich_frameset(color,depth,3,3));q.put(rich_frameset(color,depth,4,4))
            deadline=time.monotonic()+3
            while len(received)<2 and time.monotonic()<deadline: time.sleep(0.02)
            camera.close()
        self.assertEqual(len(received),2)
        self.assertEqual(received[0].shape,(48,64,3))
        self.assertTrue(np.array_equal(received[0],color))

    def test_reader_serves_capture_request_without_announce(self):
        import io, json, queue as pyq, tempfile, threading, time
        from pathlib import Path
        import numpy as np
        from dice_cup_localization.capture_rgbd import CaptureSession
        color=np.zeros((48,64,3),np.uint8);depth=np.zeros((48,64),np.uint16)
        q=pyq.Queue()
        rs=fake_rs(lambda t=3000:q.get(timeout=2))
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules',pyrealsense2=rs):
            camera=CaptureSession(self.make_args())
            camera.set_stream_sink(lambda frame: None)
            q.put(rich_frameset(color,depth,1,1));q.put(rich_frameset(color,depth,2,2))
            camera.start()
            out=Path(d)/'cap'
            result={}
            def run():
                try:
                    camera.capture(out,2,fresh=True);result['ok']=True
                except BaseException as exc:
                    result['err']=exc
            announced=io.StringIO()
            with patch('sys.stdout',announced):
                worker=threading.Thread(target=run);worker.start()
                for number in (10,11,12,13,14):
                    time.sleep(0.15);q.put(rich_frameset(color,depth,number,number))
                worker.join(20)
            camera.close()
            files=sorted(p.name for p in out.glob('frame_*'))
            meta=json.loads((out/'frame_001.json').read_text())
        self.assertTrue(result.get('ok'),result)
        self.assertEqual(files,['frame_000.json','frame_000.npz','frame_000.png',
                                'frame_001.json','frame_001.npz','frame_001.png'])
        self.assertEqual(meta['serial'],'t')
        self.assertEqual(meta['capture_timing']['fresh_discard_frames'],0)
        self.assertEqual(announced.getvalue(),'')  # reader thread must never print

    def test_legacy_capture_without_sink_still_announces(self):
        import io, json, tempfile
        from pathlib import Path
        import numpy as np
        from dice_cup_localization.capture_rgbd import CaptureSession
        color=np.zeros((48,64,3),np.uint8);depth=np.zeros((48,64),np.uint16)
        feed=iter([rich_frameset(color,depth,1,1),rich_frameset(color,depth,2,2),
                   rich_frameset(color,depth,3,3)])
        rs=fake_rs(lambda t=3000:next(feed))
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules',pyrealsense2=rs):
            camera=CaptureSession(self.make_args())
            camera.start()
            out=Path(d)/'legacy'
            announced=io.StringIO()
            with patch('sys.stdout',announced):
                camera.capture(out,1,fresh=False)
            camera.close()
            png_ok=(out/'frame_000.png').is_file()
            npz_ok=(out/'frame_000.npz').is_file()
            meta=json.loads((out/'frame_000.json').read_text())
            announced_text=announced.getvalue()
        self.assertTrue(png_ok and npz_ok)
        self.assertIn('frame_000',announced_text)
        self.assertEqual(meta['serial'],'t')
