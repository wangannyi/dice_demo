import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from cup_grasp_demo.flow.green_rtsp import RtspStreamer, rtsp_settings


class RtspSettingsTests(unittest.TestCase):
    def test_defaults_are_off(self):
        self.assertEqual(rtsp_settings({}),
                         dict(enabled=False, host='127.0.0.1', port=8554, path='/dice/seg'))

    def test_accepts_full_section(self):
        settings = rtsp_settings({'enabled': True, 'host': '127.0.0.1', 'port': 8554,
                                  'path': '/dice/seg'})
        self.assertTrue(settings['enabled'])

    def test_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            rtsp_settings({'bitrate': 4_000_000})

    def test_rejects_bad_port_or_path(self):
        for bad in ({'port': 'x'}, {'port': 0}, {'port': 65536},
                    {'path': 'dice/seg'}, {'path': '/dice /seg'}, {'host': ''}):
            with self.assertRaises(ValueError):
                rtsp_settings(bad)


class RtspStreamerTests(unittest.TestCase):
    kwargs = dict(host='127.0.0.1', port=8554, path='/dice/seg',
                  width=320, height=240, fps=6)

    def tearDown(self):
        streamer = getattr(self, 'streamer', None)
        if streamer is not None:
            streamer.close()

    def make(self, **overrides):
        kwargs = dict(self.kwargs)
        kwargs.update(overrides)
        self.streamer = RtspStreamer(**kwargs)
        return self.streamer

    def test_command_mirrors_segdetect_pipeline(self):
        streamer = self.make()
        command = streamer.command()
        self.assertEqual(command[0], 'gst-launch-1.0')
        self.assertIn('blocksize=115200', command)  # 320*240*3//2 (NV12)
        parser = command.index('rawvideoparse')
        self.assertEqual(command[parser + 1:parser + 5],
                         ['format=nv12', 'width=320', 'height=240', 'framerate=6/1'])
        self.assertNotIn('videoconvert', command)  # conversion happens in-process
        encoder = command.index('spacemith264enc')
        self.assertEqual(command[encoder + 1], 'coding-width=320')
        self.assertEqual(command[encoder + 2], 'code-hight=240')  # vendor spelling
        self.assertIn('h264parse', command)
        self.assertEqual(command[command.index('h264parse') + 1], 'config-interval=-1')
        sink = command.index('rtspclientsink')
        self.assertEqual(command[sink + 1], 'location=rtsp://127.0.0.1:8554/dice/seg')
        self.assertEqual(command[sink + 2], 'protocols=tcp')
        self.assertEqual(command[sink + 3], 'latency=0')

    def test_to_nv12_layout_and_neutral_chroma(self):
        import numpy as np
        gray = np.full((240, 320, 3), 128, np.uint8)
        payload = RtspStreamer._to_nv12(gray)
        self.assertEqual(len(payload), 320 * 240 * 3 // 2)
        y = np.frombuffer(payload[:320 * 240], np.uint8)
        uv = np.frombuffer(payload[320 * 240:], np.uint8)
        self.assertTrue(float(uv.min()) == 128.0 and float(uv.max()) == 128.0)
        self.assertTrue(110 <= float(y.mean()) <= 135)  # limited-range luma of gray 128
        red = np.zeros((240, 320, 3), np.uint8)
        red[..., 2] = 200  # BGR red
        yuv = RtspStreamer._to_nv12(red)
        chroma = np.frombuffer(yuv[320 * 240:], np.uint8)
        u, v = chroma[0::2], chroma[1::2]
        self.assertGreater(float(v.mean()), float(u.mean()))  # red lifts V over U

    def test_submit_keeps_only_newest_frame(self):
        # Build with a mocked Thread so the writer never runs; the slot
        # replacement semantics are then observed deterministically.
        with patch('cup_grasp_demo.flow.green_rtsp.threading.Thread') as thread_cls:
            streamer = self.make()
            self.assertTrue(thread_cls.called)
        first = np.zeros((240, 320, 3), np.uint8)
        second = np.ones((240, 320, 3), np.uint8)
        streamer.submit(first)
        streamer.submit(second)
        self.assertIs(streamer._slot, second)
        self.assertEqual(streamer.dropped, 1)
        self.assertEqual(streamer.sent, 0)
        streamer.close()

    def test_writer_sends_frame_bytes_to_child_stdin(self):
        streamer = self.make()
        fake_stdin = Mock()
        fake_proc = Mock()
        fake_proc.poll.return_value = None
        fake_proc.stdin = fake_stdin
        with patch('cup_grasp_demo.flow.green_rtsp.subprocess.Popen',
                   return_value=fake_proc) as popen:
            streamer.submit(np.full((240, 320, 3), 7, np.uint8))
            deadline = time.monotonic() + 3
            while streamer.sent < 1 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(streamer.sent, 1)
            self.assertEqual(popen.call_count, 1)
            self.assertEqual(popen.call_args.kwargs['stdout'], subprocess_devnull())
            written = fake_stdin.write.call_args[0][0]
            self.assertEqual(len(written), 320 * 240 * 3 // 2)  # NV12 payload
            self.assertTrue(any(written))  # not all-zero pixels
            streamer.close()
            fake_proc.terminate.assert_called()
        self.assertFalse(streamer._thread.is_alive())

    def test_spawn_failure_backs_off_before_retry(self):
        streamer = self.make(restart_backoff_s=30.0)
        attempts = []

        def refused(*args, **kwargs):
            attempts.append(time.monotonic())
            raise OSError('gst-launch missing')

        with patch('cup_grasp_demo.flow.green_rtsp.subprocess.Popen',
                   side_effect=refused):
            streamer.submit(np.zeros((240, 320, 3), np.uint8))
            deadline = time.monotonic() + 3
            while not attempts and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(len(attempts), 1)
            streamer.submit(np.zeros((240, 320, 3), np.uint8))
            time.sleep(0.3)
            self.assertEqual(len(attempts), 1)  # inside backoff: no respawn
            self.assertTrue(streamer.last_error)
        streamer.close()

    def test_child_death_is_detected_and_respawned_after_backoff(self):
        streamer = self.make(restart_backoff_s=0.0)
        first_stdin = Mock()
        first_proc = Mock()
        first_proc.poll.return_value = None
        first_proc.stdin = first_stdin
        first_stdin.write.side_effect = BrokenPipeError('child died')
        second_proc = Mock()
        second_proc.poll.return_value = None
        with patch('cup_grasp_demo.flow.green_rtsp.subprocess.Popen',
                   side_effect=[first_proc, second_proc]):
            streamer.submit(np.zeros((240, 320, 3), np.uint8))
            deadline = time.monotonic() + 3
            while streamer._spawned < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            first_proc.terminate.assert_called()  # dead child cleaned up
            streamer.submit(np.zeros((240, 320, 3), np.uint8))
            deadline = time.monotonic() + 3
            while streamer.sent < 1 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(streamer.sent, 1)
        streamer.close()
        second_proc.terminate.assert_called()


def subprocess_devnull():
    import subprocess
    return subprocess.DEVNULL


if __name__ == '__main__':
    unittest.main()
