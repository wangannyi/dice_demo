"""Focused offline checks for RGB-D units and the first scene masks."""

from contextlib import redirect_stdout
import importlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from cup_grasp_demo import grasp, vision


def bgr_from_hsv(h, s, v):
    return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]


class MaskTests(unittest.TestCase):
    def test_navy_cup_and_nearby_red_table_separate(self):
        image = np.full((40, 60, 3), bgr_from_hsv(3, 160, 170), dtype=np.uint8)
        image[0:10, :] = bgr_from_hsv(100, 210, 185)  # bright blue table
        image[13:29, 20:40] = bgr_from_hsv(105, 180, 120)  # navy cup top
        image[22:29, 20:40] = bgr_from_hsv(120, 150, 40)  # dark cup side
        image[14:22, 45:55] = bgr_from_hsv(110, 80, 30)  # another dark object
        cup = vision.segment_dark_cup(image, bbox=(15, 10, 43, 32))
        self.assertTrue(cup[17, 25])
        self.assertTrue(cup[25, 25])
        self.assertFalse(cup[17, 50])
        self.assertFalse(cup[5, 25])

        depth = np.full((40, 60), 0.50, dtype=np.float32)
        depth[cup] = 0.43
        depth[31:39, :] = 1.20  # red surface at a distant depth is not nearby
        table = vision.segment_table(image, cup, bbox=(10, 10, 50, 40), depth_m=depth)
        self.assertTrue(table[12, 19])
        self.assertFalse(table[17, 25])
        self.assertFalse(table[35, 19])
        self.assertFalse(np.any(table & cup))

    def test_green_roi_and_external_png_mask(self):
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        image[1:4, 1:5] = bgr_from_hsv(65, 200, 160)
        image[4:6, 1:5] = bgr_from_hsv(65, 200, 28)  # dark green lower cup side
        image[1:4, 8:11] = bgr_from_hsv(65, 200, 160)
        mask = vision.segment_green_cup(image, bbox=(0, 0, 6, 6))
        self.assertTrue(mask[2, 3])
        self.assertTrue(mask[5, 3])
        self.assertFalse(mask[2, 9])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mask.png'
            self.assertTrue(cv2.imwrite(str(path), mask.astype(np.uint8) * 255))
            np.testing.assert_array_equal(vision.load_mask_png(path, image.shape[:2]), mask)
            with self.assertRaises(ValueError):
                vision.load_mask_png(path, (9, 12))

    def test_depth_refinement_adds_only_foreground_near_seed(self):
        image = np.full((80, 100, 3), bgr_from_hsv(3, 160, 170), dtype=np.uint8)
        image[17:41, 32:54] = bgr_from_hsv(110, 170, 110)
        depth = np.full((80, 100), 0.50, dtype=np.float32)
        depth[17:41, 32:54] = 0.44
        depth[25:35, 31] = 0.44  # red-colored silhouette edge; depth says foreground
        depth[25:35, 30] = 0.50  # further red pixels remain table, outside dilation
        table = np.zeros((80, 100), dtype=bool)
        table[45:75, 20:80] = True
        intr = {'fx': 150., 'fy': 150., 'cx': 50., 'cy': 40.,
                'width': 100, 'height': 80, 'frame': 'color_optical',
                'dist_coeffs': [0., 0., 0., 0., 0.]}
        seed = vision.segment_dark_cup(image, (25, 10, 65, 45))
        refined, quality = vision.segment_dark_cup_depth_refined(
            image, (25, 10, 65, 45), depth, table, intr)
        self.assertFalse(seed[30, 31])
        self.assertTrue(refined[30, 31])
        self.assertFalse(refined[30, 30])
        self.assertFalse(np.any(refined & table))
        self.assertGreater(quality['added_pixels'], 0)
        self.assertEqual(quality['refined_pixels'], int(refined.sum()))
        self.assertGreater(quality['table_plane_inlier_fraction'], 0.8)
        self.assertIn('local_table_depth_foreground', quality['mask_source'])
        with self.assertRaisesRegex(ValueError, 'zero distortion'):
                vision.segment_dark_cup_depth_refined(
                    image, (25, 10, 65, 45), depth, table,
                    {**intr, 'dist_coeffs': [0.01, 0, 0, 0, 0]})

    def test_provisional_red_table_filter_reports_raw_fraction(self):
        image = np.full((80, 100, 3), bgr_from_hsv(3, 160, 170), dtype=np.uint8)
        image[10:40, 40:60] = bgr_from_hsv(65, 220, 75)
        cup = vision.segment_green_cup(image, bbox=(35, 8, 65, 42))
        depth = np.full((80, 100), 0.50, dtype=np.float32)
        depth[45:75, 10:90] = 0.50
        depth[45:75, 10:90][::3, :] = 0.54  # 1/3 of color-qualified mat is out-of-plane
        intr = {'fx': 150., 'fy': 150., 'cx': 50., 'cy': 40.,
                'width': 100, 'height': 80, 'frame': 'color_optical',
                'dist_coeffs': [0., 0., 0., 0., 0.]}
        filtered, quality = vision.segment_red_table_plane_supported(
            image, cup, (10, 45, 90, 75), depth, intr)
        self.assertFalse(np.any(filtered & cup))
        self.assertGreater(filtered.sum(), 100)
        self.assertLess(filtered.sum(), quality['candidate_pixels'])
        self.assertGreater(quality['raw_plane_inlier_fraction'], 0.5)
        self.assertLess(quality['raw_plane_inlier_fraction'], 0.8)
        self.assertFalse(quality['complete_cup_height_verified'])
        self.assertIn('provisional', quality['mask_source'])


class CaptureTests(unittest.TestCase):
    def test_snapshot_cli_preserves_default_and_passes_explicit_long_warmup(self):
        with (patch.object(grasp.vision, 'capture_frame', return_value={'fake': True}) as capture,
              patch.object(grasp, 'save_snapshot', return_value={'schema': 1}),
              redirect_stdout(io.StringIO())):
            self.assertEqual(grasp.main(['snapshot', '--dataset', '/tmp/offline-default']), 0)
            capture.assert_called_once_with('346222071954', timeout_ms=3000)
            capture.reset_mock()
            self.assertEqual(grasp.main([
                'snapshot', '--dataset', '/tmp/offline-warmup',
                '--warmup-frames', '30', '--timeout-ms', '9000',
            ]), 0)
            capture.assert_called_once_with(
                '346222071954', timeout_ms=9000, warmup_frames=30)

    def test_realsense_import_is_lazy(self):
        with patch.dict(sys.modules, {'pyrealsense2': None}):
            importlib.reload(vision)

    def test_capture_returns_raw_z16_and_scaled_metres_once(self):
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.uint16([[1000, 0, 1500], [2000, 500, 1000]])

        class FakeFrame:
            def __init__(self, data, number, timestamp, domain='hardware_clock'):
                self.data, self.number, self.timestamp, self.domain = data, number, timestamp, domain

            def get_data(self):
                return self.data

            def get_frame_number(self):
                return self.number

            def get_timestamp(self):
                return self.timestamp

            def get_frame_timestamp_domain(self):
                return self.domain

        class FakeFrames:
            def __init__(self, pair):
                self.pair = pair

            def get_color_frame(self):
                return FakeFrame(color, *self.pair[0])

            def get_depth_frame(self):
                return FakeFrame(depth, *self.pair[1])

        class FakePipeline:
            stopped = False
            calls = 0
            pairs = [
                ((0, 100.), (0, 270.)),  # startup, 170 ms unsynced
                ((1, 110.), (1, 280.)),
                ((2, 120.), (2, 290.)),
                ((3, 130.), (3, 300.)),  # must retry beyond warmup
                ((4, 140., 'hardware_clock'), (4, 141., 'system_time')),
                ((42, 101.), (43, 102.)),  # accepted pair
            ]

            def start(self, _):
                return profile

            def wait_for_frames(self, _):
                pair = self.pairs[self.calls]
                self.calls += 1
                return FakeFrames(pair)

            def stop(self):
                self.stopped = True

        class FakeConfig:
            def enable_device(self, serial):
                self.serial = serial

            def enable_stream(self, *args):
                pass

        class FakeVideoProfile:
            def as_video_stream_profile(self):
                return self

            def get_intrinsics(self):
                return types.SimpleNamespace(fx=600, fy=601, ppx=1, ppy=1,
                                             width=3, height=2,
                                             coeffs=[0, 0.01, 0, 0, 0],
                                             model='inverse_brown_conrady')

        class FakeSensor:
            def get_depth_scale(self):
                return 0.001

        class FakeDevice:
            def first_depth_sensor(self):
                return FakeSensor()

            def get_info(self, _):
                return 'fake-serial'

        class FakeProfile:
            def get_stream(self, _):
                return FakeVideoProfile()

            def get_device(self):
                return FakeDevice()

        class FakeAlign:
            def __init__(self, _):
                pass

            def process(self, frames):
                return frames

        profile = FakeProfile()
        pipeline = FakePipeline()
        fake_rs = types.SimpleNamespace(
            pipeline=lambda: pipeline, config=FakeConfig, align=FakeAlign,
            stream=types.SimpleNamespace(color='color', depth='depth'),
            format=types.SimpleNamespace(bgr8='bgr8', z16='z16'),
            camera_info=types.SimpleNamespace(serial_number='serial_number'),
        )
        with patch.dict(sys.modules, {'pyrealsense2': fake_rs}):
            frame = vision.capture_frame('fake-serial', width=3, height=2, fps=15)
        self.assertTrue(pipeline.stopped)
        self.assertEqual(frame['depth_raw'].dtype, np.uint16)
        np.testing.assert_array_equal(frame['depth_raw'], depth)
        np.testing.assert_allclose(frame['depth_m'], depth.astype(np.float32) * 0.001)
        self.assertEqual(frame['frame_id'], 42)
        self.assertEqual(frame['frame_ids'], {'color': 42, 'depth_aligned': 43})
        self.assertEqual(frame['timestamps_ms'], {'color': 101.0, 'depth_aligned': 102.0})
        self.assertEqual(frame['sync_delta_ms'], 1.0)
        self.assertEqual(frame['sync_limit_ms'], 40.0)
        self.assertEqual(frame['framesets_seen'], 6)
        self.assertEqual(frame['warmup_frames'], 3)
        self.assertEqual(pipeline.calls, 6)
        self.assertEqual(frame['intrinsics']['fx'], 600.0)
        self.assertEqual(frame['intrinsics']['dist_coeffs'], [0.0, 0.01, 0.0, 0.0, 0.0])
        self.assertEqual(frame['intrinsics']['distortion_model'], 'inverse_brown_conrady')
        self.assertEqual(frame['serial'], 'fake-serial')

    def test_capture_rejects_unsynced_frames_and_closes_pipeline(self):
        def frame(number, timestamp):
            return types.SimpleNamespace(
                get_frame_number=lambda: number,
                get_timestamp=lambda: timestamp,
                get_frame_timestamp_domain=lambda: 'hardware_clock',
            )

        frames = types.SimpleNamespace(
            get_color_frame=lambda: frame(12, 1000.),
            get_depth_frame=lambda: frame(13, 1170.),
        )
        profile = types.SimpleNamespace(
            get_stream=lambda _: types.SimpleNamespace(
                as_video_stream_profile=lambda: types.SimpleNamespace(
                    get_intrinsics=lambda: types.SimpleNamespace(
                        fx=600, fy=600, ppx=1, ppy=1,
                        width=3, height=2, coeffs=[0]*5, model='none'))),
            get_device=lambda: types.SimpleNamespace(
                first_depth_sensor=lambda: types.SimpleNamespace(get_depth_scale=lambda: .001)),
        )
        stop = Mock()
        pipeline = types.SimpleNamespace(
            start=lambda _: profile, wait_for_frames=lambda _: frames, stop=stop)
        config = types.SimpleNamespace(enable_device=lambda _: None, enable_stream=lambda *args: None)
        fake_rs = types.SimpleNamespace(
            pipeline=lambda: pipeline, config=lambda: config,
            align=lambda _: types.SimpleNamespace(process=lambda received: received),
            stream=types.SimpleNamespace(color='color', depth='depth'),
            format=types.SimpleNamespace(bgr8='bgr8', z16='z16'),
        )
        with patch.dict(sys.modules, {'pyrealsense2': fake_rs}):
            with self.assertRaisesRegex(RuntimeError, 'delta 170.000 ms exceeds 40.000 ms'):
                vision.capture_frame('fake-serial', width=3, height=2,
                                     warmup_frames=1, max_framesets=5)
            with self.assertRaisesRegex(RuntimeError, '9000 ms/45 framesets'):
                vision.capture_frame('fake-serial', width=3, height=2,
                                     warmup_frames=30, timeout_ms=9000)
        self.assertEqual(stop.call_count, 2)


if __name__ == '__main__':
    unittest.main()
