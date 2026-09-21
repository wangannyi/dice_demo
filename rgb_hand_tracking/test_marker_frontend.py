import queue
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from marker_frontend import MarkerRgbCamera


class ControlledCamera:
    def __init__(self, opened=True):
        self.opened = opened
        self.items = queue.Queue()
        self.timestamp = None
        self.settings = []
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.settings.append((key, value))

    def get(self, key):
        return 30.

    def read(self):
        item = self.items.get()
        if item is None:
            return False, None
        frame, self.timestamp = item
        return True, frame

    def push(self, frame, timestamp):
        self.items.put((frame, timestamp))

    def release(self):
        self.released = True
        self.items.put(None)


class ScriptedTracker:
    def __init__(self, rows, **options):
        self.rows = rows
        self.marker_id = options['marker_id']
        self.max_gap_s = options['max_gap_s']
        self.jump_px = 12
        self.max_speed_px_s = 600
        self.calls = []

    def update(self, frame, timestamp, epoch):
        self.calls.append((int(frame[0, 0, 0]), timestamp, epoch))
        row = dict(self.rows[len(self.calls)-1])
        row.update(timestamp_s=timestamp, marker_id=self.marker_id,
                   motion_target_valid=False)
        return row


class Tests(unittest.TestCase):
    config = {'format': 'MJPG', 'width': 160, 'height': 120}

    @staticmethod
    def row(valid=True, shift=0, reason=None):
        corners = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], float)+shift
        return {'state': 'TRACKING' if valid else 'ACQUIRING',
                'observation_valid': valid, 'reason': reason or ('confirmed' if valid else 'confirming'),
                'raw_corners_px': corners.tolist(), 'marker_corners_px': (corners+.25).tolist()}

    def camera(self, rows=None, config=None, **options):
        self.native = ControlledCamera()
        tracker_factory = None
        if rows is not None:
            def tracker_factory(**opts):
                return ScriptedTracker(rows, **opts)
        front = MarkerRgbCamera(
            '/dev/video7', config or self.config, warmup_frames=0, read_timeout_s=.05,
            camera_factory=lambda *args: self.native, tracker_factory=tracker_factory,
            clock=lambda: self.native.timestamp, **options)
        self.addCleanup(front.close)
        return front

    def publish(self, front, index, timestamp):
        frame = np.full((120, 160, 3), index, np.uint8)
        self.native.push(frame, timestamp)
        with front.condition:
            self.assertTrue(front.condition.wait_for(
                lambda: front.serial >= index or front.failed is not None, timeout=1))
        return frame

    def test_latest_frame_and_marker_identity_are_atomic_and_copies(self):
        front = self.camera([self.row() for _ in range(12)])
        for index in range(1, 13):
            self.publish(front, index, index/30)
        frame, timestamp, serial, marker, generation = front.next(0)
        self.assertEqual(serial, 12)
        self.assertEqual(int(frame[0, 0, 0]), 12)
        self.assertEqual(timestamp, 12/30)
        self.assertEqual(marker['frame_serial'], serial)
        self.assertEqual(marker['timestamp_s'], timestamp)
        self.assertEqual(marker['capture_timestamp_s'], timestamp)
        self.assertEqual(marker['camera_epoch'], 'initial')
        self.assertEqual(marker['tracking_generation'], generation)
        self.assertEqual((marker['width'], marker['height']), (160, 120))
        self.assertFalse(marker['motion_target_valid'])
        self.assertEqual(marker['raw_corners_px'][0], [30, 30])
        self.assertEqual(marker['marker_corners_px'][0], [30.25, 30.25])
        frame[:] = 99
        marker['raw_corners_px'][0][0] = 99
        fresh = front.next(0)
        self.assertEqual(int(fresh[0][0, 0, 0]), 12)
        self.assertEqual(fresh[3]['raw_corners_px'][0], [30, 30])
        self.assertEqual(len(front.tracker.calls), 12)
        stats = front.stats_snapshot()
        self.assertAlmostEqual(stats['capture_mean_fps'], 30)
        self.assertEqual(stats['frames'], 12)
        self.assertEqual(stats['latest_frame_serial'], 12)
        self.assertTrue(stats['latest_marker_observation_valid'])

    def test_hidden_loss_increments_generation_once_and_survives_reacquisition(self):
        missing = {'state': 'LOST', 'observation_valid': False, 'reason': 'marker_missing',
                   'marker_corners_px': None}
        rows = [self.row(False), self.row(False), self.row(), missing,
                missing, self.row(False), self.row(False), self.row()]
        front = self.camera(rows)
        for index in range(1, 4):
            self.publish(front, index, index/30)
        self.assertEqual(front.next(0)[4], 0)
        for index in range(4, 9):
            self.publish(front, index, index/30)
        _, _, _, marker, generation = front.next(3)
        self.assertTrue(marker['observation_valid'])
        self.assertEqual(generation, 1)
        stats = front.stats_snapshot()
        self.assertEqual(stats['generation_change_reasons'], {'tracking_invalidated': 1})
        self.assertEqual(stats['tags_detected'], 6)
        self.assertEqual(stats['tags_confirmed'], 2)

    def test_jump_gap_and_epoch_resets_count_once_per_frame(self):
        rows = [self.row(), self.row(False, 100, 'jump_reacquire'),
                self.row(shift=100), self.row(False, 100),
                self.row(shift=100), self.row(False, 100)]
        front = self.camera(rows)
        self.publish(front, 1, 1.)
        self.publish(front, 2, 1.01)
        self.assertEqual(front.next(1)[4], 1)
        self.publish(front, 3, 1.1)
        self.publish(front, 4, 2.)
        self.assertEqual(front.next(3)[4], 2)
        self.publish(front, 5, 2.01)
        front.camera_epoch = 'relocated'
        self.publish(front, 6, 2.02)
        marker = front.next(5)[3]
        self.assertEqual(marker['camera_epoch'], 'relocated')
        self.assertEqual(marker['tracking_generation'], 3)
        self.assertEqual(front.stats_snapshot()['generation_change_reasons'],
                         {'marker_jump': 1, 'tracking_invalidated': 3,
                          'capture_gap': 1, 'camera_epoch_changed': 1})

    def test_reported_camera_fps_does_not_claim_detector_rate(self):
        front = self.camera([self.row(False) for _ in range(4)])
        for index in range(1, 5):
            self.publish(front, index, index*.2)
        stats = front.stats_snapshot()
        self.assertAlmostEqual(stats['capture_mean_fps'], 5)
        self.assertEqual(stats['reported_camera_fps'], 30)
        self.assertAlmostEqual(stats['min_capture_interval_s'], .2)
        self.assertAlmostEqual(stats['max_capture_interval_s'], .2)
        self.assertTrue(stats['timestamps_strictly_forward'])

    def test_real_tracker_acquires_at_camera_rate_while_consumer_skips_frames(self):
        front = self.camera(config={'format': 'MJPG', 'width': 320, 'height': 240})
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        factory = getattr(cv2.aruco, 'generateImageMarker', None)
        code = (factory(dictionary, 40, 80) if factory else
                cv2.aruco.drawMarker(dictionary, 40, 80))
        image = np.full((240, 320, 3), 255, np.uint8)
        image[80:160, 100:180] = code[:, :, None]
        for index in range(1, 9):
            self.native.push(image, index/30)
            with front.condition:
                self.assertTrue(front.condition.wait_for(
                    lambda: front.serial >= index or front.failed is not None, timeout=1))
        _, _, serial, marker, generation = front.next(0)
        self.assertEqual(serial, 8)
        self.assertTrue(marker['observation_valid'])
        self.assertEqual(marker['state'], 'TRACKING')
        self.assertEqual(generation, 0)
        self.assertEqual(front.stats_snapshot()['tags_confirmed'], 6)

    def test_read_failure_releases_camera_and_rejects_previous_observation(self):
        front = self.camera([self.row()])
        self.publish(front, 1, 1.)
        self.native.items.put(None)
        with front.condition:
            self.assertTrue(front.condition.wait_for(lambda: front.failed is not None, timeout=1))
        with self.assertRaisesRegex(RuntimeError, 'RGB read failed'):
            front.next(0)
        front.thread.join(timeout=1)
        self.assertTrue(self.native.released)
        self.assertEqual(front.stats_snapshot()['failures'], 1)

    def test_camera_resolution_fallback_rejected_before_marker_detection(self):
        front = self.camera([self.row()])
        self.native.push(np.zeros((240, 320, 3), np.uint8), 1.)
        with front.condition:
            self.assertTrue(front.condition.wait_for(lambda: front.failed is not None, timeout=1))
        with self.assertRaisesRegex(RuntimeError, 'calibrated stream size'):
            front.next(0)
        front.thread.join(timeout=1)
        self.assertTrue(self.native.released)
        self.assertEqual(front.tracker.calls, [])

    def test_bad_timestamp_releases_reader_instead_of_reusing_old_metadata(self):
        front = self.camera([self.row(), self.row()])
        self.publish(front, 1, 1.)
        self.publish(front, 2, 1.)
        with self.assertRaisesRegex(RuntimeError, 'strictly increasing'):
            front.next(1)
        front.thread.join(timeout=1)
        self.assertFalse(front.stats_snapshot()['timestamps_strictly_forward'])
        self.assertTrue(self.native.released)

    def test_timeout_and_close_unblock_camera_and_are_idempotent(self):
        front = self.camera([])
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            front.next(0)
        self.assertTrue(self.native.released)
        self.assertFalse(front.thread.is_alive())
        front.close()
        front.close()
        self.assertFalse(front.thread.is_alive())
        self.assertTrue(self.native.released)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            front.next(0)
        self.assertEqual(front.stats_snapshot()['failures'], 0)

    def test_only_stream_properties_are_set_and_open_warmup_failures_release(self):
        self.camera([])
        self.assertEqual({key for key, _ in self.native.settings},
                         {cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_FRAME_WIDTH,
                          cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS,
                          cv2.CAP_PROP_BUFFERSIZE})
        for opened, warmup, reason in ((False, 0, 'Cannot open'), (True, 1, 'warmup')):
            native = ControlledCamera(opened)
            native.items.put(None)
            with self.subTest(opened=opened), self.assertRaisesRegex(RuntimeError, reason):
                MarkerRgbCamera('/dev/video7', self.config, warmup_frames=warmup,
                                camera_factory=lambda *args: native)
            self.assertTrue(native.released)

    def test_invalid_tracking_options_fail_before_opening_camera(self):
        factory = Mock()
        invalid = ([{'marker_id': value} for value in (-1, 50, True, 2.5, '40')]
                   + [{'max_gap_s': value} for value in (0, -1, True, '1', float('nan'))]
                   + [{'read_timeout_s': value} for value in (0, True, '1', float('inf'))]
                   + [{'warmup_frames': value} for value in (-1, True, .5)]
                   + [{'camera_epoch': value} for value in ('', None, 2)])
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                MarkerRgbCamera('/dev/video7', self.config, camera_factory=factory, **options)
        factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
