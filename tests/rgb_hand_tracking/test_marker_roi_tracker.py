import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rgb_hand_tracking.marker_roi_tracker import RoiRescueMarkerTracker
from rgb_hand_tracking.tracker import MarkerTracker


def image(x=150):
    bgr = np.full((400, 400, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    draw = getattr(cv2.aruco, 'generateImageMarker', None) or cv2.aruco.drawMarker
    marker = draw(dictionary, 40, 50)
    bgr[150:200, x:x+50] = marker[:, :, None]
    return bgr


class RoiTests(unittest.TestCase):
    def tracker(self):
        tracker = RoiRescueMarkerTracker()
        for i in range(3):
            result = tracker.update(image(), 1+i/30, 'a')
        self.assertTrue(result['observation_valid'])
        return tracker

    def forced_native_miss(self):
        original = MarkerTracker._select_current
        calls = []
        def select(tracker, bgr):
            calls.append(bgr.shape)
            return [] if len(calls) == 1 else original(tracker, bgr)
        return calls, patch.object(MarkerTracker, '_select_current', select)

    def test_current_roi_decodes_and_maps_native_coordinates(self):
        tracker = self.tracker()
        calls, context = self.forced_native_miss()
        with context:
            result = tracker.update(image(), 1.1, 'a')
        self.assertTrue(result['observation_valid'])
        self.assertEqual(result['detection_source'], 'current_frame_roi_x2')
        np.testing.assert_allclose(result['marker_center_px'], [174.5, 174.5], atol=.4)
        self.assertEqual(len(calls), 2)

    def test_occlusion_does_not_publish_previous_corners(self):
        tracker = self.tracker()
        result = tracker.update(np.full((400, 400, 3), 255, np.uint8), 1.1, 'a')
        self.assertFalse(result['observation_valid'])
        self.assertIsNone(result['marker_corners_px'])
        self.assertIsNone(tracker.previous)

    def test_native_duplicate_cannot_be_hidden_by_roi(self):
        tracker = self.tracker()
        with patch.object(MarkerTracker, '_select_current', return_value=[tracker.previous]*2):
            result = tracker.update(image(), 1.1, 'a')
        self.assertEqual(result['reason'], 'duplicate_marker_id')
        self.assertEqual(tracker.roi_attempts, 0)

    def test_epoch_change_cannot_rescue_from_old_roi(self):
        tracker = self.tracker()
        with patch.object(MarkerTracker, '_select_current', return_value=[]):
            result = tracker.update(image(), 1.1, 'b')
        self.assertFalse(result['observation_valid'])
        self.assertEqual(tracker.roi_attempts, 0)

    def test_capture_gap_cannot_rescue_from_old_roi(self):
        tracker = self.tracker()
        with patch.object(MarkerTracker, '_select_current', return_value=[]):
            result = tracker.update(image(), 2., 'a')
        self.assertFalse(result['observation_valid'])
        self.assertEqual(tracker.roi_attempts, 0)

    def test_jump_reconfirmation_applies_to_rescued_corners(self):
        tracker = self.tracker()
        _, context = self.forced_native_miss()
        with context:
            result = tracker.update(image(190), 1.1, 'a')
        self.assertFalse(result['observation_valid'])
        self.assertEqual(result['reason'], 'jump_reacquire')


if __name__ == '__main__':
    unittest.main()
