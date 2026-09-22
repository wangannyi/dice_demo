import unittest
import cv2
import numpy as np
from rgb_hand_tracking.tracker import MarkerTracker, AdaptiveFilter, calibrated_region


def frame(marker_id=40, x=100, second=False):
    image = np.full((400, 600, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, 'generateImageMarker'):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 90)
    else:
        marker = cv2.aruco.drawMarker(dictionary, marker_id, 90)
    image[100:190, x:x+90] = marker[:, :, None]
    if second:
        image[250:340, 400:490] = marker[:, :, None]
    return image


class Tests(unittest.TestCase):
    def confirmed(self):
        tracker = MarkerTracker()
        for i in range(3):
            result = tracker.update(frame(), i/30)
        self.assertTrue(result['observation_valid'])
        self.assertFalse(result['motion_target_valid'])
        self.assertIsNone(result['palm_center_px'])
        return tracker

    def test_acquire_and_loss(self):
        tracker = self.confirmed()
        result = tracker.update(np.full((400, 600, 3), 255, np.uint8), .1)
        self.assertFalse(result['observation_valid'])
        self.assertIsNone(result['marker_center_px'])
        self.assertEqual(tracker.update(frame(), .14)['state'], 'ACQUIRING')

    def test_wrong_id_and_duplicate(self):
        for image in [frame(0), frame(second=True)]:
            self.assertFalse(MarkerTracker().update(image, 0)['observation_valid'])

    def test_jump(self):
        tracker = self.confirmed()
        self.assertEqual(tracker.update(frame(x=300), .1)['reason'], 'jump_reacquire')

    def test_camera_epoch(self):
        tracker = self.confirmed()
        self.assertEqual(tracker.update(frame(), .1, 'moved')['state'], 'ACQUIRING')

    def test_gap_and_stale_timestamp(self):
        tracker = self.confirmed()
        self.assertEqual(tracker.update(frame(), 1)['state'], 'ACQUIRING')
        with self.assertRaises(ValueError):
            tracker.update(frame(), 1)

    def test_filter_jitter_and_motion(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (300, 2))
        filt = AdaptiveFilter()
        smoothed = np.array([filt.update(x, 1/30) for x in data])
        self.assertLess(smoothed[30:].std(), data[30:].std()*.7)
        for _ in range(10):
            result = filt.update(np.array([40, 40]), 1/30)
        self.assertLess(np.linalg.norm(result-[40, 40]), 2)

    def test_template(self):
        result = calibrated_region([[100, 100], [200, 100], [200, 200], [100, 200]], [[.5, .5]])
        np.testing.assert_allclose(result, [[150, 150]])


if __name__ == '__main__':
    unittest.main()
