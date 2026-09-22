import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from nero_calibration.core import pose_matrix
from tools.reprocess_dimensions import main, measured_observation


class ReprocessDimensionsTests(unittest.TestCase):
    def check_known_pose(self, width_m, height_m):
        detector = Mock()
        detector.roi = None
        detector.cfg = {'squares_x': 4, 'squares_y': 5, 'square_length_m': .0215,
                        'max_reprojection_px': 1.}
        detector.points = np.array([[x*.0215, y*.0215, 0]
                                    for y in range(1, 5) for x in range(1, 4)])
        physical = np.array([[x*width_m/4, y*height_m/5, 0]
                             for y in range(1, 5) for x in range(1, 4)])
        expected = pose_matrix([.035, -.027, .55, .3, -.2, .1])
        K = np.array([[600., 0, 320], [0, 601., 240], [0, 0, 1.]])
        D = np.zeros(5)
        pixels = cv2.projectPoints(physical, cv2.Rodrigues(expected[:3, :3])[0],
                                   expected[:3, 3], K, D)[0]
        ids = np.arange(12).reshape(-1, 1)
        image = np.zeros((480, 640, 3), np.uint8)
        with patch('cv2.aruco.detectMarkers', return_value=([], ids, [])), \
             patch('cv2.aruco.interpolateCornersCharuco', return_value=(12, pixels, ids)):
            actual, quality, _, _ = measured_observation(
                image, detector, K, D, width_m, height_m)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertLess(quality['reprojection_rms_px'], 1e-5)
        np.testing.assert_array_equal(image, np.zeros_like(image))

    def test_measured_xy_spacing_recovers_known_pose(self):
        self.check_known_pose(.0865, .108)

    def test_original_square_dimensions_preserve_known_pose(self):
        self.check_known_pose(.086, .1075)

    def test_invalid_measurements_are_rejected_before_detection(self):
        detector = Mock()
        for width, height in [(0, .1), (.1, float('nan')), (1, .1), (.1, -.1)]:
            with self.assertRaisesRegex(ValueError, 'width and height'):
                measured_observation(None, detector, None, None, width, height)
        detector.detect.assert_not_called()

    def test_version_mismatch_rejected_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'new'
            with patch('tools.reprocess_dimensions.read_dataset',
                       return_value=({'opencv_version': 'different-version'}, [])):
                with self.assertRaisesRegex(ValueError, 'recorded OpenCV version'):
                    main(['--dataset', tmp, '--output', str(output),
                          '--width-mm', '86.5', '--height-mm', '108'])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
