import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from board_rgb import BoardRgbObserver


CONFIG = {
    'type': 'charuco', 'dictionary': '4x4_50',
    'squares_x': 4, 'squares_y': 5,
    'square_length_m': .022, 'marker_length_m': .0155,
    'legacy_pattern': False,
}


def board_image(observer):
    if hasattr(observer.board, 'generateImage'):
        image = observer.board.generateImage((640, 800), marginSize=32, borderBits=1)
    else:
        image = observer.board.draw((640, 800), marginSize=32, borderBits=1)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


class BoardTests(unittest.TestCase):
    def test_synthetic_board_observation_has_pixels_only(self):
        observer = BoardRgbObserver({'board': CONFIG})
        image = board_image(observer)
        original = image.copy()
        with patch.object(cv2, 'solvePnP', side_effect=AssertionError('No PnP allowed')):
            result = observer.observe(image)
        self.assertTrue(result['valid'], result['reason'])
        self.assertEqual(result['charuco_corner_count'], 12)
        self.assertEqual(sorted(result['charuco_corner_ids']), list(range(12)))
        self.assertEqual(len(result['marker_ids']), 10)
        self.assertEqual(result['coordinate_space'], 'image_pixels')
        self.assertFalse(result['metric_pose_available'])
        self.assertFalse(result['motion_target_valid'])
        self.assertNotIn('camera_matrix', result)
        self.assertNotIn('T_camera_board', result)
        points = np.asarray(result['charuco_corners_px'])
        self.assertEqual(points.shape, (12, 2))
        self.assertTrue((points > 32).all())
        self.assertLess(points[:, 0].max(), image.shape[1])
        self.assertLess(points[:, 1].max(), image.shape[0])
        json.dumps(result)
        annotated = observer.annotate(image, result)
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.array_equal(annotated, original))

    def test_blank_frame_invalid(self):
        observer = BoardRgbObserver(CONFIG)
        result = observer.observe(np.full((480, 640, 3), 255, dtype=np.uint8))
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'no_board_markers')
        self.assertEqual(result['charuco_corner_count'], 0)
        self.assertEqual(result['marker_ids'], [])

    def test_other_marker_is_not_board(self):
        observer = BoardRgbObserver(CONFIG)
        image = np.full((300, 300, 3), 255, dtype=np.uint8)
        if hasattr(cv2.aruco, 'generateImageMarker'):
            marker = cv2.aruco.generateImageMarker(observer.dictionary, 40, 120)
        else:
            marker = cv2.aruco.drawMarker(observer.dictionary, 40, 120)
        image[80:200, 80:200] = marker[:, :, None]
        result = observer.observe(image)
        self.assertEqual(result['detected_marker_ids'], [40])
        self.assertEqual(result['marker_ids'], [])
        self.assertFalse(result['valid'])

    def test_invalid_frame_and_config(self):
        observer = BoardRgbObserver(CONFIG)
        for image in (None, np.zeros((20, 20), np.uint8), np.zeros((20, 20, 3), float)):
            with self.assertRaises(ValueError):
                observer.observe(image)
        with self.assertRaises(ValueError):
            BoardRgbObserver(dict(CONFIG, marker_length_m=.03))

    def test_default_keeps_legacy_interpolation_correspondences_unchanged(self):
        # A legacy detector reports the image's top corner row as IDs 9..11.
        # Inject those returns after actual marker detection on a board image.
        points = np.array([[100.+100*col, 100.+100*row]
                           for row in range(4) for col in range(3)], np.float32)
        old_ids = np.array([9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2], np.int32)
        with patch.object(cv2, '__version__', '4.5.4'):
            observer = BoardRgbObserver(CONFIG)
            image = board_image(observer)
            with patch.object(cv2.aruco, 'interpolateCornersCharuco', create=True,
                              return_value=(12, points[:, None], old_ids[:, None])):
                result = observer.observe(image)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['charuco_corner_ids'], old_ids.tolist())
        np.testing.assert_array_equal(result['charuco_corners_px'], points)
        self.assertEqual(result['corner_convention'], 'native')
        self.assertFalse(result['corner_ids_remapped'])
        self.assertFalse(result['metric_pose_available'])

    def test_explicit_modern_convention_normalizes_old_detector_output(self):
        points = np.array([[100.+100*col, 100.+100*row]
                           for row in range(4) for col in range(3)], np.float32)
        old_ids = np.array([9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2], np.int32)
        # A deliberately shuffled detector result also checks that point/ID
        # associations survive sorting, rather than only testing an ID formula.
        shuffle = np.array([11, 4, 0, 7, 3, 8, 2, 9, 1, 10, 5, 6])
        with patch.object(cv2, '__version__', '4.5.4'):
            observer = BoardRgbObserver(CONFIG, corner_convention='opencv_4_6')
            image = board_image(observer)
            with patch.object(cv2.aruco, 'interpolateCornersCharuco', create=True,
                              return_value=(12, points[shuffle, None], old_ids[shuffle, None])):
                result = observer.observe(image)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['charuco_corner_ids'], list(range(12)))
        np.testing.assert_array_equal(result['charuco_corners_px'], points)
        self.assertEqual(result['corner_convention'], 'opencv_4_6')
        self.assertEqual(result['native_corner_convention'], 'opencv_before_4_6')
        self.assertTrue(result['corner_ids_remapped'])

    def test_explicit_modern_convention_does_not_transform_modern_output(self):
        points = np.array([[100., 100.], [200., 100.], [300., 100.],
                           [100., 200.], [200., 200.], [300., 200.]], np.float32)
        ids = np.array([0, 1, 2, 3, 4, 5], np.int32)
        with patch.object(cv2, '__version__', '4.13.0'):
            observer = BoardRgbObserver(CONFIG, corner_convention='opencv_4_6')
            image = board_image(observer)
            with patch.object(cv2.aruco, 'interpolateCornersCharuco', create=True,
                              return_value=(6, points[:, None], ids[:, None])):
                result = observer.observe(image)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['charuco_corner_ids'], ids.tolist())
        np.testing.assert_array_equal(result['charuco_corners_px'], points)
        self.assertFalse(result['corner_ids_remapped'])
        self.assertEqual(result['native_corner_convention'], 'opencv_4_6')

    def test_actual_pattern_detection_maps_correspondences_only_when_needed(self):
        native = BoardRgbObserver(CONFIG)
        canonical = BoardRgbObserver(CONFIG, corner_convention='opencv_4_6')
        image = board_image(native)
        original = native.observe(image)
        result = canonical.observe(image)
        self.assertTrue(result['valid'], result)
        self.assertTrue(original['valid'], original)
        original_map = dict(zip(original['charuco_corner_ids'], original['charuco_corners_px']))
        if native.native_corner_convention == 'opencv_before_4_6':
            # Explicit known 4x5 printed board row correspondences.
            id_pairs = [(9, 0), (10, 1), (11, 2), (6, 3), (7, 4), (8, 5),
                        (3, 6), (4, 7), (5, 8), (0, 9), (1, 10), (2, 11)]
            result_map = dict(zip(result['charuco_corner_ids'], result['charuco_corners_px']))
            for old_id, new_id in id_pairs:
                np.testing.assert_array_equal(original_map[old_id], result_map[new_id])
        else:
            self.assertEqual(original['charuco_corner_ids'], result['charuco_corner_ids'])
            self.assertEqual(original['charuco_corners_px'], result['charuco_corners_px'])
        with self.assertRaises(ValueError):
            BoardRgbObserver(CONFIG, corner_convention='unknown')


if __name__ == '__main__':
    unittest.main()
