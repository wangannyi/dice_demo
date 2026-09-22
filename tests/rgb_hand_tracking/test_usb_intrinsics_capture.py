import unittest

import numpy as np

from rgb_hand_tracking.board_rgb import BoardRgbObserver
from rgb_hand_tracking.usb_intrinsics_capture import SampleSelector


class Tests(unittest.TestCase):
    def setUp(self):
        self.observer = BoardRgbObserver({})
        self.selector = SampleSelector(self.observer)
        board = self.observer.board
        self.points = np.asarray(board.getChessboardCorners() if hasattr(board, 'getChessboardCorners')
                                 else board.chessboardCorners, float)

    def observation(self, x=200, y=200, scale=2000):
        return {'valid': True, 'charuco_corner_count': len(self.points),
                'charuco_corner_ids': list(range(len(self.points))), 'image_size': [1280, 720],
                'charuco_corners_px': (self.points[:, :2]*scale+[x, y]).tolist()}

    def test_stationary_duplicate_not_counted_as_diverse_views(self):
        obs = self.observation()
        self.assertFalse(self.selector.consider(obs, 1)[0])
        self.assertTrue(self.selector.consider(obs, 1.2)[0])
        self.selector.consider(obs, 2.4)
        self.assertEqual(self.selector.consider(obs, 2.6)[1],
                         'duplicate_view_change_position_distance_or_tilt')
        self.assertEqual(len(self.selector.accepted), 1)

    def test_moving_view_waits_then_accepts_new_held_position(self):
        first = self.observation()
        self.selector.consider(first, 1)
        self.selector.consider(first, 1.2)
        self.selector.consider(first, 2.4)
        moved = self.observation(x=500)
        self.assertEqual(self.selector.consider(moved, 2.6)[1], 'hold_board_still')
        self.assertTrue(self.selector.consider(moved, 2.8)[0])

    def test_tiny_or_missing_board_not_saved(self):
        self.assertEqual(self.selector.consider(self.observation(scale=100), 1)[1],
                         'board_too_small_move_closer')
        self.assertFalse(self.selector.consider({'valid': False}, 2)[0])


if __name__ == '__main__':
    unittest.main()
