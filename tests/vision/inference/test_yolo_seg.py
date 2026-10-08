"""Verify the letterbox preprocessing contract shared by every inference path."""

import unittest

import numpy as np

from vision.inference.yolo_seg import preprocess


class PreprocessTests(unittest.TestCase):
    def test_letterbox_shape_dtype_and_transform(self):
        image = np.zeros((480, 640, 3), np.uint8)
        tensor, (scale, left, top, nw, nh) = preprocess(image)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual((scale, left, top, nw, nh), (1.0, 0, 80, 640, 480))
        # RGB channel swap: a pure-blue BGR image must light the blue plane
        # at a content pixel (letterbox padding stays the verified 114 grey).
        blue = np.zeros((480, 640, 3), np.uint8)
        blue[:, :] = (255, 0, 0)
        tensor, (_, left, top, nw, nh) = preprocess(blue)
        self.assertAlmostEqual(float(tensor[0, 2, 240, 320]), 1.0, places=6)
        self.assertEqual(float(tensor[0, 0, 240, 320]), 0.0)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 114/255, places=6)

    def test_non_square_input_scales_to_fit(self):
        image = np.zeros((720, 1280, 3), np.uint8)
        tensor, (scale, left, top, nw, nh) = preprocess(image)
        self.assertEqual((nw, nh), (640, 360))
        self.assertEqual((left, top), (0, 140))
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        # Content region differs from the padding rows.
        self.assertFalse(np.array_equal(tensor[0, :, top], tensor[0, :, 0]))


if __name__ == '__main__':
    unittest.main()
