"""Verify DFL decoding, mask mapping and semantic/color selection without weights."""

import unittest

import numpy as np

from dice_cup_localization.yolo_seg import decode, decode_standard2, preprocess, select_green_cup


class SegmentationTests(unittest.TestCase):
    def test_decode_cup_mask_after_letterbox(self):
        image = np.zeros((480, 640, 3), np.uint8)
        _, transform = preprocess(image)
        outputs = []
        for size in (80, 40, 20):
            outputs.extend([np.full((1, 64, size, size), -50, np.float32),
                            np.zeros((1, 80, size, size), np.float32),
                            np.zeros((1, 1, size, size), np.float32)])
        outputs.extend([np.zeros((1, 32, s, s), np.float32) for s in (80, 40, 20)])
        outputs.append(np.ones((1, 32, 160, 160), np.float32))
        for side in range(4):
            outputs[0][0, side*16+5, 40, 40] = 50
        outputs[1][0, 41, 40, 40] = .9
        outputs[9][0, 0, 40, 40] = 1
        result = decode(outputs, image.shape, transform)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['class_id'], 41)
        np.testing.assert_allclose(result[0]['bbox_xyxy'], [284, 204, 364, 284])
        self.assertEqual(result[0]['mask'].sum(), 80*80)
        self.assertFalse(result[0]['mask'][0, 0])

    def test_green_only_never_becomes_yolo_cup(self):
        image = np.full((20, 20, 3), [0, 255, 0], np.uint8)
        item = {'class_id': 0, 'mask': np.ones((20, 20), bool)}
        self.assertIsNone(select_green_cup(image, [item])[0])
        item['class_id'] = 41
        self.assertEqual(select_green_cup(image, [item])[0], 0)
        self.assertEqual(select_green_cup(image, [item, item])[1], 'multiple_green_yolo_candidates')
        item['class_id'] = 75
        self.assertIsNone(select_green_cup(image, [item])[0])
        self.assertEqual(select_green_cup(image, [item], candidate_classes=(41, 75))[0], 0)
        image[:] = [0, 0, 255]
        self.assertIsNone(select_green_cup(image, [item])[0])

    def test_invalid_layout_rejected(self):
        with self.assertRaises(ValueError):
            decode([np.zeros((1, 116, 8400))], (480, 640, 3), (1, 0, 80, 640, 480))

    def test_two_class_cap_mask_and_activated_probability(self):
        image = np.zeros((480, 640, 3), np.uint8)
        _, transform = preprocess(image)
        detection = np.zeros((1, 38, 8400), np.float32)
        prototype = np.zeros((1, 32, 160, 160), np.float32)
        detection[0, :4, 100] = [320, 320, 80, 80]
        detection[0, 4, 100] = .8
        detection[0, 6, 100] = 1
        prototype[0, 0] = 1
        results = decode_standard2([detection, prototype], image.shape, transform)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['class_id'], 0)
        np.testing.assert_allclose(results[0]['bbox_xyxy'], [280, 200, 360, 280])
        self.assertEqual(results[0]['mask'].sum(), 80*80)
        detection[0, 4, 100] = .2
        self.assertEqual(decode_standard2([detection, prototype], image.shape, transform), [])


if __name__ == '__main__':
    unittest.main()
