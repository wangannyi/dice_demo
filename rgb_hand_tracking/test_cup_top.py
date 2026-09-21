"""Cup identity and internal-top-edge behavior on deterministic RGB scenes."""
import json
import unittest

import cv2
import numpy as np

from cup_top import CupTopDetector


class FakeSegmentor:
    layout = 'standard2_cap_ground'

    def __init__(self, instances):
        self.instances = instances

    def infer(self, bgr):
        return self.instances


def scene(top=True, duplicate=False):
    image = np.full((220, 360, 3), 210, np.uint8)
    items = []
    for center in [(95, 115)] + ([(270, 115)] if duplicate else []):
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.ellipse(mask, center, (65, 77), 0, 0, 360, 255, -1)
        image[mask != 0] = (30, 100, 30)
        if top:
            cv2.ellipse(image, (center[0]-6, center[1]-20), (37, 34),
                        0, 0, 360, (55, 155, 55), -1)
        items.append({'class_id': 0, 'score': .95, 'mask': mask != 0,
                      'bbox_xyxy': [center[0]-65, center[1]-77,
                                    center[0]+66, center[1]+78]})
    return image, items


class CupTopTests(unittest.TestCase):
    def test_top_edge_center_differs_from_whole_body(self):
        image, items = scene()
        result = CupTopDetector(FakeSegmentor(items)).process(image)
        self.assertTrue(result['valid'], result)
        self.assertLess(np.linalg.norm(np.asarray(result['center_px'])-[89, 95]), 1.5)
        self.assertGreater(np.linalg.norm(np.asarray(result['center_px'])
                                         - result['body_center_px_diagnostic']), 15)
        self.assertAlmostEqual(result['ellipse_px']['diameters_px'][0], 68, delta=3)
        self.assertAlmostEqual(result['ellipse_px']['diameters_px'][1], 74, delta=3)
        self.assertFalse(result['motion_target_valid'])
        json.dumps(result)

    def test_outer_body_outline_cannot_substitute_for_top(self):
        image, items = scene(top=False)
        result = CupTopDetector(FakeSegmentor(items)).process(image)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'no_supported_internal_top_edge')
        self.assertIsNone(result['center_px'])

    def test_green_shape_without_yolo_confirmation_is_rejected(self):
        image, _ = scene()
        result = CupTopDetector(FakeSegmentor([])).process(image)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'no_green_yolo_candidate')

    def test_two_cups_are_ambiguous(self):
        image, items = scene(duplicate=True)
        result = CupTopDetector(FakeSegmentor(items)).process(image)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'multiple_green_yolo_candidates')
        self.assertIsNone(result['center_px'])

    def test_ground_class_cannot_confirm_cup(self):
        image, items = scene()
        items[0]['class_id'] = 1
        result = CupTopDetector(FakeSegmentor(items)).process(image)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'no_green_yolo_candidate')

    def test_green_top_backend_is_explicit_and_not_model_confirmed(self):
        image, _ = scene()
        result = CupTopDetector(backend='green_top').process(image)
        self.assertTrue(result['valid'], result)
        self.assertFalse(result['model_confirmed'])
        self.assertEqual(result['source'], 'green_top_geometry')
        self.assertFalse(result['motion_target_valid'])
        self.assertLess(np.linalg.norm(np.asarray(result['center_px'])-[89, 95]), 1.5)

    def test_green_board_rectangular_border_is_not_a_cup(self):
        image, _ = scene()
        cv2.rectangle(image, (205, 40), (340, 195), (30, 100, 30), 12)
        result = CupTopDetector(backend='green_top').process(image)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['quality']['green_geometry_candidate_count'], 1)
        self.assertLess(result['center_px'][0], 120)

    def test_green_top_backend_rejects_multiple_round_objects(self):
        image, _ = scene(duplicate=True)
        result = CupTopDetector(backend='green_top').process(image)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'multiple_green_geometry_candidates')

    def test_green_top_backend_still_requires_internal_edge(self):
        image, _ = scene(top=False)
        result = CupTopDetector(backend='green_top').process(image)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'no_supported_internal_top_edge')

    def test_disconnected_measured_arcs_form_one_internal_top_edge(self):
        image, _ = scene(top=False)
        for start, end in ((0, 190), (220, 310)):
            cv2.ellipse(image, (89, 95), (37, 34), 0, start, end, (55, 155, 55), 1)
        result = CupTopDetector(backend='green_top').process(image)
        self.assertTrue(result['valid'], result)
        self.assertLess(np.linalg.norm(np.asarray(result['center_px'])-[89, 95]), 1)
        self.assertTrue(result['quality']['combined_same_frame_edge_fragments'])
        self.assertGreaterEqual(result['quality']['angular_coverage'], .60)
        self.assertGreaterEqual(result['quality']['edge_support_fraction'], .60)

    def test_insufficient_visible_arcs_cannot_invent_missing_top_boundary(self):
        image, _ = scene(top=False)
        for start, end in ((0, 130), (190, 240)):
            cv2.ellipse(image, (89, 95), (37, 34), 0, start, end, (55, 155, 55), 1)
        result = CupTopDetector(backend='green_top').process(image)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'no_supported_internal_top_edge')
        self.assertIsNone(result['center_px'])

    def test_enclosed_hsv_color_dropout_does_not_erase_actual_top_edge(self):
        image, items = scene(top=False)
        # This face retains its luminance edge, but its saturation is below the
        # green mask threshold. Only the explicit green backend fills that hole.
        cv2.ellipse(image, (89, 95), (37, 34), 0, 0, 360, (75, 95, 75), -1)
        result = CupTopDetector(backend='green_top').process(image)
        self.assertTrue(result['valid'], result)
        self.assertLess(np.linalg.norm(np.asarray(result['center_px'])-[89, 95]), 1.5)
        established = CupTopDetector(FakeSegmentor(items)).process(image)
        self.assertFalse(established['valid'])
        self.assertEqual(established['reason'], 'no_supported_internal_top_edge')

    def test_two_supported_nested_internal_edges_remain_ambiguous(self):
        image, _ = scene(top=False)
        cv2.ellipse(image, (89, 95), (42, 39), 0, 0, 360, (55, 155, 55), -1)
        cv2.ellipse(image, (89, 95), (32, 30), 0, 0, 360, (80, 200, 80), -1)
        result = CupTopDetector(backend='green_top').process(image)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'ambiguous_internal_top_edges')
        self.assertIsNone(result['center_px'])

    def test_invalid_frame_or_mask_rejected(self):
        detector = CupTopDetector(FakeSegmentor([]))
        with self.assertRaises(ValueError):
            detector.process(np.zeros((10, 10), np.uint8))
        image, items = scene()
        items[0]['mask'] = np.zeros((1, 1), bool)
        with self.assertRaises(ValueError):
            CupTopDetector(FakeSegmentor(items)).process(image)

    def test_unknown_model_contract_needs_explicit_classes(self):
        class Unknown:
            pass
        with self.assertRaises(ValueError):
            CupTopDetector(Unknown())


if __name__ == '__main__':
    unittest.main()
