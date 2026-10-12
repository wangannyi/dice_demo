"""Workspace selection with a desaturated red reflection connecting to the border."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from vision.geometry.circle_rim import detect_stereo, workspace_masks, workspace_options


class RedWorkspaceTests(unittest.TestCase):
    def image(self):
        hsv = np.full((100, 160, 3), (110, 180, 180), dtype=np.uint8)
        hsv[:, 50:] = (2, 130, 180)
        hsv[40:65, 50:100] = (2, 60, 180)
        hsv[35:60, 85:110] = (60, 180, 150)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def test_desaturated_corridor_no_longer_excludes_cup_but_blue_stays_out(self):
        image = self.image()
        _, old = workspace_masks(image)
        _, adjusted = workspace_masks(image, {'min_saturation': 55})
        self.assertEqual(old[45, 95], 0)
        self.assertEqual(adjusted[45, 95], 1)
        self.assertEqual(adjusted[45, 20], 0)

    def test_omitted_options_preserve_original_mask_exactly(self):
        image = np.random.default_rng(4).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        expected = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50)
        red, workspace = workspace_masks(image)
        np.testing.assert_array_equal(red, expected)
        contours, _ = cv2.findContours(expected.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        original = np.zeros(expected.shape, np.uint8)
        cv2.drawContours(original, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
        np.testing.assert_array_equal(workspace, original)

    def test_invalid_configuration_is_rejected(self):
        for raw in ([], {'typo': 55}, {'min_value': True}, {'min_saturation': -1},
                    {'min_saturation': 255}, {'min_saturation': 55.5}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                workspace_options(raw)

    def test_missing_or_multiple_candidates_still_rejected_with_diagnostics(self):
        mask = np.zeros((100, 160), bool)
        mask[35:60, 85:110] = True
        for instances in ([], [{'mask': mask}, {'mask': mask}], [{'mask': np.zeros_like(mask)}]):
            diagnostic = {}
            with patch('vision.geometry.circle_rim.resolve_table', return_value=(None, None, None, None, 'calibrated')):
                with self.assertRaisesRegex(ValueError, 'requires one YOLO cup'):
                    detect_stereo(None, None, self.image(), None,
                                  {'min_area_px': 10, 'red_workspace': {'min_saturation': 55}},
                                  6, instances, diagnostic)
            self.assertEqual(len(diagnostic['red_workspace']['candidates']), len(instances))

    def test_no_red_region_returns_empty_workspace(self):
        red, workspace = workspace_masks(np.zeros((20, 20, 3), np.uint8))
        self.assertFalse(red.any())
        self.assertFalse(workspace.any())
