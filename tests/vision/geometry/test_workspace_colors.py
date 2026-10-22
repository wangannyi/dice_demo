"""Table colour selection without opening a camera or commanding hardware."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from cup_grasp_demo.flow import planar_scene
from vision.geometry.circle_rim import detect_stereo
from vision.geometry.workspace import perception_workspace, workspace_masks, workspace_options


class WorkspaceColorTests(unittest.TestCase):
    def scene(self):
        image = np.full((240, 320, 3), (102, 102, 102), np.uint8)
        image[:, 160:] = (55, 226, 176)  # BGR for #B0E237
        image[90:130, 210:250] = (60, 60, 60)  # cup hole
        image[10:40, 170:200] = (255, 255, 255)  # white reference
        return image

    def test_green_and_gray_select_opposite_sides_and_fill_cup_holes(self):
        image = self.scene()
        green, region = workspace_masks(image, {'color': 'green'})
        gray, gray_region = workspace_masks(image, {'color': 'gray'})
        self.assertTrue(green[50, 280])
        self.assertFalse(green[50, 50])
        self.assertFalse(green[100, 230])
        self.assertEqual(region[100, 230], 1)
        self.assertTrue(gray[50, 50])
        self.assertFalse(gray[50, 280])
        self.assertFalse(gray[100, 230])  # isolated gray cup excluded
        self.assertEqual(gray_region[50, 280], 0)

    def test_green_survives_moderate_shadow_and_desaturation(self):
        hsv = np.full((100, 100, 3), (38, 50, 80), np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        surface, _ = workspace_masks(image, {'color': 'green'})
        self.assertTrue(surface.all())

    def test_small_colored_object_is_not_a_table(self):
        image = np.full((200, 300, 3), 255, np.uint8)
        image[20:35, 20:35] = (55, 226, 176)
        surface, region = workspace_masks(image, {'color': 'green'})
        self.assertFalse(surface.any())
        self.assertFalse(region.any())

    def test_envelope_fills_edge_connected_glare_without_fabricating_table_depth(self):
        image = self.scene()
        # A low-saturation reflection reaches the outer edge and joins the cup.
        image[95:120, 220:] = (160, 160, 160)
        opts = {'color': 'green', 'region_shape': 'convex_hull'}
        surface, region = workspace_masks(image, opts)
        old_surface, old_region = workspace_masks(image, {'color': 'green'})
        self.assertFalse(old_region[110, 230])
        self.assertTrue(region[110, 230])
        self.assertFalse(surface[110, 230])
        np.testing.assert_array_equal(surface, old_surface)
        self.assertFalse(region[:, :160].any())  # opposite gray side stays out

    def test_envelope_does_not_promote_small_object_to_workspace(self):
        image = np.full((200, 300, 3), 255, np.uint8)
        image[20:35, 20:35] = (55, 226, 176)
        surface, region = workspace_masks(image, {'color': 'green', 'region_shape': 'convex_hull'})
        self.assertFalse(surface.any())
        self.assertFalse(region.any())

    def test_site_hue_override_accepts_yellow_cast_without_gray(self):
        hsv = np.full((100, 200, 3), (22, 50, 140), np.uint8)
        hsv[:, :100] = (0, 0, 102)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        surface, region = workspace_masks(image, {'color': 'green', 'hue_ranges': [[20, 55]],
                                                  'region_shape': 'convex_hull'})
        self.assertTrue(surface[:, 100:].all())
        self.assertFalse(region[:, :100].any())

    def test_configuration_validates_before_devices_open(self):
        for raw in ({'color': 'blue'}, {'hue_ranges': [[55, 25]]},
                    {'hue_ranges': [[False, 30]]}, {'max_value': 999},
                    {'min_region_fraction': float('nan')}, {'min_value': True},
                    {'color': []}, {'region_shape': 'rectangle'}, {'region_shape': []}, {'typo': 3}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                workspace_options(raw)
        with self.assertRaisesRegex(ValueError, 'remove legacy'):
            perception_workspace({'workspace': {'color': 'green'}, 'red_workspace': {}})

    def test_legacy_red_thresholds_still_work(self):
        options = perception_workspace({'red_workspace': {'min_saturation': 50}})
        self.assertEqual(options['color'], 'red')
        self.assertEqual(options['min_saturation'], 50)

    def test_stereo_filters_cups_with_the_same_workspace_setting(self):
        image = self.scene()
        mask = np.zeros(image.shape[:2], bool)
        mask[90:130, 210:250] = True
        diagnostic = {}
        with patch('vision.geometry.circle_rim.resolve_table', return_value=(None, None, None, None, 'calibrated')):
            # Ambiguous cups are still refused; inspect selection before geometry.
            with self.assertRaisesRegex(ValueError, 'requires one YOLO cup in the green workspace'):
                detect_stereo(None, None, image, None,
                              {'min_area_px': 10, 'workspace': {'color': 'green'}},
                              6, [{'mask': mask}, {'mask': mask}], diagnostic)
        self.assertTrue(all(c['accepted'] for c in diagnostic['workspace']['candidates']))
        self.assertNotIn('red_workspace', diagnostic)

    def test_table_registration_uses_green_surface_and_preserves_depth_checks(self):
        image = self.scene()
        depth = np.full(image.shape[:2], 700, np.uint16)
        depth[90:130, 210:250] = 650  # object is above the plane, never table support
        meta = {'intrinsics': {'frame': 'color_optical', 'width': 320, 'height': 240,
                              'fx': 300, 'fy': 300, 'cx': 160, 'cy': 120,
                              'dist_coeffs': [0]*5}, 'depth_scale_m': .001}
        cfg = {'plane_tolerance_mm': 6, 'green_cup': {'perception': {'workspace': {'color': 'green'}}}}
        with patch.object(planar_scene, 'load_batch', return_value=(meta, depth, image, [])), \
             patch.object(planar_scene.common, 'camera_transform', return_value=(np.eye(4), True)):
            scene, quality = planar_scene.home_scene(None, cfg)
            self.assertTrue(quality)
            self.assertAlmostEqual(scene['cup_support_base_m'][2], .7)
            self.assertLess(scene['table_fit']['rms_mm'], .01)
            self.assertEqual(scene['workspace']['color'], 'green')
            depth[:] = 0
            with self.assertRaisesRegex(ValueError, 'insufficient_valid_depth'):
                planar_scene.home_scene(None, cfg)
