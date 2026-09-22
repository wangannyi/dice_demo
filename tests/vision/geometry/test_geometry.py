"""Synthetic geometric accuracy and failure gates; no camera or robot access."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from vision.geometry.table_plane import deproject, estimate_points, localize


def scene(arc=160, tilt=0, noise=0.0002):
    """Create a partial tapered side surface and a support plane."""
    rng = np.random.default_rng(21)
    angles, heights = np.meshgrid(np.linspace(-arc/2, arc/2, 100)*np.pi/180,
                                 np.linspace(.008, .12, 80))
    radius = .0375 - .008 * heights / .12
    obj = np.column_stack((.03 + (radius*np.sin(angles)).ravel(),
                           .15 - heights.ravel(), (.70-radius*np.cos(angles)).ravel()))
    obj += rng.normal(0, noise, obj.shape)
    x, z = np.meshgrid(np.linspace(-.15, .2, 30), np.linspace(.5, .9, 30))
    table = np.column_stack((x.ravel(), np.full(x.size, .15), z.ravel()))
    c, s = np.cos(tilt), np.sin(tilt)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return obj @ rotation.T, table @ rotation.T, rotation @ np.array([.03, .09, .70])


class GeometryTests(unittest.TestCase):
    def test_known_partial_tapered_geometry(self):
        for tilt in (0, .3):
            obj, table, expected = scene(tilt=tilt)
            result = estimate_points(obj, table)
            np.testing.assert_allclose(result['center_m'], expected, atol=.002)
            self.assertAlmostEqual(result['dimensions']['observed_height_m'], .12, delta=.002)
            self.assertFalse(result['quality']['hardware_validated'])
            self.assertFalse(result['top_surface']['valid'])

    def test_visible_top_plane_axis_contact(self):
        rng = np.random.default_rng(5)
        for tilt in (0, .3):
            obj, table, _ = scene(tilt=tilt)
            angle = rng.uniform(-np.pi, np.pi, 1200)
            radius = .028 * np.sqrt(rng.random(1200))
            cap = np.column_stack((.03 + radius * np.sin(angle),
                                   np.full(1200, .03), .70 - radius * np.cos(angle)))
            rotation = np.array([[np.cos(tilt), -np.sin(tilt), 0],
                                 [np.sin(tilt), np.cos(tilt), 0], [0, 0, 1]])
            result = estimate_points(np.vstack((obj, cap @ rotation.T)), table)
            top = result['top_surface']
            self.assertTrue(top['valid'], top)
            np.testing.assert_allclose(top['center_m'], rotation @ [.03, .03, .70], atol=.002)
            self.assertFalse(top['quality']['rim_center_independently_measured'])

    def test_unobserved_top_center_rejected_without_changing_body(self):
        obj, table, _ = scene()
        rng = np.random.default_rng(8)
        angle = rng.uniform(-np.pi, np.pi, 1200)
        radius = .02 + .008 * rng.random(1200)
        cap = np.column_stack((.03 + radius * np.sin(angle),
                               np.full(1200, .03),
                               .70 - radius * np.cos(angle)))
        result = estimate_points(np.vstack((obj, cap)), table)
        self.assertFalse(result['top_surface']['valid'])
        self.assertEqual(result['top_surface']['reason'], 'top_center_unobserved')

    def test_table_contamination_tolerated(self):
        obj, table, expected = scene()
        table[:90, 1] -= .03
        result = estimate_points(obj, table)
        np.testing.assert_allclose(result['center_m'], expected, atol=.002)

    def test_background_in_object_rejected(self):
        obj, table, _ = scene()
        with self.assertRaisesRegex(ValueError, 'not_separated'):
            estimate_points(np.vstack((obj, np.tile(table, (8, 1)))), table)

    def test_too_little_visible_arc(self):
        obj, table, _ = scene(arc=35, noise=0)
        with self.assertRaisesRegex(ValueError, 'insufficient_side_sections'):
            estimate_points(obj, table)

    def test_top_only_is_unobservable(self):
        obj, table, _ = scene()
        obj[:, 1] = .03
        with self.assertRaisesRegex(ValueError, 'insufficient_side_sections'):
            estimate_points(obj, table)

    def test_leaning_axis_rejected(self):
        obj, table, _ = scene()
        obj[:, 0] += (.15 - obj[:, 1]) * .7
        with self.assertRaisesRegex(ValueError, 'upright_model_inconsistent'):
            estimate_points(obj, table)

    def test_deprojection_explicit_units(self):
        intr = {'frame': 'color_optical', 'height': 2, 'width': 2,
                'fx': 100, 'fy': 100, 'cx': 0, 'cy': 0, 'dist_coeffs': [0]*5}
        mask = np.ones((2, 2), dtype=bool)
        points = deproject(np.full((2, 2), 1000, dtype=np.uint16), mask, intr, .001)
        np.testing.assert_allclose(points[-1], [.01, .01, 1])
        points = deproject(np.full((2, 2), 1000, dtype=np.uint16), mask, intr, .0001)
        np.testing.assert_allclose(points[-1], [.001, .001, .1])
        intr['dist_coeffs'][0] = .01
        with self.assertRaisesRegex(ValueError, 'rectify'):
            deproject(np.ones((2, 2)), mask, intr, .001)

    def test_invalid_depth_returns_no_geometry(self):
        mask = np.ones((20, 20), dtype=bool)
        metadata = {'timestamp_ms': 123, 'timestamp_domain': 'hardware_clock',
                    'depth_registered_to': 'color_optical',
                    'frame_id': 'serial:123', 'instance_id': 'cup-1', 'mask_source': 'manual',
                    'depth_scale_m': .001, 'intrinsics': {
                        'frame': 'color_optical', 'height': 20, 'width': 20,
                        'fx': 100, 'fy': 100, 'cx': 10, 'cy': 10, 'dist_coeffs': [0]*5}}
        for bad in (0, np.nan, np.inf, -1):
            result = localize(np.full((20, 20), bad), mask, ~mask, metadata)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['geometry'])
            self.assertEqual(result['reason'], 'insufficient_valid_depth')
        result = localize(np.ones((20, 20)), mask, mask, metadata)
        self.assertIn('overlap', result['reason'])

    def test_degenerate_table(self):
        obj, table, _ = scene()
        table[:, 0] = 0
        with self.assertRaises(ValueError):
            estimate_points(obj, table)


class PlaneRefinementTest(unittest.TestCase):
    def test_refine_before_support_gate(self):
        from unittest.mock import Mock, patch
        from vision.geometry.table_plane import _plane, Config
        rng = np.random.default_rng(9)
        points = np.column_stack((rng.uniform(-.2,.2,(1000,2)), rng.uniform(-.003,.003,1000)))
        points = np.vstack(([[-.2,-.2,.0048],[.2,-.2,.0048],[0,.2,.0048]], points))
        initial = np.abs(points[:,2]-.0048) <= .006
        self.assertLess(initial.mean(), .8)
        chooser=Mock()
        chooser.choice.side_effect=lambda n,size,replace: np.arange(n) if size>3 else np.array([0,1,2])
        with patch('vision.geometry.table_plane.np.random.default_rng',return_value=chooser):
            _,_,fraction,_ = _plane(points,Config(plane_tolerance_m=.006))
        self.assertGreaterEqual(fraction,.8)

    def test_refinement_still_rejects_unsupported_cloud(self):
        from vision.geometry.table_plane import _plane, Config
        points=np.random.default_rng(22).uniform(-.2,.2,(3000,3))
        with self.assertRaisesRegex(ValueError,'table_plane_not_supported'):
            _plane(points,Config(plane_tolerance_m=.006))


if __name__ == '__main__':
    unittest.main()
