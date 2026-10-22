"""Low-contrast stereo observations and original-mode regression; no devices."""
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from vision.geometry.circle_rim import (
    detect_stereo, fit_circle, ir_contrast_mode, make_view, project,
)


class StereoContrastTests(unittest.TestCase):
    def render(self, centers=(0.,), contrast='none', missing_right=False):
        views = []
        angle = np.linspace(0, 2*np.pi, 360, endpoint=False)
        for index, baseline in enumerate((0., .05)):
            record = dict(
                intrinsics=dict(width=640, height=480, fx=600., fy=600.,
                                cx=320., cy=240., dist_coeffs=[0]*5),
                to_color_rotation=np.eye(3).ravel(order='F'),
                to_color_translation=[baseline, 0, 0])
            image = np.full((480, 640), 110, np.uint8)
            blank = make_view(image, record)
            if not (missing_right and index == 1):
                for x in centers:
                    points = np.c_[x+.0375*np.cos(angle), .0375*np.sin(angle),
                                   np.full(len(angle), .6)]
                    uv = project(points, blank)
                    cv2.fillPoly(image, [np.rint(uv).astype('int32')], 100)
            views.append(make_view(image, record, contrast=contrast))
        return views

    def fit(self, views, seeds=None):
        return fit_circle(views, np.array([0., 0., .665]), np.array([0., 0., -1.]),
                          [[0, 0, .6, .0375]] if seeds is None else seeds,
                          [.025, .075], [.04, .18], fixed_height=.065,
                          analytic_jacobian=True)

    def test_low_contrast_rim_recovers_metric_circle_without_relaxing_quality(self):
        with self.assertRaises(ValueError):
            self.fit(self.render())
        fit = self.fit(self.render(contrast='clahe'))
        np.testing.assert_allclose(fit['center'], [0, 0, .6], atol=.002)
        self.assertAlmostEqual(fit['radius_m']*2000, 75., delta=2.)
        self.assertEqual(fit['edge_quality']['failed_checks'], [])

    def test_missing_view_cannot_be_recovered_by_contrast(self):
        with self.assertRaises(ValueError):
            self.fit(self.render(contrast='clahe', missing_right=True))

    def test_competing_circles_remain_rejected(self):
        views = self.render(centers=(-.05, .05), contrast='clahe')
        with self.assertRaisesRegex(ValueError, 'ambiguous competing circles'):
            self.fit(views, [[-.05, 0, .6, .0375], [.05, 0, .6, .0375]])

    def test_default_keeps_original_edge_map_and_image(self):
        for view in self.render():
            edges = cv2.Canny(view['image'], 25, 60)
            expected = cv2.distanceTransform(255-edges, cv2.DIST_L2, 5)
            np.testing.assert_array_equal(view['distance'], expected)
        raw, enhanced = self.render(), self.render(contrast='clahe')
        for before, after in zip(raw, enhanced):
            np.testing.assert_array_equal(before['image'], after['image'])
            np.testing.assert_array_equal(before['R'], after['R'])
            np.testing.assert_array_equal(before['t'], after['t'])
            self.assertEqual(before['k'], after['k'])

    def test_unknown_mode_is_rejected_during_pipeline_validation(self):
        from cup_grasp_demo.flow.core import ROOT, load_config
        from cup_grasp_demo.flow.green_pipeline import validate
        cfg = load_config(ROOT/'configs/green_cup.json')
        self.assertEqual(ir_contrast_mode({}), 'none')
        for bad in ('auto', True, [], None):
            with self.subTest(value=bad):
                changed = copy.deepcopy(cfg)
                changed['green_cup']['perception']['stereo_ir_contrast'] = bad
                with self.assertRaisesRegex(ValueError, 'stereo_ir_contrast'):
                    validate(changed)

    def test_explicit_opening_quantile_is_used_and_legacy_default_preserved(self):
        image = np.full((100, 100, 3), (0, 0, 255), np.uint8)
        for raw, expected in (({}, .3), ({'opening_value_quantile': .5}, .5)):
            opts = dict(min_area_px=10, image_rim=raw)
            diagnostics = {}
            with patch('vision.geometry.circle_rim.resolve_table',
                       return_value=(None, None, None, None, 'calibrated')), \
                 patch('vision.geometry.circle_rim.image_contour',
                       side_effect=ValueError('no contour')) as contour:
                with self.assertRaisesRegex(ValueError, 'No complete image contour'):
                    detect_stereo(None, None, image, {'intrinsics': {}}, opts, 6,
                                  [{'mask': np.ones((100, 100), bool)}], diagnostics)
            self.assertEqual(contour.call_args_list[0].args[2]['image_rim']['opening_value_quantile'], expected)
            self.assertEqual(diagnostics['stereo_image_processing']['opening_value_quantile'], expected)


if __name__ == '__main__':
    unittest.main()
