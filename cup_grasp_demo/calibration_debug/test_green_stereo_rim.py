"""Independent synthetic stereo geometry and failure cases; no hardware actions."""
import unittest
import cv2
import numpy as np
from cup_grasp_demo.calibration_debug.green_stereo_rim import make_view, project, fit_circle


class StereoRimTest(unittest.TestCase):
    def setUp(self):
        self.table = np.array([0., 0., .665])
        self.normal = np.array([0., 0., -1.])
        self.views = []
        theta = np.linspace(0, 2*np.pi, 360, endpoint=False)
        points = np.c_[.0375*np.cos(theta), .0375*np.sin(theta), np.full(360, .6)]
        for baseline in (0, .05):
            record = {'intrinsics': dict(width=640, height=480, fx=600., fy=600.,
                                         cx=320., cy=240., dist_coeffs=[0]*5),
                      'to_color_rotation': np.eye(3).ravel(order='F').tolist(),
                      'to_color_translation': [baseline, 0, 0]}
            image = np.full((480, 640), 150, np.uint8)
            blank = make_view(image, record)
            uv = project(points, blank)
            cv2.fillPoly(image, [np.rint(uv).astype('int32')], 20)
            self.views.append(make_view(image, record))

    def test_fixed_height_recovers_position_and_radius(self):
        for center in ([0., 0., .6],):
            fitted = fit_circle(self.views, self.table, self.normal,
                                [[.003,.002,.59,.034],[-.002,.001,.62,.04]],
                                [.025,.075],[.04,.18],fixed_height=.065)
            self.assertAlmostEqual(fitted['height_m'],.065,places=12)
            np.testing.assert_allclose(fitted['center'],center,atol=.002)
            self.assertAlmostEqual(fitted['radius_m']*2,.075,delta=.002)
        shifted = [dict(v, k=dict(v['k'], cx=v['k']['cx']+15)) for v in self.views]
        moved = fit_circle(shifted, self.table, self.normal,
                           [[-.01,0,.6,.0375]], [.025,.075],[.04,.18],fixed_height=.065)
        self.assertAlmostEqual(moved['center'][0],-.015,delta=.002)
        bad = list(self.views)
        bad[1] = dict(bad[1], distance=np.full((480,640),30.))
        with self.assertRaises(ValueError):
            fit_circle(bad,self.table,self.normal,[[0,0,.6,.0375]],
                       [.025,.075],[.04,.18],fixed_height=.065)

    def test_height_mode_validation(self):
        from cup_grasp_demo.calibration_debug.green_stereo_rim import height_options
        self.assertEqual(height_options({})[0],'measured')
        opts=dict(height_mode='fixed',fixed_height_mm=65,geometry_method='stereo_rim',height_range_mm=[40,180])
        self.assertEqual(height_options(opts),('fixed',.065))
        for change in (dict(fixed_height_mm=0),dict(fixed_height_mm=200),dict(height_mode='typo'),dict(geometry_method='depth_band')):
            with self.assertRaises(ValueError):height_options(dict(opts,**change))

    def test_recovers_metric_height_and_diameter_without_size_reference(self):
        fitted = fit_circle(self.views, self.table, self.normal,
                            [[.003, .002, .59, .034], [-.002, .001, .62, .04]],
                            [.025, .075], [.04, .18])
        self.assertAlmostEqual(fitted['height_m'], .065, delta=.004)
        self.assertAlmostEqual(fitted['radius_m']*2, .075, delta=.002)
        np.testing.assert_allclose(fitted['center'], [0, 0, .6], atol=.004)

    def test_scale_comes_from_baseline_not_reference_size(self):
        for view in self.views:
            view['t'] *= 1.5
        fitted = fit_circle(self.views, self.table*1.5, self.normal,
                            [[.003, .002, .89, .054], [-.002, .001, .92, .06]],
                            [.025, .075], [.04, .18])
        self.assertAlmostEqual(fitted['height_m'], .0975, delta=.006)
        self.assertAlmostEqual(fitted['radius_m']*2, .1125, delta=.003)

    def test_quality_configuration_validation(self):
        from cup_grasp_demo.calibration_debug.green_stereo_rim import quality_options
        for bad in ({'min_edge_support': 1.1}, {'edge_distance_px': float('nan')},
                    {'max_center_spread_mm': True}, {'unknown': 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                quality_options(bad)

    def test_single_camera_cannot_supply_metric_scale(self):
        with self.assertRaisesRegex(ValueError, 'independent'):
            fit_circle([self.views[0]]*2, self.table, self.normal,
                       [[0, 0, .6, .0375]], [.025, .075], [.04, .18])

    def test_one_missing_rim_is_rejected(self):
        self.views[1]['distance'][:] = 30
        with self.assertRaises(ValueError):
            fit_circle(self.views, self.table, self.normal,
                       [[0, 0, .6, .0375]], [.025, .075], [.04, .18])

    def test_distorted_images_require_rectification(self):
        with self.assertRaisesRegex(ValueError, 'rectified'):
            make_view(np.zeros((480, 640), np.uint8), {
                'intrinsics': dict(width=640, height=480, dist_coeffs=[.1, 0, 0, 0, 0])})


class StereoSequenceTest(unittest.TestCase):
    def test_shared_starts_do_not_replace_independent_observations(self):
        from unittest.mock import patch
        from cup_grasp_demo.calibration_debug.green_stereo_rim import refine_sequence, QUALITY
        initial = [dict(center=[0, 0, .60], radius_m=.0375),
                   dict(center=[0, 0, .61], radius_m=.037)]
        views = [object(), object()]
        independent_results = [dict(center=[0, 0, .6001]), dict(center=[0, 0, .6102])]
        with patch('cup_grasp_demo.calibration_debug.green_stereo_rim.fit_circle',
                   side_effect=independent_results) as fit:
            result = refine_sequence(views, initial, [0, 0, 1], [0, 0, -1],
                                     [.025, .075], [.04, .18], QUALITY)
        self.assertEqual(result, independent_results)
        self.assertIs(fit.call_args_list[0].args[0], views[0])
        self.assertIs(fit.call_args_list[1].args[0], views[1])
        for call in fit.call_args_list:
            np.testing.assert_allclose(call.args[3], [[0, 0, .60, .0375], [0, 0, .61, .037]])

    def test_bad_first_search_can_be_recovered_but_each_frame_is_rechecked(self):
        from unittest.mock import patch
        from cup_grasp_demo.calibration_debug.green_stereo_rim import fit_frame_sequence, QUALITY
        good = dict(center=[0, 0, .6], radius_m=.0375)
        views = [object(), object()]
        diag = {}
        with patch('cup_grasp_demo.calibration_debug.green_stereo_rim.fit_circle',
                   side_effect=[ValueError('bad initial guess'), good, good, good]) as fit:
            result = fit_frame_sequence(views, [0, 0, 1], [0, 0, -1],
                                       [[0, 0, .7, .04]], [.025, .075], [.04, .18], QUALITY, diag)
        self.assertEqual(result, [good, good])
        self.assertIsNone(diag['initial_frames'][0])
        self.assertEqual(diag['initial_frame_errors'][0]['frame'], 0)
        self.assertIs(fit.call_args_list[2].args[0], views[0])
        self.assertIs(fit.call_args_list[3].args[0], views[1])

    def test_missing_edges_are_not_rescued_by_neighbor_frames(self):
        from unittest.mock import patch
        from cup_grasp_demo.calibration_debug.green_stereo_rim import fit_frame_sequence, QUALITY
        good = dict(center=[0, 0, .6], radius_m=.0375)
        with patch('cup_grasp_demo.calibration_debug.green_stereo_rim.fit_circle',
                   side_effect=[ValueError('missing edges'), good, ValueError('missing edges')]):
            with self.assertRaisesRegex(ValueError, 'missing edges'):
                fit_frame_sequence([object(), object()], [0, 0, 1], [0, 0, -1],
                                   [[0, 0, .7, .04]], [.025, .075], [.04, .18], QUALITY, {})

    def test_real_motion_still_fails_original_limits(self):
        from cup_grasp_demo.calibration_debug.green_stereo_rim import validate_sequence, QUALITY
        frames = [dict(center=[x, 0, .6], height_m=.065, radius_m=.0375)
                  for x in [0, 0, 0, .01, .02]]
        diag = {}
        with self.assertRaisesRegex(ValueError, 'center spread 20.00 mm'):
            validate_sequence(frames, QUALITY, diag)
        self.assertEqual(diag['center_spread_mm'], 20)

    def test_failed_batch_height_variation_remains_rejected(self):
        from cup_grasp_demo.calibration_debug.green_stereo_rim import validate_sequence, QUALITY
        heights = [.0720637, .0663968, .0659890, .0687016, .0686496]
        frames = [dict(center=[0, 0, .735-h], height_m=h, radius_m=.0373) for h in heights]
        with self.assertRaisesRegex(ValueError, 'height span 6.07 mm'):
            validate_sequence(frames, QUALITY, {})


if __name__ == '__main__':
    unittest.main()


class FixedQuorumTest(unittest.TestCase):
    def test_quorum_preserves_quality_and_records_rejected_frames(self):
        from unittest.mock import patch
        from cup_grasp_demo.calibration_debug.green_stereo_rim import fit_fixed_sequence, QUALITY
        good = dict(center=[0, 0, .6], height_m=.065, radius_m=.0375)
        diagnostic = {}
        with patch('cup_grasp_demo.calibration_debug.green_stereo_rim.fit_circle',
                   side_effect=[ValueError('weak edge'), good, ValueError('no rim'), good, good]) as fit:
            results, first = fit_fixed_sequence([None]*5, [0,0,.665], [0,0,-1], [],
                [.025,.075], [.04,.18], QUALITY, .065, 3, diagnostic)
        self.assertEqual(first, 1)
        self.assertEqual(len(results), 3)
        self.assertEqual(diagnostic['accepted_frame_indices'], [1,3,4])
        self.assertEqual(len(diagnostic['rejected_frames']), 2)
        self.assertTrue(all(call.args[6] is QUALITY for call in fit.call_args_list))

    def test_insufficient_frames_and_motion_still_fail(self):
        from unittest.mock import patch
        from cup_grasp_demo.calibration_debug.green_stereo_rim import fit_fixed_sequence, QUALITY
        good = dict(center=[0,0,.6], height_m=.065, radius_m=.0375)
        moved = dict(good, center=[.02,0,.6])
        for values, minimum in (([good,good]+[ValueError('weak')]*3,3),
                                ([good]*3+[ValueError('weak')]*2,5),
                                ([good,good,moved]+[ValueError('weak')]*2,3)):
            with patch('cup_grasp_demo.calibration_debug.green_stereo_rim.fit_circle', side_effect=values):
                with self.assertRaises(ValueError):
                    fit_fixed_sequence([None]*5,[0,0,.665],[0,0,-1],[],[.025,.075],
                                       [.04,.18],QUALITY,.065,minimum,{})
        for invalid in (True,2,6,3.5):
            with self.assertRaises(ValueError):
                fit_fixed_sequence([],[],[],[],[],[],QUALITY,.065,invalid,{})
