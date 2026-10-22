"""Candidate selection keeps every per-view quality gate; no hardware."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vision.geometry.circle_rim import RimEdgeQualityError, fit_circle


class CandidateQualityTests(unittest.TestCase):
    def fit_candidates(self, good_errors, other_errors):
        views = []
        for index, baseline in enumerate((0., .05)):
            views.append(dict(k=dict(width=640, height=480, fx=600., fy=600.,
                                     cx=320., cy=240.),
                              R=np.eye(3), t=np.array([baseline, 0., 0.]),
                              distance=np.full((480, 640), index, dtype=float)))

        def sampled_errors(distance, coordinates, **kwargs):
            side = int(distance[0, 0])
            # z=.6 and f=600: projected average x identifies the candidate.
            center_x = (np.mean(coordinates[1])-320)/1000 + views[side]['t'][0]
            source = good_errors if abs(center_x) < .01 else other_errors
            return np.broadcast_to(source[side], (120,)).copy()

        # Fixed optimizer endpoints isolate qualification/selection from search
        # convergence. Rasterized two-circle integration tests cover the search.
        with patch('vision.geometry.circle_rim.least_squares',
                   side_effect=lambda fun, start, **kw: SimpleNamespace(x=np.asarray(start))), \
             patch('vision.geometry.circle_rim.map_coordinates', side_effect=sampled_errors):
            return fit_circle(views, np.array([0., 0., .665]), np.array([0., 0., -1.]),
                              [[0., 0., .6, .038], [.02, 0., .6, .0445]],
                              [.025, .075], [.04, .18],
                              dict(max_mean_edge_error_px=1., max_view_edge_error_px=1.2,
                                   ambiguity_score_margin_px=.1))

    def test_lower_mean_but_insufficient_single_view_support_does_not_win(self):
        unsupported = np.r_[np.full(20, 4.), np.full(100, .5)]
        result = self.fit_candidates([.95, .95], [unsupported, .8])
        self.assertAlmostEqual(result['radius_m']*2000, 76.)
        self.assertEqual(result['qualified_candidate_count'], 1)
        self.assertIn('min_edge_support', result['rejected_candidate_quality'][0]['failed_checks'])
        self.assertLess(result['rejected_candidate_quality'][0]['mean_edge_error_px'],
                        result['mean_edge_error_px'])

    def test_nearby_score_but_insufficient_support_cannot_veto_valid_rim(self):
        unsupported = np.r_[np.full(20, 4.), np.full(100, .55)]
        result = self.fit_candidates([.95, .95], [unsupported, .8])
        self.assertAlmostEqual(result['radius_m']*2000, 76.)
        self.assertGreater(result['rejected_candidate_quality'][0]['mean_edge_error_px'],
                           result['mean_edge_error_px'])

    def test_single_view_mean_gate_is_not_hidden_by_good_other_view(self):
        result = self.fit_candidates([.7, .7], [1.21, .1])
        self.assertEqual(result['qualified_candidate_count'], 1)
        self.assertEqual(result['rejected_candidate_quality'][0]['failed_checks'],
                         ['max_view_edge_error_px'])

    def test_two_qualified_distinct_circles_are_still_ambiguous(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous competing circles'):
            self.fit_candidates([.95, .95], [.96, .96])

    def test_no_qualified_candidate_preserves_best_failure(self):
        unsupported = np.r_[np.full(20, 4.), np.full(100, .5)]
        with self.assertRaises(RimEdgeQualityError) as caught:
            self.fit_candidates([1.5, 1.5], [unsupported, .8])
        self.assertAlmostEqual(caught.exception.report['diameter_mm'], 89.)
        self.assertEqual(caught.exception.report['failed_checks'], ['min_edge_support'])


if __name__ == '__main__':
    unittest.main()
