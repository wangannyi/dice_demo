"""Borderline edge residual and bounded fresh-frame recovery, offline only."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from cup_grasp_demo.calibration_debug.green_stereo_rim import (
    check_edge_quality, quality_options, RimEdgeQualityError)
from cup_grasp_demo.calibration_debug.green_pipeline import Workflow

class EdgeRecoveryTest(unittest.TestCase):
    def errors(self):
        errors=np.full((2,120),(.830*240-17*2.1)/223)
        errors[0,:6]=2.1;errors[1,:11]=2.1
        return errors

    def quality(self):
        return quality_options(dict(max_mean_edge_error_px=1.,max_view_edge_error_px=1.2))

    def test_reported_boundary_accepted_without_changing_support_gate(self):
        with self.assertRaises(RimEdgeQualityError) as old:
            check_edge_quality(self.errors(),quality_options(),.065,.0749/2)
        self.assertEqual(old.exception.report['failed_checks'],['max_mean_edge_error_px'])
        report=check_edge_quality(self.errors(),self.quality(),.065,.0749/2)
        self.assertAlmostEqual(report['mean_edge_error_px'],.830)
        self.assertEqual(report['edge_support'],[.95,109/120])
        self.assertEqual(report['failed_checks'],[])

    def test_configured_partial_rim_support_accepts_reported_case(self):
        import json
        root = Path(__file__).resolve().parents[3]
        raw = json.loads((root/'configs/green_cup.json').read_text())
        quality = quality_options(raw['green_cup']['perception']['stereo_rim'])
        errors = np.full((2,120), .5)
        errors[0,:15] = 2.1
        errors[1,:3] = 2.1
        report = check_edge_quality(errors, quality, .065, .0749/2)
        self.assertEqual(report['edge_support'], [.875, .975])
        self.assertEqual(report['failed_checks'], [])
        errors[0,:19] = 2.1  # Below 85%, despite a small mean error.
        with self.assertRaises(RimEdgeQualityError) as rejected:
            check_edge_quality(errors, quality, .065, .0749/2)
        self.assertIn('min_edge_support', rejected.exception.report['failed_checks'])

    def test_missing_edges_and_bad_view_still_rejected(self):
        for errors in [np.full((2,120),3.), np.array([[.1]*120,[1.3]*120])]:
            with self.assertRaises(RimEdgeQualityError):
                check_edge_quality(errors,self.quality(),.065,.0375)

    def test_retry_only_one_fresh_frame_and_never_stale_success(self):
        error=RimEdgeQualityError(dict(mean_edge_error_px=1.1,edge_support=[.95,.91],
            height_mm=65,diameter_mm=75,failed_checks=['max_mean_edge_error_px']))
        with tempfile.TemporaryDirectory() as tmp:
            w=object.__new__(Workflow);w.root=Path(tmp);w.args=SimpleNamespace(mode='fast');w._vision=object()
            w._capture_once=Mock(side_effect=[error,'fresh result'])
            self.assertEqual(w.capture(),'fresh result');self.assertEqual(w._capture_once.call_count,2)
            w._capture_once=Mock(side_effect=error)
            with self.assertRaises(RimEdgeQualityError):w.capture()
            self.assertEqual(w._capture_once.call_count,2)
            w._capture_once=Mock(side_effect=ValueError('ambiguous competing circles'))
            with self.assertRaises(ValueError):w.capture()
            self.assertEqual(w._capture_once.call_count,1)
