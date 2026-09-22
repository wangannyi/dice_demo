"""One-class decoding, geometry selection, frozen model identity and replay."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.flow import cup_perception as vision, debug
from cup_grasp_demo.flow.core import ROOT, digest, load_config, read_json
from cup_grasp_demo.flow.cup_selection import select_cup
from cup_grasp_demo.side_grasp.preview_index import load_batch
from cup_grasp_demo.side_grasp.preview_index import section
from dice_cup_localization.geometry import Config

MODEL = 'cup_grasp_demo/models/cup_yolov8n_seg_20260918/best.q.onnx'


class DecodeTest(unittest.TestCase):
    def test_one_class_coefficients_letterbox_nms_and_float_rounding(self):
        image = np.zeros((480, 640, 3), np.uint8)
        _, transform = vision.preprocess(image)
        pred = np.zeros((1, 37, 8400), np.float32)
        proto = np.zeros((1, 32, 160, 160), np.float32)
        proto[0, 0] = 1
        pred[0, 4] = -6e-8  # Observed SIMD sigmoid roundoff on K3.
        for i, score in [(0, .9), (1, .8)]:
            pred[0, :5, i] = [320, 320, 100, 120, score]
            pred[0, 5, i] = 10  # First coefficient is channel 5, not the old channel 6.
        results = vision.decode([pred, proto], image.shape, transform, vision.DEFAULTS)
        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item['class_id'], 0)
        np.testing.assert_allclose(item['bbox_xyxy'], [270, 180, 370, 300])
        ys, xs = np.nonzero(item['mask'])
        self.assertTrue(270 <= xs.min() < xs.max() < 370)
        self.assertTrue(180 <= ys.min() < ys.max() < 300)
        self.assertGreater(len(xs), 10000)
        pred[0, 4] = 0
        self.assertEqual(vision.decode([pred, proto], image.shape, transform, vision.DEFAULTS), [])

    def test_invalid_layout_nonfinite_and_unactivated_scores_rejected(self):
        pred = np.zeros((1, 37, 8400), np.float32)
        proto = np.zeros((1, 32, 160, 160), np.float32)
        args = ((480, 640, 3), (1, 0, 80, 640, 480), vision.DEFAULTS)
        with self.assertRaisesRegex(ValueError, 'one-class'):
            vision.decode([np.zeros((1, 38, 8400)), proto], *args)
        for invalid in (float('nan'), 2., -.01):
            pred[0, 4, 0] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                vision.decode([pred, proto], *args)

    def test_legacy_default_and_invalid_configuration(self):
        self.assertEqual(vision.perception_options({}), {'backend': 'depth_geometry'})
        base = dict(backend='yolo_seg_onnx', model=MODEL)
        self.assertEqual(vision.perception_options({'cup_perception': base})['section_half_band_mm'], 4.)
        for key, value in [('confidence', True), ('mask_threshold', 0),
                           ('min_valid_depth_fraction', float('nan')),
                           ('mask_erode_px', 1.5), ('model', '/tmp/outside.onnx'),
                           ('section_half_band_mm', True), ('section_half_band_mm', .4),
                           ('section_half_band_mm', 4.1), ('section_half_band_mm', float('nan'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                vision.perception_options({'cup_perception': dict(base, **{key: value})})


class SectionBandTest(unittest.TestCase):
    def test_narrow_band_excludes_adjacent_recess_without_changing_contact_height(self):
        rng = np.random.default_rng(7)
        height, fraction = .115, 10/11
        target = height*fraction
        angle = np.linspace(0, np.radians(220), 200)
        wall = np.column_stack((.032*np.cos(angle), .032*np.sin(angle), np.full(200, target)))
        theta = rng.uniform(0, 2*np.pi, 600)
        radius = .025*np.sqrt(rng.uniform(0, 1, 600))
        recess = np.column_stack((radius*np.cos(theta), radius*np.sin(theta), np.full(600, target+.003)))
        top = wall.copy()
        top[:, 2] = height
        points = np.vstack((wall, recess, top))
        args = (points, np.zeros(3), np.array([0., 0., 1.]), fraction, Config())
        with self.assertRaisesRegex(ValueError, 'insufficient_circular_side_surface'):
            section(*args)
        result = section(*args, half_band_m=.002)
        self.assertAlmostEqual(result['contact_height_m'], target)
        self.assertAlmostEqual(result['radius_m'], .032)
        self.assertEqual(result['section_points'], 200)
        # A genuinely non-circular section at the requested height still fails.
        points[:200, 0] *= 2
        with self.assertRaisesRegex(ValueError, 'insufficient_circular_side_surface'):
            section(*args, half_band_m=.002)


class ModelGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / 'configs/green_cup.json')
        cls.cfg['cup_perception'] = dict(backend='yolo_seg_onnx', model=MODEL)
        cls.cfg['contact_height_fraction'] = 10/11
        cls.fixture = ROOT / 'cup_grasp_demo/datasets/compare_20260917_221945'
        cls.meta, cls.depth, cls.image, _ = load_batch(cls.fixture)
        cls.instances, cls.provenance = vision.infer(cls.image, vision.perception_options(cls.cfg))
        cls.assert_inference = (cls.instances, cls.provenance)

    def test_no_detection_does_not_use_geometric_fallback(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(vision, 'infer', return_value=([], self.provenance)), \
             patch('cup_grasp_demo.flow.cup_selection.depth_candidates',
                   side_effect=AssertionError('No automatic fallback')):
            path = Path(temp)
            with self.assertRaisesRegex(ValueError, '没有候选'):
                select_cup(self.depth, self.image, self.meta, self.cfg, path)
            self.assertFalse(read_json(path / 'cup_candidates.json')['passed'])
            self.assertTrue((path / 'yolo_seg.png').exists())

    def test_outside_workspace_and_missing_cup_depth_rejected(self):
        outside = deepcopy(self.instances[0])
        outside['mask'][:] = False
        outside['mask'][380:400, 580:610] = True
        with patch.object(vision, 'infer', return_value=([outside], self.provenance)):
            proposals, rejected = vision.yolo_candidates(self.depth, self.image, self.meta, self.cfg)
            self.assertEqual(proposals, [])
            self.assertEqual(rejected[0]['reason'], 'outside_red_workspace')
        missing = self.depth.copy()
        for item in self.instances:
            missing[item['mask']] = 0
        with patch.object(vision, 'infer', return_value=self.assert_inference):
            proposals, rejected = vision.yolo_candidates(missing, self.image, self.meta, self.cfg)
            self.assertEqual(proposals, [])
            self.assertEqual(rejected[0]['reason'], 'insufficient_valid_depth_in_yolo_mask')

    def test_multiple_matching_cups_stay_ambiguous(self):
        with patch.object(vision, 'infer', return_value=(self.instances*2, self.provenance)):
            with self.assertRaisesRegex(ValueError, '多个候选'):
                select_cup(self.depth, self.image, self.meta, self.cfg)

    def test_changed_model_invalidates_frozen_session(self):
        hashes = debug.source_hashes(self.cfg)
        self.assertEqual(hashes[MODEL], self.provenance['model_sha256'])
        with tempfile.TemporaryDirectory(dir=ROOT / 'cup_grasp_demo/datasets') as temp:
            directory = Path(temp)
            model = directory / 'model.onnx'
            model.write_bytes(b'original')
            session = {'source_hashes': {str(model.relative_to(ROOT)): digest(model)}}
            (directory / 'session.json').write_text(json.dumps(session))
            model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, '参数/模型已变更'):
                debug.verify_session(directory)


if __name__ == '__main__':
    unittest.main()
