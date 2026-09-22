"""Initial selection with hand distractors, mobile cups, and frozen replay."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.flow import debug
from cup_grasp_demo.flow.core import ROOT, digest, load_config, read_json
from cup_grasp_demo.flow.cup_selection import choose_candidate, select_cup, selection_options
from cup_grasp_demo.flow.grasp import observed_scene
from vision.capture.frame_io import depth_candidates, depth_proposals, load_batch
from vision.geometry.table_plane import Config


class CupSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.failed = ROOT / 'cup_grasp_demo/datasets/capture_selection_20260917_01/failed'
        cls.config_path = ROOT / 'cup_grasp_demo/flow/index_joint_center/config.json'
        cls.cfg = load_config(cls.config_path)
        cls.cfg.pop('cup_perception', None)  # Preserve the established geometry backend regression.
        cls.cfg['contact_height_fraction'] = 9 / 11  # Frozen regression section.
        cls.meta, cls.depth, cls.image, _ = load_batch(cls.failed)
        cls.proposals, _ = depth_candidates(cls.depth, cls.image, cls.meta, Config(plane_tolerance_m=.005), 9 / 11)
        cls.cup = next(p for p in cls.proposals if p[0]['bbox_xywh'][:2] == [108, 188])
        cls.hand = next(p for p in cls.proposals if p is not cls.cup)

    def test_original_failure_and_configured_selection_on_real_frames(self):
        with self.assertRaisesRegex(ValueError, 'got 2'):
            depth_proposals(self.depth, self.image, self.meta, Config(plane_tolerance_m=.005), 9 / 11)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            selected = select_cup(self.depth, self.image, self.meta, self.cfg, output)
            report = read_json(output / 'cup_candidates.json')
            self.assertTrue(report['passed'])
            self.assertEqual(selected[0]['bbox_xywh'], [108, 188, 81, 78])
            self.assertAlmostEqual(selected[0]['height_m'] * 1000, 115.32097, places=4)
            self.assertEqual(report['candidates'][0]['reasons'], ['height_outside_range', 'diameter_outside_range'])
            self.assertFalse(report['position_filter_used'])
            self.assertFalse(report['semantic_identity_verified'])
            self.assertTrue((output / 'cup_candidates.png').exists())

    def test_one_hand_candidate_is_not_mistaken_for_cup(self):
        result = choose_candidate([self.hand], self.cfg)
        self.assertFalse(result['passed'])
        self.assertIsNone(result['selected_index'])

    def test_selection_does_not_depend_on_old_cup_location(self):
        for offset in ([.2, -.1, .1], [-.1, .1, .2]):
            moved = deepcopy(self.cup[0])
            moved['center_camera_m'] = (np.array(moved['center_camera_m']) + offset).tolist()
            moved['bbox_xywh'] = [340, 250, 81, 78]
            result = choose_candidate([self.hand, (moved, self.cup[1], self.cup[2])], self.cfg)
            self.assertTrue(result['passed'])
            self.assertEqual(result['selected_index'], 1)

    def test_two_cups_are_ambiguous_not_ranked(self):
        moved = deepcopy(self.cup[0])
        moved['center_camera_m'][0] += .2
        result = choose_candidate([self.cup, (moved, self.cup[1], self.cup[2])], self.cfg)
        self.assertFalse(result['passed'])
        self.assertIsNone(result['selected_index'])
        self.assertIn('多个候选', result['reason'])

    def test_no_candidate_or_plane_failure_saves_diagnostics(self):
        for effect in ('empty', 'bad_plane'):
            with tempfile.TemporaryDirectory() as temp, patch(
                    'cup_grasp_demo.flow.cup_selection.depth_candidates') as detect:
                if effect == 'empty':
                    detect.return_value = ([], [])
                else:
                    detect.side_effect = ValueError('plane missing')
                output = Path(temp)
                with self.assertRaises(ValueError):
                    select_cup(self.depth, self.image, self.meta, self.cfg, output)
                self.assertFalse(read_json(output / 'cup_candidates.json')['passed'])
                self.assertTrue((output / 'cup_candidates.png').exists())

    def test_invalid_bounds_are_rejected(self):
        for value in ([140, 90], [90, 90], [0, 140], [90, float('nan')], [True, 140], '90,140', [90]):
            cfg = deepcopy(self.cfg)
            cfg['cup_selection']['height_range_mm'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'height_range_mm'):
                selection_options(cfg)
        with self.assertRaisesRegex(ValueError, 'mode'):
            selection_options({'cup_selection': {'mode': 'nearest'}})

    def test_legacy_unconfigured_selection_remains_strict(self):
        cfg = {k:v for k,v in self.cfg.items() if k != 'cup_selection'}
        result = choose_candidate(self.proposals, cfg)
        self.assertFalse(result['passed'])
        self.assertIn('got 2', result['reason'])
        self.assertTrue(choose_candidate([self.hand], cfg)['passed'])
        path = ROOT / 'cup_grasp_demo/datasets/debug_regression_20260917/compare_20260917_221945'
        meta, depth, image, _ = load_batch(path)
        original = depth_proposals(depth, image, meta, Config(plane_tolerance_m=.005), 9 / 11)
        selected = select_cup(depth, image, meta, cfg)
        self.assertEqual(selected[0], original[0])
        np.testing.assert_array_equal(selected[1], original[1])
        np.testing.assert_array_equal(selected[2], original[2])

    def capture_fixture(self, directory, cfg=None):
        args = SimpleNamespace(config=self.config_path, session=directory, replay=None, show=True)
        snapshot = read_json(self.failed / 'after.json')
        def camera(output, config):
            shutil.copytree(self.failed / 'rgbd', output)
        return (args, patch.object(debug, 'bridge', return_value=snapshot),
                patch.object(debug, 'capture_rgbd', side_effect=camera),
                patch.object(debug, 'load_config', return_value=cfg or self.cfg))

    def test_capture_and_frozen_grasp_geometry_both_select_same_cup(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'session'
            args, bridge, camera, config = self.capture_fixture(directory)
            with bridge as sdk, camera, config, patch.object(debug, 'show'):
                debug.capture(args)
            self.assertEqual([c.args[0] for c in sdk.call_args_list], ['snapshot', 'snapshot'])
            session, cfg = debug.verify_session(directory)
            scene = observed_scene(directory, session, cfg)
            self.assertEqual(scene['geometry']['bbox_xywh'], [108, 188, 81, 78])
            self.assertEqual(session['config']['cup_selection'], self.cfg['cup_selection'])
            self.assertGreater(scene['cup_envelope_radius_m'], scene['geometry']['radius_m'])

    def test_capture_failure_shows_candidates_and_never_freezes_target(self):
        cfg = deepcopy(self.cfg)
        cfg['cup_selection']['height_range_mm'] = [1, 2]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'failed'
            args, bridge, camera, config = self.capture_fixture(directory, cfg)
            with bridge as sdk, camera, config, patch.object(debug, 'show') as show:
                with self.assertRaisesRegex(ValueError, '没有候选'):
                    debug.capture(args)
                show.assert_called_once_with(directory / 'cup_candidates.png', True)
            self.assertEqual([c.args[0] for c in sdk.call_args_list], ['snapshot', 'snapshot'])
            self.assertFalse((directory / 'session.json').exists())

    def test_recapture_updates_same_run_and_invalidates_old_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / 'runs').mkdir()
            (directory / 'runs/receipt.json').write_text('old evidence')
            (directory / 'notes.txt').write_text('user notes')
            args, bridge, camera, config = self.capture_fixture(directory)
            with bridge, camera, config, patch.object(debug, 'show'):
                debug.capture(args)
                first, _ = debug.verify_session(directory)
                plan = dict(kind='flange_tcp_debug_plan', offline_only=False, blockers=[], screen_passed=True,
                            session_sha256=digest(directory / 'session.json'), created_epoch_s=100)
                debug.validate_plan(plan, directory, self.cfg, now=101)
                updated = deepcopy(self.cfg)
                updated['tcp_offset_in_link_mm'] = [0, 0, 10]
                updated['side_grasp']['close_gap_mm'] = 30
                with patch.object(debug, 'load_config', return_value=updated):
                    debug.capture(args)
                second, cfg = debug.verify_session(directory)
            self.assertNotEqual(first['capture_id'], second['capture_id'])
            self.assertAlmostEqual(np.linalg.norm(np.array(first['T_flange_tcp'])[:3, 3]
                                                 - np.array(second['T_flange_tcp'])[:3, 3]), .01)
            self.assertEqual(cfg['side_grasp']['close_gap_mm'], 30)
            with self.assertRaisesRegex(ValueError, 'Session changed'):
                debug.validate_plan(plan, directory, cfg, now=101)
            self.assertEqual((directory / 'runs/receipt.json').read_text(), 'old evidence')
            self.assertEqual((directory / 'notes.txt').read_text(), 'user notes')

    def test_failed_refresh_invalidates_old_capture_and_can_retry_same_path(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            args, bridge, camera, config = self.capture_fixture(directory)
            with bridge, camera, config, patch.object(debug, 'show'):
                debug.capture(args)
                (directory / 'rgbd/stale_frame.png').write_text('stale')
                with patch.object(debug, 'capture_rgbd', side_effect=RuntimeError('camera failed')):
                    with self.assertRaisesRegex(RuntimeError, 'camera failed'):
                        debug.capture(args)
                self.assertFalse((directory / 'session.json').exists())
                self.assertFalse((directory / 'target.png').exists())
                with self.assertRaisesRegex(ValueError, '最新采集尚未成功'):
                    debug.verify_session(directory)
                debug.capture(args)
                debug.verify_session(directory)
                self.assertFalse((directory / 'rgbd/stale_frame.png').exists())


if __name__ == '__main__':
    unittest.main()
