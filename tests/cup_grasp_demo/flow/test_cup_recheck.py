"""Reproduce the hand/cup ambiguity and guard against stale or ambiguous cups."""

from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.flow import grasp_cli
from cup_grasp_demo.flow.core import ROOT, load_config, read_json
from cup_grasp_demo.flow.cup_recheck import match_candidates, verify_cup
from cup_grasp_demo.side_grasp.preview_index import depth_candidates, depth_proposals, load_batch
from dice_cup_localization.geometry import Config


class CupRecheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / 'cup_grasp_demo/datasets/debug_regression_20260917/compare_20260917_221945'
        cls.failed = cls.directory / 'runs/20260917_222129_execute_grasp_a663e4'
        cls.session = read_json(cls.directory / 'session.json')
        cls.expected = cls.session['scene']['geometry']
        cls.cfg = load_config(ROOT / 'cup_grasp_demo/flow/index_joint_center/config.json')
        cls.cfg.pop('cup_perception', None)  # Legacy geometry matching fixture.
        cls.cfg['contact_height_fraction'] = 9 / 11  # Frozen regression section.
        cls.meta, cls.depth, cls.image, _ = load_batch(cls.failed)
        cls.proposals, cls.rejected = depth_candidates(cls.depth, cls.image, cls.meta,
                                                       Config(plane_tolerance_m=.005), 9 / 11)

    def test_old_single_candidate_failure_is_reproduced(self):
        with self.assertRaisesRegex(ValueError, 'exactly one geometric proposal, got 2'):
            depth_proposals(self.depth, self.image, self.meta, Config(plane_tolerance_m=.005), 9 / 11)

    def test_saved_failure_selects_cup_not_hand_without_changing_target(self):
        expected = deepcopy(self.expected)
        report = match_candidates(self.proposals, expected)
        self.assertTrue(report['passed'])
        chosen = report['candidates'][report['selected_index']]
        self.assertEqual(chosen['bbox_xywh'][:2], [107, 189])
        self.assertLess(chosen['errors_mm']['center_mm'], .4)
        self.assertLess(chosen['errors_mm']['height_mm'], .4)
        others = [x for x in report['candidates'] if not x['matches_frozen_cup']]
        self.assertEqual(len(others), 1)
        self.assertGreater(others[0]['errors_mm']['center_mm'], 200)
        self.assertEqual(expected, self.expected)

    def test_original_capture_reproduces_single_candidate(self):
        meta, depth, image, _ = load_batch(self.directory)
        geom, _, _ = depth_proposals(depth, image, meta, Config(plane_tolerance_m=.005), 9 / 11)
        np.testing.assert_allclose(geom['center_camera_m'], self.expected['center_camera_m'], atol=1e-12)

    def test_moved_missing_or_changed_size_cup_is_not_replaced_by_nearest(self):
        cup = self.proposals[1]
        for field in ('center_camera_m', 'height_m', 'radius_m'):
            geom = deepcopy(cup[0])
            if field == 'center_camera_m':
                geom[field][0] += .010
            else:
                geom[field] += .010
            with self.subTest(field=field):
                report = match_candidates([(geom, cup[1], cup[2])], self.expected)
                self.assertFalse(report['passed'])
                self.assertIsNone(report['selected_index'])
        self.assertFalse(match_candidates([], self.expected)['passed'])

    def test_two_matching_candidates_are_rejected_instead_of_ranked(self):
        cup = self.proposals[1]
        second = deepcopy(cup[0])
        second['center_camera_m'][0] += .001
        report = match_candidates([cup, (second, cup[1], cup[2])], self.expected)
        self.assertFalse(report['passed'])
        self.assertIn('歧义', report['reason'])
        self.assertIsNone(report['selected_index'])

    def test_failure_still_writes_matching_evidence(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch('cup_grasp_demo.flow.cup_recheck.depth_candidates', return_value=([], [])):
            path = Path(temp)
            with self.assertRaisesRegex(ValueError, '5 mm'):
                verify_cup(self.depth, self.image, self.meta, self.cfg, self.expected, path)
            self.assertFalse(read_json(path / 'cup_recheck.json')['passed'])
            self.assertTrue((path / 'cup_recheck.png').exists())

    def run_operator_gate(self, session, moved=False):
        """Run the real RGB-D preflight, substituting IO and stopping at the prompt."""
        planned = read_json(self.directory / 'pregrasp_plan.json')
        planned['session_path'] = str(self.directory)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            def capture(output, cfg):
                shutil.copytree(self.failed / 'rgbd', output)
            with patch.object(grasp_cli, 'read_json', return_value=planned), \
                 patch.object(grasp_cli.common, 'verify_session', return_value=(session, self.cfg)), \
                 patch.object(grasp_cli, 'validate'), \
                 patch.object(grasp_cli, 'observed_scene', return_value={}), \
                 patch.object(grasp_cli, 'make_grasp_plan', return_value=deepcopy(planned)), \
                 patch.object(grasp_cli.common, 'new_run', return_value=run), \
                 patch.object(grasp_cli.common, 'capture_rgbd', side_effect=capture), \
                 patch.object(grasp_cli.common, 'show'), \
                 patch.object(grasp_cli.common, 'bridge', return_value=dict(
                     joints_rad=planned['start_q_rad'], arm_status=0, motion_status=0,
                     joints_enabled=[True] * 7, ctrl_mode=1)) as sdk, \
                 patch('builtins.input', return_value='') as prompt:
                args = SimpleNamespace(plan=Path('unused'), execute=True, show=False)
                if moved:
                    with self.assertRaisesRegex(ValueError, '5 mm'):
                        grasp_cli.execute(args, confirm=input)
                    prompt.assert_not_called()
                else:
                    self.assertEqual(grasp_cli.execute(args, confirm=input), 0)
                    prompt.assert_called_once()
                self.assertEqual([call.args[0] for call in sdk.call_args_list], ['snapshot', 'snapshot'])
                self.assertEqual(read_json(run / 'cup_recheck.json')['passed'], not moved)
                self.assertFalse((run / 'request.json').exists())

    def test_operator_command_reaches_confirmation_on_original_failed_frames(self):
        self.run_operator_gate(self.session)

    def test_operator_command_stops_before_confirmation_if_frozen_cup_moved(self):
        session = deepcopy(self.session)
        session['scene']['geometry']['center_camera_m'][0] += .010
        self.run_operator_gate(session, moved=True)


if __name__ == '__main__':
    unittest.main()
