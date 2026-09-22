"""Open-finger TCP offset, direct closing states, and recorded collision rejection."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.calibration_debug import grasp_cli
from cup_grasp_demo.calibration_debug.core import ROOT, load_config, read_json
from cup_grasp_demo.calibration_debug.grasp import FINGER_TARGETS, make_grasp_plan, observed_scene, options
from cup_grasp_demo.calibration_debug.grasp_execution import execute, validate_sequence
from cup_grasp_demo.hand_geometry import RightRevo2Model
from nero_revo2_control.kinematics import load_model


class DirectGraspTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / 'cup_grasp_demo/calibration_debug/index_joint_center/config.json')
        cls.cfg.pop('cup_perception', None)  # Legacy geometry fixture.
        # Preserve the original 30 mm regression independently of the active profile.
        cls.cfg['side_grasp']['close_gap_mm'] = 30.0
        cls.cfg['side_grasp']['approach'] = {'enabled': False}
        cls.cfg['side_grasp']['allow_hand_cup_contact'] = False  # Frozen strict-check regression.
        cls.directory = ROOT / 'cup_grasp_demo/datasets/debug_regression_20260917/compare_20260917_221945'
        cls.session = read_json(cls.directory / 'session.json')
        cls.cfg['contact_height_fraction'] = cls.session['config']['contact_height_fraction']
        cls.candidate = read_json(cls.cfg['tcp_candidate'])
        cls.session['T_flange_tcp'] = cls.candidate['T_flange_contact_candidate']
        cls.home = read_json(cls.cfg['home'])['joints_rad']
        cls.scene = observed_scene(cls.directory, cls.session, cls.cfg)
        cls.recorded = make_grasp_plan(cls.session, cls.home, cls.cfg, cls.scene)
        # Synthetic narrow obstacle tests the successful state path, independently
        # of the actual recorded cup, which must remain blocked by collision.
        cls.synthetic = deepcopy(cls.scene)
        cls.synthetic['cup_envelope_radius_m'] = .040
        cls.clear = make_grasp_plan(cls.session, cls.home, cls.cfg, cls.synthetic)

    def test_tcp_is_fifty_mm_along_open_index_not_flange_x(self):
        link = RightRevo2Model().link_transforms_from_flange(np.eye(4))['right_index_proximal_link']
        tcp = np.array(self.candidate['T_flange_contact_candidate'])
        delta = tcp[:3, 3] - link[:3, 3]
        np.testing.assert_allclose(delta, link[:3, 2] * .05, atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(delta), .05)
        self.assertGreater(abs(delta[1]), .008)
        np.testing.assert_allclose(tcp @ self.candidate['T_contact_flange_candidate'], np.eye(4), atol=1e-12)
        self.assertFalse(self.candidate['user_reported_point_alignment_acceptable'])

    def test_legacy_pregrasp_plan_matches_the_saved_implementation(self):
        session = read_json(self.directory / 'session.json')
        cfg = deepcopy(session['config'])
        for key in ('home', 'calibration', 'orientation_reference', 'tcp_candidate', 'grasp_config'):
            cfg[key] = str(ROOT / cfg[key])
        saved = read_json(self.directory / 'pregrasp_plan.json')
        scene = observed_scene(self.directory, session, cfg)
        rebuilt = make_grasp_plan(session, saved['start_q_rad'], cfg, scene, 'home', 'pregrasp')
        def compare(actual, expected):
            if isinstance(actual, dict):
                self.assertEqual(set(actual), set(expected))
                for key in actual:
                    compare(actual[key], expected[key])
            elif isinstance(actual, (list, tuple)):
                self.assertEqual(len(actual), len(expected))
                for a, b in zip(actual, expected):
                    compare(a, b)
            elif isinstance(actual, float):
                # PC/K3 BLAS can differ in the final floating-point digits.
                self.assertAlmostEqual(actual, expected, delta=1e-9)
            else:
                self.assertEqual(actual, expected)
        compare(rebuilt, {key: saved[key] for key in rebuilt})

    def test_recorded_target_keeps_exact_gap_and_rejects_cup_overlap(self):
        plan = self.recorded
        delta = np.array(plan['close_ready_base_m']) - plan['contact_base_m']
        np.testing.assert_allclose(delta, .03 * np.array(plan['outward_base']), atol=1e-12)
        self.assertAlmostEqual(float(delta @ self.scene['cup_normal_base']), 0)
        self.assertAlmostEqual(delta[1], 0)
        self.assertEqual(plan['closing_gap_mm'], 30)
        self.assertFalse(plan['screen_passed'])
        self.assertTrue(any('cup clearance' in x for x in plan['blockers']))
        self.assertTrue(all(x['kind'] == 'arm' for x in plan['stages']))
        self.assertTrue(all(x['state'] == 'TO_CLOSE_READY' for x in plan['stages']))
        tcp = np.array(self.session['T_flange_tcp'])
        pose = np.array(load_model().fk(plan['stages'][-1]['target_q_rad']))
        np.testing.assert_allclose((pose @ tcp)[:3, 3], plan['close_ready_base_m'], atol=1e-5)

    def test_clear_synthetic_scene_closes_without_an_extra_approach(self):
        plan = self.clear
        self.assertEqual(plan['blockers'], [])
        validate_sequence(plan)
        self.assertNotIn('pregrasp_base_m', plan)
        states = list(dict.fromkeys(s['state'] for s in plan['stages']))
        self.assertEqual(states, ['TO_CLOSE_READY', 'THUMB_BASE', 'CLOSE_FINGERS'])
        hands = [s for s in plan['stages'] if s['kind'] == 'hand']
        self.assertEqual([s['target_0_100'] for s in hands], FINGER_TARGETS)
        for hand in hands:
            self.assertEqual(hand['current_q_rad'], plan['stages'][-3]['target_q_rad'])
        self.assertEqual(plan['stages'][-3]['state_completed'], 'READY')

    def test_resume_closes_only_from_the_thirty_mm_ready_pose(self):
        ready = make_grasp_plan(self.session, self.home, self.cfg, self.synthetic, until='ready')
        q = ready['stages'][-1]['target_q_rad']
        self.assertTrue(all(s['kind'] == 'arm' for s in ready['stages']))
        close = make_grasp_plan(self.session, q, self.cfg, self.synthetic, 'ready', 'grip')
        self.assertEqual(close['blockers'], [])
        self.assertEqual([s['kind'] for s in close['stages']], ['hand', 'hand'])
        with self.assertRaisesRegex(ValueError, '阶段起点'):
            make_grasp_plan(self.session, self.home, self.cfg, self.synthetic, 'ready', 'grip')
        blocked = make_grasp_plan(self.session, q, self.cfg, self.scene, 'ready', 'grip')
        self.assertTrue(blocked['blockers'])
        self.assertEqual(blocked['stages'], [])

    def test_invalid_options_and_legacy_states_are_not_silently_reinterpreted(self):
        for value in (0, float('nan'), 51):
            cfg = deepcopy(self.cfg)
            cfg['side_grasp']['close_gap_mm'] = value
            with self.assertRaisesRegex(ValueError, 'close_gap_mm'):
                options(cfg)
        for start, until in [('home', 'pregrasp'), ('home', 'contact'), ('pregrasp', 'grip')]:
            with self.assertRaisesRegex(ValueError, 'direct_close'):
                make_grasp_plan(self.session, self.home, self.cfg, self.scene, start, until)
        wrong = deepcopy(self.clear)
        wrong['stages'][0]['state'] = 'APPROACH'
        with self.assertRaisesRegex(ValueError, 'state order'):
            validate_sequence(wrong)

    def test_failed_motion_never_starts_closure(self):
        demo = SimpleNamespace(require_right_hand=lambda *_: None)
        def fail(*_):
            raise TimeoutError('incomplete joint target')
        with patch('cup_grasp_demo.calibration_debug.grasp_execution.fresh_current'), \
             patch('cup_grasp_demo.calibration_debug.grasp_execution.send_closure') as close:
            with self.assertRaisesRegex(TimeoutError, 'incomplete'):
                execute(self.clear, self.cfg, object(), object(), demo, fail, dict(stages=[]))
            close.assert_not_called()

    def test_preview_does_not_open_camera_or_robot(self):
        plan = {**self.recorded, 'session_path': str(self.directory)}
        with patch.object(grasp_cli, 'read_json', return_value=plan), \
             patch.object(grasp_cli.common, 'verify_session', return_value=(self.session, self.cfg)), \
             patch.object(grasp_cli, 'validate'), \
             patch.object(grasp_cli.common, 'bridge') as sdk, \
             patch.object(grasp_cli.common, 'capture_rgbd') as camera:
            grasp_cli.execute(SimpleNamespace(plan=Path('unused'), execute=False))
            sdk.assert_not_called()
            camera.assert_not_called()


class CurrentClosingGapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / 'cup_grasp_demo/datasets/closing_gap_20260917_01/current'
        cls.cfg = load_config(ROOT / 'cup_grasp_demo/calibration_debug/index_joint_center/config.json')
        cls.cfg.pop('cup_perception', None)  # Legacy geometry fixture.
        # Freeze the original 40 mm policy independently of operator tuning.
        cls.cfg['side_grasp'].update(close_gap_mm=40, allow_hand_cup_contact=False,
                                     approach={'enabled': False})
        cls.session = read_json(cls.directory / 'session.json')
        cls.cfg['contact_height_fraction'] = cls.session['config']['contact_height_fraction']
        cls.scene = observed_scene(cls.directory, cls.session, cls.cfg)
        cls.start = read_json(cls.directory / 'ready_plan.json')['start_q_rad']
        cls.plan = make_grasp_plan(cls.session, cls.start, cls.cfg, cls.scene, until='ready')

    def test_recorded_40mm_profile_clears_cup_without_relaxing_margins(self):
        self.assertEqual(self.cfg['side_grasp']['close_gap_mm'], 40)
        self.assertEqual(self.cfg['cup_margin_mm'], 5)
        self.assertEqual(self.cfg['table_margin_mm'], 10)
        self.assertTrue(self.plan['screen_passed'])
        self.assertEqual(self.plan['blockers'], [])
        self.assertTrue(all(s['kind'] == 'arm' for s in self.plan['stages']))
        validate_sequence(self.plan)
        offset = np.array(self.plan['close_ready_base_m']) - self.plan['contact_base_m']
        np.testing.assert_allclose(offset, .04 * np.array(self.plan['outward_base']), atol=1e-12)
        self.assertGreater(min(s['screen']['cup_lower_bound_mm'] for s in self.plan['stages']), 7)
        self.assertGreater(min(s['screen']['table_min_mm'] for s in self.plan['stages']), 21)

    def test_previous_35mm_advice_is_still_rejected_for_latest_capture(self):
        cfg = deepcopy(self.cfg)
        cfg['side_grasp']['close_gap_mm'] = 35
        plan = make_grasp_plan(self.session, self.start, cfg, self.scene, until='grip')
        self.assertFalse(plan['screen_passed'])
        self.assertTrue(any('cup clearance' in b for b in plan['blockers']))
        self.assertTrue(all(s['kind'] == 'arm' for s in plan['stages']))


if __name__ == '__main__':
    unittest.main()
