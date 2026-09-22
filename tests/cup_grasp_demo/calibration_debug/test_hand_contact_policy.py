"""Target-cup contact permission must retain table, arm and transit checks."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from cup_grasp_demo.calibration_debug.core import ROOT, Screen, make_plan, read_json
from cup_grasp_demo.calibration_debug.grasp import make_grasp_plan
from cup_grasp_demo.calibration_debug.parameters import hand_contact_allowed


class ContactScreenTest(unittest.TestCase):
    def setUp(self):
        self.screen = Screen.__new__(Screen)
        self.screen.arm = SimpleNamespace(joints=[])
        self.screen.meshes = {}
        self.add('right_pinky_distal_link', [.049, 0, .1])
        self.add('link7', [.1, 0, .1])
        self.scene = dict(cup_support_base_m=[0, 0, 0], cup_normal_base=[0, 0, 1],
                          cup_envelope_radius_m=.05, geometry=dict(height_m=.2))
        self.cfg = dict(table_margin_mm=10, cup_margin_mm=5)

    def add(self, name, point):
        vertex = np.array([point])
        self.screen.meshes[name] = (vertex, vertex, np.array([.001]))

    def check(self, enabled):
        return self.screen.check([[0.] * 7], self.scene, False, self.cfg, allow_hand_cup_contact=enabled)

    def test_only_target_hand_margin_becomes_a_warning(self):
        strict, contact = self.check(False), self.check(True)
        self.assertTrue(strict['blockers'])
        self.assertEqual(contact['blockers'], [])
        self.assertEqual(strict['cup_lower_bound_mm'], contact['cup_lower_bound_mm'])
        self.assertLess(contact['hand_cup_lower_bound_mm'], 0)
        self.assertGreater(contact['non_hand_cup_lower_bound_mm'], 5)
        self.assertTrue(contact['contact_warnings'])
        self.assertEqual(strict['table_min_mm'], contact['table_min_mm'])

    def test_hand_table_collision_remains_a_blocker(self):
        self.add('right_pinky_distal_link', [.049, 0, .005])
        self.assertTrue(any('table clearance' in b for b in self.check(True)['blockers']))

    def test_arm_and_adapter_still_block_even_when_hand_is_closer(self):
        for link in ('link7', 'revo2_flange'):
            with self.subTest(link=link):
                self.setUp()
                self.add(link, [.052, 0, .1])
                review = self.check(True)
                self.assertEqual(review['cup_link'], 'right_pinky_distal_link')
                self.assertTrue(any(b.startswith(link + ': cup clearance') for b in review['blockers']))

    def test_contact_option_is_explicit_boolean_and_direct_only(self):
        self.assertFalse(hand_contact_allowed({}))
        self.assertTrue(hand_contact_allowed(dict(strategy='direct_close', allow_hand_cup_contact=True)))
        for value in ('true', 1, None):
            with self.assertRaises(ValueError):
                hand_contact_allowed(dict(strategy='direct_close', allow_hand_cup_contact=value))
        with self.assertRaises(ValueError):
            hand_contact_allowed(dict(strategy='side_approach', allow_hand_cup_contact=True))


class RecordedContactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / 'cup_grasp_demo/datasets/clearance_10of11_20260918_02/source'
        cls.session = read_json(path / 'session.json')
        cls.saved = read_json(path / 'ready_plan.json')
        cls.scene = cls.saved['scene']
        cls.cfg = deepcopy(cls.session['config'])
        for name in ('home', 'calibration', 'orientation_reference', 'tcp_candidate', 'grasp_config'):
            cls.cfg[name] = str(ROOT / cls.cfg[name])
        cls.cfg['side_grasp']['allow_hand_cup_contact'] = True
        cls.start = cls.saved['start_q_rad']
        cls.plan = make_grasp_plan(cls.session, cls.start, cls.cfg, cls.scene, until='ready')

    def test_reported_minus3p6mm_plan_passes_with_unchanged_joints_and_table_margin(self):
        self.assertTrue(self.plan['screen_passed'])
        self.assertTrue(self.plan['contact_warnings'])
        self.assertEqual(len(self.plan['stages']), len(self.saved['stages']))
        for actual, original in zip(self.plan['stages'], self.saved['stages']):
            np.testing.assert_allclose(actual['target_q_rad'], original['target_q_rad'], atol=1e-9)
            self.assertAlmostEqual(actual['screen']['table_min_mm'], original['screen']['table_min_mm'], places=7)
            if actual['name'].startswith('lower_'):
                self.assertTrue(actual['screen']['hand_cup_contact_allowed'])
                self.assertGreater(actual['screen']['non_hand_cup_lower_bound_mm'], 5)
            else:
                self.assertNotIn('hand_cup_contact_allowed', actual['screen'])

    def test_disabling_option_restores_original_blockers(self):
        cfg = deepcopy(self.cfg)
        cfg['side_grasp']['allow_hand_cup_contact'] = False
        result = make_grasp_plan(self.session, self.start, cfg, self.scene, until='ready')
        self.assertEqual(result['blockers'], self.saved['blockers'])
        self.assertNotIn('allow_hand_cup_contact', result)

    def test_comparison_workflow_keeps_strict_check_even_with_active_contact_config(self):
        result = make_plan({**self.session, 'scene': self.scene}, self.start, self.cfg, 'tcp', 40, False)
        self.assertTrue(any('right_pinky_distal_link: cup clearance' in b for b in result['blockers']))

    def test_ready_resume_records_contact_warning_and_uses_same_two_step_closure(self):
        result = make_grasp_plan(self.session, self.plan['stages'][-1]['target_q_rad'], self.cfg,
                                 self.scene, 'ready', 'grip')
        self.assertEqual(result['blockers'], [])
        self.assertTrue(result['contact_warnings'])
        self.assertEqual([s['target_0_100'] for s in result['stages']],
                         [[0, 100, 0, 0, 0, 0], [100] * 6])

    def test_optional_radial_approach_uses_contact_policy(self):
        cfg = deepcopy(self.cfg)
        cfg['side_grasp'].update(close_gap_mm=30,
                                 approach=dict(enabled=True, start_gap_mm=40, step_mm=2, speed_percent=2))
        result = make_grasp_plan(self.session, self.start, cfg, self.scene, until='ready')
        self.assertTrue(result['screen_passed'])
        approach = [s for s in result['stages'] if s['state'] == 'APPROACH_CLOSE_READY']
        self.assertEqual(len(approach), 5)
        self.assertTrue(all(s['screen']['hand_cup_contact_allowed'] for s in approach))


if __name__ == '__main__':
    unittest.main()
