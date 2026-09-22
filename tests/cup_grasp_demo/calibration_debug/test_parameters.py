"""Configurable targets, optional approach and unchanged legacy defaults."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.calibration_debug import debug
from cup_grasp_demo.calibration_debug.core import ROOT, configured_tcp, load_config, read_json
from cup_grasp_demo.calibration_debug.grasp import make_grasp_plan, observed_scene, options
from cup_grasp_demo.calibration_debug.grasp_execution import send_closure, validate_sequence
from cup_grasp_demo.calibration_debug.parameters import approach_options, closure_targets
from cup_grasp_demo.hand_geometry import RightRevo2Model


class ParametersTest(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / 'cup_grasp_demo/calibration_debug/index_joint_center/config.json'
        self.cfg = load_config(self.path)

    def test_zero_offset_preserves_candidate_and_local_z_moves_toward_tip(self):
        original = np.array(read_json(self.cfg['tcp_candidate'])['T_flange_contact_candidate'])
        np.testing.assert_array_equal(configured_tcp(self.cfg), original)
        self.cfg['tcp_offset_in_link_mm'] = [0, 0, 10]
        shifted = configured_tcp(self.cfg)
        link = RightRevo2Model().link_transforms_from_flange(np.eye(4))['right_index_proximal_link']
        np.testing.assert_allclose(shifted[:3, 3] - original[:3, 3], .01 * link[:3, 2], atol=1e-12)
        np.testing.assert_array_equal(shifted[:3, :3], original[:3, :3])

    def test_config_check_never_reads_or_moves_hardware(self):
        with patch.object(debug, 'bridge') as sdk, patch.object(debug, 'capture_rgbd') as camera:
            self.assertEqual(debug.config_check(SimpleNamespace(config=self.path)), 0)
        sdk.assert_not_called()
        camera.assert_not_called()

    def test_invalid_new_parameters_are_rejected_during_config_load(self):
        raw = read_json(self.path)
        variants = []
        for value in ([0, 0], [0, True, 0], [0, 0, float('nan')], [0, 0, 101]):
            cfg = deepcopy(raw)
            cfg['tcp_offset_in_link_mm'] = value
            variants.append(cfg)
        for value in (True, '90', 101, -1, 80.5):
            cfg = deepcopy(raw)
            cfg['side_grasp']['thumb_base_target_0_100'] = value
            variants.append(cfg)
        for approach in (dict(enabled='yes'),
                         dict(enabled=True, start_gap_mm=raw['side_grasp']['close_gap_mm']),
                         dict(speed_percent=1.5), dict(step_mm=0), dict(step_m=2)):
            cfg = deepcopy(raw)
            cfg['side_grasp']['approach'] = approach
            variants.append(cfg)
        with tempfile.TemporaryDirectory() as temp:
            import json
            for i, cfg in enumerate(variants):
                path = Path(temp) / f'{i}.json'
                path.write_text(json.dumps(cfg))
                with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                    load_config(path)

    def test_closure_keeps_thumb_base_first_and_at_same_target(self):
        opts = dict(thumb_base_target_0_100=80, close_targets_0_100=[95, 80, 90, 85, 90, 95])
        self.assertEqual(closure_targets(opts), [[0, 80, 0, 0, 0, 0], [95, 80, 90, 85, 90, 95]])
        opts['close_targets_0_100'][1] = 100
        with self.assertRaisesRegex(ValueError, 'retain'):
            closure_targets(opts)
        with self.assertRaisesRegex(ValueError, 'six'):
            closure_targets(dict(close_targets_0_100=[100] * 5))

    def test_custom_targets_reach_sdk_arguments(self):
        cfg = dict(side_grasp=dict(thumb_base_target_0_100=80,
                                  close_targets_0_100=[95, 80, 90, 85, 90, 95]))
        targets = closure_targets(cfg['side_grasp'])
        stages = [dict(kind='hand', state=state, target_0_100=target, duration_s=.8, settle_s=.2)
                  for state, target in zip(('THUMB_BASE', 'CLOSE_FINGERS'), targets)]
        plan = dict(until_state='grip', stages=stages)
        validate_sequence(plan, cfg)
        with self.assertRaises(ValueError):
            validate_sequence(plan)  # Legacy config must reject a different recipe.
        sent, clock = [], [0.]
        hand = SimpleNamespace(position_time_ctrl=lambda **kw: sent.append(kw), get_finger_pos=lambda: None)
        demo = SimpleNamespace(FINGER_NAMES=('thumb_tip', 'thumb_base', 'index', 'middle', 'ring', 'pinky'),
                               feedback_stamp=lambda _: None)
        def sleep(delta):
            clock[0] += delta
        with patch('cup_grasp_demo.calibration_debug.grasp_execution.fresh_current', return_value={}):
            for stage in stages:
                send_closure(hand, demo, stage, lambda: None, monotonic=lambda: clock[0],
                             wallclock=lambda: 1000 + clock[0], sleep=sleep)
        self.assertEqual([list(cmd.values())[1:] for cmd in sent[::2]], targets)
        self.assertEqual(sent[1], dict(mode='time', **dict.fromkeys(demo.FINGER_NAMES, 80)))

    def test_old_pregrasp_distance_is_configurable_with_legacy_defaults(self):
        opts = dict(strategy='side_approach', pregrasp_gap_mm=40, approach_step_mm=2,
                    approach_speed_percent=2, finger_duration_s=1, finger_settle_s=.3,
                    cup_radius_percentile=98, cup_model_padding_mm=3)
        self.assertEqual(options(dict(side_grasp=opts))['pregrasp_gap_mm'], 40)
        self.assertFalse(approach_options(opts)['enabled'])
        self.assertEqual(closure_targets(opts), [[0, 100, 0, 0, 0, 0], [100] * 6])


class ApproachPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / 'cup_grasp_demo/datasets/contact_yaw20_20260917_01/current'
        cls.session = read_json(cls.directory / 'session.json')
        cls.cfg = deepcopy(cls.session['config'])
        for key in ('home', 'calibration', 'tcp_candidate', 'orientation_reference', 'grasp_config'):
            cls.cfg[key] = str(ROOT / cls.cfg[key])
        cls.scene = observed_scene(cls.directory, cls.session, cls.cfg)
        cls.home = cls.session['captured_q_rad']
        cls.custom = deepcopy(cls.cfg)
        cls.custom['side_grasp'].update(close_gap_mm=30, thumb_base_target_0_100=80,
                                       close_targets_0_100=[95, 80, 90, 85, 90, 95],
                                       approach=dict(enabled=True, start_gap_mm=40, step_mm=2, speed_percent=2))
        cls.clear = deepcopy(cls.scene)
        cls.clear['cup_envelope_radius_m'] = .03  # Synthetic test only, never a hardware scene.
        cls.plan = make_grasp_plan(cls.session, cls.home, cls.custom, cls.clear, until='grip')

    def test_optional_approach_preserves_height_orientation_and_configured_steps(self):
        self.assertFalse(self.plan['blockers'])
        validate_sequence(self.plan, self.custom)
        approach = [s for s in self.plan['stages'] if s['state'] == 'APPROACH_CLOSE_READY']
        self.assertEqual(len(approach), 5)
        self.assertTrue(all(s['speed_percent'] == 2 for s in approach))
        outward = np.array(self.plan['outward_base'])
        contact = np.array(self.plan['contact_base_m'])
        normal = np.array(self.scene['cup_normal_base'])
        for i, stage in enumerate(approach):
            delta = np.array(stage['tcp_target_base_m']) - contact
            np.testing.assert_allclose(delta, (.038 - i * .002) * outward, atol=1e-12)
            self.assertAlmostEqual(float(delta @ normal), 0)
            np.testing.assert_allclose(np.array(stage['T_base_flange'])[:3, :3],
                                       self.session['R_base_flange'], atol=1e-12)
        self.assertEqual(approach[-1]['state_completed'], 'READY')
        hands = [s['target_0_100'] for s in self.plan['stages'] if s['kind'] == 'hand']
        self.assertEqual(hands, closure_targets(self.custom['side_grasp']))

    def test_real_cup_collision_still_blocks_slow_approach_and_closure(self):
        plan = make_grasp_plan(self.session, self.home, self.custom, self.scene, until='grip')
        self.assertFalse(plan['screen_passed'])
        self.assertTrue(any('approach/' in b and 'cup clearance' in b for b in plan['blockers']))
        self.assertTrue(all(s['kind'] == 'arm' for s in plan['stages']))

    def test_disabled_approach_is_identical_to_original_direct_plan(self):
        first = make_grasp_plan(self.session, self.home, self.cfg, self.scene, until='ready')
        cfg = deepcopy(self.cfg)
        cfg['side_grasp'].update(approach=dict(enabled=False, start_gap_mm=90, step_mm=1, speed_percent=1),
                                 thumb_base_target_0_100=100, close_targets_0_100=[100] * 6)
        second = make_grasp_plan(self.session, self.home, cfg, self.scene, until='ready')
        self.assertEqual(first, second)
        self.assertNotIn('APPROACH_CLOSE_READY', [s['state'] for s in second['stages']])
