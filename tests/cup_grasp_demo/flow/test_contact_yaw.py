"""Exact XY azimuth, table height, legacy behavior, and frozen-scene planning."""
from copy import deepcopy
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.flow import debug
from cup_grasp_demo.flow.contact_geometry import contact_azimuth, contact_direction
from cup_grasp_demo.flow.core import ROOT, load_config, read_json
from cup_grasp_demo.flow.grasp import make_grasp_plan, observed_scene
from cup_grasp_demo.hand_geometry import RightRevo2Model
from nero_revo2_control.kinematics import load_model


class AzimuthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_config(ROOT / 'cup_grasp_demo/flow/index_joint_center/config.json')
        cls.rotation = np.array(load_model().fk(read_json(cfg['orientation_reference'])['joints_rad']))[:3, :3]
        link = read_json(cfg['tcp_candidate'])['link']
        cls.palm = RightRevo2Model().link_transforms_from_flange(np.eye(4))[link][:3, 0]
        cls.normal = np.array([.04, -.08, 1.])
        cls.normal /= np.linalg.norm(cls.normal)

    def direction(self, angle, side='positive'):
        return contact_direction(self.rotation, self.palm, self.normal,
                                 dict(mode='base_xy_angle_table_section', side=side, azimuth_deg=angle))

    def test_exact_xy_angle_and_constant_height_on_tilted_table(self):
        for angle in (-20, 0, 20, 90):
            with self.subTest(angle=angle):
                d, _, info = self.direction(angle)
                self.assertAlmostEqual(math.degrees(math.atan2(d[1], d[0])), angle)
                self.assertAlmostEqual(float(d @ self.normal), 0)
                self.assertAlmostEqual(np.linalg.norm(d), 1)
                self.assertEqual(info['diameter_azimuth_deg'], angle)
                self.assertEqual(info['parallel_in_base_xy'], angle == 0)
                center = np.array([.1, .5, .09])
                for point in (center + .035 * d, center + (.035 + .03) * d):
                    self.assertAlmostEqual(float(point @ self.normal), float(center @ self.normal))

    def test_hand_faces_selected_side_preserving_pitch_and_roll_relative_to_table(self):
        d, rotation, _ = self.direction(20)
        palm = rotation @ self.palm
        facing = -palm + (palm @ self.normal) * self.normal
        facing /= np.linalg.norm(facing)
        np.testing.assert_allclose(facing, d, atol=1e-12)
        self.assertAlmostEqual(float(palm @ self.normal), float((self.rotation @ self.palm) @ self.normal))
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_opposite_endpoint_preserves_same_diameter(self):
        positive, _, _ = self.direction(20)
        negative, _, info = self.direction(20, 'negative')
        np.testing.assert_allclose(positive, -negative, atol=1e-12)
        self.assertAlmostEqual(info['outward_azimuth_deg'], -160)

    def test_new_zero_angle_matches_legacy_base_x(self):
        old = contact_direction(self.rotation, self.palm, self.normal,
                                dict(mode='base_x_table_section', side='positive'))
        new = self.direction(0)
        np.testing.assert_array_equal(old[0], new[0])
        np.testing.assert_array_equal(old[1], new[1])
        self.assertNotIn('diameter_azimuth_deg', old[2])
        self.assertEqual(contact_azimuth(None), None)

    def test_invalid_angles_are_rejected_instead_of_silently_ignored(self):
        for angle in (None, True, '20', float('nan'), float('inf'), -181, 181):
            with self.subTest(angle=angle), self.assertRaisesRegex(ValueError, 'azimuth_deg'):
                self.direction(angle)
        with self.assertRaisesRegex(ValueError, 'requires'):
            contact_azimuth(dict(mode='base_x_table_section', side='positive', azimuth_deg=20))

    def test_config_validation_happens_before_any_capture(self):
        source = ROOT / 'cup_grasp_demo/flow/index_joint_center/config.json'
        cfg = read_json(source)
        cfg['contact_direction']['azimuth_deg'] = '20'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.json'
            import json
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'azimuth_deg'):
                load_config(path)


class AzimuthCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ROOT / 'cup_grasp_demo/flow/index_joint_center/config.json'
        cls.cfg = load_config(cls.config)
        cls.cfg.pop('cup_perception', None)  # Keep the original geometric azimuth regression.
        cls.cfg['contact_height_fraction'] = 9 / 11  # Preserve the original azimuth regression.
        cls.cfg['side_grasp'].update(close_gap_mm=40, allow_hand_cup_contact=False,
                                     approach={'enabled': False})
        cls.fixture = ROOT / 'cup_grasp_demo/datasets/contact_yaw20_20260917_01/current'
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        directory = Path(cls.temp.name) / 'capture'
        args = SimpleNamespace(config=cls.config, session=directory, replay=None, show=False)
        def camera(output, _cfg):
            shutil.copytree(cls.fixture / 'rgbd', output)
        with patch.object(debug, 'load_config', return_value=cls.cfg), \
             patch.object(debug, 'capture_rgbd', side_effect=camera), \
             patch.object(debug, 'bridge', return_value=read_json(cls.fixture / 'after.json')) as sdk:
            debug.capture(args)
            assert [call.args[0] for call in sdk.call_args_list] == ['snapshot', 'snapshot']
        cls.session, cls.frozen_cfg = debug.verify_session(directory)
        cls.scene = observed_scene(directory, cls.session, cls.frozen_cfg)
        cls.start = read_json(cls.fixture / 'ready_plan.json')['start_q_rad']
        cls.plan = make_grasp_plan(cls.session, cls.start, cls.frozen_cfg, cls.scene, until='ready')

    def test_live_profile_and_frozen_capture_use_plus20_without_tcp_or_height_change(self):
        self.assertEqual(self.session['config']['contact_direction']['azimuth_deg'], 20)
        d = np.array(self.scene['outward_base'])
        self.assertAlmostEqual(math.degrees(math.atan2(d[1], d[0])), 20)
        support = np.array(self.scene['cup_support_base_m'])
        normal = np.array(self.scene['cup_normal_base'])
        contact = np.array(self.session['contact_base_m'])
        self.assertAlmostEqual(float((contact - support) @ normal), self.scene['geometry']['height_m'] * 9 / 11)
        np.testing.assert_array_equal(self.session['T_flange_tcp'], read_json(self.fixture / 'session.json')['T_flange_tcp'])

    def test_plus20_40mm_passes_full_path_without_table_clearance_regression(self):
        self.assertTrue(self.plan['screen_passed'])
        self.assertEqual(self.plan['blockers'], [])
        self.assertTrue(all(stage['kind'] == 'arm' for stage in self.plan['stages']))
        self.assertGreater(min(stage['screen']['table_min_mm'] for stage in self.plan['stages']), 20)
        self.assertGreater(min(stage['screen']['cup_lower_bound_mm'] for stage in self.plan['stages']), 9)

    def test_smaller_gap_remains_blocked_and_never_adds_hand_commands(self):
        cfg = deepcopy(self.frozen_cfg)
        cfg['side_grasp']['close_gap_mm'] = 30
        plan = make_grasp_plan(self.session, self.start, cfg, self.scene, until='grip')
        self.assertFalse(plan['screen_passed'])
        self.assertTrue(any('cup clearance' in b for b in plan['blockers']))
        self.assertTrue(all(stage['kind'] == 'arm' for stage in plan['stages']))


if __name__ == '__main__':
    unittest.main()
