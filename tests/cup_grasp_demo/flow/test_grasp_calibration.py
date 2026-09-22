"""Recorded-scene, stage-transition and timed finger-command regressions."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.flow.core import ROOT, read_json
from cup_grasp_demo.flow.contact_geometry import contact_direction
from cup_grasp_demo.flow.grasp import (
    FINGER_TARGETS, GraspScreen, make_grasp_plan, observed_scene, options, target_points,
)
from cup_grasp_demo.flow import grasp_cli
from cup_grasp_demo.flow.grasp_execution import execute, send_closure, validate_sequence
from nero_revo2_control.kinematics import load_model
from cup_grasp_demo.hand_geometry import RightRevo2Model


def legacy_config():
    # Preserve the last successful 50 mm -> contact workflow independently of
    # subsequent changes to the operator's active profile and TCP.
    saved = read_json(ROOT / 'cup_grasp_demo/datasets/debug_regression_20260917/compare_20260917_221945/session.json')
    cfg = deepcopy(saved['config'])
    for key in ('home', 'calibration', 'orientation_reference', 'tcp_candidate', 'grasp_config'):
        cfg[key] = str(ROOT / cfg[key])
    return cfg, saved['T_flange_tcp']


class ContactDirectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / 'cup_grasp_demo/datasets/compare_20260917_211358'
        cls.cfg, frozen_tcp = legacy_config()
        cls.original = read_json(cls.directory / 'session.json')
        candidate = read_json(cls.cfg['tcp_candidate'])
        cls.palm = RightRevo2Model().link_transforms_from_flange(np.eye(4))[candidate['link']][:3, 0]
        cls.rotation = np.array(cls.original['R_base_flange'])
        cls.normal = np.array(cls.original['scene']['cup_normal_base'])
        cls.direction, cls.turned, cls.info = contact_direction(
            cls.rotation, cls.palm, cls.normal, cls.cfg['contact_direction'])
        cls.session = deepcopy(cls.original)
        cls.session['T_flange_tcp'] = frozen_tcp
        cls.session['R_base_flange'] = cls.turned.tolist()
        scene = cls.session['scene']
        center = np.array(scene['cup_support_base_m']) + scene['geometry']['contact_height_m'] * cls.normal
        cls.session['contact_base_m'] = (center + scene['geometry']['radius_m'] * cls.direction).tolist()
        scene['outward_base'] = cls.direction.tolist()
        scene['contact_direction'] = cls.info
        cls.scene = observed_scene(cls.directory, cls.session, cls.cfg)
        cls.plan = make_grasp_plan(cls.session, read_json(cls.cfg['home'])['joints_rad'], cls.cfg, cls.scene)

    def test_without_constraint_preserves_previous_contact_and_orientation(self):
        direction, rotation, info = contact_direction(self.rotation, self.palm, self.normal)
        np.testing.assert_allclose(direction, self.original['scene']['outward_base'], atol=1e-10)
        np.testing.assert_array_equal(rotation, self.rotation)
        self.assertIsNone(info)

    def test_x_diameter_preserves_height_on_tilted_table(self):
        contact, pre, direction, tangent = target_points(self.session)
        support = np.array(self.scene['cup_support_base_m'])
        height = self.scene['geometry']['contact_height_m']
        center = support + height * self.normal
        for point in (contact, pre):
            self.assertAlmostEqual(float((point - support) @ self.normal), height)
            self.assertAlmostEqual(float(point[1]), float(center[1]))
        self.assertGreater(float(direction[0]), 0)
        self.assertAlmostEqual(float(direction @ tangent), 0)
        self.assertAlmostEqual(float(np.linalg.norm(pre - contact)), .05)
        self.assertGreater(self.info['base_x_3d_deviation_deg'], 0)

    def test_hand_turns_with_contact_side_and_preserves_tilt(self):
        palm_before = self.rotation @ self.palm
        palm_after = self.turned @ self.palm
        outward = -palm_after + (palm_after @ self.normal) * self.normal
        outward /= np.linalg.norm(outward)
        np.testing.assert_allclose(outward, self.direction, atol=1e-12)
        self.assertAlmostEqual(float(palm_before @ self.normal), float(palm_after @ self.normal))
        np.testing.assert_allclose(self.turned.T @ self.turned, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(self.turned)), 1)

    def test_flat_table_and_opposite_side(self):
        for side, sign in [('positive', 1), ('negative', -1)]:
            direction, _, _ = contact_direction(self.rotation, self.palm, [0, 0, 1],
                                                dict(mode='base_x_table_section', side=side))
            np.testing.assert_allclose(direction, [sign, 0, 0])

    def test_invalid_constraint_does_not_fall_back_to_previous_direction(self):
        for constraint in ({'mode': 'camera_x', 'side': 'positive'},
                           {'mode': 'base_x_table_section', 'side': 'unknown'}, 'base_x'):
            with self.subTest(constraint=constraint), self.assertRaisesRegex(ValueError, 'contact_direction'):
                contact_direction(self.rotation, self.palm, self.normal, constraint)

    def test_x_contact_reaches_pregrasp_and_contact_with_unchanged_closing(self):
        self.assertEqual(self.plan['blockers'], [])
        validate_sequence(self.plan)
        self.assertEqual([x['target_0_100'] for x in self.plan['stages'] if x['kind'] == 'hand'], FINGER_TARGETS)
        arm, tcp = load_model(), np.array(self.session['T_flange_tcp'])
        for state, key in [('PREGRASP', 'pregrasp_base_m'), ('CONTACT', 'contact_base_m')]:
            stage = next(x for x in self.plan['stages'] if x.get('state_completed') == state)
            actual = np.array(arm.fk(stage['target_q_rad']))
            np.testing.assert_allclose((actual @ tcp)[:3, 3], self.plan[key], atol=1e-5)
            np.testing.assert_allclose(actual[:3, :3], self.turned, atol=1e-4)


class PlanningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / 'cup_grasp_demo/datasets/compare_20260917_211358'
        cls.cfg, frozen_tcp = legacy_config()
        cls.session = read_json(cls.directory / 'session.json')
        cls.session['T_flange_tcp'] = frozen_tcp
        cls.home = read_json(cls.cfg['home'])['joints_rad']
        cls.scene = observed_scene(cls.directory, cls.session, cls.cfg)
        cls.plan = make_grasp_plan(cls.session, cls.home, cls.cfg, cls.scene)

    def test_new_home_and_radial_pregrasp_on_tilted_table(self):
        np.testing.assert_allclose(np.degrees(self.home), [0, -70, -90, 100, -10, -5, 5])
        contact, pre, outward, tangent = target_points(self.session)
        self.assertAlmostEqual(np.linalg.norm(pre - contact), .05)
        self.assertAlmostEqual(float(outward @ tangent), 0)
        self.assertAlmostEqual(float(outward @ self.scene['cup_normal_base']), 0)
        np.testing.assert_allclose(pre - contact, .05 * outward)

    def test_missing_profile_and_fractional_speed_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '抓杯配置'):
            options({})
        cfg = deepcopy(self.cfg)
        cfg['side_grasp']['approach_speed_percent'] = 3.0
        self.assertIsInstance(options(cfg)['approach_speed_percent'], int)
        cfg['side_grasp']['approach_speed_percent'] = 3.5
        with self.assertRaisesRegex(ValueError, 'integer'):
            options(cfg)

    def test_recorded_full_plan_reaches_pregrasp_then_contact_then_closes(self):
        self.assertEqual(self.plan['blockers'], [])
        validate_sequence(self.plan)
        stages = self.plan['stages']
        self.assertEqual([x['target_0_100'] for x in stages if x['kind'] == 'hand'], FINGER_TARGETS)
        pre = next(x for x in stages if x.get('state_completed') == 'PREGRASP')
        reached = next(x for x in stages if x.get('state_completed') == 'CONTACT')
        arm, tcp = load_model(), np.array(self.session['T_flange_tcp'])
        for stage, point in [(pre, self.plan['pregrasp_base_m']), (reached, self.plan['contact_base_m'])]:
            np.testing.assert_allclose((np.array(arm.fk(stage['target_q_rad'])) @ tcp)[:3, 3], point, atol=1e-5)
        self.assertEqual(len([x for x in stages if x['state'] == 'APPROACH']), 10)
        self.assertFalse(self.plan['physical_grip_verified'])

    def test_resume_requires_correct_start_and_preserves_stage_boundary(self):
        q = next(x['target_q_rad'] for x in self.plan['stages'] if x.get('state_completed') == 'PREGRASP')
        plan = make_grasp_plan(self.session, q, self.cfg, self.scene, 'pregrasp', 'contact')
        self.assertEqual(plan['blockers'], [])
        self.assertEqual({x['state'] for x in plan['stages']}, {'APPROACH'})
        q = plan['stages'][-1]['target_q_rad']
        close = make_grasp_plan(self.session, q, self.cfg, self.scene, 'contact', 'grip')
        self.assertEqual([x['kind'] for x in close['stages']], ['hand', 'hand'])
        with self.assertRaisesRegex(ValueError, '阶段起点'):
            make_grasp_plan(self.session, self.home, self.cfg, self.scene, 'pregrasp', 'contact')

    def test_intentional_hand_contact_does_not_disable_table_or_wrist_checks(self):
        q = next(x['target_q_rad'] for x in self.plan['stages'] if x.get('state_completed') == 'CONTACT')
        scene = deepcopy(self.scene)
        screen = GraspScreen()
        scene['cup_support_base_m'] = (np.array(scene['cup_support_base_m']) + .1 * np.array(scene['cup_normal_base'])).tolist()
        self.assertTrue(any('table clearance' in x for x in screen.approach([q], scene, self.cfg)['blockers']))
        scene = deepcopy(self.scene)
        scene['cup_envelope_radius_m'] = .3
        self.assertTrue(any('cup clearance' in x for x in screen.approach([q], scene, self.cfg)['blockers']))

    def test_replay_grasp_plan_cannot_execute(self):
        with self.assertRaisesRegex(ValueError, 'live side-grasp'):
            grasp_cli.validate({**self.plan, 'offline_only': True}, self.directory, self.cfg)

    def test_preview_has_no_camera_or_robot_calls(self):
        plan = {**self.plan, 'session_path': str(self.directory)}
        with patch.object(grasp_cli, 'read_json', return_value=plan), \
             patch.object(grasp_cli.common, 'verify_session', return_value=(self.session, self.cfg)), \
             patch.object(grasp_cli, 'validate'), \
             patch.object(grasp_cli.common, 'bridge') as sdk, \
             patch.object(grasp_cli.common, 'capture_rgbd') as camera:
            grasp_cli.execute(SimpleNamespace(plan=Path('unused'), execute=False))
            sdk.assert_not_called()
            camera.assert_not_called()


class FingerTest(unittest.TestCase):
    def setUp(self):
        self.clock = 0.
        self.sent = []
        self.names = ('thumb_tip', 'thumb_base', 'index_finger', 'middle_finger', 'ring_finger', 'pinky_finger')
        self.hand = SimpleNamespace(position_time_ctrl=lambda **kw: self.sent.append(kw),
                                    get_finger_pos=lambda: None,
                                    get_finger_current=lambda: SimpleNamespace(timestamp=1000 + self.clock))
        self.demo = SimpleNamespace(FINGER_NAMES=self.names,
                                    read_fresh=lambda getter, *_: getter(),
                                    feedback_stamp=lambda x: getattr(x, 'timestamp', None),
                                    finger_values=lambda _: dict.fromkeys(self.names, 0))
        self.stage = dict(state='THUMB_BASE', target_0_100=FINGER_TARGETS[0], duration_s=1., settle_s=.3)

    def sleep(self, value):
        self.clock += value

    def test_timed_closure_without_position_feedback_is_not_reported_verified(self):
        result = send_closure(self.hand, self.demo, self.stage, lambda: None,
                              monotonic=lambda: self.clock, wallclock=lambda: 1000 + self.clock, sleep=self.sleep)
        self.assertEqual(self.sent[0], dict(mode='pos', **dict(zip(self.names, FINGER_TARGETS[0]))))
        self.assertEqual(self.sent[1], dict(mode='time', **dict.fromkeys(self.names, 100)))
        self.assertGreaterEqual(self.clock, 1.3)
        self.assertIsNone(result['position_target_reached'])
        self.assertFalse(result['physical_grip_verified'])

    def test_maximum_speed_sends_zero_time_and_retains_observation_window(self):
        result = send_closure(self.hand, self.demo, self.stage, lambda: None,
                              monotonic=lambda: self.clock, wallclock=lambda: 1000+self.clock,
                              sleep=self.sleep, maximum_speed=True, read_feedback=False)
        self.assertEqual(self.sent[1], dict(mode='time', **dict.fromkeys(self.names, 0)))
        self.assertEqual(result['speed_mode'], 'max')
        self.assertGreaterEqual(self.clock, 1.3)
        self.assertIsNone(result['position_target_reached'])

    def test_arm_movement_during_closing_aborts(self):
        with self.assertRaisesRegex(RuntimeError, 'moved'):
            send_closure(self.hand, self.demo, self.stage,
                         lambda: (_ for _ in ()).throw(RuntimeError('arm moved')),
                         monotonic=lambda: self.clock, wallclock=lambda: 1000 + self.clock, sleep=self.sleep)
        self.assertEqual(len(self.sent), 2)

    def test_sequence_failure_stops_before_second_hand_command(self):
        plan = dict(until_state='grip', stages=[
            dict(kind='hand', state=state, name=state, target_0_100=target,
                 duration_s=1., settle_s=.3, current_q_rad=[0] * 7)
            for state, target in zip(('THUMB_BASE', 'CLOSE_FINGERS'), FINGER_TARGETS)])
        self.demo.require_right_hand = lambda *_: None
        self.demo.arm_snapshot = lambda _: ([0] * 7, None, SimpleNamespace(arm_status=0, motion_status=0, ctrl_mode=1))
        robot = SimpleNamespace(get_joints_enable_status_list=lambda: [True] * 7)
        with patch('cup_grasp_demo.flow.grasp_execution.fresh_current'), \
             patch('cup_grasp_demo.flow.grasp_execution.send_closure', side_effect=RuntimeError('thumb failed')) as close:
            with self.assertRaisesRegex(RuntimeError, 'thumb failed'):
                execute(plan, {}, robot, self.hand, self.demo, lambda *_: None, dict(stages=[]))
            self.assertEqual(close.call_count, 1)

    def test_reordered_closing_recipe_rejected(self):
        stages = [dict(kind='hand', state=state, target_0_100=target, duration_s=1., settle_s=.3)
                  for state, target in zip(('THUMB_BASE', 'CLOSE_FINGERS'), reversed(FINGER_TARGETS))]
        with self.assertRaisesRegex(ValueError, 'thumb base first'):
            validate_sequence(dict(until_state='grip', stages=stages))


if __name__ == '__main__':
    unittest.main()
