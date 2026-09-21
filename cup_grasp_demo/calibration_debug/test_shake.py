"""Shake units, timing, local IK, limit rejection and workflow isolation."""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.calibration_debug import debug, pipeline_runner, shake, shake_cli
from cup_grasp_demo.calibration_debug.core import ROOT, read_json
from cup_grasp_demo.calibration_debug.parameters import shake_options

EVIDENCE = ROOT / 'cup_grasp_demo/datasets/shake_assessment_20260918'
CONFIG = ROOT / 'cup_grasp_demo/calibration_debug/index_joint_center/config.json'


class ShakeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feedback = read_json(EVIDENCE / 'controller_limits.json')
        cls.session = read_json(EVIDENCE / 'session.json')
        cls.request = {'shake': dict(frequency_hz=1., amplitude_mm=100., azimuth_deg=10., duration_s=5.)}
        cls.plan = shake.make_plan(cls.session, cls.feedback, cls.request)

    def test_user_values_preserved_and_unsafe_profile_rejected(self):
        self.assertEqual(self.plan['parameters'], self.request['shake'])
        self.assertEqual(self.plan['total_stroke_mm'], 200)
        self.assertEqual(self.plan['nominal_cycles'], 5)
        self.assertFalse(self.plan['planning_passed'])
        self.assertFalse(self.plan['execution_enabled'])
        self.assertTrue(any('J7 加速度' in item for item in self.plan['blockers']))
        self.assertGreater(self.plan['cartesian_peak_acceleration_m_s2'], 3.94)
        self.assertGreater(self.plan['joint_acceleration_limit_ratio'][6], 6)
        self.assertFalse(self.plan['uniform_retiming_reference']['applied'])

    def test_direction_is_plus_x_toward_plus_y_and_tangent_to_table(self):
        d = self.plan['direction_base']
        self.assertAlmostEqual(math.degrees(math.atan2(d[1], d[0])), 10)
        self.assertAlmostEqual(float(np.dot(d, self.session['scene']['cup_normal_base'])), 0)
        self.assertAlmostEqual(np.linalg.norm(d), 1)

    def test_total_duration_includes_smooth_start_and_stop(self):
        opts = self.request['shake']
        for duration in [5., 5.3]:
            opts = dict(opts, duration_s=duration)
            s, v, a = shake.profile(np.array([0., duration]), opts)
            np.testing.assert_allclose([s, v, a], 0, atol=1e-12)
        self.assertEqual(self.plan['samples'][0]['t_s'], 0)
        self.assertEqual(self.plan['samples'][-1]['t_s'], 5)
        np.testing.assert_allclose(self.plan['samples'][0]['q_rad'], self.plan['start_q_rad'])
        np.testing.assert_allclose(self.plan['samples'][-1]['q_rad'], self.plan['start_q_rad'])
        s, _, _ = shake.profile(np.array([1.25, 1.75]), self.request['shake'])
        np.testing.assert_allclose(s, [.1, -.1], atol=1e-12)

    def test_analytic_profile_derivatives_match_finite_differences(self):
        t = np.array([.13, .67, 1.25, 2., 4.18, 4.79])
        eps = 1e-5
        _, v, a = shake.profile(t, self.request['shake'])
        sp, vp, _ = shake.profile(t+eps, self.request['shake'])
        sm, vm, _ = shake.profile(t-eps, self.request['shake'])
        np.testing.assert_allclose(v, (sp-sm)/(2*eps), atol=1e-8)
        np.testing.assert_allclose(a, (vp-vm)/(2*eps), atol=1e-7)

    def test_smaller_amplitude_can_pass_without_claiming_execution_verified(self):
        cfg = deepcopy(self.request)
        cfg['shake']['amplitude_mm'] = 10
        plan = shake.make_plan(self.session, self.feedback, cfg)
        self.assertTrue(plan['planning_passed'], plan['blockers'])
        self.assertFalse(plan['execution_enabled'])
        self.assertGreater(len(plan['validation_pending']), 0)
        self.assertEqual(self.request['shake']['amplitude_mm'], 100)

    def test_sampled_joint_path_preserves_orientation_and_table_height(self):
        kin = shake.Kinematics()
        center = np.array(self.plan['T_base_flange_center'])
        direction = np.array(self.plan['direction_base'])
        for sample in self.plan['samples'][::17]:
            pose, _ = kin.forward(sample['q_rad'])
            np.testing.assert_allclose(pose[:3, :3], center[:3, :3], atol=1e-7)
            np.testing.assert_allclose(pose[:3, 3], center[:3, 3] +
                                       sample['displacement_m']*direction, atol=1e-7)
        self.assertGreaterEqual(self.plan['minimum_joint_margin_deg'], 1)

    def test_bad_units_types_unknown_keys_and_missing_limits_fail(self):
        for key, value in [('amplitude_mm', True), ('duration_s', float('nan')),
                           ('frequency_hz', 0), ('azimuth_deg', '10')]:
            cfg = deepcopy(self.request)
            cfg['shake'][key] = value
            with self.assertRaises(ValueError):
                shake_options(cfg)
        cfg = deepcopy(self.request)
        cfg['shake']['amplitude_cm'] = 10
        with self.assertRaises(ValueError):
            shake_options(cfg)
        feedback = deepcopy(self.feedback)
        feedback['limits'][0]['max_acceleration_rad_s2'] = 0
        with self.assertRaises(ValueError):
            shake.make_plan(self.session, feedback, self.request)

    def test_cli_offline_never_opens_hardware_and_overwrites_same_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'shake_plan.json'
            config = Path(temp)/'config.json'
            cfg = read_json(CONFIG)
            cfg['shake'] = self.request['shake']
            config.write_text(json.dumps(cfg))
            args = SimpleNamespace(config=config, session=EVIDENCE, output=output,
                                   feedback_json=EVIDENCE/'controller_limits.json')
            with patch.object(shake_cli.subprocess, 'run', side_effect=AssertionError('hardware')):
                self.assertEqual(shake_cli.plan(args), 2)
                self.assertEqual(shake_cli.plan(args), 2)
            self.assertTrue(read_json(output)['offline_only'])

    def test_no_shake_execute_flag_and_old_pipeline_default_stays_grip(self):
        with patch.object(debug, 'dispatch') as dispatch:
            debug.main(['pipeline', '--session', '/unused'])
            self.assertEqual(dispatch.call_args.args[0].until, 'grip')
        with self.assertRaises(SystemExit) as error:
            debug.main(['shake-plan', '--session', '/unused', '--execute'])
        self.assertEqual(error.exception.code, 2)

    def test_readback_allowlist_rejects_motion_mode_and_parameter_writes(self):
        spec = importlib.util.spec_from_file_location('shake_readback',
                 ROOT/'cup_grasp_demo/calibration_debug/shake_readback.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for joint in range(1, 8):
            self.assertTrue(module.is_read_only_request(0x472, bytes([joint, 1, 0, 0, 0, 0, 0, 0])))
            self.assertTrue(module.is_read_only_request(0x472, bytes([joint, 2, 0, 0, 0, 0, 0, 0])))
        for can_id in (0x150, 0x155, 0x156, 0x157, 0x170, 0x1B5, 0x471):
            self.assertFalse(module.is_read_only_request(can_id, bytes(8)))
        self.assertFalse(module.is_read_only_request(0x472, bytes([1, 1, 0, 0, 1, 0, 0, 0])))

    def test_pipeline_shake_stage_only_plans_and_surfaces_blocker(self):
        backend = pipeline_runner.Backend.__new__(pipeline_runner.Backend)
        backend.args = SimpleNamespace(session=Path('/unused'), config=CONFIG)
        with patch.object(shake_cli, 'plan', return_value=2), \
             patch.object(backend, 'move', side_effect=AssertionError('motion')):
            with self.assertRaisesRegex(RuntimeError, '未执行摇晃'):
                backend.perform('SHAKE_PLAN')
        with patch.object(shake_cli, 'plan', return_value=0), \
             patch.object(backend, 'move', side_effect=AssertionError('motion')):
            result = backend.perform('SHAKE_PLAN')
            self.assertFalse(result['execution_enabled'])

    def test_optional_pipeline_phase_pauses_then_resumes_without_repeating_grip(self):
        from cup_grasp_demo.calibration_debug.test_pipeline_runner import FakeBackend
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(session=Path(temp), config=CONFIG, status=False,
                                   until='shake-plan', mode='step', execute=True,
                                   resume=False, show=False)
            backend = FakeBackend(args)
            answers = iter([''] * 5 + ['q'])
            pipeline_runner.run(args, backend_factory=lambda _: backend,
                                ask=lambda _: next(answers))
            self.assertEqual(backend.calls, list(pipeline_runner.PHASES))
            state = read_json(Path(temp)/pipeline_runner.STATE_FILE)
            self.assertEqual(state['phases'][-1], 'SHAKE_PLAN')
            self.assertEqual(state['next_index'], 5)
            args.resume = True
            pipeline_runner.run(args, backend_factory=lambda _: backend, ask=lambda _: '')
            self.assertEqual(backend.calls, list(pipeline_runner.PHASES) + ['SHAKE_PLAN'])
            state = read_json(Path(temp)/pipeline_runner.STATE_FILE)
            self.assertEqual(state['completed_state'], 'SHAKE_PLANNED')

    def test_legacy_grasp_backend_does_not_validate_unused_shake_options(self):
        from cup_grasp_demo.calibration_debug.core import load_config
        cfg = load_config(CONFIG)
        cfg['shake'] = {'bad': 'unused'}
        args = SimpleNamespace(config=CONFIG, show=False, mode='auto', until='grip')
        with patch.object(pipeline_runner, 'load_config', return_value=cfg):
            pipeline_runner.Backend(args)
            args.until = 'shake-plan'
            with self.assertRaises(ValueError):
                pipeline_runner.Backend(args)


if __name__ == '__main__':
    unittest.main()
