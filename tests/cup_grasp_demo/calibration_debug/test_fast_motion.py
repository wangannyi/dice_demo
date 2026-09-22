"""Recorded path, fixed-delay feedback and FAST execution regressions; no hardware."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.calibration_debug import grasp_cli, shake_tracking
from cup_grasp_demo.calibration_debug.core import ROOT, Screen, load_config, read_json, write_json
from cup_grasp_demo.calibration_debug.direct_grasp import approach_stages, make_direct_plan
from cup_grasp_demo.calibration_debug.grasp import options
from cup_grasp_demo.calibration_debug.parameters import approach_options, shake_options

EVIDENCE = ROOT / 'cup_grasp_demo/datasets/fast_motion_20260918/live'


class DirectRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = read_json(EVIDENCE / 'ready_plan.json')
        cls.session = read_json(EVIDENCE / 'session.json')
        cls.cfg = load_config(EVIDENCE / 'config.json')
        cls.cfg['side_grasp']['transfer_mode'] = 'direct_checked'
        cls.scene = cls.original['scene']
        cls.plan = make_direct_plan(cls.session, cls.original['start_q_rad'], cls.cfg,
                                   cls.scene, options(cls.cfg), 'home', 'ready')

    def test_recorded_home_route_reduces_stops_and_preserves_ready_point(self):
        plan = self.plan
        self.assertFalse(plan['blockers'])
        self.assertLessEqual(len(plan['stages']), 2)
        self.assertEqual(plan['stages'][0]['name'], 'direct_to_ready')
        self.assertEqual(plan['stages'][-1]['state_completed'], 'READY')
        np.testing.assert_allclose(plan['close_ready_base_m'], self.original['close_ready_base_m'])
        self.assertEqual(plan['closing_gap_mm'], self.original['closing_gap_mm'])
        for stage in plan['stages']:
            self.assertGreaterEqual(stage['screen']['table_min_mm'], self.cfg['table_margin_mm'])

    def test_direct_path_blockage_falls_back_instead_of_skipping_screen(self):
        with patch('cup_grasp_demo.calibration_debug.direct_grasp.direct_transfer',
                   return_value=dict(stages=[], blockers=['direct table blocked'])):
            plan = make_direct_plan(self.session, self.original['start_q_rad'], self.cfg,
                                    self.scene, options(self.cfg), 'home', 'ready')
        self.assertEqual(plan['direct_fallback_reasons'], ['direct table blocked'])
        self.assertFalse(plan['blockers'])
        self.assertGreater(len(plan['stages']), 2)
        self.assertNotEqual(plan['stages'][0]['name'], 'direct_to_ready')

    def test_disabled_approach_allows_one_checked_move_to_final_gap(self):
        cfg = deepcopy(self.cfg)
        cfg['side_grasp']['approach']['enabled'] = False
        plan = make_direct_plan(self.session, self.original['start_q_rad'], cfg,
                                self.scene, options(cfg), 'home', 'ready')
        self.assertFalse(plan['blockers'])
        self.assertEqual(len(plan['stages']), 1)
        self.assertEqual(plan['closing_gap_mm'], 10)
        np.testing.assert_allclose(plan['close_ready_base_m'], self.plan['close_ready_base_m'])

    def test_rejected_combined_approach_uses_original_checked_waypoints(self):
        screen = Screen()
        original_check = screen.check
        checks = []
        def check(*args, **kwargs):
            result = original_check(*args, **kwargs)
            checks.append(result)
            if len(checks) == 1:
                result['blockers'].append('combined shortcut rejected')
            return result
        screen.check = check
        stages, blockers = approach_stages(self.session, self.cfg, self.scene, screen,
            self.plan['stages'][0]['target_q_rad'], approach_options(self.cfg['side_grasp']), 10)
        self.assertFalse(blockers)
        self.assertEqual(len(stages), 10)
        self.assertTrue(all(s['name'].startswith('approach_') for s in stages))


class DelayTest(unittest.TestCase):
    def test_twenty_second_delayed_feedback_preserves_measured_frequency_and_duration(self):
        from scipy.spatial.transform import Rotation
        from nero_revo2_control.kinematics import load_model
        plan = read_json(EVIDENCE / 'request.json')['plan']
        plan['parameters']['feedback_reference_delay_s'] = .08
        arm, monitor, rows = load_model(), shake_tracking.FeedbackMonitor(plan), []
        for index in range(0, len(plan['samples']), 2):
            sample = plan['samples'][max(0, index - 8)]
            t = np.asarray(arm.fk(sample['q_rad']))
            feedback = dict(q_rad=sample['q_rad'], enabled=[True]*7,
                            status=dict(arm_status=0, ctrl_mode=1, motion_status=1),
                            fk_flange_pose_m_rad=[*t[:3, 3], *Rotation.from_matrix(t[:3, :3]).as_euler('xyz')])
            elapsed = plan['samples'][index]['t_s']
            rows.append(dict(elapsed_s=elapsed, tracking=monitor.check(feedback, elapsed)))
        wave = shake_tracking.measured_wave(rows, plan)
        self.assertEqual(rows[-1]['elapsed_s'], 20.)
        self.assertTrue(wave['tracking_verified'])
        self.assertAlmostEqual(wave['feedback_frequency_hz'], .45, places=3)
        self.assertGreater(wave['feedback_total_stroke_mm'], 99)

    def test_recorded_failure_is_time_aligned_without_changing_reference_or_path_guards(self):
        request = read_json(EVIDENCE / 'request.json')
        report = read_json(EVIDENCE / 'actual.json')
        plan = request['plan']
        row = report['last_observation']
        with self.assertRaisesRegex(RuntimeError, '沿程跟踪偏差'):
            shake_tracking.FeedbackMonitor(plan).check(row['feedback'], row['elapsed_s'])
        before = deepcopy(plan['samples'])
        plan['parameters']['feedback_reference_delay_s'] = .08
        monitor = shake_tracking.FeedbackMonitor(plan)
        all_rows = [(r['feedback'], r['elapsed_s']) for r in report['feedback']]
        all_rows.append((row['feedback'], row['elapsed_s']))
        measured = [monitor.check(r, t) for r, t in all_rows]
        self.assertLess(max(abs(r['aligned_following_error_mm']) for r in measured), 5)
        self.assertGreater(abs(measured[-1]['raw_following_error_mm']), 15)
        self.assertEqual(before, plan['samples'])
        unsafe = deepcopy(row['feedback'])
        unsafe['fk_flange_pose_m_rad'][2] += .004
        with self.assertRaises(RuntimeError):
            monitor.check(unsafe, row['elapsed_s'])

    def test_delay_is_bounded_and_does_not_accept_stationary_hand(self):
        plan = read_json(EVIDENCE / 'request.json')['plan']
        for invalid in (-.01, .16, True, float('nan')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                shake_options({'shake': dict(plan['parameters'], feedback_reference_delay_s=invalid)})
        self.assertNotIn('feedback_reference_delay_s', shake_options({'shake': plan['parameters']}))
        plan['parameters']['feedback_reference_delay_s'] = .08
        row = read_json(EVIDENCE / 'actual.json')['feedback'][0]['feedback']
        with self.assertRaisesRegex(RuntimeError, '沿程跟踪偏差'):
            shake_tracking.FeedbackMonitor(plan).check(row, 2.9)


class FastHandoffTest(unittest.TestCase):
    def test_fast_grip_keeps_sdk_execution_but_skips_rebuild_and_preview_capture(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = read_json(EVIDENCE / 'ready_plan.json')
            plan = dict(source, session_path=str(directory), start_state='ready', until_state='grip')
            path = directory / 'grip_plan.json'
            write_json(path, plan)
            prepared = dict(plan_sha256=grasp_cli.digest(path))
            with patch.object(grasp_cli.common, 'verify_session',
                              return_value=({'replay_only': False}, {})), \
                 patch.object(grasp_cli, 'validate'), \
                 patch.object(grasp_cli, 'observed_scene', side_effect=AssertionError('duplicate scene')), \
                 patch.object(grasp_cli, 'make_grasp_plan', side_effect=AssertionError('duplicate IK')), \
                 patch.object(grasp_cli.common, 'capture_with_feedback', side_effect=AssertionError('preview')), \
                 patch.object(grasp_cli.common, 'capture_rgbd', side_effect=AssertionError('preview')), \
                 patch.object(grasp_cli.common, 'bridge', return_value={'last_state': 'GRIP_COMMANDS_SENT'}) as sdk:
                result = grasp_cli.execute(SimpleNamespace(plan=path, execute=True, show=False, fast=True),
                                           confirm=lambda _: 'GRASP', prepared=prepared)
                self.assertEqual(result, 0)
                self.assertEqual(sdk.call_args.args[0], 'run')
                request = read_json(sdk.call_args.args[3])
                self.assertTrue(request['execution_authorized'])
                prepared['plan_sha256'] = 'changed'
                with self.assertRaisesRegex(ValueError, '计划发生改变'):
                    grasp_cli.execute(SimpleNamespace(plan=path, execute=True, fast=True), prepared=prepared)


if __name__ == '__main__':
    unittest.main()
