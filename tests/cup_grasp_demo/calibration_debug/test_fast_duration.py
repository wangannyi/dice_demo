"""Recorded aborts, bounded tolerances and identical batched table screening."""

from copy import deepcopy
from contextlib import ExitStack
import math
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from cup_grasp_demo.calibration_debug.core import ROOT, Screen, cached_screen_geometry, read_json
from cup_grasp_demo.calibration_debug.parameters import shake_options
from cup_grasp_demo.calibration_debug.shake_tracking import FeedbackMonitor, check_delivery

EVIDENCE = ROOT / 'cup_grasp_demo/datasets/fast_duration_20260918/live'
WIDE = dict(tracking_error_limit_mm=40, tracking_error_grace_s=.5,
            joint_path_tolerance_deg=1, command_step_limit_deg=3, command_lag_limit_s=.15)


class DurationTest(unittest.TestCase):
    def setUp(self):
        self.plan = read_json(EVIDENCE / 'request_180346.json')['plan']

    def test_recorded_following_abort_passes_with_explicit_wider_limit(self):
        report = read_json(EVIDENCE / 'actual_175958.json')
        # This run has its own starting center and joint trajectory.
        plan = read_json(EVIDENCE / 'request_175958.json')['plan']
        last = report['last_observation']
        with self.assertRaisesRegex(RuntimeError, '15 mm'):
            FeedbackMonitor(plan).check(last['feedback'], last['elapsed_s'])
        plan['parameters'].update(WIDE)
        monitor = FeedbackMonitor(plan)
        for item in report['feedback'] + [last]:
            monitor.check(item['feedback'], item['elapsed_s'])
        changed = deepcopy(last['feedback'])
        changed['fk_flange_pose_m_rad'][2] += .004
        with self.assertRaisesRegex(RuntimeError, '高度|横向'):
            monitor.check(changed, last['elapsed_s'])

    def test_recorded_delivery_abort_is_between_old_and_new_step_limits(self):
        report = read_json(EVIDENCE / 'actual_180346.json')
        last = report['last_observation']
        sample = self.plan['samples'][round(last['elapsed_s'] * self.plan['sample_hz'])]
        q = report['commands'][-1]['target_q_rad']
        jump = math.degrees(max(abs(a-b) for a,b in zip(q, sample['q_rad'])))
        self.assertGreater(jump, 1)
        self.assertLess(jump, 3)
        with self.assertRaisesRegex(RuntimeError, '1°'):
            check_delivery(sample['q_rad'], q, last['elapsed_s'], sample['t_s'])
        check_delivery(sample['q_rad'], q, last['elapsed_s'], sample['t_s'], WIDE)
        with self.assertRaises(RuntimeError):
            check_delivery([q[0] + math.radians(3.01), *q[1:]], q, 0, 0, WIDE)
        with self.assertRaises(RuntimeError):
            check_delivery(q, q, .151, 0, WIDE)

    def test_transient_warning_recovers_but_sustained_error_stops(self):
        self.plan['parameters'].update(WIDE)
        monitor = FeedbackMonitor(self.plan)
        geometric = dict(along_m=.041, cross_m=0, height_m=0, rotation_deg=0)
        with patch('cup_grasp_demo.calibration_debug.shake_tracking.check_feedback', return_value=geometric), \
             patch('cup_grasp_demo.calibration_debug.shake_tracking.interpolate_displacement', return_value=0):
            self.assertTrue(monitor.check({}, 0)['tracking_warning'])
            self.assertTrue(monitor.check({}, .49)['tracking_warning'])
            geometric['along_m'] = 0
            self.assertFalse(monitor.check({}, .495)['tracking_warning'])
            geometric['along_m'] = .041
            monitor.check({}, 1)
            with self.assertRaisesRegex(RuntimeError, '持续'):
                monitor.check({}, 1.5)

    def test_new_fields_bounded_and_legacy_defaults_unchanged(self):
        base = self.plan['parameters']
        for name in WIDE:
            self.assertNotIn(name, shake_options({'shake': base}))
            for bad in (-1, float('nan'), True, 1000):
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    shake_options({'shake': dict(base, **{name: bad})})
        parsed = shake_options({'shake': dict(base, **WIDE)})
        self.assertEqual(parsed['duration_s'], 20)
        self.assertEqual(parsed['command_step_limit_deg'], 3)


class TableBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.screen = Screen()
        cls.plan = read_json(EVIDENCE / 'request_180346.json')['plan']
        cls.scene = dict(cup_support_base_m=[0,0,0], cup_normal_base=cls.plan['table_normal_base'],
                         cup_envelope_radius_m=.05, geometry={'height_m': .12})

    def test_batch_matches_scalar_for_every_pose_including_unsafe_sample(self):
        samples = [s['q_rad'] for s in self.plan['samples'][::29]]
        for z in (0, .1):
            scene = dict(self.scene, cup_support_base_m=[0,0,z])
            cfg = dict(table_margin_mm=10, cup_margin_mm=5)
            scalar = self.screen.check(samples, scene, True, cfg)
            batch = self.screen.check_table_batch(samples, scene, cfg)
            self.assertAlmostEqual(scalar['table_min_mm'], batch['table_min_mm'], places=8)
            self.assertEqual(scalar['table_link'], batch['table_link'])
            self.assertEqual(scalar['blockers'], batch['blockers'])
        bad = deepcopy(samples)
        bad[1][0] = np.nan
        with self.assertRaises(ValueError):
            self.screen.check_table_batch(bad, self.scene, cfg)

    def test_geometry_cache_is_scoped_to_one_run(self):
        # Empty fake models make construction observable without duplicating meshes.
        with patch.object(Screen, 'add'), patch('cup_grasp_demo.calibration_debug.core.load_model') as load:
            with cached_screen_geometry():
                a, b = Screen(), Screen()
                self.assertIs(a.meshes, b.meshes)
                self.assertEqual(load.call_count, 1)
            Screen()
            self.assertEqual(load.call_count, 2)


class ExecutorDurationTest(unittest.TestCase):
    def simulate(self, fail_after=None):
        from scipy.spatial.transform import Rotation
        from nero_revo2_control.kinematics import load_model
        from cup_grasp_demo.calibration_debug import shake_execution as sdk
        request = read_json(EVIDENCE / 'request_180346.json')
        request.update(authorized_epoch_s=1000, input_hashes={}, camera_process=None)
        plan = request['plan']
        plan['parameters'].update(WIDE)
        arm, clock, robot = load_model(), [0.], MagicMock()
        current = [plan['start_q_rad']]
        poses = {}
        def fk(q):
            key = tuple(q)
            if key not in poses:
                t = np.asarray(arm.fk(q))
                poses[key] = [*t[:3, 3], *Rotation.from_matrix(t[:3,:3]).as_euler('xyz')]
            return poses[key]
        robot.fk.side_effect = fk
        robot.move_js.side_effect = lambda q: current.__setitem__(0, list(q))
        robot.get_joint_limits_enabled.return_value = True
        robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(msg=SimpleNamespace(
            max_joint_spd=10, min_angle_limit=-10, max_angle_limit=10))
        robot.get_joint_acc_limits.return_value = SimpleNamespace(msg=SimpleNamespace(max_joint_acc=100))
        robot.get_flange_vel_acc_limits.return_value = SimpleNamespace(msg=SimpleNamespace(
            end_max_linear_vel=10, end_max_linear_acc=10))
        session = MagicMock(robot=robot)
        def row():
            return dict(q_rad=current[0], enabled=[True]*7,
                        status=dict(arm_status=0, ctrl_mode=1, motion_status=1),
                        fk_flange_pose_m_rad=fk(current[0]))
        def feedback(*args, **kwargs):
            # A slower feedback reader skips some 100 Hz reference samples.
            clock[0] += .045
            if fail_after is not None and clock[0] >= fail_after:
                raise TimeoutError('simulated feedback loss')
            return row()
        def sleep(seconds):
            clock[0] += seconds
        timers = SimpleNamespace(monotonic=lambda: clock[0], time=lambda:1000+clock[0], sleep=sleep)
        with ExitStack() as stack:
            for name, value in (
                ('time', timers), ('ShakeGuard', MagicMock()), ('control_evidence', MagicMock(return_value={})),
                ('hand_feedback', MagicMock(return_value={'positions_0_100':[100]*6})),
                ('verify_stop', MagicMock()),
            ):
                stack.enter_context(patch.object(sdk, name, value))
            for name, value in (
                ('load_sdk_runtime', MagicMock(return_value=(object(), object()))),
                ('PassivePoseSession', MagicMock(return_value=session)),
                ('stopped_window', MagicMock(return_value=([row()], {}))),
                ('joint_limits', MagicMock(return_value=[])), ('ready_blockers', MagicMock(return_value=[])),
                ('validate_target', MagicMock(side_effect=lambda q, _: q)),
                ('evidence_blockers', MagicMock(return_value=[])),
                ('take_can_control', MagicMock(return_value={'samples':[row()]})),
                ('fresh_feedback', MagicMock(side_effect=feedback)),
                ('fresh_hold', MagicMock(return_value={'target_rad':current[0]})),
            ):
                stack.enter_context(patch.object(sdk.core, name, value))
            return sdk.run(request)

    def test_full_executor_runs_twenty_seconds_with_slower_feedback(self):
        report = self.simulate()
        self.assertTrue(report['success'], report.get('error'))
        self.assertTrue(report['duration_completed'])
        self.assertTrue(report['returned_center'])
        self.assertGreaterEqual(report['reference_send_elapsed_s'], 20.)
        self.assertLess(report['reference_send_elapsed_s'], 20.1)
        self.assertGreater(len(report['commands']), 400)
        self.assertTrue(report['measured_wave']['tracking_verified'])

    def test_real_feedback_fault_still_stops_early_and_reports_incomplete_duration(self):
        report = self.simulate(fail_after=4.)
        self.assertFalse(report['success'])
        self.assertFalse(report['duration_completed'])
        self.assertIn('feedback loss', report['error'])
        self.assertLess(report['motion_elapsed_s'], 4.1)


if __name__ == '__main__':
    unittest.main()
