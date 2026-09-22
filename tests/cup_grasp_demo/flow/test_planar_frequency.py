"""Adapt timing for a real blocked pose without relaxing geometry or limits."""

from copy import deepcopy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scipy.spatial.transform import Rotation

from cup_grasp_demo.flow import planar_shake as planner
from cup_grasp_demo.flow import planar_shake_execution as executor
from cup_grasp_demo.flow.parameters import planar_shake_options, shake_options

DATA = Path(__file__).resolve().parents[3] / 'datasets/planar_frequency_fit_20260920'


class FrequencyFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feedback = json.loads((DATA / 'limits.json').read_text())
        cls.blocked = json.loads((DATA / 'planar_js_plan.json').read_text())
        cls.tcp = cls.blocked['T_flange_virtual_tcp']
        cls.opts = dict(cls.blocked['parameters'], auto_reduce_frequency=True)
        cls.adapted = planner.make_plan(cls.feedback, cls.opts, cls.tcp)

    def test_real_blocked_pose_preserves_stroke_time_and_passes_each_joint_budget(self):
        self.assertFalse(self.blocked['planning_passed'])
        p = self.adapted
        self.assertTrue(p['planning_passed'], p['blockers'])
        self.assertEqual(p['requested_parameters']['frequency_hz'], 1.44)
        self.assertLess(p['parameters']['frequency_hz'], 1.44)
        self.assertGreater(p['parameters']['frequency_hz'], 1.3)
        self.assertTrue(p['frequency_adaptation']['applied'])
        for key in ['amplitude_mm', 'duration_s', 'azimuth_deg', 'limit_utilization']:
            self.assertEqual(p['parameters'][key], self.opts[key])
        self.assertEqual(len(p['samples']), 2001)
        self.assertEqual(p['samples'][-1]['t_s'], 20.)
        self.assertLess(min(s['displacement_m'] for s in p['samples']), -.0499)
        self.assertGreater(max(s['displacement_m'] for s in p['samples']), .0499)
        for kind in ['velocity_rad_s', 'acceleration_rad_s2']:
            for demand, budget in zip(p['joint_peak_'+kind], p['joint_'+kind.replace('_rad', '_budget_rad')]):
                self.assertLessEqual(demand, budget)
        for key in ['z_mm', 'rx_deg', 'ry_deg']:
            self.assertLess(p['peak_deviation'][key], .02)

    def test_opt_out_retains_requested_frequency_and_existing_blocker(self):
        raw = {k:v for k,v in self.opts.items() if k != 'auto_reduce_frequency'}
        p = planner.make_plan(self.feedback, raw, self.tcp)
        self.assertFalse(p['planning_passed'])
        self.assertEqual(p['parameters']['frequency_hz'], 1.44)
        self.assertFalse(p['frequency_adaptation']['applied'])
        self.assertEqual(p['blockers'], self.blocked['blockers'])

    def test_executor_uses_adjusted_frequency_for_full_duration_and_measurement(self):
        p = self.adapted
        clock = [0.]
        q = list(p['start_q_rad'])
        kin = planner.Kinematics()

        def send(target):
            q[:] = target

        def poll(robot, stamps):
            flange, _ = kin.forward(q)
            row = dict(q_rad=q.copy(), enabled=[True]*7,
                       status=dict(arm_status=0, ctrl_mode=1, motion_status=1),
                       fk_flange_pose_m_rad=[*flange[:3, 3],
                           *Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')])
            return row, {'test': clock[0]}

        def sleep(duration):
            clock[0] += max(duration, .000001)

        robot = SimpleNamespace(get_joint_limits_enabled=lambda: True, move_js=send)
        receipt = {}
        executor.stream(robot, p, receipt, clock=lambda: clock[0], sleep=sleep, poll=poll)
        self.assertTrue(receipt['duration_completed'])
        self.assertGreaterEqual(receipt['motion_elapsed_s'], 20.)
        self.assertTrue(receipt['measured_wave']['tracking_verified'])
        self.assertAlmostEqual(receipt['measured_wave']['feedback_frequency_hz'],
                               p['parameters']['frequency_hz'], places=2)

    def test_unachievable_budget_is_still_blocked_at_minimum_two_cycles(self):
        feedback = deepcopy(self.feedback)
        for row in feedback['limits']:
            row['max_acceleration_rad_s2'] = .000001
        p = planner.make_plan(feedback, self.opts, self.tcp)
        self.assertFalse(p['planning_passed'])
        self.assertEqual(p['parameters']['frequency_hz'], .1)
        self.assertEqual(p['parameters']['duration_s'], 20.)
        self.assertTrue(any('加速度' in b for b in p['blockers']))

    def test_velocity_budget_also_controls_adaptation(self):
        feedback = deepcopy(self.feedback)
        for row in feedback['limits']:
            row['max_velocity_rad_s'] = .06
        p = planner.make_plan(feedback, self.opts, self.tcp)
        self.assertTrue(p['planning_passed'], p['blockers'])
        self.assertLessEqual(max(p['joint_peak_velocity_rad_s']), .06)

    def test_geometry_failure_is_not_hidden_by_frequency_policy(self):
        with patch.object(planner, 'solve_planar', side_effect=ValueError('IK blocked')):
            with self.assertRaisesRegex(ValueError, 'IK blocked'):
                planner.make_plan(self.feedback, self.opts, self.tcp)

    def test_start_near_joint_limit_reports_joint_and_required_range(self):
        feedback = deepcopy(self.feedback)
        feedback['q_after_rad'][1] = planner.Kinematics().lower[1] + math.radians(.25)
        with self.assertRaisesRegex(ValueError, '起点距关节限位不足 1°：J2=.*规划范围'):
            planner.make_plan(feedback, self.opts, self.tcp)

    def test_policy_is_boolean_and_not_accepted_by_legacy_strategy(self):
        with self.assertRaisesRegex(ValueError, 'boolean'):
            planar_shake_options(dict(self.opts, auto_reduce_frequency='true'))
        with self.assertRaises(ValueError):
            shake_options({'shake': dict(frequency_hz=1., amplitude_mm=50., duration_s=20.,
                                         azimuth_deg=10., auto_reduce_frequency=True)})


if __name__ == '__main__':
    unittest.main()
