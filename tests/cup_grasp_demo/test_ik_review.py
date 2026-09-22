"""Offline checks for the proposed grasp's flange kinematics audit."""

from __future__ import annotations

import copy
import math
import subprocess
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

from nero_calibration.core import matrix_pose
from nero_revo2_control.kinematics import load_model

from cup_grasp_demo import ik_review


def synthetic_plan(q_rad):
    model = load_model()
    current = model.fk(q_rad)
    waypoints = {}
    for name, delta_deg in [('pregrasp', 1.0), ('contact', 2.0), ('lift', 3.0)]:
        target_q = list(q_rad)
        target_q[0] += math.radians(delta_deg)
        target = model.fk(target_q)
        waypoints[name] = {
            'T_base_flange': [list(row) for row in target],
            'flange_pose_base_m_rad': matrix_pose(target),
        }
    return {
        'schema': 1, 'kind': 'read_only_side_grasp_proposal', 'units': 'm_rad',
        'snapshot_frame_id': 'synthetic:1',
        'current_flange_pose_base_m_rad': matrix_pose(current),
        'waypoints': waypoints,
    }


class GraspIKReviewTests(unittest.TestCase):
    def setUp(self):
        self.current = tuple(math.radians(v) for v in
                             (55.0, -78.0, 80.0, -45.0, 130.0, -30.0, 40.0))
        self.plan = synthetic_plan(self.current)

    def test_nearby_three_stage_fk_ik_agrees_without_claiming_clearance(self):
        result = ik_review.review_plan(self.plan, self.current)
        self.assertTrue(result['current_fk']['agreement_passed'])
        self.assertTrue(result['kinematic_checks_passed'], result['blockers'])
        self.assertEqual([item['name'] for item in result['stages']],
                         ['pregrasp', 'contact', 'lift'])
        for stage in result['stages']:
            self.assertTrue(stage['ik']['success'], stage['ik']['reason'])
            self.assertLess(stage['ik']['position_error_m'], 0.001)
            self.assertLessEqual(stage['ik']['attempt_count'], 7)
            self.assertTrue(stage['joint_path']['joint_limits_passed'])
            self.assertTrue(stage['joint_path']['continuity_passed'])
            self.assertFalse(stage['joint_path']['collision_verified'])
        self.assertFalse(result['executable'])
        self.assertFalse(result['motion_sent'])
        self.assertFalse(result['scene_collision_verified'])
        self.assertFalse(result['controller_trajectory_verified'])

    def test_clearance_required_adds_first_stage_and_cannot_be_skipped(self):
        model = load_model()
        clearance_q = list(self.current)
        clearance_q[0] += math.radians(0.5)
        clearance = model.fk(clearance_q)
        plan = copy.deepcopy(self.plan)
        plan['checks'] = {'clearance_required': True}
        plan['waypoints']['clearance'] = {
            'T_base_flange': [list(row) for row in clearance],
            'flange_pose_base_m_rad': matrix_pose(clearance),
        }
        report = ik_review.review_plan(plan, self.current)
        self.assertEqual(report['sequence'],
                         ['current', 'clearance', 'pregrasp', 'contact', 'lift'])
        self.assertEqual([stage['name'] for stage in report['stages']],
                         ['clearance', 'pregrasp', 'contact', 'lift'])
        self.assertTrue(report['kinematic_checks_passed'], report['blockers'])
        self.assertTrue(report['stages'][0]['ik']['success'])
        self.assertTrue(report['stages'][0]['joint_path']['continuity_passed'])
        self.assertFalse(report['stages'][0]['joint_path']['collision_verified'])
        del plan['waypoints']['clearance']
        with self.assertRaisesRegex(ValueError, 'clearance waypoint required'):
            ik_review.review_plan(plan, self.current)

    def test_extra_clearance_without_required_flag_preserves_three_stage_default(self):
        plan = copy.deepcopy(self.plan)
        plan['waypoints']['clearance'] = plan['waypoints']['pregrasp']
        report = ik_review.review_plan(plan, self.current)
        self.assertEqual(report['sequence'], ['current', 'pregrasp', 'contact', 'lift'])
        self.assertFalse(report['clearance_required'])

    def test_bounded_seed_search_can_recover_after_first_local_solver_failure(self):
        model = load_model()
        target = model.fk(self.current)
        original_ik = model.ik
        calls = 0

        def first_seed_fails(*args, **kwargs):
            nonlocal calls
            calls += 1
            solved = original_ik(*args, **kwargs)
            if calls == 1:
                return replace(solved, success=False, reason='injected_local_stall')
            return solved

        with patch.object(model, 'ik', side_effect=first_seed_fails):
            selected, result, attempts = ik_review._solve_bounded(model, target, self.current)
        self.assertTrue(result.success)
        self.assertEqual(selected, 'j3_plus_10deg')
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]['reason'], 'injected_local_stall')
        self.assertLessEqual(len(attempts), ik_review.MAX_IK_SEEDS_PER_STAGE)

    def test_documented_j7_can_be_inside_mechanical_range_but_outside_urdf(self):
        current = list(self.current)
        current[6] = math.radians(95.0)
        plan = synthetic_plan(current)
        result = ik_review.review_plan(plan, current)
        j7 = result['current_joint_limits'][6]
        self.assertTrue(j7['within_documented'])
        self.assertFalse(j7['within_urdf'])
        self.assertAlmostEqual(j7['urdf_range_deg'][1], 90.0, places=3)
        self.assertIn('J7', result['seed_clipped_to_urdf_axes'])
        self.assertFalse(result['kinematic_checks_passed'])
        self.assertIn('J7 current angle exceeds conservative URDF range',
                      result['blockers'])
        self.assertFalse(result['collision_verified'])

    def test_current_j2_at_minus_100_is_documented_but_outside_urdf(self):
        current = list(self.current)
        current[1] = math.radians(-100.0)
        result = ik_review.review_plan(synthetic_plan(current), current)
        j2 = result['current_joint_limits'][1]
        self.assertTrue(j2['within_documented'])
        self.assertFalse(j2['within_urdf'])
        self.assertIn('J2', result['seed_clipped_to_urdf_axes'])
        self.assertFalse(result['kinematic_checks_passed'])

    def test_k3_feedback_case_marks_both_j2_and_j7_recovery_blockers(self):
        # Fixed read-only K3 feedback from 2026-09-15; the hardware posture can change.
        current = tuple(math.radians(value) for value in
                        (40.386, -100.349, -100.184, 73.146, -1.918, -30.399, 97.660))
        sdk_limits = [(-155, 155), (-100, 100), (-158, 158),
                      (-58, 123), (-158, 158), (-42, 55), (-90, 90)]
        report = ik_review.review_plan(synthetic_plan(current), current,
                                       sdk_limits_deg=sdk_limits)
        j2, j7 = report['current_joint_limits'][1], report['current_joint_limits'][6]
        self.assertTrue(j2['within_documented'])
        self.assertFalse(j2['within_urdf'])
        self.assertFalse(j2['within_sdk'])
        self.assertFalse(j7['within_documented'])
        self.assertFalse(j7['within_urdf'])
        self.assertFalse(j7['within_sdk'])
        self.assertIn('J2 current angle exceeds SDK soft range', report['blockers'])
        self.assertIn('J7 current angle exceeds documented mechanical range', report['blockers'])
        self.assertFalse(report['kinematic_checks_passed'])

    def test_disagreeing_feedback_fk_or_waypoint_encoding_is_caught(self):
        plan = copy.deepcopy(self.plan)
        plan['current_flange_pose_base_m_rad'][0] += 0.020
        result = ik_review.review_plan(plan, self.current)
        self.assertFalse(result['current_fk']['agreement_passed'])
        self.assertFalse(result['kinematic_checks_passed'])
        plan = copy.deepcopy(self.plan)
        plan['waypoints']['contact']['flange_pose_base_m_rad'][0] += 0.005
        with self.assertRaisesRegex(ValueError, 'contact flange pose and matrix disagree'):
            ik_review.review_plan(plan, self.current)

    def test_existing_joint_feedback_is_parsed_without_control_action(self):
        payload = {'event': 'read_joints', 'joints_rad': list(self.current),
                   'sdk_limits_deg': [[-90, 90]]*7}
        completed = subprocess.CompletedProcess([], 0, stdout='noise\n'+str(payload).replace("'", '"')+'\n')
        with patch.object(ik_review.subprocess, 'run', return_value=completed) as call:
            joints, sdk_limits = ik_review.read_joints_existing_demo('can0', sys.executable)
        self.assertEqual(joints, self.current)
        self.assertEqual(len(sdk_limits), 7)
        args = call.call_args.args[0]
        self.assertEqual(args[-1], 'read-joints')
        self.assertNotIn('move-p', args)
        self.assertNotIn('move-j', args)

    def test_module_import_does_not_load_can_or_pyagxarm(self):
        isolated = subprocess.run(
            [sys.executable, '-c',
             "import cup_grasp_demo.ik_review, sys; "
             "assert 'pyAgxArm' not in sys.modules; assert 'can' not in sys.modules"],
            capture_output=True, text=True, check=False)
        self.assertEqual(isolated.returncode, 0, isolated.stderr)


if __name__ == '__main__':
    unittest.main()
