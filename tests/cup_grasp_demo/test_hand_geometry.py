"""Focused read-only checks for the official right-Revo2 fingertip screen."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cup_grasp_demo.hand_geometry import (
    RightRevo2Model, check_hand_pose, check_plan_geometry, main,
)
from nero_calibration.core import PALM


class HandGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RightRevo2Model()

    def test_mount_transform_and_palm_inversion_match(self):
        model = self.model
        self.assertEqual(len(model.collision), 17)
        np.testing.assert_allclose(model.T_flange_hand_base[:3, 3],
                                   [0.043, 0, -0.0235], atol=1e-6)
        flange = np.eye(4)
        palm = flange @ PALM
        from_flange = model.tip_transforms_from_flange(flange)
        from_palm = model.tip_transforms_from_palm(palm, PALM)
        self.assertEqual(set(from_flange), {'thumb', 'index', 'middle', 'ring', 'pinky'})
        for name in from_flange:
            np.testing.assert_allclose(from_flange[name], from_palm[name], atol=1e-10)
            self.assertTrue(np.isfinite(from_flange[name]).all())
        self.assertEqual(model.provenance['model_sources'][2]['path'].split('/')[-1],
                         'revo2_right_hand.urdf')

    def test_urdf_mimic_and_limit_validation(self):
        model = self.model
        angles = {'right_index_proximal_joint': .3}
        expected = .3*1.155
        actual = model._angles(angles)['right_index_distal_joint']
        self.assertAlmostEqual(actual, expected)
        open_tip = model.tip_transforms_from_flange(np.eye(4))['index']
        flexed_tip = model.tip_transforms_from_flange(
            np.eye(4), joint_positions_rad=angles)['index']
        self.assertGreater(np.linalg.norm(open_tip[:3, 3]-flexed_tip[:3, 3]), .01)
        for bad in ({'right_index_proximal_joint': 99.},
                    {'right_index_distal_joint': .1},
                    {'right_unknown_joint': .1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model.tip_transforms_from_flange(np.eye(4), joint_positions_rad=bad)

    def test_pose_screen_blocks_below_table_and_inside_cup(self):
        model = self.model
        flange = np.eye(4)
        tips = model.tip_transforms_from_flange(flange)
        pinky = tips['pinky'][:3, 3]
        screen = check_hand_pose(
            flange, support_center_m=pinky+[0, 0, .006],
            axis=[0, 0, 1], cup_diameter_m=.075, cup_height_m=.072,
            model=model)
        self.assertLess(screen['tips']['pinky']['table_altitude_m'], 0)
        self.assertFalse(screen['tips']['pinky']['table_margin_passed'])
        self.assertFalse(screen['table_margin_passed'])
        self.assertTrue(screen['possible_collision'])
        inside = check_hand_pose(
            flange, support_center_m=tips['index'][:3, 3]-[0, 0, .035],
            axis=[0, 0, 1], cup_diameter_m=.075, cup_height_m=.072,
            model=model)
        self.assertLessEqual(inside['tips']['index']['cup_signed_clearance_m'], 0)
        self.assertFalse(inside['tips']['index']['cup_margin_passed'])

    def test_real_candidate_contact_and_joint_linear_path(self):
        plan_path = Path('/tmp/nero_home_green_live_plan.json')
        ik_path = Path('/tmp/nero_home_green_live_ik.json')
        if not plan_path.exists() or not ik_path.exists():
            self.skipTest('Transient local home-cup review artifacts unavailable')
        plan = json.loads(plan_path.read_text())
        ik = json.loads(ik_path.read_text())
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        report = check_plan_geometry(plan, ik_review=ik, plan_sha256=plan_sha)
        self.assertEqual(report['plan_sha256'], plan_sha)
        contact = report['stages']['contact']['flange_target_pose']
        self.assertAlmostEqual(contact['tips']['pinky']['table_altitude_m'],
                               -.005, delta=.002)
        self.assertAlmostEqual(contact['tips']['ring']['table_altitude_m'],
                               .0008, delta=.002)
        self.assertLess(contact['tips']['index']['cup_signed_clearance_m'], 0)
        self.assertLess(contact['exact_collision_mesh_table_min_m'], -.005)
        self.assertFalse(report['stages']['pregrasp']['possible_collision'])
        self.assertTrue(report['stages']['contact']['possible_collision'])
        self.assertTrue(report['approach_geometry_passed'])
        self.assertFalse(report['contact_table_margin_passed'])
        self.assertFalse(report['contact_geometry_passed'])
        self.assertFalse(report['actual_hand_posture_verified'])
        self.assertEqual(report['joint_linear_paths']['clearance'][
            'first_possible_collision_sample'], None)
        self.assertIsNotNone(report['joint_linear_paths']['contact'][
            'first_possible_collision_sample'])
        self.assertFalse(report['motion_sent'])
        self.assertEqual(report['stages']['contact']['flange_target_pose'][
            'hand_posture_source'], 'nominal_urdf_zero_open_assumption')
        forged = copy.deepcopy(ik)
        forged['source_plan_frame_id'] = 'wrong-camera-frame'
        with self.assertRaises(ValueError):
            check_plan_geometry(plan, ik_review=forged)
        forged = copy.deepcopy(ik)
        forged['source_plan_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            check_plan_geometry(plan, ik_review=forged, plan_sha256=plan_sha)
        forged = copy.deepcopy(ik)
        forged['stages'][1]['ik']['joints_rad'] = list(ik['current_joints_rad'])
        with self.assertRaises(ValueError):
            check_plan_geometry(plan, ik_review=forged)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            self.assertEqual(main(['--plan', str(plan_path),
                                   '--ik-review', str(ik_path),
                                   '--output', str(output)]), 0)
            self.assertFalse(json.loads(output.read_text())['contact_geometry_passed'])
            self.assertEqual(main(['--plan', str(plan_path),
                                   '--output', str(output)]), 2)

    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            check_hand_pose(np.eye(4), support_center_m=[0, 0, 0],
                            axis=[0, 0, 0], cup_diameter_m=.075,
                            cup_height_m=.072, model=self.model)
        with self.assertRaises(ValueError):
            check_hand_pose(np.eye(4), support_center_m=[0, 0, 0],
                            axis=[0, 0, 1], cup_diameter_m=.075,
                            cup_height_m=.072, margin_m=.05, model=self.model)


if __name__ == '__main__':
    unittest.main()
