"""Focused tests for exact depth units, calibration opt-in, and target inversion."""

import ast
import tempfile
import time
import math
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from cup_grasp_demo.grasp import (
    READY_HOME_DEG, _adjacent_table_bbox, compute_side_grasp,
    load_snapshot, read_ready_home_feedback, save_snapshot,
)
from test_planning import fixture_calibration, fixture_localization
from nero_calibration.core import inverse, matrix


class GraspTests(unittest.TestCase):
    def test_ready_home_reference_matches_existing_nero_control_demo(self):
        control = (Path(__file__).resolve().parents[2] / 'nero_revo2_control'
                   / 'nero_revo2_demo.py')
        tree = ast.parse(control.read_text(encoding='utf-8'))
        ready = next(node.value for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(name, ast.Name) and name.id == 'READY_HOME_DEG'
                             for name in node.targets))
        self.assertEqual(READY_HOME_DEG, ast.literal_eval(ready))

    def test_read_ready_home_uses_existing_read_only_command_and_live_state(self):
        feedback = {
            'event': 'read_joints', 'joints_deg': list(READY_HOME_DEG),
            'joints_enabled': [True] * 7, 'arm_status': 0, 'ctrl_mode': 1,
        }
        with patch('cup_grasp_demo.grasp.subprocess.run',
                   return_value=SimpleNamespace(stdout=json.dumps(feedback))) as run:
            home = read_ready_home_feedback('can0', '/sdk/python')
        command = run.call_args.args[0]
        self.assertEqual(command[0], '/sdk/python')
        self.assertEqual(command[-1], 'read-joints')
        self.assertNotIn('--execute', command)
        self.assertTrue(home['confirmed'])
        self.assertEqual(home['max_joint_error_deg'], 0.)
        self.assertEqual(home['provenance']['kind'], 'can_joint_feedback_read_only')
        self.assertGreater(home['provenance']['host_feedback_received_time_ns'], 0)

        feedback['joints_deg'][6] += 1.1
        with patch('cup_grasp_demo.grasp.subprocess.run',
                   return_value=SimpleNamespace(stdout=json.dumps(feedback))):
            outside = read_ready_home_feedback('can0', '/sdk/python')
        self.assertFalse(outside['confirmed'])
        self.assertFalse(outside['joint_angles_arrived'])
        self.assertAlmostEqual(outside['max_joint_error_deg'], 1.1)

        feedback['joints_deg'] = list(READY_HOME_DEG)
        feedback['joints_enabled'][3] = False
        with patch('cup_grasp_demo.grasp.subprocess.run',
                   return_value=SimpleNamespace(stdout=json.dumps(feedback))):
            disabled = read_ready_home_feedback('can0', '/sdk/python')
        self.assertTrue(disabled['joint_angles_arrived'])
        self.assertFalse(disabled['confirmed'])

    def test_required_home_gates_only_opt_in_execute_readiness(self):
        calibration = fixture_calibration(True)
        localization = fixture_localization()
        localization['geometry']['dimensions'] = {'observed_side_diameter_m': .075}
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480,
                                   'fx': 600., 'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        palm = np.eye(4)
        palm[:3, :3] = np.diag([-1., -1., 1.])
        palm[:3, 3] = [.4, -.1, .6]
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        common = {'pose_source': {'kind': 'can_feedback_read_only'},
                  'now_ns': 11_000_000_000}
        default = compute_side_grasp(localization, calibration, flange, metadata, **common)
        self.assertTrue(default['checks']['execute_ready'])
        self.assertFalse(default['checks']['require_home'])
        missing = compute_side_grasp(localization, calibration, flange, metadata,
                                     require_home=True, **common)
        self.assertFalse(missing['checks']['execute_ready'])
        self.assertFalse(missing['checks']['home_gate_passed'])
        self.assertIsNone(missing['home_check'])
        failed = {'confirmed': False,
                  'provenance': {'kind': 'can_joint_feedback_read_only'}}
        not_home = compute_side_grasp(localization, calibration, flange, metadata,
                                      require_home=True, home_check=failed, **common)
        self.assertFalse(not_home['checks']['execute_ready'])
        passed = {'confirmed': True, 'joint_target_deg': list(READY_HOME_DEG),
                  'max_joint_error_deg': 0.3,
                  'provenance': {'kind': 'can_joint_feedback_read_only',
                                 'host_feedback_received_time_ns': 11_000_000_000}}
        at_home = compute_side_grasp(localization, calibration, flange, metadata,
                                     require_home=True, home_check=passed, **common)
        self.assertTrue(at_home['checks']['execute_ready'])
        self.assertTrue(at_home['checks']['home_confirmed'])
        self.assertEqual(at_home['home_check'], passed)

    def test_adjacent_table_roi_stays_under_cup_and_inside_frame(self):
        self.assertEqual(_adjacent_table_bbox((80, 60, 185, 180), (480, 640)),
                         (80, 170, 185, 240))
        self.assertEqual(_adjacent_table_bbox((80, 400, 185, 470), (480, 640)),
                         (80, 460, 185, 480))
        with self.assertRaisesRegex(ValueError, 'edge'):
            _adjacent_table_bbox((80, 420, 185, 475), (480, 640))

    def test_snapshot_preserves_z16_scale_and_rejects_asset_mismatch(self):
        frame = {
            'color_bgr': np.zeros((2, 3, 3), np.uint8),
            'depth_raw': np.uint16([[0, 1000, 1500], [500, 2000, 1000]]),
            'depth_scale_m': .001,
            'intrinsics': {'width': 3, 'height': 2, 'fx': 600., 'fy': 601.,
                           'ppx': 1., 'ppy': 1., 'dist_coeffs': [0.]*5,
                           'distortion_model': 'distortion.none'},
            'serial': 'fake-serial', 'frame': 'color_optical', 'frame_id': 42,
            'timestamps_ms': {'color': 100., 'depth_aligned': 101.},
            'timestamp_domains': {'color': 'hardware_clock', 'depth_aligned': 'hardware_clock'},
            'host_capture_time_ns': time.time_ns(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'snapshot'
            save_snapshot(source, frame)
            color, depth, metadata = load_snapshot(source)
            np.testing.assert_array_equal(color, frame['color_bgr'])
            np.testing.assert_array_equal(depth, frame['depth_raw'])
            self.assertEqual(metadata['depth_scale_m'], .001)
            self.assertEqual(depth[0, 1] * metadata['depth_scale_m'], 1.0)
            altered = np.zeros((2, 3, 3), np.uint8)
            altered[0, 0] = [255, 0, 0]
            cv2.imwrite(str(source/'color.png'), altered)
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_snapshot(source)
        frame['timestamps_ms']['depth_aligned'] = 270.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'timestamp gap'):
                save_snapshot(Path(tmp)/'unsynced', frame)

    def test_provisional_side_grasp_maps_tcp_and_keeps_orientation(self):
        calibration = fixture_calibration(False)
        localization = fixture_localization()
        localization['geometry']['dimensions'] = {'observed_side_diameter_m': .075}
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480,
                                   'fx': 600., 'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        palm = np.eye(4)
        palm[:3, :3] = np.diag([-1., -1., 1.])
        # camera center [.1,.2,.5] -> base center [.2,-.1,.8]
        palm[:3, 3] = [.4, -.1, .6]
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        source = {'kind': 'can_feedback_read_only'}
        with self.assertRaisesRegex(ValueError, 'allow_provisional'):
            compute_side_grasp(localization, calibration, flange, metadata,
                               pose_source=source, now_ns=11_000_000_000)
        result = compute_side_grasp(localization, calibration, flange, metadata,
                                    allow_provisional=True, pose_source=source,
                                    now_ns=11_000_000_000)
        self.assertTrue(result['checks']['execute_ready'])
        self.assertAlmostEqual(result['checks']['palm_x_toward_cup_dot'], 1.)
        self.assertEqual(set(result['waypoints']), {'pregrasp', 'contact', 'lift'})
        center = np.array(result['cup_base']['center_m'])
        contact = np.array(result['waypoints']['contact']['palm_pose_base_m_rad'][:3])
        pregrasp = np.array(result['waypoints']['pregrasp']['palm_pose_base_m_rad'][:3])
        lift = np.array(result['waypoints']['lift']['palm_pose_base_m_rad'][:3])
        np.testing.assert_allclose(contact, center + [.075/2+.07, 0, .025], atol=1e-12)
        np.testing.assert_allclose(pregrasp-contact, [.04, 0, .03], atol=1e-12)
        np.testing.assert_allclose(lift-contact, [0, 0, .05], atol=1e-12)
        for waypoint in result['waypoints'].values():
            np.testing.assert_allclose(
                np.asarray(waypoint['T_base_flange']) @ np.asarray(calibration['T_flange_tcp']),
                waypoint['T_base_palm'], atol=1e-12)
            np.testing.assert_allclose(np.asarray(waypoint['T_base_palm'])[:3, :3],
                                       palm[:3, :3], atol=1e-12)

    def test_old_snapshot_or_manual_pose_cannot_report_execute_ready(self):
        calibration = fixture_calibration(True)
        localization = fixture_localization()
        localization['geometry']['dimensions'] = {'observed_side_diameter_m': .075}
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480,
                                   'fx': 600., 'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        palm = np.eye(4)
        palm[:3, :3] = np.diag([-1., -1., 1.])
        palm[:3, 3] = [.4, -.1, .6]
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        result = compute_side_grasp(localization, calibration, flange, metadata,
                                    pose_source={'kind': 'manual_offline'},
                                    now_ns=11_000_000_000)
        self.assertFalse(result['checks']['execute_ready'])
        old = compute_side_grasp(localization, calibration, flange, metadata,
                                 pose_source={'kind': 'can_feedback_read_only'},
                                 now_ns=31_000_000_000)
        self.assertFalse(old['checks']['snapshot_fresh'])
        self.assertFalse(old['checks']['execute_ready'])

    def test_supervised_facing_override_remains_explicit(self):
        calibration = fixture_calibration(True)
        localization = fixture_localization()
        localization['geometry']['dimensions'] = {'observed_side_diameter_m': .075}
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480, 'fx': 600.,
                                   'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        angle = math.radians(47)
        yaw = np.array([[math.cos(angle), -math.sin(angle), 0.],
                        [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        palm = np.eye(4)
        palm[:3, :3] = yaw @ np.diag([-1., -1., 1.])
        palm[:3, 3] = [.4, -.1, .6]
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        source = {'kind': 'can_feedback_read_only'}
        default = compute_side_grasp(localization, calibration, flange, metadata,
                                     pose_source=source, now_ns=11_000_000_000)
        self.assertFalse(default['checks']['palm_facing_cup'])
        trial = compute_side_grasp(localization, calibration, flange, metadata,
                                   pose_source=source, now_ns=11_000_000_000,
                                   min_facing_dot=.65)
        self.assertTrue(trial['checks']['palm_facing_cup'])
        self.assertEqual(trial['checks']['minimum_palm_facing_dot'], .65)
        with self.assertRaisesRegex(ValueError, 'Minimum palm facing'):
            compute_side_grasp(localization, calibration, flange, metadata,
                               pose_source=source, now_ns=11_000_000_000,
                               min_facing_dot=.1)

    def test_opt_in_yaw_alignment_wraps_euler_and_requires_review(self):
        calibration = fixture_calibration(True)
        localization = fixture_localization()
        localization['geometry']['dimensions'] = {
            'observed_side_diameter_m': .075, 'observed_height_m': .08,
        }
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480, 'fx': 600.,
                                   'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        # Palm is on the outward bearing +10°, so desired inward is -170°.
        # Current palm +X at +170° needs only +20° across the Euler wrap.
        outward = np.array([math.cos(math.radians(10)), math.sin(math.radians(10)), 0.])
        palm = np.eye(4)
        yaw = math.radians(170)
        palm[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0.],
                        [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]]
        palm[:3, 3] = [.2, -.1, .8] + outward*.2
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        options = {'pose_source': {'kind': 'can_feedback_read_only'},
                   'now_ns': 11_000_000_000, 'align_palm_to_cup': True}
        unreviewed = compute_side_grasp(localization, calibration, flange, metadata,
                                         **options)
        self.assertFalse(unreviewed['checks']['execute_ready'])
        self.assertTrue(unreviewed['checks']['alignment_requires_ik_scene_review'])
        self.assertAlmostEqual(unreviewed['heading_correction_deg'], 20., places=8)
        self.assertAlmostEqual(unreviewed['checks']['target_palm_x_toward_cup_dot'], 1., places=8)
        self.assertGreater(unreviewed['checks']['target_palm_x_toward_cup_dot'],
                           unreviewed['checks']['palm_x_toward_cup_dot'])
        reviewed = compute_side_grasp(localization, calibration, flange, metadata,
                                       alignment_reviewed=True, **options)
        self.assertTrue(reviewed['checks']['execute_ready'])
        self.assertTrue(reviewed['checks']['alignment_reviewed'])
        self.assertTrue(reviewed['checks']['clearance_required'])
        clearance = reviewed['waypoints']['clearance']
        np.testing.assert_allclose(
            clearance['palm_pose_base_m_rad'][:3], palm[:3, 3] + [0., 0., .08], atol=1e-12)
        np.testing.assert_allclose(np.asarray(clearance['T_base_palm'])[:3, :3],
                                   palm[:3, :3], atol=1e-12)
        for stage in ('pregrasp', 'contact', 'lift'):
            waypoint = reviewed['waypoints'][stage]
            target_palm = np.asarray(waypoint['T_base_palm'])
            self.assertGreaterEqual(waypoint['palm_altitude_from_table_m'], 0.)
            np.testing.assert_allclose(
                np.asarray(waypoint['T_base_flange']) @ np.asarray(calibration['T_flange_tcp']),
                target_palm, atol=1e-12)
            # Translation/rotation can be recovered after the +/-pi Euler wrap.
            from nero_calibration.core import pose_matrix
            np.testing.assert_allclose(pose_matrix(waypoint['palm_pose_base_m_rad']),
                                       target_palm, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'requires'):
            compute_side_grasp(localization, calibration, flange, metadata,
                               alignment_reviewed=True, now_ns=11_000_000_000)

    def test_explicit_home_yaw_roll_preserves_other_targets_and_raises_only_pregrasp(self):
        calibration = fixture_calibration(True)
        localization = fixture_localization()
        localization['geometry']['dimensions'] = {
            'observed_side_diameter_m': .075, 'observed_height_m': .08,
        }
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480,
                                   'fx': 600., 'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        palm = np.eye(4)
        palm[:3, :3] = np.diag([-1., -1., 1.])
        palm[:3, 3] = [.4, -.1, .8]
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        home = {'confirmed': True,
                'provenance': {'kind': 'can_joint_feedback_read_only',
                               'host_feedback_received_time_ns': 11_000_000_000}}
        common = {'pose_source': {'kind': 'can_feedback_read_only'},
                  'now_ns': 11_000_000_000}
        baseline = compute_side_grasp(localization, calibration, flange, metadata,
                                      **common)
        candidate = compute_side_grasp(
            localization, calibration, flange, metadata,
            require_home=True, home_check=home,
            base_yaw_deg=32., palm_x_roll_deg=25., **common)
        self.assertFalse(candidate['checks']['execute_ready'])
        self.assertTrue(candidate['checks']['alignment_requires_ik_scene_review'])
        self.assertTrue(candidate['checks']['clearance_required'])
        reviewed = compute_side_grasp(
            localization, calibration, flange, metadata,
            require_home=True, home_check=home, alignment_reviewed=True,
            base_yaw_deg=32., palm_x_roll_deg=25., **common)
        self.assertTrue(reviewed['checks']['execute_ready'])
        self.assertTrue(reviewed['checks']['home_confirmed'])
        self.assertEqual(reviewed['orientation_provenance']['rotation_order'],
                         'Rz_base(yaw) @ R_HOME_palm @ Rx_palm(roll)')
        self.assertEqual(reviewed['orientation_provenance']['base_yaw_deg'], 32.)
        self.assertEqual(reviewed['orientation_provenance']['local_palm_x_roll_deg'], 25.)
        self.assertEqual(reviewed['orientation_provenance']['pregrasp_extra_base_z_m'], .04)
        self.assertAlmostEqual(reviewed['checks']['target_palm_x_toward_cup_dot'],
                               math.cos(math.radians(32.)), places=12)
        yaw, roll = math.radians(32), math.radians(25)
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.],
                       [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
        rx = np.array([[1., 0., 0.], [0., math.cos(roll), -math.sin(roll)],
                       [0., math.sin(roll), math.cos(roll)]])
        expected_rotation = rz @ palm[:3, :3] @ rx
        clearance = np.asarray(reviewed['waypoints']['clearance']['T_base_palm'])
        np.testing.assert_allclose(clearance[:3, 3], palm[:3, 3] + [0., 0., .08], atol=1e-12)
        np.testing.assert_allclose(clearance[:3, :3], palm[:3, :3], atol=1e-12)
        for stage in ('pregrasp', 'contact', 'lift'):
            waypoint = reviewed['waypoints'][stage]
            target_palm = np.asarray(waypoint['T_base_palm'])
            np.testing.assert_allclose(target_palm[:3, :3], expected_rotation, atol=1e-12)
            np.testing.assert_allclose(
                np.asarray(waypoint['T_base_flange'])
                @ np.asarray(calibration['T_flange_tcp']), target_palm, atol=1e-12)
            baseline_position = np.asarray(baseline['waypoints'][stage]['T_base_palm'])[:3, 3]
            expected_delta = [0., 0., .04] if stage == 'pregrasp' else [0., 0., 0.]
            np.testing.assert_allclose(target_palm[:3, 3]-baseline_position,
                                       expected_delta, atol=1e-12)

    def test_explicit_home_orientation_requires_paired_bounded_exclusive_inputs(self):
        calibration = fixture_calibration(True)
        localization = fixture_localization()
        localization['geometry']['dimensions'] = {'observed_side_diameter_m': .075}
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480,
                                   'fx': 600., 'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        palm = np.eye(4)
        palm[:3, :3] = np.diag([-1., -1., 1.])
        palm[:3, 3] = [.4, -.1, .8]
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        common = {'now_ns': 11_000_000_000, 'require_home': True}
        with self.assertRaisesRegex(ValueError, 'provided together'):
            compute_side_grasp(localization, calibration, flange, metadata,
                               base_yaw_deg=32., **common)
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            compute_side_grasp(localization, calibration, flange, metadata,
                               align_palm_to_cup=True, base_yaw_deg=32.,
                               palm_x_roll_deg=25., **common)
        with self.assertRaisesRegex(ValueError, 'requires --require-home'):
            compute_side_grasp(localization, calibration, flange, metadata,
                               base_yaw_deg=32., palm_x_roll_deg=25.,
                               now_ns=11_000_000_000)
        for yaw, roll in ((46., 25.), (32., 36.), (float('nan'), 25.)):
            with self.assertRaisesRegex(ValueError, 'finite within'):
                compute_side_grasp(localization, calibration, flange, metadata,
                                   base_yaw_deg=yaw, palm_x_roll_deg=roll,
                                   **common)
        no_home = compute_side_grasp(localization, calibration, flange, metadata,
                                     base_yaw_deg=32., palm_x_roll_deg=25.,
                                     alignment_reviewed=True,
                                     pose_source={'kind': 'can_feedback_read_only'}, **common)
        self.assertFalse(no_home['checks']['execute_ready'])
        self.assertFalse(no_home['orientation_provenance']['home_reference_verified'])

    def test_yaw_alignment_rejects_large_unreviewed_reorientation(self):
        calibration = fixture_calibration(True)
        localization = fixture_localization()
        localization['geometry']['dimensions'] = {
            'observed_side_diameter_m': .075, 'observed_height_m': .08,
        }
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480, 'fx': 600.,
                                   'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        palm = np.eye(4)
        palm[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        palm[:3, 3] = [.4, -.1, .6]  # inward target is base -X; +X is base +Y.
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        with self.assertRaisesRegex(ValueError, '70°'):
            compute_side_grasp(localization, calibration, flange, metadata,
                               align_palm_to_cup=True, now_ns=11_000_000_000)

    def test_yaw_alignment_preserves_rigid_tilt_and_input_pose(self):
        calibration = fixture_calibration(True)
        localization = fixture_localization()
        localization['geometry']['dimensions'] = {
            'observed_side_diameter_m': .075, 'observed_height_m': .08,
        }
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480, 'fx': 600.,
                                   'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        pitch = math.radians(20)
        cy, sy = math.cos(math.radians(135)), math.sin(math.radians(135))
        cp, sp = math.cos(pitch), math.sin(pitch)
        yaw = np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
        tilt = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
        palm = np.eye(4)
        palm[:3, :3] = tilt @ yaw
        palm[:3, 3] = [.4, -.1, .8]
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        flange_before = flange.copy()
        result = compute_side_grasp(localization, calibration, flange, metadata,
                                    align_palm_to_cup=True, now_ns=11_000_000_000)
        np.testing.assert_allclose(flange, flange_before, atol=0)
        for waypoint in result['waypoints'].values():
            target = np.asarray(waypoint['T_base_palm'])
            np.testing.assert_allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-12)
            # Base +Z yaw rotates the palm heading while retaining its +Z tilt.
            self.assertAlmostEqual(target[2, 2], palm[2, 2], places=12)

    def test_nominal_height_only_moves_waypoint_center_with_provenance(self):
        calibration = fixture_calibration(True)
        localization = fixture_localization()
        localization['geometry']['dimensions'] = {
            'observed_side_diameter_m': .075, 'observed_height_m': .08,
        }
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480, 'fx': 600.,
                                   'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        palm = np.eye(4)
        palm[:3, :3] = np.diag([-1., -1., 1.])
        palm[:3, 3] = [.4, -.1, .6]
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        observed = compute_side_grasp(localization, calibration, flange, metadata,
                                      now_ns=11_000_000_000)
        nominal = compute_side_grasp(localization, calibration, flange, metadata,
                                     now_ns=11_000_000_000, nominal_cup_height_m=.072)
        expected_center = (np.asarray(nominal['cup_base']['support_center_m'])
                           + np.asarray(nominal['cup_base']['axis'])*.036)
        np.testing.assert_allclose(nominal['planning_cup_center_base_m'], expected_center,
                                   atol=1e-12)
        np.testing.assert_allclose(nominal['cup_base']['center_m'],
                                   observed['cup_base']['center_m'], atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(nominal['waypoints']['contact']['palm_pose_base_m_rad'][:3])
            - np.asarray(observed['waypoints']['contact']['palm_pose_base_m_rad'][:3]),
            [0., 0., -.004], atol=1e-12)
        self.assertEqual(nominal['center_provenance']['source'],
                         'support_plus_explicit_nominal_height_half')
        self.assertAlmostEqual(nominal['center_provenance']['nominal_minus_observed_height_m'],
                               -.008)
        self.assertFalse(nominal['center_provenance']['geometry_hardware_validated'])

    def test_base_z_axis_assumption_preserves_measurement_and_needs_both_reviews(self):
        calibration = fixture_calibration(True)
        localization = fixture_localization()
        measured_camera_axis = np.array([-.4, .3, math.sqrt(.75)])
        support_camera = np.array([.1, .2, .46])
        localization['geometry']['axis'] = measured_camera_axis.tolist()
        localization['geometry']['support_center_m'] = support_camera.tolist()
        localization['geometry']['center_m'] = (
            support_camera + measured_camera_axis*.04).tolist()
        localization['geometry']['dimensions'] = {
            'observed_side_diameter_m': .075, 'observed_height_m': .08,
        }
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480, 'fx': 600.,
                                   'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        palm = np.eye(4)
        palm[:3, :3] = np.diag([-1., -1., 1.])
        palm[:3, 3] = [.4, -.1, .8]
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        common = {'pose_source': {'kind': 'can_feedback_read_only'},
                  'now_ns': 11_000_000_000, 'nominal_cup_height_m': .072}
        normal = compute_side_grasp(localization, calibration, flange, metadata,
                                    **common)
        self.assertFalse(normal['checks']['axis_upright'])
        self.assertFalse(normal['checks']['execute_ready'])
        self.assertNotIn('clearance', normal['waypoints'])
        provisional = compute_side_grasp(
            localization, calibration, flange, metadata,
            assume_base_z_upright=True, align_palm_to_cup=True, **common)
        self.assertFalse(provisional['checks']['execute_ready'])
        self.assertTrue(provisional['checks']['axis_override_requires_review'])
        axis_only = compute_side_grasp(
            localization, calibration, flange, metadata,
            assume_base_z_upright=True, axis_reviewed=True,
            align_palm_to_cup=True, **common)
        self.assertFalse(axis_only['checks']['execute_ready'])
        reviewed = compute_side_grasp(
            localization, calibration, flange, metadata,
            assume_base_z_upright=True, axis_reviewed=True,
            align_palm_to_cup=True, alignment_reviewed=True, **common)
        self.assertTrue(reviewed['checks']['execute_ready'])
        self.assertFalse(reviewed['checks']['measured_axis_upright'])
        self.assertTrue(reviewed['checks']['planning_axis_upright'])
        self.assertAlmostEqual(reviewed['checks']['measured_axis_base_z_dot'],
                               math.sqrt(.75))
        self.assertAlmostEqual(reviewed['checks']['planning_axis_base_z_dot'], 1.)
        np.testing.assert_allclose(reviewed['cup_base']['axis'],
                                   normal['cup_base']['axis'], atol=1e-12)
        np.testing.assert_allclose(reviewed['planning_cup_axis_base'], [0., 0., 1.],
                                   atol=1e-12)
        np.testing.assert_allclose(
            reviewed['planning_cup_center_base_m'],
            np.asarray(reviewed['cup_base']['support_center_m']) + [0., 0., .036],
            atol=1e-12)
        self.assertEqual(reviewed['axis_provenance']['source'],
                         'explicit_base_z_upright_assumption')
        self.assertFalse(reviewed['axis_provenance']['mount_and_table_orientation_verified'])
        self.assertIn('clearance', reviewed['waypoints'])
        with self.assertRaisesRegex(ValueError, 'nominal-cup-height'):
            compute_side_grasp(localization, calibration, flange, metadata,
                               assume_base_z_upright=True, now_ns=11_000_000_000)
        with self.assertRaisesRegex(ValueError, 'axis-reviewed requires'):
            compute_side_grasp(localization, calibration, flange, metadata,
                               axis_reviewed=True, now_ns=11_000_000_000)

    def test_base_z_axis_assumption_rejects_large_mount_discrepancy(self):
        calibration = fixture_calibration(True)
        localization = fixture_localization()
        localization['geometry']['axis'] = [0., .8, .6]
        localization['geometry']['dimensions'] = {
            'observed_side_diameter_m': .075, 'observed_height_m': .08,
        }
        metadata = {'serial': calibration['camera']['serial'],
                    'intrinsics': {'width': 640, 'height': 480, 'fx': 600.,
                                   'fy': 600., 'cx': 320., 'cy': 240.,
                                   'dist_coeffs': [0.]*5},
                    'host_capture_time_ns': 10_000_000_000,
                    'frame_id': 'fake:42'}
        palm = np.eye(4)
        palm[:3, :3] = np.diag([-1., -1., 1.])
        palm[:3, 3] = [.4, -.1, .8]
        flange = matrix(palm @ inverse(calibration['T_flange_tcp']))
        with self.assertRaisesRegex(ValueError, '45°'):
            compute_side_grasp(localization, calibration, flange, metadata,
                               assume_base_z_upright=True, nominal_cup_height_m=.072,
                               now_ns=11_000_000_000)


if __name__ == '__main__':
    unittest.main()
