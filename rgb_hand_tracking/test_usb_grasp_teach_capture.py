"""Teaching pixels stay separate from valid physical calibration/motion."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

import usb_grasp_teach_capture as teach
from test_usb_marker_registration_capture import Board, Camera, Geometry, PoseClient


class TeachGeometry(Geometry):
    def _board_pose(self, observation):
        pose, rotation, translation = super()._board_pose(observation)
        pose['valid'] = True
        return pose, rotation, translation

    def observe(self, board, cup):
        return {'cup_top': {'valid': False}}


class Cup:
    def process(self, frame):
        return {'valid': False}


class Tests(unittest.TestCase):
    def sample(self, pose_modify=None, marker_modify=None, stats=None):
        pose = PoseClient(pose_modify)
        camera = Camera(pose, marker_modify)
        if stats:
            camera.stats.update(stats)
        return teach.capture_reference(camera, pose, Board(), TeachGeometry(), Cup(),
                                       'stream_1', clock=lambda: 0., sleep=lambda _: None)

    def test_twelve_same_frame_images_bracketed_by_zero_tx_stationary_fk(self):
        sample, frames = self.sample()
        self.assertEqual(len(frames), 12)
        self.assertEqual(sample['stationarity']['snapshot_count'], 32)
        self.assertIsNone(sample['T_marker_contact'])
        self.assertIsNone(sample['selected_marker_pose_index'])
        self.assertFalse(sample['physical_branch_verified'])
        self.assertFalse(sample['motion_target_valid'])
        for row in sample['repeated_rgb_observations']:
            self.assertEqual(row['source'], row['hand_marker_pose_candidates']['source'])
            self.assertLess(row['software_bracket']['before']['request_end_monotonic_s'],
                            row['source']['timestamp_s'])
            self.assertLess(row['source']['timestamp_s'],
                            row['software_bracket']['after']['request_start_monotonic_s'])

    def test_unstable_3d_is_recorded_without_making_calibration_valid(self):
        original = teach.registration.ippe_square_candidates
        calls = []

        def candidate(*args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(result)
            if len(calls) == 2:
                result['candidates'][1]['T_camera_hand_marker'][0][3] += .004
            return result

        with patch.object(teach.registration, 'ippe_square_candidates', candidate):
            sample, _ = self.sample()
        self.assertGreater(sample['raw_ippe_repeatability'][1]['maximum_pairwise_position_m'], .003)
        self.assertFalse(sample['physical_branch_verified'])
        self.assertFalse(sample['motion_target_valid'])

    def test_marker_source_mismatch_loss_and_generation_change_reject(self):
        for mutate in (lambda m, f: m.update(frame_serial=123),
                       lambda m, f: m.update(observation_valid=False),
                       lambda m, f: m.update(camera_epoch='other'),
                       lambda m, f: m.update(tracking_generation=int(f[0, 0, 0]))):
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.sample(marker_modify=mutate)
        with self.assertRaises(ValueError):
            self.sample(stats={'tracking_generation': 2})

    def test_motion_and_stale_feedback_reject(self):
        def move(row):
            if row['snapshot_index'] >= 10:
                row['fk_flange_pose_m_rad'][0] = .0006
        with self.assertRaises(ValueError):
            self.sample(pose_modify=move)
        with self.assertRaises(ValueError):
            self.sample(pose_modify=lambda row: row['packet_ages_s'].update(joint_7=.101))

    def test_pixel_corner_jump_rejects_not_smoothed_away(self):
        def move(marker, frame):
            if int(frame[0, 0, 0]) == 4:
                marker['raw_corners_px'] = (np.array(marker['raw_corners_px'])+2).tolist()
        with self.assertRaisesRegex(ValueError, 'corners'):
            self.sample(marker_modify=move)

    def test_unique_rim_point_uses_top_plane_circle_and_minimum_image_y(self):
        T = np.eye(4)
        T[2, 3] = .6
        cup = {'valid': True, 'center_board_m': [0., 0., -.0635], 'radius_m': .025}
        point = teach.cup_rim_target(cup, T, Geometry.camera_matrix, Geometry.distortion)
        np.testing.assert_allclose(point['contact_point_board_candidate_m'], [0., -.025, -.0635],
                                   atol=1e-8)
        self.assertLess(point['contact_point_px'][1], 60)
        self.assertFalse(point['motion_target_valid'])
        self.assertFalse(point['independent_metric_accuracy_validated'])

    def test_invalid_or_body_only_cup_and_bad_height_geometry_reject(self):
        valid = {'valid': True, 'center_board_m': [0., 0., -.0635], 'radius_m': .025}
        T = np.eye(4)
        T[2, 3] = .6
        for mutate in (lambda c: c.update(valid=False), lambda c: c.update(radius_m=True),
                       lambda c: c.update(center_board_m=[float('nan'), 0, 0]),
                       lambda c: c.pop('center_board_m')):
            cup = deepcopy(valid)
            mutate(cup)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                teach.cup_rim_target(cup, T, Geometry.camera_matrix, Geometry.distortion)

    def test_cup_motion_rejects_and_occlusion_never_verifies_fixed_cup(self):
        target = {'cup_center_board_m': [0, 0, -.0635], 'fitted_radius_m': .025}
        metric = {'valid': True, 'center_board_m': [0, 0, -.0635], 'radius_m': .025}
        rows = [{'cup_metric': deepcopy(metric)} for _ in range(6)]
        self.assertTrue(teach.cup_fixed_verification(rows, target)['verified'])
        rows[-1]['cup_metric']['center_board_m'][0] = .003
        with self.assertRaisesRegex(ValueError, 'Cup moved'):
            teach.cup_fixed_verification(rows, target)
        self.assertFalse(teach.cup_fixed_verification([{'cup_metric': {'valid': False}}],
                                                     target)['verified'])
        self.assertFalse(teach.cup_fixed_verification(rows[:2], target)['verified'])

    def test_occluded_relocated_cup_saves_pose_without_reusing_old_rim(self):
        sample, frames = self.sample()
        fields = teach.grasp_pose_fields(sample, Geometry.camera_matrix, Geometry.distortion)
        self.assertEqual(len(frames), 12)
        self.assertTrue(fields['pose_record_valid'])
        self.assertTrue(fields['operator_pregrasp_pose_confirmed'])
        self.assertFalse(fields['cup_reference_valid'])
        self.assertIsNone(fields['rim_target'])
        self.assertFalse(fields['previous_cup_target_reused'])
        self.assertFalse(fields['physical_contact_verified'])
        self.assertFalse(fields['reference_activation_valid'])
        self.assertIsNone(fields['T_marker_contact'])
        self.assertNotIn('operator_contact_placement_confirmed', fields)

    def test_new_visible_cup_baseline_comes_only_from_current_rows(self):
        sample, _ = self.sample()
        for row in sample['repeated_rgb_observations']:
            row['cup_metric'] = {'valid': True, 'center_board_m': [.1, 0, -.0635], 'radius_m': .025}
        result = teach.grasp_pose_fields(sample, Geometry.camera_matrix, Geometry.distortion)
        self.assertTrue(result['cup_reference_valid'])
        self.assertEqual(result['rim_target']['cup_center_board_m'], [.1, 0, -.0635])
        self.assertFalse(result['rim_target_is_physical_contact_goal'])
        self.assertFalse(result['motion_target_valid'])
        self.assertFalse(result['physical_contact_verified'])

    def test_unstable_current_cup_does_not_discard_stopped_pose(self):
        sample, _ = self.sample()
        for row in sample['repeated_rgb_observations']:
            row['cup_metric'] = {'valid': True, 'center_board_m': [0, 0, -.0635], 'radius_m': .025}
        sample['repeated_rgb_observations'][-1]['cup_metric']['center_board_m'][0] = .003
        fields = teach.grasp_pose_fields(sample, Geometry.camera_matrix, Geometry.distortion)
        self.assertTrue(fields['pose_record_valid'])
        self.assertFalse(fields['cup_reference_valid'])
        self.assertIsNone(fields['rim_target'])
        self.assertEqual(fields['current_cup_reference']['reason'], 'unstable_current_cup_top_geometry')


if __name__ == '__main__':
    unittest.main()
