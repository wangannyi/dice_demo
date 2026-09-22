import copy
from pathlib import Path
import unittest

import cv2
import numpy as np

from rgb_hand_tracking.marker_registration import (
    DATASET_SCHEMA, POSE_SCHEMA, RECOVERY_SCHEMA, RESULT_SCHEMA, _core,
    ippe_square_candidates, recover_table_board, solve_registration,
)


MARKER = {'dictionary': 'DICT_4X4_50', 'marker_id': 40, 'marker_length_m': .03}


def synthetic_report(transform, source):
    alternative = transform @ _core().pose_matrix([.01, .02, 0, .01, .02, 0])
    return {'schema': POSE_SCHEMA, **MARKER, 'source': source,
            'candidates': [
                {'candidate_index': i, 'T_camera_hand_marker': t.tolist(),
                 'all_corners_positive_depth': True, 'reprojection_rms_px': i+.1}
                for i, t in enumerate((transform, alternative))]}


def synthetic_dataset(count=24, moving_camera=True):
    core = _core()
    rng = np.random.default_rng(821)
    bb = core.pose_matrix([.42, -.18, .1, .05, .02, .08])
    fm = core.pose_matrix([.02, .03, .11, -.4, .2, .1])
    samples = []
    for i in range(count):
        bf = core.pose_matrix([*rng.uniform(-.3, .3, 2), rng.uniform(.3, .5),
                               *rng.uniform(-.8, .8, 3)])
        camera_jitter = rng.uniform(-.2, .2, 6) if moving_camera else np.zeros(6)
        bc = core.pose_matrix(np.array([0, 0, 1.2, np.pi, 0, 0])+camera_jitter)
        cb = core.inverse(bc) @ bb
        cm = core.inverse(bc) @ bf @ fm
        source = {'frame_serial': i+1, 'timestamp_s': i+1.,
                  'camera_epoch': f'camera_position_{i}' if moving_camera else 'fixed'}
        samples.append({'T_base_flange': bf.tolist(), 'T_camera_table_board': cb.tolist(),
                        'source': source, 'stationary': True,
                        'hand_marker_pose_candidates': synthetic_report(cm, source),
                        'selected_marker_pose_index': 0})
    return bb, fm, {'schema': DATASET_SCHEMA, 'table_board_frame_id': 'pinned_board_1',
                    'marker_attachment_epoch': 'hand_marker_20260916_03',
                    'table_board_fixed_during_collection': True,
                    'hand_marker': MARKER.copy(), 'samples': samples}


class MarkerRegistrationTests(unittest.TestCase):
    def test_moving_camera_cancels_and_existing_gates_are_preserved(self):
        for moving in (False, True):
            bb, fm, dataset = synthetic_dataset(moving_camera=moving)
            result = solve_registration(dataset)
            self.assertEqual(result['schema'], RESULT_SCHEMA)
            self.assertTrue(result['quality_passed'])
            np.testing.assert_allclose(result['T_base_board'], bb, atol=1e-8)
            np.testing.assert_allclose(result['T_flange_marker'], fm, atol=1e-8)
            self.assertEqual(result['holdout_indices'], [3, 7, 11, 15, 19, 23])
            self.assertEqual(result['thresholds'], {'position_m': .005, 'angle_deg': 2.})
            self.assertEqual(result['distinct_pose_count'], 24)
            self.assertIsNone(result['T_marker_contact'])
            self.assertIsNone(result['physical_palm_m'])
            self.assertFalse(result['physical_branch_verified'])
            self.assertFalse(result['independent_validation_passed'])
            self.assertFalse(result['motion_target_valid'])

    def test_quality_failure_from_held_out_pose_is_retained(self):
        _, _, dataset = synthetic_dataset()
        candidate = dataset['samples'][3]['hand_marker_pose_candidates']['candidates'][0]
        candidate['T_camera_hand_marker'][0][3] += .1
        result = solve_registration(dataset)
        self.assertFalse(result['quality_passed'])
        self.assertTrue(any(p > .005 for p, _ in result['holdout_errors_m_deg']))
        self.assertFalse(result['execution_enabled'])

    def test_single_home_and_too_few_poses_are_rejected(self):
        _, _, dataset = synthetic_dataset(11)
        with self.assertRaisesRegex(ValueError, '12 distinct stationary poses'):
            solve_registration(dataset)
        _, _, dataset = synthetic_dataset()
        for sample in dataset['samples']:
            sample['T_base_flange'] = dataset['samples'][0]['T_base_flange']
        with self.assertRaisesRegex(ValueError, 'Duplicate stationary pose'):
            solve_registration(dataset)

    def test_holdout_cannot_copy_training_pose_even_with_many_distinct_poses(self):
        bb, fm, dataset = synthetic_dataset()
        core = _core()
        # Index 3 is held out; index 0 is training. Preserve valid geometry so
        # residual gates alone would accept this leaked validation pose.
        duplicate = dataset['samples'][3]
        duplicate['T_base_flange'] = dataset['samples'][0]['T_base_flange']
        cm = (core.matrix(duplicate['T_camera_table_board']) @ core.inverse(bb)
              @ core.matrix(duplicate['T_base_flange']) @ fm)
        duplicate['hand_marker_pose_candidates'] = synthetic_report(cm, duplicate['source'])
        adapted = []
        for sample in dataset['samples']:
            cm = core.matrix(sample['hand_marker_pose_candidates']['candidates'][0]
                             ['T_camera_hand_marker'])
            adapted.append({'T_base_flange': sample['T_base_flange'],
                            'T_camera_board': (core.inverse(core.matrix(
                                sample['T_camera_table_board'])) @ cm).tolist()})
        self.assertTrue(core.solve(adapted, np.eye(4))['quality_passed'])
        with self.assertRaisesRegex(ValueError, 'Duplicate stationary pose'):
            solve_registration(dataset)

    def test_single_axis_rotation_is_rejected_by_unchanged_solver(self):
        _, _, dataset = synthetic_dataset()
        for i, sample in enumerate(dataset['samples']):
            sample['T_base_flange'] = _core().pose_matrix([i*.01, 0, .3, 0, 0, i*.1]).tolist()
        with self.assertRaisesRegex(ValueError, 'at least two axes'):
            solve_registration(dataset)

    def test_unknown_candidate_never_defaults_to_lower_reprojection_branch(self):
        _, _, dataset = synthetic_dataset()
        del dataset['samples'][0]['selected_marker_pose_index']
        with self.assertRaisesRegex(ValueError, 'Explicit selected_marker_pose_index'):
            solve_registration(dataset)
        for invalid in (True, -1, 2, 0., '0'):
            dataset['samples'][0]['selected_marker_pose_index'] = invalid
            with self.assertRaisesRegex(ValueError, 'Explicit selected_marker_pose_index'):
                solve_registration(dataset)

    def test_explicit_second_candidate_is_honored(self):
        bb, fm, dataset = synthetic_dataset()
        for sample in dataset['samples']:
            report = sample['hand_marker_pose_candidates']
            first, second = report['candidates']
            first['T_camera_hand_marker'], second['T_camera_hand_marker'] = (
                second['T_camera_hand_marker'], first['T_camera_hand_marker'])
            sample['selected_marker_pose_index'] = 1
        result = solve_registration(dataset)
        self.assertTrue(result['quality_passed'])
        self.assertEqual(result['selected_marker_pose_indices'], [1]*24)
        np.testing.assert_allclose(result['T_base_board'], bb, atol=1e-8)
        np.testing.assert_allclose(result['T_flange_marker'], fm, atol=1e-8)

    def test_legacy_schema_cannot_be_mistaken_for_marker_registration(self):
        _, _, dataset = synthetic_dataset()
        dataset['schema'] = 1
        with self.assertRaisesRegex(ValueError, 'explicit RGB hand-marker'):
            solve_registration(dataset)
        dataset['schema'] = DATASET_SCHEMA
        dataset['samples'][0]['hand_marker_pose_candidates']['schema'] = 1
        with self.assertRaisesRegex(ValueError, 'explicit hand-marker candidate schema'):
            solve_registration(dataset)

    def test_source_identity_stationarity_and_fixed_board_contract(self):
        _, _, original = synthetic_dataset()
        mutations = [
            lambda d: d.update(table_board_fixed_during_collection=False),
            lambda d: d['samples'][0].update(stationary=False),
            lambda d: d['samples'][0]['hand_marker_pose_candidates'].update(
                source={**d['samples'][0]['source'], 'frame_serial': 999}),
            lambda d: d['samples'][1].update(source=d['samples'][0]['source']),
            lambda d: d['samples'][1]['source'].update(timestamp_s=0.),
            lambda d: d['samples'][0]['hand_marker_pose_candidates']['candidates'][0].update(
                all_corners_positive_depth=False),
            lambda d: d['samples'][0]['hand_marker_pose_candidates'].update(marker_length_m=.031),
        ]
        for mutate in mutations:
            dataset = copy.deepcopy(original)
            mutate(dataset)
            with self.assertRaises(ValueError):
                solve_registration(dataset)

    def test_moved_board_recovery_requires_independent_verification(self):
        _, fm, dataset = synthetic_dataset()
        registration = solve_registration(dataset)
        new_board = _core().pose_matrix([-.1, .22, .04, -.04, .06, -.3])
        bf = _core().matrix(dataset['samples'][0]['T_base_flange'])
        new_camera = _core().pose_matrix([.04, .01, 1.1, np.pi+.1, .05, -.04])
        source = {'frame_serial': 501, 'timestamp_s': 50., 'camera_epoch': 'new_camera'}
        cm = _core().inverse(new_camera) @ bf @ fm
        inputs = dict(T_base_flange=bf, T_camera_table_board=_core().inverse(new_camera) @ new_board,
                      hand_marker_pose_candidates=synthetic_report(cm, source),
                      selected_marker_pose_index=0, source=source, stationary=True,
                      marker_attachment_epoch=dataset['marker_attachment_epoch'],
                      table_board_frame_id='repositioned_board_2')
        result = recover_table_board(registration, **inputs)
        self.assertEqual(result['schema'], RECOVERY_SCHEMA)
        np.testing.assert_allclose(result['T_base_board'], new_board, atol=1e-8)
        self.assertEqual(result['status'], 'requires_independent_validation')
        self.assertFalse(result['independent_validation_passed'])
        self.assertFalse(result['motion_target_valid'])
        self.assertFalse(result['execution_enabled'])
        self.assertIsNone(result['T_marker_contact'])
        self.assertIsNone(result['physical_palm_m'])
        for change in ({'marker_attachment_epoch': 'new_attachment'},
                       {'stationary': False}, {'selected_marker_pose_index': None}):
            with self.assertRaises(ValueError):
                recover_table_board(registration, **{**inputs, **change})
        registration['quality_passed'] = False
        with self.assertRaisesRegex(ValueError, 'quality-passed'):
            recover_table_board(registration, **inputs)

    def test_ippe_preserves_two_branches_raw_pixels_and_actual_residuals(self):
        K = np.array([[760., 0, 631.], [0, 765., 358.], [0, 0, 1]])
        D = np.array([.13, -.12, -.003, -.003, 0])
        objects = np.array([[-.015, .015, 0], [.015, .015, 0],
                            [.015, -.015, 0], [-.015, -.015, 0]])
        true_pose = _core().pose_matrix([.12, -.04, .55, np.pi-.25, .15, -.1])
        rvec = cv2.Rodrigues(true_pose[:3, :3])[0]
        corners = cv2.projectPoints(objects, rvec, true_pose[:3, 3], K, D)[0].reshape(4, 2)
        report = ippe_square_candidates(corners, K, D)
        self.assertEqual(report['schema'], POSE_SCHEMA)
        self.assertIsNone(report['selected_candidate_index'])
        self.assertFalse(report['physical_branch_verified'])
        np.testing.assert_array_equal(report['raw_corners_px'], corners)
        self.assertEqual(len(report['candidates']), 2)
        self.assertTrue(all(c['all_corners_positive_depth'] for c in report['candidates']))
        self.assertTrue(all(c['printed_face_toward_camera'] for c in report['candidates']))
        best = min(report['candidates'], key=lambda c: c['reprojection_rms_px'])
        np.testing.assert_allclose(best['T_camera_hand_marker'], true_pose, atol=1e-7)
        for candidate in report['candidates']:
            errors = np.linalg.norm(np.asarray(candidate['reprojected_corners_px'])-corners, axis=1)
            np.testing.assert_allclose(candidate['corner_reprojection_errors_px'], errors)
            self.assertAlmostEqual(candidate['reprojection_rms_px'], np.sqrt(np.mean(errors**2)))

    def test_ippe_rejects_degenerate_geometry_and_bad_intrinsics(self):
        corners = [[0., 0.], [30., 0.], [30., 30.], [0., 30.]]
        K = np.array([[760., 0, 631.], [0, 765., 358.], [0, 0, 1]])
        for changes in ({'raw_corners_px': [[0., 0.]]*4},
                        {'raw_corners_px': [[0., 0.], [30., 30.], [30., 0.], [0., 30.]]},
                        {'camera_matrix': np.eye(4)}, {'distortion_coefficients': [0]*4},
                        {'marker_length_m': True}, {'marker_id': 50}):
            args = {'raw_corners_px': corners, 'camera_matrix': K,
                    'distortion_coefficients': np.zeros(5), **changes}
            with self.assertRaises(ValueError):
                ippe_square_candidates(**args)

    def test_solver_core_source_remains_the_original_dependency(self):
        self.assertEqual(Path(_core().__file__).name, 'core.py')
        self.assertEqual(Path(_core().__file__).parent.name, 'nero_calibration')


if __name__ == '__main__':
    unittest.main()
