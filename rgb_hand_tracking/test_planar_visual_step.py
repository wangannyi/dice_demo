from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from planar_visual_step import (DETECTOR, build_calibration, create_calibration,
                                main, propose_axis_step)


def evidence(axis, before_tag, after_tag, before_pose, after_pose):
    epoch = 'optical-epoch:cartesian_probe:'+axis
    target = [.004, 0., 0., 0., 0., 0., 0.] if axis == 'x' else [.004, .004, 0., 0., 0., 0., 0.]
    feedback = {'q_rad': target, 'fk_flange_pose_m_rad': after_pose,
                'status': {'arm_status': 0, 'ctrl_mode': 1, 'motion_status': 0},
                'enabled': [True]*7}
    sample = {'feedback': feedback, 'verification': {'reached': True, 'checks': {'passed': True}}}
    sdk = {'success': True, 'event': 'cartesian_microprobe_reached',
           'requested_probe': {'axis': axis, 'distance_m': .001},
           'proposal': {'target_position_m': after_pose[:3]}, 'q_target_rad': target,
           'baseline_feedback': [{'fk_flange_pose_m_rad': before_pose}],
           'target_verification': {'fresh_stable_samples': 10, 'samples': [deepcopy(sample) for _ in range(10)]},
           'move_j_sent_monotonic_s': 2., 'strict_settled_monotonic_s': 2.5}
    rows = []
    for index in range(20):
        center = np.array(before_tag if index < 10 else after_tag, float)
        corners = (center+np.array([[-15., -15.], [15., -15.], [15., 15.], [-15., 15.]])).tolist()
        timestamp = 1.+index*.01 if index < 10 else 3.+(index-10)*.01
        marker = {'raw_corners_px': corners, 'camera_epoch': epoch, 'tracking_generation': 0,
                  'frame_serial': index+1, 'timestamp_s': timestamp, 'observation_valid': True}
        rows.append({'timestamp_s': timestamp, 'frame_serial': index+1, 'camera_epoch': epoch,
                     'tracking_generation': 0, 'read_age_s': .01, 'marker': marker,
                     'same_frame_current_tag_board_cup_valid': True,
                     'cup_top': {'valid': True, 'center_px': [300., 400.]},
                     'board': {'valid': True, 'charuco_corner_ids': list(range(12)),
                               'charuco_corners_px': [[800.+i*5, 600.] for i in range(12)]}})
    return {'success': True, 'axis': axis, 'marker_corner_detector': DETECTOR,
            'intrinsics_sha256': 'a'*64, 'rows': rows}, sdk


def pair():
    x = evidence('x', [500., 200.], [500., 198.8], [0.]*6, [.001, 0., 0., 0., 0., 0.])
    y = evidence('y', [500., 198.8], [498.7, 198.8], [.001, 0., 0., 0., 0., 0.],
                 [.001, .001, 0., 0., 0., 0.])
    return [*x, *y]


class PlanarVisualStepTests(unittest.TestCase):
    def proposal(self, calibration=None, current=None, target=None, **kwargs):
        c = calibration or build_calibration(*pair())
        arguments = {'timestamp_s': 100., 'now_s': 100.05, 'camera_epoch': c['camera_epoch'],
                     'intrinsics_sha256': c['intrinsics_sha256'], 'tracking_generation': 0,
                     'stable_generation': 0}
        arguments.update(kwargs)
        return propose_axis_step(c, current or c['local_center_tag_px'], target or c['virtual_goal_px'],
                                 **arguments)

    def test_measured_columns_use_tag_response_and_actual_fk_mm(self):
        data = pair()
        result = build_calibration(*data)
        self.assertTrue(result['valid'], result)
        np.testing.assert_allclose(result['J_px_per_mm'], [[0., -1.3], [-1.2, 0.]], atol=1e-10)
        self.assertFalse(result['physical_registration'])
        self.assertFalse(result['columns_subtract_cup_noise'])
        self.assertEqual(result['virtual_goal_px'], [500., 200.])
        np.testing.assert_allclose(result['local_center_tag_px'], [498.7, 198.8], atol=1e-10)
        data[1]['target_verification']['samples'][-1]['feedback']['fk_flange_pose_m_rad'][0] = .0009
        data[2]['rows'] = deepcopy(data[2]['rows'])
        data[3]['baseline_feedback'][0]['fk_flange_pose_m_rad'][0] = .0009
        changed = build_calibration(*data)
        self.assertTrue(changed['valid'], changed)
        self.assertAlmostEqual(changed['J_px_per_mm'][1][0], -1.2/.9)

    def test_epoch_intrinsics_and_detector_mismatch_rejected(self):
        for change in ['epoch', 'intrinsics', 'detector', 'generation', 'same_frame']:
            data = pair()
            if change == 'epoch':
                for row in data[2]['rows']:
                    row['camera_epoch'] = row['marker']['camera_epoch'] = 'moved:cartesian_probe:y'
            elif change == 'intrinsics':
                data[2]['intrinsics_sha256'] = 'b'*64
            elif change == 'detector':
                data[0]['marker_corner_detector'] = 'adaptive-default'
            elif change == 'generation':
                data[0]['rows'][10]['tracking_generation'] = data[0]['rows'][10]['marker']['tracking_generation'] = 1
            else:
                data[0]['rows'][10]['marker']['frame_serial'] += 1
            result = build_calibration(*data)
            self.assertFalse(result['valid'], change)

    def test_insufficient_pre_or_post_samples_rejected(self):
        for start in [0, 10]:
            data = pair()
            for row in data[0]['rows'][start:start+3]:
                row['same_frame_current_tag_board_cup_valid'] = False
            result = build_calibration(*data)
            self.assertFalse(result['valid'])
            self.assertIn('eight', result['error'])

    def test_weak_or_uncertain_response_rejected(self):
        data = pair()
        for row in data[0]['rows'][10:]:
            row['marker']['raw_corners_px'] = (np.asarray(row['marker']['raw_corners_px'])+[0., 1.05]).tolist()
        result = build_calibration(*data)
        self.assertFalse(result['valid'])
        self.assertIn('weak or uncertain', result['error'])
        data = pair()
        for i, row in enumerate(data[0]['rows']):
            row['marker']['raw_corners_px'] = (np.asarray(row['marker']['raw_corners_px'])+
                                              [0., .24 if i % 10 < 5 else -.24]).tolist()
        result = build_calibration(*data)
        self.assertFalse(result['valid'])
        self.assertIn('weak or uncertain', result['error'])

    def test_parallel_responses_have_no_rank(self):
        data = pair()
        for row in data[2]['rows'][10:]:
            row['marker']['raw_corners_px'] = (np.asarray(row['marker']['raw_corners_px'])+[1.3, -1.3]).tolist()
        result = build_calibration(*data)
        self.assertFalse(result['valid'])
        self.assertIn('rank or conditioning', result['error'])

    def test_cup_or_board_movement_rejected_without_subtracting_it(self):
        for kind in ['cup', 'board']:
            data = pair()
            for row in data[0]['rows'][10:]:
                if kind == 'cup':
                    row['cup_top']['center_px'][0] += 1.1
                else:
                    row['board']['charuco_corners_px'][0][0] += .3
            result = build_calibration(*data)
            self.assertFalse(result['valid'], result)
            self.assertIn('moved', result['error'])

    def test_stale_row_and_sdk_height_or_angle_gate_fail(self):
        for change in ['stale', 'height', 'angle', 'joint', 'motion']:
            data = pair()
            sample = data[1]['target_verification']['samples'][-1]
            if change == 'stale':
                data[0]['rows'][0]['read_age_s'] = .16
            elif change == 'height':
                sample['feedback']['fk_flange_pose_m_rad'][2] = .00011
            elif change == 'angle':
                sample['feedback']['fk_flange_pose_m_rad'][3] = .001
            elif change == 'joint':
                sample['feedback']['q_rad'][6] += .0004
            else:
                sample['feedback']['status']['motion_status'] = 1
            self.assertFalse(build_calibration(*data)['valid'], change)

    def test_proposal_selects_largest_axis_and_clamps_to_one_mm(self):
        result = self.proposal()
        self.assertTrue(result['valid'], result)
        self.assertFalse(result['stop'])
        self.assertEqual(result['axis'], 'x')
        self.assertAlmostEqual(result['distance_mm'], -1.)
        self.assertLess(np.linalg.norm(result['predicted_remaining_error_px']), result['pixel_error_norm_px'])

    def test_quarter_pixel_stop_and_maximum_frame_age(self):
        c = build_calibration(*pair())
        current = np.asarray(c['virtual_goal_px'])+[.1, .1]
        result = self.proposal(calibration=c, current=current.tolist())
        self.assertTrue(result['valid'])
        self.assertTrue(result['stop'])
        self.assertIsNone(result['axis'])
        for arguments in [{'now_s': 100.16}, {'now_s': 99.99}, {'max_age_s': .2},
                          {'tracking_generation': 1}, {'camera_epoch': 'moved'},
                          {'intrinsics_sha256': 'b'*64}, {'max_step_mm': 1.01}]:
            result = self.proposal(**arguments)
            self.assertFalse(result['valid'], arguments)
            self.assertIsNone(result['distance_mm'])

    def test_local_three_mm_radius_is_around_y_post_not_original_goal(self):
        c = build_calibration(*pair())
        matrix = np.asarray(c['J_px_per_mm'])
        centre = np.asarray(c['local_center_tag_px'])
        near = (centre+matrix @ np.array([2.9, 0.])).tolist()
        self.assertTrue(self.proposal(calibration=c, current=near)['valid'])
        far = (centre+matrix @ np.array([3.01, 0.])).tolist()
        self.assertFalse(self.proposal(calibration=c, current=far)['valid'])
        self.assertFalse(self.proposal(calibration=c, target=far)['valid'])

    def test_corrupted_rank_matrix_cannot_use_prior_valid_flag(self):
        c = build_calibration(*pair())
        c['J_px_per_mm'] = [[1., 1.], [0., 0.]]
        self.assertFalse(self.proposal(calibration=c)['valid'])

    def test_selected_axis_endpoint_must_also_remain_local(self):
        c = build_calibration(*pair())
        centre, matrix = np.asarray(c['local_center_tag_px']), np.asarray(c['J_px_per_mm'])
        current = (centre+matrix @ np.array([2.8, .5])).tolist()
        target = (centre+matrix @ np.array([2., 2.2])).tolist()
        result = self.proposal(calibration=c, current=current, target=target)
        self.assertFalse(result['valid'])
        self.assertIn('leave', result['error'])

    def test_local_json_creation_and_cli_atomic_output(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, value in enumerate(pair()):
                path = Path(directory)/str(i)
                path.write_text(json.dumps(value))
                paths.append(path)
            result = create_calibration(*paths)
            self.assertTrue(result['valid'], result)
            self.assertEqual(len(result['source_files_sha256']), 4)
            output = Path(directory)/'calibration.json'
            argv = ['--x-scene', str(paths[0]), '--x-sdk', str(paths[1]),
                    '--y-scene', str(paths[2]), '--y-sdk', str(paths[3]), '--output', str(output)]
            with patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(main(argv), 0)
            self.assertTrue(json.loads(output.read_text())['valid'])
            self.assertFalse(list(Path(directory).glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
