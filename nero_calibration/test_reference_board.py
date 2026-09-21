import copy
import unittest

import numpy as np

from core import inverse, pose_matrix
from reference_board import DIRECTION, register, restore, summarize


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.bc = pose_matrix([.12, -.3, .65, .2, -.1, .7])
        self.bb = pose_matrix([.31, -.12, .02, 0., 0., .3])
        self.camera = {'backend': 'realsense', 'serial': 'test', 'frame': 'color_optical'}
        self.calibration = {'schema': 1, 'mode': 'eye_to_hand', 'direction': DIRECTION,
                            'quality_passed': False, 'T_base_camera': self.bc.tolist(),
                            'T_flange_tcp': np.eye(4).tolist(), 'camera': self.camera,
                            'all_sample_errors_m_deg': [[.0123, 3.645]]}
        self.board = {'type': 'charuco', 'dictionary': '4x4_50', 'squares_x': 4,
                      'squares_y': 5, 'square_length_m': .0215, 'marker_length_m': .015,
                      'legacy_pattern': False, 'image_roi_xyxy': [10, 10, 300, 400]}

    def observation(self, bc):
        cb = inverse(bc) @ self.bb
        return {'schema': 1, 'mode': 'reference_observation', 'reference_id': 'table',
                'opencv_version': 'same', 'camera': copy.deepcopy(self.camera),
                'board': copy.deepcopy(self.board), 'T_camera_board': cb.tolist(),
                'records': [{'T_camera_board': cb.tolist(),
                             'quality': {'corners': 12, 'reprojection_rms_px': .15}}
                            for _ in range(10)]}

    def test_known_transform_and_moved_camera_recovery(self):
        obs = self.observation(self.bc)
        original = copy.deepcopy(self.calibration)
        reg = register(self.calibration, obs, True)
        np.testing.assert_allclose(reg['T_base_board'], self.bb, atol=1e-10)
        moved = pose_matrix([-.25, .14, .72, -.3, .5, -.6])
        obs2 = self.observation(moved)
        obs2['board']['image_roi_xyxy'] = [200, 100, 600, 470]
        result = restore(reg, obs2, True)
        np.testing.assert_allclose(result['T_base_camera'], moved, atol=1e-10)
        self.assertFalse(result['quality_passed'])
        self.assertFalse(result['independent_accuracy_verified'])
        self.assertEqual(result['all_sample_errors_m_deg'], original['all_sample_errors_m_deg'])
        self.assertEqual(self.calibration, original)
        np.testing.assert_allclose(restore(reg, obs, True)['T_base_camera'], self.bc, atol=1e-10)

    def test_failed_calibration_requires_opt_in_at_both_steps(self):
        obs = self.observation(self.bc)
        with self.assertRaisesRegex(ValueError, 'allow-provisional'):
            register(self.calibration, obs)
        reg = register(self.calibration, obs, True)
        with self.assertRaisesRegex(ValueError, 'allow-provisional'):
            restore(reg, obs)
        reg['quality_passed'] = True
        self.assertFalse(restore(reg, obs, True)['quality_passed'])

    def test_passed_calibration_retains_default_without_opt_in(self):
        self.calibration['quality_passed'] = True
        obs = self.observation(self.bc)
        self.assertTrue(restore(register(self.calibration, obs), obs)['quality_passed'])

    def test_reject_mismatched_camera_board_or_version(self):
        obs = self.observation(self.bc)
        reg = register(self.calibration, obs, True)
        changes = [('camera', 'serial', 'other'), ('board', 'square_length_m', .022),
                   (None, 'reference_id', 'other'), (None, 'opencv_version', 'different')]
        for group, key, value in changes:
            with self.subTest(key=key):
                changed = copy.deepcopy(obs)
                (changed[group] if group else changed)[key] = value
                with self.assertRaises(ValueError):
                    restore(reg, changed, True)
        changed = copy.deepcopy(obs)
        changed['camera']['serial'] = 'other'
        with self.assertRaises(ValueError):
            register(self.calibration, changed, True)

    def test_reject_incomplete_noisy_or_unstable_observations(self):
        records = self.observation(self.bc)['records']
        with self.assertRaises(ValueError):
            summarize(records[:9])
        for key, value in [('corners', 11), ('reprojection_rms_px', float('nan')),
                           ('reprojection_rms_px', .6)]:
            changed = copy.deepcopy(records)
            changed[0]['quality'][key] = value
            with self.assertRaises(ValueError):
                summarize(changed)
        changed = copy.deepcopy(records)
        changed[0]['T_camera_board'][0][3] += .01
        with self.assertRaisesRegex(ValueError, 'unstable'):
            summarize(changed)

    def test_reject_forged_mean(self):
        obs = self.observation(self.bc)
        obs['T_camera_board'][0][3] += .01
        with self.assertRaisesRegex(ValueError, 'mean disagrees'):
            register(self.calibration, obs, True)


if __name__ == '__main__':
    unittest.main()
