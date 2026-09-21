"""Focused checks for transform direction and provisional-result handling."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cup_grasp_demo.planning import load_calibration, map_localization, plan_palm_target


SERIAL = '346222071954'


def fixture_calibration(passed=False):
    # 90 degree base <- camera rotation, and a nontrivial flange <- palm offset.
    bc = np.eye(4)
    bc[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    bc[:3, 3] = [.4, -.2, .3]
    ft = np.eye(4)
    ft[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    ft[:3, 3] = [.08, .01, -.02]
    return {'schema': 1, 'mode': 'eye_to_hand', 'quality_passed': passed,
            'direction': 'T_base_camera maps camera optical coordinates into arm base',
            'T_base_camera': bc.tolist(), 'T_flange_tcp': ft.tolist(), 'tcp': 'palm',
            'camera': {'backend': 'realsense', 'serial': SERIAL, 'frame': 'color_optical',
                       'width': 640, 'height': 480,
                       'camera_matrix': [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
                       'dist_coeffs': [0, 0, 0, 0, 0]}}


def fixture_localization():
    return {'schema_version': 1, 'valid': True, 'frame': 'color_optical', 'units': 'm',
            'frame_id': f'{SERIAL}:42', 'camera_serial': SERIAL,
            'instance_id': 'cup-1', 'timestamp_ms': 1234,
            'geometry': {'center_m': [.1, .2, .5], 'support_center_m': [.1, .2, .46],
                         'axis': [0, 0, 1]}}


class PlanningTests(unittest.TestCase):
    def test_provisional_gate_requires_explicit_override_at_load_and_plan(self):
        result = fixture_calibration()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'result.json'
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'allow_provisional'):
                load_calibration(path, expected_camera_serial=SERIAL)
            loaded = load_calibration(path, expected_camera_serial=SERIAL,
                                      allow_provisional=True)
            self.assertEqual(len(loaded['_source']['sha256']), 64)
            with self.assertRaisesRegex(ValueError, 'allow_provisional'):
                map_localization(fixture_localization(), loaded, camera_serial=SERIAL)
            plan = plan_palm_target(fixture_localization(), loaded,
                                    camera_serial=SERIAL, palm_position_base_m=[.3, .1, .2],
                                    palm_orientation_base=np.eye(3), allow_provisional=True)
            self.assertTrue(plan['provisional'])
            self.assertFalse(plan['source_quality_passed'])

    def test_camera_to_base_direction_and_axis(self):
        mapped = map_localization(fixture_localization(), fixture_calibration(True),
                                  camera_serial=SERIAL)
        np.testing.assert_allclose(mapped['center_m'], [.2, -.1, .8], atol=1e-12)
        np.testing.assert_allclose(mapped['support_center_m'], [.2, -.1, .76], atol=1e-12)
        np.testing.assert_allclose(mapped['axis'], [0, 0, 1], atol=1e-12)

    def test_tcp_inverse_recovers_target_palm_pose(self):
        calibration = fixture_calibration(True)
        palm_orientation = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        target = plan_palm_target(fixture_localization(), calibration,
                                  camera_serial=SERIAL, palm_position_base_m=[.3, .1, .2],
                                  palm_orientation_base=palm_orientation)
        actual = np.asarray(target['T_base_flange_target']) @ np.asarray(calibration['T_flange_tcp'])
        np.testing.assert_allclose(actual, target['T_base_palm_target'], atol=1e-12)
        self.assertEqual(target['orientation_source'], 'specified_palm_orientation')
        current = np.eye(4)
        current[:3, :3] = palm_orientation
        preserved = plan_palm_target(fixture_localization(), calibration,
                                     camera_serial=SERIAL, palm_position_base_m=[.4, .2, .3],
                                     current_T_base_palm=current)
        np.testing.assert_allclose(np.asarray(preserved['T_base_palm_target'])[:3, :3],
                                   palm_orientation, atol=1e-12)

    def test_reject_frame_serial_direction_and_invalid_geometry(self):
        result, localization = fixture_calibration(True), fixture_localization()
        with self.assertRaisesRegex(ValueError, 'serial'):
            map_localization(localization, result, camera_serial='other-device')
        wrong = copy.deepcopy(localization)
        wrong['camera_serial'] = 'other-device'
        with self.assertRaisesRegex(ValueError, 'serial'):
            map_localization(wrong, result, camera_serial=SERIAL)
        wrong = copy.deepcopy(localization)
        wrong['frame'] = 'depth_optical'
        with self.assertRaisesRegex(ValueError, 'color_optical'):
            map_localization(wrong, result, camera_serial=SERIAL)
        wrong = copy.deepcopy(result)
        wrong['direction'] = 'T_camera_base maps base into camera'
        with self.assertRaisesRegex(ValueError, 'direction'):
            map_localization(localization, wrong, camera_serial=SERIAL)
        wrong = copy.deepcopy(localization)
        wrong['geometry']['axis'] = [0, 0, 2]
        with self.assertRaisesRegex(ValueError, 'unit vector'):
            map_localization(wrong, result, camera_serial=SERIAL)

    def test_reject_ambiguous_orientation_and_other_tcp(self):
        calibration = fixture_calibration(True)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            plan_palm_target(fixture_localization(), calibration, camera_serial=SERIAL,
                             palm_position_base_m=[.3, .1, .2])
        calibration['tcp'] = 'flange'
        with self.assertRaisesRegex(ValueError, 'palm TCP'):
            plan_palm_target(fixture_localization(), calibration, camera_serial=SERIAL,
                             palm_position_base_m=[.3, .1, .2], palm_orientation_base=np.eye(3))


if __name__ == '__main__':
    unittest.main()
