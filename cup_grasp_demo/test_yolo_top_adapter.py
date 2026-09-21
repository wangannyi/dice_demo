"""Same-frame YOLO top source and HOME-before-capture regressions."""

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from cup_grasp_demo.grasp import save_snapshot
from cup_grasp_demo.yolo_top_adapter import build_source_record
from nero_calibration.core import PALM, matrix_pose
from nero_revo2_control.kinematics import load_model


Q = np.radians((40.705, -70.001, -101.037, 77.192,
                -91.354, -37.301, -24.419))
SDK = [[-155, 155], [-100, 100], [-158, 158], [-58, 123],
       [-158, 158], [-42, 55], [-90, 90]]


class YoloTopAdapterTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        captured_ns = time.time_ns()-int(2e9)
        frame = {
            'color_bgr': np.zeros((24, 32, 3), dtype=np.uint8),
            'depth_raw': np.full((24, 32), 500, dtype=np.uint16),
            'depth_scale_m': .001,
            'intrinsics': {'width': 32, 'height': 24, 'fx': 50., 'fy': 50.,
                           'ppx': 16., 'ppy': 12., 'dist_coeffs': [0.]*5,
                           'distortion_model': 'none'},
            'serial': 'fixture-camera', 'frame': 'color_optical', 'frame_id': 77,
            'timestamps_ms': {'color': 1000., 'depth_aligned': 1000.1},
            'timestamp_domains': {'color': 'fixture', 'depth_aligned': 'fixture'},
            'host_capture_time_ns': captured_ns,
        }
        self.meta = save_snapshot(self.root/'snapshot', frame)
        self.result_dir = self.root/'recognition'
        self.result_dir.mkdir()
        recognized = {'frame_id': self.meta['frame_id'], 'selected_instance': 0,
                      'model': {'sha256': 'a'*64},
                      'model_profile': 'dice_cap2',
                      'red_workspace': {'config_sha256': 'r'*64},
                      'instances': [{'class_id': 0, 'green_fraction': .7,
                                     'red_workspace': {'valid': True,
                                                       'mask_inside_fraction': 1.,
                                                       'red_context_fraction': .9}}]}
        (self.result_dir/'recognition.json').write_text(json.dumps(recognized))
        geometry = {
            'schema_version': 1, 'frame': 'color_optical', 'units': 'm',
            'frame_id': self.meta['frame_id'], 'instance_id': 'cup-0',
            'timestamp_ms': self.meta['timestamp_ms'],
            'timestamp_domain': self.meta['timestamp_domain'], 'valid': True,
            'geometry': {'center_m': [0., 0., .536],
                         'support_center_m': [0., 0., .5],
                         'axis': [0., 0., 1.],
                         'dimensions': {'observed_height_m': .072,
                                        'observed_side_diameter_m': .075},
                         'quality': {'table_support_center_inside_red_workspace': True},
                         'top_surface': {
                             'valid': True, 'center_m': [0., 0., .572],
                             'normal': [0., 0., 1.],
                             'center_definition':
                                 'side_axis_intersection_with_visible_top_plane',
                             'quality': {'rim_center_independently_measured': False,
                                         'coaxial_cup_assumption': True,
                                         'center_support_points': 30},
                             'reason': None}},
            'input_provenance': {'kind': 'cup_grasp_demo_snapshot',
                                 'sha256_color': self.meta['sha256_color'],
                                 'sha256_depth': self.meta['sha256_depth']},
            'model_sha256': 'a'*64,
            'model_profile': 'dice_cap2',
            'red_workspace_config_sha256': 'r'*64,
        }
        (self.result_dir/'geometry.json').write_text(json.dumps(geometry))
        self.calibration = self.root/'calibration.json'
        self.calibration.write_text(json.dumps({
            'schema': 1, 'mode': 'eye_to_hand', 'quality_passed': False,
            'direction': 'T_base_camera maps camera optical coordinates into arm base',
            'tcp': 'palm', 'T_base_camera': np.eye(4).tolist(),
            'T_flange_tcp': PALM.tolist(),
            'camera': {'backend': 'realsense', 'frame': 'color_optical',
                       'serial': 'fixture-camera', 'width': 32, 'height': 24,
                       'camera_matrix': [[50., 0., 16.], [0., 50., 12.], [0., 0., 1.]],
                       'dist_coeffs': [0.]*5},
        }))
        self.home = {'schema': 1, 'kind': 'nero_green_cup_state_machine',
                     'phase': 'PLAN', 'snapshot_frame_id': self.meta['frame_id'],
                     'home_arrival_time_ns': captured_ns-int(1e9),
                     'home_arrival_joints_rad': Q.tolist()}
        self.status = {'event': 'arm_status', 'joints_rad': Q.tolist(),
                       'flange_m_rad': matrix_pose(load_model().fk(Q))}
        self.joints = {'event': 'read_joints', 'joints_rad': Q.tolist(),
                       'sdk_limits_deg': SDK, 'arm_status': 0, 'ctrl_mode': 1,
                       'joints_enabled': [True]*7}
        self.arm_read_ns = captured_ns+int(.5e9)
        self.now_ns = self.arm_read_ns+int(.1e9)

    def tearDown(self):
        self.folder.cleanup()

    def build(self, **overrides):
        values = {'home_state': self.home, 'arm_read_time_ns': self.arm_read_ns,
                  'allow_provisional': True, 'now_ns': self.now_ns}
        values.update(overrides)
        return build_source_record(self.root/'snapshot', self.result_dir,
                                   self.calibration, self.status, self.joints,
                                   **values)

    def test_same_frame_visible_top_maps_into_base_and_preserves_quality(self):
        source = self.build()
        self.assertFalse(source['checks']['execute_ready'])
        self.assertTrue(source['checks']['home_confirmed'])
        self.assertEqual(source['source_role'],
                         'yolo_visible_top_source_only_no_side_motion')
        np.testing.assert_allclose(
            source['cup_base']['measured_top_surface']['center_base_m'],
            [0., 0., .572], atol=1e-8)
        self.assertFalse(source['cup_base']['measured_top_surface']['quality']
                         ['rim_center_independently_measured'])
        self.assertTrue(source['recognition']['red_workspace']
                        ['selected_green_cap_in_red_mat'])
        self.assertEqual(source['recognition']['red_workspace']['source_frame_id'],
                         self.meta['frame_id'])

    def test_old_or_mismatched_frame_and_home_are_rejected(self):
        result_path = self.result_dir/'geometry.json'
        result = json.loads(result_path.read_text())
        result['input_provenance']['sha256_color'] = 'f'*64
        result_path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'exact current RGB-D'):
            self.build()
        result['input_provenance']['sha256_color'] = self.meta['sha256_color']
        result_path.write_text(json.dumps(result))
        old_home = copy.deepcopy(self.home)
        old_home['home_arrival_time_ns'] = self.meta['host_capture_time_ns']+1
        with self.assertRaisesRegex(ValueError, 'current HOME followed'):
            self.build(home_state=old_home)
        changed_home = copy.deepcopy(self.home)
        changed_home['home_arrival_joints_rad'][2] += .2
        with self.assertRaisesRegex(ValueError, 'Arm moved after HOME'):
            self.build(home_state=changed_home)

    def test_blue_mat_instance_or_red_support_loss_cannot_be_a_formal_target(self):
        path = self.result_dir/'recognition.json'
        report = json.loads(path.read_text())
        report['instances'][0]['red_workspace']['valid'] = False
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'red-mat mask/table support'):
            self.build()


if __name__ == '__main__':
    unittest.main()
