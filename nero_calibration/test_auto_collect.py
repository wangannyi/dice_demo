"""Offline checks for automatic hand-eye teaching and replay."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from auto_collect import (board_corners, check_path, inside_window, interpolate,
                          make_plan, outline, run_plan, taught_route, validate_window)


class FakeJoint:
    lower_rad = -2.
    upper_rad = 2.


class FakeModel:
    joints = [FakeJoint()] * 7

    def fk(self, q):
        t = np.eye(4)
        t[0, 3] = q[0]
        return t


class AutoCollectionTests(unittest.TestCase):
    def setUp(self):
        self.camera = {'serial': 'mock-camera', 'width': 640, 'height': 480,
                       'camera_matrix': [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],
                       'dist_coeffs': [0]*5}
        self.board = {'squares_x': 4, 'squares_y': 5, 'square_length_m': .02}
        self.manifest = {'camera': self.camera, 'board': self.board,
                         'T_flange_tcp': np.eye(4).tolist()}
        self.calibration = {'T_base_camera': np.eye(4).tolist(),
                            'T_tcp_board': self.pose(.5).tolist()}

    @staticmethod
    def pose(z):
        t = np.eye(4)
        t[2, 3] = z
        return t

    def test_outer_board_projection_and_window(self):
        points = outline(self.pose(.5), self.camera, board_corners(self.board, {}))
        np.testing.assert_allclose(points[0], [320, 240], atol=1e-6)
        np.testing.assert_allclose(points[2], [480, 440], atol=1e-6)
        self.assertTrue(inside_window(points, [300, 220, 500, 460], 8))
        self.assertFalse(inside_window(points, [300, 220, 470, 460], 8))

    def test_joint_path_checks_interpolated_board_visibility(self):
        start = [0.]*7
        target = [.04]+[0.]*6
        path = check_path(start, [target], FakeModel(), self.manifest,
                          self.calibration, [300, 220, 600, 470], 1., 5.)
        self.assertGreater(len(path), 2)
        self.assertEqual(path[-1]['capture_sample_index'], 0)
        with self.assertRaisesRegex(ValueError, 'Board leaves drawn window'):
            check_path(start, [[.08]+[0.]*6], FakeModel(), self.manifest,
                       self.calibration, [300, 220, 600, 470], 1., 5.)
        self.assertEqual(len(interpolate(start, target, 1.)), 3)

    def test_trace_records_path_and_rejects_lost_board(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            samples = [{'time_unix_s': float(i+1), 'joints_rad': [i*.01]+[0.]*6,
                        'T_camera_board': self.pose(.5).tolist(),
                        'frame_joint_observations': [{}]*3} for i in range(2)]
            frame = {'time_unix_s': 1.5, 'joints': {'joints_rad': [.005]+[0.]*6},
                     'T_camera_board': self.pose(.5).tolist(),
                     'image_width': 640, 'image_height': 480}
            trace = dataset/'teaching_frames.jsonl'
            trace.write_text(json.dumps(frame)+'\n')
            route, capture, used = taught_route(dataset, samples, self.manifest,
                                                 self.calibration, [300, 220, 500, 460], 5.)
            self.assertEqual(capture, [0, None, 1])
            self.assertEqual(route[1][0], .005)
            self.assertEqual(used, trace)
            frame['T_camera_board'] = None
            trace.write_text(json.dumps(frame)+'\n')
            with self.assertRaisesRegex(ValueError, 'not detected'):
                taught_route(dataset, samples, self.manifest, self.calibration,
                             [300, 220, 500, 460], 5.)
            frame['T_camera_board'] = self.pose(.5).tolist()
            frame['time_unix_s'] = 1.1
            trace.write_text(json.dumps(frame)+'\n')
            samples[1]['time_unix_s'] = 5.
            with self.assertRaisesRegex(ValueError, 'Unrecorded arm movement'):
                taught_route(dataset, samples, self.manifest, self.calibration,
                             [300, 220, 500, 460], 5.)

    def test_window_must_match_camera_and_board(self):
        data = {'camera': self.camera, 'board': self.board,
                'window_xyxy': [0, 0, 640, 480]}
        self.assertEqual(validate_window(data, self.manifest), data['window_xyxy'])
        with self.assertRaisesRegex(ValueError, 'window'):
            validate_window({**data, 'window_xyxy': [0, 0, 641, 480]}, self.manifest)

    def make_taught_session(self, root):
        source = root/'taught'
        source.mkdir()
        manifest = {**self.manifest, 'schema': 1, 'mode': 'eye_to_hand'}
        (source/'manifest.json').write_text(json.dumps(manifest))
        calibration = {**self.calibration, 'camera': self.camera,
                       'T_flange_tcp': np.eye(4).tolist(), 'quality_passed': True}
        calibration_path = root/'calibration.json'
        calibration_path.write_text(json.dumps(calibration))
        (source/'board_window.json').write_text(json.dumps({
            'window_xyxy': [300, 220, 640, 470], 'camera': self.camera,
            'board': self.board}))
        frames = []
        for i in range(8):
            sample = {'T_base_flange': np.eye(4).tolist(),
                      'T_base_tcp': np.eye(4).tolist(),
                      'T_camera_board': self.pose(.5).tolist(),
                      'time_unix_s': float(i+1),
                      'joints_rad': [i*.01]+[0.]*6,
                      'frame_joint_observations': [{}]*3}
            (source/f'sample_{i:04d}.json').write_text(json.dumps(sample))
            if i:
                frames.append({'time_unix_s': i+.5,
                               'joints': {'joints_rad': [(i-.5)*.01]+[0.]*6},
                               'T_camera_board': self.pose(.5).tolist(),
                               'image_width': 640, 'image_height': 480})
        (source/'teaching_frames.jsonl').write_text(
            ''.join(json.dumps(row)+'\n' for row in frames))
        return source, calibration_path

    def test_plan_then_mock_run_writes_all_samples_without_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, calibration = self.make_taught_session(root)
            plan_path, output = root/'plan.json', root/'automatic'
            with patch('auto_collect.load_model', return_value=FakeModel()):
                plan = make_plan(source, calibration, plan_path, margin_px=5.)
            self.assertEqual(plan['path_source'], 'per_frame_teaching_trace')
            self.assertEqual(sum(w['capture_sample_index'] is not None
                                 for w in plan['waypoints']), 8)
            camera, arm, detector = Mock(), Mock(), Mock()
            camera.info = self.camera
            arm.read_joints.return_value = {'joints_rad': [0.]*7}
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            sample = {'T_base_flange': np.eye(4).tolist(),
                      'T_camera_board': self.pose(.5).tolist(),
                      'quality': {'corners': 12, 'reprojection_rms_px': .1}}
            with patch('auto_collect.RealSenseCamera', return_value=camera), \
                 patch('auto_collect.NeroFeedback', return_value=arm), \
                 patch('auto_collect.CharucoDetector', return_value=detector), \
                 patch('auto_collect.load_model', return_value=FakeModel()), \
                 patch('auto_collect.require_arm_ready'), \
                 patch('auto_collect.live_board'), \
                 patch('auto_collect.move_and_watch') as move, \
                 patch('auto_collect.capture_sample', return_value=(sample, frame, frame)):
                self.assertEqual(run_plan(plan_path, output, 'can0', 10), 8)
            self.assertGreater(move.call_count, 8)
            self.assertEqual(len(list(output.glob('sample_*.json'))), 8)
            self.assertFalse((output/'AUTO_INCOMPLETE.json').exists())
            arm.close.assert_called_once()
            camera.close.assert_called_once()
            (source/'teaching_frames.jsonl').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'changed after planning'):
                run_plan(plan_path, root/'another', 'can0', 10)

    def test_replay_fps_override_keeps_teaching_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, calibration = self.make_taught_session(root)
            taught = json.loads((source/'manifest.json').read_text())
            taught['camera']['fps'] = 15
            taught['board']['image_profile'] = {'color_resolution': [640, 480], 'fps': 15}
            (source/'manifest.json').write_text(json.dumps(taught))
            solved = json.loads(calibration.read_text())
            solved['camera']['fps'] = 15
            calibration.write_text(json.dumps(solved))
            window = json.loads((source/'board_window.json').read_text())
            window['camera']['fps'] = 15
            window['board'] = taught['board']
            (source/'board_window.json').write_text(json.dumps(window))
            with patch('auto_collect.load_model', return_value=FakeModel()):
                make_plan(source, calibration, root/'plan.json', margin_px=5.)
            camera = Mock()
            camera.info = dict(self.camera, fps=6)
            with patch('auto_collect.RealSenseCamera', return_value=camera) as open_camera, \
                 patch('auto_collect.NeroFeedback', side_effect=RuntimeError('mock stop')):
                with self.assertRaisesRegex(RuntimeError, 'mock stop'):
                    run_plan(root/'plan.json', root/'automatic', 'can0', 10, fps=6)
            self.assertEqual(open_camera.call_args.kwargs['image_profile']['fps'], 6)
            camera.close.assert_called_once()
            self.assertEqual(json.loads((source/'manifest.json').read_text())['camera']['fps'], 15)


if __name__ == '__main__':
    unittest.main()
