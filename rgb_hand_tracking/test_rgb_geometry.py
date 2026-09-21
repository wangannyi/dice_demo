import copy
import json
import unittest

import cv2
import numpy as np

from board_rgb import BoardRgbObserver
from rgb_geometry import CalibratedRgbGeometry


BOARD = {'type': 'charuco', 'dictionary': '4x4_50',
         'squares_x': 4, 'squares_y': 5, 'square_length_m': .022,
         'marker_length_m': .0155, 'legacy_pattern': False,
         'board_face_above_cup_support_m': .004,
         'previous_setup_height_reverified': True}
CAMERA = {'backend': 'v4l2_rgb', 'device': '/dev/video7',
          'format': 'MJPG', 'width': 1280, 'height': 720}
CONFIG = {'camera': CAMERA, 'board': BOARD,
          'cup': {'height_m': .0675, 'top_radius_m': .025}}
K = np.array([[760., 0., 631.], [0., 765., 358.], [0., 0., 1.]])
D = np.array([.13, -.12, -.0035, -.0033, 0.])
REPORT = {'quality_passed': True, 'image_size': [1280, 720],
          'camera': CAMERA, 'board': BOARD, 'distortion_model': 'opencv_brown_5',
          'camera_matrix': K.tolist(), 'distortion_coefficients': D.tolist()}
RVEC = np.array([.35, -.25, .12])
TVEC = np.array([-.15, -.10, .65])
CUP_CENTER = np.array([.24, .16, -.0635])


def observations(center=CUP_CENTER, radii=None, angles=None):
    observer = BoardRgbObserver({'board': BOARD})
    objects = (observer.board.getChessboardCorners()
               if hasattr(observer.board, 'getChessboardCorners')
               else observer.board.chessboardCorners)
    objects = np.asarray(objects, dtype=float).reshape(-1, 3)
    pixels = cv2.projectPoints(objects, RVEC, TVEC, K, D)[0].reshape(-1, 2)
    board = {'valid': True, 'image_size': [1280, 720],
             'charuco_corner_ids': list(range(len(objects))),
             'charuco_corners_px': pixels.tolist()}
    if angles is None:
        angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
    if radii is None:
        radii = np.full(len(angles), .025)
    circle = center+np.column_stack((radii*np.cos(angles), radii*np.sin(angles),
                                    np.zeros(len(angles))))
    edges = cv2.projectPoints(circle, RVEC, TVEC, K, D)[0].reshape(-1, 2)
    cup = {'valid': True, 'source': 'synthetic_known_top_edge',
           'edge_points_px': edges.tolist(),
           # Deliberately wrong: neither field may replace the projected edges.
           'center_px': [10., 10.], 'ellipse_px': {'center_px': [10., 10.]}}
    return board, cup


class RgbGeometryTests(unittest.TestCase):
    def setUp(self):
        self.geometry = CalibratedRgbGeometry(copy.deepcopy(CONFIG), copy.deepcopy(REPORT))

    def test_tilted_board_and_elevated_top_circle_recovered_from_edges(self):
        board, cup = observations()
        result = self.geometry.observe(board, cup)
        self.assertTrue(result['geometry_valid'], result)
        pose = result['board_pose']
        expected = np.eye(4)
        expected[:3, :3] = cv2.Rodrigues(RVEC)[0]
        expected[:3, 3] = TVEC
        np.testing.assert_allclose(pose['T_camera_board'], expected, atol=1e-6)
        self.assertLess(pose['reprojection_rms_px'], 1e-5)
        np.testing.assert_allclose(result['cup_top']['center_board_m'], CUP_CENTER, atol=1e-6)
        self.assertAlmostEqual(result['cup_top']['radius_m'], .025, places=6)
        self.assertLess(result['cup_top']['radial_rms_m'], 1e-6)
        self.assertAlmostEqual(result['cup_top']['top_above_board_m'], .0635)
        self.assertEqual(pose['board_positive_z'], 'into_table')
        self.assertFalse(result['independent_metric_accuracy_validated'])
        self.assertFalse(result['motion_target_valid'])
        self.assertNotIn('palm', result)
        json.dumps(result, allow_nan=False)

    def test_projecting_elevated_cup_onto_table_produces_material_bias(self):
        board, cup = observations()
        result = self.geometry.observe(board, cup)
        transform = np.asarray(result['board_pose']['T_camera_board'])
        wrong = self.geometry._project_to_plane(
            cup['edge_points_px'], transform[:3, :3], transform[:3, 3], 0.)
        error = np.linalg.norm(wrong[:, :2].mean(axis=0)-CUP_CENTER[:2])
        self.assertGreater(error, .015)
        # The proper height-plane result remains within a micrometer here;
        # this is numeric recovery evidence, not an independent real-world test.
        np.testing.assert_allclose(result['cup_top']['center_board_m'], CUP_CENTER, atol=1e-6)

    def test_unknown_or_unverified_face_height_preserves_only_board_pose(self):
        board, cup = observations()
        for height, verified, reason in (
                (None, True, 'board_face_above_cup_support_height_unknown'),
                (.004, False, 'current_board_face_above_cup_support_height_unverified')):
            with self.subTest(height=height, verified=verified):
                config = copy.deepcopy(CONFIG)
                config['board']['board_face_above_cup_support_m'] = height
                config['board']['previous_setup_height_reverified'] = verified
                result = CalibratedRgbGeometry(config, REPORT).observe(board, cup)
                self.assertTrue(result['board_pose']['valid'])
                self.assertFalse(result['geometry_valid'])
                self.assertEqual(result['cup_top']['reason'], reason)
                self.assertIsNone(result['cup_top']['center_board_m'])

    def test_missing_frame_geometry_does_not_reuse_previous_pose_or_circle(self):
        board, cup = observations()
        self.assertTrue(self.geometry.observe(board, cup)['geometry_valid'])
        result = self.geometry.observe({'valid': False}, cup)
        self.assertFalse(result['board_pose']['valid'])
        self.assertIsNone(result['board_pose']['T_camera_board'])
        self.assertFalse(result['cup_top']['valid'])
        result = self.geometry.observe(board, {'valid': False})
        self.assertTrue(result['board_pose']['valid'])
        self.assertFalse(result['cup_top']['valid'])
        self.assertIsNone(result['cup_top']['center_board_m'])

    def test_calibration_quality_mode_board_pattern_and_parameters_checked(self):
        mutations = ('quality', 'report_size', 'camera_size', 'camera_device', 'camera_format',
                     'square', 'marker', 'dictionary', 'legacy', 'matrix_nan', 'fx',
                     'cx', 'skew', 'distortion_nan', 'distortion_size', 'distortion_large',
                     'distortion_model')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                report = copy.deepcopy(REPORT)
                if mutation == 'quality':
                    report['quality_passed'] = 1
                elif mutation == 'report_size':
                    report['image_size'] = [640, 360]
                elif mutation == 'camera_size':
                    report['camera']['width'] = 640
                elif mutation == 'camera_device':
                    report['camera']['device'] = '/dev/video0'
                elif mutation == 'camera_format':
                    report['camera']['format'] = 'YUYV'
                elif mutation in ('square', 'marker'):
                    report['board'][mutation+'_length_m'] = .020 if mutation == 'square' else .014
                elif mutation == 'dictionary':
                    report['board']['dictionary'] = '5x5_50'
                elif mutation == 'legacy':
                    report['board']['legacy_pattern'] = True
                elif mutation == 'matrix_nan':
                    report['camera_matrix'][0][0] = float('nan')
                elif mutation == 'fx':
                    report['camera_matrix'][0][0] = -760.
                elif mutation == 'cx':
                    report['camera_matrix'][0][2] = 1600.
                elif mutation == 'skew':
                    report['camera_matrix'][0][1] = 10.
                elif mutation == 'distortion_nan':
                    report['distortion_coefficients'][0] = float('nan')
                elif mutation == 'distortion_size':
                    report['distortion_coefficients'] = [0.]*8
                elif mutation == 'distortion_large':
                    report['distortion_coefficients'][2] = 1.
                else:
                    report['distortion_model'] = 'fisheye'
                with self.assertRaises(ValueError):
                    CalibratedRgbGeometry(CONFIG, report)

    def test_bad_board_corner_observations_rejected(self):
        mutations = ('size', 'ids_duplicate', 'ids_float', 'ids_range', 'nan',
                     'outside', 'too_few', 'collinear', 'reprojection')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                board, cup = observations()
                if mutation == 'size':
                    board['image_size'] = [640, 360]
                elif mutation == 'ids_duplicate':
                    board['charuco_corner_ids'][1] = 0
                elif mutation == 'ids_float':
                    board['charuco_corner_ids'][0] = 0.
                elif mutation == 'ids_range':
                    board['charuco_corner_ids'][0] = 99
                elif mutation == 'nan':
                    board['charuco_corners_px'][0][0] = float('nan')
                elif mutation == 'outside':
                    board['charuco_corners_px'][0][0] = 1300.
                elif mutation == 'too_few':
                    board['charuco_corner_ids'] = board['charuco_corner_ids'][:4]
                    board['charuco_corners_px'] = board['charuco_corners_px'][:4]
                elif mutation == 'collinear':
                    board['charuco_corners_px'] = [[200.+i, 200.] for i in range(12)]
                else:
                    board['charuco_corners_px'][0][0] += 25.
                result = self.geometry.observe(board, cup)
                self.assertFalse(result['geometry_valid'])
                self.assertFalse(result['board_pose']['valid'])

    def test_cup_edges_required_and_bad_circles_rejected(self):
        mutations = ('missing', 'short', 'duplicate', 'nan', 'radius', 'radial_error', 'arc')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                if mutation == 'radius':
                    board, cup = observations(radii=np.full(72, .070))
                elif mutation == 'radial_error':
                    board, cup = observations(radii=np.tile([.019, .031], 36))
                elif mutation == 'arc':
                    board, cup = observations(angles=np.linspace(0., np.pi/3, 24))
                else:
                    board, cup = observations()
                    if mutation == 'missing':
                        del cup['edge_points_px']
                    elif mutation == 'short':
                        cup['edge_points_px'] = cup['edge_points_px'][:11]
                    elif mutation == 'duplicate':
                        cup['edge_points_px'] = [cup['edge_points_px'][0]]*20
                    else:
                        cup['edge_points_px'][0][0] = float('nan')
                result = self.geometry.observe(board, cup)
                self.assertTrue(result['board_pose']['valid'])
                self.assertFalse(result['cup_top']['valid'])
                self.assertFalse(result['geometry_valid'])
                json.dumps(result, allow_nan=False)

    def test_user_approximate_radius_does_not_override_measured_circle(self):
        config = copy.deepcopy(CONFIG)
        config['cup']['top_radius_m'] = .030
        result = CalibratedRgbGeometry(config, REPORT).observe(*observations())
        self.assertTrue(result['geometry_valid'])
        self.assertAlmostEqual(result['cup_top']['radius_m'], .025, places=6)
        self.assertAlmostEqual(result['cup_top']['radius_minus_approximate_user_radius_m'],
                               -.005, places=6)

    def test_parallel_and_negative_depth_ray_intersections_rejected(self):
        rotation = cv2.Rodrigues(RVEC)[0]
        with self.assertRaisesRegex(ValueError, 'behind_camera'):
            self.geometry._project_to_plane([[631., 358.]], rotation, TVEC, 2.)
        parallel = cv2.Rodrigues(np.array([0., np.pi/2, 0.]))[0]
        with self.assertRaisesRegex(ValueError, 'parallel'):
            self.geometry._project_to_plane([[631., 358.]], parallel, TVEC, .0635)


if __name__ == '__main__':
    unittest.main()
