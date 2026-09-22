"""Read-only metric board/cup geometry from a calibrated RGB observation.

The OpenCV ChArUco board has its origin at a printed-pattern corner and +Z
into the table when its printed face points at the overhead camera. A physical
height above that face therefore has board Z = -height. This module produces
no hand contact estimate, robot/base transform, or executable motion target.
"""
import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from rgb_hand_tracking.board_rgb import BoardRgbObserver


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Invalid '+name)
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid '+name) from exc
    if not np.isfinite(result):
        raise ValueError('Invalid '+name)
    return result


def _image_size(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in value)):
        raise ValueError('Invalid image_size')
    return list(value)


class CalibratedRgbGeometry:
    """Compute fresh same-frame geometry; never open devices or reuse a pose."""

    def __init__(self, config, calibration_report):
        if not isinstance(config, dict) or not isinstance(calibration_report, dict):
            raise ValueError('Expected configuration and calibration dictionaries')
        camera = config.get('camera', {})
        calibrated_camera = calibration_report.get('camera', {})
        self.image_size = _image_size([camera.get('width'), camera.get('height')])
        if (calibration_report.get('quality_passed') is not True
                or _image_size(calibration_report.get('image_size')) != self.image_size
                or _image_size([calibrated_camera.get('width'),
                                calibrated_camera.get('height')]) != self.image_size):
            raise ValueError('Calibration must pass quality checks and match image size')
        for key in ('backend', 'device', 'format'):
            if camera.get(key) != calibrated_camera.get(key):
                raise ValueError('Calibration camera mode mismatch: '+key)
        if calibration_report.get('distortion_model') != 'opencv_brown_5':
            raise ValueError('Only calibrated opencv_brown_5 distortion is supported')
        try:
            self.camera_matrix = np.asarray(calibration_report['camera_matrix'], dtype=float)
            self.distortion = np.asarray(calibration_report['distortion_coefficients'], dtype=float)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError('Invalid camera intrinsics') from exc
        width, height = self.image_size
        matrix = self.camera_matrix
        if (matrix.shape != (3, 3) or self.distortion.shape != (5,)
                or not np.isfinite(matrix).all() or not np.isfinite(self.distortion).all()
                or not np.allclose(matrix[2], [0., 0., 1.], atol=1e-12, rtol=0)
                or abs(matrix[0, 1]) > 1e-12 or abs(matrix[1, 0]) > 1e-12
                or not (.05*width <= matrix[0, 0] <= 10*width)
                or not (.05*height <= matrix[1, 1] <= 10*height)
                or not (0 <= matrix[0, 2] < width and 0 <= matrix[1, 2] < height)
                or (np.abs(self.distortion[[0, 1, 4]]) > 5).any()
                or (np.abs(self.distortion[[2, 3]]) > .2).any()):
            raise ValueError('Nonfinite or unreasonable camera intrinsics')
        board_config = config.get('board')
        calibrated_board = calibration_report.get('board')
        if not isinstance(board_config, dict) or not isinstance(calibrated_board, dict):
            raise ValueError('Missing calibrated board configuration')
        for key in ('type', 'dictionary', 'squares_x', 'squares_y', 'legacy_pattern'):
            if key not in board_config or board_config[key] != calibrated_board.get(key):
                raise ValueError('Calibrated printed board mismatch: '+key)
        for key in ('square_length_m', 'marker_length_m'):
            if not np.isclose(_scalar(board_config.get(key), key),
                              _scalar(calibrated_board.get(key), key), atol=1e-12, rtol=0):
                raise ValueError('Calibrated printed board mismatch: '+key)
        observer = BoardRgbObserver({'board': board_config})
        points = (observer.board.getChessboardCorners()
                  if hasattr(observer.board, 'getChessboardCorners')
                  else observer.board.chessboardCorners)
        self.board_points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.minimum_corners = observer.minimum
        self.board_config = dict(board_config)
        cup = config.get('cup', {})
        self.cup_height_m = _scalar(cup.get('height_m'), 'cup height')
        if not 0 < self.cup_height_m < 1:
            raise ValueError('Invalid cup height')
        self.approximate_radius_m = cup.get('top_radius_m')
        if self.approximate_radius_m is not None:
            self.approximate_radius_m = _scalar(self.approximate_radius_m, 'cup radius')
            if not 0 < self.approximate_radius_m < 1:
                raise ValueError('Invalid cup radius')

    def _pixels(self, value, minimum):
        points = np.asarray(value, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum
                or not np.isfinite(points).all() or (points < 0).any()
                or (points[:, 0] >= self.image_size[0]).any()
                or (points[:, 1] >= self.image_size[1]).any()):
            raise ValueError('invalid_pixel_points')
        return points

    def _board_pose(self, observation):
        if not isinstance(observation, dict) or observation.get('valid') is not True:
            raise ValueError('board_observation_invalid')
        if _image_size(observation.get('image_size')) != self.image_size:
            raise ValueError('board_image_size_mismatch')
        raw_ids = observation.get('charuco_corner_ids')
        if (not isinstance(raw_ids, (list, tuple)) or len(raw_ids) < self.minimum_corners
                or any(not isinstance(i, int) or isinstance(i, bool) for i in raw_ids)
                or len(set(raw_ids)) != len(raw_ids)
                or any(i < 0 or i >= len(self.board_points) for i in raw_ids)):
            raise ValueError('invalid_charuco_corner_ids')
        pixels = self._pixels(observation.get('charuco_corners_px'), self.minimum_corners)
        if len(raw_ids) != len(pixels):
            raise ValueError('charuco_id_point_count_mismatch')
        objects = self.board_points[raw_ids]
        if np.linalg.matrix_rank(objects[:, :2]-objects[:, :2].mean(axis=0)) < 2:
            raise ValueError('collinear_board_model_corners')
        if np.linalg.matrix_rank(pixels-pixels.mean(axis=0), tol=1e-7) < 2:
            raise ValueError('collinear_charuco_pixels')
        success, rvec, tvec = cv2.solvePnP(
            objects, pixels, self.camera_matrix, self.distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            raise ValueError('board_pnp_failed')
        rotation = cv2.Rodrigues(rvec)[0]
        translation = tvec.reshape(3)
        if ((objects @ rotation.T+translation)[:, 2] <= 0).any():
            raise ValueError('board_behind_camera')
        if rotation[2, 2] <= 0:
            raise ValueError('board_printed_face_normal_reversed')
        reprojection = cv2.projectPoints(
            objects, rvec, tvec, self.camera_matrix, self.distortion)[0].reshape(-1, 2)
        errors = np.linalg.norm(reprojection-pixels, axis=1)
        rms = float(np.sqrt(np.mean(errors**2)))
        if not np.isfinite(rms) or rms > 1.2 or errors.max() > 3.:
            raise ValueError('board_reprojection_error_too_large')
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, translation
        pose = {'valid': True, 'reason': 'ok', 'T_camera_board': transform.tolist(),
                'reprojection_rms_px': rms, 'maximum_reprojection_error_px': float(errors.max()),
                'corner_count': len(raw_ids), 'board_origin': 'printed_pattern_corner',
                'board_positive_z': 'into_table',
                'metric_accuracy_independently_validated': False}
        return pose, rotation, translation

    def _project_to_plane(self, pixels, rotation, translation, height):
        """Undistort pixels, then intersect camera rays with board Z=-height."""
        pixels = self._pixels(pixels, 1)
        if hasattr(cv2, 'undistortPointsIter'):
            normalized = cv2.undistortPointsIter(
                pixels.reshape(-1, 1, 2), self.camera_matrix, self.distortion, None, None,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-12))
        else:
            normalized = cv2.undistortPoints(
                pixels.reshape(-1, 1, 2), self.camera_matrix, self.distortion)
        normalized = normalized.reshape(-1, 2)
        if not np.isfinite(normalized).all():
            raise ValueError('undistortion_failed')
        camera_rays = np.column_stack((normalized, np.ones(len(normalized))))
        # Check the inverse-distortion result rather than trusting convergence.
        check = cv2.projectPoints(camera_rays, np.zeros(3), np.zeros(3),
                                  self.camera_matrix, self.distortion)[0].reshape(-1, 2)
        if np.linalg.norm(check-pixels, axis=1).max() > .01:
            raise ValueError('undistortion_did_not_converge')
        origin = -rotation.T @ translation
        rays = camera_rays @ rotation
        if (np.abs(rays[:, 2]) < 1e-8).any():
            raise ValueError('ray_parallel_to_height_plane')
        distance = (-height-origin[2])/rays[:, 2]
        points = origin+distance[:, None]*rays
        if (not np.isfinite(points).all() or (distance <= 0).any()
                or ((points @ rotation.T+translation)[:, 2] <= 0).any()):
            raise ValueError('height_plane_intersection_behind_camera')
        return points

    def _cup_geometry(self, observation, rotation, translation):
        if not isinstance(observation, dict) or observation.get('valid') is not True:
            raise ValueError('cup_observation_invalid')
        if ('image_size' in observation
                and _image_size(observation['image_size']) != self.image_size):
            raise ValueError('cup_image_size_mismatch')
        face_height = self.board_config.get('board_face_above_cup_support_m')
        if face_height is None:
            raise ValueError('board_face_above_cup_support_height_unknown')
        if self.board_config.get('previous_setup_height_reverified') is not True:
            raise ValueError('current_board_face_above_cup_support_height_unverified')
        face_height = _scalar(face_height, 'board_face_above_cup_support_height')
        if not 0 <= face_height < self.cup_height_m:
            raise ValueError('invalid_board_face_above_cup_support_height')
        height = self.cup_height_m-face_height
        pixels = self._pixels(observation.get('edge_points_px'), 12)
        if len(np.unique(pixels, axis=0)) < 12:
            raise ValueError('too_few_distinct_cup_edge_points')
        points = self._project_to_plane(pixels, rotation, translation, height)
        xy = points[:, :2]
        mean = xy.mean(axis=0)
        centered = xy-mean
        # Center the fit for numerical stability at positions far from the board.
        design = np.column_stack((2*centered, np.ones(len(centered))))
        fitted, _, rank, _ = np.linalg.lstsq(design, (centered**2).sum(axis=1), rcond=None)
        if rank < 3:
            raise ValueError('degenerate_cup_edge_circle')
        center = fitted[:2]+mean
        distances = np.linalg.norm(xy-center, axis=1)
        radius = float(distances.mean())
        rms = float(np.sqrt(np.mean((distances-radius)**2)))
        if not np.isfinite([*center, radius, rms]).all():
            raise ValueError('nonfinite_cup_circle')
        if not .008 <= radius <= .060:
            raise ValueError('cup_circle_radius_out_of_range')
        if rms > .002:
            raise ValueError('cup_circle_radial_error_too_large')
        angles = np.sort(np.arctan2(xy[:, 1]-center[1], xy[:, 0]-center[0]))
        largest_gap = np.diff(np.r_[angles, angles[0]+2*np.pi]).max()
        coverage = float(2*np.pi-largest_gap)
        if coverage < np.pi:
            raise ValueError('insufficient_cup_edge_angular_coverage')
        return {'valid': True, 'reason': 'ok',
                'center_board_m': [float(center[0]), float(center[1]), -height],
                'radius_m': radius, 'radial_rms_m': rms, 'edge_point_count': len(points),
                'edge_angular_coverage_deg': float(np.degrees(coverage)),
                'top_above_board_m': height,
                'source': 'undistorted_edge_rays_height_plane_circle_fit',
                'source_observation': observation.get('source'),
                'approximate_user_radius_m': self.approximate_radius_m,
                'radius_minus_approximate_user_radius_m': (
                    radius-self.approximate_radius_m
                    if self.approximate_radius_m is not None else None),
                'metric_accuracy_independently_validated': False,
                'motion_target_valid': False}

    def observe(self, board_observation, cup_observation):
        """Return JSON-compatible geometry from these observations exclusively."""
        result = {'geometry_valid': False, 'reason': None, 'coordinate_space': 'board_meters',
                  'board_pose': {'valid': False, 'reason': None, 'T_camera_board': None},
                  'cup_top': {'valid': False, 'reason': 'board_pose_unavailable',
                              'center_board_m': None},
                  'independent_metric_accuracy_validated': False, 'motion_target_valid': False}
        try:
            pose, rotation, translation = self._board_pose(board_observation)
        except (ValueError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
            result['board_pose']['reason'] = str(exc)
            result['reason'] = 'board_pose_invalid'
            return result
        result['board_pose'] = pose
        try:
            result['cup_top'] = self._cup_geometry(cup_observation, rotation, translation)
        except (ValueError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
            result['cup_top']['reason'] = str(exc)
            result['reason'] = 'cup_metric_geometry_invalid'
            return result
        result['geometry_valid'] = True
        result['reason'] = 'ok'
        return result
