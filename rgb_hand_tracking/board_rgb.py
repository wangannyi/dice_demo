"""Same-frame, read-only ChArUco observations in image pixels.

Board dimensions describe the printed pattern. They do not provide camera
intrinsics or make the returned observations a metric camera/robot pose.
"""
import cv2
import numpy as np


class BoardRgbObserver:
    """Observe an existing BGR frame without camera access or pose estimation."""

    def __init__(self, config, *, corner_convention='native'):
        if corner_convention not in ('native', 'opencv_4_6'):
            raise ValueError('Unsupported ChArUco corner convention')
        try:
            major, minor = (int(x) for x in cv2.__version__.split('.')[:2])
        except (ValueError, AttributeError) as exc:
            raise RuntimeError('Cannot determine native OpenCV corner convention') from exc
        self.corner_convention = corner_convention
        self.native_corner_convention = ('opencv_before_4_6' if (major, minor) < (4, 6)
                                         else 'opencv_4_6')
        self.remap_corner_ids = (corner_convention == 'opencv_4_6'
                                 and self.native_corner_convention == 'opencv_before_4_6')
        cfg = dict(config.get('board', config))
        if cfg.get('type', 'charuco') != 'charuco':
            raise ValueError('Only charuco boards are supported')
        sx, sy = cfg.get('squares_x', 4), cfg.get('squares_y', 5)
        square = cfg.get('square_length_m', .022)
        marker = cfg.get('marker_length_m', .0155)
        minimum = cfg.get('min_corners', 6)
        if not (isinstance(sx, int) and not isinstance(sx, bool)
                and isinstance(sy, int) and not isinstance(sy, bool)
                and sx >= 3 and sy >= 3 and np.isfinite([square, marker]).all()
                and 0 < marker < square < 1 and isinstance(minimum, int)
                and not isinstance(minimum, bool) and 4 <= minimum <= (sx-1)*(sy-1)):
            raise ValueError('Invalid board dimensions or min_corners')
        if not hasattr(cv2, 'aruco'):
            raise RuntimeError('OpenCV aruco support is required')
        name = cfg.get('dictionary', '4x4_50').lower()
        supported = {'4x4_50', '4x4_100', '5x5_50', '5x5_100', '6x6_50', '6x6_100'}
        if name not in supported:
            raise ValueError('Unsupported ArUco dictionary')
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, 'DICT_'+name.upper()))
        if hasattr(cv2.aruco, 'CharucoBoard_create'):
            self.board = cv2.aruco.CharucoBoard_create(sx, sy, square, marker, self.dictionary)
        else:
            self.board = cv2.aruco.CharucoBoard((sx, sy), square, marker, self.dictionary)
        legacy = bool(cfg.get('legacy_pattern', False))
        if hasattr(self.board, 'setLegacyPattern'):
            self.board.setLegacyPattern(legacy)
        elif sy % 2 == 0 and not legacy:
            raise ValueError('Modern even-row board requires newer OpenCV')
        ids = self.board.getIds() if hasattr(self.board, 'getIds') else self.board.ids
        self.board_marker_ids = set(int(x) for x in np.asarray(ids).ravel())
        self.minimum = minimum
        self.corner_columns = sx-1
        self.corner_rows = sy-1
        self.cfg = cfg
        self._aruco_detector = (cv2.aruco.ArucoDetector(self.dictionary)
                                if hasattr(cv2.aruco, 'ArucoDetector') else None)
        self._charuco_detector = (cv2.aruco.CharucoDetector(self.board)
                                  if hasattr(cv2.aruco, 'CharucoDetector') else None)

    def observe(self, bgr):
        """Return JSON-compatible pixel observations; invalid frames raise."""
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8
                or bgr.ndim != 3 or bgr.shape[2] != 3 or not bgr.size):
            raise ValueError('Expected a nonempty uint8 BGR image')
        if self._aruco_detector is not None:
            corners, ids, _ = self._aruco_detector.detectMarkers(bgr)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(bgr, self.dictionary)
        all_ids = [] if ids is None else [int(x) for x in np.asarray(ids).ravel()]
        matching = [i for i, marker_id in enumerate(all_ids)
                    if marker_id in self.board_marker_ids]
        marker_ids = [all_ids[i] for i in matching]
        marker_corners = [np.asarray(corners[i], dtype=float).reshape(4, 2).tolist()
                          for i in matching]
        result = {
            'valid': False,
            'reason': 'no_board_markers',
            'coordinate_space': 'image_pixels',
            'corner_convention': self.corner_convention,
            'native_corner_convention': self.native_corner_convention,
            'corner_ids_remapped': self.remap_corner_ids,
            'opencv_version': cv2.__version__,
            'image_size': [int(bgr.shape[1]), int(bgr.shape[0])],
            'detected_marker_ids': all_ids,
            'marker_ids': marker_ids,
            'marker_corners_px': marker_corners,
            'charuco_corner_ids': [],
            'charuco_corners_px': [],
            'charuco_corner_count': 0,
            'metric_pose_available': False,
            'motion_target_valid': False,
        }
        if not matching:
            return result
        if hasattr(cv2.aruco, 'interpolateCornersCharuco'):
            selected_corners = [corners[i] for i in matching]
            selected_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1, 1)
            _, cc, ci = cv2.aruco.interpolateCornersCharuco(
                selected_corners, selected_ids, bgr, self.board)
        else:
            # Recent OpenCV exposes detectBoard instead of the legacy function.
            # The detector has no cameraMatrix/distCoeffs and performs no PnP.
            if self._charuco_detector is None:
                raise RuntimeError('OpenCV ChArUco interpolation support is required')
            cc, ci, _, _ = self._charuco_detector.detectBoard(bgr)
        if ci is None or cc is None:
            result['reason'] = 'too_few_charuco_corners'
            return result
        points = np.asarray(cc, dtype=float).reshape(-1, 2)
        corner_ids = np.asarray(ci, dtype=np.int32).ravel()
        if self.remap_corner_ids:
            if ((corner_ids < 0).any()
                    or (corner_ids >= self.corner_columns*self.corner_rows).any()):
                result['reason'] = 'out_of_range_charuco_corner_ids'
                return result
            # OpenCV 4.6 changed the board coordinate/ID convention. Earlier
            # interpolation numbers corner rows from the opposite board edge.
            # Normalize correspondences before PnP, rather than flipping an
            # estimated pose or changing the physical height-plane sign.
            corner_ids = ((self.corner_rows-1-corner_ids//self.corner_columns)
                          * self.corner_columns+corner_ids % self.corner_columns)
            order = np.argsort(corner_ids, kind='stable')
            corner_ids, points = corner_ids[order], points[order]
        result['charuco_corner_ids'] = [int(x) for x in corner_ids]
        result['charuco_corners_px'] = points.tolist()
        result['charuco_corner_count'] = len(corner_ids)
        if len(set(result['charuco_corner_ids'])) != len(corner_ids):
            result['reason'] = 'duplicate_charuco_corner_ids'
        elif not np.isfinite(points).all():
            result['reason'] = 'nonfinite_charuco_corners'
        elif len(corner_ids) < self.minimum:
            result['reason'] = 'too_few_charuco_corners'
        elif np.linalg.matrix_rank(points-points.mean(axis=0), tol=1e-7) < 2:
            result['reason'] = 'collinear_charuco_corners'
        else:
            result['valid'] = True
            result['reason'] = 'ok'
        return result

    def annotate(self, bgr, observation):
        """Return a copy with board corners labelled; never modify input."""
        vis = bgr.copy()
        for marker_id, corners in zip(observation['marker_ids'], observation['marker_corners_px']):
            polygon = np.rint(corners).astype(np.int32)
            cv2.polylines(vis, [polygon], True, (0, 200, 255), 1)
            cv2.putText(vis, str(marker_id), tuple(polygon[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 200, 255), 1)
        for corner_id, point in zip(observation['charuco_corner_ids'], observation['charuco_corners_px']):
            pixel = tuple(np.rint(point).astype(int))
            cv2.circle(vis, pixel, 3, (0, 255, 0), -1)
            cv2.putText(vis, str(corner_id), (pixel[0]+4, pixel[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 0), 1)
        return vis
