"""Read-only arm feedback and camera capture; no motion/mode/enable calls."""
import time
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from nero_calibration.core import distance, pose_matrix


class NeroFeedback:
    def __init__(self, channel):
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
        self.robot = AgxArmFactory.create_arm(create_agx_arm_config(
            robot=ArmModel.NERO, firmeware_version=NeroFW.V120,
            interface='socketcan', channel=channel))
        self.last_timestamp = None
        try:
            self.robot.connect()
            # Let all multi-frame fields initialize, then require fresh observations.
            time.sleep(1.)
            for _ in range(5):
                self.read()
        except BaseException:
            self.close()
            raise

    def read(self):
        deadline = time.monotonic() + .5
        while time.monotonic() < deadline:
            if self.robot.has_comm_error():
                raise RuntimeError('CAN feedback error')
            msg = self.robot.get_flange_pose()
            if msg is not None and msg.timestamp != self.last_timestamp:
                self.last_timestamp = msg.timestamp
                return pose_matrix(list(msg.msg))
            time.sleep(.01)
        raise RuntimeError('No new flange feedback within 500 ms')

    def read_joints(self):
        """Require a newly received seven-axis observation; never query/move joints."""
        first = self.robot.get_joint_angles()
        stamp = None if first is None else first.timestamp
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if self.robot.has_comm_error():
                raise ValueError('CAN error while recording calibration joints')
            msg = self.robot.get_joint_angles()
            if msg is not None and (first is None or msg.timestamp != stamp):
                q = np.asarray(list(msg.msg), dtype=float)
                if q.shape != (7,) or not np.isfinite(q).all():
                    raise ValueError('Invalid seven-axis calibration feedback')
                return dict(joints_rad=q.tolist(), joints_deg=np.degrees(q).tolist(),
                            sdk_timestamp=msg.timestamp, observed_epoch_s=time.time())
            time.sleep(.01)
        raise ValueError('No fresh joints for calibration pose recording')

    def close(self):
        self.robot.disconnect()


def assert_still(poses, position_m=.0005, angle_deg=.2):
    if len(poses) < 2:
        raise ValueError('Need multiple stationary observations')
    if any(p > position_m or a > angle_deg for p, a in
           (distance(poses[0], t) for t in poses[1:])):
        raise ValueError('Arm moved during capture; sample rejected')


def opencv_distortion(model, coeffs):
    coeffs = np.asarray(coeffs, dtype=float)
    if coeffs.shape != (5,) or not np.isfinite(coeffs).all():
        raise ValueError('Invalid camera distortion coefficients')
    if model in ('distortion.none', 'distortion.brown_conrady'):
        return np.zeros(5) if model == 'distortion.none' else coeffs
    if model in ('distortion.inverse_brown_conrady', 'distortion.modified_brown_conrady') and np.all(coeffs == 0):
        return np.zeros(5)
    raise ValueError(f'Unsupported nonzero camera distortion model: {model}; rectify first')


class RealSenseCamera:
    def __init__(self, serial=None, image_profile=None):
        import pyrealsense2 as rs
        from nero_calibration.image_profile import profile_options
        resolution, fps, self.crop = profile_options(image_profile)
        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if not serial and len(rs.context().query_devices()) != 1:
            raise ValueError('Expected one RealSense camera; specify --serial')
        if serial:
            cfg.enable_device(serial)
        # RGB only: hand-eye board pose does not need depth.
        cfg.enable_stream(rs.stream.color, *resolution, rs.format.bgr8, fps)
        try:
            profile = self.pipeline.start(cfg)
            self.started = True
            stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = stream.get_intrinsics()
            self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
            if self.crop is not None:
                self.K[0, 2] -= self.crop[0]
                self.K[1, 2] -= self.crop[1]
            self.D = opencv_distortion(str(intr.model), intr.coeffs)
            self.info = {'backend': 'realsense', 'serial': profile.get_device().get_info(rs.camera_info.serial_number),
                         'frame': 'color_optical', 'width': intr.width, 'height': intr.height,
                         'camera_matrix': self.K.tolist(), 'dist_coeffs': self.D.tolist(),
                         'distortion_model': str(intr.model)}
            if self.crop is not None:
                self.info.update(width=self.crop[2], height=self.crop[3],
                                 source_resolution=resolution, crop_xywh=list(self.crop), fps=fps)
            for _ in range(15):
                self.capture()
        except BaseException:
            self.close()
            raise

    def capture(self):
        frame = self.pipeline.wait_for_frames(3000).get_color_frame()
        if not frame:
            raise RuntimeError('Missing RGB frame')
        from nero_calibration.image_profile import crop_image
        return crop_image(np.asanyarray(frame.get_data()), self.crop)

    def close(self):
        if getattr(self, 'started', False):
            self.pipeline.stop()
            self.started = False


class CharucoDetector:
    def __init__(self, cfg):
        import cv2
        self.cv = cv2
        self.cfg = cfg
        if cfg.get('type') != 'charuco':
            raise ValueError('Only charuco is supported')
        sx, sy = cfg['squares_x'], cfg['squares_y']
        sq, mk = cfg['square_length_m'], cfg['marker_length_m']
        if not (isinstance(sx, int) and isinstance(sy, int) and sx >= 3 and sy >= 3
                and 0 < mk < sq < 1 and cfg.get('min_corners', 6) >= 4
                and cfg.get('max_reprojection_px', 1.) > 0):
            raise ValueError('Invalid board dimensions/quality thresholds')
        dictionaries = {n: getattr(cv2.aruco, 'DICT_'+n.upper())
                        for n in ['4x4_50', '4x4_100', '5x5_50', '5x5_100', '6x6_50', '6x6_100']}
        # OpenCV constants use uppercase X (DICT_4X4_50).
        if cfg['dictionary'] not in dictionaries:
            raise ValueError('Unsupported dictionary')
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionaries[cfg['dictionary']])
        if hasattr(cv2.aruco, 'CharucoBoard_create'):
            self.board = cv2.aruco.CharucoBoard_create(sx, sy, sq, mk, self.dictionary)
        else:
            self.board = cv2.aruco.CharucoBoard((sx, sy), sq, mk, self.dictionary)
        if hasattr(self.board, 'setLegacyPattern'):
            self.board.setLegacyPattern(bool(cfg.get('legacy_pattern', False)))
        elif cfg.get('legacy_pattern', False) is False and sy % 2 == 0:
            raise ValueError('Even-row modern pattern requires newer OpenCV; use matching legacy_pattern')
        self.points = (self.board.getChessboardCorners() if hasattr(self.board, 'getChessboardCorners')
                       else self.board.chessboardCorners)
        self.roi = cfg.get('image_roi_xyxy')
        if self.roi is not None:
            if (not isinstance(self.roi, (list, tuple)) or len(self.roi) != 4
                    or any(type(v) is not int for v in self.roi)
                    or not (0 <= self.roi[0] < self.roi[2]
                            and 0 <= self.roi[1] < self.roi[3])):
                raise ValueError('image_roi_xyxy must be integer [x0, y0, x1, y1]')

        self.excluded = cfg.get('image_exclude_rois_xyxy', [])
        if not isinstance(self.excluded, list):
            raise ValueError('image_exclude_rois_xyxy must be a list')
        for box in self.excluded:
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or any(type(v) is not int for v in box)
                    or not 0 <= box[0] < box[2] or not 0 <= box[1] < box[3]):
                raise ValueError('Invalid excluded detection rectangle')

    def detect(self, image, K, D):
        cv = self.cv
        detection_image = image
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            if x1 > image.shape[1] or y1 > image.shape[0]:
                raise ValueError('image_roi_xyxy exceeds image bounds')
            # Keep full-image coordinates and intrinsics; never fall back to another board.
            detection_image = np.full_like(image, 255)
            detection_image[y0:y1, x0:x1] = image[y0:y1, x0:x1]
        if self.excluded:
            detection_image = detection_image.copy()
            for ex0, ey0, ex1, ey1 in self.excluded:
                if ex1 > image.shape[1] or ey1 > image.shape[0]:
                    raise ValueError('Excluded detection rectangle exceeds image bounds')
                detection_image[ey0:ey1, ex0:ex1] = 255
        corners, ids, _ = cv.aruco.detectMarkers(detection_image, self.dictionary)
        if ids is None:
            raise ValueError('No ChArUco markers detected')
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate marker IDs; isolate one board with image_roi_xyxy')
        _, cc, ci = cv.aruco.interpolateCornersCharuco(corners, ids, detection_image, self.board,
                                                      cameraMatrix=K, distCoeffs=D)
        if ci is None or len(ci) < self.cfg.get('min_corners', 6):
            raise ValueError('Too few ChArUco corners')
        obj = np.asarray(self.points[ci.ravel()], dtype=np.float64)
        img = np.asarray(cc, dtype=np.float64).reshape(-1, 2)
        if np.linalg.matrix_rank(obj-obj.mean(axis=0), tol=1e-7) < 2:
            raise ValueError('Detected corners are collinear')
        ok, rv, tv = cv.solvePnP(obj, img, K, D, flags=cv.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(tv).all():
            raise ValueError('Board pose solve failed')
        R = cv.Rodrigues(rv)[0]
        if np.min((R @ obj.T + tv)[2]) <= 0:
            raise ValueError('Board behind camera')
        projected = cv.projectPoints(obj, rv, tv, K, D)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((projected-img)**2, axis=1))))
        if not np.isfinite(rms) or rms > self.cfg.get('max_reprojection_px', 1.):
            raise ValueError(f'Reprojection RMS too large: {rms:.3f} px')
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = R, tv.ravel()
        vis = image.copy()
        cv.aruco.drawDetectedCornersCharuco(vis, cc, ci)
        cv.drawFrameAxes(vis, K, D, rv, tv, self.cfg['square_length_m'] * 2)
        quality = {'corners': len(ci), 'reprojection_rms_px': rms}
        if self.excluded:
            quality['image_exclude_rois_xyxy'] = self.excluded
            for ex0, ey0, ex1, ey1 in self.excluded:
                cv.rectangle(vis, (ex0, ey0), (ex1-1, ey1-1), (0, 0, 255), 2)
            cv.putText(vis, 'RED BOX: excluded fixed board', (8, image.shape[0]-12),
                       cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
        if self.roi is not None:
            quality['image_roi_xyxy'] = list(self.roi)
            cv.rectangle(vis, (x0, y0), (x1-1, y1-1), (255, 128, 0), 1)
        return t, quality, vis
