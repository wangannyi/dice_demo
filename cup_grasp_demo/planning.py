"""Pure camera-to-base geometry and TCP inversion for a *proposed* cup target.

No camera, CAN, SDK, segmentation, grasp-offset, or motion dependency lives here.
The caller must choose the base-frame palm target position and orientation.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from cup_grasp_demo.flow.transforms import inverse, matrix


def _point(value, name):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f'{name} must be three finite meters')
    return point


def _rotation(value, name):
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f'{name} must be a rigid 3x3 rotation')
    candidate = np.eye(4)
    candidate[:3, :3] = rotation
    return matrix(candidate)[:3, :3]


def validate_calibration(result, *, expected_camera_serial, allow_provisional=False):
    """Validate an eye-to-hand result; failed quality requires explicit override."""
    if not isinstance(result, dict) or result.get('schema') != 1 or result.get('mode') != 'eye_to_hand':
        raise ValueError('Expected eye_to_hand calibration schema 1')
    if type(result.get('quality_passed')) is not bool:
        raise ValueError('Calibration quality_passed must be boolean')
    if not result['quality_passed'] and not allow_provisional:
        raise ValueError('Calibration did not pass validation; set allow_provisional explicitly')
    if result.get('direction') != 'T_base_camera maps camera optical coordinates into arm base':
        raise ValueError('Calibration transform direction is not camera to base')
    camera = result.get('camera')
    if not isinstance(camera, dict) or camera.get('frame') != 'color_optical':
        raise ValueError('Calibration camera must be color_optical')
    if camera.get('backend') != 'realsense':
        raise ValueError('Expected RealSense calibration camera')
    serial = camera.get('serial')
    if not isinstance(serial, str) or not serial:
        raise ValueError('Calibration camera serial is missing')
    if not isinstance(expected_camera_serial, str) or not expected_camera_serial:
        raise ValueError('Expected camera serial is required')
    if serial != expected_camera_serial:
        raise ValueError('Calibration camera serial does not match live camera')
    if any(not isinstance(camera.get(k), int) or camera[k] <= 0 for k in ('width', 'height')):
        raise ValueError('Invalid calibrated camera resolution')
    intrinsics = np.asarray(camera.get('camera_matrix'), dtype=float)
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0):
        raise ValueError('Invalid calibrated camera intrinsics')
    distortion = np.asarray(camera.get('dist_coeffs'), dtype=float)
    if distortion.shape != (5,) or not np.isfinite(distortion).all():
        raise ValueError('Invalid calibrated distortion coefficients')
    matrix(result['T_base_camera'])
    matrix(result['T_flange_tcp'])
    return result


def load_calibration(path, *, expected_camera_serial, allow_provisional=False):
    """Load one result without modifying it; include a digest for audit logs."""
    source = Path(path)
    raw = source.read_bytes()
    result = validate_calibration(json.loads(raw), expected_camera_serial=expected_camera_serial,
                                  allow_provisional=allow_provisional).copy()
    result['_source'] = {'path': str(source.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}
    return result


def map_localization(localization, calibration, *, camera_serial, allow_provisional=False):
    """Map a geometry.localize output from color optical coordinates to arm base."""
    validate_calibration(calibration, expected_camera_serial=camera_serial,
                         allow_provisional=allow_provisional)
    if not isinstance(localization, dict) or localization.get('schema_version') != 1:
        raise ValueError('Expected localization schema 1')
    if localization.get('valid') is not True or not isinstance(localization.get('geometry'), dict):
        raise ValueError('Localization is invalid')
    if localization.get('frame') != 'color_optical' or localization.get('units') != 'm':
        raise ValueError('Localization must be meters in color_optical')
    if (localization.get('camera_serial') is not None
            and localization['camera_serial'] != camera_serial):
        raise ValueError('Localization camera serial does not match live camera')
    geom = localization['geometry']
    center = _point(geom['center_m'], 'cup center')
    support = _point(geom['support_center_m'], 'cup support center')
    axis = _point(geom['axis'], 'cup axis')
    norm = np.linalg.norm(axis)
    if not np.isclose(norm, 1.0, atol=1e-3, rtol=0):
        raise ValueError('Cup axis must be a unit vector')
    transform = matrix(calibration['T_base_camera'])
    rotation, translation = transform[:3, :3], transform[:3, 3]
    return {
        'frame': 'base', 'units': 'm',
        'center_m': (rotation @ center + translation).tolist(),
        'support_center_m': (rotation @ support + translation).tolist(),
        'axis': (rotation @ (axis / norm)).tolist(),
        'dimensions': geom.get('dimensions'),
        'localization_quality': geom.get('quality'),
        'source': {'camera_serial': camera_serial, 'camera_frame': 'color_optical',
                   'frame_id': localization.get('frame_id'),
                   'instance_id': localization.get('instance_id'),
                   'timestamp_ms': localization.get('timestamp_ms'),
                   'timestamp_domain': localization.get('timestamp_domain'),
                   'quality_passed': calibration['quality_passed'],
                   'calibration': calibration.get('_source')},
        'T_base_camera': transform.tolist(),
    }


def plan_palm_target(localization, calibration, *, camera_serial,
                     palm_position_base_m, current_T_base_palm=None,
                     palm_orientation_base=None, allow_provisional=False):
    """Form a palm target and corresponding flange target; send no command.

    Exactly one orientation source is required. `current_T_base_palm` preserves
    the current palm orientation while replacing only the proposed position.
    `palm_orientation_base` is an explicit 3x3 base-frame desired orientation.
    """
    if (current_T_base_palm is None) == (palm_orientation_base is None):
        raise ValueError('Specify exactly one palm orientation source')
    if calibration.get('tcp') != 'palm':
        raise ValueError('Palm planner requires a palm TCP calibration')
    mapped = map_localization(localization, calibration, camera_serial=camera_serial,
                              allow_provisional=allow_provisional)
    palm_target = np.eye(4)
    palm_target[:3, 3] = _point(palm_position_base_m, 'palm target position')
    if current_T_base_palm is not None:
        palm_target[:3, :3] = matrix(current_T_base_palm)[:3, :3]
        orientation_source = 'current_palm_orientation'
    else:
        palm_target[:3, :3] = _rotation(palm_orientation_base, 'palm orientation')
        orientation_source = 'specified_palm_orientation'
    flange_tcp = matrix(calibration['T_flange_tcp'])
    flange_target = matrix(palm_target @ inverse(flange_tcp))
    return {
        'cup_base': mapped,
        'tcp': calibration.get('tcp', 'unknown'),
        'orientation_source': orientation_source,
        'T_flange_tcp': flange_tcp.tolist(),
        'T_base_palm_target': palm_target.tolist(),
        'T_base_flange_target': flange_target.tolist(),
        'source_quality_passed': calibration['quality_passed'],
        'provisional': not calibration['quality_passed'],
        'calibration_holdout_errors_m_deg': calibration.get('holdout_errors_m_deg'),
        'calibration_thresholds': calibration.get('thresholds'),
    }
