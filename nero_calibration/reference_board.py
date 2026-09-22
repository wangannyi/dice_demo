#!/usr/bin/env python3
"""Observe/register a fixed table board and restore extrinsics. Never opens CAN."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from nero_calibration.calibrate import write_image_new, write_new
from nero_calibration.core import average, distance, inverse, matrix, matrix_pose


GEOMETRY_KEYS = ('type', 'dictionary', 'squares_x', 'squares_y',
                 'square_length_m', 'marker_length_m', 'legacy_pattern')
DIRECTION = 'T_base_camera maps camera optical coordinates into arm base'


def same_camera_geometry(first, second):
    """FPS changes acquisition timing, not the spatial camera calibration."""
    return ({key: value for key, value in first.items() if key != 'fps'} ==
            {key: value for key, value in second.items() if key != 'fps'})


def load_source(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), {'path': str(Path(path).resolve()),
                             'sha256': hashlib.sha256(raw).hexdigest()}


def require_quality(calibration, allow_provisional):
    if (calibration.get('schema') != 1 or calibration.get('mode') != 'eye_to_hand'
            or calibration.get('direction') != DIRECTION
            or type(calibration.get('quality_passed')) is not bool):
        raise ValueError('Expected an eye_to_hand result with explicit quality status')
    matrix(calibration['T_base_camera'])
    matrix(calibration['T_flange_tcp'])
    if not calibration['quality_passed'] and not allow_provisional:
        raise ValueError('Failed hand-eye quality requires --allow-provisional')


def summarize(records):
    if len(records) < 10:
        raise ValueError('Need at least 10 reference observations')
    poses = [matrix(r['T_camera_board']) for r in records]
    for r in records:
        q = r['quality']
        if (q['corners'] < 12 or not np.isfinite(q['reprojection_rms_px'])
                or not 0 <= q['reprojection_rms_px'] <= .5):
            raise ValueError('Reference requires 12 corners and RMS <= 0.5 px')
    mean = average(poses)
    errors = [distance(mean, t) for t in poses]
    position = max(e[0] for e in errors)
    angle = max(e[1] for e in errors)
    if position > .002 or angle > 1.:
        raise ValueError('Reference observations unstable: require <= 2 mm / 1 degree')
    return mean, {'frame_count': len(records), 'max_position_deviation_mm': position*1000,
                  'max_angle_deviation_deg': angle,
                  'max_reprojection_rms_px': max(r['quality']['reprojection_rms_px']
                                                 for r in records)}


def check_observation(observation):
    if (observation.get('schema') != 1 or observation.get('mode') != 'reference_observation'
            or not observation.get('reference_id') or not observation.get('opencv_version')):
        raise ValueError('Expected a named reference-board observation')
    camera = observation['camera']
    if (camera.get('backend') != 'realsense' or camera.get('frame') != 'color_optical'
            or not camera.get('serial')):
        raise ValueError('Expected a RealSense color optical camera')
    mean, quality = summarize(observation['records'])
    if not np.allclose(mean, matrix(observation['T_camera_board']), atol=1e-9, rtol=0):
        raise ValueError('Reference mean disagrees with saved observations')
    return mean, quality


def register(calibration, observation, allow_provisional=False):
    require_quality(calibration, allow_provisional)
    cb, quality = check_observation(observation)
    if not same_camera_geometry(calibration['camera'], observation['camera']):
        raise ValueError('Camera identity/intrinsics differ from hand-eye calibration')
    recorded_version = calibration.get('board_measurement', {}).get('opencv_version')
    if recorded_version and recorded_version != observation['opencv_version']:
        raise ValueError('OpenCV board coordinate conventions must match')
    return {'schema': 1, 'mode': 'fixed_reference_registration',
            'reference_id': observation['reference_id'], 'board': observation['board'],
            'opencv_version': observation['opencv_version'],
            'T_base_board': (matrix(calibration['T_base_camera']) @ cb).tolist(),
            'T_camera_board_at_registration': cb.tolist(),
            'camera_at_registration': observation['camera'],
            'source_calibration': copy.deepcopy(calibration),
            'quality_passed': calibration['quality_passed'],
            'provisional_opt_in': bool(allow_provisional),
            'observation_quality': quality, 'independent_accuracy_verified': False,
            'created_unix_s': time.time()}


def restore(registration, observation, allow_provisional=False):
    if (registration.get('schema') != 1
            or registration.get('mode') != 'fixed_reference_registration'):
        raise ValueError('Expected a fixed-reference registration')
    calibration = registration['source_calibration']
    require_quality(calibration, allow_provisional)
    cb, quality = check_observation(observation)
    if (registration['reference_id'] != observation['reference_id']
            or any(registration['board'][k] != observation['board'][k] for k in GEOMETRY_KEYS)):
        raise ValueError('Reference identity/geometry changed; register again')
    if registration['opencv_version'] != observation['opencv_version']:
        raise ValueError('OpenCV board coordinate conventions must match')
    if not same_camera_geometry(registration['camera_at_registration'], observation['camera']):
        raise ValueError('Camera identity/intrinsics changed; recalibrate before reuse')
    bc = matrix(registration['T_base_board']) @ inverse(cb)
    result = copy.deepcopy(calibration)
    result.update({'T_base_camera': bc.tolist(), 'pose_base_camera_m_rad': matrix_pose(bc),
                   'camera': observation['camera'], 'created_unix_s': time.time(),
                   'extrinsic_method': 'fixed_reference_restore',
                   'reference_id': registration['reference_id'],
                   'provisional_opt_in': bool(allow_provisional),
                   'reference_observation_quality': quality,
                   'independent_accuracy_verified': False,
                   'residual_scope': 'Retained residuals describe the source hand-eye fit, not the restored camera pose.'})
    return result


def observe(board_path, serial, reference_id, output, frame_count):
    from nero_calibration.sensors import CharucoDetector, RealSenseCamera
    if not 10 <= frame_count <= 120:
        raise ValueError('--frames must be between 10 and 120')
    cfg, board_source = load_source(board_path)
    if not reference_id.strip() or cfg.get('image_roi_xyxy') is None:
        raise ValueError('A reference ID and an explicit board ROI are required')
    if cfg['squares_x'] != 4 or cfg['squares_y'] != 5:
        raise ValueError('This reference workflow requires the current 4x5 board')
    detector = CharucoDetector(cfg)
    output.mkdir(parents=True, exist_ok=False)
    camera = None
    try:
        camera = RealSenseCamera(serial, **({"image_profile": cfg["image_profile"]} if "image_profile" in cfg else {}))
        records = []
        for i in range(frame_count):
            image = camera.capture()
            write_image_new(output/f'frame_{i:03d}.png', image)
            t, quality, vis = detector.detect(image, camera.K, camera.D)
            write_image_new(output/f'frame_{i:03d}_detected.png', vis)
            records.append({'T_camera_board': t.tolist(), 'quality': quality,
                            'time_unix_s': time.time()})
        mean, quality = summarize(records)
        record = {'schema': 1, 'mode': 'reference_observation', 'reference_id': reference_id,
                  'board': cfg, 'board_source': board_source, 'opencv_version': cv2.__version__,
                  'camera': camera.info, 'records': records, 'T_camera_board': mean.tolist(),
                  'quality': quality, 'created_unix_s': time.time()}
        write_new(output/'observation.json', record)
        return record
    finally:
        if camera is not None:
            camera.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    obs = sub.add_parser('observe', help='Capture stationary RGB reference frames')
    obs.add_argument('--board', type=Path, required=True)
    obs.add_argument('--serial', required=True)
    obs.add_argument('--reference-id', required=True)
    obs.add_argument('--frames', type=int, default=10)
    obs.add_argument('--output', type=Path, required=True, help='New output directory')
    for command, source in [('register', 'calibration'), ('restore', 'registration')]:
        p = sub.add_parser(command, help='Offline transform calculation; no hardware access')
        p.add_argument('--'+source, type=Path, required=True)
        p.add_argument('--observation', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True, help='New output JSON')
        p.add_argument('--allow-provisional', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'observe':
        record = observe(args.board, args.serial, args.reference_id, args.output, args.frames)
        print(json.dumps({'output': str(args.output), **record['quality']}))
    else:
        source_name = 'calibration' if args.command == 'register' else 'registration'
        source, origin = load_source(getattr(args, source_name))
        observation, obs_origin = load_source(args.observation)
        function = register if args.command == 'register' else restore
        result = function(source, observation, args.allow_provisional)
        result['sources'] = {source_name: origin, 'observation': obs_origin}
        write_new(args.output, result)
        print(json.dumps({'output': str(args.output), 'quality_passed': result['quality_passed'],
                          'provisional_opt_in': result['provisional_opt_in']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
