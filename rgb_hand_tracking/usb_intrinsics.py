"""Offline USB RGB intrinsics calibration with dataset integrity and quality checks.

Five-parameter Brown distortion: k1, k2, p1, p2, k3; optionally fix k3 at zero.
No camera or robot access.
Holdout PnP residuals are pixel fit diagnostics, never metric accuracy evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from board_rgb import BoardRgbObserver


QUALITY_THRESHOLDS = {
    'minimum_views': 20,
    'maximum_training_rms_px': .8,
    'maximum_holdout_rms_px': 1.2,
    'minimum_corner_hull_fraction': .40,
    'minimum_corner_grid_cells_3x3': 6,
    'minimum_tilted_views_15deg': 4,
    'minimum_maximum_tilt_deg': 25.,
    'minimum_normal_spread_deg': 15.,
    'minimum_board_scale_ratio': 1.5,
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _board_points(observer):
    points = (observer.board.getChessboardCorners()
              if hasattr(observer.board, 'getChessboardCorners')
              else observer.board.chessboardCorners)
    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


def _validate_sample(root, sample, image_size, board_points):
    rel = sample.get('image_path')
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        raise ValueError('Sample image_path must be a relative dataset path')
    path = (root / rel).resolve()
    if root not in path.parents:
        raise ValueError('Sample image_path leaves the dataset')
    if not path.is_file():
        raise ValueError('Sample image is missing: '+rel)
    digest = _sha256(path)
    if sample.get('sha256_image') != digest:
        raise ValueError('Sample image SHA256 mismatch: '+rel)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or (image.shape[1], image.shape[0]) != image_size:
        raise ValueError('Actual sample image size mismatch: '+rel)
    obs = sample.get('observation', {})
    if obs.get('image_size') != list(image_size):
        raise ValueError('Observation image size mismatch: '+rel)
    timestamp = sample.get('timestamp_s')
    if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)):
        raise ValueError('Invalid sample timestamp: '+rel)
    identity = {'image_path': rel, 'sha256_image': digest,
                'timestamp_s': float(timestamp)}
    if obs.get('valid') is not True:
        return None, dict(identity, reason='invalid_board_observation')
    ids = obs.get('charuco_corner_ids', [])
    if (not isinstance(ids, list) or len(ids) < 6
            or any(not isinstance(x, int) or isinstance(x, bool)
                   or x < 0 or x >= len(board_points) for x in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('Invalid or duplicate ChArUco corner IDs: '+rel)
    try:
        corners = np.asarray(obs.get('charuco_corners_px'), dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid ChArUco corner coordinates: '+rel) from exc
    if (corners.shape != (len(ids), 2) or not np.isfinite(corners).all()
            or (corners < 0).any()
            or (corners[:, 0] >= image_size[0]).any()
            or (corners[:, 1] >= image_size[1]).any()
            or np.linalg.matrix_rank(corners-corners.mean(axis=0), tol=1e-5) < 2):
        raise ValueError('Invalid/out-of-image/collinear corner coordinates: '+rel)
    identity['observation_sha256'] = hashlib.sha256(json.dumps(
        {'ids': ids, 'corners': corners.tolist()}, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return dict(identity, object_points=board_points[ids], image_points=corners), None


def _residual(objects, pixels, rvec, tvec, matrix, distortion):
    predicted = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((predicted-pixels)**2, axis=1))))


def _pose_stats(rvec):
    rotation = cv2.Rodrigues(rvec)[0]
    normal = rotation[:, 2]
    # A planar board normal may face toward or away from the camera.
    if normal[2] < 0:
        normal = -normal
    tilt = math.degrees(math.acos(float(np.clip(normal[2], -1., 1.))))
    return normal, tilt


def solve_dataset(dataset_path, output_path, *, fix_k3=False):
    """Verify and fit a manifest dataset; write a NEW JSON report, never overwrite.

    dataset_path is a directory containing manifest.json, or the manifest itself.
    An unsuccessful quality check still writes all diagnostics with passed=false.
    Integrity failures or too few usable views raise ValueError and write nothing.
    fix_k3 estimates k1, k2, p1, p2 while constraining k3 to zero.
    """
    dataset = Path(dataset_path).resolve()
    manifest = dataset / 'manifest.json' if dataset.is_dir() else dataset
    root = manifest.parent
    output = Path(output_path)
    if output.exists():
        raise FileExistsError('Refusing to overwrite calibration output: '+str(output))
    cfg = json.loads(manifest.read_text())
    if cfg.get('schema') != 1:
        raise ValueError('Unsupported dataset manifest schema')
    camera = cfg.get('camera', {})
    if camera.get('backend') != 'v4l2_rgb':
        raise ValueError('Expected a v4l2_rgb USB dataset')
    width, height = camera.get('width'), camera.get('height')
    if (any(not isinstance(n, int) or isinstance(n, bool) or n < 64
            for n in (width, height))):
        raise ValueError('Invalid camera width/height')
    size = (width, height)
    board_config = cfg.get('board')
    if not isinstance(board_config, dict):
        raise ValueError('Board configuration is required')
    observer = BoardRgbObserver({'board': board_config})
    points = _board_points(observer)
    samples = cfg.get('samples')
    if not isinstance(samples, list):
        raise ValueError('samples must be a list')
    records, skipped = [], []
    seen_paths = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError('Each sample must be an object')
        if sample.get('image_path') in seen_paths:
            raise ValueError('Repeated sample image_path')
        seen_paths.add(sample.get('image_path'))
        record, reject = _validate_sample(root, sample, size, points)
        if record is not None:
            records.append(record)
        else:
            skipped.append(reject)
    if len(records) < 8:
        raise ValueError('At least 8 valid board views are needed to fit and hold out 3')
    # Choose throughout the sequence, rather than using only its final pose.
    holdout_count = max(3, int(math.ceil(len(records)*.2)))
    holdout_ids = set(int(x) for x in np.linspace(0, len(records)-1, holdout_count))
    training_ids = [i for i in range(len(records)) if i not in holdout_ids]
    objects = [records[i]['object_points'] for i in training_ids]
    pixels = [records[i]['image_points'] for i in training_ids]
    calibration_flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    calibration_options = {'flags': calibration_flags} if fix_k3 else {}
    try:
        rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            objects, pixels, size, None, None,
            criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 100, 1e-9),
            **calibration_options)
    except cv2.error as exc:
        raise ValueError('OpenCV calibration failed: '+str(exc)) from exc
    if (not math.isfinite(rms) or not np.isfinite(matrix).all()
            or not np.isfinite(distortion).all() or distortion.size != 5):
        raise ValueError('Calibration returned nonfinite/unsupported camera parameters')
    poses = {i: (rv, tv) for i, rv, tv in zip(training_ids, rvecs, tvecs)}
    views, normals, areas, depths, holdout_errors = [], [], [], [], []
    holdout_failed = False
    for i, record in enumerate(records):
        group = 'holdout' if i in holdout_ids else 'training'
        if group == 'holdout':
            ok, rv, tv = cv2.solvePnP(record['object_points'], record['image_points'],
                                     matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or not np.isfinite(rv).all() or not np.isfinite(tv).all():
                holdout_failed = True
                views.append(dict(image_path=record['image_path'], group=group,
                                  pnp_valid=False, rms_px=None, tilt_deg=None))
                continue
        else:
            rv, tv = poses[i]
        error = _residual(record['object_points'], record['image_points'], rv, tv,
                          matrix, distortion)
        normal, tilt = _pose_stats(rv)
        normals.append(normal)
        area = float(cv2.contourArea(cv2.convexHull(record['image_points'])))
        areas.append(area)
        depths.append(float(np.asarray(tv).ravel()[2]))
        if group == 'holdout':
            holdout_errors.append((error**2)*len(record['image_points']))
        views.append(dict(image_path=record['image_path'], group=group,
                          pnp_valid=True, corner_count=len(record['image_points']),
                          rms_px=error, tilt_deg=tilt,
                          corner_hull_fraction=area/(width*height),
                          board_depth_m=float(np.asarray(tv).ravel()[2])))
    all_pixels = np.concatenate([r['image_points'] for r in records])
    hull_fraction = float(cv2.contourArea(cv2.convexHull(all_pixels)))/(width*height)
    cells = set(tuple(int(x) for x in np.minimum((p/np.array(size)*3).astype(int), 2))
                for p in all_pixels)
    tilts = [v['tilt_deg'] for v in views if v['tilt_deg'] is not None]
    spread = max((math.degrees(math.acos(float(np.clip(np.dot(a, b), -1., 1.))))
                  for a in normals for b in normals), default=0.)
    scale_ratio = math.sqrt(max(areas)/min(areas)) if areas and min(areas) > 0 else 0.
    holdout_n = sum(len(records[i]['image_points']) for i in holdout_ids)
    holdout_rms = math.sqrt(sum(holdout_errors)/holdout_n) if not holdout_failed else None
    t = QUALITY_THRESHOLDS
    reasons = []
    if len(records) < t['minimum_views']:
        reasons.append('too_few_views')
    if rms > t['maximum_training_rms_px']:
        reasons.append('high_training_reprojection_error')
    if holdout_rms is None or holdout_rms > t['maximum_holdout_rms_px']:
        reasons.append('high_or_invalid_holdout_pnp_reprojection_error')
    if hull_fraction < t['minimum_corner_hull_fraction']:
        reasons.append('insufficient_image_corner_coverage')
    if len(cells) < t['minimum_corner_grid_cells_3x3']:
        reasons.append('insufficient_image_grid_coverage')
    if sum(angle >= 15. for angle in tilts) < t['minimum_tilted_views_15deg']:
        reasons.append('too_few_tilted_views')
    if max(tilts, default=0.) < t['minimum_maximum_tilt_deg']:
        reasons.append('insufficient_maximum_tilt')
    if spread < t['minimum_normal_spread_deg']:
        reasons.append('insufficient_tilt_direction_diversity')
    if scale_ratio < t['minimum_board_scale_ratio']:
        reasons.append('insufficient_board_scale_diversity')
    if any(depth <= 0 for depth in depths):
        reasons.append('board_pose_behind_camera')
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    # Broad sanity only; these are NOT a USB product-specific lens specification.
    if not (.2*width < fx < 10*width and .2*height < fy < 10*width
            and .5 < fx/fy < 2 and -.1*width < cx < 1.1*width
            and -.1*height < cy < 1.1*height):
        reasons.append('implausible_camera_parameters')
    sources = {'usb_intrinsics.py': _sha256(Path(__file__).resolve()),
               'board_rgb.py': _sha256(Path(__file__).resolve().with_name('board_rgb.py'))}
    report = {
        'schema': 1, 'kind': 'usb_rgb_intrinsics_calibration',
        'camera': camera, 'image_size': list(size), 'board': board_config,
        'distortion_model': 'opencv_brown_5',
        'distortion_order': ['k1', 'k2', 'p1', 'p2', 'k3'],
        'calibration_flags': calibration_flags,
        'calibration_flag_names': ['CALIB_FIX_K3'] if fix_k3 else [],
        'estimated_distortion_parameters': ['k1', 'k2', 'p1', 'p2']
        if fix_k3 else ['k1', 'k2', 'p1', 'p2', 'k3'],
        'fixed_distortion_parameters': {'k3': 0.0} if fix_k3 else {},
        'camera_matrix': matrix.tolist(), 'distortion_coefficients': distortion.ravel().tolist(),
        'quality_passed': not reasons, 'quality_reasons': reasons, 'quality_thresholds': t,
        'accepted_views': len(records), 'training_views': len(training_ids),
        'holdout_views': len(holdout_ids), 'training_rms_px': float(rms),
        'holdout_pnp_rms_px': holdout_rms,
        'coverage': {'corner_hull_fraction': hull_fraction,
                     'corner_grid_cells_3x3': len(cells),
                     'occupied_corner_grid_cells': [list(x) for x in sorted(cells)],
                     'board_scale_ratio': scale_ratio},
        'pose_diversity': {'maximum_tilt_deg': max(tilts, default=0.),
                           'tilted_views_15deg': sum(angle >= 15. for angle in tilts),
                           'normal_spread_deg': spread},
        'views': views, 'skipped_samples': skipped,
        'provenance': {'manifest_sha256': _sha256(manifest), 'source_sha256': sources,
                       'samples': [{k: v for k, v in r.items()
                                    if k not in ('object_points', 'image_points')}
                                   for r in records], 'opencv_version': cv2.__version__},
        'motion_target_valid': False,
        'limitations': [
            'Holdout PnP fits each held-out board pose; residuals are in pixels.',
            'Pixel residuals and quality_passed do not establish millimeter accuracy.',
            'Calibration is tied to this image size, lens focus and camera mode.',
            'Board/base transform and physical palm contact transform remain separate.',
        ],
    }
    encoded = json.dumps(report, indent=2, allow_nan=False)+'\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(encoded)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fix-k3', action='store_true',
                        help='Fix k3 at zero; estimate k1, k2, p1 and p2 only')
    args = parser.parse_args()
    report = solve_dataset(args.dataset, args.output, fix_k3=args.fix_k3)
    print(json.dumps({k: report[k] for k in (
        'quality_passed', 'quality_reasons', 'accepted_views',
        'training_rms_px', 'holdout_pnp_rms_px')}, indent=2))
    return 0 if report['quality_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
