#!/usr/bin/env python3
"""Offline board-dimension correction from saved RGB images. No hardware access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate import read_dataset, write_new  # noqa: E402
from core import distance, matrix, solve  # noqa: E402
from sensors import CharucoDetector  # noqa: E402


def measured_observation(image, detector, K, D, width_m, height_m):
    """Retain the recorded detector/ROI; solve PnP with measured X/Y grid spacing."""
    if not (np.isfinite([width_m, height_m]).all() and 0 < width_m < 1
            and 0 < height_m < 1):
        raise ValueError('Board width and height must be positive and below one metre')
    detector.detect(image, K, D)  # Apply the original marker, ROI and quality checks.
    a = image
    if detector.roi is not None:
        x0, y0, x1, y1 = detector.roi
        a = np.full_like(image, 255)
        a[y0:y1, x0:x1] = image[y0:y1, x0:x1]
    if detector.cfg.get("image_exclude_rois_xyxy"):
        a = a.copy()
        for x0, y0, x1, y1 in detector.cfg["image_exclude_rois_xyxy"]:
            a[y0:y1, x0:x1] = 255
    mc, mi, _ = cv2.aruco.detectMarkers(a, detector.dictionary)
    _, cc, ci = cv2.aruco.interpolateCornersCharuco(
        mc, mi, a, detector.board, cameraMatrix=K, distCoeffs=D)
    obj = np.asarray(detector.points[ci.ravel()], dtype=np.float64).copy()
    cfg = detector.cfg
    obj[:, 0] *= width_m / (cfg['squares_x'] * cfg['square_length_m'])
    obj[:, 1] *= height_m / (cfg['squares_y'] * cfg['square_length_m'])
    uv = np.asarray(cc, dtype=np.float64).reshape(-1, 2)
    ok, rv, tv = cv2.solvePnP(obj, uv, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('Measured-board pose solve failed')
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
    matrix(T)
    if np.min((T[:3, :3] @ obj.T + tv)[2]) <= 0:
        raise ValueError('Measured board behind camera')
    projected = cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected-uv)**2, axis=1))))
    if not np.isfinite(rms) or rms > cfg.get('max_reprojection_px', 1.):
        raise ValueError(f'Measured-board reprojection RMS too large: {rms:.3f} px')
    return T, {'corners': len(ci), 'reprojection_rms_px': rms}, ci.ravel().tolist(), uv.tolist()


def metrics(result):
    held = np.asarray(result['holdout_errors_m_deg'])
    all_fit = np.asarray(result['all_sample_errors_m_deg'])
    return {'quality_passed': result['quality_passed'],
            'holdout_max_position_mm': float(held[:, 0].max()*1000),
            'holdout_max_angle_deg': float(held[:, 1].max()),
            'all_fit_max_position_mm': float(all_fit[:, 0].max()*1000),
            'all_fit_max_angle_deg': float(all_fit[:, 1].max())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--width-mm', type=float, required=True, help='Grid width without white border')
    parser.add_argument('--height-mm', type=float, required=True, help='Grid height without white border')
    args = parser.parse_args(argv)
    meta, samples = read_dataset(args.dataset)
    if meta.get('opencv_version') != cv2.__version__:
        raise ValueError('Run with the recorded OpenCV version to preserve board corner conventions')
    args.output.mkdir(parents=True, exist_ok=False)
    detector = CharucoDetector(meta['board'])
    K = np.asarray(meta['camera']['camera_matrix'])
    D = np.asarray(meta['camera']['dist_coeffs'])
    observations, corner_records, hashes = [], [], {}
    for i, sample in enumerate(samples):
        stem = f'sample_{i:04d}'
        image_file = args.dataset/(stem+'.png')
        image = cv2.imread(str(image_file))
        if image is None:
            raise ValueError(f'Cannot read {image_file}')
        T, quality, ids, pixels = measured_observation(
            image, detector, K, D, args.width_mm/1000, args.height_mm/1000)
        observations.append({**sample, 'T_camera_board': T.tolist(), 'quality': quality})
        corner_records.append({'source': stem, 'corner_ids': ids, 'pixels': pixels})
        for path in [image_file, args.dataset/(stem+'.json')]:
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_file = args.dataset/'manifest.json'
    hashes[manifest_file.name] = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    measurement = {'grid_width_mm': args.width_mm, 'grid_height_mm': args.height_mm,
                   'spacing_x_mm': args.width_mm/meta['board']['squares_x'],
                   'spacing_y_mm': args.height_mm/meta['board']['squares_y'],
                   'source_dataset': str(args.dataset.resolve()), 'opencv_version': cv2.__version__,
                   'recorded_board_config': meta['board'], 'source_sha256': hashes,
                   'method': 'Original marker detection and corner interpolation; measured X/Y grid points for PnP. Original flange poses, sample order, holdouts and gates retained.',
                   'scope': 'Offline derived observations; not a resumable capture session or a live board config.'}
    baseline = solve(samples, meta['T_flange_tcp'])
    result = solve(observations, meta['T_flange_tcp'])
    result.update({'sample_count': len(samples), 'camera': meta['camera'],
                   'tcp': meta['tcp'], 'board_measurement': measurement})
    delta = distance(np.asarray(baseline['T_base_camera']), np.asarray(result['T_base_camera']))
    comparison = {'original': metrics(baseline), 'corrected': metrics(result),
                  'camera_transform_change_mm_deg': [delta[0]*1000, delta[1]],
                  'holdout_samples_1based': [i+1 for i in result['holdout_indices']]}
    write_new(args.output/'measurement.json', measurement)
    write_new(args.output/'observations.json', observations)
    write_new(args.output/'image_corners.json', corner_records)
    write_new(args.output/'result.json', result)
    write_new(args.output/'comparison.json', comparison)
    print(json.dumps({'output': str(args.output), **comparison}), flush=True)
    return 0 if result['quality_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
