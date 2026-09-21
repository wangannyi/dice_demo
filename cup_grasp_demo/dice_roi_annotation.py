"""Deproject an operator-reviewed visible dice disk from aligned D435i RGB-D.

The operator supplies its pixel circle after inspecting the exact formal frame.
White annulus depth supports the center; this command never controls hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from nero_calibration.core import matrix


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def annotate(snapshot, calibration, geometry, *, center_px, radius_px):
    snapshot, calibration = Path(snapshot), Path(calibration)
    meta = json.loads((snapshot/'metadata.json').read_bytes())
    if (meta.get('camera_backend') != 'realsense' or
            meta.get('frame') != 'color_optical' or
            meta.get('depth_registered_to') != 'color_optical' or
            _sha(snapshot/'color.png') != meta.get('sha256_color') or
            _sha(snapshot/'depth.npz') != meta.get('sha256_depth')):
        raise ValueError('Dice annotation needs exact aligned RGB-D hashes')
    color = cv2.imread(str(snapshot/'color.png'))
    with np.load(snapshot/'depth.npz') as archive:
        raw = np.asarray(archive['depth_raw'])
    if color is None or raw.shape != color.shape[:2]:
        raise ValueError('Dice RGB/depth image size differs')
    x, y = map(float, center_px)
    r = float(radius_px)
    height, width = raw.shape
    if (not all(math.isfinite(v) for v in (x, y, r)) or
            r < 15 or r > 100 or x-r >= width or x+r < 0 or
            y-r >= height or y+r < 0):
        raise ValueError('Reviewed dice pixel circle has no overlap with aligned frame')
    yy, xx = np.ogrid[:height, :width]
    distance = np.hypot(xx-x, yy-y)
    annulus = (distance >= .45*r) & (distance <= .90*r)
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 150)
    depth = raw.astype(np.float64)*float(meta['depth_scale_m'])
    intr = meta['intrinsics']
    fx, fy = float(intr['fx']), float(intr['fy'])
    cx, cy = float(intr['cx']), float(intr['cy'])
    if min(fx, fy) <= 0 or not all(math.isfinite(v) for v in (fx, fy, cx, cy)):
        raise ValueError('Dice frame intrinsics are invalid')
    geo_path = Path(geometry)
    geo = json.loads(geo_path.read_bytes())
    provenance = geo.get('input_provenance', {})
    table = geo.get('geometry', {})
    quality = table.get('quality', {})
    normal = np.asarray(table.get('axis'), dtype=float)
    support = np.asarray(table.get('support_center_m'), dtype=float)
    if (geo.get('frame_id') != meta['frame_id'] or
            provenance.get('sha256_color') != meta['sha256_color'] or
            provenance.get('sha256_depth') != meta['sha256_depth'] or
            normal.shape != (3,) or support.shape != (3,) or
            not np.isfinite(normal).all() or not np.isfinite(support).all() or
            abs(np.linalg.norm(normal)-1.) > .01 or
            float(quality.get('table_rms_m', 1.)) > .003 or
            float(quality.get('table_inlier_fraction', 0.)) < .5):
        raise ValueError('Dice disk needs the exact-frame supported table plane')
    denominator = (normal[0]*(xx-cx)/fx + normal[1]*(yy-cy)/fy + normal[2])
    valid = annulus & white & (depth > .3) & (depth < 1.5) & (abs(denominator) > .1)
    if np.count_nonzero(valid) < 30:
        raise ValueError('Visible dice disk arc lacks thirty white RGB-D points')
    table_depth = float(normal @ support)/denominator[valid]
    residual = depth[valid]-table_depth
    # Dice may rise above the disk.  Select its lower visible surface near
    # the tabletop, while keeping the ROI center on the common table plane.
    disk_depth_offset = float(np.percentile(residual, 80))
    supporting = abs(residual-disk_depth_offset) <= .003
    if (not -.02 <= disk_depth_offset <= .01 or
            np.count_nonzero(supporting) < 30):
        raise ValueError('White arc lacks a supported surface within 20 mm of the table')
    p90 = float(np.percentile(abs(residual[supporting]-disk_depth_offset), 90))
    if p90 > .003:
        raise ValueError('White disk surface depth residual exceeds 3 mm')
    center_denominator = normal[0]*(x-cx)/fx+normal[1]*(y-cy)/fy+normal[2]
    if abs(center_denominator) <= .1:
        raise ValueError('Dice disk center ray is parallel to table plane')
    z = float(normal @ support)/center_denominator
    if not .3 <= z <= 1.5:
        raise ValueError('Dice disk table projection depth is implausible')
    optical = np.array([(x-cx)*z/fx, (y-cy)*z/fy, z])
    record = json.loads(calibration.read_bytes())
    if (record.get('mode') != 'eye_to_hand' or
            record.get('camera', {}).get('serial') != meta.get('serial') or
            record.get('camera', {}).get('frame') != 'color_optical'):
        raise ValueError('Dice ROI camera does not match eye-to-hand result')
    transform = matrix(record['T_base_camera'])
    base = transform[:3, :3]@optical+transform[:3, 3]
    radius_m = r*z/min(fx, fy)
    if not .02 <= radius_m <= .12:
        raise ValueError('Dice disk metric radius is implausible')
    return {
        'schema': 1, 'kind': 'measured_dice_roi', 'frame': 'base',
        'source_frame_id': meta['frame_id'],
        'snapshot_sha256_color': meta['sha256_color'],
        'snapshot_sha256_depth': meta['sha256_depth'],
        'calibration_sha256': _sha(calibration),
        'table_geometry_sha256': _sha(geo_path),
        'center_pixel_xy': [x, y], 'radius_pixel': r,
        'center_camera_optical_m': optical.tolist(),
        'center_base_m': base.tolist(), 'radius_m': radius_m,
        'depth_support_points': int(np.count_nonzero(supporting)),
        'center_projection': 'same_frame_table_plane_under_visible_white_disk',
        'disk_surface_depth_offset_from_table_m': disk_depth_offset,
        'depth_p90_residual_m': p90,
        'annotation_reviewed': True,
        'quality_passed': bool(record.get('quality_passed')),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True,
                        help='Exact-frame cup geometry with measured table plane')
    parser.add_argument('--center-px', type=float, nargs=2, required=True)
    parser.add_argument('--radius-px', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = annotate(args.snapshot, args.calibration, args.geometry,
                          center_px=args.center_px, radius_px=args.radius_px)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'output': str(args.output),
                          'center_base_m': result['center_base_m']}))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'error': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
