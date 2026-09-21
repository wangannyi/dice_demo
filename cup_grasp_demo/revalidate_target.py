"""Check a newly captured RGB-D frame against the exact planned green cup.

This is a short, independent stationary-target check after slow YOLO planning.
It never labels a cup touch, hand hold, or full-scene collision as verified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from cup_grasp_demo.grasp import load_snapshot


MAX_FRESH_AGE_S = 10.


def check_stationary_cup(original_dir, recognition_dir, fresh_dir, *, now_ns):
    old_color, old_depth, old_meta = load_snapshot(original_dir)
    new_color, new_depth, new_meta = load_snapshot(fresh_dir)
    geometry = json.loads((Path(recognition_dir)/'geometry.json').read_bytes())
    recognition = json.loads((Path(recognition_dir)/'recognition.json').read_bytes())
    if (old_meta['serial'] != new_meta['serial']
            or old_meta.get('camera_backend') != 'realsense'
            or new_meta.get('camera_backend') != 'realsense'
            or old_meta.get('frame') != 'color_optical'
            or new_meta.get('frame') != 'color_optical'
            or geometry.get('frame_id') != old_meta['frame_id']
            or recognition.get('frame_id') != old_meta['frame_id']
            or recognition.get('model_profile') != 'dice_cap2'
            or geometry.get('model_sha256') != recognition.get('model', {}).get('sha256')
            or recognition.get('selected_instance') is None):
        raise ValueError('Fresh check source does not match planned camera/YOLO cup')
    if (new_meta['host_capture_time_ns'] <= old_meta['host_capture_time_ns']
            or not 0 <= (now_ns-new_meta['host_capture_time_ns'])/1e9 <= MAX_FRESH_AGE_S):
        raise ValueError('Stationary-cup RGB-D check is stale or predates YOLO')
    if old_color.shape != new_color.shape or old_depth.shape != new_depth.shape:
        raise ValueError('Camera dimensions changed after YOLO')
    old_intr = old_meta['intrinsics']
    new_intr = new_meta['intrinsics']
    if any(abs(float(old_intr[key])-float(new_intr[key])) > .1
           for key in ('fx', 'fy', 'cx', 'cy')):
        raise ValueError('Camera intrinsics changed after YOLO')
    top = geometry.get('geometry', {}).get('top_surface', {})
    center = np.asarray(top.get('center_m'), dtype=float)
    if (top.get('valid') is not True or center.shape != (3,)
            or not np.isfinite(center).all() or center[2] <= 0):
        raise ValueError('Original YOLO visible top is invalid')
    with np.load(Path(recognition_dir)/'localization_input.npz', allow_pickle=False) as bundle:
        object_mask = bundle['object_mask'].copy()
        table_mask = bundle['table_mask'].copy()
        source_depth = bundle['depth'].copy()
    if (object_mask.dtype != np.bool_ or table_mask.dtype != np.bool_
            or object_mask.shape != old_depth.shape
            or table_mask.shape != old_depth.shape
            or not np.array_equal(source_depth, old_depth)):
        raise ValueError('Saved YOLO mask/depth differs from original RGB-D')
    u = int(round(float(old_intr['cx'])+float(old_intr['fx'])*center[0]/center[2]))
    v = int(round(float(old_intr['cy'])+float(old_intr['fy'])*center[1]/center[2]))
    yy, xx = np.ogrid[:old_depth.shape[0], :old_depth.shape[1]]
    top_patch = (xx-u)**2+(yy-v)**2 <= 12**2
    center_valid = top_patch & object_mask & (old_depth > 0) & (new_depth > 0)
    table_valid = table_mask & (old_depth > 0) & (new_depth > 0)
    if int(center_valid.sum()) < 100 or int(table_valid.sum()) < 1000:
        raise ValueError('Fresh RGB-D lost measured cup top or static red-mat depth')
    center_delta_mm = new_depth[center_valid].astype(np.int32)-old_depth[center_valid].astype(np.int32)
    table_delta_mm = new_depth[table_valid].astype(np.int32)-old_depth[table_valid].astype(np.int32)
    old_hsv = cv2.cvtColor(old_color, cv2.COLOR_BGR2HSV)
    new_hsv = cv2.cvtColor(new_color, cv2.COLOR_BGR2HSV)
    lo, hi = (35, 55, 35), (90, 255, 255)
    old_green = cv2.inRange(old_hsv, lo, hi) > 0
    new_green = cv2.inRange(new_hsv, lo, hi) > 0
    old_fraction = float(np.mean(old_green[center_valid]))
    new_fraction = float(np.mean(new_green[center_valid]))
    static_color_abs = np.abs(new_color[table_valid].astype(np.int16)
                              -old_color[table_valid].astype(np.int16))
    metrics = {
        'top_pixel_uv': [u, v], 'center_depth_support_points': int(center_valid.sum()),
        'table_depth_support_points': int(table_valid.sum()),
        'center_depth_median_delta_mm': float(np.median(center_delta_mm)),
        'center_depth_p90_abs_delta_mm': float(np.quantile(np.abs(center_delta_mm), .9)),
        'table_depth_median_delta_mm': float(np.median(table_delta_mm)),
        'table_depth_p90_abs_delta_mm': float(np.quantile(np.abs(table_delta_mm), .9)),
        'old_center_green_fraction': old_fraction,
        'fresh_center_green_fraction': new_fraction,
        'table_color_p90_abs_channel_delta': float(np.quantile(static_color_abs, .9)),
    }
    if (abs(metrics['center_depth_median_delta_mm']) > 3
            or metrics['center_depth_p90_abs_delta_mm'] > 8
            or abs(metrics['table_depth_median_delta_mm']) > 3
            or metrics['table_depth_p90_abs_delta_mm'] > 8
            or old_fraction < .75 or new_fraction < .75
            or metrics['table_color_p90_abs_channel_delta'] > 20):
        raise ValueError(f'Cup or fixed-camera red-mat scene changed: {metrics}')
    if not math.isfinite(metrics['center_depth_median_delta_mm']):
        raise ValueError('Nonfinite cup depth comparison')
    return {
        'stationary_cup_verified': True, 'original_frame_id': old_meta['frame_id'],
        'fresh_frame_id': new_meta['frame_id'],
        'original_color_sha256': old_meta['sha256_color'],
        'original_depth_sha256': old_meta['sha256_depth'],
        'fresh_color_sha256': new_meta['sha256_color'],
        'fresh_depth_sha256': new_meta['sha256_depth'],
        'fresh_capture_time_ns': new_meta['host_capture_time_ns'],
        'checked_at_ns': now_ns, 'metrics': metrics,
    }
