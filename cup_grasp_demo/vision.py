"""Bounded RGB-D capture and simple cup/table masks for a first grasp demo.

All coordinates are in the RealSense color optical frame. ``depth_raw`` is
unaltered Z16 data aligned to color; multiply it by ``depth_scale_m`` exactly
once to obtain metres. No camera device is opened at module import time.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


def capture_frame(
    serial: Optional[str] = None,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    timeout_ms: int = 3000,
    warmup_frames: int = 3,
    max_framesets: Optional[int] = None,
    max_sync_delta_ms: float = 40.0,
) -> dict:
    """Open a D435i, select one time-consistent aligned pair, and close it.

    The color image is BGR uint8, as requested from RealSense. ``depth_raw``
    remains uint16 in device units; ``depth_m`` is a convenience float32 copy.
    Device timestamps must share a domain and differ by at most 40 ms at 15
    fps (also capped at one frame period). Startup frames are discarded. The
    total acquisition wait and number of framesets are bounded; no unsynced
    pair is returned. The default budget stays at 30 framesets for the usual
    three-frame warmup and grows to warmup+15 when longer warmup is requested.
    Frame timestamps are device milliseconds, not host time.
    """
    if max_framesets is None:
        max_framesets = max(30, warmup_frames + 15)
    if (fps <= 0 or timeout_ms < 100 or warmup_frames < 0
            or max_framesets <= warmup_frames or not np.isfinite(max_sync_delta_ms)
            or max_sync_delta_ms <= 0):
        raise ValueError('Invalid RealSense capture synchronization limits')
    sync_limit_ms = min(float(max_sync_delta_ms), 1000.0 / fps)
    import pyrealsense2 as rs  # K3-only dependency; no device at import time.

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(str(serial))
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        depth_scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
        align = rs.align(rs.stream.color)
        deadline_ns = time.monotonic_ns() + int(timeout_ms * 1_000_000)
        last_rejection = 'no framesets arrived'
        selected = False
        for framesets_seen in range(1, max_framesets + 1):
            remaining_ms = (deadline_ns - time.monotonic_ns()) // 1_000_000
            if remaining_ms <= 0:
                break
            frames = align.process(pipeline.wait_for_frames(int(remaining_ms)))
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                last_rejection = 'missing color or aligned depth frame'
                continue
            if framesets_seen <= warmup_frames:
                last_rejection = 'startup warmup frames'
                continue
            color_frame_id = int(color_frame.get_frame_number())
            depth_frame_id = int(depth_frame.get_frame_number())
            color_timestamp_ms = float(color_frame.get_timestamp())
            depth_timestamp_ms = float(depth_frame.get_timestamp())
            color_domain = str(color_frame.get_frame_timestamp_domain())
            depth_domain = str(depth_frame.get_frame_timestamp_domain())
            if color_frame_id <= 0 or depth_frame_id <= 0:
                last_rejection = f'startup frame ID color={color_frame_id}, depth={depth_frame_id}'
                continue
            if (not np.isfinite(color_timestamp_ms) or not np.isfinite(depth_timestamp_ms)
                    or color_timestamp_ms < 0 or depth_timestamp_ms < 0):
                last_rejection = 'invalid device timestamp'
                continue
            if not color_domain or color_domain != depth_domain:
                last_rejection = f'timestamp domains differ: {color_domain}, {depth_domain}'
                continue
            sync_delta_ms = abs(color_timestamp_ms - depth_timestamp_ms)
            if sync_delta_ms > sync_limit_ms:
                last_rejection = f'color/depth delta {sync_delta_ms:.3f} ms exceeds {sync_limit_ms:.3f} ms'
                continue
            selected = True
            break
        if not selected:
            raise RuntimeError(
                f'No synchronized RealSense pair within {timeout_ms} ms/{max_framesets} framesets: '
                f'{last_rejection}')

        # Copy buffers before pipeline.stop() releases their backing storage.
        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data()).copy()
        if color_bgr.shape != (height, width, 3) or depth_raw.shape != (height, width):
            raise RuntimeError('RealSense color and aligned depth dimensions differ')
        if depth_raw.dtype != np.uint16:
            raise RuntimeError(f'Expected Z16 depth; got {depth_raw.dtype}')
        depth_m = depth_raw.astype(np.float32) * depth_scale_m
        return {
            'color_bgr': color_bgr,
            'depth_raw': depth_raw,
            'depth_m': depth_m,
            'depth_scale_m': depth_scale_m,
            'intrinsics': {
                'fx': float(intr.fx), 'fy': float(intr.fy),
                'ppx': float(intr.ppx), 'ppy': float(intr.ppy),
                'width': int(intr.width), 'height': int(intr.height),
                'dist_coeffs': [float(value) for value in intr.coeffs],
                'distortion_model': str(intr.model),
            },
            'serial': profile.get_device().get_info(rs.camera_info.serial_number),
            'frame': 'color_optical',
            'frame_id': color_frame_id,
            'frame_ids': {'color': color_frame_id, 'depth_aligned': depth_frame_id},
            'timestamps_ms': {
                'color': color_timestamp_ms, 'depth_aligned': depth_timestamp_ms,
            },
            'timestamp_domains': {
                'color': color_domain, 'depth_aligned': depth_domain,
            },
            'sync_delta_ms': float(sync_delta_ms),
            'sync_limit_ms': float(sync_limit_ms),
            'framesets_seen': int(framesets_seen),
            'warmup_frames': int(warmup_frames),
            'host_capture_time_ns': time.time_ns(),
        }
    finally:
        if started:
            pipeline.stop()


def _check_color(color_bgr: np.ndarray) -> tuple[int, int]:
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3 or color_bgr.dtype != np.uint8:
        raise ValueError('color_bgr must be an HxWx3 uint8 image')
    return color_bgr.shape[:2]


def _roi(shape: tuple[int, int], bbox: Optional[Sequence[int]]) -> np.ndarray:
    height, width = shape
    roi = np.zeros((height, width), dtype=bool)
    if bbox is None:
        roi[:] = True
        return roi
    if len(bbox) != 4:
        raise ValueError('bbox must be (x_min, y_min, x_max, y_max)')
    x_min, y_min, x_max, y_max = (int(value) for value in bbox)
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError('bbox must be non-empty and inside the image')
    roi[y_min:y_max, x_min:x_max] = True
    return roi


def segment_green_cup(
    color_bgr: np.ndarray,
    bbox: Optional[Sequence[int]] = None,
    hsv_low: tuple[int, int, int] = (35, 45, 25),
    hsv_high: tuple[int, int, int] = (90, 255, 255),
) -> np.ndarray:
    """Return a green-object mask, optionally restricted to an XYXY ROI."""
    shape = _check_color(color_bgr)
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    return (cv2.inRange(hsv, np.array(hsv_low, dtype=np.uint8),
                        np.array(hsv_high, dtype=np.uint8)) > 0) & _roi(shape, bbox)


def segment_red_table_plane_supported(
    color_bgr: np.ndarray,
    object_mask: np.ndarray,
    bbox: Sequence[int],
    depth_m: np.ndarray,
    intrinsics: dict,
    min_candidate_plane_fraction: float = 0.50,
    plane_tolerance_m: float = 0.003,
) -> tuple[np.ndarray, dict]:
    """Keep measured plane inliers from a local red tabletop candidate.

    A textured mat may give less than the geometry module's required 80%
    initial plane support. This explicitly reports the raw support fraction,
    then passes only fitted inliers to that unchanged strict geometry gate.
    The supplied bbox must cover one local red tabletop area near the cup.
    """
    from dice_cup_localization.geometry import Config, _plane

    shape = _check_color(color_bgr)
    depth_m = np.asarray(depth_m)
    if depth_m.shape != shape or not np.issubdtype(depth_m.dtype, np.floating):
        raise ValueError('depth_m must be aligned HxW floating metres')
    if object_mask.shape != shape or object_mask.dtype != np.bool_:
        raise ValueError('object_mask must be an aligned HxW boolean mask')
    if (intrinsics.get('frame') != 'color_optical'
            or (intrinsics.get('height'), intrinsics.get('width')) != shape):
        raise ValueError('Intrinsics must describe this color optical image')
    coeffs = np.asarray(intrinsics.get('dist_coeffs'), dtype=float)
    if coeffs.shape != (5,) or not np.isfinite(coeffs).all() or np.any(coeffs != 0):
        raise ValueError('Require explicit zero distortion coefficients; rectify first')
    fx, fy = float(intrinsics['fx']), float(intrinsics['fy'])
    cx = float(intrinsics.get('cx', intrinsics.get('ppx')))
    cy = float(intrinsics.get('cy', intrinsics.get('ppy')))
    if not np.isfinite((fx, fy, cx, cy)).all() or min(fx, fy) <= 0:
        raise ValueError('Invalid color optical pinhole intrinsics')
    if (not 0.4 <= min_candidate_plane_fraction <= 1.0
            or not np.isfinite(plane_tolerance_m) or plane_tolerance_m <= 0):
        raise ValueError('Invalid provisional tabletop fit limits')

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    red = ((cv2.inRange(hsv, (0, 45, 35), (15, 255, 255)) > 0)
           | (cv2.inRange(hsv, (170, 45, 35), (179, 255, 255)) > 0))
    valid = np.isfinite(depth_m) & (depth_m >= 0.1) & (depth_m <= 2.0)
    candidate = red & _roi(shape, bbox) & valid & ~object_mask
    y, x = np.nonzero(candidate)
    if len(x) < Config().min_points:
        raise ValueError('Local red tabletop has too few valid candidate pixels')
    z = depth_m[y, x].astype(float)
    points = np.column_stack(((x-cx)*z/fx, (y-cy)*z/fy, z))
    config = Config(min_plane_fraction=float(min_candidate_plane_fraction),
                    plane_tolerance_m=float(plane_tolerance_m))
    origin, normal, raw_fraction, plane_rms_m = _plane(points, config)
    residual = np.abs((points - origin) @ normal)
    keep = residual <= plane_tolerance_m
    table_mask = np.zeros(shape, dtype=bool)
    table_mask[y[keep], x[keep]] = True
    return table_mask, {
        'mask_source': 'local_red_hsv_depth_plane_inliers_provisional',
        'candidate_pixels': int(len(x)),
        'inlier_pixels': int(np.count_nonzero(keep)),
        'raw_plane_inlier_fraction': float(raw_fraction),
        'kept_fraction': float(np.mean(keep)),
        'plane_rms_m': float(plane_rms_m),
        'plane_tolerance_m': float(plane_tolerance_m),
        'min_candidate_plane_fraction': float(min_candidate_plane_fraction),
        'table_bbox_xyxy': [int(value) for value in bbox],
        'hardware_validated': False,
        'complete_cup_height_verified': False,
        'provisional_reason': ('Plane mask keeps only local inliers from a textured surface; '
                               'observed RGB-D height may omit the complete cup'),
    }


def segment_dark_cup(
    color_bgr: np.ndarray,
    bbox: Sequence[int],
    hue_min: int = 85,
    hue_max: int = 165,
    saturation_min: int = 35,
    value_max: int = 145,
    neutral_dark_value_max: int = 65,
) -> np.ndarray:
    """Segment the dark navy cup on a red table inside a supplied XYXY ROI.

    The ROI is required because other dark objects (hand and board) are
    present in the full scene. Bright blue tabletop pixels are excluded by
    ``value_max``. Retain the largest connected candidate in the ROI.
    """
    shape = _check_color(color_bgr)
    roi = _roi(shape, bbox)
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    navy = (h >= hue_min) & (h <= hue_max) & (s >= saturation_min) & (v <= value_max)
    neutral_dark = v <= neutral_dark_value_max
    red_surface = ((h <= 15) | (h >= 170)) & (s >= 45)
    candidate = (navy | neutral_dark) & ~red_surface & roi
    candidate_u8 = candidate.astype(np.uint8)
    # The blue/red mat seam is a thin dark line which can join the cup mask.
    # A 9 px opening removes that line while preserving the ~80 px cup body.
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN,
                                    np.ones((9, 9), dtype=np.uint8))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE,
                                    np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)
    if count <= 1:
        return np.zeros(shape, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest) & roi


def segment_dark_cup_depth_refined(
    color_bgr: np.ndarray,
    bbox: Sequence[int],
    depth_m: np.ndarray,
    table_mask: np.ndarray,
    intrinsics: dict,
    min_above_table_m: float = 0.006,
    expansion_px: int = 1,
) -> tuple[np.ndarray, dict]:
    """Add depth-supported silhouette edge pixels to the HSV cup mask.

    The supplied table mask must identify one local supporting tabletop in
    this same aligned color/depth frame. No tabletop is inferred from the full
    scene. The current HSV cup mask remains the seed; only pixels within an
    explicit small dilation and above the fitted table plane are added. The
    diagnostic dictionary records the source and plane fit for review.
    """
    from dice_cup_localization.geometry import Config, _plane

    shape = _check_color(color_bgr)
    depth_m = np.asarray(depth_m)
    table_mask = np.asarray(table_mask)
    if depth_m.shape != shape or not np.issubdtype(depth_m.dtype, np.floating):
        raise ValueError('depth_m must be aligned HxW floating metres')
    if table_mask.shape != shape or table_mask.dtype != np.bool_:
        raise ValueError('table_mask must be an aligned HxW boolean mask')
    if (not np.isfinite(min_above_table_m) or min_above_table_m <= 0
            or expansion_px < 1 or expansion_px > 4):
        raise ValueError('Invalid foreground threshold or expansion')
    if (intrinsics.get('frame') != 'color_optical'
            or (intrinsics.get('height'), intrinsics.get('width')) != shape):
        raise ValueError('Intrinsics must describe this color optical image')
    coeffs = np.asarray(intrinsics.get('dist_coeffs'), dtype=float)
    if coeffs.shape != (5,) or not np.isfinite(coeffs).all() or np.any(coeffs != 0):
        raise ValueError('Require explicit zero distortion coefficients; rectify first')
    fx, fy = float(intrinsics['fx']), float(intrinsics['fy'])
    cx = float(intrinsics.get('cx', intrinsics.get('ppx')))
    cy = float(intrinsics.get('cy', intrinsics.get('ppy')))
    if not np.isfinite((fx, fy, cx, cy)).all() or min(fx, fy) <= 0:
        raise ValueError('Invalid color optical pinhole intrinsics')

    seed = segment_dark_cup(color_bgr, bbox)
    if np.any(seed & table_mask):
        raise ValueError('Cup seed and local tabletop masks overlap')
    valid = np.isfinite(depth_m) & (depth_m >= 0.1) & (depth_m <= 2.0)
    table_y, table_x = np.nonzero(table_mask & valid)
    if len(table_x) < Config().min_points:
        raise ValueError('Local tabletop has too few valid depth pixels')
    table_z = depth_m[table_y, table_x].astype(float)
    table_points = np.column_stack(((table_x-cx)*table_z/fx,
                                    (table_y-cy)*table_z/fy, table_z))
    plane_origin, plane_normal, plane_fraction, plane_rms_m = _plane(table_points, Config())

    kernel_size = 2 * int(expansion_px) + 1
    vicinity = cv2.dilate(seed.astype(np.uint8),
                          np.ones((kernel_size, kernel_size), dtype=np.uint8)) != 0
    vicinity &= _roi(shape, bbox) & ~table_mask & valid
    y, x = np.nonzero(vicinity)
    z = depth_m[y, x].astype(float)
    points = np.column_stack(((x-cx)*z/fx, (y-cy)*z/fy, z))
    heights = (points - plane_origin) @ plane_normal
    additions = np.zeros(shape, dtype=bool)
    supported = (heights > min_above_table_m) & (heights < Config().max_height_m)
    additions[y[supported], x[supported]] = True
    refined = seed | additions
    return refined, {
        'mask_source': 'dark_cup_hsv_roi+local_table_depth_foreground',
        'seed_pixels': int(np.count_nonzero(seed)),
        'added_pixels': int(np.count_nonzero(additions & ~seed)),
        'refined_pixels': int(np.count_nonzero(refined)),
        'table_plane_inlier_fraction': float(plane_fraction),
        'table_plane_rms_m': float(plane_rms_m),
        'foreground_threshold_m': float(min_above_table_m),
        'expansion_px': int(expansion_px),
        'table_support_pixels': int(len(table_x)),
    }


def segment_table(
    color_bgr: np.ndarray,
    object_mask: np.ndarray,
    bbox: Optional[Sequence[int]] = None,
    depth_m: Optional[np.ndarray] = None,
    object_depth_m: Optional[float] = None,
    max_depth_delta_m: float = 0.20,
    min_depth_m: float = 0.10,
    max_depth_m: float = 2.0,
) -> np.ndarray:
    """Return nearby red/blue tabletop pixels, excluding the object mask.

    When aligned depth is supplied, retain valid tabletop pixels near the
    median object depth. This mode is a surface candidate, not a plane fit.
    """
    shape = _check_color(color_bgr)
    if object_mask.shape != shape:
        raise ValueError('object_mask shape must match color image')
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, (0, 45, 35), (15, 255, 255)) > 0
    red_high = cv2.inRange(hsv, (170, 45, 35), (179, 255, 255)) > 0
    blue = cv2.inRange(hsv, (90, 45, 35), (140, 255, 255)) > 0
    mask = (red_low | red_high | blue) & _roi(shape, bbox) & ~object_mask.astype(bool)
    if depth_m is not None:
        if depth_m.shape != shape:
            raise ValueError('aligned depth_m shape must match color image')
        valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
        if object_depth_m is None:
            object_valid = valid & object_mask.astype(bool)
            if np.any(object_valid):
                object_depth_m = float(np.median(depth_m[object_valid]))
        mask &= valid
        if object_depth_m is not None:
            mask &= np.abs(depth_m - float(object_depth_m)) <= max_depth_delta_m
    return mask


def load_mask_png(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a user-supplied mask without silently resizing its pixel geometry."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != shape:
        raise ValueError(f'Mask PNG shape {mask.shape} differs from image {shape}')
    return mask > 0
