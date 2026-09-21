"""Read-only upright cup geometry in the color optical frame (meters)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    """Initial geometric gates; thresholds require real-camera acceptance."""

    min_points: int = 100
    min_valid_fraction: float = 0.5
    plane_tolerance_m: float = 0.003
    min_plane_fraction: float = 0.8
    min_height_m: float = 0.03
    max_height_m: float = 0.30
    min_diameter_m: float = 0.02
    max_diameter_m: float = 0.20
    max_circle_rms_m: float = 0.003
    min_arc_deg: float = 100.0
    max_center_spread_m: float = 0.005
    top_min_points: int = 80
    top_max_plane_rms_m: float = 0.0025
    top_max_tilt_deg: float = 15.0
    top_center_support_radius_m: float = 0.008
    top_min_center_support_points: int = 20


def _top_surface(points, support_center, axis, cfg):
    """Intersect the fitted cup axis with a plane measured on its visible top.

    A circular rim is not independently recovered. The output keeps this
    coaxial assumption visible to a downstream contact planner.
    """
    invalid = {'valid': False, 'center_m': None, 'normal': None,
               'center_definition': 'side_axis_intersection_with_visible_top_plane',
               'quality': {'rim_center_independently_measured': False,
                           'coaxial_cup_assumption': True}, 'reason': None}
    heights = (points - support_center) @ axis
    selected = points[heights >= np.quantile(heights, 0.9)]
    if len(selected) < cfg.top_min_points:
        invalid['reason'] = 'insufficient_top_depth'
        return invalid
    point = selected.mean(axis=0)
    _, singular, vectors = np.linalg.svd(selected - point, full_matrices=False)
    if singular[1] / np.sqrt(len(selected)) < 0.004:
        invalid['reason'] = 'top_patch_too_narrow'
        return invalid
    normal = vectors[-1]
    if normal @ axis < 0:
        normal = -normal
    tilt = float(np.degrees(np.arccos(np.clip(normal @ axis, -1.0, 1.0))))
    rms = float(np.sqrt(np.mean(((selected - point) @ normal) ** 2)))
    if tilt > cfg.top_max_tilt_deg or rms > cfg.top_max_plane_rms_m:
        invalid['reason'] = 'top_surface_not_planar_or_upright'
        invalid['quality'].update(top_plane_rms_m=rms, top_normal_tilt_deg=tilt)
        return invalid
    center_height = float(((point - support_center) @ normal) / (axis @ normal))
    center = support_center + axis * center_height
    lateral = selected - center
    lateral -= np.outer(lateral @ axis, axis)
    nearby = selected[np.linalg.norm(lateral, axis=1) < cfg.top_center_support_radius_m]
    if len(nearby) < cfg.top_min_center_support_points:
        invalid['reason'] = 'top_center_unobserved'
        invalid['quality'].update(center_support_points=len(nearby),
                                  top_plane_rms_m=rms, top_normal_tilt_deg=tilt)
        return invalid
    near_residual = float(np.median(np.abs((nearby - center) @ normal)))
    if near_residual > cfg.top_max_plane_rms_m:
        invalid['reason'] = 'top_center_depth_inconsistent'
        return invalid
    return {'valid': True, 'center_m': center.tolist(), 'normal': normal.tolist(),
            'center_definition': 'side_axis_intersection_with_visible_top_plane',
            'quality': {'top_plane_rms_m': rms, 'top_normal_tilt_deg': tilt,
                        'top_patch_points': len(selected), 'center_support_points': len(nearby),
                        'center_near_depth_median_residual_m': near_residual,
                        'rim_center_independently_measured': False,
                        'coaxial_cup_assumption': True, 'hardware_validated': False},
            'reason': None}


def deproject(depth, mask, intrinsics, depth_scale_m):
    """Deproject registered, rectified depth with color intrinsics.

    Nonzero distortion is rejected. Raw depth values are multiplied by the
    explicitly supplied sensor scale; uint16 does not imply millimeters.
    """
    depth = np.asarray(depth)
    mask = np.asarray(mask)
    if depth.ndim != 2 or depth.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError('Expected a 2D depth image and matching boolean mask')
    if not np.issubdtype(depth.dtype, np.number) or np.iscomplexobj(depth):
        raise ValueError('Depth must be real numeric data')
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError('Invalid depth_scale_m')
    if intrinsics.get('frame') != 'color_optical':
        raise ValueError('Input must be registered to color_optical')
    if (intrinsics['height'], intrinsics['width']) != depth.shape:
        raise ValueError('Intrinsics resolution mismatch')
    coeffs = np.asarray(intrinsics.get('dist_coeffs', []), dtype=float)
    if coeffs.shape != (5,) or not np.isfinite(coeffs).all() or np.any(coeffs != 0):
        raise ValueError('Require explicit zero distortion coefficients; rectify first')
    fx, fy, cx, cy = [float(intrinsics[k]) for k in ('fx', 'fy', 'cx', 'cy')]
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError('Invalid pinhole intrinsics')
    valid = mask & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    z = depth[valid].astype(float) * depth_scale_m
    if not np.isfinite(z).all():
        raise ValueError('Depth scale overflow')
    return np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))


def _plane(points, cfg):
    # Deterministic RANSAC limits CPU and rejects contaminated table masks.
    rng = np.random.default_rng(7)
    sample = points[rng.choice(len(points), min(len(points), 3000), replace=False)]
    best = np.zeros(len(sample), dtype=bool)
    for _ in range(100):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-10:
            continue
        normal /= length
        keep = np.abs((sample - a) @ normal) <= cfg.plane_tolerance_m
        if keep.sum() > best.sum():
            best = keep
    # A three-point RANSAC hypothesis can sit just below the support threshold.
    # Judge the refined least-squares plane, not this preliminary hypothesis.
    if best.sum() < 3:
        raise ValueError('table_plane_not_supported')
    center = sample[best].mean(axis=0)
    _, singular, axes = np.linalg.svd(sample[best] - center, full_matrices=False)
    if singular[1] / np.sqrt(best.sum()) < 0.01:
        raise ValueError('table_region_too_narrow')
    normal = axes[-1]
    # Choose the normal pointing from the support plane toward the camera.
    if normal @ center > 0:
        normal = -normal
    residual = np.abs((sample - center) @ normal)
    keep = residual <= cfg.plane_tolerance_m
    if keep.mean() < cfg.min_plane_fraction:
        raise ValueError('table_plane_not_supported')
    return center, normal, float(keep.mean()), float(np.sqrt(np.mean(residual[keep] ** 2)))


def _circle(xy, cfg):
    origin = xy.mean(axis=0)
    local = xy - origin
    matrix = np.column_stack((2 * local, np.ones(len(local))))
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(local ** 2, axis=1), rcond=None)
    if rank != 3:
        raise ValueError('degenerate_side_surface')
    center = solution[:2] + origin
    distances = np.linalg.norm(xy - center, axis=1)
    radius = float(np.median(distances))
    rms = float(np.sqrt(np.mean((distances - radius) ** 2)))
    angles = np.sort(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
    arc = float(np.degrees(2 * np.pi - np.max(np.diff(np.r_[angles, angles[0] + 2*np.pi]))))
    if rms > cfg.max_circle_rms_m or arc < cfg.min_arc_deg:
        raise ValueError('insufficient_circular_side_surface')
    if not cfg.min_diameter_m <= 2 * radius <= cfg.max_diameter_m:
        raise ValueError('diameter_out_of_range')
    return center, radius, rms, arc


def estimate_points(object_points, table_points, *, cfg=Config()):
    """Fit side sections of an upright rotational body on an observed table.

    Returns a model center at half the observed height, not a mass centroid or
    a measured hidden backside. An upright axis is assumed, not independently
    measured. No grasp pose or robot base transform is produced.
    """
    obj, table = (np.asarray(p, dtype=float) for p in (object_points, table_points))
    for points in (obj, table):
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < cfg.min_points
                or not np.isfinite(points).all() or np.any(points[:, 2] <= 0)):
            raise ValueError('insufficient_or_invalid_points')
    origin, axis, plane_fraction, plane_rms = _plane(table, cfg)
    heights = (obj - origin) @ axis
    foreground = (heights > 2 * cfg.plane_tolerance_m) & (heights < cfg.max_height_m)
    if foreground.mean() < 0.7:
        raise ValueError('object_not_separated_from_table')
    obj, heights = obj[foreground], heights[foreground]
    height = float(np.quantile(heights, 0.99))
    if not cfg.min_height_m <= height <= cfg.max_height_m:
        raise ValueError('height_out_of_range')
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    basis = np.column_stack((first, np.cross(axis, first)))
    xy = (obj - origin) @ basis
    sections = []
    # Exclude the top cap; a filled disk cannot identify the side circle.
    for low, high in zip(np.linspace(.15, .75, 5)[:-1], np.linspace(.15, .75, 5)[1:]):
        keep = (heights >= low * height) & (heights < high * height)
        if keep.sum() < 20:
            continue
        try:
            center, radius, rms, arc = _circle(xy[keep], cfg)
        except ValueError:
            continue
        sections.append((center, radius, rms, arc, float(np.median(heights[keep]))))
    if len(sections) < 3:
        raise ValueError('insufficient_side_sections')
    centers = np.array([s[0] for s in sections])
    center_xy = np.median(centers, axis=0)
    spread = float(np.max(np.linalg.norm(centers - center_xy, axis=1)))
    if spread > cfg.max_center_spread_m:
        raise ValueError('upright_model_inconsistent')
    base = origin + basis @ center_xy
    return {
        'center_m': (base + axis * height / 2).tolist(),
        'support_center_m': base.tolist(),
        'axis': axis.tolist(),
        'axis_source': 'support_plane_normal_upright_assumption',
        'center_definition': 'axis_midpoint_from_support_to_observed_top',
        'top_surface': _top_surface(obj, base, axis, cfg),
        'dimensions': {'observed_height_m': height,
                       'observed_side_diameter_m': float(np.median([2*s[1] for s in sections])),
                       'sections': [{'height_m': s[4], 'diameter_m': 2*s[1]} for s in sections]},
        'quality': {'table_inlier_fraction': plane_fraction, 'table_rms_m': plane_rms,
                    'side_rms_m': max(s[2] for s in sections),
                    'min_side_arc_deg': min(s[3] for s in sections),
                    'section_center_spread_m': spread, 'object_points': len(obj),
                    'side_sections': len(sections), 'hardware_validated': False,
                    'axis_measured_independently': False,
                    'hidden_surface_measured': False,
                    'complete_height_verified': False},
    }


def localize(depth, object_mask, table_mask, metadata, *, cfg=Config()):
    """Return JSON-safe per-frame geometry or an explicit invalid result.

    Metadata must identify the acquisition time/domain and the mask instance.
    The masks must belong to this depth/color pair; association is upstream.
    """
    timestamp = float(metadata['timestamp_ms'])
    if not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError('Invalid acquisition timestamp')
    for key in ('timestamp_domain', 'instance_id', 'frame_id', 'mask_source'):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise ValueError(f'Missing {key}')
    result = {key: metadata[key] for key in
              ('timestamp_domain', 'instance_id', 'frame_id', 'mask_source')}
    result.update(schema_version=1, frame='color_optical', units='m',
                  timestamp_ms=timestamp, valid=False, geometry=None)
    try:
        if metadata.get('depth_registered_to') != 'color_optical':
            raise ValueError('Require depth registered to color_optical')
        if np.asarray(object_mask).dtype != np.bool_ or np.asarray(table_mask).dtype != np.bool_:
            raise ValueError('Masks must be boolean')
        if np.shape(object_mask) != np.shape(table_mask):
            raise ValueError('Mask size mismatch')
        if np.any(object_mask & table_mask):
            raise ValueError('Object and table masks overlap')
        obj = deproject(depth, object_mask, metadata['intrinsics'], metadata['depth_scale_m'])
        table = deproject(depth, table_mask, metadata['intrinsics'], metadata['depth_scale_m'])
        fraction = len(obj) / max(1, int(np.count_nonzero(object_mask)))
        if fraction < cfg.min_valid_fraction:
            raise ValueError('insufficient_valid_depth')
        geometry = estimate_points(obj, table, cfg=cfg)
        geometry['quality']['valid_depth_fraction'] = fraction
        result.update(valid=True, geometry=geometry, reason=None)
    except ValueError as exc:
        result['reason'] = str(exc)
    return result
