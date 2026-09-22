"""Green open-cup image rim with independently checked near-rim depth support."""

import cv2
import numpy as np
from dice_cup_localization.geometry import Config, _circle, deproject

DEFAULTS = dict(
    contour_source="green_outline",
    opening_value_quantile=0.5,
    hue_range=[35, 95],
    min_saturation=45,
    min_value=20,
    roi_expand_px=4,
    contour_error_px=2.0,
    min_image_arc_deg=270,
    min_axis_ratio=0.45,
    depth_band_px=2.5,
    depth_tolerance_mm=3.0,
    depth_sectors=36,
    min_depth_sector_fraction=0.5,
    min_depth_arc_deg=180,
    min_depth_points=40,
)


def options(opts):
    raw = opts.get("image_rim", {})
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("Unknown green_cup.perception.image_rim option")
    out = dict(DEFAULTS, **raw)
    if out["contour_source"] not in ("green_outline", "opening"):
        raise ValueError("image_rim.contour_source")
    ranges = dict(
        opening_value_quantile=(0.2, 0.7),
        min_saturation=(0, 255),
        min_value=(0, 255),
        roi_expand_px=(0, 10),
        contour_error_px=(0.25, 5),
        min_image_arc_deg=(180, 360),
        min_axis_ratio=(0.2, 1),
        depth_band_px=(0.5, 5),
        depth_tolerance_mm=(0.5, 8),
        depth_sectors=(12, 72),
        min_depth_sector_fraction=(0.25, 1),
        min_depth_arc_deg=(90, 360),
        min_depth_points=(10, 1000),
    )
    for key, (low, high) in ranges.items():
        value = out[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not np.isfinite(value)
            or not low <= value <= high
        ):
            raise ValueError("image_rim." + key)
    for key in ("roi_expand_px", "depth_sectors", "min_depth_points"):
        if type(out[key]) is not int:
            raise ValueError("image_rim." + key + " must be integer")
    hue = np.asarray(out["hue_range"], dtype=float)
    if (
        hue.shape != (2,)
        or not np.isfinite(hue).all()
        or not 0 <= hue[0] < hue[1] <= 179
    ):
        raise ValueError("image_rim.hue_range")
    for name in ("reference_dimensions_mm", "reference_tolerance_mm"):
        ref = opts.get(name)
        if ref is not None:
            if not isinstance(ref, dict) or set(ref) != {"height", "diameter"}:
                raise ValueError(name + " requires height and diameter")
            if any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not np.isfinite(v)
                or v <= 0
                for v in ref.values()
            ):
                raise ValueError(name + " must be positive finite mm")
    if ("reference_dimensions_mm" in opts) != ("reference_tolerance_mm" in opts):
        raise ValueError(
            "Reference dimensions and tolerances must be configured together"
        )
    return out


def arc_degrees(angles):
    angles = np.sort(np.mod(angles, 2 * np.pi))
    if len(angles) < 2:
        return 0.0
    return float(
        np.degrees(2 * np.pi - np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi])))
    )


def ellipse_coordinates(uv, ellipse):
    center, diameters, degrees = ellipse
    theta = np.radians(degrees)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    local = (np.asarray(uv) - center) @ rotation
    radii = np.asarray(diameters) / 2
    normed = local / radii
    return normed, rotation, radii


def image_contour(image, mask, opts, diagnostic):
    o = options(opts)
    expand = o["roi_expand_px"]
    roi = cv2.dilate(mask.astype("uint8"), np.ones((2 * expand + 1,) * 2, "uint8")) > 0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = (
        (hsv[:, :, 0] >= o["hue_range"][0])
        & (hsv[:, :, 0] <= o["hue_range"][1])
        & (hsv[:, :, 1] >= o["min_saturation"])
        & (hsv[:, :, 2] >= o["min_value"])
        & roi
    )
    contours, _ = cv2.findContours(
        green.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [
        c
        for c in contours
        if len(c) >= 20 and cv2.contourArea(c) >= opts["min_area_px"] * 0.5
    ]
    if not contours:
        raise ValueError("YOLO 区域内没有完整绿色杯口外沿")
    contour = max(contours, key=cv2.contourArea)
    if o["contour_source"] == "opening":
        # The silhouette includes the lower outer wall when seen obliquely.
        # Recover the cavity boundary and bridge holes made by white dice.
        silhouette = np.zeros(green.shape, np.uint8)
        cv2.drawContours(silhouette, [contour], -1, 1, cv2.FILLED)
        values = hsv[:, :, 2][green & (silhouette > 0)]
        threshold = float(np.quantile(values, o["opening_value_quantile"]))
        dark = ((hsv[:, :, 2] <= threshold) & green & (silhouette > 0)).astype("uint8")
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cavities, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cavities:
            raise ValueError("未找到独立杯口开口区域")
        cavity = max(cavities, key=cv2.contourArea)
        area_fraction = cv2.contourArea(cavity) / cv2.contourArea(contour)
        if not 0.25 <= area_fraction <= 0.9:
            raise ValueError("杯口开口与杯体外轮廓无法可靠分离")
        contour = cv2.convexHull(cavity)
        if len(contour) < 6:
            raise ValueError("杯口开口轮廓支撑点不足")
        diagnostic.update(contour_source="opening", opening_value_threshold=threshold,
                          opening_area_fraction=area_fraction,
                          contour_definition="opening boundary; outer lip thickness not measured")
    full = contour[:, 0, :].astype(float)
    selected = np.ones(len(full), bool)
    for _ in range(3):
        if selected.sum() < (6 if o["contour_source"] == "opening" else 20):
            raise ValueError("杯口图像轮廓支撑点不足")
        ellipse = cv2.fitEllipse(full[selected].astype("float32"))
        coords, _, radii = ellipse_coordinates(full, ellipse)
        if min(radii) <= 0:
            raise ValueError("杯口椭圆退化")
        residual = abs(np.linalg.norm(coords, axis=1) - 1) * min(radii)
        selected = residual <= o["contour_error_px"]
    coords, rotation, radii = ellipse_coordinates(full, ellipse)
    arc = arc_degrees(np.arctan2(coords[selected, 1], coords[selected, 0]))
    diagnostic.update(
        image_ellipse=dict(
            center_uv=list(ellipse[0]),
            diameters_px=list(ellipse[1]),
            angle_deg=float(ellipse[2]),
        ),
        contour_uv=full.tolist(),
        image_arc_deg=arc,
        contour_inlier_fraction=float(selected.mean()),
    )
    if (
        min(radii) / max(radii) < o["min_axis_ratio"]
        or arc < o["min_image_arc_deg"]
        or selected.mean() < 0.7
    ):
        raise ValueError("杯口图像外沿不完整或不是合格椭圆")
    if (
        min(full[:, 0]) <= 1
        or min(full[:, 1]) <= 1
        or max(full[:, 0]) >= image.shape[1] - 2
        or max(full[:, 1]) >= image.shape[0] - 2
    ):
        raise ValueError("杯口轮廓被画面边界裁切")
    return ellipse, green, contour


def estimate(depth, image, meta, mask, table, normal, opts, diagnostic):
    """Fit an image rim; use angularly distributed depth to determine rim-plane height.

    An upright cup is assumed: the mouth plane is parallel to the support plane.
    The 2-D ellipse alone never supplies metric scale or authorizes movement.
    """
    o = options(opts)
    ellipse, green, contour = image_contour(image, mask, opts, diagnostic)
    yy, xx = np.indices(depth.shape)
    pixels = np.c_[xx.ravel(), yy.ravel()]
    coords, rotation, radii = ellipse_coordinates(pixels, ellipse)
    distance = abs(np.linalg.norm(coords, axis=1) - 1) * min(radii)
    band = (distance.reshape(depth.shape) <= o["depth_band_px"]) & green
    valid = band & (depth > 0) & np.isfinite(depth)
    uv = np.c_[xx[valid], yy[valid]]
    points = deproject(depth, valid, meta["intrinsics"], meta["depth_scale_m"])
    heights = (points - table) @ normal
    normalized, _, _ = ellipse_coordinates(uv, ellipse)
    angles = np.mod(np.arctan2(normalized[:, 1], normalized[:, 0]), 2 * np.pi)
    sectors = np.minimum(
        (angles / (2 * np.pi) * o["depth_sectors"]).astype(int), o["depth_sectors"] - 1
    )
    # Equal angular votes prevent a dense tiny patch from determining the height.
    medians = np.full(o["depth_sectors"], np.nan)
    for sector in range(o["depth_sectors"]):
        vals = heights[sectors == sector]
        if len(vals) >= 2:
            medians[sector] = np.median(vals)
    tolerance = o["depth_tolerance_mm"] / 1000
    diagnostic.update(
        depth_uv=uv.tolist(),
        depth_height_mm=(heights * 1000).tolist(),
        sector_height_mm=[float(x * 1000) if np.isfinite(x) else None for x in medians],
        depth_band_px=o["depth_band_px"],
        section_height_fraction=1.0,
    )
    low, high = np.asarray(opts["height_range_mm"]) / 1000
    candidates = medians[np.isfinite(medians) & (medians >= low) & (medians <= high)]
    if not len(candidates):
        raise ValueError("杯口外沿附近没有杯高范围内的可靠深度")
    counts = np.array([np.sum(abs(medians - h) <= tolerance) for h in candidates])
    height = float(candidates[np.argmax(counts)])
    supported = np.isfinite(medians) & (abs(medians - height) <= tolerance)
    height = float(np.median(medians[supported]))
    supported = np.isfinite(medians) & (abs(medians - height) <= tolerance)
    keep = supported[sectors] & (abs(heights - height) <= tolerance)
    arc = arc_degrees(angles[keep])
    sector_fraction = float(supported.mean())
    diagnostic.update(
        height_candidate_mm=height * 1000,
        accepted_depth_uv=uv[keep].tolist(),
        depth_sector_fraction=sector_fraction,
        depth_arc_deg=arc,
        depth_inlier_points=int(keep.sum()),
    )
    if (
        sector_fraction < o["min_depth_sector_fraction"]
        or arc < o["min_depth_arc_deg"]
        or keep.sum() < o["min_depth_points"]
    ):
        raise ValueError(
            f"杯口深度覆盖不足：高度候选 {height * 1000:.1f} mm，角区覆盖 {sector_fraction:.0%}，圆弧 {arc:.1f}°；不输出有效三维圆心"
        )
    # Independent separated height clusters indicate mixed table/wall/rim depths.
    for h in candidates:
        if (
            abs(h - height) > 2 * tolerance
            and np.sum(abs(medians - h) <= tolerance) >= o["depth_sectors"] * 0.25
        ):
            raise ValueError("杯口附近存在互相矛盾的深度高度，无法确定杯口平面")
    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    outline = np.c_[np.cos(theta), np.sin(theta)] * radii @ rotation.T + ellipse[0]
    intr = meta["intrinsics"]
    rays = np.c_[
        (outline[:, 0] - intr["cx"]) / intr["fx"],
        (outline[:, 1] - intr["cy"]) / intr["fy"],
        np.ones(len(outline)),
    ]
    denominator = rays @ normal
    if np.any(abs(denominator) < 1e-6):
        raise ValueError("杯口射线与桌面近似平行")
    z = (np.asarray(table) @ normal + height) / denominator
    if np.any(z <= 0):
        raise ValueError("杯口射线交点不在相机前方")
    ring = rays * z[:, None]
    u = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
    u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(normal, u)))
    center, radius, rms, _ = _circle(
        (ring - table) @ basis,
        Config(
            min_diameter_m=opts["diameter_range_mm"][0] / 1000,
            max_diameter_m=opts["diameter_range_mm"][1] / 1000,
            max_circle_rms_m=opts["rim_rms_mm"] / 1000,
            min_arc_deg=350,
        ),
    )
    center3 = table + basis @ center + height * normal
    measured = dict(height=height * 1000, diameter=radius * 2000)
    if "reference_dimensions_mm" in opts:
        deviations = {
            k: measured[k] - opts["reference_dimensions_mm"][k] for k in measured
        }
        diagnostic["reference_deviation_mm"] = deviations
        if any(
            abs(v) > opts["reference_tolerance_mm"][k] for k, v in deviations.items()
        ):
            raise ValueError(
                f"视觉尺寸与实测参考不符：估计 {measured} mm，参考 {opts['reference_dimensions_mm']} mm"
            )
    diagnostic.update(
        valid=True, rim_center_camera_m=center3.tolist(), diameter_mm=radius * 2000
    )
    return dict(
        height_m=height,
        rim_center_camera_m=center3.tolist(),
        radius_m=radius,
        rim_rms_mm=rms * 1000,
        visible_arc_deg=arc,
        rim_points=int(keep.sum()),
        table_point_camera_m=np.asarray(table).tolist(),
        table_normal_camera=np.asarray(normal).tolist(),
        rim_basis_camera=basis.tolist(),
        image_ellipse=diagnostic["image_ellipse"],
        depth_sector_fraction=sector_fraction,
        section_height_fraction=1.0,
        geometry_method="image_rim_depth",
        upright_rim_parallel_table_assumption=True,
    ), contour


def overlay(image, diagnostics):
    """A failed estimate still shows the contour and the rejected depth evidence."""
    canvas = image.copy()
    for item in diagnostics.get("candidates", []):
        ellipse = item.get("image_ellipse")
        if ellipse:
            cv2.ellipse(
                canvas,
                (
                    tuple(ellipse["center_uv"]),
                    tuple(ellipse["diameters_px"]),
                    ellipse["angle_deg"],
                ),
                (255, 255, 0),
                1,
            )
        for uv in item.get("depth_uv", []):
            cv2.circle(canvas, tuple(np.rint(uv).astype(int)), 1, (0, 165, 255), -1)
        for uv in item.get("accepted_depth_uv", []):
            cv2.circle(canvas, tuple(np.rint(uv).astype(int)), 1, (0, 255, 0), -1)
    label = (
        "FIXED HEIGHT - VISUAL RIM POSITION" if diagnostics.get("valid") and diagnostics.get("height_mode") == "fixed" else
        "RIM DEPTH VALID"
        if diagnostics.get("valid")
        else "RIM DEPTH INVALID - NO MOTION TARGET"
    )
    cv2.putText(
        canvas,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0) if diagnostics.get("valid") else (0, 0, 255),
        1,
    )
    return canvas
