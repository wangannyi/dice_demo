"""Open cup rim geometry inside a YOLO cap mask; high rim excludes the cavity."""

import cv2
import numpy as np
from dice_cup_localization.geometry import Config, _circle, _plane, deproject


def rim_geometry(points, table, normal, opts):
    points, table, normal = map(
        lambda x: np.asarray(x, dtype=float), (points, table, normal)
    )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 100
        or not np.isfinite(points).all()
    ):
        raise ValueError("绿色杯有效深度点不足")
    heights = (points - table) @ normal
    selected = heights > 0.02
    if selected.sum() < 100:
        raise ValueError("杯体未与桌面分离")
    points, heights = points[selected], heights[selected]
    # Only the high green band supplies a rim; no flat visible cup top is assumed.
    height = float(np.quantile(heights, 0.98))
    if not opts["height_range_mm"][0] <= height * 1000 <= opts["height_range_mm"][1]:
        raise ValueError("杯高超出绿色杯配置范围")
    rim = points[abs(heights - height) <= opts["rim_band_mm"] / 1000]
    if len(rim) < 40:
        raise ValueError("杯口边缘深度不足")
    axis = np.eye(3)[np.argmin(abs(normal))]
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(normal, u)))
    cfg = Config(
        min_diameter_m=0,
        max_diameter_m=float("inf"),
        max_circle_rms_m=opts["rim_rms_mm"] / 1000,
        min_arc_deg=opts["min_arc_deg"],
    )
    center, radius, rms, arc = _circle((rim - table) @ basis, cfg)
    minimum, maximum = opts["diameter_range_mm"]
    diameter = 2 * radius * 1000
    if not minimum <= diameter <= maximum:
        raise ValueError(
            f"diameter_out_of_range: 拟合直径 {diameter:.1f} mm，允许 {minimum:g}–{maximum:g} mm；"
            f"拟合高度 {height * 1000:.1f} mm，圆弧 {arc:.1f}°，残差 {rms * 1000:.1f} mm"
        )
    center3 = table + basis @ center + height * normal
    return dict(
        height_m=height,
        rim_center_camera_m=center3.tolist(),
        radius_m=radius,
        rim_rms_mm=rms * 1000,
        visible_arc_deg=arc,
        rim_points=len(rim),
        table_point_camera_m=table.tolist(),
        table_normal_camera=normal.tolist(),
        rim_basis_camera=basis.tolist(),
    )


def detect(
    depth, image, meta, opts, plane_tolerance_mm, *, instances, diagnostics=None
):
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(valid=False, candidates=[])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (
        ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165))
        & (hsv[:, :, 1] > 70)
        & (hsv[:, :, 2] > 50)
    )
    contours, _ = cv2.findContours(
        red.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("没有检测到红色工作区")
    workspace = np.zeros(depth.shape, np.uint8)
    cv2.drawContours(workspace, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    table, normal, fraction, rms = _plane(
        deproject(depth, red, meta["intrinsics"], meta["depth_scale_m"]),
        Config(plane_tolerance_m=plane_tolerance_mm / 1000),
    )
    candidates = []
    rejected = []
    for item in instances:
        mask = np.asarray(item["mask"], dtype=np.uint8) * 255
        if mask.shape != depth.shape:
            raise ValueError("YOLO mask resolution mismatch")
        area = int(np.count_nonzero(mask))
        if (
            area < opts["min_area_px"]
            or np.count_nonzero((mask > 0) & (workspace > 0)) / area < 0.9
        ):
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        diagnostic = dict(valid=False)
        diagnostics["candidates"].append(diagnostic)
        try:
            if opts.get("geometry_method", "depth_band") == "image_rim_depth":
                from cup_grasp_demo.calibration_debug.green_image_rim import estimate

                geometry, contour = estimate(
                    depth, image, meta, mask > 0, table, normal, opts, diagnostic
                )
            else:
                geometry = rim_geometry(
                    deproject(
                        depth, mask > 0, meta["intrinsics"], meta["depth_scale_m"]
                    ),
                    table,
                    normal,
                    opts,
                )
        except ValueError as exc:
            diagnostic["error"] = str(exc)
            rejected.append(str(exc))
            continue
        candidates.append((geometry, mask, contour))
    if len(candidates) != 1:
        raise ValueError(
            f"需要唯一绿色开口杯，合格候选数：{len(candidates)}；深度拒绝原因：{rejected}"
        )
    diagnostics["valid"] = True
    geometry, mask, contour = candidates[0]
    geometry["table_fit"] = dict(inlier_fraction=fraction, rms_mm=rms * 1000)
    return geometry, mask, contour
