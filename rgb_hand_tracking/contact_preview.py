"""Offline hypothesis visualization, NOT an attachment calibration or motion plan."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def ray_plane(pixel, K, D, point, normal):
    xy = cv2.undistortPoints(np.asarray(pixel, float).reshape(1, 1, 2), K, D)[0, 0]
    ray = np.r_[xy, 1.0]
    denominator = normal @ ray
    if abs(denominator) < 1e-8:
        raise ValueError("Ray parallel to plane")
    scale = (normal @ point) / denominator
    if scale <= 0:
        raise ValueError("Plane intersection behind camera")
    return ray * scale


def project(point, K, D):
    return cv2.projectPoints(
        np.asarray(point, float).reshape(1, 3), np.zeros(3), np.zeros(3), K, D
    )[0][0, 0]


def main(root, annotation):
    from PIL import Image, ImageDraw, ImageFont

    camera = json.loads((root / "hi_camera.json").read_text())
    K, D = np.array(camera["K"]), np.array(camera["D"])
    rows = json.loads((root / "guided/observations.json").read_text())
    row = next(r for r in rows if r["valid"])
    index = row["frame"]
    corners = np.array(row["marker"]["raw_corners_px"], float)
    s = 0.015
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)
    solutions = cv2.solvePnPGeneric(obj, corners, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    thickness = json.loads(
        (Path(__file__).parent / "config/palm_model_geometry.json").read_text()
    )["nominal_back_to_palm_m"]
    hypotheses = []
    for rv, tv in zip(solutions[1], solutions[2]):
        R = cv2.Rodrigues(rv)[0]
        t = tv.ravel()
        dorsal = ray_plane(row["stable_visual_palm_px"], K, D, t, R[:, 2])
        contact = dorsal - R[:, 2] * thickness
        error = np.sqrt(
            np.mean(
                np.sum(
                    (cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(-1, 2) - corners)
                    ** 2,
                    axis=1,
                )
            )
        )
        hypotheses.append(
            {
                "marker_rvec": rv.ravel().tolist(),
                "marker_tvec_m": t.tolist(),
                "reprojection_rms_px": float(error),
                "contact_candidate_camera_m": contact.tolist(),
                "contact_candidate_px": project(contact, K, D).tolist(),
            }
        )
    hypotheses.sort(key=lambda h: h["reprojection_rms_px"])
    ann = json.loads(annotation.read_text())
    image_path = root / f"hi_{index}.png"
    if (
        ann["frame_index"] != index
        or ann["image_sha256"] != hashlib.sha256(image_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Manual cup annotation belongs to a different image")
    rim = np.asarray(ann["pixels"], float)
    if rim.ndim != 2 or rim.shape[1] != 2 or len(rim) < 6 or not np.isfinite(rim).all():
        raise ValueError("Invalid rim annotation")
    board = json.loads((root / "hi_observations.json").read_text())[index]
    T = np.array(board["T_camera_board"])
    Rb, tb = T[:3, :3], T[:3, 3]
    top_origin = tb - Rb[:, 2] * 0.0635
    pts = np.array(
        [Rb.T @ (ray_plane(uv, K, D, top_origin, Rb[:, 2]) - tb) for uv in rim]
    )
    A = np.c_[2 * pts[:, :2], np.ones(len(pts))]
    rhs = np.sum(pts[:, :2] ** 2, axis=1)
    fit = np.linalg.lstsq(A, rhs, rcond=None)[0]
    center = fit[:2]
    radius = np.sqrt(fit[2] + center @ center)
    center_camera = Rb @ np.r_[center, -0.0635] + tb
    cup_px = project(center_camera, K, D)
    result = {
        "status": "hypotheses_for_human_review_only",
        "frame": index,
        "motion_target_valid": False,
        "model_thickness_m": thickness,
        "marker_size_m": 0.03,
        "contact_hypotheses": hypotheses,
        "assumptions": [
            "Visual proxy is projected onto marker plane",
            "Local palm contact lies 26.6mm inward along marker normal",
            "Exact marker location and tilt on hand model NOT registered",
        ],
        "cup": {
            "method": "manual_top_boundary_plus_board_plane_circle_fit",
            "manually_selected_pixels": rim.tolist(),
            "top_above_board_m": 0.0635,
            "center_camera_m": center_camera.tolist(),
            "center_px": cup_px.tolist(),
            "fitted_radius_m": float(radius),
            "user_radius_m": 0.025,
            "radial_rms_m": float(
                np.sqrt(
                    np.mean((np.linalg.norm(pts[:, :2] - center, axis=1) - radius) ** 2)
                )
            ),
        },
        "independent_metric_accuracy_validated": False,
    }
    (root / "contact_preview.json").write_text(json.dumps(result, indent=2))
    original = Image.open(root / f"hi_{index}.png").convert("RGB")
    im = Image.new("RGB", (1280, 960), "white")
    im.paste(original, (0, 0))
    draw = ImageDraw.Draw(im)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 19)

    def mark(point, label, position, color):
        x, y = point
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=3)
        draw.line([tuple(point), position], fill=color, width=2)
        box = draw.textbbox(position, label, font=font)
        draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), fill="white")
        draw.text(position, label, font=font, fill=color)

    mark(corners.mean(axis=0), "A 标记中心", (855, 15), "#f54242")
    mark(hypotheses[0]["contact_candidate_px"], "B 掌面候选点", (870, 100), "#ff9b00")
    mark(hypotheses[1]["contact_candidate_px"], "B′ 另一姿态解", (885, 155), "#a0a0a0")
    mark(cup_px, "C 杯顶圆心候选", (290, 65), "#16dd82")
    for x, y in rim:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#16dd82")
    lines = [
        "核对图：不是运动目标。B 为隐藏掌面接触点在图像上的候选投影。",
        "B 假设：视觉参考点落在标记平面，再沿标记内法向偏移模型厚度 26.6 mm。",
        "B′ 表示单目平面标记的另一姿态解；标记与模型的安装位置尚未完整配准。",
        "C 来自人工选取杯顶边缘与已知高度的拟合，尚未接入自动杯顶检测。",
        "请核对 B 对应的掌面位置是否为你希望贴合杯顶边缘的接触位置。",
    ]
    for j, line in enumerate(lines):
        draw.text((24, 745 + j * 38), line, font=small, fill="#222222")
    im.save(root / "contact_review.png")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("--cup-annotation", type=Path, required=True)
    args = parser.parse_args()
    main(args.capture_root, args.cup_annotation)
