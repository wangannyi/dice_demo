"""Match a frozen cup to fresh geometric candidates before arm motion."""

import cv2
import numpy as np

from cup_grasp_demo.flow.core import write_json
from cup_grasp_demo.side_grasp.preview_index import depth_candidates
from dice_cup_localization.geometry import Config


def match_candidates(proposals, expected):
    """Require one match inside fixed gates; never retarget to the nearest object."""
    rows, matched = [], []
    for index, (geom, _, _) in enumerate(proposals):
        delta = np.asarray(geom['center_camera_m']) - expected['center_camera_m']
        errors = dict(center_mm=float(np.linalg.norm(delta) * 1000),
                      height_mm=abs(float(geom['height_m'] - expected['height_m'])) * 1000,
                      radius_mm=abs(float(geom['radius_m'] - expected['radius_m'])) * 1000)
        accepted = all(np.isfinite(value) and value <= 5 for value in errors.values())
        if accepted:
            matched.append(index)
        rows.append(dict(index=index, bbox_xywh=geom['bbox_xywh'],
                         center_camera_m=geom['center_camera_m'],
                         errors_mm=errors, matches_frozen_cup=accepted))
    passed = len(matched) == 1
    if passed:
        reason = '唯一候选与冻结杯位、杯高和半径匹配'
    elif not matched:
        reason = '没有候选同时满足原杯位、杯高和半径的 5 mm 门限；请检查杯位、遮挡或重新 capture'
    else:
        reason = '多个候选符合原杯位，目标存在歧义；停止执行'
    return dict(passed=passed, reason=reason, selected_index=matched[0] if passed else None,
                thresholds_mm=dict(center=5, height=5, radius=5), candidates=rows,
                frozen_target_unchanged=True, semantic_identity_verified=False)


def verify_cup(depth, image, meta, cfg, expected, output):
    """Save the matching evidence on success and on detection/matching failure."""
    from cup_grasp_demo.flow.cup_perception import perception_options, yolo_candidates
    backend = perception_options(cfg)['backend']
    try:
        if backend == 'yolo_seg_onnx':
            proposals, rejected = yolo_candidates(depth, image, meta, cfg, output)
        else:
            proposals, rejected = depth_candidates(
                depth, image, meta, Config(plane_tolerance_m=cfg['plane_tolerance_mm'] / 1000),
                cfg['contact_height_fraction'])
        report = match_candidates(proposals, expected)
        report['rejected_components'] = rejected
        if backend == 'yolo_seg_onnx' and not proposals and rejected:
            report['reason'] = ('YOLO 候选未通过工作区/深度几何检查：'
                                + '; '.join(row['reason'] for row in rejected)
                                + '；查看 yolo_seg.json')
    except ValueError as exc:
        report = dict(passed=False, reason=f'杯位复核无法提取几何候选：{exc}', candidates=[],
                      selected_index=None, frozen_target_unchanged=True)
    report['backend'] = backend
    write_json(output / 'cup_recheck.json', report)
    if backend == 'yolo_seg_onnx' and (output / 'yolo_seg.png').exists():
        image = cv2.imread(str(output / 'yolo_seg.png'))
    overlay = image.copy()
    for row in report['candidates']:
        selected = row['index'] == report['selected_index']
        color = (0, 255, 0) if selected else (0, 180, 255)
        x, y, w, h = row['bbox_xywh']
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        label = f"{row['index']}: " + ('matched cup' if selected else 'other candidate')
        cv2.putText(overlay, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    cv2.putText(overlay, 'Cup recheck: ' + ('PASS' if report['passed'] else 'STOP'),
                (8, overlay.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (0, 255, 0) if report['passed'] else (0, 0, 255), 2)
    if not cv2.imwrite(str(output / 'cup_recheck.png'), overlay):
        raise OSError('Could not save cup recheck image')
    print(f"杯位复核：{report['reason']}；记录：{output / 'cup_recheck.json'}")
    if not report['passed']:
        raise ValueError(report['reason'])
    errors = report['candidates'][report['selected_index']]['errors_mm']
    print(f"杯心差 {errors['center_mm']:.2f} mm；杯高差 {errors['height_mm']:.2f} mm；半径差 {errors['radius_mm']:.2f} mm")
    return report
