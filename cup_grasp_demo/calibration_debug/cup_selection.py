"""Optional size-based initial cup selection; never lock to an old cup position."""

import math

import cv2

from cup_grasp_demo.calibration_debug.core import write_json
from cup_grasp_demo.side_grasp.preview_index import depth_candidates
from dice_cup_localization.geometry import Config


def selection_options(cfg):
    opts = cfg.get('cup_selection')
    if opts is None:
        return None
    if not isinstance(opts, dict) or opts.get('mode') != 'size_range':
        raise ValueError('cup_selection.mode must be size_range')
    for key in ('height_range_mm', 'diameter_range_mm'):
        values = opts.get(key)
        if (not isinstance(values, (list, tuple)) or len(values) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in values)
                or not 0 < values[0] < values[1]):
            raise ValueError(f'Invalid cup_selection.{key}')
    return opts


def choose_candidate(proposals, cfg):
    opts = selection_options(cfg)
    rows, matches = [], []
    for index, (geom, _, _) in enumerate(proposals):
        height, diameter = geom['height_m'] * 1000, geom['radius_m'] * 2000
        reasons = []
        if opts is not None:
            for name, value, key in (('height', height, 'height_range_mm'),
                                     ('diameter', diameter, 'diameter_range_mm')):
                lo, hi = opts[key]
                if not math.isfinite(value) or not lo <= value <= hi:
                    reasons.append(f'{name}_outside_range')
        if not reasons:
            matches.append(index)
        rows.append(dict(index=index, bbox_xywh=geom['bbox_xywh'], height_mm=height,
                         diameter_mm=diameter, accepted=not reasons, reasons=reasons))
    passed = len(matches) == 1
    if passed:
        reason = '唯一几何候选' if opts is None else '唯一候选符合杯高和截面直径范围'
    elif opts is None:
        reason = f'Expected exactly one geometric proposal, got {len(proposals)}'
    elif not matches:
        reason = '没有候选符合杯高和截面直径范围；查看 cup_candidates.png 和尺寸配置'
    else:
        reason = '多个候选符合杯子尺寸；无法唯一选杯，查看 cup_candidates.png'
    return dict(passed=passed, reason=reason, selected_index=matches[0] if passed else None,
                mode='unique_geometry' if opts is None else opts['mode'], size_ranges_mm=opts,
                candidates=rows, semantic_identity_verified=False, position_filter_used=False)


def save_report(output, image, report):
    write_json(output / 'cup_candidates.json', report)
    overlay = image.copy()
    for row in report['candidates']:
        selected = row['index'] == report['selected_index']
        color = (0, 255, 0) if selected else (0, 180, 255)
        x, y, w, h = row['bbox_xywh']
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        label = f"{row['index']}: H{row['height_mm']:.0f} D{row['diameter_mm']:.0f} mm"
        cv2.putText(overlay, label, (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1)
    cv2.putText(overlay, 'Cup selection: ' + ('PASS' if report['passed'] else 'STOP'),
                (8, overlay.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (0, 255, 0) if report['passed'] else (0, 0, 255), 2)
    if not cv2.imwrite(str(output / 'cup_candidates.png'), overlay):
        raise OSError('Could not save cup candidate image')
    print(f"首次选杯：{report['reason']}；记录：{output / 'cup_candidates.json'}")


def select_cup(depth, image, meta, cfg, output=None):
    selection_options(cfg)
    from cup_grasp_demo.calibration_debug.cup_perception import perception_options, yolo_candidates
    backend = perception_options(cfg)['backend']
    try:
        if backend == 'yolo_seg_onnx':
            proposals, rejected = yolo_candidates(depth, image, meta, cfg, output)
        else:
            proposals, rejected = depth_candidates(
                depth, image, meta, Config(plane_tolerance_m=cfg['plane_tolerance_mm'] / 1000),
                cfg['contact_height_fraction'])
        report = choose_candidate(proposals, cfg)
        report['rejected_components'] = rejected
        if backend == 'yolo_seg_onnx' and not proposals and rejected:
            report['reason'] = ('YOLO 候选未通过工作区/深度几何检查：'
                                + '; '.join(row['reason'] for row in rejected)
                                + '；查看 yolo_seg.json')
    except ValueError as exc:
        report = dict(passed=False, reason=f'杯子几何提取失败：{exc}', candidates=[],
                      selected_index=None, semantic_identity_verified=False)
    if output is not None:
        if backend == 'yolo_seg_onnx' and (output / 'yolo_seg.png').exists():
            image = cv2.imread(str(output / 'yolo_seg.png'))
        report['backend'] = backend
        save_report(output, image, report)
    if not report['passed']:
        raise ValueError(report['reason'])
    return proposals[report['selected_index']]
