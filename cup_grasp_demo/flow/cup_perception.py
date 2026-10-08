"""Optional single-class YOLOv8-seg cup masks with registered depth geometry."""

import ast
from functools import lru_cache
import hashlib
import importlib
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from vision.capture.frame_io import section
from vision.geometry.table_plane import Config, _plane, deproject
from vision.inference.yolo_seg import preprocess

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = dict(confidence=.35, iou_threshold=.45, mask_threshold=.5,
                min_workspace_fraction=.9, min_valid_depth_fraction=.5,
                mask_erode_px=1, section_half_band_mm=4., ort_package_dir=None)


def perception_options(cfg):
    raw = cfg.get('cup_perception', {'backend': 'depth_geometry'})
    if not isinstance(raw, dict):
        raise ValueError('cup_perception must be an object')
    if raw.get('backend') == 'depth_geometry':
        if set(raw) != {'backend'}:
            raise ValueError('depth_geometry does not take YOLO parameters')
        return dict(raw)
    if raw.get('backend') != 'yolo_seg_onnx':
        raise ValueError('cup_perception.backend must be depth_geometry or yolo_seg_onnx')
    if set(raw) - (set(DEFAULTS) | {'backend', 'model'}):
        raise ValueError('Unknown cup_perception parameter')
    if not isinstance(raw.get('model'), str) or not raw['model'].strip():
        raise ValueError('cup_perception.model is required')
    opts = dict(DEFAULTS, **raw)
    model = Path(opts['model'])
    resolved = (model if model.is_absolute() else ROOT / model).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError('Cup model must be inside dice_demo for portable session hashing')
    opts['model'] = str(resolved.relative_to(ROOT))
    for key in ('confidence', 'iou_threshold', 'mask_threshold',
                'min_workspace_fraction', 'min_valid_depth_fraction'):
        value = opts[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value < 1):
            raise ValueError(f'cup_perception.{key} must be between 0 and 1')
    value = opts['mask_erode_px']
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
        raise ValueError('cup_perception.mask_erode_px must be an integer in 0..5')
    value = opts['section_half_band_mm']
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not .5 <= value <= 4.):
        raise ValueError('cup_perception.section_half_band_mm must be in 0.5..4 mm')
    if opts['ort_package_dir'] is not None and not isinstance(opts['ort_package_dir'], str):
        raise ValueError('cup_perception.ort_package_dir must be null or a string')
    return opts


def model_path(cfg):
    opts = perception_options(cfg)
    return ROOT / opts['model'] if opts['backend'] == 'yolo_seg_onnx' else None


def decode(outputs, image_shape, transform, opts):
    """Decode this one-cap-class export: xywh, cap probability, 32 coefficients."""
    if (len(outputs) != 2 or outputs[0].shape != (1, 37, 8400)
            or outputs[1].shape != (1, 32, 160, 160)):
        raise ValueError('Expected one-class YOLO-seg outputs [1,37,8400] and [1,32,160,160]')
    if not all(np.isfinite(x).all() for x in outputs):
        raise ValueError('Nonfinite YOLO output')
    detection, prototypes = outputs
    scores = detection[0, 4]
    if np.min(scores) < -1e-6 or np.max(scores) > 1+1e-6:
        raise ValueError('Expected activated class probabilities')
    # SIMD sigmoid can undershoot zero by one float32 rounding step.
    scores = np.clip(scores, 0., 1.)
    indices = np.flatnonzero(scores >= opts['confidence'])
    boxes, anchors = [], []
    for index in indices:
        cx, cy, w, h = detection[0, :4, index]
        if w > 0 and h > 0:
            boxes.append([float(cx-w/2), float(cy-h/2), float(w), float(h)])
            anchors.append(int(index))
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, [float(scores[i]) for i in anchors],
                            opts['confidence'], opts['iou_threshold'])
    scale, left, top, nw, nh = transform
    height, width = image_shape[:2]
    py, px = np.indices((160, 160))
    results = []
    for kept in np.asarray(keep).reshape(-1):
        index = anchors[int(kept)]
        x, y, w, h = boxes[int(kept)]
        box = [max(0., (x-left)/scale), max(0., (y-top)/scale),
               min(float(width), (x+w-left)/scale), min(float(height), (y+h-top)/scale)]
        if box[2]-box[0] < 1 or box[3]-box[1] < 1:
            continue
        logits = (detection[0, 5:, index] @ prototypes[0].reshape(32, -1)).reshape(160, 160)
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        probabilities *= ((px >= x/4) & (px < (x+w)/4) & (py >= y/4) & (py < (y+h)/4))
        probabilities = cv2.resize(probabilities, (640, 640))[top:top+nh, left:left+nw]
        mask = cv2.resize(probabilities, (width, height)) > opts['mask_threshold']
        roi = np.zeros((height, width), bool)
        x1, y1, x2, y2 = box
        roi[int(y1):math.ceil(y2), int(x1):math.ceil(x2)] = True
        mask &= roi
        results.append(dict(class_id=0, class_name='cap', confidence=float(scores[index]),
                            bbox_xyxy=box, mask=mask))
    return results


@lru_cache(maxsize=2)
def _session(path, sha256, package_dir):
    # Keep the existing vision environment; add only ORT's package location if needed.
    for location in (os.environ.get('DICE_ORT_PACKAGE_DIR'),):
        if location and location not in sys.path:
            sys.path.insert(0, location)
    try:
        ort = importlib.import_module('onnxruntime')
    except ModuleNotFoundError as exc:
        if exc.name != 'onnxruntime':
            raise
        if package_dir and package_dir not in sys.path:
            sys.path.append(package_dir)
        ort = importlib.import_module('onnxruntime')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640]
            or inputs[0].type != 'tensor(float)'
            or [x.shape for x in outputs] != [[1, 37, 8400], [1, 32, 160, 160]]):
        raise ValueError('Unsupported cup ONNX input/output contract')
    metadata = session.get_modelmeta().custom_metadata_map
    if metadata.get('task') != 'segment' or ast.literal_eval(metadata.get('names', '{}')) != {0: 'cap'}:
        raise ValueError('Require segmentation model with class 0 named cap')
    return session, dict(model_sha256=sha256, runtime=ort.__version__,
                         providers=session.get_providers(), classes={'0': 'cap'},
                         input_shape=inputs[0].shape, output_shapes=[x.shape for x in outputs])


def infer(image, opts):
    path = ROOT / opts['model']
    try:
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        start = time.monotonic()
        session, provenance = _session(str(path), sha256, opts['ort_package_dir'])
        tensor, transform = preprocess(image)
        inference_start = time.monotonic()
        outputs = session.run(None, {session.get_inputs()[0].name: tensor})
        inference_s = time.monotonic() - inference_start
        instances = decode(outputs, image.shape, transform, opts)
        return instances, dict(provenance, model=opts['model'], inference_s=inference_s,
                               total_s=time.monotonic()-start)
    except Exception as exc:
        raise ValueError(f'YOLO-seg 推理失败，不回退几何选杯：{exc}') from exc


def yolo_candidates(depth, image, meta, cfg, output=None):
    opts = perception_options(cfg)
    if opts['backend'] != 'yolo_seg_onnx':
        raise ValueError('YOLO backend is not selected')
    proposals, rejected, instances = [], [], []
    report = dict(backend=opts['backend'], options=opts, instances=[],
                  fallback_used=False, independent_semantic_verification=False)
    overlay = image.copy()
    try:
        instances, provenance = infer(image, opts)
        report['model'] = provenance
        for index, item in enumerate(instances):
            row = {k: v for k, v in item.items() if k != 'mask'}
            row.update(instance=index, mask_pixels=int(item['mask'].sum()), geometry_accepted=False)
            report['instances'].append(row)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = (((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165))
               & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50))
        contours, _ = cv2.findContours(red.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError('Red workspace not visible')
        workspace = np.zeros(depth.shape, 'uint8')
        cv2.fillConvexPoly(workspace, cv2.convexHull(max(contours, key=cv2.contourArea)), 1)
        geometry_cfg = Config(plane_tolerance_m=cfg['plane_tolerance_mm']/1000)
        intr, scale = meta['intrinsics'], meta['depth_scale_m']
        all_masks = np.zeros(depth.shape, 'uint8')
        for item in instances:
            all_masks |= item['mask'].astype('uint8')
        table_mask = red & ~(cv2.dilate(all_masks, np.ones((21, 21), 'uint8')) > 0)
        for index, item in enumerate(instances):
            row = report['instances'][index]
            mask = item['mask']
            try:
                if mask.sum() < geometry_cfg.min_points:
                    raise ValueError('insufficient_mask_pixels')
                row['workspace_fraction'] = float((workspace[mask] > 0).mean())
                if row['workspace_fraction'] < opts['min_workspace_fraction']:
                    raise ValueError('outside_red_workspace')
                mask = mask & (workspace > 0)
                ring = (cv2.dilate(mask.astype('uint8'), np.ones((101, 101), 'uint8')) > 0) & table_mask
                table_points = deproject(depth, ring, intr, scale)
                if len(table_points) < geometry_cfg.min_points:
                    raise ValueError('insufficient_table_depth')
                table, normal, table_fraction, rms = _plane(table_points, geometry_cfg)
                radius = opts['mask_erode_px']
                if radius:
                    mask = cv2.erode(mask.astype('uint8'), np.ones((2*radius+1, 2*radius+1), 'uint8')) > 0
                row['valid_depth_fraction'] = float(((depth > 0) & np.isfinite(depth))[mask].mean()) if mask.any() else 0.
                if row['valid_depth_fraction'] < opts['min_valid_depth_fraction']:
                    raise ValueError('insufficient_valid_depth_in_yolo_mask')
                points = deproject(depth, mask, intr, scale)
                heights = (points-table) @ normal
                points = points[(heights > .015) & (heights < .30)]
                if len(points) < geometry_cfg.min_points:
                    raise ValueError('insufficient_cup_depth')
                # Keep only the requested height band: a wide band on a rounded
                # inverted cup can include its recessed bottom as well as its wall.
                cup_heights = (points-table) @ normal
                height = float(np.quantile(cup_heights, .99))
                contact_height = cfg['contact_height_fraction'] * height
                half_band = opts['section_half_band_mm'] / 1000
                count = int((abs(cup_heights-contact_height) < half_band).sum())
                row['section_fit'] = dict(
                    half_band_mm=opts['section_half_band_mm'], points=count,
                    required_points=geometry_cfg.min_points,
                    contact_height_mm=contact_height*1000,
                    max_circle_rms_mm=geometry_cfg.max_circle_rms_m*1000,
                    min_arc_deg=geometry_cfg.min_arc_deg)
                if count < geometry_cfg.min_points:
                    raise ValueError(f'insufficient_section_points: {count} < {geometry_cfg.min_points}')
                geom = section(points, table, normal, cfg['contact_height_fraction'], geometry_cfg,
                               half_band_m=half_band)
                row['section_fit'].update(circle_rms_mm=geom['circle_rms_m']*1000,
                                          visible_arc_deg=geom['visible_arc_deg'])
                geom['section_fit'] = dict(row['section_fit'])
                ys, xs = np.nonzero(mask)
                geom.update(table_fraction=table_fraction, table_rms_m=rms,
                            table_point_camera_m=table.tolist(),
                            bbox_xywh=[int(xs.min()), int(ys.min()), int(np.ptp(xs)+1), int(np.ptp(ys)+1)],
                            segmentation=dict(backend=opts['backend'], class_id=0, class_name='cap',
                                              confidence=item['confidence'], model_sha256=provenance['model_sha256']))
                row.update(geometry_accepted=True, height_mm=geom['height_m']*1000,
                           diameter_mm=geom['radius_m']*2000)
                proposals.append((geom, mask, points))
            except ValueError as exc:
                row['reason'] = str(exc)
                rejected.append(dict(instance=index, reason=str(exc)))
        report['geometric_candidates'] = len(proposals)
        return proposals, rejected
    except ValueError as exc:
        report['error'] = str(exc)
        raise
    finally:
        if output is not None:
            from cup_grasp_demo.flow.core import write_json
            for item, row in zip(instances, report['instances']):
                mask = item['mask']
                color = (0, 200, 0) if row['geometry_accepted'] else (0, 120, 255)
                overlay[mask] = (.6*overlay[mask]+.4*np.array(color)).astype('uint8')
                x1, y1, x2, y2 = map(int, item['bbox_xyxy'])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, f"cap {item['confidence']:.2f}", (x1, max(16, y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
            write_json(output / 'yolo_seg.json', report)
            if not cv2.imwrite(str(output / 'yolo_seg.png'), overlay):
                raise OSError('Could not write YOLO mask overlay')
