"""YOLOv8 DFL segmentation and green-cup instance selection; no hardware IO."""

import hashlib
from pathlib import Path

import cv2
import numpy as np


def preprocess(image):
    """Letterbox a BGR image to the verified 640-square float RGB input."""
    h, w = image.shape[:2]
    scale = min(640 / w, 640 / h)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (640 - nw) // 2, (640 - nh) // 2
    padded = np.full((640, 640, 3), 114, np.uint8)
    padded[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], np.float32)/255
    return tensor, (scale, left, top, nw, nh)


def decode(outputs, image_shape, transform, confidence=.25):
    """Decode the verified 13-head COCO export, with class-aware NMS."""
    expected = []
    for size in (80, 40, 20):
        expected.extend([(1, 64, size, size), (1, 80, size, size), (1, 1, size, size)])
    expected.extend([(1, 32, s, s) for s in (80, 40, 20)])
    expected.append((1, 32, 160, 160))
    if len(outputs) != 13 or any(o.shape != s for o, s in zip(outputs, expected)):
        raise ValueError('Unsupported YOLO-seg output layout')
    if not all(np.isfinite(o).all() for o in outputs):
        raise ValueError('Nonfinite model output')
    boxes, scores, labels, coefficients = [], [], [], []
    for branch, size in enumerate((80, 40, 20)):
        cls = outputs[3*branch+1][0]
        ids = cls.argmax(axis=0)
        conf = cls.max(axis=0)
        yy, xx = np.nonzero(conf >= confidence)
        if len(xx) == 0:
            continue
        logits = outputs[3*branch][0, :, yy, xx].reshape(-1, 4, 16)
        prob = np.exp(logits - logits.max(axis=2, keepdims=True))
        distances = (prob*np.arange(16)).sum(axis=2)/prob.sum(axis=2)
        centers = np.column_stack((xx+.5, yy+.5))
        xy1 = (centers-distances[:, :2])*(640/size)
        xy2 = (centers+distances[:, 2:])*(640/size)
        boxes.extend(np.column_stack((xy1, xy2-xy1)).tolist())
        scores.extend(conf[yy, xx].tolist())
        labels.extend(ids[yy, xx].tolist())
        coefficients.extend(outputs[9+branch][0, :, yy, xx])
    scale, left, top, nw, nh = transform
    height, width = image_shape[:2]
    results = []
    for label in sorted(set(labels)):
        indices = [i for i, value in enumerate(labels) if value == label]
        keep = cv2.dnn.NMSBoxes([boxes[i] for i in indices], [scores[i] for i in indices],
                                confidence, .45)
        for kept in np.asarray(keep).reshape(-1):
            i = indices[int(kept)]
            x, y, w, h = boxes[i]
            x1, y1 = max(0, (x-left)/scale), max(0, (y-top)/scale)
            x2, y2 = min(width, (x+w-left)/scale), min(height, (y+h-top)/scale)
            if x2 <= x1 or y2 <= y1:
                continue
            logits = np.asarray(coefficients[i]) @ outputs[12][0].reshape(32, -1)
            prob = 1/(1+np.exp(-np.clip(logits.reshape(160, 160), -80, 80)))
            prob = cv2.resize(prob, (640, 640))[top:top+nh, left:left+nw]
            mask = cv2.resize(prob, (width, height)) > .5
            roi = np.zeros((height, width), bool)
            roi[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = True
            mask &= roi
            results.append({'class_id': label, 'score': scores[i],
                            'bbox_xyxy': [x1, y1, x2, y2], 'mask': mask})
    return results


def decode_standard2(outputs, image_shape, transform, confidence=.35):
    """Decode the verified two-class cap/ground Ultralytics layout.

    Class channels are already probabilities in this exact SpaceMIT export.
    Box rows are cx, cy, width, height in 640-square model pixels.
    """
    if (len(outputs) != 2 or outputs[0].shape != (1, 38, 8400)
            or outputs[1].shape != (1, 32, 160, 160)):
        raise ValueError('Unsupported cap/ground standard2 output layout')
    detection, prototype = outputs
    if not np.isfinite(detection).all() or not np.isfinite(prototype).all():
        raise ValueError('Nonfinite cap model output')
    classes = detection[0, 4:6]
    labels = classes.argmax(axis=0)
    scores = classes.max(axis=0)
    anchors = np.flatnonzero(scores >= confidence)
    scale, left, top, nw, nh = transform
    height, width = image_shape[:2]
    candidates = []
    for index in anchors:
        cx, cy, box_w, box_h = detection[0, :4, index]
        x1 = max(0.0, float((cx-box_w/2-left)/scale))
        y1 = max(0.0, float((cy-box_h/2-top)/scale))
        x2 = min(float(width), float((cx+box_w/2-left)/scale))
        y2 = min(float(height), float((cy+box_h/2-top)/scale))
        if x2-x1 < 1 or y2-y1 < 1:
            continue
        candidates.append((int(index), int(labels[index]), float(scores[index]),
                           [x1, y1, x2, y2]))
    results = []
    proto_flat = prototype[0].reshape(32, -1)
    for label in sorted(set(item[1] for item in candidates)):
        group = [item for item in candidates if item[1] == label]
        boxes = [[r[0], r[1], r[2]-r[0], r[3]-r[1]] for _, _, _, r in group]
        keep = cv2.dnn.NMSBoxes(boxes, [item[2] for item in group], confidence, .45)
        for local_index in np.asarray(keep).reshape(-1):
            anchor, _, score, box = group[int(local_index)]
            logits = detection[0, 6:38, anchor] @ proto_flat
            prob = cv2.resize((logits.reshape(160, 160) > 0).astype(np.uint8), (640, 640),
                              interpolation=cv2.INTER_NEAREST)
            prob = prob[top:top+nh, left:left+nw]
            mask = cv2.resize(prob, (width, height), interpolation=cv2.INTER_NEAREST) != 0
            roi = np.zeros((height, width), bool)
            x1, y1, x2, y2 = box
            roi[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = True
            mask &= roi
            results.append({'class_id': label, 'score': score, 'bbox_xyxy': box, 'mask': mask})
    return results


def select_green_cup(image, instances, min_green_fraction=.35, candidate_classes=(41,)):
    """Select a single COCO cup (41) using green fraction inside its YOLO mask.

    Multiple eligible cups are ambiguous; never silently choose a different cup.
    HSV is a filter after model inference, not an object detector.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 55, 35), (90, 255, 255)) != 0
    candidates = []
    for index, instance in enumerate(instances):
        mask = instance['mask']
        count = int(mask.sum())
        fraction = float(green[mask].mean()) if count else 0.
        instance['green_fraction'] = fraction
        if instance['class_id'] in candidate_classes and count >= 100 and fraction >= min_green_fraction:
            candidates.append(index)
    if len(candidates) != 1:
        return None, 'no_green_yolo_candidate' if not candidates else 'multiple_green_yolo_candidates'
    return candidates[0], None


class YoloSegmentor:
    """Load existing ONNX weights with an explicit available provider."""

    def __init__(self, model, provider='CPUExecutionProvider'):
        import onnxruntime as ort
        if provider not in ort.get_available_providers():
            raise ValueError(f'Provider unavailable: {provider}')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=[provider])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640]:
            raise ValueError('Unsupported model input')
        output_shapes = [tuple(x.shape) for x in self.session.get_outputs()]
        self.layout = 'standard2_cap_ground' if output_shapes == [
            (1, 38, 8400), (1, 32, 160, 160)] else 'spacemit13_coco'
        self.provenance = {'path': str(Path(model).resolve()),
                           'sha256': hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                           'source': 'existing K3 model cache; upstream URL from SDK downloader',
                           'upstream_url': 'https://archive.spacemit.com/spacemit-ai/model_zoo/'
                                           'vision/yolov8_seg/' + Path(model).name,
                           'runtime': ort.__version__, 'providers': self.session.get_providers(),
                           'layout': self.layout,
                           'outputs': [{'name': x.name, 'shape': x.shape}
                                       for x in self.session.get_outputs()]}

    def infer(self, image):
        tensor, transform = preprocess(image)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: tensor})
        if self.layout == 'standard2_cap_ground':
            return decode_standard2(outputs, image.shape, transform)
        return decode(outputs, image.shape, transform)
