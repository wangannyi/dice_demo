"""Auto-adapt inference output shape and class names from model metadata.

Replaces the hard-coded [1,38,8400]/cap/ground contract in the original
green_yolo.py; a new model just needs metadata names + standard YOLOv8-seg
output layout.
"""
import ast
import numpy as np


def parse_class_names(metadata):
    """Read the names map from ORT custom metadata; tolerate string or dict."""
    names = metadata.get("names", "{}")
    if isinstance(names, dict):
        return names
    try:
        parsed = ast.literal_eval(names) if isinstance(names, str) else names
    except (ValueError, SyntaxError):
        raise ValueError("Model metadata names is not a valid mapping")
    if not isinstance(parsed, dict):
        raise ValueError("Model metadata names must map id -> class")
    return parsed


def adapt_outputs(outputs, class_names):
    """Normalize a YOLOv8-seg output pair to (detection, proto).

    Supports:
      - multi-class: detection [1, 4+nc, N], proto [1, 32, H, W]
        (scores not sigmoid'd; class scores occupy channels 4..4+nc)
      - legacy two-class green model: [1, 38, 8400] + [1, 32, 160, 160]
        (cap/ground with in-graph sigmoid; folded to single-class here)
    """
    if len(outputs) != 2:
        raise ValueError(f"Expected 2 outputs (detection+proto), got {len(outputs)}")
    det, proto = outputs
    if det.ndim != 3 or proto.ndim != 4:
        raise ValueError("Unexpected output ranks")
    nc = len(class_names)
    if nc < 1:
        raise ValueError("Model has no class names")
    if not all(np.isfinite(x).all() for x in (det, proto)):
        raise ValueError("Nonfinite YOLO outputs")
    if det.shape[1] == 4 + nc:
        # Standard layout: 4 box + nc class channels (no objectness)
        scores = det[:, 4:4+nc, :]
        if scores.min() < -1e-6 or scores.max() > 1 + 1e-6:
            raise ValueError("Class scores outside [0,1]")
        best = scores.argmax(axis=1, keepdims=True)
        best_score = np.take_along_axis(scores, best, axis=1)
        detection = np.concatenate([det[:, :4, :], best_score, det[:, 4+nc:, :]], axis=1)
        return detection, proto, int(best[0, 0, 0])
    # Legacy green model: det [1, 4+nc+32, N] — mask coefficients are the
    # last 32 channels; classes already in-graph sigmoid'd.
    if det.shape[1] == 4 + nc + 32 and nc == 2:
        scores = det[:, 4:4+nc, :]
        if scores.min() < -1e-6 or scores.max() > 1 + 1e-6:
            raise ValueError("Invalid class scores")
        # Fold to single class: keep selected class score, drop the other.
        detection = np.concatenate(
            [det[:, :4, :], scores[:, 0:1, :], det[:, 4+nc:, :]], axis=1)
        return detection, proto, 0
    raise ValueError(
        f"Cannot adapt model output: det {det.shape}, proto {proto.shape}, classes {nc}")
