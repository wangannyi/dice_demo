"""YOLO-seg preprocessing; decoding lives in cup_perception (production path)."""

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
