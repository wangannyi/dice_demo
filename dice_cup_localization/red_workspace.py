"""Explicit red-mat work area in one D435i color image geometry."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


class RedWorkspace:
    """Conservative color-image polygon; no dependency on YOLO ground class."""

    def __init__(self, config_path, image_shape):
        config_path = Path(config_path)
        data = config_path.read_bytes()
        self.config = json.loads(data)
        if (self.config.get('schema_version') != 1 or self.config.get('frame') != 'color_optical'
                or (self.config.get('height'), self.config.get('width')) != image_shape[:2]):
            raise ValueError('Red workspace config does not match color image')
        vertices = np.asarray(self.config['polygon_xy'], np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError('Invalid red workspace polygon')
        self.mask = np.zeros(image_shape[:2], np.uint8)
        cv2.fillPoly(self.mask, [vertices], 255)
        self.mask = self.mask != 0
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.path = str(config_path.resolve())
        self.vertices = vertices

    def evaluate(self, image, instance_mask, other_masks=()):
        """Check cap mask inside polygon and red context outside all instances."""
        mask = np.asarray(instance_mask, dtype=bool)
        count = int(mask.sum())
        inside = float((mask & self.mask).sum()/count) if count else 0.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = self.config['red_hsv']
        low = cv2.inRange(hsv, (red['hue_low'][0], red['saturation_min'], red['value_min']),
                          (red['hue_low'][1], 255, 255)) != 0
        high = cv2.inRange(hsv, (red['hue_high'][0], red['saturation_min'], red['value_min']),
                           (red['hue_high'][1], 255, 255)) != 0
        kernel_size = int(self.config['context_dilation_px'])
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError('Invalid red workspace dilation size')
        neighborhood = cv2.dilate(mask.astype(np.uint8),
                                  np.ones((kernel_size, kernel_size), np.uint8)) != 0
        context = neighborhood & self.mask & ~mask
        for other in other_masks:
            context &= ~np.asarray(other, dtype=bool)
        context_count = int(context.sum())
        red_fraction = float(((low | high) & context).sum()/context_count) if context_count else 0.0
        valid = (inside >= self.config['min_instance_mask_inside_fraction']
                 and context_count >= 100
                 and red_fraction >= self.config['min_red_context_fraction'])
        return {'valid': bool(valid), 'mask_inside_fraction': inside,
                'red_context_fraction': red_fraction, 'red_context_pixels': context_count}

    def contains_projected(self, point_m, intrinsics):
        """Check a 3-D support point against the color-image red polygon."""
        x, y, z = [float(v) for v in point_m]
        if not np.isfinite([x, y, z]).all() or z <= 0:
            return False
        u = intrinsics['fx'] * x/z + intrinsics['cx']
        v = intrinsics['fy'] * y/z + intrinsics['cy']
        return cv2.pointPolygonTest(self.vertices, (float(u), float(v)), False) >= 0
