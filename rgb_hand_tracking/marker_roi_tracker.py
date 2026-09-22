"""Optional same-frame ROI re-detection for small, intermittently missed tags.

Previous corners only propose a search area. The current RGB crop must decode
the correct ID again; loss, epoch/gap/jump and confirmation gates remain active.
"""
import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from rgb_hand_tracking.tracker import MarkerTracker


class RoiRescueMarkerTracker(MarkerTracker):
    def __init__(self, **options):
        super().__init__(**options)
        self.roi_attempts = self.roi_rescues = 0
        self.detection_source = 'native_full_frame'

    def _select_current(self, bgr):
        self.detection_source = 'native_full_frame'
        selected = super()._select_current(bgr)
        if selected or self.previous is None:
            return selected
        # Native duplicate IDs remain a rejection; ROI cannot hide one.
        margin = np.linalg.norm(self.previous-np.roll(self.previous, 1, axis=0), axis=1).max()
        lo = np.floor(self.previous.min(axis=0)-margin).astype(int)
        hi = np.ceil(self.previous.max(axis=0)+margin).astype(int)
        height, width = bgr.shape[:2]
        x0, y0 = np.maximum(lo, [0, 0])
        x1, y1 = np.minimum(hi, [width, height])
        if x1-x0 < 24 or y1-y0 < 24:
            return []
        crop = cv2.resize(bgr[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        self.roi_attempts += 1
        candidates = super()._select_current(crop)
        if len(candidates) != 1:
            return []
        self.roi_rescues += 1
        self.detection_source = 'current_frame_roi_x2'
        return [candidates[0]/2+np.array([x0, y0])]

    def update(self, *args, **kwargs):
        result = super().update(*args, **kwargs)
        result['detection_source'] = self.detection_source
        return result

    def rescue_stats(self):
        return {'roi_attempts': self.roi_attempts, 'roi_decoded': self.roi_rescues,
                'strategy': 'native_full_frame_then_current_rgb_roi_x2',
                'stale_corners_published': False}
