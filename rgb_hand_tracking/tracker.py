"""Read-only RGB marker tracker. Pixel observations are NOT robot motion targets."""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


class AdaptiveFilter:
    """One Euro filter: stronger smoothing at rest, less lag during motion."""
    def __init__(self, min_cutoff=2.0, beta=0.02, derivative_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.raw = self.value = self.derivative = None

    @staticmethod
    def alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

    def update(self, value, dt):
        value = np.asarray(value, dtype=float)
        if self.raw is None:
            self.raw = value.copy()
            self.value = value.copy()
            self.derivative = np.zeros_like(value)
            return self.value.copy()
        a = self.alpha(self.derivative_cutoff, dt)
        self.derivative += a * ((value - self.raw) / dt - self.derivative)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.derivative, axis=-1, keepdims=True)
        self.value += self.alpha(cutoff, dt) * (value - self.value)
        self.raw = value.copy()
        return self.value.copy()


class MarkerTracker:
    """Marker ID association + quality gates + reacquisition + adaptive smoothing.

    Detects afresh each frame; filtering maintains temporal continuity. No stale
    predictions on loss. A camera epoch change clears all temporal state.
    """
    def __init__(self, marker_id=40, acquire_frames=3, min_side_px=24,
                 max_gap_s=0.3, jump_px=12, max_speed_px_s=600):
        self.marker_id = marker_id
        self.acquire_frames = acquire_frames
        self.min_side_px = min_side_px
        self.max_gap_s = max_gap_s
        self.jump_px = jump_px
        self.max_speed_px_s = max_speed_px_s
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        factory = getattr(cv2.aruco, 'DetectorParameters_create', None)
        self.params = factory() if factory else cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.last_t = None
        self.epoch = None
        self.reset()

    def reset(self):
        self.previous = None
        self.count = 0
        self.smoother = AdaptiveFilter()

    def _select_current(self, bgr):
        if hasattr(cv2.aruco, 'ArucoDetector'):
            corners, ids, _ = cv2.aruco.ArucoDetector(self.dictionary, self.params).detectMarkers(bgr)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(bgr, self.dictionary, parameters=self.params)
        return [] if ids is None else [c.reshape(4, 2) for c, i in zip(corners, ids.ravel())
                                      if i == self.marker_id]

    def update(self, bgr, timestamp_s, camera_epoch='initial'):
        if not np.isfinite(timestamp_s) or (self.last_t is not None and timestamp_s <= self.last_t):
            raise ValueError('Require finite strictly increasing capture timestamps')
        dt = None if self.last_t is None else timestamp_s - self.last_t
        self.last_t = timestamp_s
        if camera_epoch != self.epoch or (dt is not None and dt > self.max_gap_s):
            self.reset()
        self.epoch = camera_epoch
        selected = self._select_current(bgr)
        result = {'timestamp_s': timestamp_s, 'marker_id': self.marker_id,
                  'state': 'LOST', 'observation_valid': False,
                  'motion_target_valid': False, 'marker_corners_px': None,
                  'marker_center_px': None, 'palm_center_px': None}
        if len(selected) != 1:
            self.reset()
            result['reason'] = 'marker_missing' if not selected else 'duplicate_marker_id'
            return result
        raw = selected[0].astype(float)
        sides = np.linalg.norm(raw - np.roll(raw, 1, axis=0), axis=1)
        if min(sides) < self.min_side_px or max(sides) / min(sides) > 4:
            self.reset()
            result['reason'] = 'marker_too_small_or_oblique'
            return result
        jumped = self.previous is not None and np.max(np.linalg.norm(raw-self.previous, axis=1)) > self.jump_px + self.max_speed_px_s * dt
        if jumped:
            self.reset()
        self.count += 1
        self.previous = raw
        filtered = self.smoother.update(raw, dt or 1/30)
        valid = self.count >= self.acquire_frames
        result.update(state='TRACKING' if valid else 'ACQUIRING', observation_valid=valid,
                      reason='confirmed' if valid else ('jump_reacquire' if jumped else 'confirming'),
                      raw_corners_px=raw.tolist(), marker_corners_px=filtered.tolist(),
                      marker_center_px=filtered.mean(axis=0).tolist())
        return result


def calibrated_region(corners, template_points):
    """Project measured marker-plane template into RGB; not a 3D palm estimate.

    Template units: marker width=1; corners (0,0),(1,0),(1,1),(0,1).
    Caller must calibrate template after attachment. Off-plane points require 3D.
    """
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    h = cv2.getPerspectiveTransform(source, np.asarray(corners, np.float32))
    points = np.asarray(template_points, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, h).reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='JSON list of image_path, timestamp_s, optional camera_epoch')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tracker = MarkerTracker()
    rows = []
    for item in json.loads(args.manifest.read_text()):
        path = Path(item['image_path'])
        if not path.is_absolute():
            path = args.manifest.parent / path
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Cannot read {path}')
        rows.append(tracker.update(frame, item['timestamp_s'], item.get('camera_epoch', 'initial')))
    with args.output.open('x') as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
