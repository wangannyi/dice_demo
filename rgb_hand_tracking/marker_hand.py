"""Marker-guided hand2 bridge; outputs visual proxies, never physical palm targets."""
import argparse
import copy
import json
import math
from numbers import Integral, Real
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from rgb_hand_tracking.tracker import MarkerTracker, calibrated_region

PALM = [0, 5, 9, 13, 17]


def candidate_rects(corners, selected=None, *, search_mode='legacy'):
    if search_mode not in ('legacy', 'omni'):
        raise ValueError('search_mode must be legacy or omni')
    c = np.asarray(corners, float)
    side = np.linalg.norm(c-np.roll(c, 1, axis=0), axis=1).mean()
    angle = math.atan2(*(c[1]-c[0])[::-1]) + math.pi/2
    if selected is not None:
        params = [selected]
    elif search_mode == 'legacy':
        # Installation-specific hypothesis: marker +X points towards fingers.
        params = [(u, scale, delta) for u in (.5, 1.25, 2.)
                  for scale in (6., 8., 10.) for delta in (-.35, 0., .35)]
    else:
        # Attachment orientation is unknown; keep the center on the observed code.
        params = [(.5, scale, direction*math.pi/6) for scale in (6., 8., 10.)
                  for direction in range(12)]
    rects = []
    for u, scale, delta in params:
        center = calibrated_region(c, [[u, .5]])[0]
        rects.append([float(center[0]), float(center[1]), side*scale, side*scale, angle+delta])
    return params, rects


def normalized_palm(points, corners):
    h = cv2.getPerspectiveTransform(np.asarray(corners, np.float32),
                                   np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32))
    return cv2.perspectiveTransform(np.asarray(points, np.float32)[PALM, None], h)[:, 0]


def acceptable(hand, corners, shape):
    pts = np.asarray(hand['landmarks_px'], float)
    if hand['presence'] < .7 or pts.shape != (21, 2) or not np.isfinite(pts).all():
        return False
    palm = pts[PALM]
    height, width = shape[:2]
    if (palm < 0).any() or (palm[:, 0] >= width).any() or (palm[:, 1] >= height).any():
        return False
    local = normalized_palm(pts, corners)
    # Reject unrelated hands and collapsed skeletons even with high presence.
    return (np.linalg.norm(local.mean(axis=0)-[.5, .5]) < 2
            and .5 < np.linalg.norm(local[0]-local[2]) < 4
            and abs(cv2.contourArea(palm.astype(np.float32))) > 100)


class Bridge:
    def __init__(self, binary, root):
        self.process = subprocess.Popen([str(binary), str(root/'hand_detector.onnx'),
                                         str(root/'hand_landmarks.onnx')],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def infer(self, image, rects):
        self.process.stdin.write(json.dumps({'image': str(image), 'rects': rects})+'\n')
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('Landmark worker exited')
            if line.startswith('{'):
                return json.loads(line)['candidates']

    def close(self):
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def _validate_options(marker_id, max_gap_s, search_mode):
    if isinstance(marker_id, bool) or not isinstance(marker_id, Integral) or not 0 <= marker_id < 50:
        raise ValueError('marker_id must be an integer in DICT_4X4_50 range 0..49')
    if (isinstance(max_gap_s, bool) or not isinstance(max_gap_s, Real)
            or not math.isfinite(max_gap_s) or max_gap_s <= 0):
        raise ValueError('max_gap_s must be finite and positive')
    if search_mode not in ('legacy', 'omni'):
        raise ValueError('search_mode must be legacy or omni')


class MarkerGuidedObserver:
    """Persistent marker-guided landmarks for one source frame at a time.

    An external marker observation must describe exactly this image's capture
    timestamp. Its producer should pass a tracking_generation that advances on
    every acquisition reset, including losses between slow landmark calls.
    Changes of generation or camera epoch discard the old ROI and template.
    The marker-plane template is a visual proxy, never a physical palm point.
    """
    def __init__(self, binary, model_dir=None, *, marker_id=40, max_gap_s=.3,
                 search_mode='legacy'):
        _validate_options(marker_id, max_gap_s, search_mode)
        root = Path(__file__).resolve().parent
        self.marker_id = int(marker_id)
        self.search_mode = search_mode
        self.tracker = MarkerTracker(marker_id=self.marker_id, max_gap_s=float(max_gap_s))
        self.worker = Bridge(binary, Path(model_dir) if model_dir is not None else root/'vendor/hand2/models')
        self.closed = False
        self.last_t = None
        self.epoch = None
        self.tracking_generation = None
        self.reset()

    def reset(self):
        self.selected = None
        self.samples = []
        self.template = None
        self.trusted_streak = 0

    def close(self):
        if not self.closed:
            self.closed = True
            self.worker.close()

    def _infer(self, image_path, rects):
        self.landmark_calls += len(rects)
        try:
            return self.worker.infer(Path(image_path), rects)
        except Exception:
            self.close()
            raise

    def observe(self, frame, image_path, timestamp_s, camera_epoch='initial', *,
                marker_observation=None, tracking_generation=None):
        start = time.monotonic()
        self.landmark_calls = 0
        self.from_tracking = False
        row = self._observe(frame, image_path, timestamp_s, camera_epoch,
                            marker_observation=marker_observation,
                            tracking_generation=tracking_generation)
        row.update(processing_ms=(time.monotonic()-start)*1000, detector_calls=0,
                   landmark_calls=self.landmark_calls,
                   from_tracking=self.from_tracking and self.landmark_calls == 1)
        return row

    def _observe(self, frame, image_path, timestamp_s, camera_epoch, *,
                 marker_observation, tracking_generation):
        if self.closed:
            raise RuntimeError('Marker observer is closed')
        if (isinstance(timestamp_s, bool) or not isinstance(timestamp_s, Real)
                or not math.isfinite(timestamp_s)
                or (self.last_t is not None and timestamp_s <= self.last_t)):
            raise ValueError('Require finite strictly increasing capture timestamps')
        if (tracking_generation is not None and (isinstance(tracking_generation, bool)
                or not isinstance(tracking_generation, Integral) or tracking_generation < 0)):
            raise ValueError('tracking_generation must be a nonnegative integer')
        if marker_observation is None:
            marker = self.tracker.update(frame, timestamp_s, camera_epoch)
        else:
            # Copy so a high-frequency producer cannot change the observation
            # while its corresponding landmarks are being inferred.
            marker = copy.deepcopy(marker_observation)
            if (marker.get('timestamp_s') != timestamp_s
                    or marker.get('marker_id') != self.marker_id
                    or ('camera_epoch' in marker and marker['camera_epoch'] != camera_epoch)):
                raise ValueError('External marker observation must match source timestamp, ID and epoch')
        if camera_epoch != self.epoch or tracking_generation != self.tracking_generation:
            self.reset()
        self.last_t = timestamp_s
        self.epoch = camera_epoch
        self.tracking_generation = tracking_generation
        row = {'marker': marker, 'valid': False, 'motion_target_valid': False,
               'physical_palm_m': None, 'source': 'marker_guided_visual_proxy',
               'source_timestamp_s': timestamp_s, 'camera_epoch': camera_epoch,
               'tracking_generation': tracking_generation}
        if not marker['observation_valid']:
            self.reset()
            row['reason'] = 'marker_unconfirmed'
            return row
        c = np.asarray(marker['marker_corners_px'], float)
        if c.shape != (4, 2) or not np.isfinite(c).all():
            self.reset()
            raise ValueError('Valid marker observation must contain four finite corners')
        self.from_tracking = self.selected is not None
        params, rects = candidate_rects(c, self.selected, search_mode=self.search_mode)
        candidates = self._infer(image_path, rects)
        good = [(h, param) for h, param in zip(candidates, params) if acceptable(h, c, frame.shape)]
        if not good and self.selected is not None:
            params, rects = candidate_rects(c, search_mode=self.search_mode)
            good = [(h, param) for h, param in zip(self._infer(image_path, rects), params)
                    if acceptable(h, c, frame.shape)]
        if not good:
            self.selected, self.samples = None, []
            self.trusted_streak = 0
            row['reason'] = 'no_associated_landmarks'
            return row
        hand, self.selected = max(good, key=lambda pair: pair[0]['presence'])
        pts = np.asarray(hand['landmarks_px'])
        local = normalized_palm(pts, c)
        if self.template is not None and np.linalg.norm(local-self.template, axis=1).max() > .8:
            self.selected, self.samples = None, []
            self.trusted_streak = 0
            row['reason'] = 'palm_geometry_disagreement'
            return row
        self.trusted_streak += 1
        if self.template is None:
            self.samples.append(local)
            if len(self.samples) >= 3:
                stacked = np.array(self.samples)
                median = np.median(stacked, axis=0)
                if np.max(np.linalg.norm(stacked-median, axis=2)) > .5:
                    self.samples = self.samples[-1:]
                else:
                    self.template = median
        raw = pts[PALM].mean(axis=0)
        _, selected_rects = candidate_rects(c, self.selected, search_mode=self.search_mode)
        row.update(presence=hand['presence'], landmarks_px=pts.tolist(),
                   raw_visual_palm_px=raw.tolist(), roi_parameters=list(self.selected),
                   selected_roi_px=selected_rects[0])
        if self.template is not None and self.trusted_streak >= 3:
            proxy = calibrated_region(c, [self.template.mean(axis=0)])[0]
            row.update(valid=True, reason='visual_template_confirmed',
                       stable_visual_palm_px=proxy.tolist(),
                       template_marker_units=self.template.tolist())
        else:
            row['reason'] = 'template_confirming'
        return row


def run(manifest, output, binary, *, marker_id=40, max_gap_s=.3, search_mode='legacy'):
    observer = MarkerGuidedObserver(binary, marker_id=marker_id, max_gap_s=max_gap_s,
                                   search_mode=search_mode)
    rows = []
    created = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        created = True
        for i, item in enumerate(json.loads(manifest.read_text())):
            path = Path(item['image_path'])
            if not path.is_absolute():
                path = manifest.parent/path
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f'Cannot read {path}')
            row = observer.observe(frame, path, item['timestamp_s'], item.get('camera_epoch', 'initial'))
            row['frame'] = i
            if 'landmarks_px' in row:
                if row['valid']:
                    proxy = row['stable_visual_palm_px']
                    cv2.circle(frame, tuple(np.round(proxy).astype(int)), 7, (255, 255, 0), 2)
                for pt in row['landmarks_px']:
                    cv2.circle(frame, tuple(np.round(pt).astype(int)), 2, (0, 0, 255), -1)
                c = row['marker']['marker_corners_px']
                cv2.aruco.drawDetectedMarkers(frame, [np.asarray(c, np.float32)[None]],
                                              np.array([[observer.marker_id]], np.int32))
                cv2.imwrite(str(output/f'frame_{i:03d}.jpg'), frame)
            rows.append(row)
    finally:
        observer.close()
        if created:
            (output/'observations.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    print(json.dumps({'frames': len(rows), 'landmarks': sum('presence' in r for r in rows),
                      'visual_proxy_valid': sum(r['valid'] for r in rows)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--binary', type=Path, default=Path(__file__).resolve().parent/'build/landmark_bridge')
    p.add_argument('--marker-id', type=int, default=40, help='DICT_4X4_50 ID, 0..49')
    p.add_argument('--max-gap-s', type=float, default=.3,
                   help='Maximum capture interval before marker acquisition resets')
    p.add_argument('--search-mode', choices=('legacy', 'omni'), default='legacy')
    args = p.parse_args()
    run(args.manifest, args.output, args.binary, marker_id=args.marker_id,
        max_gap_s=args.max_gap_s, search_mode=args.search_mode)
