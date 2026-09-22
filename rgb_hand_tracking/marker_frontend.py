"""Latest-frame RGB acquisition and marker tracking, without robot I/O.

Marker continuity is independent of slow landmark inference. The consumer gets
one RGB frame and its own marker result, never metadata from a later frame.
"""
from collections import Counter
from copy import deepcopy
import math
from numbers import Integral, Real
import threading
import time

import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from rgb_hand_tracking.tracker import MarkerTracker


class MarkerRgbCamera:
    """Read/detect continuously; keep only the latest same-frame observation."""

    def __init__(self, device, config, *, camera_epoch='initial', marker_id=40,
                 max_gap_s=.3, tracker_options=None, camera_factory=None,
                 tracker_factory=None, clock=time.monotonic, warmup_frames=30,
                 read_timeout_s=2):
        if (isinstance(marker_id, bool) or not isinstance(marker_id, Integral)
                or not 0 <= marker_id < 50):
            raise ValueError('marker_id must be an integer in DICT_4X4_50 range 0..49')
        if (isinstance(max_gap_s, bool) or not isinstance(max_gap_s, Real)
                or not math.isfinite(max_gap_s) or max_gap_s <= 0):
            raise ValueError('max_gap_s must be finite and positive')
        if not isinstance(camera_epoch, str) or not camera_epoch:
            raise ValueError('camera_epoch must be a nonempty string')
        if (not isinstance(warmup_frames, int) or isinstance(warmup_frames, bool)
                or warmup_frames < 0):
            raise ValueError('warmup_frames must be a nonnegative integer')
        if (isinstance(read_timeout_s, bool) or not isinstance(read_timeout_s, Real)
                or not math.isfinite(read_timeout_s) or read_timeout_s <= 0):
            raise ValueError('read_timeout_s must be finite and positive')
        options = dict(tracker_options or {})
        options.update(marker_id=marker_id, max_gap_s=max_gap_s)
        self.tracker = (tracker_factory or MarkerTracker)(**options)
        self.camera_epoch = camera_epoch
        self.image_size = (config['height'], config['width'])
        self.clock = clock
        self.read_timeout_s = read_timeout_s
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.serial = 0
        self.latest = None
        self.failed = None
        self.closed = False
        self.tracking_generation = 0
        self._previous_valid = False
        self._previous_epoch = None
        self._previous_t = None
        self._previous_raw = None
        self._stats = {
            'frames': 0, 'tags_detected': 0, 'tags_confirmed': 0,
            'failures': 0, 'timestamps_strictly_forward': True,
            'first_capture_timestamp_s': None, 'last_capture_timestamp_s': None,
            'min_capture_interval_s': None, 'max_capture_interval_s': None,
            'marker_processing_ms_total': 0., 'marker_processing_ms_max': 0.,
        }
        self._states = Counter()
        self._generation_reasons = Counter()
        self.camera = (camera_factory or cv2.VideoCapture)(device, cv2.CAP_V4L2)
        try:
            if not self.camera.isOpened():
                raise RuntimeError('Cannot open RGB camera')
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config['format']))
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(warmup_frames):
                ok, _ = self.camera.read()
                if not ok:
                    raise RuntimeError('RGB warmup read failed')
            self.reported_fps = self.camera.get(cv2.CAP_PROP_FPS)
        except BaseException:
            self.camera.release()
            raise
        self.thread = threading.Thread(target=self._read, daemon=True,
                                       name='rgb-marker-reader')
        try:
            self.thread.start()
        except BaseException:
            self.camera.release()
            raise

    def _generation_event(self, marker, timestamp, epoch):
        reasons = []
        dt = None if self._previous_t is None else timestamp-self._previous_t
        if self._previous_epoch is not None and epoch != self._previous_epoch:
            reasons.append('camera_epoch_changed')
        if dt is not None and dt > self.tracker.max_gap_s:
            reasons.append('capture_gap')
        raw = marker.get('raw_corners_px')
        if raw is not None and self._previous_raw is not None and dt is not None:
            jump_px = getattr(self.tracker, 'jump_px', None)
            speed = getattr(self.tracker, 'max_speed_px_s', None)
            if (jump_px is not None and speed is not None and
                    np.linalg.norm(np.asarray(raw)-self._previous_raw, axis=1).max()
                    > jump_px+speed*dt):
                reasons.append('marker_jump')
        if marker.get('reason') == 'jump_reacquire' and 'marker_jump' not in reasons:
            reasons.append('marker_jump')
        if self._previous_valid and not marker['observation_valid']:
            reasons.append('tracking_invalidated')
        if reasons:
            self.tracking_generation += 1
            self._generation_reasons.update(reasons)
        self._previous_valid = bool(marker['observation_valid'])
        self._previous_epoch = epoch
        self._previous_t = timestamp
        self._previous_raw = None if raw is None else np.asarray(raw, float).copy()

    def _read(self):
        try:
            while not self.stop.is_set():
                ok, image = self.camera.read()
                if self.stop.is_set():
                    break
                if not ok:
                    raise RuntimeError('RGB read failed')
                timestamp = float(self.clock())
                if (not math.isfinite(timestamp) or
                        (self._previous_t is not None and timestamp <= self._previous_t)):
                    with self.condition:
                        self._stats['timestamps_strictly_forward'] = False
                    raise RuntimeError('Require finite strictly increasing capture timestamps')
                if (not isinstance(image, np.ndarray) or image.ndim != 3 or
                        image.shape[2] != 3 or image.dtype != np.uint8):
                    raise RuntimeError('Require BGR uint8 RGB frame')
                if image.shape[:2] != self.image_size:
                    raise RuntimeError('RGB frame size differs from calibrated stream size')
                # Own the pixels before another native read can reuse its buffer.
                image = image.copy()
                epoch = self.camera_epoch
                started = time.perf_counter()
                marker = self.tracker.update(image, timestamp, epoch)
                elapsed_ms = (time.perf_counter()-started)*1000
                if marker.get('timestamp_s') != timestamp:
                    raise RuntimeError('Marker result capture timestamp mismatch')
                if marker.get('marker_id') != self.tracker.marker_id:
                    raise RuntimeError('Marker result ID mismatch')
                with self.condition:
                    previous_t = self._previous_t
                    self._generation_event(marker, timestamp, epoch)
                    self.serial += 1
                    marker = deepcopy(marker)
                    marker.update(frame_serial=self.serial, capture_timestamp_s=timestamp,
                                  camera_epoch=epoch, tracking_generation=self.tracking_generation,
                                  width=image.shape[1], height=image.shape[0],
                                  motion_target_valid=False)
                    self.latest = (image, timestamp, self.serial, marker,
                                   self.tracking_generation)
                    stats = self._stats
                    stats['frames'] += 1
                    stats['tags_detected'] += int(
                        marker.get('raw_corners_px') is not None or marker.get('reason') in
                        ('marker_too_small_or_oblique', 'duplicate_marker_id'))
                    stats['tags_confirmed'] += int(bool(marker['observation_valid']))
                    if stats['first_capture_timestamp_s'] is None:
                        stats['first_capture_timestamp_s'] = timestamp
                    stats['last_capture_timestamp_s'] = timestamp
                    if previous_t is not None:
                        dt = timestamp-previous_t
                        low = stats['min_capture_interval_s']
                        high = stats['max_capture_interval_s']
                        stats['min_capture_interval_s'] = dt if low is None else min(low, dt)
                        stats['max_capture_interval_s'] = dt if high is None else max(high, dt)
                    stats['marker_processing_ms_total'] += elapsed_ms
                    stats['marker_processing_ms_max'] = max(stats['marker_processing_ms_max'],
                                                            elapsed_ms)
                    self._states.update([marker['state']])
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.failed = f'{type(exc).__name__}: {exc}'
                self._stats['failures'] += 1
                self.condition.notify_all()
        finally:
            self.camera.release()
            with self.condition:
                self.condition.notify_all()

    def next(self, previous):
        """Wait for a newer result, skipping every frame the consumer missed."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.failed is not None or self.stop.is_set() or self.serial > previous,
                timeout=self.read_timeout_s)
            if self.failed is not None:
                raise RuntimeError(self.failed)
            if self.stop.is_set():
                raise RuntimeError('RGB marker reader closed')
            if self.serial > previous:
                image, timestamp, serial, marker, generation = self.latest
                return image.copy(), timestamp, serial, deepcopy(marker), generation
        # A timed-out consumer must not leave a background reader owning /dev/video7.
        self.close()
        raise RuntimeError('RGB marker read timed out')

    def stats_snapshot(self):
        with self.condition:
            stats = deepcopy(self._stats)
            duration = (None if stats['frames'] < 2 else
                        stats['last_capture_timestamp_s']-stats['first_capture_timestamp_s'])
            stats.update(
                capture_mean_fps=None if duration is None else (stats['frames']-1)/duration,
                marker_processing_ms_mean=(stats['marker_processing_ms_total']/stats['frames']
                                           if stats['frames'] else None),
                reported_camera_fps=self.reported_fps,
                tracking_generation=self.tracking_generation,
                latest_frame_serial=self.serial,
                latest_timestamp_s=None if self.latest is None else self.latest[1],
                latest_marker_observation_valid=(False if self.latest is None else
                                                  bool(self.latest[3]['observation_valid'])),
                generation_change_reasons=dict(self._generation_reasons),
                marker_states=dict(self._states), failure_reason=self.failed,
                queue_policy='latest_only')
            return stats

    def close(self):
        if self.closed:
            return
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=self.read_timeout_s)
        if self.thread.is_alive():
            # V4L2 read can block after a disconnect. Release to unblock cleanup.
            self.camera.release()
            self.thread.join(timeout=self.read_timeout_s)
        self.closed = True
        if self.thread.is_alive():
            raise RuntimeError('RGB marker reader did not stop after camera release')
