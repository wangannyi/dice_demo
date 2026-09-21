"""Collect distinct, stationary ChArUco views from USB RGB; no robot IO."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from board_rgb import BoardRgbObserver


class SampleSelector:
    def __init__(self, observer, interval_s=1.2, novelty=0.012, motion_px=3.):
        self.observer = observer
        self.interval_s, self.novelty, self.motion_px = interval_s, novelty, motion_px
        board = observer.board
        self.points = np.asarray(board.getChessboardCorners() if hasattr(board, 'getChessboardCorners')
                                 else board.chessboardCorners, float)
        cfg = observer.cfg
        x = cfg.get('squares_x', 4)*cfg.get('square_length_m', .022)
        y = cfg.get('squares_y', 5)*cfg.get('square_length_m', .022)
        self.outline = np.array([[0, 0], [x, 0], [x, y], [0, y]], np.float32)
        self.previous = None
        self.accepted = []
        self.last_saved = None

    def consider(self, observation, timestamp_s):
        if not observation['valid'] or observation['charuco_corner_count'] < 8:
            self.previous = None
            return False, 'need_at_least_8_clear_corners'
        ids = np.asarray(observation['charuco_corner_ids'], int)
        pixels = np.asarray(observation['charuco_corners_px'], float)
        width, height = observation['image_size']
        distances = np.linalg.norm(pixels[:, None]-pixels[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        if np.median(distances.min(axis=1)) < 12:
            self.previous = None
            return False, 'board_too_small_move_closer'
        current = (ids, pixels, timestamp_s)
        previous, self.previous = self.previous, current
        if previous is None or timestamp_s-previous[2] > .8:
            return False, 'hold_board_still'
        shared, prior_indices, now_indices = np.intersect1d(previous[0], ids, return_indices=True)
        if len(shared) < 6 or np.median(np.linalg.norm(
                previous[1][prior_indices]-pixels[now_indices], axis=1)) > self.motion_px:
            return False, 'hold_board_still'
        if self.last_saved is not None and timestamp_s-self.last_saved < self.interval_s:
            return False, 'minimum_sample_interval'
        matrix, _ = cv2.findHomography(self.points[ids, :2], pixels, method=0)
        if matrix is None:
            return False, 'homography_failed'
        outline = cv2.perspectiveTransform(self.outline[None], matrix)[0]
        descriptor = outline/[width, height]
        if not np.isfinite(descriptor).all():
            return False, 'invalid_board_projection'
        if self.accepted and min(float(np.sqrt(np.mean((descriptor-p)**2)))
                                 for p in self.accepted) < self.novelty:
            return False, 'duplicate_view_change_position_distance_or_tilt'
        self.accepted.append(descriptor)
        self.last_saved = timestamp_s
        return True, 'accepted'


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='/dev/video7')
    parser.add_argument('--config', type=Path, default=root/'config/usb7_relative_grasp.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--target', type=int, default=24)
    args = parser.parse_args()
    if not 0 < args.seconds <= 180 or args.target < 20:
        parser.error('duration must be 0..180 seconds; target must be at least 20')
    config = json.loads(args.config.read_text())
    observer = BoardRgbObserver(config['board'])
    selector = SampleSelector(observer)
    args.output.mkdir(parents=True, exist_ok=False)
    camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError('Cannot open USB camera')
    samples = []
    dataset = {'schema': 1, 'camera': {**config['camera'], 'device': args.device},
               'board': config['board'], 'samples': samples,
               'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
               'motion_target_valid': False,
               'note': 'Camera fixed; operator moves board between stationary views.'}

    def persist():
        path = args.output/'manifest.json'
        temporary = args.output/'manifest.json.tmp'
        temporary.write_text(json.dumps(dataset, indent=2)+'\n')
        temporary.replace(path)

    try:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config['camera']['format']))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(30):
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('USB warmup failed')
        actual_width, actual_height = frame.shape[1], frame.shape[0]
        dataset['camera'].update(width=actual_width, height=actual_height,
                                 reported_fps=camera.get(cv2.CAP_PROP_FPS),
                                 timestamp='host_monotonic_after_read_not_exposure')
        if (actual_width, actual_height) != (config['camera']['width'], config['camera']['height']):
            raise RuntimeError('Camera resolution differs from configured runtime resolution')
        for flag, key in (('--list-ctrls', 'v4l2_controls_at_capture'), ('--info', 'v4l2_device_info')):
            result = subprocess.run(['v4l2-ctl', '-d', args.device, flag],
                                    capture_output=True, text=True, check=False, timeout=5)
            dataset['camera'][key] = result.stdout
        persist()
        print('READY: move only the board; keep each view still for 2-3 seconds.', flush=True)
        start, last_status, last_preview = time.monotonic(), 0., 0.
        reason = 'initial'
        while time.monotonic()-start < args.seconds and len(samples) < args.target:
            ok, frame = camera.read()
            timestamp = time.monotonic()
            if not ok:
                raise RuntimeError('USB capture read failed')
            observation = observer.observe(frame)
            accepted, reason = selector.consider(observation, timestamp)
            if accepted:
                filename = f'sample_{len(samples):03d}.png'
                path = args.output/filename
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError('Cannot save calibration view')
                samples.append({'image_path': filename, 'timestamp_s': timestamp,
                                'sha256_image': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'observation': observation})
                persist()
                cv2.imwrite(str(args.output/f'sample_{len(samples)-1:03d}_annotated.jpg'),
                            observer.annotate(frame, observation))
                print(f'SAVED {len(samples)}/{args.target}: {observation["charuco_corner_count"]} corners', flush=True)
            if timestamp-last_preview >= 1.:
                preview = observer.annotate(frame, observation)
                cv2.putText(preview, f'{len(samples)}/{args.target} {reason}', (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                cv2.imwrite(str(args.output/'preview.jpg'), preview)
                last_preview = timestamp
            if timestamp-last_status >= 10.:
                print(f'PROGRESS {len(samples)}/{args.target}: {reason}', flush=True)
                last_status = timestamp
        dataset['completion'] = {'elapsed_s': time.monotonic()-start, 'saved_views': len(samples),
                                  'target_views': args.target, 'last_reason': reason,
                                  'target_reached': len(samples) >= args.target}
        persist()
        print(json.dumps(dataset['completion']), flush=True)
    finally:
        try:
            persist()
        finally:
            camera.release()


if __name__ == '__main__':
    main()
