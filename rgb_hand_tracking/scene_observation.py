"""Same-frame USB RGB observations with optional cup geometry; no robot commands."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

PALM = [0, 5, 9, 13, 17]


class AutomaticHandBridge:
    def __init__(self, binary, model_dir):
        self.process = subprocess.Popen(
            [str(binary), str(model_dir/'hand_detector.onnx'), str(model_dir/'hand_landmarks.onnx')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def infer(self, image_path, frame_id, reset_tracking=False):
        self.process.stdin.write(json.dumps({'image': str(image_path), 'frame_id': frame_id,
                                            'reset_tracking': int(reset_tracking)})+'\n')
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('Automatic hand worker exited')
            if line.startswith('{'):
                return json.loads(line)

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=5)


class SceneObserver:
    """Every module sees the same lossless stored RGB frame, without cache reuse."""
    def __init__(self, hand_bridge, cup_detector, board_observer, geometry_observer=None,
                 *, hand_backend='no_marker'):
        if hand_backend not in ('no_marker', 'marker_guided'):
            raise ValueError('Unknown RGB hand backend')
        self.hand_bridge = hand_bridge
        self.hand_backend = hand_backend
        self.cup_detector = cup_detector
        self.board_observer = board_observer
        self.geometry_observer = geometry_observer
        self.last_timestamp = None
        self.epoch = None

    def observe(self, frame, image_path, frame_id, timestamp_s, camera_epoch='initial', *,
                marker_observation=None, tracking_generation=None, frame_serial=None):
        if (not np.isfinite(timestamp_s) or
                (self.last_timestamp is not None and timestamp_s <= self.last_timestamp)):
            raise ValueError('Require finite strictly increasing capture timestamps')
        if frame_id < 0 or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError('Require BGR uint8 image and nonnegative frame ID')
        stored = cv2.imread(str(image_path))
        if stored is None or not np.array_equal(stored, frame):
            raise ValueError('Stored hand input differs from the observed RGB frame')
        start = time.monotonic()
        if self.hand_backend == 'marker_guided':
            if marker_observation is not None:
                expected = {'capture_timestamp_s': timestamp_s, 'camera_epoch': camera_epoch,
                            'width': frame.shape[1], 'height': frame.shape[0],
                            'frame_serial': frame_serial,
                            'tracking_generation': tracking_generation}
                if any(marker_observation.get(k) != v for k, v in expected.items()):
                    raise ValueError('Marker result frame identity mismatch')
            guided = self.hand_bridge.observe(
                frame, image_path, timestamp_s, camera_epoch,
                marker_observation=marker_observation, tracking_generation=tracking_generation)
            if (guided['source_timestamp_s'] != timestamp_s
                    or guided['camera_epoch'] != camera_epoch):
                raise ValueError('Marker-guided hand result frame identity mismatch')
            hand = {'valid': bool(guided['valid']), 'reason': guided['reason'],
                    'source': 'marker_guided_visual_proxy',
                    'landmarks_px': guided.get('landmarks_px'),
                    'raw_visual_palm_px': guided.get('raw_visual_palm_px'),
                    'visual_palm_proxy_px': guided.get('stable_visual_palm_px'),
                    'physical_palm_m': None, 'marker': guided['marker'],
                    'tracking_generation': tracking_generation,
                    'roi_parameters': guided.get('roi_parameters'),
                    'selected_roi_px': guided.get('selected_roi_px'),
                    'presence': guided.get('presence'),
                    'from_tracking': guided.get('from_tracking', False),
                    'template_marker_units': guided.get('template_marker_units'),
                    'stats': {key: guided[key] for key in
                              ('processing_ms', 'detector_calls', 'landmark_calls')}}
            if hand['valid']:
                points = np.asarray(hand['landmarks_px'], dtype=float)
                proxy = np.asarray(hand['visual_palm_proxy_px'], dtype=float)
                if (points.shape != (21, 2) or proxy.shape != (2,)
                        or not np.isfinite(points).all() or not np.isfinite(proxy).all()):
                    raise ValueError('Invalid marker-guided hand coordinates')
        else:
            native = self.hand_bridge.infer(image_path, frame_id, camera_epoch != self.epoch)
            if (native.get('frame_id') != frame_id or native.get('width') != frame.shape[1]
                    or native.get('height') != frame.shape[0]):
                raise ValueError('Hand result frame identity mismatch')
            candidates = native['hands']
            if len(candidates) > 1:
                raise ValueError('Single-hand worker returned multiple hands')
            hand = {'valid': bool(candidates), 'reason': None if candidates else 'hand_not_detected',
                    'source': 'hand2_single_track_no_marker', 'landmarks_px': None,
                    'visual_palm_proxy_px': None, 'physical_palm_m': None}
            if candidates:
                points = np.asarray(candidates[0]['landmarks_px'], dtype=float)
                if points.shape != (21, 2) or not np.isfinite(points).all():
                    raise ValueError('Invalid hand worker coordinates')
                hand.update(candidates[0], visual_palm_proxy_px=points[PALM].mean(axis=0).tolist())
            hand['stats'] = {key: native[key] for key in
                             ('processing_ms', 'detector_calls', 'landmark_calls')}
        cup_start = time.monotonic()
        cup = self.cup_detector.process(frame)
        cup_elapsed = (time.monotonic()-cup_start)*1000
        board_start = time.monotonic()
        board = self.board_observer.observe(frame)
        board_elapsed = (time.monotonic()-board_start)*1000
        geometry = None
        if self.geometry_observer is not None:
            geometry_start = time.monotonic()
            geometry = self.geometry_observer.observe(board, cup)
            geometry_elapsed = (time.monotonic()-geometry_start)*1000
        relative = None
        if hand['valid'] and cup['valid']:
            center = np.asarray(cup['center_px'], dtype=float)
            relative = {'cup_center_minus_visual_palm_px':
                        (center-np.asarray(hand['visual_palm_proxy_px'])).tolist(),
                        'coordinate_space': 'image_pixels', 'physical_palm_used': False}
        self.last_timestamp, self.epoch = timestamp_s, camera_epoch
        observation = {'schema': 1, 'kind': 'usb_rgb_scene_observation', 'frame_id': frame_id,
                'capture_timestamp_s': float(timestamp_s), 'camera_epoch': camera_epoch,
                'source': {'image_path': str(Path(image_path).resolve()),
                           'sha256_image': hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
                           'sha256_bgr_pixels': hashlib.sha256(frame.tobytes()).hexdigest(),
                           'width': frame.shape[1], 'height': frame.shape[0]},
                'hand': hand, 'cup_top': cup, 'board': board, 'relative_pixels': relative,
                'pixel_observation_valid': bool(hand['valid'] and cup['valid'] and board['valid']),
                'metric_observation_valid': False, 'motion_target_valid': False,
                'physical_relative_m': None,
                'processing_ms': {'hand': hand['stats']['processing_ms'], 'cup': cup_elapsed,
                                  'board': board_elapsed, 'total': (time.monotonic()-start)*1000}}
        if self.geometry_observer is not None:
            observation['metric_geometry'] = geometry
            observation['processing_ms']['metric_geometry'] = geometry_elapsed
        # Cup geometry does not register the visual hand proxy to the physical palm.
        return observation


def annotate(frame, observation):
    result = frame.copy()
    hand = observation['hand']
    marker = hand.get('marker')
    if marker and marker.get('marker_corners_px') is not None:
        cv2.aruco.drawDetectedMarkers(result,
            [np.asarray(marker['marker_corners_px'], np.float32)[None]],
            np.array([[marker['marker_id']]], np.int32))
    if hand['valid']:
        points = np.round(hand['landmarks_px']).astype(int)
        chains = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (5, 9, 10, 11, 12),
                  (9, 13, 14, 15, 16), (13, 17, 18, 19, 20), (0, 17)]
        for chain in chains:
            for a, b in zip(chain, chain[1:]):
                cv2.line(result, tuple(points[a]), tuple(points[b]), (0, 255, 0), 1)
        for point in points:
            cv2.circle(result, tuple(point), 2, (0, 0, 255), -1)
        proxy = tuple(np.round(hand['visual_palm_proxy_px']).astype(int))
        cv2.circle(result, proxy, 5, (255, 255, 0), 2)
    cup = observation['cup_top']
    if cup['valid']:
        center = tuple(np.round(cup['center_px']).astype(int))
        cv2.drawMarker(result, center, (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
        ellipse = cup.get('ellipse_px')
        if ellipse:
            cv2.ellipse(result, (tuple(ellipse['center_px']), tuple(ellipse['diameters_px']),
                                ellipse['angle_deg']), (0, 255, 255), 2)
        if hand['valid']:
            cv2.line(result, proxy, center, (255, 255, 0), 1)
    board = observation['board']
    for point in board.get('charuco_corners_px', []):
        cv2.circle(result, tuple(np.round(point).astype(int)), 3, (255, 0, 255), -1)
    geometry = observation.get('metric_geometry')
    mode = 'PIXELS ONLY' if geometry is None else 'PALM PIXELS / CUP GEOMETRY'
    cv2.putText(result, f"frame {observation['frame_id']} H:{int(hand['valid'])} "
                f"C:{int(cup['valid'])} B:{int(board['valid'])} {mode}",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
    if geometry is not None:
        metric_cup = geometry['cup_top']
        if metric_cup['valid']:
            x, y, z = np.asarray(metric_cup['center_board_m'])*1000
            text = (f"Cup in board mm: ({x:.1f}, {y:.1f}, {z:.1f}) "
                    f"r={metric_cup['radius_m']*1000:.1f}; metric accuracy unverified")
        else:
            text = f"Cup geometry invalid: {metric_cup.get('reason', geometry.get('reason'))}"
        cv2.putText(result, text, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)
    return result


def load_metric_geometry(config, config_path):
    """Load explicitly configured intrinsics before opening camera or hand worker."""
    intrinsics_file = config.get('camera', {}).get('intrinsics_file')
    if not isinstance(intrinsics_file, str) or not intrinsics_file.strip():
        raise ValueError('--metric-geometry requires camera.intrinsics_file')
    path = Path(intrinsics_file)
    if not path.is_absolute():
        path = config_path.resolve().parent/path
    if not path.is_file():
        raise ValueError(f'Configured USB intrinsics file does not exist: {path}')
    calibration_report = json.loads(path.read_text())
    if not isinstance(calibration_report, dict):
        raise ValueError('USB intrinsics report must be a JSON object')
    from rgb_hand_tracking.rgb_geometry import CalibratedRgbGeometry
    return CalibratedRgbGeometry(config, calibration_report)


class LatestRgbCamera:
    def __init__(self, device, config):
        self.camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            raise RuntimeError('Cannot open RGB camera')
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config['format']))
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(30):
            ok, _ = self.camera.read()
            if not ok:
                self.camera.release()
                raise RuntimeError('RGB warmup read failed')
        self.reported_fps = self.camera.get(cv2.CAP_PROP_FPS)
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.failed = False
        self.serial = 0
        self.latest = None
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while not self.stop.is_set():
            ok, image = self.camera.read()
            with self.condition:
                if not ok:
                    self.failed = True
                    self.condition.notify_all()
                    return
                self.serial += 1
                self.latest = (image, time.monotonic(), self.serial)
                self.condition.notify_all()

    def next(self, previous):
        with self.condition:
            self.condition.wait_for(lambda: self.failed or self.serial > previous, timeout=2)
            if self.failed or self.serial <= previous:
                raise RuntimeError('RGB read failed or timed out')
            image, timestamp, serial = self.latest
            return image.copy(), timestamp, serial

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError('RGB reader did not stop')
        self.camera.release()


def check_marker_publication(observation, frontend_stats):
    """Suppress a hand result if marker continuity changed during inference.

    The frontend's newer corners never replace the recorded frame's corners.
    This check only invalidates a result whose tracking session has ended.
    """
    current_generation = frontend_stats['tracking_generation']
    observed_generation = observation['hand']['tracking_generation']
    current = (current_generation == observed_generation
               and frontend_stats.get('failure_reason') is None
               and frontend_stats.get('latest_marker_observation_valid', True))
    observation['marker_publication'] = {
        'tracking_generation_at_publication': current_generation,
        'latest_frame_serial': frontend_stats['latest_frame_serial'],
        'frontend_failure_reason': frontend_stats.get('failure_reason'),
        'continuity_valid': current}
    if not current:
        observation['hand'].update(valid=False, reason='marker_continuity_changed_during_processing',
                                   visual_palm_proxy_px=None)
        observation['pixel_observation_valid'] = False
        observation['relative_pixels'] = None


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path)
    source.add_argument('--device')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--config', type=Path, default=root/'config/usb7_relative_grasp.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hand-binary', type=Path, default=root/'build/hand_observation_bridge')
    parser.add_argument('--hand-backend', choices=('no_marker', 'marker_guided'), default='no_marker',
                        help='Explicitly select hand localization; default remains marker-free')
    parser.add_argument('--marker-binary', type=Path, default=root/'build/landmark_bridge')
    parser.add_argument('--marker-id', type=int, default=40)
    parser.add_argument('--marker-max-gap-s', type=float, default=.3,
                        help='Marker capture gap; live marker frontend runs independently of landmarks')
    parser.add_argument('--marker-search-mode', choices=('legacy', 'omni'), default='omni')
    parser.add_argument('--marker-roi-rescue', action='store_true',
                        help='Explicit same-frame re-detection of missed small hand tags')
    parser.add_argument('--rgb-profile', type=Path,
                        help='Explicit live marker RGB controls; read back and restore on exit')
    parser.add_argument('--model', type=Path,
                        default=root.parent/'dice_cup_localization/models/best.q.onnx')
    parser.add_argument('--cup-backend', choices=('green_top', 'yolo_confirmed'), default='green_top')
    parser.add_argument('--ort-package-dir', type=Path)
    parser.add_argument('--metric-geometry', action='store_true',
                        help='Explicitly enable calibrated board/cup geometry; palm remains pixels only')
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error('seconds must be between 0 and 60')
    if args.rgb_profile and (args.manifest or args.hand_backend != 'marker_guided'):
        parser.error('--rgb-profile requires live --device and --hand-backend marker_guided')
    if args.marker_roi_rescue and (args.manifest or args.hand_backend != 'marker_guided'):
        parser.error('--marker-roi-rescue requires live marker_guided capture')
    if args.ort_package_dir:
        sys.path.append(str(args.ort_package_dir))
    from rgb_hand_tracking.board_rgb import BoardRgbObserver
    from rgb_hand_tracking.cup_top import CupTopDetector
    config = json.loads(args.config.read_text())
    profile = None
    if args.rgb_profile:
        from rgb_hand_tracking.camera_profile import CameraBrightnessProfile
        try:
            profile = CameraBrightnessProfile(args.device, json.loads(args.rgb_profile.read_text()),
                                              config['camera'])
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    geometry = None
    if args.metric_geometry:
        try:
            geometry = load_metric_geometry(config, args.config)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    segmentor = None
    if args.cup_backend == 'yolo_confirmed':
        sys.path.insert(0, str(root.parent/'dice_cup_localization'))
        from yolo_seg import YoloSegmentor
        segmentor = YoloSegmentor(args.model)
    cup = CupTopDetector(segmentor, backend=args.cup_backend)
    board = BoardRgbObserver(config['board'], corner_convention=(
        'opencv_4_6' if args.metric_geometry else 'native'))
    args.output.mkdir(parents=True, exist_ok=False)
    if args.hand_backend == 'marker_guided':
        from rgb_hand_tracking.marker_hand import MarkerGuidedObserver
        hand = MarkerGuidedObserver(args.marker_binary, marker_id=args.marker_id,
                                   max_gap_s=args.marker_max_gap_s,
                                   search_mode=args.marker_search_mode)
    else:
        hand = AutomaticHandBridge(args.hand_binary, root/'vendor/hand2/models')
    camera = None
    rows, manifest = [], []
    report = None
    try:
        processor = SceneObserver(hand, cup, board, geometry, hand_backend=args.hand_backend)
        if args.manifest:
            items = json.loads(args.manifest.read_text())
        else:
            if profile is not None:
                profile.apply()
            if args.hand_backend == 'marker_guided':
                from rgb_hand_tracking.marker_frontend import MarkerRgbCamera
                tracker_factory = None
                if args.marker_roi_rescue:
                    from rgb_hand_tracking.marker_roi_tracker import RoiRescueMarkerTracker
                    tracker_factory = RoiRescueMarkerTracker
                camera = MarkerRgbCamera(args.device, config['camera'],
                                         camera_epoch=config['camera']['camera_epoch'],
                                         marker_id=args.marker_id, max_gap_s=args.marker_max_gap_s,
                                         tracker_factory=tracker_factory)
            else:
                camera = LatestRgbCamera(args.device, config['camera'])
            if profile is not None:
                profile.check_active()
            items = None
        start, previous = time.monotonic(), 0
        with (args.output/'observations.jsonl').open('w') as log:
            index = 0
            while ((items is not None and index < len(items)) or
                   (items is None and time.monotonic()-start < args.seconds)):
                marker_observation, generation = None, None
                if items is not None:
                    item = items[index]
                    image_path = Path(item['image_path'])
                    if not image_path.is_absolute():
                        image_path = args.manifest.parent/image_path
                    frame = cv2.imread(str(image_path))
                    if frame is None:
                        raise ValueError(f'Cannot read {image_path}')
                    timestamp = item['timestamp_s']
                    epoch = item.get('camera_epoch', config['camera']['camera_epoch'])
                    serial = None
                else:
                    if args.hand_backend == 'marker_guided':
                        frame, timestamp, serial, marker_observation, generation = camera.next(previous)
                    else:
                        frame, timestamp, serial = camera.next(previous)
                    previous = serial
                    epoch = config['camera']['camera_epoch']
                origin_path = image_path if items is not None else None
                image_path = args.output/f'frame_{index}.png'
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError('Cannot save lossless RGB frame')
                observation = processor.observe(
                    frame, image_path, index, timestamp, epoch,
                    marker_observation=marker_observation, tracking_generation=generation,
                    frame_serial=serial)
                if args.hand_backend == 'marker_guided' and camera is not None:
                    check_marker_publication(observation, camera.stats_snapshot())
                if origin_path is not None:
                    observation['source'].update(origin_image_path=str(origin_path.resolve()),
                                                 origin_image_sha256=hashlib.sha256(origin_path.read_bytes()).hexdigest())
                observation.update(capture_serial=serial, capture_clock='saved_manifest' if items is not None
                                   else 'host_monotonic_after_read_not_exposure',
                                   result_age_ms=None if items is not None else (time.monotonic()-timestamp)*1000)
                log.write(json.dumps(observation, allow_nan=False)+'\n')
                log.flush()
                if not cv2.imwrite(str(args.output/f'annotated_{index}.jpg'), annotate(frame, observation)):
                    raise RuntimeError('Cannot save annotation')
                rows.append(observation)
                manifest.append({'image_path': str(image_path.resolve()), 'timestamp_s': timestamp,
                                 'camera_epoch': epoch})
                index += 1
        elapsed = time.monotonic()-start
        if profile is not None:
            profile.check_active()
        report = {'frames': len(rows), 'hand_frames': sum(r['hand']['valid'] for r in rows),
                  'cup_top_frames': sum(r['cup_top']['valid'] for r in rows),
                  'board_frames': sum(r['board']['valid'] for r in rows),
                  'all_pixel_frames': sum(r['pixel_observation_valid'] for r in rows),
                  'elapsed_s': elapsed, 'output_fps': len(rows)/elapsed,
                  'mode': 'replay' if items is not None else 'live_rgb',
                  'scene_contract': config['scene_contract'], 'metric_observation_valid': False,
                  'motion_target_valid': False, 'cup_backend': args.cup_backend,
                  'cup_model': segmentor.provenance if segmentor is not None else None,
                  'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest()}
        if geometry is not None:
            report.update(metric_geometry_enabled=True,
                          board_pose_frames=sum(r['metric_geometry']['board_pose']['valid'] for r in rows),
                          cup_metric_frames=sum(r['metric_geometry']['cup_top']['valid'] for r in rows),
                          geometry_frames=sum(r['metric_geometry']['geometry_valid'] for r in rows),
                          independent_metric_accuracy_validated=False)
        if args.hand_backend == 'marker_guided':
            report.update(hand_backend='marker_guided', marker_id=args.marker_id,
                          marker_search_mode=args.marker_search_mode,
                          marker_max_gap_s=args.marker_max_gap_s,
                          hand_landmark_frames=sum(r['hand']['landmarks_px'] is not None for r in rows),
                          marker_confirmed_processed_frames=sum(
                              r['hand']['marker']['observation_valid'] for r in rows),
                          marker_frontend=None if camera is None else camera.stats_snapshot())
            if args.marker_roi_rescue:
                report['marker_roi_rescue'] = camera.tracker.rescue_stats()
        (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        if profile is None:
            print(json.dumps(report, indent=2))
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            try:
                hand.close()
            finally:
                if profile is not None:
                    try:
                        profile.restore()
                    finally:
                        (args.output/'camera_profile.json').write_text(
                            json.dumps(profile.record, indent=2, allow_nan=False)+'\n')
                        if report is not None:
                            report['rgb_acquisition_profile'] = profile.record
                            (args.output/'summary.json').write_text(
                                json.dumps(report, indent=2, allow_nan=False)+'\n')
    if profile is not None and report is not None:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
