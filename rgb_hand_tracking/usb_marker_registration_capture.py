"""One stationary USB RGB registration sample; passive SDK feedback only.

The SDK helper runs in its own Python runtime. Three raw RGB images contain
the hand marker and fixed tabletop board together. FK software brackets the
after-read timestamps; this is deliberately not exposure synchronization.
No branch is selected, physical palm inferred, configuration changed or robot
command sent. Importing this module opens no device.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from rgb_hand_tracking.board_rgb import BoardRgbObserver
from rgb_hand_tracking.marker_frontend import MarkerRgbCamera
from rgb_hand_tracking import marker_registration as registration
from rgb_hand_tracking.rgb_geometry import CalibratedRgbGeometry
from rgb_hand_tracking.tracker import MarkerTracker


PACKETS = ('joint_12', 'joint_34', 'joint_56', 'joint_7')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_window_tracker(**options):
    """Pin only this collector's raw detector; preserve existing defaults."""
    tracker = MarkerTracker(**options)
    tracker.params.cornerRefinementWinSize = 2
    if hasattr(tracker.params, 'relativeCornerRefinmentWinSize'):
        tracker.params.relativeCornerRefinmentWinSize = 1.0
    return tracker


def detector_metadata(tracker):
    values = {}
    for key in dir(tracker.params):
        if key.startswith('_'):
            continue
        value = getattr(tracker.params, key)
        if isinstance(value, (str, int, float, bool)):
            values[key] = value
    return {'opencv_version': cv2.__version__, 'parameters': values,
            'pose_corners': 'raw_corners_px_without_temporal_filter',
            'board_corner_convention': 'opencv_4_6'}


class PassivePoseClient:
    """Bounded JSONL exchange with a persistent read-only SDK subprocess."""

    def __init__(self, command, *, timeout_s=5, process_factory=subprocess.Popen):
        if not np.isfinite(timeout_s) or not 0 < timeout_s <= 15:
            raise ValueError('Require finite helper timeout in (0, 15] seconds')
        self.timeout_s, self.request_index, self.closed = timeout_s, 0, False
        self.closed_report = None
        self.messages = queue.Queue()
        self.process = process_factory(command, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, text=True, bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True,
                                       name='passive-pose-jsonl-reader')
        self.reader.start()
        try:
            self.ready = self._receive('ready')
            if self.ready.get('schema') != 'passive_nero_pose_bridge_v1':
                raise RuntimeError('Unsupported passive pose helper schema')
        except BaseException:
            self.close(strict=False)
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                self.messages.put(json.loads(line))
            self.messages.put(RuntimeError('Passive pose helper exited'))
        except Exception as exc:
            self.messages.put(exc)

    def _receive(self, event, request_id=None):
        try:
            row = self.messages.get(timeout=self.timeout_s)
        except queue.Empty as exc:
            raise RuntimeError('Passive pose helper timed out') from exc
        if isinstance(row, Exception):
            raise RuntimeError('Passive pose helper protocol failure') from row
        if (not isinstance(row, dict) or row.get('event') != event
                or (request_id is not None and row.get('request_id') != request_id)):
            raise RuntimeError('Unexpected passive pose helper response: '+str(row))
        if row.get('tx_attempts') != 0 or row.get('actual_tx_count') != 0:
            raise RuntimeError('Passive helper reports transmission activity')
        return row

    def snapshot(self):
        if self.closed:
            raise RuntimeError('Passive helper already closed')
        self.request_index += 1
        request_id = str(self.request_index)
        try:
            self.process.stdin.write(json.dumps({'op': 'snapshot',
                                                'request_id': request_id})+'\n')
            self.process.stdin.flush()
            return self._receive('snapshot', request_id)
        except BaseException:
            self.close(strict=False)
            raise

    def close(self, *, strict=True):
        if self.closed:
            return self.closed_report
        self.closed = True
        error = None
        try:
            if strict:
                try:
                    request_id = 'close-'+str(self.request_index)
                    self.process.stdin.write(json.dumps({'op': 'close',
                                                        'request_id': request_id})+'\n')
                    self.process.stdin.flush()
                    self.closed_report = self._receive('closed', request_id)
                    if self.closed_report.get('success') is not True:
                        raise RuntimeError('Passive helper did not disconnect successfully')
                except Exception as exc:
                    error = exc
            if self.process.stdin and not self.process.stdin.closed:
                try:
                    self.process.stdin.close()  # EOF requests finally/disconnect.
                except Exception as exc:
                    if strict and error is None:
                        error = exc
            try:
                code = self.process.wait(timeout=self.timeout_s)
                if strict and code != 0 and error is None:
                    error = RuntimeError('Passive helper exited with nonzero status')
            except subprocess.TimeoutExpired:
                if strict and error is None:
                    error = RuntimeError('Passive helper disconnect timed out')
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        finally:
            if self.process.stdout:
                self.process.stdout.close()
            self.reader.join(timeout=2)
        if error is not None:
            raise RuntimeError('Passive helper final verification failed') from error
        return self.closed_report


def checked_snapshot(row, previous=None):
    """Validate fresh four-packet feedback and convert flange FK m/rad."""
    if (not isinstance(row, dict) or row.get('event') != 'snapshot'
            or row.get('read_only') is not True or row.get('tx_attempts') != 0
            or row.get('actual_tx_count') != 0):
        raise ValueError('Require read-only zero-transmission SDK snapshot')
    for field, length in (('q_rad', 7), ('fk_flange_pose_m_rad', 6)):
        value = row.get(field)
        if (not isinstance(value, (list, tuple)) or len(value) != length
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
            raise ValueError('Require numeric SDK joint/FK vectors')
    q = np.asarray(row['q_rad'], float)
    fk = np.asarray(row['fk_flange_pose_m_rad'], float)
    if q.shape != (7,) or fk.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(fk).all():
        raise ValueError('Require finite seven joints and six-value flange FK')
    times = {}
    for field in ('packet_timestamps_before_epoch_s', 'packet_timestamps_after_epoch_s',
                  'packet_ages_s'):
        value = row.get(field)
        if (not isinstance(value, dict) or set(value) != set(PACKETS)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not np.isfinite(v) for v in value.values())):
            raise ValueError('Require four finite SDK packet timestamps/ages')
        times[field] = value
    if times['packet_timestamps_before_epoch_s'] != times['packet_timestamps_after_epoch_s']:
        raise ValueError('SDK packets changed while snapshot was copied')
    ages = list(times['packet_ages_s'].values())
    if min(ages) < 0 or max(ages) > .1:
        raise ValueError('SDK packet stale')
    span = row.get('packet_span_s')
    actual_span = max(times['packet_timestamps_after_epoch_s'].values())-min(
        times['packet_timestamps_after_epoch_s'].values())
    if (isinstance(span, bool) or not isinstance(span, (int, float))
            or not np.isfinite(span) or not 0 <= span <= .05
            or not np.isclose(span, actual_span, atol=1e-6, rtol=0)):
        raise ValueError('SDK packet span invalid')
    fields = ('request_start_monotonic_s', 'copy_start_monotonic_s',
              'copy_end_monotonic_s', 'request_end_monotonic_s')
    monotonic = [row.get(key) for key in fields]
    if (any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not np.isfinite(v) or v < 0 for v in monotonic)
            or monotonic != sorted(monotonic)):
        raise ValueError('SDK monotonic time order invalid')
    if previous is not None:
        if monotonic[0] <= previous['request_end_monotonic_s']:
            raise ValueError('SDK snapshot request did not advance')
        if any(times['packet_timestamps_after_epoch_s'][key]
               <= previous['packet_timestamps_after_epoch_s'][key] for key in PACKETS):
            raise ValueError('All four SDK packets must advance')
    return registration._core().pose_matrix(fk)


def stationarity_evidence(snapshots):
    if len(snapshots) < 8:
        raise ValueError('Need at least eight SDK snapshots')
    duration = snapshots[-1]['request_end_monotonic_s']-snapshots[0]['request_start_monotonic_s']
    if duration < .35:
        raise ValueError('Stationary SDK window must cover at least .35 seconds')
    transforms, previous = [], None
    for row in snapshots:
        transforms.append(checked_snapshot(row, previous))
        previous = row
    distances = [registration._core().distance(a, b)
                 for i, a in enumerate(transforms) for b in transforms[i+1:]]
    maximum_m, maximum_deg = (max(v[0] for v in distances), max(v[1] for v in distances))
    if maximum_m > .0005 or maximum_deg > .2:
        raise ValueError('Flange moved during stationary capture')
    return {'snapshot_count': len(snapshots), 'duration_s': duration,
            'maximum_pairwise_flange_distance_m': maximum_m,
            'maximum_pairwise_flange_angle_deg': maximum_deg,
            'thresholds': {'position_m': .0005, 'angle_deg': .2},
            'stationary': True, 'snapshots': deepcopy(snapshots)}


def capture_stationary_sample(camera, pose_client, board, geometry, *, camera_epoch,
                              marker_id=40, marker_length_m=.03, timeout_s=10,
                              clock=time.monotonic, sleep=time.sleep):
    """Collect a validated sample in memory; caller owns resource cleanup."""
    if not np.isfinite(timeout_s) or not 0 < timeout_s <= 60:
        raise ValueError('Require finite capture timeout in (0, 60] seconds')
    deadline = clock()+timeout_s
    snapshots, frames, observations = [], [], []

    def snapshot():
        if clock() > deadline:
            raise RuntimeError('Stationary capture timed out')
        row = pose_client.snapshot()
        checked_snapshot(row, snapshots[-1] if snapshots else None)
        snapshots.append(row)
        return row

    for index in range(8):
        if index:
            sleep(.05)
        snapshot()
    stationarity_evidence(snapshots)
    serial, generation = 0, None
    for index in range(3):
        before = snapshot()
        while True:
            if clock() > deadline:
                raise RuntimeError('Newer RGB capture timed out')
            frame, timestamp, next_serial, marker, current_generation = camera.next(serial)
            if (not isinstance(next_serial, int) or isinstance(next_serial, bool)
                    or next_serial <= serial or not np.isfinite(timestamp)):
                raise ValueError('RGB source serial/timestamp did not advance')
            serial = next_serial
            if timestamp > before['request_end_monotonic_s']:
                break
        after = snapshot()
        if not timestamp < after['request_start_monotonic_s']:
            raise ValueError('RGB is outside the software FK bracket')
        expected = {'marker_id': marker_id, 'camera_epoch': camera_epoch,
                    'capture_timestamp_s': timestamp, 'timestamp_s': timestamp,
                    'frame_serial': serial, 'tracking_generation': current_generation,
                    'width': frame.shape[1], 'height': frame.shape[0]}
        if (any(marker.get(key) != value for key, value in expected.items())
                or marker.get('observation_valid') is not True
                or marker.get('raw_corners_px') is None):
            raise ValueError('Require confirmed same-frame raw hand marker')
        if generation is not None and generation != current_generation:
            raise ValueError('Marker continuity changed during capture')
        generation = current_generation
        source = {'frame_serial': serial, 'timestamp_s': timestamp,
                  'camera_epoch': camera_epoch}
        board_observation = board.observe(frame)
        pose, _, _ = geometry._board_pose(board_observation)
        candidates = registration.ippe_square_candidates(
            marker['raw_corners_px'], geometry.camera_matrix, geometry.distortion,
            marker_length_m=marker_length_m, marker_id=marker_id, source=source)
        pair_distance = registration._core().distance(checked_snapshot(before), checked_snapshot(after))
        if pair_distance[0] > .0005 or pair_distance[1] > .2:
            raise ValueError('Before/after flange moved across RGB capture')
        frames.append(frame.copy())
        observations.append({
            'source': source, 'tracking_generation': generation,
            'T_camera_table_board': pose['T_camera_board'], 'table_board_pose': pose,
            'board_observation': board_observation,
            'hand_marker_pose_candidates': candidates, 'selected_marker_pose_index': None,
            'software_bracket': {'before': deepcopy(before), 'after': deepcopy(after),
                                 'capture_timestamp_monotonic_after_read_s': timestamp,
                                 'flange_distance_m': pair_distance[0],
                                 'flange_angle_deg': pair_distance[1],
                                 'hardware_synchronized': False}})
    evidence = stationarity_evidence(snapshots)
    branch_stability = []
    for branch in range(2):
        transforms = [registration._core().inverse(row['T_camera_table_board'])
                      @ np.asarray(row['hand_marker_pose_candidates']['candidates'][branch]
                                    ['T_camera_hand_marker']) for row in observations]
        distances = [registration._core().distance(a, b)
                     for i, a in enumerate(transforms) for b in transforms[i+1:]]
        maximum_m, maximum_deg = max(v[0] for v in distances), max(v[1] for v in distances)
        if maximum_m > .002 or maximum_deg > 1:
            raise ValueError('Same-index hand-marker branch is unstable across RGB captures')
        branch_stability.append({'candidate_index': branch,
                                 'maximum_pairwise_position_m': maximum_m,
                                 'maximum_pairwise_angle_deg': maximum_deg})
    stats = camera.stats_snapshot()
    if (stats.get('failure_reason') is not None or stats.get('tracking_generation') != generation
            or stats.get('latest_marker_observation_valid') is not True):
        raise ValueError('Marker/camera continuity invalidated before capture completed')
    representative = min(range(3), key=lambda i: observations[i]['table_board_pose']
                         ['reprojection_rms_px']+min(c['reprojection_rms_px'] for c in
                         observations[i]['hand_marker_pose_candidates']['candidates']))
    chosen = deepcopy(observations[representative])
    chosen.update(stationary=True, T_base_flange=checked_snapshot(
        observations[representative]['software_bracket']['before']).tolist(),
        repeated_rgb_observations=observations, representative_rgb_index=representative,
        stationarity=evidence, branch_repeatability=branch_stability,
        marker_frontend=stats, physical_branch_verified=False,
        hardware_synchronized=False, motion_target_valid=False)
    return chosen, frames


@contextmanager
def locked_session(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path/'.capture.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield path


def load_session(path, metadata):
    """Resume only an identical attachment/calibration/source contract."""
    path = Path(path)
    manifest = path/'dataset.json'
    if not manifest.exists():
        if any(p.name != '.capture.lock' for p in path.iterdir()):
            raise ValueError('Uncommitted session artifacts require manual review')
        return dict(deepcopy(metadata), schema=registration.DATASET_SCHEMA, samples=[],
                    execution_enabled=False, motion_target_valid=False)
    dataset = json.loads(manifest.read_text())
    if (dataset.get('schema') != registration.DATASET_SCHEMA
            or any(dataset.get(key) != value for key, value in metadata.items())
            or not isinstance(dataset.get('samples'), list)):
        raise ValueError('Session calibration/attachment/source metadata mismatch')
    expected_directories = {f'sample_{index:03d}' for index in range(len(dataset['samples']))}
    if ({p.name for p in path.glob('sample_*')} != expected_directories
            or any(path.glob('.*.tmp'))):
        raise ValueError('Uncommitted session artifacts require manual review')
    for sample in dataset['samples']:
        for image in sample['images']:
            image_path = (path/image['path']).resolve()
            if not image_path.is_relative_to(path.resolve()) or sha256(image_path) != image['sha256']:
                raise ValueError('Session original image SHA mismatch')
    return dataset


def commit_sample(path, dataset, sample, frames):
    """Rename the complete lossless sample, then atomically publish the dataset."""
    path = Path(path)
    index = len(dataset['samples'])
    name = f'sample_{index:03d}'
    target = path/name
    if target.exists():
        raise ValueError('Uncommitted sample directory requires manual review')
    token = uuid.uuid4().hex
    stage, pending = path/f'.{name}.{token}.tmp', path/f'.dataset.{token}.tmp'
    stage.mkdir()
    renamed = False
    try:
        saved = deepcopy(sample)
        saved['sample_index'], saved['images'] = index, []
        for image_index, frame in enumerate(frames):
            filename = f'frame_{image_index}.png'
            output = stage/filename
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError('Cannot save raw RGB PNG')
            decoded = cv2.imread(str(output))
            if decoded is None or not np.array_equal(decoded, frame):
                raise RuntimeError('Stored PNG pixels differ from captured RGB')
            image = {'path': str(Path(name)/filename), 'sha256': sha256(output),
                     'width': frame.shape[1], 'height': frame.shape[0]}
            saved['images'].append(image)
            saved['repeated_rgb_observations'][image_index]['source'].update(
                image_path=image['path'], image_sha256=image['sha256'])
            saved['repeated_rgb_observations'][image_index]['hand_marker_pose_candidates']['source'].update(
                image_path=image['path'], image_sha256=image['sha256'])
        selected = saved['images'][saved['representative_rgb_index']]
        saved['source'].update(image_path=selected['path'], image_sha256=selected['sha256'])
        saved['hand_marker_pose_candidates']['source'].update(
            image_path=selected['path'], image_sha256=selected['sha256'])
        updated = deepcopy(dataset)
        updated['samples'].append(saved)
        pending.write_text(json.dumps(updated, indent=2, allow_nan=False)+'\n')
        stage.rename(target)
        renamed = True
        pending.replace(path/'dataset.json')
        return updated
    except BaseException:
        if renamed:
            shutil.rmtree(target)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if pending.exists():
            pending.unlink()


def optical_controls(text):
    return {name: int(value) for name, value in re.findall(
        r'^\s*(zoom_absolute|focus_absolute|focus_automatic)\s+.*?\bvalue=(-?\d+)',
        text, flags=re.MULTILINE)}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('capture-one', help='One stopped manual pose; no arm motion')
    capture.add_argument('--config', type=Path, required=True)
    capture.add_argument('--session', type=Path, required=True)
    capture.add_argument('--table-board-frame-id', required=True)
    capture.add_argument('--marker-attachment-epoch', required=True)
    capture.add_argument('--sdk-python', type=Path, required=True)
    capture.add_argument('--pose-helper', type=Path, default=root/'passive_pose_bridge.py')
    capture.add_argument('--can-interface', default='can0')
    capture.add_argument('--marker-id', type=int, default=40)
    capture.add_argument('--marker-length-m', type=float, default=.03)
    capture.add_argument('--timeout-s', type=float, default=10)
    args = parser.parse_args()
    if (not args.table_board_frame_id.strip() or not args.marker_attachment_epoch.strip()
            or not args.can_interface.strip() or not 0 <= args.marker_id < 50
            or not np.isfinite(args.marker_length_m) or not 0 < args.marker_length_m < 1
            or not np.isfinite(args.timeout_s) or not 0 < args.timeout_s <= 60):
        parser.error('Require nonempty epochs/channel, valid marker ID/side and timeout <=60s')
    config = json.loads(args.config.read_text())
    camera_config = config['camera']
    intrinsic_path = Path(camera_config['intrinsics_file'])
    if not intrinsic_path.is_absolute():
        intrinsic_path = args.config.resolve().parent/intrinsic_path
    calibration = json.loads(intrinsic_path.read_text())
    geometry = CalibratedRgbGeometry(config, calibration)
    board = BoardRgbObserver(config['board'], corner_convention='opencv_4_6')
    tracker = fixed_window_tracker(marker_id=args.marker_id)
    detector = detector_metadata(tracker)
    control_command = ['v4l2-ctl', '-d', camera_config['device'], '--list-ctrls']
    controls_before = subprocess.check_output(control_command, text=True, timeout=5)
    expected = optical_controls(calibration['camera'].get('v4l2_controls_at_capture', ''))
    if optical_controls(controls_before) != expected:
        raise ValueError('Current optical controls differ from USB intrinsics calibration')
    sources = ['usb_marker_registration_capture.py', 'marker_registration.py',
               'marker_frontend.py', 'tracker.py', 'board_rgb.py', 'rgb_geometry.py']
    metadata = {'table_board_frame_id': args.table_board_frame_id,
                'table_board_fixed_during_collection': True,
                'marker_attachment_epoch': args.marker_attachment_epoch,
                'hand_marker': {'dictionary': 'DICT_4X4_50', 'marker_id': args.marker_id,
                                'marker_length_m': args.marker_length_m},
                'camera': {key: camera_config[key] for key in
                           ('backend', 'device', 'width', 'height', 'format')},
                'board': deepcopy(config['board']), 'intrinsics_sha256': sha256(intrinsic_path),
                'camera_matrix': geometry.camera_matrix.tolist(),
                'distortion_coefficients': geometry.distortion.tolist(),
                'detector': detector, 'optical_controls': expected,
                'source_sha256': {name: sha256(root/name) for name in sources},
                'pose_helper_sha256': sha256(args.pose_helper),
                'rigid_math_source_sha256': sha256(registration._core().__file__)}
    # A new acquisition stream has its own identity even though physical camera
    # pose/configuration may be unchanged; native frame serials restart at one.
    epoch = camera_config['camera_epoch']+':capture:'+uuid.uuid4().hex
    camera = client = None
    with locked_session(args.session) as session:
        dataset = load_session(session, metadata)
        try:
            client = PassivePoseClient([str(args.sdk_python), str(args.pose_helper),
                                        '--channel', args.can_interface])
            if any(previous['pose_helper_ready'].get('source_runtime_info')
                   != client.ready.get('source_runtime_info') for previous in dataset['samples']):
                raise ValueError('Passive helper SDK source/runtime changed within session')
            camera = MarkerRgbCamera(camera_config['device'], camera_config,
                                     camera_epoch=epoch, marker_id=args.marker_id,
                                     tracker_factory=fixed_window_tracker)
            sample, frames = capture_stationary_sample(
                camera, client, board, geometry, camera_epoch=epoch,
                marker_id=args.marker_id, marker_length_m=args.marker_length_m,
                timeout_s=args.timeout_s)
            sample['pose_helper_ready'] = deepcopy(client.ready)
            sample['camera_configuration_epoch'] = camera_config['camera_epoch']
            sample['v4l2_controls_before'] = controls_before
            sample['v4l2_controls_after'] = subprocess.check_output(
                control_command, text=True, timeout=5)
            if sample['v4l2_controls_after'] != controls_before:
                raise ValueError('Camera controls changed during capture')
        finally:
            try:
                if camera is not None:
                    camera.close()
            finally:
                if client is not None:
                    client.close()
        sample['pose_helper_closed'] = deepcopy(client.closed_report)
        updated = commit_sample(session, dataset, sample, frames)
        print(json.dumps({'dataset': str((session/'dataset.json').resolve()),
                          'sample_count': len(updated['samples']),
                          'selected_marker_pose_index': None,
                          'execution_enabled': False}, allow_nan=False))


if __name__ == '__main__':
    main()
