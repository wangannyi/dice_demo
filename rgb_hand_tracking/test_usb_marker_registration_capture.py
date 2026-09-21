"""CPU/mock contracts for passive stopped-pose capture and recoverable storage."""
from copy import deepcopy
import io
import json
from pathlib import Path
import queue
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import usb_marker_registration_capture as capture


def pose_snapshot(index, *, translate=0):
    start = 10+index*.06
    stamps = {name: 1000+start for name in capture.PACKETS}
    return {'event': 'snapshot', 'snapshot_index': index, 'q_rad': [0.]*7,
            'fk_flange_pose_m_rad': [translate, 0, .2, 0, 0, 0],
            'packet_timestamps_before_epoch_s': stamps.copy(),
            'packet_timestamps_after_epoch_s': stamps.copy(),
            'packet_ages_s': dict.fromkeys(capture.PACKETS, .002), 'packet_span_s': 0.,
            'request_start_monotonic_s': start, 'copy_start_monotonic_s': start+.003,
            'copy_end_monotonic_s': start+.004, 'request_end_monotonic_s': start+.01,
            'read_only': True, 'tx_attempts': 0, 'actual_tx_count': 0}


class PoseClient:
    def __init__(self, modify=None):
        self.index, self.modify, self.last = 0, modify, None

    def snapshot(self):
        self.index += 1
        row = pose_snapshot(self.index)
        if self.modify:
            self.modify(row)
        self.last = row
        return row


class Geometry:
    camera_matrix = np.array([[500., 0, 80], [0, 500, 60], [0, 0, 1]])
    distortion = np.zeros(5)

    def _board_pose(self, observation):
        transform = np.eye(4)
        transform[2, 3] = .6
        return {'T_camera_board': transform.tolist(), 'reprojection_rms_px': .1}, None, None


class Board:
    def observe(self, frame):
        return {'valid': True, 'image_value': int(frame[0, 0, 0])}


class Camera:
    def __init__(self, pose, modify=None):
        self.pose, self.modify, self.index = pose, modify, 0
        self.stats = {'failure_reason': None, 'tracking_generation': 0,
                      'latest_marker_observation_valid': True}

    def next(self, previous):
        self.index += 1
        timestamp = self.pose.last['request_end_monotonic_s']+.01
        frame = np.full((120, 160, 3), self.index, np.uint8)
        half = .015
        objects = np.array([[-half, half, 0], [half, half, 0],
                            [half, -half, 0], [-half, -half, 0]])
        corners = cv2.projectPoints(objects, np.array([.3, .1, 0]),
                                    np.array([.03, 0, .5]), Geometry.camera_matrix,
                                    Geometry.distortion)[0].reshape(4, 2)
        marker = {'marker_id': 40, 'camera_epoch': 'stream_1',
                  'timestamp_s': timestamp, 'capture_timestamp_s': timestamp,
                  'frame_serial': self.index, 'tracking_generation': 0,
                  'width': 160, 'height': 120, 'observation_valid': True,
                  'raw_corners_px': corners.tolist(),
                  # Deliberately false filtered points: pose must use raw only.
                  'marker_corners_px': (corners+15).tolist()}
        if self.modify:
            self.modify(marker, frame)
        return frame, timestamp, self.index, marker, marker['tracking_generation']

    def stats_snapshot(self):
        return deepcopy(self.stats)


class Tests(unittest.TestCase):
    def sample(self, pose_modify=None, marker_modify=None, camera_stats=None):
        pose = PoseClient(pose_modify)
        camera = Camera(pose, marker_modify)
        if camera_stats:
            camera.stats.update(camera_stats)
        return capture.capture_stationary_sample(
            camera, pose, Board(), Geometry(), camera_epoch='stream_1',
            clock=lambda: 0., sleep=lambda seconds: None)

    def test_three_raw_same_frame_images_and_fk_brackets_never_select_branch(self):
        sample, frames = self.sample()
        self.assertTrue(sample['stationary'])
        self.assertEqual(sample['stationarity']['snapshot_count'], 14)
        self.assertEqual(len(frames), 3)
        self.assertIsNone(sample['selected_marker_pose_index'])
        self.assertFalse(sample['physical_branch_verified'])
        self.assertFalse(sample['hardware_synchronized'])
        self.assertFalse(sample['motion_target_valid'])
        for index, row in enumerate(sample['repeated_rgb_observations']):
            self.assertEqual(row['board_observation']['image_value'], index+1)
            bracket = row['software_bracket']
            self.assertLess(bracket['before']['request_end_monotonic_s'], row['source']['timestamp_s'])
            self.assertLess(row['source']['timestamp_s'], bracket['after']['request_start_monotonic_s'])
            candidates = row['hand_marker_pose_candidates']
            self.assertEqual(candidates['source'], row['source'])
            expected = np.eye(4)
            expected[:3, :3] = cv2.Rodrigues(np.array([.3, .1, 0]))[0]
            expected[:3, 3] = [.03, 0, .5]
            np.testing.assert_allclose(candidates['candidates'][0]['T_camera_hand_marker'],
                                       expected, atol=1e-8)
            self.assertEqual(len(candidates['candidates']), 2)
            self.assertIsNone(candidates['selected_candidate_index'])

    def test_snapshot_rejects_stale_mixed_nonadvance_and_transmission(self):
        cases = [lambda row: row['packet_ages_s'].update(joint_7=.101),
                 lambda row: row['packet_timestamps_after_epoch_s'].update(joint_7=999),
                 lambda row: row.update(tx_attempts=1),
                 lambda row: row.update(q_rad=[0]*6)]
        for modify in cases:
            with self.subTest(modify=modify):
                row = pose_snapshot(2)
                modify(row)
                with self.assertRaises(ValueError):
                    capture.checked_snapshot(row, pose_snapshot(1))
        with self.assertRaisesRegex(ValueError, 'advance'):
            capture.checked_snapshot(pose_snapshot(1), pose_snapshot(1))

    def test_flange_move_rejects_the_entire_in_memory_sample(self):
        def modify(row):
            if row['snapshot_index'] >= 10:
                row['fk_flange_pose_m_rad'][0] = .0006
        with self.assertRaisesRegex(ValueError, 'moved'):
            self.sample(pose_modify=modify)

    def test_initial_window_requires_count_and_elapsed_duration(self):
        with self.assertRaisesRegex(ValueError, 'eight'):
            capture.stationarity_evidence([pose_snapshot(1)])
        short = [pose_snapshot(i) for i in range(1, 9)]
        for index, row in enumerate(short):
            for field in ('request_start_monotonic_s', 'copy_start_monotonic_s',
                          'copy_end_monotonic_s', 'request_end_monotonic_s'):
                row[field] = 10+index*.01
        with self.assertRaisesRegex(ValueError, '.35'):
            capture.stationarity_evidence(short)

    def test_stale_latest_frame_is_skipped_until_after_before_snapshot(self):
        pose = PoseClient()

        class StaleFirstCamera(Camera):
            def next(self, previous):
                if self.index == 0:
                    self.index = 1
                    return np.zeros((120, 160, 3), np.uint8), 1., 1, {}, 0
                return super().next(previous)

        camera = StaleFirstCamera(pose)
        sample, _ = capture.capture_stationary_sample(
            camera, pose, Board(), Geometry(), camera_epoch='stream_1',
            clock=lambda: 0, sleep=lambda seconds: None)
        self.assertEqual(sample['repeated_rgb_observations'][0]['source']['frame_serial'], 2)

    def test_identity_loss_or_hidden_generation_change_rejects_capture(self):
        cases = [lambda marker, frame: marker.update(camera_epoch='wrong'),
                 lambda marker, frame: marker.update(frame_serial=123),
                 lambda marker, frame: marker.update(observation_valid=False),
                 lambda marker, frame: marker.update(tracking_generation=int(frame[0, 0, 0]))]
        for modify in cases:
            with self.subTest(modify=modify), self.assertRaises(ValueError):
                self.sample(marker_modify=modify)
        with self.assertRaisesRegex(ValueError, 'continuity'):
            self.sample(camera_stats={'tracking_generation': 1})
        with self.assertRaisesRegex(ValueError, 'continuity'):
            self.sample(camera_stats={'failure_reason': 'disconnected'})

    def test_same_branch_jump_is_rejected_without_swapping_or_low_rms_selection(self):
        original = capture.registration.ippe_square_candidates
        calls = []

        def candidate(*args, **kwargs):
            row = original(*args, **kwargs)
            calls.append(row)
            if len(calls) == 2:
                row['candidates'][1]['T_camera_hand_marker'][0][3] += .003
            return row

        with patch.object(capture.registration, 'ippe_square_candidates', candidate):
            with self.assertRaisesRegex(ValueError, 'branch is unstable'):
                self.sample()

    def test_fixed_detector_window_does_not_change_tracker_default(self):
        native = capture.MarkerTracker()
        original = native.params.cornerRefinementWinSize
        fixed = capture.fixed_window_tracker()
        self.assertEqual(fixed.params.cornerRefinementWinSize, 2)
        self.assertEqual(native.params.cornerRefinementWinSize, original)
        self.assertEqual(capture.detector_metadata(fixed)['parameters']['cornerRefinementWinSize'], 2)
        if hasattr(fixed.params, 'relativeCornerRefinmentWinSize'):
            self.assertEqual(fixed.params.relativeCornerRefinmentWinSize, 1.)

    def test_atomic_append_resume_checks_image_sha_and_attachment(self):
        sample, frames = self.sample()
        metadata = {'marker_attachment_epoch': 'attachment_1', 'intrinsics_sha256': 'abc',
                    'table_board_frame_id': 'fixed_board'}
        with tempfile.TemporaryDirectory() as directory:
            with capture.locked_session(directory) as path:
                dataset = capture.load_session(path, metadata)
                updated = capture.commit_sample(path, dataset, sample, frames)
                self.assertEqual(len(updated['samples']), 1)
                restored = capture.load_session(path, metadata)
                self.assertEqual(restored, updated)
                self.assertIsNone(restored['samples'][0]['selected_marker_pose_index'])
                for row in restored['samples'][0]['repeated_rgb_observations']:
                    self.assertEqual(row['source'], row['hand_marker_pose_candidates']['source'])
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    capture.load_session(path, dict(metadata, marker_attachment_epoch='moved'))
                (path/updated['samples'][0]['images'][0]['path']).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'SHA'):
                    capture.load_session(path, metadata)

    def test_failed_png_save_leaves_valid_previous_dataset_and_no_sample(self):
        sample, frames = self.sample()
        with tempfile.TemporaryDirectory() as directory:
            with capture.locked_session(directory) as path:
                dataset = capture.load_session(path, {'marker_attachment_epoch': 'fixed'})
                with patch.object(capture.cv2, 'imwrite', return_value=False):
                    with self.assertRaisesRegex(RuntimeError, 'save'):
                        capture.commit_sample(path, dataset, sample, frames)
                self.assertEqual([p.name for p in path.iterdir()], ['.capture.lock'])
                self.assertEqual(capture.load_session(path, {'marker_attachment_epoch': 'fixed'}), dataset)

    def test_dataset_publish_failure_rolls_back_new_sample(self):
        sample, frames = self.sample()
        with tempfile.TemporaryDirectory() as directory:
            with capture.locked_session(directory) as path:
                dataset = capture.load_session(path, {})
                with patch.object(Path, 'replace', side_effect=OSError('disk failure')):
                    with self.assertRaises(OSError):
                        capture.commit_sample(path, dataset, sample, frames)
                self.assertEqual([p.name for p in path.iterdir()], ['.capture.lock'])

    def test_resume_refuses_orphan_data_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with capture.locked_session(directory) as path:
                (path/'sample_000').mkdir()
                with self.assertRaisesRegex(ValueError, 'manual review'):
                    capture.load_session(path, {})

    def test_resume_refuses_leftover_sample_after_previous_manifest_commit(self):
        sample, frames = self.sample()
        with tempfile.TemporaryDirectory() as directory:
            with capture.locked_session(directory) as path:
                capture.commit_sample(path, capture.load_session(path, {}), sample, frames)
                (path/'sample_001').mkdir()
                with self.assertRaisesRegex(ValueError, 'manual review'):
                    capture.load_session(path, {})

    def test_optical_controls_ignore_brightness_and_preserve_zoom_focus(self):
        text = ' brightness 0x01 (int) : value=50\n zoom_absolute 0x02 (int) : value=100\n'
        self.assertEqual(capture.optical_controls(text), {'zoom_absolute': 100})

    def test_helper_timeout_and_kill_cleanup_are_bounded(self):
        class Process:
            stdin = io.StringIO()
            stdout = io.StringIO(json.dumps({'event': 'ready', 'schema': 'passive_nero_pose_bridge_v1',
                                             'tx_attempts': 0, 'actual_tx_count': 0})+'\n')
            terminated = killed = False

            def wait(self, timeout):
                if not self.killed:
                    raise subprocess.TimeoutExpired('fake_helper', timeout)
                return 0

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

        process = Process()
        client = capture.PassivePoseClient(['unused'], process_factory=lambda *args, **kwargs: process)
        client.messages = queue.Queue()  # Simulate the running helper ceasing to answer.
        client.timeout_s = .01
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            client.snapshot()
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertTrue(client.closed)
        client.close()

    def test_helper_normal_close_checks_success_and_transmissions(self):
        ready = {'event': 'ready', 'schema': 'passive_nero_pose_bridge_v1',
                 'tx_attempts': 0, 'actual_tx_count': 0}
        for final in ({'event': 'closed', 'request_id': 'close-0', 'success': True,
                       'tx_attempts': 0, 'actual_tx_count': 0},
                      {'event': 'closed', 'request_id': 'close-0', 'success': False,
                       'tx_attempts': 0, 'actual_tx_count': 0},
                      {'event': 'closed', 'request_id': 'close-0', 'success': True,
                       'tx_attempts': 1, 'actual_tx_count': 0}):
            class Process:
                stdin = io.StringIO()
                stdout = io.StringIO(json.dumps(ready)+'\n'+json.dumps(final)+'\n')

                def wait(self, timeout):
                    return 0

            with self.subTest(final=final):
                process = Process()
                client = capture.PassivePoseClient(['unused'], process_factory=lambda *args, **kwargs: process)
                if final['success'] and final['tx_attempts'] == 0:
                    self.assertEqual(client.close(), final)
                else:
                    with self.assertRaisesRegex(RuntimeError, 'final verification'):
                        client.close()
                self.assertTrue(client.closed)

    def test_capture_timeout_fails_before_requesting_pose_or_frame(self):
        ticks = iter((0., 11.))
        pose = PoseClient()
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            capture.capture_stationary_sample(
                Camera(pose), pose, Board(), Geometry(), camera_epoch='stream_1',
                timeout_s=10, clock=lambda: next(ticks), sleep=lambda seconds: None)
        self.assertEqual(pose.index, 0)

    def test_abort_reaps_helper_even_when_closing_stdin_raises_broken_pipe(self):
        class BrokenInput(io.StringIO):
            def close(self):
                super().close()
                raise BrokenPipeError('helper stopped reading')

        class Process:
            stdin = BrokenInput()
            stdout = io.StringIO(json.dumps({'event': 'ready', 'schema': 'passive_nero_pose_bridge_v1',
                                             'tx_attempts': 0, 'actual_tx_count': 0})+'\n')
            waited = False

            def wait(self, timeout):
                self.waited = True
                return 0

        process = Process()
        client = capture.PassivePoseClient(['unused'], process_factory=lambda *args, **kwargs: process)
        client.close(strict=False)
        self.assertTrue(process.waited)
        self.assertTrue(client.closed)


if __name__ == '__main__':
    unittest.main()
