import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from rgb_hand_tracking.scene_observation import SceneObserver, check_marker_publication, load_metric_geometry


class Hand:
    def __init__(self):
        self.present = True
        self.wrong_id = False
        self.resets = []

    def infer(self, path, frame_id, reset):
        self.resets.append(reset)
        return {'frame_id': frame_id+int(self.wrong_id), 'width': 160, 'height': 120,
                'processing_ms': 10, 'detector_calls': 0, 'landmark_calls': 1,
                'hands': [{'landmarks_px': [[30, 40]]*21, 'presence': .98,
                           'from_tracking': True}] if self.present else []}


class Cup:
    def __init__(self):
        self.valid = True
        self.frame = None

    def process(self, frame):
        self.frame = frame
        return {'valid': self.valid, 'center_px': [100, 50] if self.valid else None,
                'reason': None if self.valid else 'cup_not_detected'}


class Board:
    def __init__(self):
        self.valid = True
        self.frame = None

    def observe(self, frame):
        self.frame = frame
        return {'valid': self.valid, 'charuco_corners_px': []}


class Geometry:
    def __init__(self):
        self.calls = []

    def observe(self, board, cup):
        self.calls.append((board, cup))
        valid = board['valid'] and cup['valid']
        return {'geometry_valid': valid, 'board_pose': {'valid': board['valid']},
                'cup_top': {'valid': valid, 'center_board_m': [.1, .2, .0635] if valid else None},
                'independent_metric_accuracy_validated': False, 'motion_target_valid': False}


class MarkerHand:
    def __init__(self):
        self.valid = True
        self.wrong_epoch = False
        self.calls = []

    def observe(self, frame, path, timestamp, epoch, **context):
        self.calls.append((frame, path, timestamp, epoch, context))
        return {'valid': self.valid, 'reason': 'visual_template_confirmed',
                'source_timestamp_s': timestamp, 'camera_epoch': 'wrong' if self.wrong_epoch else epoch,
                'marker': context['marker_observation'],
                'landmarks_px': [[30, 40]]*21 if self.valid else None,
                'stable_visual_palm_px': [35, 45] if self.valid else None,
                'raw_visual_palm_px': [30, 40], 'processing_ms': 10,
                'detector_calls': 0, 'landmark_calls': 1}


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.frame = np.zeros((120, 160, 3), np.uint8)
        self.path = Path(self.temp.name)/'frame.png'
        cv2.imwrite(str(self.path), self.frame)
        self.hand = Hand()
        self.cup, self.board = Cup(), Board()
        self.observer = SceneObserver(self.hand, self.cup, self.board)

    def observe(self, timestamp, epoch='initial'):
        return self.observer.observe(self.frame, self.path, 1, timestamp, epoch)

    def test_same_frame_pixel_delta_never_claims_metric_target(self):
        row = self.observe(1)
        self.assertTrue(row['pixel_observation_valid'])
        self.assertEqual(row['relative_pixels']['cup_center_minus_visual_palm_px'], [70, 10])
        self.assertFalse(row['metric_observation_valid'])
        self.assertFalse(row['motion_target_valid'])
        self.assertIsNone(row['physical_relative_m'])
        self.assertIsNone(row['hand']['physical_palm_m'])
        self.assertNotIn('metric_geometry', row)
        self.assertEqual(set(row['processing_ms']), {'hand', 'cup', 'board', 'total'})

    def test_enabled_geometry_receives_current_same_frame_observations(self):
        geometry = Geometry()
        self.observer = SceneObserver(self.hand, self.cup, self.board, geometry)
        row = self.observe(1)
        self.assertIs(self.cup.frame, self.frame)
        self.assertIs(self.board.frame, self.frame)
        self.assertIs(geometry.calls[0][0], row['board'])
        self.assertIs(geometry.calls[0][1], row['cup_top'])
        self.assertTrue(row['metric_geometry']['geometry_valid'])
        self.assertIn('metric_geometry', row['processing_ms'])

    def test_valid_cup_geometry_cannot_create_physical_palm_or_motion_target(self):
        self.observer = SceneObserver(self.hand, self.cup, self.board, Geometry())
        row = self.observe(1)
        self.assertTrue(row['metric_geometry']['cup_top']['valid'])
        self.assertIsNone(row['hand']['physical_palm_m'])
        self.assertIsNone(row['physical_relative_m'])
        self.assertFalse(row['metric_observation_valid'])
        self.assertFalse(row['motion_target_valid'])

    def test_missing_current_observations_do_not_reuse_previous_geometry(self):
        geometry = Geometry()
        self.observer = SceneObserver(self.hand, self.cup, self.board, geometry)
        first = self.observe(1)
        self.cup.valid = False
        second = self.observe(2)
        self.assertFalse(second['metric_geometry']['geometry_valid'])
        self.assertIsNone(second['metric_geometry']['cup_top']['center_board_m'])
        self.cup.valid, self.board.valid = True, False
        third = self.observe(3)
        self.assertFalse(third['metric_geometry']['geometry_valid'])
        self.assertFalse(third['metric_geometry']['board_pose']['valid'])
        self.assertIsNone(third['metric_geometry']['cup_top']['center_board_m'])
        self.assertEqual(len(geometry.calls), 3)
        self.assertTrue(first['metric_geometry']['geometry_valid'])

    def test_explicit_intrinsics_requirement_and_config_relative_path(self):
        config_path = Path(self.temp.name)/'config.json'
        for filename in (None, '', 'absent.json'):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                load_metric_geometry({'camera': {'intrinsics_file': filename}}, config_path)
        report_path = Path(self.temp.name)/'intrinsics.json'
        report = {'camera_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
        report_path.write_text(json.dumps(report))
        constructor = Mock()
        module = SimpleNamespace(CalibratedRgbGeometry=constructor)
        with patch.dict(sys.modules, {'rgb_hand_tracking.rgb_geometry': module}):
            for filename in ('intrinsics.json', str(report_path)):
                config = {'camera': {'intrinsics_file': filename}}
                self.assertIs(load_metric_geometry(config, config_path), constructor.return_value)
                constructor.assert_called_with(config, report)

    def test_mismatch_does_not_accept_other_frame_hand(self):
        self.hand.wrong_id = True
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.observe(1)

    def test_different_stored_rgb_not_used_as_same_frame(self):
        self.frame[0, 0] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.observe(1)

    def test_missing_hand_does_not_reuse_previous_proxy(self):
        self.observe(1)
        self.hand.present = False
        row = self.observe(2)
        self.assertFalse(row['pixel_observation_valid'])
        self.assertIsNone(row['relative_pixels'])
        self.assertIsNone(row['hand']['visual_palm_proxy_px'])

    def test_epoch_reset_and_timestamp_order(self):
        self.observe(1)
        self.observe(2)
        self.observe(3, 'moved')
        self.assertEqual(self.hand.resets, [True, False, True])
        with self.assertRaises(ValueError):
            self.observe(3)

    def marker_case(self, timestamp=1, epoch='initial', generation=0, serial=9):
        self.marker_hand = MarkerHand()
        self.observer = SceneObserver(self.marker_hand, self.cup, self.board,
                                      hand_backend='marker_guided')
        self.marker = {'capture_timestamp_s': timestamp, 'timestamp_s': timestamp,
                       'camera_epoch': epoch, 'width': 160, 'height': 120,
                       'frame_serial': serial, 'tracking_generation': generation,
                       'marker_id': 40, 'observation_valid': True}
        return dict(marker_observation=self.marker, tracking_generation=generation,
                    frame_serial=serial)

    def test_marker_same_source_uses_template_proxy_without_metric_target(self):
        context = self.marker_case()
        row = self.observer.observe(self.frame, self.path, 1, 1, **context)
        self.assertIs(self.marker_hand.calls[0][0], self.frame)
        self.assertIs(self.cup.frame, self.frame)
        self.assertIs(self.board.frame, self.frame)
        self.assertTrue(row['pixel_observation_valid'])
        self.assertEqual(row['relative_pixels']['cup_center_minus_visual_palm_px'], [65, 5])
        self.assertEqual(row['hand']['raw_visual_palm_px'], [30, 40])
        self.assertEqual(row['hand']['source'], 'marker_guided_visual_proxy')
        self.assertFalse(row['motion_target_valid'])
        self.assertIsNone(row['hand']['physical_palm_m'])

    def test_marker_other_frame_metadata_rejected_before_inference(self):
        for key, value in [('capture_timestamp_s', 2), ('frame_serial', 10),
                           ('tracking_generation', 1), ('width', 170), ('camera_epoch', 'moved')]:
            with self.subTest(key=key):
                context = self.marker_case()
                context['marker_observation'][key] = value
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    self.observer.observe(self.frame, self.path, 1, 1, **context)
                self.assertEqual(self.marker_hand.calls, [])

    def test_marker_loss_does_not_reuse_visual_proxy(self):
        context = self.marker_case()
        self.observer.observe(self.frame, self.path, 1, 1, **context)
        self.marker_hand.valid = False
        context['marker_observation'].update(timestamp_s=2, capture_timestamp_s=2)
        row = self.observer.observe(self.frame, self.path, 2, 2, **context)
        self.assertFalse(row['pixel_observation_valid'])
        self.assertIsNone(row['relative_pixels'])
        self.assertIsNone(row['hand']['visual_palm_proxy_px'])

    def test_marker_hand_epoch_mismatch_rejected(self):
        context = self.marker_case()
        self.marker_hand.wrong_epoch = True
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.observer.observe(self.frame, self.path, 1, 1, **context)

    def test_hidden_marker_loss_during_inference_invalidates_publication(self):
        context = self.marker_case()
        row = self.observer.observe(self.frame, self.path, 1, 1, **context)
        marker = row['hand']['marker']
        check_marker_publication(row, {'tracking_generation': 1, 'latest_frame_serial': 200})
        self.assertFalse(row['hand']['valid'])
        self.assertFalse(row['pixel_observation_valid'])
        self.assertIsNone(row['relative_pixels'])
        self.assertIsNone(row['hand']['visual_palm_proxy_px'])
        self.assertIs(row['hand']['marker'], marker)
        self.assertEqual(marker['capture_timestamp_s'], 1)
        self.assertFalse(row['marker_publication']['continuity_valid'])

    def test_reader_failure_during_inference_invalidates_publication(self):
        context = self.marker_case()
        row = self.observer.observe(self.frame, self.path, 1, 1, **context)
        check_marker_publication(row, {'tracking_generation': 0, 'latest_frame_serial': 9,
                                      'failure_reason': 'RGB read failed'})
        self.assertFalse(row['hand']['valid'])
        self.assertIsNone(row['relative_pixels'])
        self.assertEqual(row['marker_publication']['frontend_failure_reason'], 'RGB read failed')


if __name__ == '__main__':
    unittest.main()
