"""Saved-image pairing gates do not create physical calibration or motion."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rgb_hand_tracking import grasp_image_reference as reference


TAG = [[15., 15.], [25., 15.], [25., 25.], [15., 25.]]


class Board:
    minimum = 6
    cfg = {'squares_x': 4, 'squares_y': 5}
    drift = 0.
    single_corner_drift = 0.
    sparse = False

    def __init__(self, config, *, corner_convention):
        if corner_convention != 'opencv_4_6':
            raise AssertionError('Wrong replay convention')

    def observe(self, frame):
        code = int(frame[0, 0, 0])
        drift = self.drift if code >= 20 else 0.
        ids = list(range(6))
        if self.sparse and code % 12 >= 7:
            ids = list(range(6, 12))
        return {'valid': True, 'corner_convention': 'opencv_4_6', 'charuco_corner_ids': ids,
                'charuco_corners_px': [[10.+i+drift+(self.single_corner_drift if code >= 20 and i == 0 else 0.),
                                       5.+i % 2] for i in ids]}


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.a = self.dataset('grasp', 'grasp_pose_record', 1)
        self.b = self.dataset('cup', 'before_contact', 20)
        self.board_patch = patch.object(reference, 'BoardRgbObserver', Board)
        self.tag_patch = patch.object(reference, '_raw_tag_observation', lambda _: np.array(TAG))
        self.board_patch.start()
        self.tag_patch.start()
        Board.drift, Board.single_corner_drift, Board.sparse = 0., 0., False

    def tearDown(self):
        self.tag_patch.stop()
        self.board_patch.stop()
        self.tmp.cleanup()

    def dataset(self, name, purpose, code):
        root = self.root/name
        root.mkdir()
        images, rows = [], []
        for i in range(12):
            frame = np.full((40, 50, 3), code+i, dtype=np.uint8)
            png = root/f'frame_{i}.png'
            self.assertTrue(cv2.imwrite(str(png), frame))
            digest = hashlib.sha256(png.read_bytes()).hexdigest()
            image = {'path': png.name, 'sha256': digest, 'width': 50, 'height': 40}
            source = {'frame_serial': i+1, 'timestamp_s': float(i+1), 'camera_epoch': 'camera:stream:'+name,
                      'image_path': png.name, 'image_sha256': digest}
            images.append(image)
            rows.append({'source': source, 'tracking_generation': 0, 'raw_marker_corners_px': deepcopy(TAG),
                         'hand_marker_pose_candidates': {'source': deepcopy(source), 'raw_corners_px': deepcopy(TAG)},
                         'cup_top': {'valid': True, 'source': 'green_top_geometry', 'center_px': [30., 30.]},
                         'cup_metric': {'valid': True, 'source': 'undistorted_edge_rays_height_plane_circle_fit',
                                        'center_board_m': [.1, .1, -.0635], 'radius_m': .025}})
        data = {'schema': reference.SCHEMA, 'kind': reference.SCHEMA, 'purpose': purpose,
                'samples': [{'images': images, 'repeated_rgb_observations': rows, 'stationary': True,
                             'physical_branch_verified': False, 'selected_marker_pose_index': None,
                             'T_marker_contact': None, 'motion_target_valid': False,
                             'marker_corner_statistics': {'median_px': deepcopy(TAG)}}],
                'intrinsics_sha256': 'a'*64, 'marker_attachment_epoch': 'attached',
                'camera_configuration_epoch': 'camera', 'execution_enabled': False,
                'motion_target_valid': False, 'physical_palm_transform_valid': False,
                'reference_activation_valid': False, 'pose_record_valid': True,
                'operator_pregrasp_pose_confirmed': True, 'reference_marker_corners_px': deepcopy(TAG),
                'reference_q_rad': [0.]*7, 'source_sha256': {'original.py': 'b'*64}}
        path = root/'dataset.json'
        path.write_text(json.dumps(data))
        return path

    def mutate(self, path, change):
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))

    def pair(self):
        return reference.pair_image_reference(self.a, self.b, {})

    def test_valid_pair_preserves_originals_and_all_inactive_flags(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = self.pair()
        self.assertTrue(result['image_pair_valid'])
        self.assertTrue(result['fixed_view_valid'])
        self.assertEqual(result['goal']['relative_corners_8D_px'], [-15., -15., -5., -15., -5., -5., -15., -5.])
        self.assertEqual(result['goal']['detector_contract']['cornerRefinementWinSize'], 2)
        self.assertFalse(result['goal']['detector_contract']['Win5_comparison_allowed'])
        self.assertEqual(result['sources'][0]['original_capture_source_SHA256'], {'original.py': 'b'*64})
        for key in reference.FALSE_FLAGS:
            self.assertIs(result[key], False)
        self.assertIsNone(result['T_marker_contact'])
        self.assertIsNone(result['selected_marker_pose_index'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_identity_mismatch_rejects_each_identity(self):
        original = self.b.read_bytes()
        for key in ('intrinsics_sha256', 'marker_attachment_epoch', 'camera_configuration_epoch'):
            self.b.write_bytes(original)
            self.mutate(self.b, lambda d: d.update({key: 'different'}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity mismatch'):
                self.pair()

    def test_modified_png_is_rejected_before_replay(self):
        png = self.b.parent/'frame_0.png'
        png.write_bytes(png.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.pair()

    def test_board_movement_median_or_maximum_rejects(self):
        Board.drift = .8
        with self.assertRaisesRegex(ValueError, 'view moved'):
            self.pair()
        Board.drift, Board.single_corner_drift = 0., 1.6
        with self.assertRaisesRegex(ValueError, 'view moved'):
            self.pair()

    def test_each_common_board_id_needs_eight_frames(self):
        Board.sparse = True
        with self.assertRaisesRegex(ValueError, 'repeated common board'):
            self.pair()

    def test_cup_center_and_radius_instability_reject(self):
        original = self.b.read_bytes()
        for field, value in (('center_board_m', [.103, .1, -.0635]), ('radius_m', .028)):
            self.b.write_bytes(original)
            self.mutate(self.b, lambda d: d['samples'][0]['repeated_rgb_observations'][-1]['cup_metric'].update({field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'cup center or radius is unstable'):
                self.pair()

    def test_fewer_than_eight_cup_top_frames_rejects_not_body_diagnostics(self):
        self.mutate(self.b, lambda d: [r['cup_metric'].update(valid=False)
                                    for r in d['samples'][0]['repeated_rgb_observations'][:5]])
        with self.assertRaisesRegex(ValueError, 'eight fresh'):
            self.pair()

    def test_source_identity_and_ippe_corner_disagreement_reject(self):
        original = self.a.read_bytes()
        for field in ('source', 'corners'):
            self.a.write_bytes(original)
            def change(d):
                row = d['samples'][0]['repeated_rgb_observations'][0]
                if field == 'source':
                    row['hand_marker_pose_candidates']['source']['frame_serial'] = 333
                else:
                    row['hand_marker_pose_candidates']['raw_corners_px'][0][0] += 1.
            self.mutate(self.a, change)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'source identity|IPPE source corners'):
                self.pair()

    def test_wrong_window_corner_bias_rejects_without_smoothing(self):
        with patch.object(reference, '_raw_tag_observation', lambda _: np.asarray(TAG)+1.):
            with self.assertRaisesRegex(ValueError, 'Win2 replay disagrees'):
                self.pair()

    def test_actual_detector_pins_win2_and_rejects_duplicate_id40(self):
        self.tag_patch.stop()
        try:
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            generate = getattr(cv2.aruco, 'generateImageMarker', None) or cv2.aruco.drawMarker
            marker = cv2.cvtColor(generate(dictionary, 40, 90), cv2.COLOR_GRAY2BGR)
            frame = np.full((180, 280, 3), 255, dtype=np.uint8)
            frame[40:130, 30:120] = marker
            if hasattr(cv2.aruco, 'ArucoDetector'):
                with patch.object(cv2.aruco, 'ArucoDetector', wraps=cv2.aruco.ArucoDetector) as spy:
                    corners = reference._raw_tag_observation(frame)
                    parameters = spy.call_args.args[1]
            else:
                with patch.object(cv2.aruco, 'detectMarkers', wraps=cv2.aruco.detectMarkers) as spy:
                    corners = reference._raw_tag_observation(frame)
                    parameters = spy.call_args.kwargs['parameters']
            self.assertEqual(corners.shape, (4, 2))
            self.assertEqual(parameters.cornerRefinementWinSize, 2)
            self.assertEqual(parameters.cornerRefinementMethod, cv2.aruco.CORNER_REFINE_SUBPIX)
            frame[40:130, 150:240] = marker
            with self.assertRaisesRegex(ValueError, 'unique ID40'):
                reference._raw_tag_observation(frame)
        finally:
            self.tag_patch.start()

    def test_duplicate_image_and_dimensions_reject(self):
        original = self.a.read_bytes()
        self.mutate(self.a, lambda d: d['samples'][0]['images'][0].update(width=99))
        with self.assertRaisesRegex(ValueError, 'dimensions mismatch'):
            self.pair()
        self.a.write_bytes(original)
        self.mutate(self.a, lambda d: d['samples'][0]['images'].__setitem__(1, deepcopy(d['samples'][0]['images'][0])))
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.pair()

    def test_activation_claims_are_rejected(self):
        self.mutate(self.a, lambda d: d.update(motion_target_valid=True))
        with self.assertRaisesRegex(ValueError, 'inactive'):
            self.pair()

    def test_input_physical_registration_and_selected_branch_reject(self):
        original = self.a.read_bytes()
        for key, value in (('physical_registration_valid', True), ('selected_marker_pose_index', 0)):
            self.a.write_bytes(original)
            self.mutate(self.a, lambda d: d.update({key: value}))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Image-only'):
                self.pair()

    def test_cli_creates_new_json_but_refuses_existing_sources(self):
        config = self.root/'board.json'
        config.write_text('{}')
        output = self.root/'pair.json'
        argv = ['pair', '--grasp-dataset', str(self.a), '--cup-dataset', str(self.b),
                '--board-config', str(config), '--output', str(output)]
        with patch.object(sys, 'argv', argv), patch('builtins.print'):
            reference.main()
        saved = json.loads(output.read_text())
        self.assertTrue(saved['fixed_view_valid'])
        self.assertFalse(saved['motion_target_valid'])
        self.assertEqual(saved['board_configuration_source']['SHA256'], hashlib.sha256(config.read_bytes()).hexdigest())
        original = self.a.read_bytes()
        argv[-1] = str(self.a)
        with patch.object(sys, 'argv', argv), patch.object(sys, 'stderr'), self.assertRaises(SystemExit):
            reference.main()
        self.assertEqual(self.a.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
