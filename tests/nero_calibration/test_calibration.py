import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch, Mock
import numpy as np
import cv2
from nero_calibration.core import pose_matrix, matrix_pose, inverse, matrix, tcp_transform, PALM, HAND_BASE, solve
from nero_calibration.sensors import CharucoDetector, assert_still, opencv_distortion, NeroFeedback
from nero_calibration.calibrate import read_dataset, read_resume_dataset, write_new, main, capture_sample

ROOT = Path(__file__).resolve().parents[2] / 'nero_calibration'


def synthetic():
    rng = np.random.default_rng(123)
    bc = pose_matrix([.42, -.18, .7, .2, -.3, .4])
    fm = pose_matrix([.02, .03, .11, -.4, .2, .1])
    samples = []
    for _ in range(24):
        bf = pose_matrix([*rng.uniform(-.3, .3, 3), *rng.uniform(-.8, .8, 3)])
        cm = inverse(bc) @ bf @ fm
        samples.append({'T_base_flange': bf.tolist(), 'T_camera_board': cm.tolist()})
    return bc, fm, samples


class CalibrationTests(unittest.TestCase):
    def make_collection_session(self, path, poses=None):
        board = json.loads((ROOT/'config/board_perceptive.json').read_text())
        camera = {'backend': 'realsense', 'serial': 'test-camera', 'frame': 'color_optical',
                  'width': 640, 'height': 480,
                  'camera_matrix': [[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]],
                  'dist_coeffs': [0.] * 5, 'distortion_model': 'distortion.inverse_brown_conrady'}
        write_new(path/'manifest.json', {'schema': 1, 'mode': 'eye_to_hand',
                  'camera': camera, 'board': board, 'tcp': 'palm',
                  'T_flange_tcp': PALM.tolist()})
        for i, flange in enumerate(poses or [np.eye(4)]):
            stem = f'sample_{i:04d}'
            write_new(path/(stem+'.json'), {'T_base_flange': flange.tolist(),
                      'T_camera_board': pose_matrix([0, 0, .5, 0, 0, 0]).tolist(),
                      'T_base_tcp': (flange @ PALM).tolist(),
                      'quality': {'corners': 12, 'reprojection_rms_px': .1}})
            (path/(stem+'.png')).write_bytes(b'old color')
            (path/(stem+'_detected.png')).write_bytes(b'old annotation')
        return board, camera

    def test_default_is_flange(self):
        np.testing.assert_array_equal(tcp_transform(), np.eye(4))

    def test_custom_parent_composition(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'tcp.json'
            pose = [.01, -.02, .03, .2, -.3, .4]
            for parent, t in [('flange', np.eye(4)), ('hand_base', HAND_BASE), ('palm', PALM)]:
                p.write_text(json.dumps({'units': 'm_rad', 'relative_to': parent, 'pose': pose}))
                np.testing.assert_allclose(tcp_transform('custom', p), t @ pose_matrix(pose))
            p.write_text(json.dumps({'units': 'mm', 'relative_to': 'palm', 'pose': pose}))
            with self.assertRaises(ValueError):
                tcp_transform('custom', p)

    def test_matrix_tcp_matches_palm_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'tcp.json'
            for parent, base in [('flange', np.eye(4)), ('hand_base', HAND_BASE), ('palm', PALM)]:
                offset = pose_matrix([.012, -.006, .028, .2, -.1, .3])
                p.write_text(json.dumps({'units': 'm', 'relative_to': parent,
                                         'matrix': offset.tolist()}))
                np.testing.assert_allclose(tcp_transform('custom', p), base @ offset)
            p.write_text(json.dumps({'units': 'm', 'relative_to': 'flange',
                                     'matrix': PALM.tolist()}))
            np.testing.assert_allclose(tcp_transform('custom', p), tcp_transform('palm'))
            p.write_text(json.dumps({'units': 'm', 'relative_to': 'flange',
                                     'matrix': np.eye(4).tolist()}))
            np.testing.assert_allclose(tcp_transform('custom', p), tcp_transform('flange'))

    def test_invalid_matrix_tcp_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'tcp.json'
            def check(config):
                p.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    tcp_transform('custom', p)
            baseline = {'units': 'm', 'relative_to': 'flange',
                        'matrix': np.eye(4).tolist()}
            check({**baseline, 'pose': [0]*6})
            check({'units': 'm', 'relative_to': 'flange'})
            check({**baseline, 'units': 'mm'})
            check({**baseline, 'matrix': np.diag([-1, 1, 1, 1]).tolist()})
            check({**baseline, 'matrix': [[1, 0, 0, 0]]*4})

    def test_tcp_target_roundtrip(self):
        target = pose_matrix([.3, .2, .5, .4, -.3, .8])
        for tcp in [np.eye(4), PALM, pose_matrix([.04, .07, -.02, .2, .4, -.9])]:
            np.testing.assert_allclose((target @ inverse(tcp)) @ tcp, target, atol=1e-12)

    def test_pose_roundtrip_and_gimbal_lock(self):
        for pitch in [-np.pi/2, -.2, np.pi/2]:
            t = pose_matrix([.1, .2, -.3, .4, pitch, -.7])
            np.testing.assert_allclose(pose_matrix(matrix_pose(t)), t, atol=1e-8)

    def test_invalid_transforms(self):
        for bad in [np.zeros((4, 4)), np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan)]:
            with self.assertRaises(ValueError):
                matrix(bad)
        with self.assertRaises(ValueError):
            pose_matrix([0, 0, 0, 0, 0, float('nan')])

    def test_known_eye_to_hand_and_tcp_invariance(self):
        bc, fm, samples = synthetic()
        for tcp in [np.eye(4), PALM, pose_matrix([.1, .02, .03, .2, -.4, .3])]:
            result = solve(samples, tcp)
            self.assertTrue(result['quality_passed'])
            np.testing.assert_allclose(result['T_base_camera'], bc, atol=1e-8)
            np.testing.assert_allclose(result['T_tcp_board'], inverse(tcp) @ fm, atol=1e-8)

    def test_outlier_fails_quality(self):
        _, _, samples = synthetic()
        samples[3]['T_camera_board'][0][3] += .1
        self.assertFalse(solve(samples, np.eye(4))['quality_passed'])

    def test_degenerate_rotation_rejected(self):
        _, _, samples = synthetic()
        for i, s in enumerate(samples):
            s['T_base_flange'] = pose_matrix([i*.01, 0, .3, 0, 0, i*.03]).tolist()
        with self.assertRaisesRegex(ValueError, 'diversity'):
            solve(samples, np.eye(4))
        with self.assertRaises(ValueError):
            solve(samples[:5], np.eye(4))

    def test_roi_isolates_duplicate_boards_without_changing_coordinates(self):
        cfg = json.loads((ROOT/'config/board_4x5_21p5mm.json').read_text())
        full = CharucoDetector(cfg)
        board = full.board
        generated = (board.generateImage((200, 250)) if hasattr(board, 'generateImage')
                     else board.draw((200, 250)))
        single = np.full((480, 640), 255, np.uint8)
        single[100:350, 60:260] = generated
        two = single.copy()
        two[100:350, 400:600] = generated
        single = cv2.cvtColor(single, cv2.COLOR_GRAY2BGR)
        two = cv2.cvtColor(two, cv2.COLOR_GRAY2BGR)
        before = two.copy()
        K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        expected, _, _ = full.detect(single, K, np.zeros(5))
        with self.assertRaisesRegex(ValueError, 'Duplicate marker IDs'):
            full.detect(two, K, np.zeros(5))
        roi = CharucoDetector({**cfg, 'image_roi_xyxy': [20, 40, 320, 440]})
        actual, quality, _ = roi.detect(two, K, np.zeros(5))
        np.testing.assert_allclose(actual, expected, atol=1e-9)
        np.testing.assert_array_equal(two, before)
        self.assertEqual(quality['corners'], 12)
        self.assertEqual(quality['image_roi_xyxy'], [20, 40, 320, 440])
        # Removing the hand target must not select the table target outside the ROI.
        two[:, :320] = 255
        with self.assertRaisesRegex(ValueError, 'No ChArUco markers'):
            roi.detect(two, K, np.zeros(5))

    def test_exclusion_keeps_full_frame_coordinates_and_rejects_fixed_board_alone(self):
        cfg = json.loads((ROOT/'config/board_4x5_21p5mm.json').read_text())
        full = CharucoDetector(cfg)
        board = full.board
        pattern = (board.generateImage((200,250)) if hasattr(board,'generateImage') else board.draw((200,250)))
        image = np.full((720,960,3),255,np.uint8)
        image[450:700,50:250] = cv2.cvtColor(pattern,cv2.COLOR_GRAY2BGR)
        K=np.array([[900.,0.,480.],[0.,900.,360.],[0.,0.,1.]])
        expected,_,_=full.detect(image,K,np.zeros(5))
        image[450:700,600:800] = cv2.cvtColor(pattern,cv2.COLOR_GRAY2BGR)
        before=image.copy()
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            full.detect(image,K,np.zeros(5))
        detector=CharucoDetector(dict(cfg,image_exclude_rois_xyxy=[[580,430,820,720]]))
        actual,q,_=detector.detect(image,K,np.zeros(5))
        np.testing.assert_allclose(actual,expected,atol=1e-9)
        np.testing.assert_array_equal(image,before)
        self.assertEqual(q['corners'],12)
        from tools.reprocess_dimensions import measured_observation
        measured, _, _, _ = measured_observation(image, detector, K, np.zeros(5), .0865, .108)
        single = image.copy()
        single[:, 580:] = 255
        reference, _, _, _ = measured_observation(single, full, K, np.zeros(5), .0865, .108)
        np.testing.assert_allclose(measured, reference, atol=1e-9)
        image[:,:300]=255
        with self.assertRaisesRegex(ValueError,'No ChArUco markers'):
            detector.detect(image,K,np.zeros(5))

    def test_roi_rejects_invalid_and_out_of_bounds_regions(self):
        cfg = json.loads((ROOT/'config/board_4x5_21p5mm.json').read_text())
        for bad in [[0, 0, 0, 100], [-1, 0, 20, 20], [0, 0, 1.5, 10],
                    [False, 0, 10, 10], [0, 0, 10], '0,0,10,10']:
            with self.subTest(roi=bad), self.assertRaisesRegex(ValueError, 'image_roi_xyxy'):
                CharucoDetector({**cfg, 'image_roi_xyxy': bad})
        detector = CharucoDetector({**cfg, 'image_roi_xyxy': [0, 0, 641, 480]})
        with self.assertRaisesRegex(ValueError, 'exceeds image bounds'):
            detector.detect(np.full((480, 640, 3), 255, np.uint8), np.eye(3), np.zeros(5))

    def test_real_board_image_detection(self):
        cfg = json.loads((ROOT/'config/board_perceptive.json').read_text())
        detector = CharucoDetector(cfg)
        board = detector.board
        # Four squares span 400 pixels; depth follows fx * physical square size / square pixels.
        generated = board.generateImage((400, 500)) if hasattr(board, 'generateImage') else board.draw((400, 500))
        img = np.full((700, 600), 255, np.uint8)
        img[100:600, 100:500] = generated
        K = np.array([[1000., 0, 300], [0, 1000., 350], [0, 0, 1]])
        T, quality, _ = detector.detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), K, np.zeros(5))
        self.assertGreaterEqual(quality['corners'], 6)
        square_pixels = 400 / cfg['squares_x']
        expected_depth = K[0, 0] * cfg['square_length_m'] / square_pixels
        self.assertAlmostEqual(T[2, 3], expected_depth, delta=.002)
        with self.assertRaises(ValueError):
            detector.detect(np.full((700, 600, 3), 255, np.uint8), K, np.zeros(5))

    def test_distortion_compatibility(self):
        np.testing.assert_array_equal(opencv_distortion('distortion.inverse_brown_conrady', [0]*5), [0]*5)
        with self.assertRaises(ValueError):
            opencv_distortion('distortion.inverse_brown_conrady', [.1, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            opencv_distortion('distortion.modified_brown_conrady', [.1, 0, 0, 0, 0])

    def test_stationarity(self):
        t = np.eye(4)
        assert_still([t, t])
        with self.assertRaises(ValueError):
            assert_still([t, pose_matrix([.005, 0, 0, 0, 0, 0])])

    def test_stale_feedback_is_rejected(self):
        reader = NeroFeedback.__new__(NeroFeedback)
        reader.last_timestamp = 100.
        reader.robot = Mock()
        reader.robot.has_comm_error.return_value = False
        reader.robot.get_flange_pose.return_value = Mock(timestamp=100., msg=[0]*6)
        with patch('nero_calibration.sensors.time.monotonic', side_effect=[0., .1, .6]), patch('nero_calibration.sensors.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'No new'):
                reader.read()

    def test_capture_brackets_pose_and_rejects_motion(self):
        arm, camera, detector = Mock(), Mock(), Mock()
        arm.read.return_value = np.eye(4)
        arm.read_joints.return_value = dict(joints_rad=[0.]*7,joints_deg=[0.]*7)
        camera.K, camera.D = np.eye(3), np.zeros(5)
        camera.capture.return_value = np.zeros((10, 10, 3), np.uint8)
        detector.detect.return_value = (pose_matrix([0, 0, .5, 0, 0, 0]),
                                        {'reprojection_rms_px': .1}, camera.capture.return_value)
        with patch('nero_calibration.calibrate.time.sleep'):
            sample, _, _ = capture_sample(arm, camera, detector)
            np.testing.assert_allclose(sample['T_base_flange'], np.eye(4))
            arm.read.side_effect = [np.eye(4)]*8 + [pose_matrix([.01, 0, 0, 0, 0, 0])]*6
            with self.assertRaisesRegex(ValueError, 'moved'):
                capture_sample(arm, camera, detector)
        self.assertEqual([c[0] for c in arm.method_calls if c[0] not in ('read', 'read_joints')], [])

    def test_dataset_overwrite_protected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'result.json'
            write_new(p, {'a': 1})
            with self.assertRaises(FileExistsError):
                write_new(p, {'a': 2})
            self.assertEqual(json.loads(p.read_text()), {'a': 1})

    def test_collect_existing_directory_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'session'
            path.mkdir()
            self.make_collection_session(path)
            original = (path/'manifest.json').read_bytes()
            with patch('nero_calibration.sensors.RealSenseCamera') as camera:
                with self.assertRaises(FileExistsError):
                    main(['collect', '--tcp', 'palm', '--dataset', str(path)])
                camera.assert_not_called()
            self.assertEqual((path/'manifest.json').read_bytes(), original)

    def test_invalidated_session_cannot_resume_or_solve(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.make_collection_session(path)
            write_new(path/'INVALIDATED.json', {'reason': 'Camera moved'})
            original = {p.name: p.read_bytes() for p in path.iterdir()}
            with patch('nero_calibration.sensors.RealSenseCamera') as camera:
                with self.assertRaisesRegex(ValueError, 'invalidated'):
                    main(['collect', '--resume', '--tcp', 'palm', '--dataset', str(path)])
                camera.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'invalidated'):
                main(['solve', '--dataset', str(path), '--output', str(path/'result.json')])
            self.assertEqual({p.name: p.read_bytes() for p in path.iterdir()}, original)

    def test_collect_resume_appends_next_index_and_filters_old_pose(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'session'
            path.mkdir()
            poses = [pose_matrix([i * .02, 0, 0, 0, 0, 0]) for i in range(10)]
            _, camera_info = self.make_collection_session(path, poses)
            previous = {p.name: p.read_bytes() for p in path.iterdir()}
            cam, arm = Mock(), Mock()
            cam.info = camera_info
            image = np.zeros((10, 10, 3), np.uint8)
            duplicate = {'T_base_flange': poses[-1].tolist(),
                         'T_camera_board': np.eye(4).tolist(),
                         'quality': {'corners': 12, 'reprojection_rms_px': .1}}
            flange = pose_matrix([.25, 0, 0, 0, 0, 0])
            fresh = {**duplicate, 'T_base_flange': flange.tolist()}
            with patch('nero_calibration.sensors.RealSenseCamera', return_value=cam), \
                 patch('nero_calibration.sensors.NeroFeedback', return_value=arm), \
                 patch('nero_calibration.calibrate.capture_sample', side_effect=[(duplicate, image, image),
                                                               (fresh, image, image)]), \
                 patch('builtins.input', side_effect=['', '', 'q']), \
                 patch('builtins.print'):
                self.assertEqual(main(['collect', '--tcp', 'palm', '--dataset', str(path),
                                       '--resume']), 0)
            for name, contents in previous.items():
                self.assertEqual((path/name).read_bytes(), contents)
            self.assertEqual(len(list(path.glob('sample_*.json'))), 11)
            self.assertTrue((path/'sample_0010.png').is_file())
            self.assertTrue((path/'sample_0010_detected.png').is_file())
            np.testing.assert_allclose(json.loads((path/'sample_0010.json').read_text())['T_base_tcp'],
                                       flange @ PALM)
            cam.close.assert_called_once()
            arm.close.assert_called_once()

    def test_collect_interrupt_closes_devices_despite_second_sigint(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'session'
            path.mkdir()
            _, camera_info = self.make_collection_session(path)
            original = {p.name: p.read_bytes() for p in path.iterdir()}
            cam, arm = Mock(), Mock()
            cam.info = camera_info
            previous_handler = signal.getsignal(signal.SIGINT)

            def stop_camera():
                self.assertEqual(signal.getsignal(signal.SIGINT), signal.SIG_IGN)
                signal.raise_signal(signal.SIGINT)

            cam.close.side_effect = stop_camera
            with patch('nero_calibration.sensors.RealSenseCamera', return_value=cam), \
                 patch('nero_calibration.sensors.NeroFeedback', return_value=arm), \
                 patch('builtins.input', side_effect=KeyboardInterrupt), \
                 patch('builtins.print'):
                self.assertEqual(main(['collect', '--tcp', 'palm', '--dataset', str(path),
                                       '--resume']), 130)
            self.assertEqual(signal.getsignal(signal.SIGINT), previous_handler)
            arm.close.assert_called_once()
            cam.close.assert_called_once()
            self.assertEqual({p.name: p.read_bytes() for p in path.iterdir()}, original)

    def test_collect_eof_exits_normally_and_closes_devices(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'session'
            path.mkdir()
            _, camera_info = self.make_collection_session(path)
            cam, arm = Mock(), Mock()
            cam.info = camera_info
            with patch('nero_calibration.sensors.RealSenseCamera', return_value=cam), \
                 patch('nero_calibration.sensors.NeroFeedback', return_value=arm), \
                 patch('builtins.input', side_effect=EOFError), \
                 patch('builtins.print'):
                self.assertEqual(main(['collect', '--tcp', 'palm', '--dataset', str(path),
                                       '--resume']), 0)
            arm.close.assert_called_once()
            cam.close.assert_called_once()

    def test_resume_rejects_manifest_and_sample_mismatches_before_hardware(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'session'
            path.mkdir()
            board, _ = self.make_collection_session(path)
            changed_board = {**board, 'square_length_m': .03}
            with self.assertRaisesRegex(ValueError, 'board'):
                read_resume_dataset(path, changed_board, 'palm', PALM)
            with self.assertRaisesRegex(ValueError, 'TCP'):
                read_resume_dataset(path, board, 'flange', np.eye(4))
            (path/'sample_0000_detected.png').unlink()
            with self.assertRaisesRegex(ValueError, 'missing images'):
                read_resume_dataset(path, board, 'palm', PALM)
            (path/'sample_0000_detected.png').write_bytes(b'old annotation')
            (path/'sample_0002.json').write_bytes((path/'sample_0000.json').read_bytes())
            with patch('nero_calibration.sensors.RealSenseCamera') as camera:
                with self.assertRaisesRegex(ValueError, 'numbering'):
                    main(['collect', '--tcp', 'palm', '--dataset', str(path), '--resume'])
                camera.assert_not_called()

    def test_resume_rejects_camera_mismatch_before_arm(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'session'
            path.mkdir()
            _, camera_info = self.make_collection_session(path)
            with patch('nero_calibration.sensors.RealSenseCamera') as camera:
                with self.assertRaisesRegex(ValueError, 'serial'):
                    main(['collect', '--tcp', 'palm', '--dataset', str(path), '--resume',
                          '--serial', 'another-camera'])
                camera.assert_not_called()
            cam = Mock()
            cam.info = {**camera_info, 'camera_matrix': [[601., 0., 320.],
                                                         [0., 600., 240.], [0., 0., 1.]]}
            with patch('nero_calibration.sensors.RealSenseCamera', return_value=cam), \
                 patch('nero_calibration.sensors.NeroFeedback') as arm:
                with self.assertRaisesRegex(ValueError, 'identity/intrinsics'):
                    main(['collect', '--tcp', 'palm', '--dataset', str(path), '--resume'])
                arm.assert_not_called()
            cam.close.assert_called_once()

    def test_dataset_tcp_matches_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            write_new(p/'manifest.json', {'schema': 1, 'mode': 'eye_to_hand',
                      'T_flange_tcp': PALM.tolist()})
            flange = pose_matrix([.2, .3, .4, -.2, .1, .8])
            sample = {'T_base_flange': flange.tolist(),
                      'T_camera_board': np.eye(4).tolist(),
                      'T_base_tcp': (flange @ PALM).tolist()}
            write_new(p/'sample_0000.json', sample)
            _, loaded = read_dataset(p)
            self.assertEqual(len(loaded), 1)
            sample['T_base_tcp'][0][3] += .01
            (p/'sample_0000.json').write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                read_dataset(p)

    def test_cli_solve_saved_dataset(self):
        _, _, samples = synthetic()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            write_new(p/'manifest.json', {'schema': 1, 'mode': 'eye_to_hand',
                      'T_flange_tcp': PALM.tolist(), 'tcp': 'palm', 'camera': {'frame': 'color_optical'}, 'board': {}})
            for i, s in enumerate(samples):
                write_new(p/f'sample_{i:04d}.json', s)
            with patch('builtins.print'):
                self.assertEqual(main(['solve', '--dataset', d, '--output', str(p/'result.json')]), 0)
            self.assertTrue(json.loads((p/'result.json').read_text())['quality_passed'])


if __name__ == '__main__':
    unittest.main()
