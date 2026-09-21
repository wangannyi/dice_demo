import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from board_rgb import BoardRgbObserver
from usb_intrinsics import _board_points, solve_dataset


BOARD = {'type': 'charuco', 'dictionary': '4x4_50',
         'squares_x': 4, 'squares_y': 5, 'square_length_m': .022,
         'marker_length_m': .0155, 'legacy_pattern': False}
MATRIX = np.array([[950., 0, 640.], [0, 940., 360.], [0, 0, 1.]])
DISTORTION = np.array([-.12, .05, .002, -.001, -.01])


def dataset(root, frontal=False, distortion=None):
    distortion = DISTORTION if distortion is None else distortion
    points = _board_points(BoardRgbObserver(BOARD))
    center = points.mean(axis=0)
    samples = []
    for i in range(30):
        if frontal:
            rv = np.zeros(3)
            uv, depth = (640., 360.), .4
        else:
            rx = [-.55, .15, .45][i % 3]
            ry = [-.45, .3][(i // 3) % 2]
            rv = np.array([rx, ry, .08*(i % 4)])
            uv = ([240., 640., 1040.][i % 3], [160., 360., 560.][(i // 3) % 3])
            depth = [.33, .45, .62][(i // 9) % 3]
        rotated_center = cv2.Rodrigues(rv)[0] @ center
        desired_center = np.array([(uv[0]-640)*depth/950.,
                                   (uv[1]-360)*depth/940., depth])
        tv = desired_center-rotated_center
        pixels = cv2.projectPoints(points, rv, tv, MATRIX, distortion)[0].reshape(-1, 2)
        assert np.all(pixels >= 0) and np.all(pixels[:, 0] < 1280) and np.all(pixels[:, 1] < 720)
        image = np.full((720, 1280, 3), i, np.uint8)
        path = root / ('sample_%03d.png' % i)
        cv2.imwrite(str(path), image)
        samples.append({'image_path': path.name, 'timestamp_s': i*.3,
                        'sha256_image': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'observation': {'valid': True, 'image_size': [1280, 720],
                                        'charuco_corner_ids': list(range(len(points))),
                                        'charuco_corners_px': pixels.tolist()}})
    cfg = {'schema': 1, 'camera': {'backend': 'v4l2_rgb', 'device': '/dev/video7',
                                 'width': 1280, 'height': 720, 'format': 'MJPG'},
           'board': BOARD, 'samples': samples}
    (root / 'manifest.json').write_text(json.dumps(cfg))
    return cfg


class UsbIntrinsicsTests(unittest.TestCase):
    def test_known_parameters_recovered_with_holdout_and_no_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset(root)
            report = solve_dataset(root, root / 'intrinsics.json')
            self.assertTrue(report['quality_passed'], report['quality_reasons'])
            self.assertEqual(report['training_views'], 24)
            self.assertEqual(report['holdout_views'], 6)
            np.testing.assert_allclose(report['camera_matrix'], MATRIX, atol=.02)
            np.testing.assert_allclose(report['distortion_coefficients'], DISTORTION, atol=.002)
            self.assertLess(report['holdout_pnp_rms_px'], .001)
            self.assertFalse(report['motion_target_valid'])
            self.assertEqual(report['calibration_flags'], 0)
            self.assertEqual(report['fixed_distortion_parameters'], {})
            self.assertEqual(len(report['provenance']['samples']), 30)
            self.assertEqual(report['provenance']['manifest_sha256'],
                             hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                solve_dataset(root, root / 'intrinsics.json')

    def test_fixed_k3_recovers_known_camera_and_reports_constraint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            distortion = DISTORTION.copy()
            distortion[4] = 0.
            dataset(root, distortion=distortion)
            report = solve_dataset(root, root / 'intrinsics.json', fix_k3=True)
            self.assertTrue(report['quality_passed'], report['quality_reasons'])
            np.testing.assert_allclose(report['camera_matrix'], MATRIX, atol=.02)
            np.testing.assert_allclose(report['distortion_coefficients'], distortion, atol=.002)
            self.assertEqual(report['distortion_coefficients'][4], 0.)
            self.assertLess(report['holdout_pnp_rms_px'], .001)
            self.assertEqual(report['calibration_flags'], cv2.CALIB_FIX_K3)
            self.assertEqual(report['calibration_flag_names'], ['CALIB_FIX_K3'])
            self.assertEqual(report['estimated_distortion_parameters'], ['k1', 'k2', 'p1', 'p2'])
            self.assertEqual(report['fixed_distortion_parameters'], {'k3': 0.})

    def test_repeated_frontal_views_rejected_despite_small_fit_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset(root, frontal=True)
            report = solve_dataset(root, root / 'intrinsics.json')
            self.assertFalse(report['quality_passed'])
            self.assertIn('insufficient_image_corner_coverage', report['quality_reasons'])
            self.assertIn('insufficient_board_scale_diversity', report['quality_reasons'])
            self.assertIn('too_few_tilted_views', report['quality_reasons'])

    def test_hash_actual_size_observation_size_and_ids_are_verified(self):
        mutations = ('hash', 'image_size', 'observation_size', 'corner_ids', 'path')
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg = dataset(root)
                sample = cfg['samples'][0]
                if mutation == 'hash':
                    sample['sha256_image'] = '0'*64
                elif mutation == 'image_size':
                    image_path = root / sample['image_path']
                    cv2.imwrite(str(image_path), np.zeros((360, 640, 3), np.uint8))
                    sample['sha256_image'] = hashlib.sha256(image_path.read_bytes()).hexdigest()
                elif mutation == 'observation_size':
                    sample['observation']['image_size'] = [640, 360]
                elif mutation == 'corner_ids':
                    sample['observation']['charuco_corner_ids'][1] = 0
                else:
                    sample['image_path'] = '../outside.png'
                (root / 'manifest.json').write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    solve_dataset(root, root / 'output.json')
                self.assertFalse((root / 'output.json').exists())

    def test_too_few_usable_views_do_not_write_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dataset(root)
            for sample in cfg['samples'][7:]:
                sample['observation']['valid'] = False
            (root / 'manifest.json').write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                solve_dataset(root, root / 'output.json')
            self.assertFalse((root / 'output.json').exists())

    def test_insufficient_count_cannot_pass_with_diverse_clean_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dataset(root)
            cfg['samples'] = cfg['samples'][:18]
            (root / 'manifest.json').write_text(json.dumps(cfg))
            report = solve_dataset(root / 'manifest.json', root / 'output.json')
            self.assertFalse(report['quality_passed'])
            self.assertIn('too_few_views', report['quality_reasons'])


if __name__ == '__main__':
    unittest.main()
