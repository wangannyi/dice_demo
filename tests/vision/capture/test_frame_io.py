"""Contact geometry regressions independent of cameras, CAN, and learned models."""
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from vision.capture.frame_io import load_batch, ray_surface_x, section
from vision.geometry.table_plane import Config


class IndexPreviewTests(unittest.TestCase):
    def test_batch_discards_saturated_depth_before_temporal_median(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / 'rgbd').mkdir()
            meta = dict(serial='test', intrinsics={}, depth_scale_m=.001,
                        depth_registered_to='color_optical')
            for i, depth in enumerate(([[0, 65535], [700, 700]],
                                       [[65535, 0], [65535, 702]],
                                       [[65535, 65535], [0, 704]])):
                path = run / 'rgbd' / f'frame_{i:03d}'
                path.with_suffix('.json').write_text(json.dumps(meta))
                np.savez(path.with_suffix('.npz'), depth=np.array(depth, np.uint16),
                         color=np.zeros((2, 2, 3), np.uint8))
            _, depth, _, _ = load_batch(run)
            np.testing.assert_array_equal(depth, [[0, 0], [700, 702]])
            for i in (1, 2):
                (run / 'rgbd' / f'frame_{i:03d}.json').unlink()
            _, depth, _, _ = load_batch(run, min_frames=1)
            np.testing.assert_array_equal(depth, [[0, 0], [700, 700]])

    def test_section_uses_tilted_table_height_and_local_radius(self):
        # A tapered cylinder: using the old 0.6H radius must yield a different target.
        theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        heights = np.repeat(np.linspace(0, .12, 121), len(theta))
        angle = np.tile(theta, 121)
        radius = .025 + .10 * heights
        points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), heights))
        rotation = np.array([[1, 0, 0], [0, .8, -.6], [0, .6, .8]])
        origin = np.array([.1, -.2, .8])
        points = points @ rotation.T + origin
        normal = rotation[:, 2]
        current = section(points, origin, normal, 9 / 11, Config())
        old = section(points, origin, normal, .6, Config())
        center = np.array(current['center_camera_m'])
        self.assertAlmostEqual(float((center - origin) @ normal), current['height_m'] * 9 / 11)
        self.assertLess(np.linalg.norm(center - origin - current['contact_height_m'] * normal), 1e-6)
        self.assertGreater(current['radius_m'] - old['radius_m'], .002)
        self.assertAlmostEqual(current['radius_m'], .025 + .1 * current['contact_height_m'], delta=.0002)

    def test_surface_ray_chooses_outer_skin_and_rejects_missing_mesh(self):
        tri = np.array([[[x, -1., -1.], [x, 1., -1.], [x, 0., 1.]] for x in [-.006, .005]])
        self.assertAlmostEqual(ray_surface_x(tri, 0, 0), .005)
        with self.assertRaisesRegex(ValueError, 'No mesh surface'):
            ray_surface_x(tri, 3, 3)

    def test_endpoint_fractions_are_not_side_contacts(self):
        for fraction in [0, 1, -1, float('nan')]:
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                section(np.ones((200, 3)), np.zeros(3), np.array([0, 0, 1]), fraction, Config())


if __name__ == '__main__':
    unittest.main()
