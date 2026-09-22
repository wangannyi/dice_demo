"""Contact geometry regressions independent of cameras, CAN, and learned models."""
import unittest

import numpy as np

from cup_grasp_demo.side_grasp.preview_index import ray_surface_x, section
from dice_cup_localization.geometry import Config


class IndexPreviewTests(unittest.TestCase):
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
