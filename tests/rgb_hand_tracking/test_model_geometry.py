import unittest
import numpy as np
from rgb_hand_tracking.model_geometry import intersect_x, section


class Tests(unittest.TestCase):
    def test_known_parallel_surfaces(self):
        # Two parallel 20 mm-separated squares, each split along its diagonal.
        triangles = []
        for x in (-.012, .008):
            a,b,c,d = [[x,-.02,-.02],[x,.02,-.02],[x,.02,.02],[x,-.02,.02]]
            triangles.extend([[a,b,c],[a,c,d]])
        tri = np.array(triangles)
        np.testing.assert_allclose(intersect_x(tri, 0, 0), [-.012,.008])
        self.assertAlmostEqual(section(tri, 0, 0)['thickness_m'], .02)
        with self.assertRaises(ValueError):
            section(tri, .1, 0)

    def test_parallel_ray_does_not_create_false_hits(self):
        tri = np.array([[[0,0,0],[1,0,0],[0,0,1]]],float)
        self.assertEqual(len(intersect_x(tri, 0, .2)), 0)


if __name__ == '__main__':
    unittest.main()
