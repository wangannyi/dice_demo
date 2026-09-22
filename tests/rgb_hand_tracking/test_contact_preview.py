import unittest
import numpy as np
from rgb_hand_tracking.contact_preview import ray_plane, project


class Tests(unittest.TestCase):
    def test_plane_intersection_roundtrip(self):
        k=np.array([[900.,0,640],[0,900,360],[0,0,1]])
        d=np.zeros(5)
        point=ray_plane([700,400],k,d,np.array([0,0,.5]),np.array([0,0,1]))
        self.assertAlmostEqual(point[2],.5)
        np.testing.assert_allclose(project(point,k,d),[700,400])

    def test_invalid_intersections(self):
        k=np.eye(3)
        with self.assertRaises(ValueError):
            ray_plane([0,0],k,np.zeros(5),np.array([1,0,0]),np.array([1,0,0]))
        with self.assertRaises(ValueError):
            ray_plane([0,0],k,np.zeros(5),np.array([0,0,-1]),np.array([0,0,1]))


if __name__=='__main__':
    unittest.main()
