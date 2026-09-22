"""Batch FK preserves the scalar solver's geometry at all input joint samples."""
import unittest
import numpy as np
from cup_grasp_demo.calibration_debug.shake import Kinematics


class BatchFKTest(unittest.TestCase):
    def test_same_transforms_as_scalar_including_limits(self):
        kin=Kinematics()
        rng=np.random.default_rng(211)
        qs=np.vstack([rng.uniform(kin.lower,kin.upper,(128,7)),kin.lower,kin.upper,np.zeros(7)])
        expected=np.array([kin.forward(q)[0] for q in qs])
        np.testing.assert_allclose(kin.forward_batch(qs),expected,atol=1e-13,rtol=0)
        np.testing.assert_allclose(kin.forward_batch(qs)[0],kin.forward_batch(qs[:1])[0],atol=1e-13,rtol=0)

    def test_invalid_samples_fail_before_computation(self):
        kin=Kinematics()
        for qs in ([],[0]*7,[[0]*6],[[float('nan')]*7]):
            with self.subTest(qs=qs),self.assertRaises(ValueError):kin.forward_batch(qs)


if __name__=='__main__':unittest.main()
