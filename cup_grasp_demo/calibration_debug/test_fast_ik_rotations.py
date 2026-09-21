"""FAST FK must preserve the model pose, Jacobian and constrained IK solution."""
import unittest
from unittest.mock import patch
from scipy.spatial.transform import Rotation
import numpy as np
from cup_grasp_demo.calibration_debug.green_cup_planning import batched_rotation_forward, solve, vertical_targets, proper_rotation_vector
from cup_grasp_demo.calibration_debug.shake import Kinematics


class FastIKTests(unittest.TestCase):
    def test_rotation_log_matches_scipy_including_zero_and_pi(self):
        rng = np.random.default_rng(39)
        vectors = list(Rotation.random(1000, random_state=rng).as_rotvec())
        axis = np.array([1., -2., 3.]); axis /= np.linalg.norm(axis)
        vectors += [axis*a for a in [0.,1e-12,1e-8,1e-4,np.pi-1e-3,np.pi-1e-5,np.pi,np.pi+1e-6]]
        for vector in vectors:
            matrix = Rotation.from_rotvec(vector).as_matrix()
            np.testing.assert_allclose(proper_rotation_vector(matrix),Rotation.from_matrix(matrix).as_rotvec(),atol=1e-10,rtol=1e-10)

    def test_older_rotation_api_falls_back_to_checked_conversion(self):
        kin = Kinematics()
        seed = np.radians([25, -80, -100, 70, 0, -13, 5])
        target = kin.forward(seed + np.radians([5, 3, -2, 4, 2, 1, -3]))[0]
        with patch('cup_grasp_demo.calibration_debug.green_cup_planning._ROTATION_ASSUME_VALID', False):
            checked = solve(target, seed, [0, -13, 5], fast_fk=True)
        fast = solve(target, seed, [0, -13, 5], fast_fk=True)
        np.testing.assert_allclose(fast, checked, atol=1e-6)

    def test_vertical_lift_preserves_tool_pose(self):
        kin = Kinematics()
        seed = np.radians([25, -80, -100, 70, 0, -13, 5])
        tcp = np.eye(4)
        tcp[:3, 3] = [.10, .02, .03]
        for distance in (.05, -.05):
            slow = vertical_targets(seed, tcp, distance, [0, -13, 5], single_target=True)
            fast = vertical_targets(seed, tcp, distance, [0, -13, 5], single_target=True, fast_fk=True)
            np.testing.assert_allclose(kin.forward(fast[-1])[0] @ tcp,
                                       kin.forward(slow[-1])[0] @ tcp, atol=1e-6)

    def test_random_joint_poses_and_jacobians_match(self):
        kin = Kinematics()
        rng = np.random.default_rng(42)
        for q in rng.uniform(kin.lower, kin.upper, size=(100, 7)):
            actual, jac = batched_rotation_forward(kin, q)
            expected, expected_jac = kin.forward(q)
            np.testing.assert_allclose(actual, expected, atol=2e-15)
            np.testing.assert_allclose(jac, expected_jac, atol=2e-15)

    def test_solver_preserves_endpoint_and_wrist_preference(self):
        kin = Kinematics()
        seed = np.radians([25, -80, -100, 70, 0, -13, 5])
        target = kin.forward(seed + np.radians([5, 3, -2, 4, 2, 1, -3]))[0]
        slow = solve(target, seed, [0, -13, 5])
        fast = solve(target, seed, [0, -13, 5], fast_fk=True)
        np.testing.assert_allclose(fast, slow, atol=1e-6)
