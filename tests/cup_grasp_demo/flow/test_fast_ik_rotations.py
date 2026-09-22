"""FAST FK must preserve the model pose, Jacobian and constrained IK solution."""
import unittest
from unittest.mock import patch
from scipy.spatial.transform import Rotation
import numpy as np
from cup_grasp_demo.flow.green_cup_planning import batched_rotation_forward, solve, vertical_targets, proper_rotation_vector
from cup_grasp_demo.flow.shake import Kinematics


class FastIKTests(unittest.TestCase):
    def test_lift_minimax_preserves_tcp_endpoint_and_reduces_travel(self):
        kin = Kinematics()
        start = np.array([.94984, -1.35877, -2.07918, 1.57910, .07915, -.46152, -.71604])
        tcp = np.eye(4)
        tcp[:3, 3] = [.1, .02, .03]
        old = np.array(vertical_targets(start, tcp, .05, [15, -13, 5], single_target=True, fast_fk=True)[0])
        new = np.array(vertical_targets(start, tcp, .05, [15, -13, 5], single_target=True, fast_fk=True, minimize_travel=True)[0])
        target = kin.forward(start)[0] @ tcp
        target[2, 3] += .05
        np.testing.assert_allclose(kin.forward(new)[0] @ tcp, target, atol=1e-6)
        self.assertLess(np.max(abs(new-start)), np.max(abs(old-start)))

    def test_lift_minimax_failure_keeps_original_solution(self):
        from types import SimpleNamespace
        from cup_grasp_demo.flow.green_cup_planning import minimize_joint_travel
        start = np.radians([25, -80, -100, 70, 0, -13, 5])
        with patch('cup_grasp_demo.flow.green_cup_planning.minimize', return_value=SimpleNamespace(success=False)):
            np.testing.assert_array_equal(minimize_joint_travel(start, start, Kinematics().forward(start)[0]), start)

    def test_dogbox_reachable_targets_preserve_pose_and_joint_limits(self):
        kin = Kinematics()
        seed = np.radians([25, -80, -100, 70, 0, -13, 5])
        for delta in ([5, 3, -2, 4, 2, 1, -3], [-8, 5, 4, 2, -3, 2, 7], [12, 4, -8, 8, 6, 3, -10]):
            target = kin.forward(seed + np.radians(delta))[0]
            q = solve(target, seed, [0, -13, 5], fast_fk=True, method='dogbox')
            pose = kin.forward(q)[0]
            self.assertLessEqual(np.linalg.norm(pose[:3, 3] - target[:3, 3]), .0015)
            self.assertLessEqual(np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec()), np.radians(1))
            self.assertTrue(np.all(q >= kin.lower + np.radians(1)))
            self.assertTrue(np.all(q <= kin.upper - np.radians(1)))

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
        with patch('cup_grasp_demo.flow.green_cup_planning._ROTATION_ASSUME_VALID', False):
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
