import math
import unittest

import numpy as np

from rgb_hand_tracking.cartesian_microstep import MAX_JOINT_DELTA_RAD, plan_microstep


LIMITS = [[-2., 2.] for _ in range(7)]


def redundant_fk(q, *, wrap=False):
    x, y, z, r, p, yaw, redundant = q
    angle = math.pi-1e-6+yaw+.5*x
    if wrap:
        angle = (angle+math.pi) % (2*math.pi)-math.pi
    return [.25*math.sin(x)+.1*redundant, .25*y, .25*z+.01*x*x,
            r+.2*x, p-.1*y, angle]


class CartesianMicrostepTests(unittest.TestCase):
    def plan(self, **kwargs):
        return plan_microstep(kwargs.pop('fk', redundant_fk),
                              kwargs.pop('q_rad', [0.]*7),
                              kwargs.pop('joint_limits_rad', LIMITS),
                              axis=kwargs.pop('axis', 'x'),
                              distance_m=kwargs.pop('distance_m', .001), **kwargs)

    def test_redundant_positive_with_coupled_rotation_and_z(self):
        result = self.plan()
        self.assertTrue(result['valid'], result)
        self.assertTrue(all(result['constraint_checks'].values()))
        self.assertAlmostEqual(result['predicted_displacement_m'][0], .001, delta=2e-5)
        self.assertLess(abs(result['predicted_displacement_m'][2]), 1e-4)
        self.assertLess(result['orientation_error_rad'], math.radians(.05))
        self.assertLessEqual(max(abs(q) for q in result['delta_q_rad']), MAX_JOINT_DELTA_RAD)
        self.assertFalse(result['contact_point_experimentally_verified'])

    def test_negative_y_and_smaller_x_have_requested_sign(self):
        for axis, amount in [('y', -.001), ('x', .00001)]:
            with self.subTest(axis=axis, amount=amount):
                result = self.plan(axis=axis, distance_m=amount)
                self.assertTrue(result['valid'], result)
                index = 0 if axis == 'x' else 1
                self.assertAlmostEqual(result['predicted_displacement_m'][index], amount,
                                       delta=min(2e-5, abs(amount)*.02))

    def test_euler_wrap_does_not_change_plan(self):
        plain = self.plan()
        wrapped = self.plan(fk=lambda q: redundant_fk(q, wrap=True))
        self.assertTrue(wrapped['valid'], wrapped)
        np.testing.assert_allclose(wrapped['target_q_rad'], plain['target_q_rad'], atol=1e-9)
        self.assertLess(wrapped['orientation_error_rad'], 1e-5)

    def test_invalid_inputs_never_publish_target(self):
        cases = [{'axis': 'z'}, {'distance_m': 0.}, {'distance_m': .00101},
                 {'distance_m': float('nan')}, {'distance_m': True},
                 {'q_rad': [0.]*6}, {'q_rad': [False]+[0.]*6},
                 {'q_rad': [float('inf')]+[0.]*6},
                 {'joint_limits_rad': [[1., -1.]]*7},
                 {'joint_margin_rad': 3.}, {'joint_margin_rad': -.01},
                 {'max_joint_delta_rad': math.radians(.36)},
                 {'finite_difference_rad': 0.}, {'damping': 0.},
                 {'max_iterations': True}, {'max_iterations': 0}]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = self.plan(**arguments)
                self.assertFalse(result['valid'])
                self.assertIsNone(result['target_q_rad'])
                self.assertEqual(result['error']['code'], 'invalid_input')

    def test_starting_outside_margin_is_not_silently_clipped(self):
        result = self.plan(q_rad=[1.999]+[0.]*6)
        self.assertFalse(result['valid'])
        self.assertEqual(result['error']['code'], 'invalid_input')
        self.assertNotIn('candidate_q_rad', result)

    def test_rank_deficient_fk_is_rejected_even_if_x_reachable(self):
        result = self.plan(fk=lambda q: [.25*q[0], .25*q[1], .25*q[2], 0., 0., 0.])
        self.assertFalse(result['valid'])
        self.assertIsNone(result['target_q_rad'])
        self.assertEqual(result['error']['code'], 'rank_insufficient')

    def test_soft_boundary_unreachable_no_calls_outside_limits(self):
        limits = [[-.002, .003]]+LIMITS[1:]
        samples = []

        def limited(q):
            self.assertTrue(all(lo <= v <= hi for v, (lo, hi) in zip(q, limits)))
            samples.append(q)
            return [.25*q[0], .25*q[1], .25*q[2], q[3], q[4], q[5]]

        result = self.plan(fk=limited, joint_limits_rad=limits, joint_margin_rad=0.)
        self.assertFalse(result['valid'])
        self.assertEqual(result['error']['code'], 'unreachable_within_bounds')
        self.assertIsNone(result['target_q_rad'])
        self.assertAlmostEqual(result['candidate_q_rad'][0], .003)
        self.assertGreater(len(samples), 10)

    def test_joint_delta_cap_rejects_unreachable_task(self):
        result = self.plan(fk=lambda q: [.05*q[0]+.02*q[6], .25*q[1], .25*q[2],
                                       q[3], q[4], q[5]])
        self.assertFalse(result['valid'])
        self.assertIsNone(result['target_q_rad'])
        self.assertTrue(result['constraint_checks']['joint_delta'])
        self.assertEqual(result['error']['code'], 'unreachable_within_bounds')

    def test_finite_difference_stays_in_supplied_margin_box(self):
        margin = math.radians(.1)
        q = [2.-margin]+[0.]*6

        def checked(values):
            self.assertTrue(all(-2.+margin <= v <= 2.-margin for v in values))
            return [.25*values[0], .25*values[1], .25*values[2],
                    values[3], values[4], values[5]]

        result = self.plan(fk=checked, q_rad=q, distance_m=-.001)
        self.assertTrue(result['valid'], result)

    def test_malformed_fk_and_fk_failure_are_not_bypassed(self):
        def broken(q):
            raise RuntimeError('FK unavailable')

        for fk in [lambda q: [0.]*5, lambda q: [float('nan')]*6, broken]:
            result = self.plan(fk=fk)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['target_q_rad'])
            self.assertEqual(result['error']['code'], 'fk_failure')

    def test_failure_during_jacobian_is_reported_as_fk_failure(self):
        def at_start_only(q):
            if any(q):
                raise OSError('FK failed off start')
            return redundant_fk(q)

        result = self.plan(fk=at_start_only)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['target_q_rad'])
        self.assertEqual(result['error']['code'], 'fk_failure')


if __name__ == '__main__':
    unittest.main()
