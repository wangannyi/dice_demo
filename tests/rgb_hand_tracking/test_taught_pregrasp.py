"""Lifted taught endpoints stay bounded and never validate robot execution."""
import math
import unittest

import numpy as np

from rgb_hand_tracking.taught_pregrasp import plan_taught_pregrasp


LIMITS = [[-2., 2.] for _ in range(7)]

# The SDK's saved Nero modified-DH table, replayed as pure numerical FK.
NERO_MDH = ((.138, 0., 0., 0.), (0., 0., math.pi/2, math.pi),
            (.31, 0., math.pi/2, math.pi), (0., 0., math.pi/2, math.pi),
            (.27001, 0., math.pi/2, math.pi/2), (0., 0., math.pi/2, math.pi/2),
            (.0235, 0., math.pi/2, 0.))
TAUGHT_Q_19 = np.radians([30.008, -99.998, -98.003, 105.001, -79.999, -.001, -10.003]).tolist()
SDK_LIMITS_30 = [[-2.705261, 2.705261], [-1.74533, 1.74533], [-2.757621, 2.757621],
                 [-1.012291, 2.146755], [-2.757621, 2.757621], [-.733039, .959932],
                 [-1.570797, 1.570797]]


def nero_mdh_fk(q):
    transform = np.eye(4)
    for angle, (d, a, alpha, offset) in zip(q, NERO_MDH):
        ca, sa = math.cos(alpha), math.sin(alpha)
        c, s = math.cos(angle+offset), math.sin(angle+offset)
        # Rx(alpha) Tx(a) Rz(q+offset) Tz(d).
        transform = transform @ np.array([[c, -s, 0., a],
                                          [ca*s, ca*c, -sa, -d*sa],
                                          [sa*s, sa*c, ca, d*ca],
                                          [0., 0., 0., 1.]])
    rotation = transform[:3, :3]
    return [*transform[:3, 3], math.atan2(rotation[2, 1], rotation[2, 2]),
            math.asin(float(np.clip(-rotation[2, 0], -1., 1.))),
            math.atan2(rotation[1, 0], rotation[0, 0])]


def redundant_fk(q):
    a, b, c, r, p, y, redundant = q
    return [.25*math.sin(a)+.1*redundant, .25*b+.05*redundant,
            .25*c+.01*a*a, r+.2*a, p-.1*b, y+.3*c]


class TaughtPregraspTests(unittest.TestCase):
    def plan(self, **kwargs):
        return plan_taught_pregrasp(kwargs.pop('fk', redundant_fk),
                                   kwargs.pop('taught_q_rad', [0.]*7),
                                   kwargs.pop('joint_limits_rad', LIMITS),
                                   operator_same_cup_scene_confirmed=kwargs.pop('operator_same_cup_scene_confirmed', True),
                                   **kwargs)

    def test_planted_target_retains_full_rotation_and_is_model_only(self):
        planted = np.array([.2, -.3, .4, .1, -.2, .25, .05])
        # The supplied redundant FK has a known exact lifted solution: change
        # c by +.08 rad and yaw by -.024 rad, retaining position X/Y and R.
        expected = planted.copy()
        expected[2] += .08
        expected[5] -= .024
        result = self.plan(taught_q_rad=planted.tolist(), current_q_rad=[0.]*7)
        self.assertTrue(result['model_candidate_valid'], result)
        actual = redundant_fk(result['candidate_q_rad'])
        np.testing.assert_allclose(actual, redundant_fk(expected), atol=2e-5)
        self.assertLessEqual(result['position_error_m'], .00002)
        self.assertLessEqual(result['rotation_error_rad'], math.radians(.05))
        self.assertEqual(result['jacobian_shape'], [6, 7])
        for key in ('physical_contact_verified', 'physical_registration_valid', 'physical_branch_verified',
                    'physical_palm_transform_valid', 'trajectory_validated', 'hand_geometry_collision_validated',
                    'visual_visibility_validated', 'motion_target_valid', 'execution_enabled'):
            self.assertIs(result[key], False)
        for key in ('T_marker_contact', 'selected_marker_pose_index', 'trajectory', 'target_q_rad'):
            self.assertIsNone(result[key])
        self.assertFalse(result['current_reference']['used_as_IK_seed'])
        self.assertFalse(result['current_reference']['path_to_candidate_validated'])

    def test_joint2_boundary_explicit_seed_clamp_keeps_original_target(self):
        taught = [0., math.radians(-99.998), 0., 0., 0., 0., 0.]
        limits = [v.copy() for v in LIMITS]
        limits[1] = [math.radians(-100.), math.radians(100.)]
        margin = math.radians(.1)
        calls = []

        def boundary_fk(q):
            self.assertTrue(all(lo <= v <= hi for v, (lo, hi) in zip(q, limits)))
            if calls:
                self.assertTrue(all(lo+margin <= v <= hi-margin for v, (lo, hi) in zip(q, limits)))
            calls.append(q)
            # Redundant joint7 compensates the numerical seed's J2 clamp, so
            # the actual original flange Y remains reachable inside the margin.
            return [.25*q[0], .25*(q[1]+q[6]), .25*q[2], q[3], q[4], q[5]]

        result = self.plan(fk=boundary_fk, taught_q_rad=taught, joint_limits_rad=limits)
        self.assertTrue(result['valid'], result)
        clamp = result['seed_clamp']
        self.assertTrue(clamp['applied'])
        self.assertEqual([r['joint_number'] for r in clamp['joints']], [2])
        self.assertAlmostEqual(math.degrees(clamp['joints'][0]['seed_rad']), -99.9)
        self.assertAlmostEqual(math.degrees(clamp['joints'][0]['delta_rad']), .098)
        self.assertFalse(clamp['reference_pose_changed_by_clamp'])
        self.assertAlmostEqual(result['target_flange_position_m'][1], .25*taught[1])
        self.assertTrue(all(result['constraint_checks'].values()))
        self.assertGreater(len(calls), 15)

    def test_unreachable_margin_cannot_be_bypassed(self):
        taught = [0., math.radians(-99.998), 0., 0., 0., 0., 0.]
        limits = [v.copy() for v in LIMITS]
        limits[1] = [math.radians(-100.), math.radians(100.)]

        def fk(q):
            return [.25*q[0], .25*q[1], .25*q[2], q[3], q[4], q[5]]

        result = self.plan(fk=fk, taught_q_rad=taught, joint_limits_rad=limits)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['candidate_q_rad'])
        self.assertIsNone(result['target_q_rad'])
        self.assertEqual(result['error']['code'], 'unreachable_within_bounds')
        self.assertTrue(result['constraint_checks']['joint_soft_limits_with_margin'])

    def test_taught_relative_ten_degree_cap_cannot_be_bypassed(self):
        def weak_z_fk(q):
            return [.25*q[0], .25*q[1], .02*q[2], q[3], q[4], q[5]]

        result = self.plan(fk=weak_z_fk)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['candidate_q_rad'])
        self.assertEqual(result['error']['code'], 'unreachable_within_bounds')
        self.assertTrue(result['constraint_checks']['joint_delta_from_original_taught'])

    def test_seed_clamp_must_fit_cap_relative_to_original_taught(self):
        taught = [0., math.radians(-99.998), 0., 0., 0., 0., 0.]
        limits = [v.copy() for v in LIMITS]
        limits[1] = [math.radians(-100.), math.radians(100.)]

        def forbidden(q):
            self.fail('An empty bounded seed box must reject before FK')

        result = self.plan(fk=forbidden, taught_q_rad=taught, joint_limits_rad=limits,
                           max_joint_delta_rad=math.radians(.05))
        self.assertFalse(result['valid'])
        self.assertIsNone(result['candidate_q_rad'])
        self.assertEqual(result['error']['code'], 'invalid_input')

    def test_euler_wrap_retains_rotation_with_identical_candidate(self):
        def fk(q, wrap=False):
            pose = redundant_fk(q)
            pose[5] += math.pi-1e-6
            if wrap:
                pose[5] = (pose[5]+math.pi) % (2*math.pi)-math.pi
            return pose

        plain = self.plan(fk=fk)
        wrapped = self.plan(fk=lambda q: fk(q, wrap=True))
        self.assertTrue(plain['valid'], plain)
        self.assertTrue(wrapped['valid'], wrapped)
        np.testing.assert_allclose(plain['candidate_q_rad'], wrapped['candidate_q_rad'], atol=1e-9)

    def test_invalid_lift_limits_and_confirmation_never_query_fk(self):
        def forbidden(q):
            self.fail('Invalid input must not query FK')

        cases = [{'lift_m': v} for v in (0., -.020, .00099, .03001, float('nan'), True)]
        cases += [{'operator_same_cup_scene_confirmed': False}, {'operator_same_cup_scene_confirmed': 1},
                  {'joint_margin_rad': 0.}, {'joint_margin_rad': math.radians(.099)},
                  {'max_joint_delta_rad': math.radians(10.01)}, {'taught_q_rad': [2.001]+[0.]*6},
                  {'current_q_rad': [2.001]+[0.]*6}, {'joint_limits_rad': [[1., -1.]]*7},
                  {'taught_q_rad': [False]+[0.]*6}, {'taught_q_rad': [0.]*6},
                  {'finite_difference_rad': 0.}, {'damping': 0.}, {'max_iterations': True}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                result = self.plan(fk=forbidden, **kwargs)
                self.assertFalse(result['valid'])
                self.assertIsNone(result['candidate_q_rad'])
                self.assertEqual(result['error']['code'], 'invalid_input')

    def test_rank_deficient_fk_rejects_even_if_lift_reachable(self):
        result = self.plan(fk=lambda q: [.25*q[0], .25*q[1], .25*q[2], 0., 0., 0.])
        self.assertFalse(result['valid'])
        self.assertIsNone(result['candidate_q_rad'])
        self.assertEqual(result['error']['code'], 'rank_insufficient')

    def test_unknown_malformed_or_failed_fk_never_substitutes_pose(self):
        def broken(q):
            raise OSError('FK unavailable')

        for fk in (lambda q: None, lambda q: [0.]*5, lambda q: [float('nan')]*6,
                   lambda q: ['unknown']*6, lambda q: [True]+[0.]*5, broken):
            with self.subTest(fk=fk):
                result = self.plan(fk=fk)
                self.assertFalse(result['valid'])
                self.assertIsNone(result['candidate_q_rad'])
                self.assertEqual(result['error']['code'], 'fk_failure')

    def test_fk_failure_during_jacobian_does_not_publish_partial_candidate(self):
        def only_reference(q):
            if any(q):
                raise RuntimeError('FK invalid away from taught pose')
            return redundant_fk(q)

        result = self.plan(fk=only_reference)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['candidate_q_rad'])
        self.assertEqual(result['error']['code'], 'fk_failure')

    def test_positive_endpoints_one_and_thirty_mm(self):
        for lift in (.001, .030):
            result = self.plan(lift_m=lift)
            self.assertTrue(result['valid'], result)
            self.assertAlmostEqual(result['predicted_displacement_from_taught_m'][2], lift, delta=.00002)

    def test_extended_ninety_mm_requires_explicit_opt_in_and_vector_caps(self):
        caps = np.radians([30., 30., 30., 30., 10., 30., 10.]).tolist()
        planted = [.2, -.3, .4, .1, -.2, .25, .05]
        expected = planted.copy()
        expected[2] += .36
        expected[5] -= .108
        result = self.plan(taught_q_rad=planted, lift_m=.090,
                           allow_extended_lift=True, per_joint_delta_caps_rad=caps)
        self.assertTrue(result['valid'], result)
        np.testing.assert_allclose(redundant_fk(result['candidate_q_rad']), redundant_fk(expected), atol=2e-5)
        self.assertTrue(all(result['constraint_checks'].values()))
        self.assertEqual(result['planner_configuration']['per_joint_delta_caps_rad'], caps)
        self.assertTrue(result['planner_configuration']['allow_extended_lift'])
        self.assertEqual(result['planner_configuration']['maximum_lift_m'], .120)
        self.assertFalse(result['physical_contact_verified'])
        self.assertFalse(result['hand_geometry_collision_validated'])
        self.assertFalse(result['execution_enabled'])
        self.assertIsNone(result['target_q_rad'])
        self.assertIsNone(result['trajectory'])
        without_opt_in = self.plan(taught_q_rad=planted, lift_m=.090, per_joint_delta_caps_rad=caps)
        self.assertFalse(without_opt_in['valid'])
        self.assertIsNone(without_opt_in['candidate_q_rad'])
        # Extended height does not silently enlarge the original scalar cap.
        without_caps = self.plan(taught_q_rad=planted, lift_m=.090, allow_extended_lift=True)
        self.assertFalse(without_caps['valid'])
        self.assertIsNone(without_caps['candidate_q_rad'])

    def test_explicit_caps_respect_protected_joint5_and_joint7_ceilings(self):
        ceilings = np.radians([30., 30., 30., 30., 10., 30., 10.])
        for index in range(7):
            invalid = ceilings.copy()
            invalid[index] += math.radians(.001)
            with self.subTest(joint=index+1):
                result = self.plan(per_joint_delta_caps_rad=invalid.tolist())
                self.assertFalse(result['valid'])
                self.assertIsNone(result['candidate_q_rad'])
                self.assertEqual(result['error']['code'], 'invalid_input')
        # Reaching these lifts would require protected J5/J7 >10 degrees.
        def coupled_joint5(q):
            return [.25*q[0], .25*q[1], .25*q[2], q[3], q[4]+q[2], q[5]]

        def joint7_lift(q):
            return [.25*q[0], .25*q[1], .25*q[6], q[3], q[4], q[5]]

        for fk in (coupled_joint5, joint7_lift):
            with self.subTest(fk=fk):
                result = self.plan(fk=fk, lift_m=.090, allow_extended_lift=True,
                                   per_joint_delta_caps_rad=ceilings.tolist())
                self.assertFalse(result['valid'])
                self.assertIsNone(result['candidate_q_rad'])
                self.assertTrue(result['constraint_checks']['joint_delta_from_original_taught'])

    def test_extended_inputs_cannot_bypass_limits_margin_or_shape(self):
        caps = np.radians([30., 30., 30., 30., 10., 30., 10.]).tolist()
        cases = [{'per_joint_delta_caps_rad': [1.]*6}, {'per_joint_delta_caps_rad': [[.1]]*7},
                 {'per_joint_delta_caps_rad': [True]+[.1]*6},
                 {'per_joint_delta_caps_rad': [float('nan')]+[.1]*6},
                 {'per_joint_delta_caps_rad': [0.]+[.1]*6},
                 {'allow_extended_lift': 1}, {'lift_m': .12001}, {'joint_margin_rad': 0.},
                 {'taught_q_rad': [2.001]+[0.]*6}, {'max_joint_delta_rad': math.radians(30.)}]
        for changes in cases:
            options = {'lift_m': .090, 'allow_extended_lift': True, 'per_joint_delta_caps_rad': caps}
            options.update(changes)
            with self.subTest(changes=changes):
                result = self.plan(**options)
                self.assertFalse(result['valid'])
                self.assertIsNone(result['candidate_q_rad'])
                self.assertEqual(result['error']['code'], 'invalid_input')

    def test_original_twenty_mm_default_matches_explicit_original_caps(self):
        default = self.plan()
        explicit = self.plan(lift_m=.020, allow_extended_lift=False,
                             per_joint_delta_caps_rad=[math.radians(10.)]*7)
        self.assertTrue(default['valid'], default)
        self.assertTrue(explicit['valid'], explicit)
        np.testing.assert_array_equal(default['candidate_q_rad'], explicit['candidate_q_rad'])
        self.assertEqual(default['planner_configuration']['joint_cap_policy'], 'scalar_default')
        self.assertEqual(default['planner_configuration']['per_joint_delta_caps_rad'], None)
        self.assertEqual(default['tolerances']['maximum_delta_from_taught_rad'], math.radians(10.))

    def test_extended_lift_still_checks_supplied_soft_limit_box(self):
        caps = np.radians([30., 30., 30., 30., 10., 30., 10.]).tolist()
        limits = [v.copy() for v in LIMITS]
        limits[2] = [-.2, .2]
        calls = []

        def checked(q):
            self.assertTrue(all(lo+math.radians(.1) <= v <= hi-math.radians(.1)
                                for v, (lo, hi) in zip(q, limits)))
            calls.append(q)
            return [.25*q[0], .25*q[1], .25*q[2], q[3], q[4], q[5]]

        result = self.plan(fk=checked, joint_limits_rad=limits, lift_m=.090,
                           allow_extended_lift=True, per_joint_delta_caps_rad=caps)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['candidate_q_rad'])
        self.assertEqual(result['error']['code'], 'unreachable_within_bounds')
        self.assertTrue(result['constraint_checks']['joint_soft_limits_with_margin'])
        self.assertGreater(len(calls), 10)

    def test_exact_cap_endpoint_survives_subtraction_roundoff_without_epsilon(self):
        cap = math.radians(10.)
        taught = [0.]*6+[TAUGHT_Q_19[6]]
        calls = []

        def boundary_fk(q):
            self.assertGreaterEqual(q[6], taught[6]-cap)
            self.assertLessEqual(q[6], taught[6]+cap)
            calls.append(q)
            # The target is slightly beyond the box, but its exact boundary
            # pose is within 20 um. This forces np.clip onto the legal endpoint.
            return [.25*q[0], .25*q[1], -(.020/(cap+1e-5))*q[6], q[3], q[4], q[5]]

        result = self.plan(fk=boundary_fk, taught_q_rad=taught)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['candidate_q_rad'][6], taught[6]-cap)
        self.assertGreater(abs(result['candidate_q_rad'][6]-taught[6]), cap)
        diagnostics = result['joint_bound_diagnostics']
        self.assertTrue(diagnostics['exact_effective_box_verified'])
        self.assertFalse(diagnostics['raw_subtraction_cap_check'])
        self.assertGreater(diagnostics['raw_subtraction_cap_excess_rad'][6], 0.)
        self.assertEqual(diagnostics['epsilon_expansion_rad'], 0.)
        self.assertLessEqual(result['position_error_m'], .00002)
        self.assertTrue(all(result['constraint_checks'].values()))

    def test_saved_nero_mdh_preserves_real_preview_position_rejection(self):
        np.testing.assert_allclose(nero_mdh_fk(TAUGHT_Q_19),
                                   [.05449596660824044, .3235256496484632, .06923605675190876,
                                    -1.1912563324862577, -.07459765799337183, 2.3129938485861063],
                                   atol=1e-14, rtol=0)
        caps = np.radians([30., 30., 30., 30., 10., 30., 10.]).tolist()
        result = self.plan(fk=nero_mdh_fk, taught_q_rad=TAUGHT_Q_19,
                           joint_limits_rad=SDK_LIMITS_30, lift_m=.090,
                           allow_extended_lift=True, per_joint_delta_caps_rad=caps)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['candidate_q_rad'])
        self.assertAlmostEqual(result['position_error_m'], .000024110490, delta=1e-9)
        self.assertGreater(result['position_error_m'], .00002)
        self.assertTrue(result['constraint_checks']['joint_delta_from_original_taught'])
        self.assertTrue(result['constraint_checks']['joint_soft_limits_with_margin'])
        self.assertTrue(result['constraint_checks']['taught_rotation_retained'])
        self.assertEqual(result['solver_termination'], 'weighted_objective_stalled')
        self.assertFalse(result['joint_bound_diagnostics']['raw_subtraction_cap_check'])

    def test_explicit_merit_weight_nero_mdh_accepts_without_changing_pose_gates(self):
        caps = np.radians([30., 30., 30., 30., 10., 30., 10.]).tolist()
        result = self.plan(fk=nero_mdh_fk, taught_q_rad=TAUGHT_Q_19,
                           joint_limits_rad=SDK_LIMITS_30, lift_m=.090,
                           allow_extended_lift=True, per_joint_delta_caps_rad=caps,
                           rotation_length_m=.08)
        self.assertTrue(result['model_candidate_valid'], result)
        self.assertAlmostEqual(result['position_error_m'], .000015939838, delta=1e-9)
        self.assertAlmostEqual(math.degrees(result['rotation_error_rad']), .046128688, delta=1e-6)
        self.assertTrue(all(result['constraint_checks'].values()))
        self.assertEqual(result['tolerances']['position_m'], .00002)
        self.assertEqual(result['tolerances']['rotation_rad'], math.radians(.05))
        self.assertEqual(result['tolerances']['joint_margin_rad'], math.radians(.1))
        self.assertEqual(result['planner_configuration']['rotation_length_m'], .08)
        self.assertFalse(result['execution_enabled'])
        self.assertIsNone(result['target_q_rad'])
        bounds = result['effective_joint_bounds_rad']
        self.assertTrue(all(lo <= v <= hi for v, (lo, hi) in zip(result['candidate_q_rad'], bounds)))

    def test_invalid_merit_weight_rejects_before_fk_and_default_stays_point_one(self):
        def forbidden(q):
            self.fail('Invalid merit parameters must not query FK')

        for value in (0., -.08, .04999, .10001, float('nan'), True):
            with self.subTest(value=value):
                result = self.plan(fk=forbidden, rotation_length_m=value)
                self.assertFalse(result['valid'])
                self.assertIsNone(result['candidate_q_rad'])
                self.assertEqual(result['error']['code'], 'invalid_input')
        default, explicit = self.plan(), self.plan(rotation_length_m=.1)
        self.assertTrue(default['valid'], default)
        self.assertTrue(explicit['valid'], explicit)
        np.testing.assert_array_equal(default['candidate_q_rad'], explicit['candidate_q_rad'])
        self.assertEqual(default['planner_configuration']['rotation_length_m'], .1)


if __name__ == '__main__':
    unittest.main()
