"""Pure injected-FK planning of a lifted, operator-taught flange pose.

The reference is the ORIGINAL taught flange pose, even when its joints must
be clamped for a soft-limit-safe numerical seed. A candidate is an endpoint
model prediction only. No palm transform, physical contact, IPPE branch,
trajectory, collision clearance, visual visibility or execution is validated.
"""
import math

import numpy as np

from cartesian_microstep import (
    FKFailure, ROTATION_LENGTH_M, _bounded_step, _error, _jacobian,
    _numbers, _pose, _rotation, _rotation_vector)


MIN_LIFT_M = .001
MAX_LIFT_M = .030
MAX_EXTENDED_LIFT_M = .120
MAX_JOINT_CHANGE_RAD = math.radians(10.)
PER_JOINT_CAP_CEILINGS_RAD = tuple(math.radians(v) for v in (30., 30., 30., 30., 10., 30., 10.))
MIN_SOFT_MARGIN_RAD = math.radians(.1)
POSITION_TOLERANCE_M = .00002
ROTATION_TOLERANCE_RAD = math.radians(.05)
MIN_MERIT_ROTATION_LENGTH_M = .05
MAX_MERIT_ROTATION_LENGTH_M = .1


def plan_taught_pregrasp(fk, taught_q_rad, joint_limits_rad, *, lift_m=.020,
                        current_q_rad=None, operator_same_cup_scene_confirmed=False,
                        joint_margin_rad=MIN_SOFT_MARGIN_RAD,
                        max_joint_delta_rad=MAX_JOINT_CHANGE_RAD,
                        allow_extended_lift=False, per_joint_delta_caps_rad=None,
                        rotation_length_m=ROTATION_LENGTH_M,
                        finite_difference_rad=1e-5, damping=1e-5, max_iterations=80):
    """Return an inactive, bounded IK endpoint candidate; failures publish null.

    FK: seven radians -> flange [x,y,z,roll,pitch,yaw] in base metres/radians,
    with Rz(yaw) Ry(pitch) Rx(roll). Default lift is base +Z, 1..30 mm;
    allow_extended_lift=True explicitly permits 1..120 mm. Full rotation
    is retained relative to the original taught FK. Taught/current references
    may be evaluated inside hard limits; all new IK probes/candidates stay in
    supplied limits plus >=0.1 degree margin. Default joint cap is original
    taught +/-10 degrees. An explicit seven-element per-joint cap vector can
    replace it, with ceilings [30,30,30,30,10,30,10] degrees; these limits do
    not relax supplied joint limits, pose tolerances or soft-limit margin.
    rotation_length_m in [.05, .1] changes only the solver's rotational merit
    weight; actual rotation acceptance remains <=0.05 degree, position <=20 um.
    Current joints, if provided, are diagnostic only and do not set the seed.

    Operator confirmation is a scene assertion, not measured cup immobility.
    Caller must verify the real path, whole hand clearance, scene, feedback
    freshness and visibility separately before considering a motion command.
    """
    report = {'schema': 'offline_taught_pregrasp_model_v1', 'valid': False,
              'model_candidate_valid': False, 'candidate_q_rad': None,
              'target_q_rad': None, 'error': None, 'iterations': 0,
              'physical_contact_verified': False, 'physical_registration_valid': False,
              'physical_branch_verified': False, 'physical_palm_transform_valid': False,
              'T_marker_contact': None, 'selected_marker_pose_index': None,
              'trajectory': None, 'trajectory_validated': False,
              'hand_geometry_collision_validated': False, 'visual_visibility_validated': False,
              'motion_target_valid': False, 'execution_enabled': False,
              'scene_identity_independently_verified': False,
              'scope': 'Injected FK endpoint model only; no SDK/device access or motion commands'}
    stage = 'invalid_input'
    try:
        if not callable(fk) or operator_same_cup_scene_confirmed is not True:
            raise ValueError('Callable FK and explicit operator same-cup/same-scene confirmation required')
        taught = _numbers(taught_q_rad, (7,), 'seven taught joints')
        limits = _numbers(joint_limits_rad, (7, 2), 'seven supplied joint limits')
        lift, margin, cap, increment, damp, rotation_length = _numbers(
            [lift_m, joint_margin_rad, max_joint_delta_rad, finite_difference_rad, damping,
             rotation_length_m], (6,), 'planner parameters')
        if type(allow_extended_lift) is not bool:
            raise ValueError('Extended lift requires an explicit boolean opt-in')
        lift_ceiling = MAX_EXTENDED_LIFT_M if allow_extended_lift else MAX_LIFT_M
        caps = (np.full(7, cap) if per_joint_delta_caps_rad is None else
                _numbers(per_joint_delta_caps_rad, (7,), 'seven per-joint delta caps'))
        ceilings = np.asarray(PER_JOINT_CAP_CEILINGS_RAD)
        report['planner_configuration'] = {
            'lift_m': float(lift), 'allow_extended_lift': allow_extended_lift,
            'maximum_lift_m': lift_ceiling,
            'joint_margin_rad': float(margin), 'scalar_max_joint_delta_rad': float(cap),
            'per_joint_delta_caps_rad': None if per_joint_delta_caps_rad is None else caps.tolist(),
            'effective_per_joint_delta_caps_rad': caps.tolist(),
            'per_joint_cap_ceilings_rad': ceilings.tolist(),
            'joint_cap_policy': 'scalar_default' if per_joint_delta_caps_rad is None else 'explicit_per_joint',
            'rotation_length_m': float(rotation_length), 'rotation_weight_is_merit_only': True,
            'finite_difference_rad': float(increment), 'damping': float(damp),
            'max_iterations': max_iterations}
        if not MIN_LIFT_M <= lift <= lift_ceiling:
            raise ValueError('Lift must be positive base +Z in the explicitly permitted range')
        if (margin < MIN_SOFT_MARGIN_RAD or not 0 < cap <= MAX_JOINT_CHANGE_RAD
                or not 0 < increment <= 1e-3 or not 0 < damp <= .01
                or isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
                or not 1 <= max_iterations <= 100):
            raise ValueError('Invalid soft margin, taught-relative joint cap or solver parameters')
        if np.any(caps <= 0) or np.any(caps > ceilings):
            raise ValueError('Per-joint caps exceed fixed [30,30,30,30,10,30,10] degree ceilings or are nonpositive')
        if not MIN_MERIT_ROTATION_LENGTH_M <= rotation_length <= MAX_MERIT_ROTATION_LENGTH_M:
            raise ValueError('Rotational merit length must be in [.05, .1] metres; pose gates remain fixed')
        hard_lower, hard_upper = limits[:, 0], limits[:, 1]
        soft_lower, soft_upper = hard_lower+margin, hard_upper-margin
        if (np.any(hard_lower >= hard_upper) or np.any(soft_lower >= soft_upper)
                or np.any(taught < hard_lower) or np.any(taught > hard_upper)):
            raise ValueError('Taught reference or margin violates supplied joint limits')
        current = None if current_q_rad is None else _numbers(current_q_rad, (7,), 'seven current joints')
        if current is not None and (np.any(current < hard_lower) or np.any(current > hard_upper)):
            raise ValueError('Current reference violates supplied hard joint limits')
        lower, upper = np.maximum(soft_lower, taught-caps), np.minimum(soft_upper, taught+caps)
        if np.any(lower >= upper):
            raise ValueError('No nonempty soft-limit box within taught-relative joint cap')
        seed = np.clip(taught, lower, upper)
        clamps = [{'joint_index': i, 'joint_number': i+1,
                   'taught_rad': float(taught[i]), 'seed_rad': float(seed[i]),
                   'delta_rad': float(seed[i]-taught[i]),
                   'soft_lower_rad': float(soft_lower[i]), 'soft_upper_rad': float(soft_upper[i])}
                  for i in range(7) if seed[i] != taught[i]]
        report.update({'operator_same_cup_scene_confirmed': True,
                       'taught_q_rad': taught.tolist(), 'seed_q_rad': seed.tolist(),
                       'seed_clamp': {'applied': bool(clamps), 'joints': clamps,
                                      'reference_pose_changed_by_clamp': False,
                                      'policy': 'Clamp search seed only; target remains original taught FK'},
                       'effective_joint_bounds_rad': np.column_stack((lower, upper)).tolist(),
                       'supplied_joint_limits_rad': limits.tolist(), 'requested_lift_m': float(lift),
                       'lift_frame': 'base', 'lift_axis': 'positive_z',
                       'tolerances': {'position_m': POSITION_TOLERANCE_M,
                                      'rotation_rad': ROTATION_TOLERANCE_RAD,
                                      'maximum_delta_from_taught_rad': float(caps.max()),
                                      'maximum_delta_from_taught_per_joint_rad': caps.tolist(),
                                      'joint_margin_rad': float(margin)}})
        stage = 'fk_failure'
        original = _pose(fk, taught)
        target_position = original[:3]+[0., 0., lift]
        target_rotation = _rotation(original)
        report.update({'taught_fk_flange_m_rad': original.tolist(),
                       'target_flange_position_m': target_position.tolist(),
                       'target_flange_rotation_matrix': target_rotation.tolist(),
                       'target_reference': 'original_operator_taught_flange_not_clamped_seed'})
        if current is not None:
            report['current_reference'] = {'q_rad': current.tolist(), 'fk_flange_m_rad': _pose(fk, current).tolist(),
                                           'used_as_IK_seed': False, 'path_to_candidate_validated': False}
        q, after = seed.copy(), _pose(fk, seed)
        rotation_gain = rotation_length/ROTATION_LENGTH_M

        def merit_error(pose):
            value = _error(pose, target_position, target_rotation)
            if rotation_gain != 1.:
                value[3:] *= rotation_gain
            return value

        converged = False
        termination = None
        for iteration in range(max_iterations+1):
            report['iterations'] = iteration
            error = merit_error(after)
            stage = 'rank_insufficient'
            jacobian, singular = _jacobian(fk, q, lower, upper, increment)
            if rotation_gain != 1.:
                jacobian[3:] *= rotation_gain
                singular = np.linalg.svd(jacobian, compute_uv=False)
                if singular[-1] <= 1e-6 or singular[0]/singular[-1] > 1e6:
                    raise ArithmeticError('Weighted six-dimensional FK Jacobian is rank deficient or ill-conditioned')
            report['jacobian_shape'] = list(jacobian.shape)
            report['jacobian_singular_values'] = singular.tolist()
            stage = 'unreachable_within_bounds'
            if (np.linalg.norm(error[:3]) <= POSITION_TOLERANCE_M
                    and np.linalg.norm(error[3:])/rotation_length <= ROTATION_TOLERANCE_RAD):
                converged = True
                termination = 'pose_tolerances_met'
                break
            if iteration == max_iterations:
                termination = 'iteration_limit'
                break
            step = _bounded_step(jacobian, error, q, lower, upper, damp)
            improved = False
            for exponent in range(12):
                trial = np.clip(q+step*(.5**exponent), lower, upper)
                stage = 'fk_failure'
                trial_pose = _pose(fk, trial)
                trial_error = merit_error(trial_pose)
                stage = 'unreachable_within_bounds'
                if np.linalg.norm(trial_error) < np.linalg.norm(error)-1e-12:
                    q, after, improved = trial, trial_pose, True
                    break
            if not improved:
                termination = 'weighted_objective_stalled'
                break
        position_error = float(np.linalg.norm(after[:3]-target_position))
        rotation_error = float(np.linalg.norm(_rotation_vector(_rotation(after) @ target_rotation.T)))
        # The search box was constructed from taught +/- caps and intersected
        # with exact soft limits. Subtracting a nonzero taught joint again can
        # round a legal box endpoint to cap+1 ULP. Test the same bounds that
        # constrain all probes instead of expanding limits with an epsilon.
        box_verified = bool(np.all(q >= lower) and np.all(q <= upper))
        raw_delta = q-taught
        checks = {'position_target': position_error <= POSITION_TOLERANCE_M,
                  'taught_rotation_retained': rotation_error <= ROTATION_TOLERANCE_RAD,
                  'joint_delta_from_original_taught': box_verified,
                  'joint_soft_limits_with_margin': bool(np.all(q >= soft_lower) and np.all(q <= soft_upper)),
                  'positive_lift_range': bool(MIN_LIFT_M <= lift <= lift_ceiling),
                  'full_six_dimensional_Jacobian': jacobian.shape == (6, 7)}
        report.update({'predicted_fk_flange_m_rad': after.tolist(),
                       'predicted_displacement_from_taught_m': (after[:3]-original[:3]).tolist(),
                       'position_error_m': position_error, 'rotation_error_rad': rotation_error,
                       'constraint_checks': checks,
                       'solver_termination': termination,
                       'joint_bound_diagnostics': {
                           'basis': 'exact_effective_joint_bounds_without_epsilon_expansion',
                           'exact_effective_box_verified': box_verified,
                           'epsilon_expansion_rad': 0.,
                           'raw_delta_from_taught_q_rad': raw_delta.tolist(),
                           'configured_delta_caps_rad': caps.tolist(),
                           'raw_subtraction_cap_excess_rad': (np.abs(raw_delta)-caps).tolist(),
                           'raw_subtraction_cap_check': bool(np.all(np.abs(raw_delta) <= caps)),
                           'slack_to_lower_bound_rad': (q-lower).tolist(),
                           'slack_to_upper_bound_rad': (upper-q).tolist(),
                           'diagnostics_are_motion_targets': False}})
        if not converged or not all(checks.values()):
            raise ArithmeticError('Solver did not find a lifted taught endpoint satisfying all bounded pose constraints')
        report.update({'valid': True, 'model_candidate_valid': True,
                       'candidate_q_rad': q.tolist(), 'delta_from_taught_q_rad': (q-taught).tolist()})
        if current is not None:
            report['current_reference']['endpoint_joint_difference_to_candidate_rad'] = (q-current).tolist()
    except (ValueError, TypeError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as exc:
        report['error'] = {'code': 'fk_failure' if isinstance(exc, FKFailure) else stage,
                           'message': type(exc).__name__+': '+str(exc)}
    return report
