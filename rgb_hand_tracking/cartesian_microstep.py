"""Offline bounded flange X/Y planner; imports never access SDK or hardware.

Injected FK accepts seven joint radians and returns [x,y,z,roll,pitch,yaw]
in metres/radians, with SDK Rz(yaw) Ry(pitch) Rx(roll) convention. A valid
plan predicts endpoint flange pose only. Preserving a contact point additionally
assumes an unchanged rigid attachment and finger posture, and needs hardware
verification. NumPy is the only non-standard runtime dependency.
"""
import math

import numpy as np


MAX_TRANSLATION_M = .001
MAX_JOINT_DELTA_RAD = math.radians(.35)
DEFAULT_JOINT_MARGIN_RAD = math.radians(.1)
Z_TOLERANCE_M = .0001
ORIENTATION_TOLERANCE_RAD = math.radians(.05)
XY_TOLERANCE_M = .00002
ROTATION_LENGTH_M = .1


class FKFailure(RuntimeError):
    """Invalid injected FK output or failure; no substitute pose is used."""


def _numbers(value, shape, label):
    raw = np.asarray(value, dtype=object)
    if raw.shape != shape or any(isinstance(v, (bool, np.bool_)) for v in raw.flat):
        raise ValueError('Invalid '+label)
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite '+label)
    return result


def _rotation(pose):
    r, p, y = pose[3:]
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                             math.sin(p), math.cos(y), math.sin(y))
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def _rotation_vector(rotation):
    angle = math.acos(float(np.clip((np.trace(rotation)-1)/2, -1., 1.)))
    skew = np.array([rotation[2, 1]-rotation[1, 2],
                     rotation[0, 2]-rotation[2, 0], rotation[1, 0]-rotation[0, 1]])
    if angle < 1e-7:
        return skew/2
    if math.pi-angle < 1e-5:
        _, vectors = np.linalg.eigh((rotation+rotation.T)/2)
        axis = vectors[:, -1]
        if axis @ skew < 0:
            axis = -axis
        return axis*angle
    return skew*(angle/(2*math.sin(angle)))


def _pose(fk, q):
    try:
        return _numbers(fk(q.tolist()), (6,), 'FK pose (m/rad)')
    except Exception as exc:
        raise FKFailure(type(exc).__name__+': '+str(exc)) from exc


def _error(pose, target_position, target_rotation):
    return np.concatenate((target_position-pose[:3], ROTATION_LENGTH_M*
                           _rotation_vector(target_rotation @ _rotation(pose).T)))


def _jacobian(fk, q, lower, upper, increment):
    columns = []
    for index in range(7):
        lo, hi = q.copy(), q.copy()
        lo[index] = max(lower[index], q[index]-increment)
        hi[index] = min(upper[index], q[index]+increment)
        width = hi[index]-lo[index]
        if width <= 1e-12:
            columns.append(np.zeros(6))
            continue
        a, b = _pose(fk, lo), _pose(fk, hi)
        columns.append(np.concatenate((b[:3]-a[:3], ROTATION_LENGTH_M*
                                       _rotation_vector(_rotation(b) @ _rotation(a).T)))/width)
    jacobian = np.column_stack(columns)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    if singular[-1] <= 1e-6 or singular[0]/singular[-1] > 1e6:
        raise ArithmeticError('Six-dimensional FK Jacobian is rank deficient or ill-conditioned')
    return jacobian, singular


def _bounded_step(jacobian, error, q, lower, upper, damping):
    """Damped active-set solve; every proposed joint remains in the full box."""
    step, free = np.zeros(7), np.ones(7, dtype=bool)
    for _ in range(8):
        if not free.any():
            break
        j = jacobian[:, free]
        residual = error-jacobian[:, ~free] @ step[~free]
        step[free] = j.T @ np.linalg.solve(j @ j.T+damping**2*np.eye(6), residual)
        clipped = np.clip(q+step, lower, upper)-q
        violated = free & (np.abs(step-clipped) > 1e-12)
        step[violated] = clipped[violated]
        if not violated.any():
            break
        free[violated] = False
    return np.clip(q+step, lower, upper)-q


def plan_microstep(fk, q_rad, joint_limits_rad, *, axis, distance_m,
                   joint_margin_rad=DEFAULT_JOINT_MARGIN_RAD,
                   max_joint_delta_rad=MAX_JOINT_DELTA_RAD,
                   finite_difference_rad=1e-5, damping=1e-5, max_iterations=40):
    """Return a JSON-compatible plan; invalid plans never publish target joints.

    Axis is base-frame 'x' or 'y'; 0 < abs(distance_m) <= 1 mm. Defaults:
    per-joint delta <= 0.35 degrees, soft-limit margin 0.1 degrees, Z error
    <= 0.1 mm, rotation geodesic error <= 0.05 degrees, XY target error
    <= min(0.02 mm, 2% of requested distance). This conservative planner
    requires all six flange task directions to be locally observable.
    Caller owns FK purity, SDK feedback freshness, control and verification.
    """
    report = {'schema': 'offline_cartesian_microstep_v1', 'valid': False,
              'target_q_rad': None, 'error': None, 'iterations': 0,
              'scope': 'FK prediction only; no device access or motion commands',
              'contact_point_experimentally_verified': False}
    stage = 'invalid_input'
    try:
        if not callable(fk) or axis not in ('x', 'y'):
            raise ValueError('Expected callable FK and base-frame axis x or y')
        q0 = _numbers(q_rad, (7,), 'seven joints')
        limits = _numbers(joint_limits_rad, (7, 2), 'seven joint limits')
        distance, margin, cap, increment, damp = _numbers(
            [distance_m, joint_margin_rad, max_joint_delta_rad, finite_difference_rad, damping],
            (5,), 'planner parameters')
        if not 0 < abs(distance) <= MAX_TRANSLATION_M:
            raise ValueError('Translation must be nonzero and at most 1 mm')
        if (margin < 0 or not 0 < cap <= MAX_JOINT_DELTA_RAD
                or not 0 < increment <= 1e-3 or not 0 < damp <= .01
                or isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
                or not 1 <= max_iterations <= 100):
            raise ValueError('Invalid margin, joint cap, finite difference, damping or iteration bound')
        low_soft, high_soft = limits[:, 0]+margin, limits[:, 1]-margin
        if (np.any(limits[:, 0] >= limits[:, 1]) or np.any(low_soft >= high_soft)
                or np.any(q0 < low_soft) or np.any(q0 > high_soft)):
            raise ValueError('Starting joints or margin violate supplied soft limits')
        lower, upper = np.maximum(low_soft, q0-cap), np.minimum(high_soft, q0+cap)
        stage = 'fk_failure'
        before = _pose(fk, q0)
        target_position, target_rotation = before[:3].copy(), _rotation(before)
        target_position[0 if axis == 'x' else 1] += distance
        report.update({'axis': axis, 'requested_distance_m': float(distance),
                       'q_before_rad': q0.tolist(), 'fk_before_m_rad': before.tolist(),
                       'target_position_m': target_position.tolist(),
                       'effective_joint_bounds_rad': np.column_stack((lower, upper)).tolist(),
                       'tolerances': {'xy_m': min(XY_TOLERANCE_M, abs(float(distance))*.02),
                                      'z_m': Z_TOLERANCE_M,
                                      'orientation_rad': ORIENTATION_TOLERANCE_RAD,
                                      'joint_delta_rad': float(cap),
                                      'joint_margin_rad': float(margin)}})
        q, after = q0.copy(), before.copy()
        converged = False
        for iteration in range(max_iterations+1):
            report['iterations'] = iteration
            error = _error(after, target_position, target_rotation)
            stage = 'rank_insufficient'
            jacobian, singular = _jacobian(fk, q, lower, upper, increment)
            report['jacobian_singular_values'] = singular.tolist()
            stage = 'unreachable_within_bounds'
            if (np.linalg.norm(error[:2]) <= report['tolerances']['xy_m']
                    and abs(error[2]) <= Z_TOLERANCE_M
                    and np.linalg.norm(error[3:])/ROTATION_LENGTH_M <= ORIENTATION_TOLERANCE_RAD):
                converged = True
                break
            if iteration == max_iterations:
                break
            step = _bounded_step(jacobian, error, q, lower, upper, damp)
            improved = False
            for exponent in range(12):
                candidate = np.clip(q+step*(.5**exponent), lower, upper)
                stage = 'fk_failure'
                candidate_pose = _pose(fk, candidate)
                candidate_error = _error(candidate_pose, target_position, target_rotation)
                stage = 'unreachable_within_bounds'
                if np.linalg.norm(candidate_error) < np.linalg.norm(error)-1e-12:
                    q, after, improved = candidate, candidate_pose, True
                    break
            if not improved:
                break
        displacement = after[:3]-before[:3]
        orientation_error = float(np.linalg.norm(_rotation_vector(_rotation(after) @ target_rotation.T)))
        checks = {'xy_target': float(np.linalg.norm(after[:2]-target_position[:2]))
                  <= report['tolerances']['xy_m'], 'z_retained': abs(float(displacement[2]))
                  <= Z_TOLERANCE_M, 'orientation_retained': orientation_error
                  <= ORIENTATION_TOLERANCE_RAD, 'joint_delta': bool(np.all(np.abs(q-q0) <= cap+1e-12)),
                  'joint_soft_limits_with_margin': bool(np.all(q >= low_soft) and np.all(q <= high_soft))}
        report.update({'candidate_q_rad': q.tolist(), 'fk_after_m_rad': after.tolist(),
                       'delta_q_rad': (q-q0).tolist(), 'predicted_displacement_m': displacement.tolist(),
                       'orientation_error_rad': orientation_error, 'constraint_checks': checks})
        if not converged or not all(checks.values()):
            raise ArithmeticError('Requested displacement cannot be reached within bounded pose constraints')
        report['valid'], report['target_q_rad'] = True, q.tolist()
    except (ValueError, TypeError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as exc:
        report['error'] = {'code': 'fk_failure' if isinstance(exc, FKFailure) else stage,
                           'message': type(exc).__name__+': '+str(exc)}
    return report
