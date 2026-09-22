"""Green-cup IK and full sampled path checks, with soft wrist preferences."""

import math
import inspect
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation
from cup_grasp_demo.flow.shake import Kinematics
from cup_grasp_demo.flow.core import Screen

_ROTATION_ASSUME_VALID = 'assume_valid' in inspect.signature(Rotation.from_matrix).parameters


def proper_rotation_vector(matrix):
    """Log of an already validated proper rotation; stable fallback near pi."""
    vector = .5 * np.array([matrix[2, 1]-matrix[1, 2],
                             matrix[0, 2]-matrix[2, 0], matrix[1, 0]-matrix[0, 1]])
    sine = math.sqrt(float(vector @ vector))
    cosine = .5 * (float(matrix[0, 0]+matrix[1, 1]+matrix[2, 2])-1.)
    angle = math.atan2(sine, cosine)
    if math.pi - angle < 1e-4:
        return Rotation.from_matrix(matrix).as_rotvec()
    return vector if sine < 1e-8 else (angle / sine) * vector


def batched_rotation_forward(kin, q):
    """Same FK/Jacobian, with one scipy rotation conversion for all seven axes."""
    rotations = Rotation.from_rotvec(np.asarray(kin.axes) * np.asarray(q)[:, None]).as_matrix()
    transform = np.eye(4)
    pivots, axes = [], []
    for origin, axis, rotation in zip(kin.origins, kin.axes, rotations):
        transform = transform @ origin
        pivots.append(transform[:3, 3].copy())
        axes.append(transform[:3, :3] @ axis)
        transform[:3, :3] = transform[:3, :3] @ rotation
    axes, pivots = np.asarray(axes), np.asarray(pivots)
    return transform, np.concatenate((np.cross(axes, transform[:3, 3] - pivots).T, axes.T))


def solve(target, seed, wrist_deg, *, fast_fk=False, method='trf'):
    if method not in ('trf', 'dogbox'):
        raise ValueError('IK method must be trf or dogbox')
    kin = Kinematics()
    forward = (lambda q: batched_rotation_forward(kin, q)) if fast_fk else kin.forward
    reference = np.radians(wrist_deg)
    rotation = np.asarray(target[:3, :3])
    # FK rotations are products of proper rotations. Validate the fixed target
    # once; avoid repeating scipy's orthogonality projection on every iterate.
    rotation_options = ({'assume_valid': True} if fast_fk and _ROTATION_ASSUME_VALID
                        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0)
                        and abs(np.linalg.det(rotation) - 1.) < 1e-12 else {})
    wrist_derivative = np.c_[np.zeros((3, 4)), .001 * np.eye(3)]

    # Reuse FK between scipy's residual and Jacobian calls at the same q.
    cached = [None, None, None]

    def evaluate(q):
        if cached[0] is not None and np.array_equal(q, cached[0]):
            return cached[1], cached[2]
        pose, geometric = forward(q)
        error_rotation = rotation @ pose[:3, :3].T
        phi = (proper_rotation_vector(error_rotation) if rotation_options else
               Rotation.from_matrix(error_rotation).as_rotvec())
        x, y, z = phi
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        theta = np.linalg.norm(phi)
        coefficient = (1 / 12 + theta**2 / 720 if theta < 1e-4
                       else (1 - .5 * theta / np.tan(.5 * theta)) / theta**2)
        right_inverse = np.eye(3) + .5 * skew + coefficient * (skew @ skew)
        if fast_fk:
            # Allocate independent results for scipy, without rebuilding the
            # concatenation descriptors and constant wrist blocks each time.
            value = np.empty(9)
            value[:3] = 10 * (pose[:3, 3] - target[:3, 3])
            value[3:6] = phi
            value[6:] = .001 * (q[4:] - reference)
            derivative = np.empty((9, 7))
            derivative[:3] = 10 * geometric[:3]
            derivative[3:6] = -right_inverse @ geometric[3:]
            derivative[6:] = wrist_derivative
        else:
            value = np.r_[10 * (pose[:3, 3] - target[:3, 3]), phi,
                          .001 * (q[4:] - reference)]
            derivative = np.vstack((10 * geometric[:3], -right_inverse @ geometric[3:],
                                    wrist_derivative))
        cached[:] = [q.copy(), value, derivative]
        return value, derivative

    def residual(q):
        return evaluate(q)[0]

    def jacobian(q):
        return evaluate(q)[1]

    lower = kin.lower + math.radians(1)
    upper = kin.upper - math.radians(1)
    starts = [np.asarray(seed).copy()]
    preferred = starts[0].copy()
    preferred[4:] = reference
    starts.append(preferred)
    solutions = []
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            jac=jacobian,
            method=method,
            max_nfev=150,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        pose, _ = kin.forward(result.x)
        position = np.linalg.norm(pose[:3, 3] - target[:3, 3])
        angle = np.linalg.norm(
            Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec()
        )
        if position <= 0.0015 and angle <= math.radians(1):
            solutions.append(
                (
                    np.linalg.norm(result.x[4:] - reference)
                    + 0.1 * np.linalg.norm(result.x - seed),
                    result.x,
                )
            )
    if not solutions:
        if method == 'dogbox':
            return solve(target, seed, wrist_deg, fast_fk=fast_fk, method='trf')
        raise ValueError("绿色杯目标 IK 无合格解；保留当前姿态")
    return min(solutions, key=lambda item: item[0])[1]


def held_cup_clearance(qs, T_flange_cup, radius, height, scene):
    poses = Kinematics().forward_batch(qs) @ T_flange_cup
    n = np.asarray(scene["cup_normal_base"])
    p = np.asarray(scene["cup_support_base_m"])
    alignment = poses[:, :3, 2] @ n
    support = radius * np.sqrt(np.maximum(0, 1 - alignment**2)) + height / 2 * abs(
        alignment
    )
    return float(np.min((poses[:, :3, 3] - p) @ n - support)) * 1000


def arm_plan(start, targets, scene, cfg, *, held=None, cup_margin_mm=0):
    kin = Kinematics()
    stages = []
    samples = []
    q = np.asarray(start)
    for i, target in enumerate(targets):
        target = np.asarray(target)
        checked = kin.model.check_joint_path(q.tolist(), target.tolist())
        if not checked.kinematic_checks_passed:
            raise ValueError("关节路径检查未通过")
        samples.extend(checked.samples_rad)
        stages.append(
            dict(
                name=f"green_move_{i}",
                current_q_rad=q.tolist(),
                target_q_rad=target.tolist(),
            )
        )
        q = target
    review = Screen(table_only=True).check_table_batch(samples, scene, cfg)
    if held is not None:
        gap = held_cup_clearance(
            samples, held["T_flange_cup"], held["radius_m"], held["height_m"], scene
        )
        review["held_cup_min_mm"] = gap
        if gap < cup_margin_mm:
            review["blockers"].append(f"持杯桌面间隙不足：{gap:.1f} mm")
    return dict(
        kind="green_arm_plan",
        start_q_rad=list(start),
        stages=stages,
        blockers=review["blockers"],
        table_screen=review,
        replan_context=dict(
            held=None if held is None else dict(held, T_flange_cup=np.asarray(held['T_flange_cup']).tolist()),
            cup_margin_mm=cup_margin_mm),
    )


def minimize_joint_travel(start, candidate, target):
    """Redistribute redundant motion without changing the Cartesian endpoint.

    Equal per-axis acceleration budgets make the largest angular displacement
    the rest-to-rest timing bottleneck. The normal solver remains the fallback;
    arm_plan still checks the resulting joint path and held-cup clearance.
    """
    kin = Kinematics()
    start, candidate = np.asarray(start), np.asarray(candidate)
    lower, upper = kin.lower + math.radians(1), kin.upper - math.radians(1)
    def equality(x):
        pose = batched_rotation_forward(kin, x[:7])[0]
        return np.r_[10 * (pose[:3, 3] - target[:3, 3]),
                     Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec()]
    def travel(x):
        delta = x[:7] - start
        return np.r_[x[7] - delta, x[7] + delta]
    original = float(np.max(abs(candidate-start)))
    result = minimize(lambda x: x[7], np.r_[candidate, original], method='SLSQP',
                      bounds=list(zip(lower, upper)) + [(0., max(original, 1e-6))],
                      constraints=[dict(type='eq', fun=equality), dict(type='ineq', fun=travel)],
                      options=dict(ftol=1e-10, maxiter=60))
    if (result.success and np.all(np.isfinite(result.x))
            and np.max(abs(equality(result.x))) < 1e-6
            and np.all(result.x[:7] >= lower) and np.all(result.x[:7] <= upper)
            and np.max(abs(result.x[:7]-start)) < original):
        return result.x[:7]
    return candidate


def vertical_targets(start, tcp, dz, wrist_deg, *, single_target=False, fast_fk=False, minimize_travel=False):
    kin = Kinematics()
    flange, _ = kin.forward(start)
    pose = flange @ tcp
    targets = []
    q = np.asarray(start)
    distances = [dz] if single_target else np.linspace(0, dz, max(2, math.ceil(abs(dz) / 0.01) + 1))[1:]
    for distance in distances:
        target = pose.copy()
        target[2, 3] += distance
        flange_target = target @ np.linalg.inv(tcp)
        previous = q
        q = solve(flange_target, q, wrist_deg, **({'fast_fk': True} if fast_fk else {}))
        if minimize_travel:
            q = minimize_joint_travel(previous, q, flange_target)
        targets.append(q.tolist())
    return targets
