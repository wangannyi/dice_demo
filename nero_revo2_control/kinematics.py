"""Offline Nero seven-axis kinematics from the official URDF joint chain.

All transforms are base_link -> link7 (the arm flange). This module reads only a
local URDF and uses the Python standard library. It never opens CAN, the SDK,
or a camera. Joint-path checks here cover limits and interpolation only; they
cannot establish clearance from the hand, table, cup, or other obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, ceil, cos, isfinite, pi, radians, sin, sqrt
from pathlib import Path
from typing import Sequence
from xml.etree import ElementTree


Matrix4 = tuple[tuple[float, float, float, float], ...]
Vector7 = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]
    lower_rad: float
    upper_rad: float


@dataclass(frozen=True)
class IKResult:
    success: bool
    joints_rad: Vector7
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    reason: str


@dataclass(frozen=True)
class JointPathCheck:
    samples_rad: tuple[Vector7, ...]
    joint_limits_passed: bool
    continuity_passed: bool
    singularity_warning: bool
    collision_verified: bool
    scene_collision_verified: bool
    reason: str

    @property
    def kinematic_checks_passed(self) -> bool:
        return self.joint_limits_passed and self.continuity_passed

    @property
    def sample_count(self) -> int:
        return len(self.samples_rad)

    @property
    def joint_limits_ok(self) -> bool:
        return self.joint_limits_passed

    @property
    def continuity_ok(self) -> bool:
        return self.continuity_passed


def _identity() -> Matrix4:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _matmul(a: Matrix4, b: Matrix4) -> Matrix4:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def _rotation_transform(
    r: Sequence[Sequence[float]], p: Sequence[float] = (0.0, 0.0, 0.0)
) -> Matrix4:
    return tuple(
        tuple((r[i][j] if j < 3 else p[i]) for j in range(4)) for i in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def _rpy_transform(xyz: Sequence[float], rpy: Sequence[float]) -> Matrix4:
    roll, pitch, yaw = rpy
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    # URDF fixed-axis roll/pitch/yaw: Rz(yaw) Ry(pitch) Rx(roll).
    r = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return _rotation_transform(r, xyz)


def _axis_rotation(axis: Sequence[float], angle: float) -> Matrix4:
    x, y, z = axis
    c, s = cos(angle), sin(angle)
    v = 1.0 - c
    r = (
        (c + x * x * v, x * y * v - z * s, x * z * v + y * s),
        (y * x * v + z * s, c + y * y * v, y * z * v - x * s),
        (z * x * v - y * s, z * y * v + x * s, c + z * z * v),
    )
    return _rotation_transform(r)


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Sequence[float]) -> float:
    return sqrt(sum(x * x for x in v))


def _rotation_error(target: Matrix4, current: Matrix4) -> tuple[float, float, float]:
    """World-frame SO(3) log of R_target R_current^T."""
    r = [
        [sum(target[i][k] * current[j][k] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    c = max(-1.0, min(1.0, (r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0))
    theta = acos(c)
    skew = (r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1])
    if theta < 1e-6:
        return tuple(v / 2.0 for v in skew)
    if pi - theta < 1e-5:
        # Robust axis for half-turns; the solver still treats these as a large
        # jump rather than declaring success at an unstable orientation.
        diag = [sqrt(max(0.0, (r[i][i] + 1.0) / 2.0)) for i in range(3)]
        i = max(range(3), key=lambda k: diag[k])
        axis = [0.0, 0.0, 0.0]
        axis[i] = diag[i]
        if axis[i] > 1e-9:
            for j in range(3):
                if j != i:
                    axis[j] = (r[i][j] + r[j][i]) / (4.0 * axis[i])
        if sum(axis[k] * skew[k] for k in range(3)) < 0:
            axis = [-v for v in axis]
        return tuple(theta * v for v in axis)
    scale = theta / (2.0 * sin(theta))
    return tuple(scale * v for v in skew)


def _solve_linear(
    a: Sequence[Sequence[float]], b: Sequence[float]
) -> list[float] | None:
    """Partial-pivot elimination for the six-dimensional DLS normal matrix."""
    n = len(b)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        row = max(range(col, n), key=lambda i: abs(m[i][col]))
        if abs(m[row][col]) < 1e-14:
            return None
        m[col], m[row] = m[row], m[col]
        pivot = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= pivot
        for i in range(n):
            if i == col:
                continue
            factor = m[i][col]
            for j in range(col, n + 1):
                m[i][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]


def _parse_vector(raw: str, count: int = 3) -> tuple[float, ...]:
    values = tuple(float(item) for item in raw.split())
    if len(values) != count:
        raise ValueError(f"expected {count} components, got {raw!r}")
    return values


class NeroModel:
    """Seven URDF revolute joints from base_link through link7."""

    base_frame = "base_link"
    flange_frame = "link7"

    def __init__(self, joints: Sequence[Joint], source: Path):
        if len(joints) != 7:
            raise ValueError(f"Nero model requires seven joints, got {len(joints)}")
        for i, joint in enumerate(joints):
            if joint.name != f"joint{i + 1}":
                raise ValueError(f"unexpected joint name {joint.name!r}")
            expected_parent = "base_link" if i == 0 else f"link{i}"
            if joint.parent != expected_parent or joint.child != f"link{i + 1}":
                raise ValueError(f"unexpected Nero chain at {joint.name}")
        self.joints = tuple(joints)
        self.source = source
        self.limits_rad = tuple((j.lower_rad, j.upper_rad) for j in joints)

    def _check_q(self, q_rad: Sequence[float]) -> Vector7:
        if len(q_rad) != 7:
            raise ValueError("Nero requires seven joint angles in radians")
        q = tuple(float(v) for v in q_rad)
        if any(not isfinite(v) for v in q):
            raise ValueError("joint angles must be finite")
        return q  # type: ignore[return-value]

    def within_limits(self, q_rad: Sequence[float], margin_rad: float = 0.0) -> bool:
        q = self._check_q(q_rad)
        if margin_rad < 0.0:
            raise ValueError("margin_rad must be nonnegative")
        return all(
            lo + margin_rad <= angle <= hi - margin_rad
            for angle, (lo, hi) in zip(q, self.limits_rad)
        )

    def _forward(
        self, q_rad: Sequence[float]
    ) -> tuple[
        Matrix4,
        tuple[tuple[float, float, float], ...],
        tuple[tuple[float, float, float], ...],
    ]:
        q = self._check_q(q_rad)
        t = _identity()
        pivots = []
        axes = []
        for joint, angle in zip(self.joints, q):
            t = _matmul(t, _rpy_transform(joint.origin_xyz, joint.origin_rpy))
            pivots.append((t[0][3], t[1][3], t[2][3]))
            axes.append(
                tuple(
                    sum(t[i][k] * joint.axis_xyz[k] for k in range(3)) for i in range(3)
                )
            )
            t = _matmul(t, _axis_rotation(joint.axis_xyz, angle))
        return t, tuple(pivots), tuple(axes)

    def fk(self, q_rad: Sequence[float]) -> Matrix4:
        """Return homogeneous base_link->link7 transform, meters/radians."""
        return self._forward(q_rad)[0]

    def ik(
        self,
        target_matrix: Sequence[Sequence[float]],
        seed_rad: Sequence[float],
        *,
        position_tolerance_m: float = 1e-4,
        orientation_tolerance_rad: float = 1e-3,
        max_iterations: int = 250,
        damping: float = 0.05,
        max_joint_step_rad: float = 0.16,
    ) -> IKResult:
        """Solve a 6D target locally with damped least squares and seven joint bounds.

        This is a local seeded solver. A failed result or one successful branch
        does not prove reachability, trajectory clearance, or grasp feasibility.
        """
        if len(target_matrix) != 4 or any(len(row) != 4 for row in target_matrix):
            raise ValueError("target_matrix must be 4x4")
        target = tuple(tuple(float(v) for v in row) for row in target_matrix)
        if any(not isfinite(v) for row in target for v in row):
            raise ValueError("target_matrix must be finite")
        if any(abs(target[3][j] - (1.0 if j == 3 else 0.0)) > 1e-8 for j in range(4)):
            raise ValueError("target_matrix must be homogeneous")
        columns = [tuple(target[i][j] for i in range(3)) for j in range(3)]
        if any(
            abs(
                sum(columns[i][k] * columns[j][k] for k in range(3))
                - (1.0 if i == j else 0.0)
            )
            > 1e-4
            for i in range(3)
            for j in range(3)
        ):
            raise ValueError("target rotation must be orthonormal")
        if (
            sum(_cross(columns[0], columns[1])[k] * columns[2][k] for k in range(3))
            < 0.9999
        ):
            raise ValueError("target rotation must be right-handed")
        if (
            position_tolerance_m <= 0
            or orientation_tolerance_rad <= 0
            or max_iterations < 1
        ):
            raise ValueError("tolerances and max_iterations must be positive")
        if damping <= 0 or max_joint_step_rad <= 0:
            raise ValueError("damping and max_joint_step_rad must be positive")
        q = list(self._check_q(seed_rad))
        if not self.within_limits(q):
            raise ValueError("IK seed is outside URDF joint limits")
        best_q = tuple(q)
        best_metric = float("inf")
        best_pos = float("inf")
        best_ori = float("inf")
        reason = "maximum_iterations"
        for iteration in range(max_iterations + 1):
            current, pivots, axes = self._forward(q)
            ep = tuple(target[i][3] - current[i][3] for i in range(3))
            er = _rotation_error(target, current)
            pos_error, ori_error = _norm(ep), _norm(er)
            metric = (
                pos_error / position_tolerance_m + ori_error / orientation_tolerance_rad
            )
            if metric < best_metric:
                best_metric, best_q, best_pos, best_ori = (
                    metric,
                    tuple(q),
                    pos_error,
                    ori_error,
                )
            if (
                pos_error <= position_tolerance_m
                and ori_error <= orientation_tolerance_rad
            ):
                return IKResult(
                    True, tuple(q), pos_error, ori_error, iteration, "converged"
                )
            if iteration == max_iterations:
                break
            end = (current[0][3], current[1][3], current[2][3])
            jac = []
            for pivot, axis in zip(pivots, axes):
                offset = tuple(end[i] - pivot[i] for i in range(3))
                jac.append(_cross(axis, offset) + axis)
            # Rows position xyz + angular xyz; orientation is in radians. No
            # stiffness command or robot motion is involved.
            jrows = [[jac[col][row] for col in range(7)] for row in range(6)]
            normal = [
                [
                    sum(jrows[i][k] * jrows[j][k] for k in range(7))
                    + (damping * damping if i == j else 0.0)
                    for j in range(6)
                ]
                for i in range(6)
            ]
            y = _solve_linear(normal, ep + er)
            if y is None:
                reason = "numerically_singular"
                break
            delta = [
                sum(jrows[row][col] * y[row] for row in range(6)) for col in range(7)
            ]
            largest = max(abs(v) for v in delta)
            if largest > max_joint_step_rad:
                delta = [v * max_joint_step_rad / largest for v in delta]
            accepted = False
            for backtrack in range(12):
                scale = 0.5**backtrack
                candidate = [
                    max(lo, min(hi, q[i] + scale * delta[i]))
                    for i, (lo, hi) in enumerate(self.limits_rad)
                ]
                candidate_t = self.fk(candidate)
                cand_ep = tuple(target[i][3] - candidate_t[i][3] for i in range(3))
                cand_er = _rotation_error(target, candidate_t)
                cand_metric = (
                    _norm(cand_ep) / position_tolerance_m
                    + _norm(cand_er) / orientation_tolerance_rad
                )
                if cand_metric < metric - 1e-8:
                    q = candidate
                    accepted = True
                    break
            if not accepted:
                reason = "stalled_or_joint_bound"
                break
        return IKResult(False, best_q, best_pos, best_ori, iteration, reason)  # type: ignore[arg-type]

    def check_joint_path(
        self,
        start_rad: Sequence[float],
        goal_rad: Sequence[float],
        *,
        max_sample_step_rad: float = radians(2.0),
        max_samples: int = 1000,
        singularity_window_rad: float = radians(3.0),
    ) -> JointPathCheck:
        """Sample a joint-linear path; check only joint bounds and continuity.

        `collision_verified` remains False even when these checks pass. The
        official URDF visual/collision meshes, mounted Revo2, and measured scene
        need separate validated geometry and a collision checker.
        """
        start, goal = self._check_q(start_rad), self._check_q(goal_rad)
        if max_sample_step_rad <= 0 or max_samples < 2 or singularity_window_rad < 0:
            raise ValueError("invalid path sampling settings")
        steps = max(
            1, ceil(max(abs(b - a) for a, b in zip(start, goal)) / max_sample_step_rad)
        )
        if steps + 1 > max_samples:
            return JointPathCheck(
                (), False, False, False, False, False, "max_samples_exceeded"
            )
        samples = tuple(
            tuple(a + (b - a) * i / steps for a, b in zip(start, goal))
            for i in range(steps + 1)
        )
        limits_pass = all(self.within_limits(q) for q in samples)
        continuity_pass = all(
            max(abs(b - a) for a, b in zip(samples[i], samples[i + 1]))
            <= max_sample_step_rad + 1e-12
            for i in range(steps)
        )
        singularity_warning = any(
            abs(q[1]) <= singularity_window_rad or abs(q[3]) <= singularity_window_rad
            for q in samples
        )
        reason = "joint_checks_passed_collision_unverified"
        if not limits_pass:
            reason = "joint_limit_violation"
        elif not continuity_pass:
            reason = "joint_step_violation"
        return JointPathCheck(
            samples,
            limits_pass,
            continuity_pass,
            singularity_warning,
            False,
            False,
            reason,
        )


def load_model(urdf_path: str | Path | None = None) -> NeroModel:
    """Load the bundled official Nero arm URDF's seven revolute joints."""
    source = (
        Path(urdf_path)
        if urdf_path is not None
        else Path(__file__).resolve().parent / "models" / "nero_description.urdf"
    )
    root = ElementTree.parse(source).getroot()
    joints = []
    for i in range(1, 8):
        node = root.find(f"./joint[@name='joint{i}']")
        if node is None or node.get("type") != "revolute":
            raise ValueError(f"missing revolute joint{i} in {source}")
        origin = node.find("origin")
        axis = node.find("axis")
        lower = node.find("limit")
        parent = node.find("parent")
        child = node.find("child")
        if any(item is None for item in (origin, axis, lower, parent, child)):
            raise ValueError(f"joint{i} lacks kinematic metadata")
        xyz = _parse_vector(origin.get("xyz", "0 0 0"))
        rpy = _parse_vector(origin.get("rpy", "0 0 0"))
        axis_xyz = _parse_vector(axis.get("xyz", "0 0 1"))
        axis_length = _norm(axis_xyz)
        if axis_length < 1e-12:
            raise ValueError(f"joint{i} axis is zero")
        axis_xyz = tuple(v / axis_length for v in axis_xyz)
        joints.append(
            Joint(
                f"joint{i}",
                parent.get("link", ""),
                child.get("link", ""),
                xyz,
                rpy,
                axis_xyz,
                float(lower.attrib["lower"]),
                float(lower.attrib["upper"]),
            )
        )
    return NeroModel(joints, source)
