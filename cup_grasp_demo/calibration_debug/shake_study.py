"""Read-only comparison of fixed pose, planar yaw freedom and J1/J4/J7 motion."""

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from cup_grasp_demo.calibration_debug.shake import Kinematics, read_limits, profile
from cup_grasp_demo.calibration_debug.parameters import shake_options, number


def tcp_state(kin, q, tcp):
    flange, jac = kin.forward(q)
    pose = flange @ tcp
    offset = pose[:3, 3] - flange[:3, 3]
    linear = jac[:3] + np.cross(jac[3:].T, offset).T
    roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")
    if abs(math.cos(pitch)) < 0.05:
        raise ValueError("TCP RPY near gimbal lock; choose another reference frame")
    # R = Rz(yaw) Ry(pitch) Rx(roll); spatial angular velocity -> RPY rates.
    e = np.array(
        [
            [math.cos(yaw) * math.cos(pitch), -math.sin(yaw), 0],
            [math.sin(yaw) * math.cos(pitch), math.cos(yaw), 0],
            [-math.sin(pitch), 0, 1],
        ]
    )
    rates = np.linalg.solve(e, jac[3:])
    return pose, np.r_[linear, rates], np.array([roll, pitch, yaw])


def wrap(values):
    return (values + np.pi) % (2 * np.pi) - np.pi


def motion_cost(cfg):
    raw = cfg.get("shake_study", {})
    if not isinstance(raw, dict) or set(raw) - {"joint_motion_cost"}:
        raise ValueError("Unknown shake_study fields")
    values = raw.get("joint_motion_cost", [1, 4, 4, 1, 4, 4, 1])
    if not isinstance(values, list) or len(values) != 7:
        raise ValueError("shake_study.joint_motion_cost requires seven costs")
    return np.array([number(v, "joint_motion_cost", 0.1, 100) for v in values])


def solve_planar(
    kin, seed, tcp, initial, rpy0, direction, distance, selected, cost=None
):
    q = np.asarray(seed).copy()
    scale = np.sqrt(
        np.ones(len(selected)) if cost is None else np.asarray(cost)[selected]
    )
    for _ in range(40):
        pose, jac, rpy = tcp_state(kin, q, tcp)
        error = np.r_[
            distance - np.dot(direction, pose[:3, 3] - initial[:3, 3]),
            initial[2, 3] - pose[2, 3],
            0.15 * wrap(rpy0[:2] - rpy[:2]),
        ]
        if np.max(np.abs(error)) < 1e-8:
            return q
        task = np.vstack([direction @ jac[:3], jac[2], 0.15 * jac[3:5]])[:, selected]
        step = np.linalg.lstsq(task / scale, error, rcond=1e-8)[0] / scale
        step *= min(1.0, 0.04 / max(np.max(np.abs(step)), 1e-12))
        q[selected] += step
        if np.any(q < kin.lower) or np.any(q > kin.upper):
            break
    raise ValueError("当前分支无法同时满足平面进度及 z/rx/ry 约束")


def compare(feedback, cfg, tcp):
    opts = shake_options(cfg)
    cost = motion_cost(cfg)
    utilization = opts.get("limit_utilization", 0.8)
    (lo, hi, vel, acc), _ = read_limits(feedback)
    q0 = np.asarray(feedback["q_after_rad"])
    kin = Kinematics()
    kin.lower, kin.upper = (
        np.maximum(kin.lower, lo) + math.radians(1),
        np.minimum(kin.upper, hi) - math.radians(1),
    )
    tcp = np.asarray(tcp)
    initial, jac, rpy0 = tcp_state(kin, q0, tcp)
    angle = math.radians(opts["azimuth_deg"])
    direction = np.array([math.cos(angle), math.sin(angle), 0.0])
    selected = [0, 3, 6]
    constraint = np.vstack([jac[2], 0.15 * jac[3:5]])[:, selected]
    singular = np.linalg.svd(constraint, compute_uv=False)
    rank = int(np.sum(singular > 1e-7))
    report = dict(
        kind="shake_constraint_study",
        execution_enabled=False,
        parameters=opts,
        joint_motion_cost=cost.tolist(),
        start_q_deg=np.rad2deg(q0).tolist(),
        T_flange_virtual_tcp=tcp.tolist(),
        reference="配置中的刚性虚拟 TCP；闭指后不等同于实际食指接触点",
        axes="基座坐标 XYZ，R=Rz(rz) Ry(ry) Rx(rx)；固定 z/rx/ry",
        joint147_constraint_rank=rank,
        joint147_constraint_singular_values=singular.tolist(),
        joint147_local_nullity=3 - rank,
        strategies=[],
    )
    amplitude = opts["amplitude_mm"] / 1000
    distances = np.linspace(0, amplitude, math.ceil(amplitude / 0.001) + 1)
    times = np.linspace(0, opts["duration_s"], math.ceil(opts["duration_s"] * 1000) + 1)
    ds, dv, da = profile(times, opts)
    for mode in (
        "fixed_pose_7j",
        "planar_yaw_free_7j",
        "planar_yaw_free_j147_primary",
        "planar_yaw_free_j147",
    ):
        result = dict(strategy=mode, kinematics_passed=False, blockers=[])
        report["strategies"].append(result)
        try:
            if np.any(q0 < kin.lower) or np.any(q0 > kin.upper):
                raise ValueError("起点距关节限位不足 1°")
            if mode.endswith("j147") and rank == 3:
                raise ValueError(
                    "J1/J4/J7 的 z/rx/ry 约束雅可比满秩；局部没有非零连续运动自由度"
                )
            branches = {}
            for sign in (-1, 1):
                q, rows = q0.copy(), [q0.copy()]
                for d in distances[1:]:
                    if mode == "fixed_pose_7j":
                        target = initial.copy()
                        target[:3, 3] += direction * sign * d
                        q = kin.solve(q, target @ np.linalg.inv(tcp))
                    else:
                        q = solve_planar(
                            kin,
                            q,
                            tcp,
                            initial,
                            rpy0,
                            direction,
                            sign * d,
                            selected if mode.endswith("j147") else list(range(7)),
                            cost if mode.endswith("j147_primary") else None,
                        )
                    rows.append(q.copy())
                branches[sign] = rows
            knots = np.r_[-distances[1:][::-1], distances]
            spline = CubicSpline(
                knots, np.r_[branches[-1][1:][::-1], branches[1]], axis=0
            )
            values = spline(ds)
            if np.any(values < kin.lower) or np.any(values > kin.upper):
                raise ValueError("插值路径超出关节范围")
            v = np.max(np.abs(spline(ds, 1) * dv[:, None]), axis=0)
            a = np.max(
                np.abs(spline(ds, 2) * dv[:, None] ** 2 + spline(ds, 1) * da[:, None]),
                axis=0,
            )
            peaks = dict(
                z_mm=0.0,
                rx_deg=0.0,
                ry_deg=0.0,
                rz_deg=0.0,
                perpendicular_xy_mm=0.0,
                progress_mm=0.0,
            )
            for d in np.linspace(-amplitude, amplitude, 201):
                pose, _, rpy = tcp_state(kin, spline(d), tcp)
                delta = pose[:3, 3] - initial[:3, 3]
                angles = np.rad2deg(wrap(rpy - rpy0))
                errors = dict(
                    z_mm=abs(delta[2]) * 1000,
                    rx_deg=abs(angles[0]),
                    ry_deg=abs(angles[1]),
                    rz_deg=abs(angles[2]),
                    perpendicular_xy_mm=abs(
                        np.dot([-direction[1], direction[0], 0], delta)
                    )
                    * 1000,
                    progress_mm=abs(np.dot(direction, delta) - d) * 1000,
                )
                peaks = {k: max(peaks[k], errors[k]) for k in peaks}
            if (
                max(peaks["z_mm"], peaks["progress_mm"]) > 0.02
                or max(peaks["rx_deg"], peaks["ry_deg"]) > 0.02
            ):
                raise ValueError("插值后 z/rx/ry 或平面进度约束超差")
            minimum_scale = max(
                float(np.max(v / (vel * utilization))),
                math.sqrt(float(np.max(a / (acc * utilization)))),
            )
            scale = max(1.0, minimum_scale)
            result.update(
                kinematics_passed=True,
                joint_peak_to_peak_deg=np.rad2deg(np.ptp(values, axis=0)).tolist(),
                joint_peak_velocity_rad_s=v.tolist(),
                joint_peak_acceleration_rad_s2=a.tolist(),
                peak_deviation=peaks,
                joint_budget_passed=scale <= 1,
                time_scale_to_joint_budget=scale,
                retimed_frequency_hz=opts["frequency_hz"] / scale,
                retimed_duration_s=opts["duration_s"] * scale,
                uniform_time_joint_budget_frequency_hz=opts["frequency_hz"]
                / minimum_scale,
                corresponding_duration_s=opts["duration_s"] * minimum_scale,
                moving_joints=[i + 1 for i in range(7) if v[i] > 1e-6],
                note="仅关节预算参考；未验证碰撞、SDK 跟踪、持杯和骰子，不是实测最大频率",
            )
        except ValueError as error:
            result["blockers"].append(str(error))
    return report
