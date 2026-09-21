"""Isolated JS trial planner: fixed virtual-TCP z/roll/pitch, free yaw/lateral XY."""

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from cup_grasp_demo.calibration_debug.parameters import planar_shake_options
from cup_grasp_demo.calibration_debug.shake import Kinematics, profile, read_limits
from cup_grasp_demo.calibration_debug.shake_study import solve_planar, tcp_state, wrap

KIND = "planar_js_shake_trial"


def options(raw):
    return planar_shake_options(raw)


def make_plan(feedback, raw, tcp):
    opts = options(raw)
    requested = dict(opts)
    (lo, hi, vel, acc), cart = read_limits(feedback)
    acc = np.minimum(acc, opts["joint_acceleration_cap_rad_s2"])
    utilization = opts.get("limit_utilization", .97)
    kin = Kinematics()
    kin.lower = np.maximum(kin.lower, lo) + math.radians(1)
    kin.upper = np.minimum(kin.upper, hi) - math.radians(1)
    q0 = np.asarray(feedback["q_after_rad"], dtype=float)
    tcp = np.asarray(tcp, dtype=float)
    if q0.shape != (7,) or not np.isfinite(q0).all():
        raise ValueError("Invalid starting joints")
    if tcp.shape != (4, 4) or not np.isfinite(tcp).all():
        raise ValueError("Invalid virtual TCP transform")
    np.testing.assert_allclose(tcp[3], [0, 0, 0, 1], atol=1e-10)
    np.testing.assert_allclose(tcp[:3, :3].T @ tcp[:3, :3], np.eye(3), atol=1e-8)
    if np.linalg.det(tcp[:3, :3]) < .999999:
        raise ValueError("TCP rotation must be proper")
    if np.any(q0 < kin.lower) or np.any(q0 > kin.upper):
        details = [f"J{i+1}={math.degrees(q0[i]):.3f}°，规划范围 "
                   f"[{math.degrees(kin.lower[i]):.3f}°, {math.degrees(kin.upper[i]):.3f}°]"
                   for i in range(7) if not kin.lower[i] <= q0[i] <= kin.upper[i]]
        raise ValueError("起点距关节限位不足 1°：" + "；".join(details))
    initial, _, rpy0 = tcp_state(kin, q0, tcp)
    angle = math.radians(opts["azimuth_deg"])
    direction = np.array([math.cos(angle), math.sin(angle), 0.])
    amplitude = opts["amplitude_mm"] / 1000
    distances = np.linspace(0, amplitude, math.ceil(amplitude / .001) + 1)
    branches = {}
    for sign in (-1, 1):
        q, rows = q0.copy(), [q0.copy()]
        for d in distances[1:]:
            q = solve_planar(kin, q, tcp, initial, rpy0, direction, sign * d,
                             list(range(7)), opts["joint_motion_cost"])
            rows.append(q.copy())
        branches[sign] = rows
    spline = CubicSpline(np.r_[-distances[1:][::-1], distances],
                         np.r_[branches[-1][1:][::-1], branches[1]], axis=0)
    duration = opts["duration_s"]
    dense = np.linspace(0, duration, math.ceil(duration * 1000) + 1)
    def dynamics():
        ds, dv, da = profile(dense, opts)
        q = spline(ds)
        velocity = np.max(np.abs(spline(ds, 1) * dv[:, None]), axis=0)
        acceleration = np.max(np.abs(spline(ds, 2) * dv[:, None] ** 2
                                     + spline(ds, 1) * da[:, None]), axis=0)
        return q, velocity, acceleration

    qs, v, a = dynamics()
    attempts = []
    minimum = max(.1, 2. / duration)
    for _ in range(12):
        scale = max(float(np.max(v / (vel * utilization))),
                    math.sqrt(float(np.max(a / (acc * utilization)))))
        attempts.append(dict(frequency_hz=opts['frequency_hz'], demand_scale=scale))
        if not opts.get('auto_reduce_frequency', False) or scale <= 1.:
            break
        # Keep amplitude and duration. Reuse the geometric path, then recheck
        # the complete new time profile; scaling alone is not the acceptance test.
        candidate = max(minimum, math.floor(opts['frequency_hz'] / scale * 100) / 100)
        if candidate >= opts['frequency_hz']:
            break
        opts['frequency_hz'] = candidate
        qs, v, a = dynamics()
    blockers = []
    if np.any(qs < kin.lower) or np.any(qs > kin.upper):
        blockers.append("插值路径超出关节范围")
    for i in range(7):
        if v[i] > vel[i] * utilization:
            blockers.append(f"J{i+1} 速度需求 {v[i]:.3f} 超出预算 {vel[i]*utilization:.3f} rad/s")
        if a[i] > acc[i] * utilization:
            blockers.append(f"J{i+1} 加速度需求 {a[i]:.3f} 超出预算 {acc[i]*utilization:.3f} rad/s²")
    times = np.linspace(0, duration, math.ceil(duration * 100) + 1)
    displacements, _, _ = profile(times, opts)
    samples, deviation = [], np.zeros(5)
    sample_joints = spline(displacements)
    flanges = kin.forward_batch(sample_joints)
    poses = flanges @ tcp
    rpys = Rotation.from_matrix(poses[:, :3, :3]).as_euler("xyz")
    if np.any(np.abs(np.cos(rpys[:, 1])) < .05):
        raise ValueError("TCP RPY near gimbal lock; choose another reference frame")
    flange_rpys = Rotation.from_matrix(flanges[:, :3, :3]).as_euler("xyz")
    for t, d, q, pose, rpy, flange, flange_rpy in zip(
            times, displacements, sample_joints, poses, rpys, flanges, flange_rpys):
        delta = pose[:3, 3] - initial[:3, 3]
        angles = np.rad2deg(wrap(rpy - rpy0))
        errors = [abs(delta[2])*1000, abs(angles[0]), abs(angles[1]), abs(angles[2]),
                  abs(np.dot([-direction[1], direction[0], 0], delta))*1000]
        deviation = np.maximum(deviation, errors)
        if max(errors[:3]) > .02 or abs(np.dot(direction, delta) - d) > .00002:
            raise ValueError("插值后的 Z/Rx/Ry 或沿程约束不满足")
        samples.append(dict(t_s=float(t), q_rad=q.tolist(), displacement_m=float(d),
                            flange_pose_m_rad=[*flange[:3, 3],
                                *flange_rpy]))
    # JS targets are joint-space targets. These values remain visible as a
    # reference, not a claim that Cartesian MoveL limits govern JS firmware.
    return dict(kind=KIND, backend="move_js", parameters=opts, sample_hz=100,
                requested_parameters=requested,
                frequency_adaptation=dict(enabled=opts.get('auto_reduce_frequency', False),
                    applied=opts['frequency_hz'] != requested['frequency_hz'],
                    requested_hz=requested['frequency_hz'], effective_hz=opts['frequency_hz'],
                    attempts=attempts, amplitude_preserved=True, duration_preserved=True),
                start_q_rad=q0.tolist(), T_flange_virtual_tcp=tcp.tolist(),
                T_base_tcp_center=initial.tolist(), tcp_rpy_center_rad=rpy0.tolist(),
                direction_base=direction.tolist(), samples=samples,
                joint_peak_velocity_rad_s=v.tolist(), joint_peak_acceleration_rad_s2=a.tolist(),
                joint_limits_rad=np.c_[lo, hi].tolist(),
                joint_velocity_budget_rad_s=(vel*utilization).tolist(),
                joint_acceleration_budget_rad_s2=(acc*utilization).tolist(),
                cartesian_controller_reference=dict(velocity_m_s=float(cart[0]), acceleration_m_s2=float(cart[1])),
                peak_deviation=dict(zip(["z_mm", "rx_deg", "ry_deg", "rz_deg", "lateral_mm"], deviation.tolist())),
                total_stroke_mm=2*opts["amplitude_mm"], duration_includes_ramps=True,
                performance_error_action="record", planning_passed=not blockers,
                blockers=blockers, execution_enabled=False,
                reference="法兰固连的虚拟 TCP；闭指后不是实际食指表面接触点")
