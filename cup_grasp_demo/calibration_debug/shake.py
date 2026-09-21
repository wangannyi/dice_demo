"""Plan a table-tangent shake without importing or commanding hardware.

Planning checks a local IK branch and configured velocity/acceleration limits.
It does not certify closed-finger clearance, grip retention or contact forces.
"""

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from cup_grasp_demo.calibration_debug.parameters import shake_options
from nero_revo2_control.kinematics import _rpy_transform, load_model


KIND = 'shake_design_plan'
JOINT_MARGIN_RAD = math.radians(1)
LIMIT_UTILIZATION = .8
SAMPLE_HZ = 100


def direction_on_table(normal, azimuth_deg):
    normal = np.asarray(normal, dtype=float)
    if (normal.shape != (3,) or not np.isfinite(normal).all()
            or not .99 <= np.linalg.norm(normal) <= 1.01):
        raise ValueError('Invalid table normal')
    normal = normal / np.linalg.norm(normal)
    if normal[2] < .9:
        raise ValueError('Table normal must point upwards and be near horizontal')
    angle = math.radians(azimuth_deg)
    xy = np.array([math.cos(angle), math.sin(angle)])
    direction = np.r_[xy, -np.dot(normal[:2], xy) / normal[2]]
    return direction / np.linalg.norm(direction)


def profile(t, opts):
    """Return displacement, velocity and acceleration; total duration includes ramps."""
    frequency, duration = opts['frequency_hz'], opts['duration_s']
    t = np.asarray(t, dtype=float)
    ramp = min(1 / frequency, duration / 4)
    w, wd, wdd = np.ones_like(t), np.zeros_like(t), np.zeros_like(t)
    for mask, x, sign in ((t < ramp, t / ramp, 1),
                          (t > duration - ramp, (duration - t) / ramp, -1)):
        x = np.clip(x[mask], 0, 1)
        w[mask] = 10 * x**3 - 15 * x**4 + 6 * x**5
        wd[mask] = sign * (30 * x**2 - 60 * x**3 + 30 * x**4) / ramp
        wdd[mask] = (60 * x - 180 * x**2 + 120 * x**3) / ramp**2
    omega = 2 * math.pi * frequency
    sn, cs = np.sin(omega * t), np.cos(omega * t)
    amplitude = opts['amplitude_mm'] / 1000
    return (amplitude * w * sn,
            amplitude * (wd * sn + w * omega * cs),
            amplitude * (wdd * sn + 2 * wd * omega * cs - w * omega**2 * sn))


class Kinematics:
    def __init__(self):
        self.model = load_model()
        self.lower, self.upper = np.array(self.model.limits_rad).T
        self.origins = [np.array(_rpy_transform(j.origin_xyz, j.origin_rpy))
                        for j in self.model.joints]
        self.axes = [np.array(j.axis_xyz) for j in self.model.joints]

    def forward(self, q):
        transform = np.eye(4)
        pivots, axes = [], []
        for origin, axis, angle in zip(self.origins, self.axes, q):
            transform = transform @ origin
            pivots.append(transform[:3, 3].copy())
            axes.append(transform[:3, :3] @ axis)
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            transform = transform @ rotation
        axes, pivots = np.array(axes), np.array(pivots)
        jacobian = np.r_[np.cross(axes, transform[:3, 3] - pivots).T, axes.T]
        return transform, jacobian

    def forward_batch(self, qs):
        """The same FK for every input sample, without unused per-sample Jacobians."""
        qs = np.asarray(qs, dtype=float)
        if qs.ndim != 2 or qs.shape[1] != 7 or not len(qs) or not np.isfinite(qs).all():
            raise ValueError('Expected finite N x 7 joint samples')
        transform = np.broadcast_to(np.eye(4), (len(qs), 4, 4)).copy()
        for i, (origin, axis) in enumerate(zip(self.origins, self.axes)):
            rotation = np.broadcast_to(np.eye(4), (len(qs), 4, 4)).copy()
            rotation[:, :3, :3] = Rotation.from_rotvec(qs[:, i, None] * axis).as_matrix()
            transform = transform @ origin @ rotation
        return transform

    def solve(self, seed, target):
        q = np.array(seed, dtype=float)
        for _ in range(25):
            current, jacobian = self.forward(q)
            error = np.r_[target[:3, 3] - current[:3, 3],
                          Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()]
            if np.linalg.norm(error[:3]) < 1e-9 and np.linalg.norm(error[3:]) < 1e-9:
                return q
            dq = np.linalg.lstsq(jacobian, error, rcond=None)[0]
            dq *= min(1, .05 / max(np.max(np.abs(dq)), 1e-12))
            q += dq
            if np.any(q < self.lower) or np.any(q > self.upper):
                break
        raise ValueError('当前局部 IK 分支无法覆盖完整行程；不是全工作空间不可达的证明')


def read_limits(feedback):
    if feedback.get('success') is not True:
        raise ValueError('Controller limit query did not succeed')
    rows = feedback.get('limits', [])
    if len(rows) != 7 or [r.get('joint') for r in rows] != list(range(1, 8)):
        raise ValueError('Require seven ordered joint limits')
    keys = ('min_angle_rad', 'max_angle_rad', 'max_velocity_rad_s', 'max_acceleration_rad_s2')
    values = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    if (not np.isfinite(values).all() or np.any(values[:, 0] >= values[:, 1])
            or np.any(values[:, 2:] <= 0)):
        raise ValueError('Invalid joint limit values')
    flange = feedback['flange_limits']
    cart = np.array([flange['end_max_linear_vel'], flange['end_max_linear_acc']], dtype=float)
    if not np.isfinite(cart).all() or np.any(cart <= 0):
        raise ValueError('Invalid flange limits')
    return values.T, cart


def make_plan(session, feedback, cfg):
    opts = shake_options(cfg)
    utilization = opts.get('limit_utilization', LIMIT_UTILIZATION)
    (lower, upper, velocity, acceleration), cart = read_limits(feedback)
    q0 = np.array(feedback['q_after_rad'], dtype=float)
    if q0.shape != (7,) or not np.isfinite(q0).all():
        raise ValueError('Require seven finite starting joints')
    normal = session['scene']['cup_normal_base']
    direction = direction_on_table(normal, opts['azimuth_deg'])
    kin = Kinematics()
    kin.lower, kin.upper = np.maximum(kin.lower, lower), np.minimum(kin.upper, upper)
    if np.any(q0 < kin.lower + JOINT_MARGIN_RAD) or np.any(q0 > kin.upper - JOINT_MARGIN_RAD):
        raise ValueError('Starting joints require at least 1 degree of joint-limit margin')
    t0, _ = kin.forward(q0)
    np.testing.assert_allclose(t0, kin.model.fk(q0), atol=1e-10)
    duration, frequency = opts['duration_s'], opts['frequency_hz']
    amplitude = opts['amplitude_mm'] / 1000
    ramp = min(1 / frequency, duration / 4)
    times = np.linspace(0, duration, math.ceil(duration * SAMPLE_HZ) + 1)
    # Dense dynamics screening is separate from the 100 Hz reference output.
    dense = np.linspace(0, duration, math.ceil(duration * 1000) + 1)
    ds, dv, da = profile(dense, opts)
    displacement, _, _ = profile(times, opts)
    result = dict(kind=KIND, parameters=opts, execution_enabled=False,
                  planning_passed=False, motion_commands_sent=0, blockers=[],
                  start_q_rad=q0.tolist(), T_base_flange_center=t0.tolist(),
                  direction_base=direction.tolist(), total_stroke_mm=2 * opts['amplitude_mm'],
                  duration_includes_ramps=True, ramp_s_each_end=ramp,
                  nominal_cycles=frequency * duration, sample_hz=SAMPLE_HZ,
                  configured_limit_utilization=utilization,
                  trajectory_states=[dict(state='RAMP_UP', start_s=0., end_s=ramp),
                                     dict(state='SHAKE', start_s=ramp, end_s=duration-ramp),
                                     dict(state='RAMP_DOWN', start_s=duration-ramp, end_s=duration),
                                     dict(state='CENTER_HOLD', start_s=duration)],
                  validation_pending=['整臂/闭合手形/桌边通路检查', '杯口贴桌与动态抓握验证',
                                      '实际频率和摆幅跟踪验证', '骰子翻滚与点数变化验证'],
                  pose_reference='闭手后的当前法兰位姿；不使用张开食指 TCP 追踪闭合手形',
                  cartesian_peak_velocity_m_s=float(np.max(np.abs(dv))),
                  cartesian_peak_acceleration_m_s2=float(np.max(np.abs(da))))
    ratios = [result['cartesian_peak_velocity_m_s'] / cart[0],
              result['cartesian_peak_acceleration_m_s2'] / cart[1]]
    for label, ratio, demand, limit in zip(('末端线速度', '末端线加速度'), ratios,
                                          (result['cartesian_peak_velocity_m_s'],
                                           result['cartesian_peak_acceleration_m_s2']), cart):
        if ratio > utilization:
            result['blockers'].append(f'{label}需求 {demand:.3f}，配置限值 {limit:.3f}，'
                                      f'占用 {ratio:.1%}，超过规划预算 {utilization:.0%}')
    paths = {}
    try:
        steps = math.ceil(amplitude / .0005)
        distances = np.linspace(0, amplitude, steps + 1)
        for sign in (-1, 1):
            q, rows = q0.copy(), [q0.copy()]
            for distance in distances[1:]:
                target = t0.copy()
                target[:3, 3] += direction * distance * sign
                q = kin.solve(q, target)
                if np.min(np.minimum(q - kin.lower, kin.upper - q)) < JOINT_MARGIN_RAD:
                    raise ValueError('摇晃路径距关节限位不足 1°')
                rows.append(q.copy())
            paths[sign] = rows
        knots = np.r_[-distances[1:][::-1], distances]
        spline = CubicSpline(knots, np.r_[paths[-1][1:][::-1], paths[1]], axis=0)
        q = spline(ds)
        qv = spline(ds, 1) * dv[:, None]
        qa = spline(ds, 2) * dv[:, None]**2 + spline(ds, 1) * da[:, None]
        peak_v, peak_a = np.max(np.abs(qv), axis=0), np.max(np.abs(qa), axis=0)
        v_ratio, a_ratio = peak_v / velocity, peak_a / acceleration
        result.update(joint_peak_velocity_rad_s=peak_v.tolist(),
                      joint_peak_acceleration_rad_s2=peak_a.tolist(),
                      joint_velocity_limit_ratio=v_ratio.tolist(),
                      joint_acceleration_limit_ratio=a_ratio.tolist(),
                      minimum_joint_margin_deg=float(np.rad2deg(
                          np.min(np.minimum(q - kin.lower, kin.upper - q)))))
        if result['minimum_joint_margin_deg'] < 1:
            raise ValueError('插值轨迹距关节限位不足 1°')
        for index in range(7):
            for label, ratio, demand, limit in (('速度', v_ratio[index], peak_v[index], velocity[index]),
                                               ('加速度', a_ratio[index], peak_a[index], acceleration[index])):
                if ratio > utilization:
                    result['blockers'].append(f'J{index+1} {label}需求 {demand:.3f}，配置限值 {limit:.3f}，'
                                              f'占用 {ratio:.1%}，超过规划预算 {utilization:.0%}')
        max_pos, max_rot = 0., 0.
        for distance, angles in zip((knots[1:]+knots[:-1])/2,
                                    spline((knots[1:]+knots[:-1])/2)):
            transform, _ = kin.forward(angles)
            max_pos = max(max_pos, float(np.linalg.norm(transform[:3, 3] - t0[:3, 3] - distance * direction)))
            max_rot = max(max_rot, float(np.linalg.norm(Rotation.from_matrix(
                t0[:3, :3] @ transform[:3, :3].T).as_rotvec())))
        result.update(interpolation_position_error_mm=max_pos*1000,
                      interpolation_rotation_error_deg=math.degrees(max_rot))
        if max_pos > .00001 or max_rot > math.radians(.01):
            raise ValueError('插值后的位姿误差超限')
        scale = max(1., float(max(v_ratio))/utilization,
                    math.sqrt(float(max(a_ratio))/utilization),
                    ratios[0]/utilization, math.sqrt(ratios[1]/utilization))
        result['uniform_retiming_reference'] = dict(
            applied=False, time_scale=scale, frequency_hz=frequency/scale,
            duration_s=duration*scale, ramp_s_each_end=ramp*scale,
            note='保持本次轨迹形状和周期数，所有时间同步拉长；仅供参考，不自动改写四个参数')
        result['samples'] = [dict(t_s=float(t), displacement_m=float(s), q_rad=angles.tolist())
                             for t, s, angles in zip(times, displacement, spline(displacement))]
    except ValueError as error:
        result['blockers'].append(str(error))
    result['planning_passed'] = not result['blockers']
    return result
