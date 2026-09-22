"""Stdlib-only feedback checks and measurements for the isolated SDK runner."""

import bisect
import math
import statistics

from cup_grasp_demo.flow.parameters import shake_orientation_limit


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def rotation(pose):
    r, p, y = pose[3:]
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def interpolate_displacement(samples, elapsed, times=None):
    times = times if times is not None else [s["t_s"] for s in samples]
    i = bisect.bisect_right(times, elapsed)
    if i == 0:
        return samples[0]["displacement_m"]
    if i == len(samples):
        return samples[-1]["displacement_m"]
    a, b = samples[i - 1], samples[i]
    u = (elapsed - a["t_s"]) / (b["t_s"] - a["t_s"])
    return a["displacement_m"] * (1 - u) + b["displacement_m"] * u


def state_ok(row, stopped=False):
    state = row["status"]
    return (
        state["arm_status"] == 0
        and state["ctrl_mode"] == 1
        and state["motion_status"] in ((0,) if stopped else (0, 1))
        and row["enabled"] == [True] * 7
    )


def check_feedback(row, plan, *, expected_s=None, joint_bounds=None):
    if not state_ok(row):
        raise RuntimeError("摇晃时 NORMAL/CAN/七轴使能状态异常")
    q = row["q_rad"]
    if joint_bounds is None:
        refs = [s["q_rad"] for s in plan["samples"]]
        joint_bounds = [(min(r[i] for r in refs), max(r[i] for r in refs)) for i in range(7)]
    if (
        len(q) != 7
        or not all(math.isfinite(v) for v in q)
        or any(
            q[i] < joint_bounds[i][0] - math.radians(plan['parameters'].get('joint_path_tolerance_deg', .15))
            or q[i] > joint_bounds[i][1] + math.radians(plan['parameters'].get('joint_path_tolerance_deg', .15))
            for i in range(7)
        )
    ):
        raise RuntimeError("关节反馈越出计划路径范围")
    pose = row["fk_flange_pose_m_rad"]
    center = plan["T_base_flange_center"]
    direction = plan["direction_base"]
    if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise RuntimeError("Invalid flange feedback")
    delta = [pose[i] - center[i][3] for i in range(3)]
    along = dot(delta, direction)
    cross = math.sqrt(sum((delta[i] - along * direction[i]) ** 2 for i in range(3)))
    height = dot(delta, plan["table_normal_base"])
    r = rotation(pose)
    angle = math.acos(
        min(
            1.0,
            max(
                -1.0,
                (sum(r[i][j] * center[i][j] for i in range(3) for j in range(3)) - 1)
                / 2,
            ),
        )
    )
    violations = []
    if abs(along) > plan["parameters"]["amplitude_mm"] / 1000 + 0.005:
        violations.append(f"沿程位移 {along * 1000:.2f} mm 超出幅度加 5 mm")
    if cross > 0.003:
        violations.append(f"横向偏差 {cross * 1000:.2f} mm > 3 mm")
    if abs(height) > 0.002:
        violations.append(f"离桌高度变化 {height * 1000:.2f} mm > 2 mm")
    orientation_limit = shake_orientation_limit(plan["parameters"])
    if orientation_limit is not None and angle > math.radians(orientation_limit):
        violations.append(
            f"法兰朝向偏差 {math.degrees(angle):.3f}° > {orientation_limit:g}°"
        )
    limit = plan['parameters'].get('tracking_error_limit_mm', 15.)
    if expected_s is not None and abs(along - expected_s) * 1000 > limit:
        violations.append(
            f"沿程跟踪偏差 {abs(along - expected_s) * 1000:.2f} mm > {limit:g} mm"
        )
    if violations:
        raise RuntimeError("；".join(violations))
    return dict(
        along_m=along, cross_m=cross, height_m=height, rotation_deg=math.degrees(angle)
    )


class FeedbackMonitor:
    """Compare measured motion with a fixed-delay reference, never advance targets."""

    def __init__(self, plan):
        self.plan = plan
        self.samples = plan['samples']
        self.times = [s['t_s'] for s in self.samples]
        self.delay = plan['parameters'].get('feedback_reference_delay_s', 0.)
        self.error_since = None
        self.bounds = [(min(s['q_rad'][i] for s in self.samples),
                        max(s['q_rad'][i] for s in self.samples)) for i in range(7)]

    def check(self, row, elapsed):
        raw = interpolate_displacement(self.samples, elapsed, self.times)
        reference = interpolate_displacement(self.samples, max(0., elapsed - self.delay), self.times)
        tracking = check_feedback(row, self.plan, joint_bounds=self.bounds)
        limit = self.plan['parameters'].get('tracking_error_limit_mm', 15.)
        error = abs(tracking['along_m'] - reference) * 1000
        if error > limit:
            if self.error_since is None:
                self.error_since = elapsed
            if elapsed - self.error_since >= self.plan['parameters'].get('tracking_error_grace_s', 0.):
                raise RuntimeError(f'沿程跟踪偏差 {error:.2f} mm > {limit:g} mm，持续 {elapsed - self.error_since:.3f} s')
        else:
            self.error_since = None
        tracking.update(reference_displacement_m=reference, reference_delay_s=self.delay,
                        tracking_warning=error > limit,
                        raw_following_error_mm=(tracking['along_m'] - raw) * 1000,
                        aligned_following_error_mm=(tracking['along_m'] - reference) * 1000)
        return tracking


def check_delivery(q, previous_q, elapsed, scheduled, parameters=None):
    parameters = parameters or {}
    lag = parameters.get('command_lag_limit_s', .08)
    step = parameters.get('command_step_limit_deg', 1.)
    if (
        elapsed - scheduled > lag
        or elapsed < scheduled - 0.011
        or max(abs(a - b) for a, b in zip(q, previous_q)) > math.radians(step)
    ):
        raise RuntimeError(f"参考点延迟超过 {lag * 1000:g} ms 或相邻目标超过 {step:g}°，停止补发")


def measured_wave(rows, plan):
    """Frequency comes from same-direction feedback zero crossings, not commands."""
    t = [r["elapsed_s"] for r in rows]
    x = [r["tracking"]["along_m"] for r in rows]
    crossings = {1: [], -1: []}
    sign = 0
    armed = False
    last = None
    for i in range(1, len(x)):
        if abs(x[i - 1]) > 0.002:
            sign = 1 if x[i - 1] > 0 else -1
            armed = True
        if armed and x[i - 1] * x[i] <= 0 and x[i] != x[i - 1]:
            last = t[i - 1] - x[i - 1] * (t[i] - t[i - 1]) / (x[i] - x[i - 1])
        if armed and last is not None and x[i] * sign < -0.002:
            crossings[-sign].append(last)
            sign, armed, last = -sign, False, None
    periods = [b - a for vals in crossings.values() for a, b in zip(vals, vals[1:])]
    frequency = 1 / statistics.median(periods) if periods else None
    stroke = (max(x) - min(x)) * 1000 if x else 0.0
    target = plan["parameters"]
    return dict(
        feedback_frequency_hz=frequency,
        feedback_reference_delay_s=target.get('feedback_reference_delay_s', 0.),
        orientation_error_limit_deg=shake_orientation_limit(target),
        orientation_limit_enabled=shake_orientation_limit(target) is not None,
        feedback_total_stroke_mm=stroke,
        feedback_min_mm=min(x) * 1000 if x else None,
        feedback_max_mm=max(x) * 1000 if x else None,
        zero_crossings_s=crossings,
        measured_period_count=len(periods),
        max_table_height_error_mm=max(
            (abs(r["tracking"]["height_m"]) * 1000 for r in rows), default=0
        ),
        max_orientation_error_deg=max(
            (r["tracking"]["rotation_deg"] for r in rows), default=0
        ),
        tracking_verified=frequency is not None
        and abs(frequency / target["frequency_hz"] - 1) <= 0.1
        and 0.8 * 2 * target["amplitude_mm"]
        <= stroke
        <= 1.1 * 2 * target["amplitude_mm"],
        measurement_source="actual joint feedback + SDK FK; not independent visual measurement",
    )
