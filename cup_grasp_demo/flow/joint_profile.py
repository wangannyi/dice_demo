"""Bounded, synchronized joint reciprocation. No SDK, numpy or hardware imports."""

import math
import statistics

from cup_grasp_demo.flow.parameters import number

KIND = "joint_reciprocation_test"
DEFAULTS = dict(
    joints=[7],
    amplitude_deg=[5.0],
    velocity_deg_s=[20.0],
    acceleration_deg_s2=[60.0],
    cycles=3,
    phase_delay_deg=None,
    limit_utilization=0.97,
    controller_speed_percent=97,
    controller_acceleration_rad_s2=None,
    tracking_error_deg=5.0,
    tracking_error_action="stop",
    envelope_margin_deg=None,
    feedback_reference_delay_s=0.0,
    measurement_window_s=0.12,
    inactive_error_deg=0.5,
    command_step_deg=3.0,
    command_lag_s=0.15,
    command_rate_hz=None,
    command_mode="smooth_profile",
    endpoint_blend=0.5,
)


def options(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("Unknown joint-test configuration fields")
    result = dict(DEFAULTS, **raw)
    if result["command_mode"] not in ("smooth_profile", "blended", "fixed_endpoints"):
        raise ValueError("command_mode must be smooth_profile, blended or fixed_endpoints")
    result["endpoint_blend"] = number(
        result["endpoint_blend"], "endpoint_blend", 0, 1
    )
    if result["command_rate_hz"] is not None:
        result["command_rate_hz"] = number(result["command_rate_hz"], "command_rate_hz", 20, 200, True)
    targets = result["controller_acceleration_rad_s2"]
    if targets is not None:
        if not isinstance(targets, list) or len(targets) != 7:
            raise ValueError("controller_acceleration_rad_s2 requires seven values ordered J1..J7")
        result["controller_acceleration_rad_s2"] = [
            number(v, "controller_acceleration_rad_s2", .01, 5.0) for v in targets]
        if any(abs(v * 100 - round(v * 100)) > 1e-8 for v in targets):
            raise ValueError("Controller acceleration resolution is 0.01 rad/s²")
    if result["tracking_error_action"] not in ("stop", "record"):
        raise ValueError("tracking_error_action must be stop or record")
    joints = result["joints"]
    if (
        not isinstance(joints, list)
        or not joints
        or any(type(j) is not int or not 1 <= j <= 7 for j in joints)
        or len(set(joints)) != len(joints)
    ):
        raise ValueError("joints must contain 1..7 distinct integer joint numbers from 1 to 7")
    for key, low, high in [
        ("amplitude_deg", -60, 60),
        ("velocity_deg_s", 0.1, 300),
        ("acceleration_deg_s2", 0.1, 1000),
    ]:
        values = result[key]
        if not isinstance(values, list) or len(values) != len(joints):
            raise ValueError(key + " must contain one value per selected joint")
        result[key] = [number(v, key, low, high) for v in values]
    if any(abs(v) < 0.1 for v in result["amplitude_deg"]):
        raise ValueError("Each signed amplitude must have magnitude >= 0.1 degree")
    phases = result['phase_delay_deg']
    if phases is not None:
        if not isinstance(phases, list) or len(phases) != len(joints):
            raise ValueError('phase_delay_deg must contain one value per selected joint')
        result['phase_delay_deg'] = [number(v, 'phase_delay_deg', 0, 360) for v in phases]
    for key, low, high, integer in [
        ("cycles", 1, 20, True),
        ("limit_utilization", 0.1, 1.0, False),
        ("controller_speed_percent", 1, 100, True),
        ("tracking_error_deg", 0.1, 10, False),
        ("feedback_reference_delay_s", 0.0, 0.2, False),
        ("measurement_window_s", 0.04, 0.5, False),
        ("inactive_error_deg", 0.05, 1, False),
        ("command_step_deg", 0.1, 20, False),
        ("command_lag_s", 0.02, 0.2, False),
    ]:
        result[key] = number(result[key], key, low, high, integer)
    # Legacy plans used the tracking threshold as the envelope margin too.
    margin = result["envelope_margin_deg"]
    result["envelope_margin_deg"] = number(
        result["tracking_error_deg"] if margin is None else margin,
        "envelope_margin_deg", 0.1, 10,
    )
    return result


def leg(start, end, speed, acceleration):
    distance = abs(end - start)
    ta = min(speed / acceleration, math.sqrt(distance / acceleration))
    peak = acceleration * ta
    cruise = (
        0.0
        if distance <= speed * speed / acceleration
        else max(0.0, distance / peak - ta)
    )
    return dict(
        start=start,
        end=end,
        accel=acceleration,
        accel_s=ta,
        cruise_s=cruise,
        peak=peak,
        duration_s=2 * ta + cruise,
    )


def value(segment, t):
    t = min(max(t, 0.0), segment["duration_s"])
    a, ta, tc, v = (segment[k] for k in ("accel", "accel_s", "cruise_s", "peak"))
    if t < ta:
        position, velocity, acceleration = 0.5 * a * t * t, a * t, a
    elif t < ta + tc:
        position, velocity, acceleration = 0.5 * a * ta * ta + v * (t - ta), v, 0.0
    else:
        left = segment["duration_s"] - t
        position, velocity, acceleration = (
            abs(segment["end"] - segment["start"]) - 0.5 * a * left * left,
            a * left,
            -a,
        )
    sign = 1 if segment["end"] > segment["start"] else -1
    return segment["start"] + sign * position, sign * velocity, sign * acceleration


def trajectory(raw):
    cfg = options(raw)
    amplitudes = [abs(v) for v in cfg["amplitude_deg"]]
    speed = min(v / a for v, a in zip(cfg["velocity_deg_s"], amplitudes))
    acceleration = min(v / a for v, a in zip(cfg["acceleration_deg_s2"], amplitudes))
    points = [0.0, 1.0] + [-1.0, 1.0] * cfg["cycles"] + [0.0]
    segments, elapsed = [], 0.0
    for start, end in zip(points, points[1:]):
        item = leg(start, end, speed, acceleration)
        item["start_s"] = elapsed
        segments.append(item)
        elapsed += item["duration_s"]
    if (cfg["tracking_error_action"] == "stop"
            and cfg["feedback_reference_delay_s"] > 0.1 * (2 * segments[1]["duration_s"])):
        raise ValueError(
            "feedback_reference_delay_s must not exceed 10% of a cycle in stop mode; "
            f"maximum {0.2 * segments[1]['duration_s']:.6f}s"
        )
    if elapsed > 60:
        raise ValueError(
            f"Bounded test duration {elapsed:.2f}s exceeds 60s; reduce cycles/amplitude"
        )
    return cfg, segments, elapsed


def at(segments, t):
    for item in segments:
        if t <= item["start_s"] + item["duration_s"]:
            return value(item, t - item["start_s"])
    return 0.0, 0.0, 0.0


def joint_values(plan, t):
    """Phase lag delays each complete profile; all axes start/end at rest at center."""
    cfg = plan['parameters']
    phases = cfg.get('phase_delay_deg')
    if not phases or not any(phases):
        u, _, _ = at(plan['segments'], t)
        return [q + a * u for q, a in zip(plan['start_q_rad'], plan['amplitude_rad'])]
    delays = dict(zip(cfg['joints'], phases))
    period = 2 * plan['segments'][1]['duration_s']
    return [q + a * at(plan['segments'], t - period * delays.get(i + 1, 0) / 360)[0]
            for i, (q, a) in enumerate(zip(plan['start_q_rad'], plan['amplitude_rad']))]


def endpoint_value(segments, t):
    """Return the fixed destination of the active half-cycle."""
    if t < 0:
        return 0.0
    for item in segments:
        if t < item["start_s"] + item["duration_s"]:
            return item["end"]
    return 0.0


def command_values(plan, t):
    """Return the target sent to MoveJS for the selected command strategy."""
    cfg = plan["parameters"]
    mode = cfg.get("command_mode", "smooth_profile")
    smooth = joint_values(plan, t)
    if mode == "smooth_profile":
        return smooth
    phases = cfg.get("phase_delay_deg")
    if not phases or not any(phases):
        u = endpoint_value(plan["segments"], t)
        endpoint = [
            q + a * u for q, a in zip(plan["start_q_rad"], plan["amplitude_rad"])
        ]
    else:
        delays = dict(zip(cfg["joints"], phases))
        period = 2 * plan["segments"][1]["duration_s"]
        endpoint = [
            q + a * endpoint_value(
                plan["segments"], t - period * delays.get(i + 1, 0) / 360
            )
            for i, (q, a) in enumerate(zip(plan["start_q_rad"], plan["amplitude_rad"]))
        ]
    if mode == "fixed_endpoints":
        return endpoint
    blend = cfg["endpoint_blend"]
    return [a + blend * (b - a) for a, b in zip(smooth, endpoint)]


def make_plan(feedback, raw, model_limits):
    cfg, segments, duration = trajectory(raw)
    duration += max(cfg.get('phase_delay_deg') or [0]) / 360 * 2 * segments[1]['duration_s']
    if duration > 60:
        raise ValueError('Phase-delayed test duration exceeds 60s')
    q0 = feedback["q_after_rad"]
    if len(q0) != 7 or not all(math.isfinite(v) for v in q0):
        raise ValueError("Require seven finite starting angles")
    limits = feedback["limits"]
    if not feedback.get("success") or [v["joint"] for v in limits] != list(range(1, 8)):
        raise ValueError("Require successful ordered live limits")
    amplitudes = [0.0] * 7
    for j, a in zip(cfg["joints"], cfg["amplitude_deg"]):
        amplitudes[j - 1] = math.radians(a)
    peak = max(s["peak"] for s in segments)
    peak_acc = segments[0]["accel"]
    v = [abs(a) * peak for a in amplitudes]
    a = [abs(v) * peak_acc for v in amplitudes]
    blockers = []
    for i, row in enumerate(limits):
        for key in (
            "min_angle_rad",
            "max_angle_rad",
            "max_velocity_rad_s",
            "max_acceleration_rad_s2",
        ):
            if not math.isfinite(row[key]):
                raise ValueError("Non-finite controller limits")
        if (
            row["min_angle_rad"] >= row["max_angle_rad"]
            or row["max_velocity_rad_s"] <= 0
            or row["max_acceleration_rad_s2"] <= 0
        ):
            raise ValueError("Invalid controller limits")
        low = max(row["min_angle_rad"], model_limits[i][0]) + math.radians(1)
        high = min(row["max_angle_rad"], model_limits[i][1]) - math.radians(1)
        if not low <= q0[i] - abs(amplitudes[i]) <= q0[i] + abs(amplitudes[i]) <= high:
            blockers.append(f"J{i + 1}: 端点距关节限位不足 1°")
        for label, demand, limit in [
            ("速度", v[i], row["max_velocity_rad_s"]),
            ("加速度", a[i], row["max_acceleration_rad_s2"]),
        ]:
            if demand > limit * cfg["limit_utilization"] + 1e-10:
                blockers.append(
                    f"J{i + 1} {label}需求 {demand:.4f} 超过 {cfg['limit_utilization']:.0%} 预算 {limit * cfg['limit_utilization']:.4f}"
                )
    count = math.ceil(duration * 100)
    samples = []
    for i in range(count + 1):
        t = i * duration / count
        samples.append(
            dict(t_s=t, q_rad=joint_values(dict(parameters=cfg, segments=segments,
                                               start_q_rad=q0, amplitude_rad=amplitudes), t))
        )
    samples[0]["q_rad"] = list(q0)
    samples[-1]["q_rad"] = list(q0)
    return dict(
        kind=KIND,
        parameters=cfg,
        start_q_rad=q0,
        amplitude_rad=amplitudes,
        segments=segments,
        samples=samples,
        duration_s=duration,
        cycle_period_s=2 * segments[1]["duration_s"],
        reference_frequency_hz=1 / (2 * segments[1]["duration_s"]),
        joint_peak_velocity_rad_s=v,
        joint_peak_acceleration_rad_s2=a,
        triangular_profile=segments[1]["cruise_s"] == 0,
        warnings=([
            "反馈补偿延迟超过周期 10%；record 模式仅记录，不阻止规划。"
            "请结合原始误差、实际幅度和频率评估跟踪效果。"
        ] if cfg["feedback_reference_delay_s"] > 0.2 * segments[1]["duration_s"] else []),
        blockers=blockers,
        planning_passed=not blockers,
        execution_enabled=False,
        physical_limit_verified=False,
    )


def reference_errors(row, plan, elapsed, delay=0.0):
    target = command_values(plan, max(0.0, min(elapsed - delay, plan['duration_s'])))
    return [math.degrees(q - expected) for q, expected in zip(row['q_rad'], target)]


def tracking_exceedances(errors, cfg):
    return [j for j in cfg["joints"] if abs(errors[j - 1]) > cfg["tracking_error_deg"]]


def feedback_check(row, plan, elapsed):
    from cup_grasp_demo.flow.shake_tracking import state_ok

    if not state_ok(row):
        raise RuntimeError("电机测试反馈状态异常")
    cfg = plan["parameters"]
    if len(row["q_rad"]) != 7:
        raise RuntimeError("缺少七轴反馈")
    errors = reference_errors(
        row, plan, elapsed, cfg.get("feedback_reference_delay_s", 0.0)
    )
    for i, (q, center, amp) in enumerate(
        zip(row["q_rad"], plan["start_q_rad"], plan["amplitude_rad"])
    ):
        margin = cfg.get("envelope_margin_deg")
        if margin is None:
            margin = cfg["tracking_error_deg"]
        envelope = math.radians(margin if amp else cfg["inactive_error_deg"])
        if not math.isfinite(q) or abs(q - center) > abs(amp) + envelope:
            raise RuntimeError(f"J{i + 1} 超出往返包络或未选中关节发生运动")
        tolerance = cfg["tracking_error_deg"] if amp else cfg["inactive_error_deg"]
        if (
            abs(errors[i]) > tolerance
            and (not amp or cfg.get("tracking_error_action", "stop") == "stop")
        ):
            delay = cfg.get("feedback_reference_delay_s", 0.0)
            raise RuntimeError(
                f"J{i + 1} 跟踪误差 {errors[i]:.2f}° 超限 "
                f"(阈值 {tolerance:g}°，参考延迟 {delay:.3f}s)"
            )
    return errors


def packet_samples(rows, joint, max_age_s=.1):
    """Use packet receipt timestamps; identical packet copies count once."""
    key = (
        "joint_12",
        "joint_12",
        "joint_34",
        "joint_34",
        "joint_56",
        "joint_56",
        "joint_7",
    )[joint - 1]
    samples, previous_stamp, sources = [], None, set()
    for item in rows:
        row = item["feedback"]
        q = row.get("q_rad", [])
        if len(q) != 7 or not all(math.isfinite(v) for v in q):
            continue
        t = item["elapsed_s"]
        stamp = (
            row.get("sdk_snapshot", {})
            .get("packet_timestamps_after_epoch_s", {})
            .get(key)
        )
        observed = row.get("observed_epoch_s")
        if isinstance(stamp, (int, float)) and isinstance(observed, (int, float)):
            if not math.isfinite(stamp) or not 0 <= observed - stamp <= max_age_s:
                continue
            if previous_stamp is not None and stamp <= previous_stamp:
                continue
            previous_stamp = stamp
            t -= observed - stamp
            source = "joint_packet_receipt"
        else:
            source = "observation_time_fallback"
        if not math.isfinite(t) or (samples and t <= samples[-1][0]):
            continue
        samples.append((t, math.degrees(q[joint - 1])))
        sources.add(source)
    return samples, sorted(sources)


def solve3(matrix, vector):
    """Pivoted 3x3 solve for a local quadratic fit, without SDK numpy dependency."""
    rows = [list(a) + [b] for a, b in zip(matrix, vector)]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda k: abs(rows[k][i]))
        rows[i], rows[pivot] = rows[pivot], rows[i]
        if abs(rows[i][i]) < 1e-12:
            raise ValueError("Degenerate derivative window")
        divisor = rows[i][i]
        rows[i] = [v / divisor for v in rows[i]]
        for j in range(3):
            if j != i:
                factor = rows[j][i]
                rows[j] = [a - factor * b for a, b in zip(rows[j], rows[i])]
    return [rows[i][3] for i in range(3)]


def window_derivatives(samples, window_s):
    """Estimate velocity/acceleration over a fixed full window; no edge extrapolation."""
    from bisect import bisect_left, bisect_right

    times = [s[0] for s in samples]
    half = window_s / 2
    result = []
    for t, q in samples:
        if t - half < times[0] or t + half > times[-1]:
            continue
        selected = samples[bisect_left(times, t - half) : bisect_right(times, t + half)]
        if len(selected) < 5 or selected[-1][0] - selected[0][0] < 0.8 * window_s:
            continue
        x = [(s - t) / half for s, _ in selected]
        y = [v - q for _, v in selected]
        moments = [sum(v**power for v in x) for power in range(5)]
        rhs = [sum(a**power * b for a, b in zip(x, y)) for power in range(3)]
        try:
            _, slope, curve = solve3(
                [[moments[i + j] for j in range(3)] for i in range(3)], rhs
            )
        except ValueError:
            continue
        result.append((t, slope / half, 2 * curve / half**2))
    return result


def measurements(rows, plan):
    """Windowed angle-feedback estimates; never substitute command derivatives."""
    result = []
    window = plan["parameters"].get("measurement_window_s", 0.12)
    # Measurement must accept the same packet ages the execution loop accepted;
    # a stricter filter here silently drops valid samples and can flip
    # tracking_verified to false on a run that actually tracked.
    max_age_s = plan["parameters"].get("feedback_freshness_limit_s", .1)
    for j, requested in zip(
        plan["parameters"]["joints"], plan["parameters"]["amplitude_deg"]
    ):
        samples, sources = packet_samples(rows, j, max_age_s=max_age_s)
        if len(samples) < 3:
            continue
        derivatives = window_derivatives(samples, window)
        center = math.degrees(plan["start_q_rad"][j - 1])
        sign = 1 if requested > 0 else -1
        crossings = []
        for a, b in zip(samples, samples[1:]):
            x, y = sign * (a[1] - center), sign * (b[1] - center)
            if x < 0 <= y and y > x:
                crossings.append(a[0] + (b[0] - a[0]) * (-x) / (y - x))
        periods = [b - a for a, b in zip(crossings, crossings[1:])]
        freq = 1 / statistics.median(periods) if periods else None
        stroke = max(q for _, q in samples) - min(q for _, q in samples)
        result.append(
            dict(
                joint=j,
                stroke_deg=stroke,
                frequency_hz=freq,
                reference_frequency_hz=plan["reference_frequency_hz"],
                estimated_peak_velocity_deg_s=max(
                    (abs(v) for _, v, _ in derivatives), default=None
                ),
                estimated_peak_acceleration_deg_s2=max(
                    (abs(a) for _, _, a in derivatives), default=None
                ),
                derivative_window_s=window,
                derivative_estimate_count=len(derivatives),
                unique_feedback_samples=len(samples),
                timestamp_sources=sources,
                complete_periods_measured=len(periods),
                tracking_verified=freq is not None
                and abs(freq / plan["reference_frequency_hz"] - 1) <= 0.1
                and 0.8 <= stroke / (2 * abs(requested)) <= 1.1,
            )
        )
    return dict(
        joints=result,
        derivative_note="按关节包接收时间去重，固定时间窗二次拟合估算；会平滑尖峰，不是电机额定能力。数据不足为 null",
        tracking_verified=bool(result) and all(r["tracking_verified"] for r in result),
    )
