"""MoveJS transport for the existing joint-position command interface."""

import math
import time


def delivery_options(raw=None):
    """Keep legacy defaults; the release config explicitly selects full budget."""
    options = dict(limit_utilization=.97, acceleration_cap_rad_s2=5.,
                   tracking_error_deg=5., tracking_error_action='stop',
                   envelope_margin_deg=5., velocity_cap_deg_s=50., profile='quintic',
                   feedback_freshness_limit_s=.1)
    raw = raw or {}
    if not isinstance(raw, dict) or set(raw) - set(options):
        raise ValueError('Invalid joint_delivery options')
    options.update(raw)
    if options['profile'] not in ('quintic', 'trapezoid'):
        raise ValueError('joint_delivery.profile must be quintic or trapezoid')
    for name, low, high in (('limit_utilization', .1, 1.),
                            ('acceleration_cap_rad_s2', .1, 5.),
                            ('tracking_error_deg', .1, 10.),
                            ('envelope_margin_deg', .1, 10.),
                            ('feedback_freshness_limit_s', .1, .5)):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'joint_delivery.{name} must be {low}..{high}')
    cap = options['velocity_cap_deg_s']
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float))
                            or not math.isfinite(cap) or cap <= 0):
        raise ValueError('joint_delivery.velocity_cap_deg_s must be positive or null')
    if options['tracking_error_action'] not in ('record', 'stop'):
        raise ValueError('joint_delivery.tracking_error_action must be record or stop')
    return options


def smooth_duration(start, target, velocity, acceleration):
    """Quintic rest-to-rest path: peak normalized speed 1.875, accel 10/sqrt(3)."""
    return max(
        0.04,
        *[
            max(
                1.875 * abs(b - a) / v,
                math.sqrt((10 / math.sqrt(3)) * abs(b - a) / acc),
            )
            for a, b, v, acc in zip(start, target, velocity, acceleration)
        ],
    )


def trapezoid_profile(start, target, velocity, acceleration):
    """Synchronized rest-to-rest motion, bounded by every moving joint.

    Return duration, acceleration interval and normalized peak velocity.
    Zero-distance paths retain a short constant-position delivery interval.
    """
    moving = [(abs(b-a), v, acc) for a, b, v, acc in
              zip(start, target, velocity, acceleration) if abs(b-a) > 1e-12]
    if not moving:
        return .04, .02, 0.
    vmax = min(v / distance for distance, v, _ in moving)
    amax = min(acc / distance for distance, _, acc in moving)
    peak = min(vmax, math.sqrt(amax))
    ramp = peak / amax
    duration = 1 / peak + ramp
    scale = max(1., .04 / duration)
    return duration * scale, ramp * scale, peak / scale


def trapezoid_position(elapsed, duration, ramp, peak):
    if peak == 0:
        return min(1., max(0., elapsed / duration))
    elapsed = min(duration, max(0., elapsed))
    if elapsed < ramp:
        return .5 * peak / ramp * elapsed**2
    if elapsed > duration - ramp:
        return 1 - .5 * peak / ramp * (duration - elapsed)**2
    return peak * (elapsed - .5 * ramp)


class ServoJointRobot:
    """Adapt legacy command handlers to bounded MoveJS streaming, without SDK edits."""

    def __init__(
        self,
        robot,
        demo,
        events,
        *,
        options=None,
        repeat_final=True,
        batch_limits=False,
        sleep=time.sleep,
        wallclock=time.time,
        monotonic=time.monotonic,
    ):
        self.robot, self.demo, self.events = robot, demo, events
        self.sleep, self.wallclock, self.monotonic = sleep, wallclock, monotonic
        self.speed = 5
        self.options = delivery_options(options)
        self.repeat_final = repeat_final
        self.batch_limits = batch_limits
        self.on_motion_tick = None

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def set_speed_percent(self, speed):
        if not 1 <= speed <= 100:
            raise ValueError("MoveJS speed must be 1..100")
        self.speed = speed
        self.robot.set_speed_percent(speed)

    def check_feedback(self, status=None):
        status = status if status is not None else self.robot.get_arm_status()
        joints = self.robot.get_joint_angles()
        limit = self.options['feedback_freshness_limit_s']
        fresh = True
        for label, feedback in (("status", status), ("joints", joints)):
            stamp = self.demo.feedback_stamp(feedback)
            if stamp is None or not 0 <= self.wallclock() - stamp <= limit:
                fresh = False
                break
        if fresh:
            state, joint_values = status.msg, list(joints.msg)
        else:
            # Cached readers (fast_cached_feedback) can lag the CAN feedback
            # stream. One bounded fresh read—the fast_feedback.arm_snapshot
            # fallback pattern—before declaring delivery failure.
            joint_values, _, state = self.demo.arm_snapshot(self.robot)
        self.demo.check_comm(self.robot)
        if (
            state.arm_status != 0
            or state.ctrl_mode != 1
            or getattr(state, "mode_feedback", None) != 1
            or state.motion_status not in (0, 1)
            or self.robot.get_joints_enable_status_list() != [True] * 7
        ):
            raise RuntimeError(
                "MoveJS delivery: require NORMAL/CAN joint mode and seven enabled joints"
            )
        if len(joint_values) != 7 or not all(math.isfinite(x) for x in joint_values):
            raise RuntimeError("MoveJS delivery: invalid joint feedback")
        return list(joint_values)

    def move_js(self, target):
        target = tuple(float(x) for x in target)
        if len(target) != 7 or not all(math.isfinite(x) for x in target):
            raise ValueError("MoveJS delivery requires seven finite absolute targets")
        event = dict(
            target_q_rad=list(target),
            sdk_method="move_js",
            sent_count=0,
            mode_confirmed=False,
            delivery_completed=False,
        )
        self.events.append(event)
        automatic = self.robot.get_auto_set_motion_mode_enabled()
        try:
            self.robot.set_auto_set_motion_mode_enabled(False)
            self.robot.set_motion_mode("js")
            status = self.demo.read_fresh(self.robot.get_arm_status, 1.0, "MoveJS mode")
            start = self.check_feedback(status)
            event["mode_confirmed"] = True
            velocity, acceleration, bounds = [], [], []
            event["live_limits"] = []
            limit_started = self.monotonic()
            if self.batch_limits:
                from cup_grasp_demo.flow.batched_limits import read_limits
                pairs = read_limits(self.robot, sleep=self.sleep, monotonic=self.monotonic)
            else:
                pairs = None
            for i in range(7):
                if pairs is None:
                    v = self.robot.get_joint_angle_vel_limits(i + 1, timeout=0.5, min_interval=0)
                    a = self.robot.get_joint_acc_limits(i + 1, timeout=0.5, min_interval=0)
                else:
                    v, a = pairs[i]
                if v is None or a is None:
                    raise RuntimeError(f"J{i + 1}: missing live limits for MoveJS")
                low, high = v.msg.min_angle_limit, v.msg.max_angle_limit
                vel, acc = v.msg.max_joint_spd, a.msg.max_joint_acc
                if (
                    not all(math.isfinite(x) for x in (low, high, vel, acc))
                    or not low <= start[i] <= high
                    or not low <= target[i] <= high
                    or vel <= 0
                    or acc <= 0
                ):
                    raise ValueError(f"J{i + 1}: invalid MoveJS path or limits")
                event["live_limits"].append(dict(joint=i+1, min_angle_rad=low,
                    max_angle_rad=high, max_velocity_rad_s=vel, max_acceleration_rad_s2=acc))
                # Host interpolation enforces speed; do not rely on JS speed-percent behavior.
                budget = self.options['limit_utilization']
                speed_budget = vel * self.speed / 100 * budget
                cap = self.options['velocity_cap_deg_s']
                velocity.append(speed_budget if cap is None else min(speed_budget, math.radians(cap)))
                acceleration.append(min(acc, self.options['acceleration_cap_rad_s2']) * budget)
                bounds.append((low, high))
            event['limits_read_s'] = self.monotonic() - limit_started
            event['limits_batched'] = self.batch_limits
            duration = smooth_duration(start, target, velocity, acceleration)
            profile = self.options['profile']
            if profile == 'trapezoid':
                duration, ramp, normalized_peak = trapezoid_profile(start, target, velocity, acceleration)
            else:
                normalized_peak = 1.875 / duration
            if duration > 120:
                raise ValueError("MoveJS transition exceeds 120 seconds")
            if max(
                abs(a - b) for a, b in zip(start, self.check_feedback())
            ) > math.radians(0.1):
                raise RuntimeError(
                    "MoveJS starting posture changed while querying limits"
                )
            event.update(duration_s=duration, profile=profile, start_q_rad=start,
                         acceleration_budget_rad_s2=acceleration,
                         tracking_error_action=self.options['tracking_error_action'],
                         tracking_exceeded_samples=0, tracking_max_error_deg=0.)
            event["motion_started_epoch_s"] = self.wallclock()
            began = last_time = self.monotonic()
            peak_speed = normalized_peak * max(abs(b-a) for a, b in zip(start, target))
            sample_period = min(0.02, math.radians(0.5) / peak_speed) if peak_speed else 0.02
            max_phase_step = math.radians(0.9) / peak_speed if peak_speed else 0.08
            phase_elapsed = 0.0
            event.update(sample_period_s=sample_period, phase_delay_s=0.0)
            last = start
            while True:
                now = self.monotonic()
                if now - last_time > 0.08:
                    raise RuntimeError(
                        "MoveJS scheduling gap exceeds 80 ms; no catch-up jump"
                    )
                actual = self.check_feedback()
                margin = math.radians(self.options['envelope_margin_deg'])
                for i, (a, b, observed) in enumerate(zip(start, target, actual)):
                    if not bounds[i][0] <= observed <= bounds[i][1]:
                        raise RuntimeError(f'J{i+1}: actual angle outside controller limits')
                    if not min(a, b)-margin <= observed <= max(a, b)+margin:
                        raise RuntimeError(f'J{i+1}: actual angle outside planned envelope')
                error_deg = math.degrees(max(abs(a - b) for a, b in zip(actual, last)))
                event['tracking_max_error_deg'] = max(event['tracking_max_error_deg'], error_deg)
                if error_deg > self.options['tracking_error_deg']:
                    event['tracking_exceeded_samples'] += 1
                    if self.options['tracking_error_action'] == 'stop':
                        raise RuntimeError('MoveJS transition tracking error exceeds configured threshold')
                elapsed = now - last_time
                advance = min(elapsed, max_phase_step)
                phase_elapsed += advance
                event['phase_delay_s'] += elapsed - advance
                t = min(phase_elapsed / duration, 1.0)
                u = 10 * t**3 - 15 * t**4 + 6 * t**5
                if profile == 'trapezoid':
                    u = trapezoid_position(t * duration, duration, ramp, normalized_peak)
                q = [a + (b - a) * u for a, b in zip(start, target)]
                if max(abs(a - b) for a, b in zip(q, last)) > math.radians(1):
                    raise RuntimeError("MoveJS command step exceeds 1 degree")
                self.robot.move_js(q)
                event["sent_count"] += 1
                if self.on_motion_tick is not None:
                    self.on_motion_tick()
                last, last_time = q, now
                if t >= 1:
                    break
                self.sleep(sample_period)
            # Repeat the final streamed sample so all four joint packets can settle.
            for _ in range(2 if self.repeat_final else 0):
                self.sleep(0.02)
                self.check_feedback()
                self.robot.move_js(list(target))
                event["sent_count"] += 1
            event["delivery_completed"] = True
            event["actual_delivery_s"] = self.monotonic() - began
        except BaseException as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.robot.set_auto_set_motion_mode_enabled(automatic)

    # The shared Nero CLI still names its command "move-j". All its joint writes
    # are redirected here; this alias never invokes the SDK move_j method.
    move_j = move_js


class JSModeProxy:
    """Translate only the legacy CAN handoff mode, scoped to this session."""

    def __init__(self, robot):
        self.robot = robot

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def set_motion_mode(self, mode):
        self.robot.set_motion_mode("js" if mode == "j" else mode)


def take_js_control(core, session, baseline, *, timeout_s):
    original = session.robot
    session.robot = JSModeProxy(original)
    try:
        return core.take_can_control(session, baseline, timeout_s=timeout_s)
    finally:
        session.robot = original


def fresh_js_hold(core, session, limits, *, joint_max_age_s=.1):
    feedback = core.fresh_feedback(session, joint_max_age_s=joint_max_age_s)
    if (
        feedback["status"]["arm_status"] != 0
        or feedback["status"]["ctrl_mode"] != 1
        or feedback["enabled"] != [True] * 7
    ):
        raise RuntimeError(
            "Cannot request MoveJS hold without fresh enabled CAN feedback"
        )
    target = core.validate_target(feedback["q_rad"], limits)
    session.robot.set_joint_limits_enabled(True)
    session.robot.move_js(target)
    return dict(requested=True, sdk_method="move_js", target_rad=target)
