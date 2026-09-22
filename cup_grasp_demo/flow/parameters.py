"""Shared configuration validation; standard library only for the SDK process."""

import math


DEFAULT_FINGER_TARGETS = [[0, 100, 0, 0, 0, 0], [100] * 6]


def planar_shake_options(raw):
    """Planar JS budget; shared by the standalone and pipeline entry points."""
    if not isinstance(raw, dict):
        raise ValueError('planar shake options must be an object')
    extra = {'joint_motion_cost', 'joint_acceleration_cap_rad_s2', 'limit_utilization',
             'auto_reduce_frequency'}
    result = shake_options({'shake': {k: v for k, v in raw.items() if k not in extra}})
    if 'limit_utilization' in raw:
        result['limit_utilization'] = number(raw['limit_utilization'], 'limit_utilization', .1, 1.)
    costs = raw.get('joint_motion_cost', [1, 4, 4, 1, 4, 4, 1])
    if not isinstance(costs, list) or len(costs) != 7:
        raise ValueError('joint_motion_cost requires seven costs')
    result['joint_motion_cost'] = [number(v, 'joint_motion_cost', .1, 100) for v in costs]
    result['joint_acceleration_cap_rad_s2'] = number(
        raw.get('joint_acceleration_cap_rad_s2', 5), 'joint_acceleration_cap_rad_s2', .1, 5)
    if 'auto_reduce_frequency' in raw:
        if not isinstance(raw['auto_reduce_frequency'], bool):
            raise ValueError('auto_reduce_frequency must be a boolean')
        result['auto_reduce_frequency'] = raw['auto_reduce_frequency']
    return result


def effective_shake_options(cfg):
    """Validate the selected strategy; configurations without one retain legacy behavior."""
    raw = cfg.get('shake')
    if isinstance(raw, dict) and raw.get('strategy') == 'planar_js':
        return dict(strategy='planar_js', **planar_shake_options(
            {k: v for k, v in raw.items() if k != 'strategy'}))
    return shake_options(cfg)


def shake_options(cfg):
    """Four motion inputs and an optional orientation monitoring threshold."""
    raw = cfg.get('shake')
    names = {'frequency_hz', 'amplitude_mm', 'azimuth_deg', 'duration_s'}
    optional = {'orientation_error_limit_deg', 'feedback_reference_delay_s',
                'tracking_error_limit_mm', 'tracking_error_grace_s',
                'joint_path_tolerance_deg', 'command_step_limit_deg', 'command_lag_limit_s',
                'limit_utilization'}
    if not isinstance(raw, dict) or not names <= set(raw) or set(raw) - names - optional:
        raise ValueError('shake requires frequency_hz, amplitude_mm, azimuth_deg, duration_s')
    result = {key: number(raw[key], 'shake.' + key, low, high) for key, low, high in (
        ('frequency_hz', .1, 5), ('amplitude_mm', 1, 200),
        ('azimuth_deg', -180, 180), ('duration_s', .5, 60))}
    if result['frequency_hz'] * result['duration_s'] < 2:
        raise ValueError('shake duration must contain at least two nominal cycles, including ramps')
    if 'orientation_error_limit_deg' in raw:
        result['orientation_error_limit_deg'] = shake_orientation_limit(raw)
    if 'feedback_reference_delay_s' in raw:
        result['feedback_reference_delay_s'] = number(
            raw['feedback_reference_delay_s'], 'shake.feedback_reference_delay_s', 0, .15)
        if result['feedback_reference_delay_s'] > .1 / result['frequency_hz']:
            raise ValueError('feedback_reference_delay_s must not exceed 10% of a cycle')
    for key, low, high in (
        ('tracking_error_limit_mm', 1, 100), ('tracking_error_grace_s', 0, 1),
        ('joint_path_tolerance_deg', .15, 2), ('command_step_limit_deg', 1, 5),
        ('command_lag_limit_s', .08, .2), ('limit_utilization', .1, .97),
    ):
        if key in raw:
            result[key] = number(raw[key], 'shake.' + key, low, high)
    return result


def shake_orientation_limit(opts):
    """Null records orientation only; omitted fields retain the legacy 1° limit."""
    value = opts.get('orientation_error_limit_deg', 1.0)
    if value is None:
        return None
    value = number(value, 'shake.orientation_error_limit_deg', 0, 180)
    if value == 0:
        raise ValueError('shake.orientation_error_limit_deg must be > 0; use null to disable')
    return value


def hand_contact_allowed(opts):
    enabled = opts.get('allow_hand_cup_contact', False)
    if not isinstance(enabled, bool):
        raise ValueError('side_grasp.allow_hand_cup_contact must be boolean')
    if enabled and opts.get('strategy', 'side_approach') != 'direct_close':
        raise ValueError('allow_hand_cup_contact is only supported by direct_close')
    return enabled


def number(value, name, low, high, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high
            or (integer and int(value) != value)):
        raise ValueError(f'{name} must be {low}..{high}' + (' integer' if integer else ''))
    return int(value) if integer else float(value)


def closure_targets(opts):
    """Only the thumb base moves first; the second step retains its target."""
    thumb = number(opts.get('thumb_base_target_0_100', 100), 'thumb_base_target_0_100', 0, 100, True)
    close = opts.get('close_targets_0_100', [100] * 6)
    if not isinstance(close, (list, tuple)) or len(close) != 6:
        raise ValueError('close_targets_0_100 requires six channels')
    close = [number(v, 'close_targets_0_100', 0, 100, True) for v in close]
    if close[1] != thumb:
        raise ValueError('close_targets_0_100[1] must retain thumb_base_target_0_100')
    return [[0, thumb, 0, 0, 0, 0], close]


def approach_options(opts):
    raw = opts.get('approach', {})
    if not isinstance(raw, dict) or set(raw) - {'enabled', 'start_gap_mm', 'step_mm', 'speed_percent'}:
        raise ValueError('Invalid side_grasp.approach fields')
    enabled = raw.get('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('side_grasp.approach.enabled must be boolean')
    result = dict(enabled=enabled,
                  start_gap_mm=number(raw.get('start_gap_mm', 40), 'approach.start_gap_mm', 5, 100),
                  step_mm=number(raw.get('step_mm', 2), 'approach.step_mm', 1, 5),
                  speed_percent=number(raw.get('speed_percent', 2), 'approach.speed_percent', 1, 5, True))
    if enabled:
        if opts.get('strategy', 'side_approach') != 'direct_close':
            raise ValueError('side_grasp.approach is only supported by direct_close')
        gap = number(opts['close_gap_mm'], 'close_gap_mm', 5, 50)
        if gap >= result['start_gap_mm']:
            raise ValueError('Enabled approach requires close_gap_mm < approach.start_gap_mm')
    return result


def tcp_offset(cfg):
    value = cfg.get('tcp_offset_in_link_mm', [0, 0, 0])
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('tcp_offset_in_link_mm requires three coordinates')
    return [number(v, 'tcp_offset_in_link_mm', -100, 100) for v in value]
