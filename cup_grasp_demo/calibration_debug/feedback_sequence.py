"""Pure configuration validation for a two-pose feedback hand sequence."""
import math


def sequence_values(raw, initial, minimum_interval):
    if raw is None:
        return [list(initial)], minimum_interval
    if not isinstance(raw, dict):
        raise ValueError('hand_sequence must be an object')
    poses = raw.get('poses')
    cycles = raw.get('cycles', 3)
    interval = raw.get('interval_s', minimum_interval)
    if (not isinstance(poses, list) or len(poses) != 2
            or any(not isinstance(p, list) or len(p) != 6 or
                   any(type(v) is not int or not 0 <= v <= 100 for v in p) for p in poses)):
        raise ValueError('hand_sequence.poses requires two six-channel poses')
    if poses[0] != initial:
        raise ValueError('hand_0_100 must equal the first sequence pose')
    if type(cycles) is not int or not 1 <= cycles <= 20:
        raise ValueError('hand_sequence.cycles must be 1..20')
    if (type(interval) not in (int, float) or not math.isfinite(interval)
            or not minimum_interval <= interval <= 5):
        raise ValueError(f'hand_sequence.interval_s must be {minimum_interval}..5')
    # A, B, A, B, A ...: one cycle is A -> B -> A; finish at A.
    return [list(poses[0])] + [list(p) for _ in range(cycles) for p in (poses[1], poses[0])], interval
