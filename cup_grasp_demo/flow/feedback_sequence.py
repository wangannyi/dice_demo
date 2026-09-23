"""Pure configuration validation for a two-pose feedback hand sequence."""
import math


def sequence_values(raw, initial, completion_duration):
    if raw is None:
        return [list(initial)], completion_duration
    if not isinstance(raw, dict):
        raise ValueError('hand_sequence must be an object')
    poses = raw.get('poses')
    cycles = raw.get('cycles', 3)
    interval = raw.get('interval_s', completion_duration)
    return_to_initial = raw.get('return_to_initial', True)
    if (not isinstance(poses, list) or len(poses) != 2
            or any(not isinstance(p, list) or len(p) != 6 or
                   any(type(v) is not int or not 0 <= v <= 100 for v in p) for p in poses)):
        raise ValueError('hand_sequence.poses requires two six-channel poses')
    if poses[0] != initial:
        raise ValueError('hand_0_100 must equal the first sequence pose')
    if type(cycles) is not int or not 1 <= cycles <= 20:
        raise ValueError('hand_sequence.cycles must be 1..20')
    if type(return_to_initial) is not bool:
        raise ValueError('hand_sequence.return_to_initial must be boolean')
    # A new target may intentionally overlap the preceding physical movement.
    # Revo2 position/time frame pairs are sent atomically; 50 ms prevents a
    # transition from overtaking its predecessor without forcing us to wait
    # for that predecessor to reach its final pose.
    if (type(interval) not in (int, float) or not math.isfinite(interval)
            or not .05 <= interval <= 5):
        raise ValueError('hand_sequence.interval_s must be 0.05..5')
    # Default: A, B, A, B, A ...; one cycle is A -> B -> A. A one-way
    # reveal may set return_to_initial=false and finish at B.
    pair = (poses[1], poses[0]) if return_to_initial else (poses[1],)
    return [list(poses[0])] + [list(p) for _ in range(cycles) for p in pair], interval
