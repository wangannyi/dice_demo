"""Evaluate saved residuals against the active workflow without rewriting history."""
import copy
import hashlib
import json
import math


def thresholds(config):
    values = config.get('quality_thresholds', {})
    return {'position_m': values.get('position_mm', 5.)/1000,
            'angle_deg': values.get('angle_deg', 2.)}


def evaluate(data, config):
    if not isinstance(data, dict):
        raise ValueError('缺少有效标定结果，无法评估质量')
    result = copy.deepcopy(data)
    if 'source_calibration' in result:
        result['source_calibration'] = evaluate(result['source_calibration'], config)
        result['quality_passed'] = result['source_calibration']['quality_passed']
        return result
    limits = thresholds(config)
    # Incomplete historical evidence must never turn a failure into a pass.
    groups = [result.get(key) for key in ('all_sample_errors_m_deg', 'holdout_errors_m_deg')]
    complete = all(isinstance(group, list) and group for group in groups)
    if complete:
        complete = all(isinstance(pair, (list, tuple)) and len(pair) == 2
                       and all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in pair)
                       for group in groups for pair in group)
    if complete:
        result['quality_passed'] = all(p <= limits['position_m'] and a <= limits['angle_deg']
                                       for group in groups for p, a in group)
    else:
        result['quality_passed'] = False
    result['thresholds'] = limits
    return result


def approval_key(data, config):
    evaluated = evaluate(data, config)
    quality = evaluated.get('source_calibration', evaluated)
    return hashlib.sha256(json.dumps(quality, sort_keys=True, allow_nan=False).encode()).hexdigest()
