"""A lift accepts configured small drift without runtime geometric replanning."""
import math
from types import SimpleNamespace
import pytest
from cup_grasp_demo.flow.hardware import request_start_tolerance, validate_start, StartPositionChanged


def config():
    return dict(pipeline_strategy='green_open_cup', start_tolerance_deg=.25,
                green_cup=dict(lift_start_tolerance_deg=.5))


@pytest.mark.parametrize('degrees', [.137, .206, .254, .298, .5])
def test_small_lift_drift_is_accepted(degrees):
    plan = dict(kind='green_arm_plan', start_q_rad=[0.]*7)
    state = SimpleNamespace(arm_status=0, motion_status=0, ctrl_mode=1)
    current = [0.]*5 + [math.radians(degrees), 0.]
    bound = request_start_tolerance({'allow_lift_start_drift': True}, plan, config())
    validate_start(plan, current, state, [True]*7, bound)
    assert plan['start_q_rad'] == [0.]*7


def test_large_drift_and_faults_still_stop():
    plan = dict(kind='green_arm_plan', start_q_rad=[0.]*7)
    state = SimpleNamespace(arm_status=0, motion_status=0, ctrl_mode=1)
    with pytest.raises(StartPositionChanged):
        validate_start(plan, [0.]*5+[math.radians(.501), 0.], state, [True]*7, .5)
    for field in ('arm_status', 'motion_status'):
        setattr(state, field, 1)
        with pytest.raises(RuntimeError):
            validate_start(plan, [0.]*7, state, [True]*7, .5)
        setattr(state, field, 0)
    with pytest.raises(RuntimeError):
        validate_start(plan, [0.]*7, state, [False]*7, .5)


def test_other_phases_and_unconfigured_lifts_keep_original_tolerance():
    cfg = config(); plan = dict(kind='green_arm_plan')
    assert request_start_tolerance({}, plan, cfg) == .25
    cfg['green_cup'] = {}
    assert request_start_tolerance({'allow_lift_start_drift': True}, plan, cfg) == .25


@pytest.mark.parametrize('value', [True, 0, .51, float('nan'), '0.5'])
def test_invalid_lift_bound_rejected(value):
    cfg = config(); cfg['green_cup']['lift_start_tolerance_deg'] = value
    with pytest.raises(ValueError):
        request_start_tolerance({'allow_lift_start_drift': True}, dict(kind='green_arm_plan'), cfg)


def test_non_green_plan_cannot_use_lift_override():
    with pytest.raises(ValueError):
        request_start_tolerance({'allow_lift_start_drift': True}, dict(kind='other'), config())
