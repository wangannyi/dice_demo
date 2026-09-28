"""Runtime feedback budgets must not contaminate validated trajectory options."""
import copy
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cup_grasp_demo.flow import joint_execution as execution
from cup_grasp_demo.flow import joint_profile as profile


def make_plan():
    limits = [dict(joint=j, min_angle_rad=-3, max_angle_rad=3,
                   max_velocity_rad_s=4, max_acceleration_rad_s2=5)
              for j in range(1, 8)]
    return profile.make_plan(
        dict(success=True, q_after_rad=[0.] * 7, limits=limits),
        dict(joints=[1], amplitude_deg=[2.], velocity_deg_s=[20.],
             acceleration_deg_s2=[60.], cycles=1), [(-3, 3)] * 7)


@pytest.mark.parametrize('budget', [.1, .3])
def test_run_revalidates_live_plan_without_mutating_parameters(budget):
    plan = make_plan()
    before = copy.deepcopy(plan)
    request = dict(plan=plan, channel='can0', load_context='green_cup_held',
                   feedback_freshness_limit_s=budget)
    robot = Mock()
    robot.get_joint_angle_vel_limits.return_value = SimpleNamespace(msg=SimpleNamespace(
        min_angle_limit=-3, max_angle_limit=3, max_joint_spd=4))
    robot.get_joint_acc_limits.return_value = SimpleNamespace(msg=SimpleNamespace(max_joint_acc=5))
    original_live_plan = execution.live_plan
    verified = []

    def check_live(robot, candidate):
        verified.append(original_live_plan(robot, candidate))
        raise RuntimeError('test stops before motion')

    with ExitStack() as stack:
        stack.enter_context(patch.object(execution, 'validate_request',
                                         side_effect=lambda req, now: profile.options(req['plan']['parameters'])))
        stack.enter_context(patch.object(execution.core, 'load_sdk_runtime', return_value=(Mock(), Mock())))
        stack.enter_context(patch.object(execution.core, 'PassivePoseSession'))
        stack.enter_context(patch.object(execution, 'JointGuard'))
        stack.enter_context(patch.object(execution, 'persistent_stopped_window', return_value=[]))
        stack.enter_context(patch.object(execution, 'check_start_rows'))
        stack.enter_context(patch.object(execution.core, 'joint_limits', return_value=[(-3, 3)] * 7))
        stack.enter_context(patch.object(execution.shared, 'control_evidence', return_value={}))
        stack.enter_context(patch.object(execution.shared, 'control_conflicts', return_value=[]))
        stack.enter_context(patch.object(execution, 'live_plan', side_effect=check_live))
        measured = stack.enter_context(patch.object(execution, 'measurements', return_value=[]))
        # FK checks are independent of this regression; no SDK or bus is used.
        plan['model_flange_checks'] = []
        before['model_flange_checks'] = []
        report = execution.run(request, connected=robot, connection_evidence={})
    assert report['error'] == 'RuntimeError: test stops before motion'
    assert len(verified) == 1
    assert plan == before
    assert report['motion_attempted'] is False
    robot.move_js.assert_not_called()
    robot.set_speed_percent.assert_not_called()
    measured.assert_called_once_with([], plan, max_age_s=budget)


def test_measurement_budget_is_explicit_and_default_remains_strict():
    plan = make_plan()
    rows = [dict(elapsed_s=i * .05, feedback=dict(
        q_rad=[i * .01] * 7, observed_epoch_s=100. + i * .05,
        sdk_snapshot=dict(packet_timestamps_after_epoch_s={
            'joint_12': 100. + i * .05 - .2}))) for i in range(8)]
    assert profile.measurements(rows, plan)['joints'] == []
    assert len(profile.measurements(rows, plan, max_age_s=.3)['joints']) == 1
    with pytest.raises(ValueError, match='Unknown joint-test'):
        profile.options(dict(plan['parameters'], feedback_freshness_limit_s=.3))
