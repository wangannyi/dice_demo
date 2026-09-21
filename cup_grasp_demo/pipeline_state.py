"""Pure, opt-in state transitions for the green-cup NERO demo.

Each motion is one separately reviewed ``move_j`` or Revo2 hand command.
An acknowledgement of position never substitutes for observed cup contact or
hold.  After a scene change the RGB-D frame and plan are invalidated.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


PHASES = (
    'WAIT_SCENE_CLEAR', 'HOME', 'WAIT_CONTROL_HANDOFF',
    'CAPTURE', 'LOCALIZE', 'PLAN',
    'APPROACH', 'CONTACT', 'GRIP', 'VERIFY_GRIP', 'SHAKE', 'REVEAL',
    'VERIFY_REVEAL',
    'DONE', 'PAUSED', 'ERROR',
)
APPROACH_STAGES = ('prep', 'clearance', 'pretop', 'guarded_hover')
PROBE_STAGES = ('probe_1', 'probe_2', 'contact_candidate')
SHAKE_STAGES = ('tiny_shake_a', 'tiny_shake_b')
REVEAL_STAGES = ('reveal_lift', 'reveal_exit')


class PipelineTransitionError(ValueError):
    """The event cannot safely advance this pipeline."""


def new_state(config_sha256: str) -> dict[str, Any]:
    if len(config_sha256) != 64:
        raise PipelineTransitionError('Configuration needs a SHA-256 identity')
    return {
        'schema': 1, 'kind': 'nero_green_cup_state_machine',
        'config_sha256': config_sha256, 'phase': 'WAIT_SCENE_CLEAR',
        'scene_id': None, 'snapshot_frame_id': None, 'plan_sha256': None,
        'target_red_roi_xyxy': None, 'target_red_roi_sha256': None,
        'path_scene_sha256': None,
        'scene_observed_at_ns': None, 'home_arrival_time_ns': None,
        'home_arrival_joints_rad': None,
        'home_source': None, 'user_designated_home_joints_rad': None,
        'home_move_count': 0, 'approach_completed': [], 'probe_completed': [],
        'reviewed_approach_sequence': None,
        'shake_completed': [], 'reveal_completed': [],
        'reveal_last_arrival_time_ns': None, 'contact_observed': False,
        'hold_observed': False, 'hold_observed_at_ns': None,
        'cup_revealed': False,
        'events': [], 'pause_reason': None, 'command_may_have_been_sent': False,
    }


def _require_evidence(evidence: dict[str, Any], *names: str) -> None:
    if not isinstance(evidence, dict) or any(evidence.get(name) is None for name in names):
        raise PipelineTransitionError('Event lacks evidence: ' + ', '.join(names))


def _append(state: dict[str, Any], event: str, evidence: dict[str, Any]) -> None:
    state['events'].append({'event': event, 'phase_after': state['phase'],
                            'evidence': deepcopy(evidence)})


def _seven_joints(value: Any, name: str) -> tuple[float, ...]:
    if (not isinstance(value, (list, tuple)) or len(value) != 7
            or any(type(item) not in (int, float) or not math.isfinite(item)
                   for item in value)):
        raise PipelineTransitionError(f'{name} needs seven finite radians')
    return tuple(float(item) for item in value)


def apply_event(state: dict[str, Any], event: str,
                evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a new state after one verified event; never mutate the input."""
    state = deepcopy(state)
    evidence = {} if evidence is None else deepcopy(evidence)
    phase = state.get('phase')
    if state.get('schema') != 1 or state.get('kind') != 'nero_green_cup_state_machine' \
            or phase not in PHASES:
        raise PipelineTransitionError('Invalid pipeline state')
    if event == 'pause':
        _require_evidence(evidence, 'reason')
        state['phase'] = 'PAUSED'
        state['pause_reason'] = str(evidence['reason'])
        state['command_may_have_been_sent'] = bool(
            evidence.get('command_may_have_been_sent', False))
        _append(state, event, evidence)
        return state
    if phase == 'PAUSED':
        if event != 'reset_after_scene_change':
            raise PipelineTransitionError('A paused run needs a new scene and live state')
        _require_evidence(evidence, 'new_scene_id', 'live_arm_status', 'live_camera_frame_id')
        if evidence['new_scene_id'] == state['scene_id']:
            raise PipelineTransitionError('A scene change needs a different scene identity')
        state.update(phase='WAIT_SCENE_CLEAR', scene_id=None,
                     snapshot_frame_id=None, plan_sha256=None,
                     target_red_roi_xyxy=None, target_red_roi_sha256=None,
                     path_scene_sha256=None,
                     scene_observed_at_ns=None, home_arrival_time_ns=None,
                     home_arrival_joints_rad=None,
                     home_source=None, user_designated_home_joints_rad=None,
                     home_move_count=0,
                     reviewed_approach_sequence=None,
                     approach_completed=[], probe_completed=[], shake_completed=[],
                     reveal_completed=[], reveal_last_arrival_time_ns=None,
                     contact_observed=False, hold_observed=False,
                     hold_observed_at_ns=None, cup_revealed=False,
                     pause_reason=None, command_may_have_been_sent=False)
        _append(state, event, evidence)
        return state
    if phase in ('DONE', 'ERROR'):
        raise PipelineTransitionError('A terminal run cannot advance')
    if phase == 'WAIT_SCENE_CLEAR' and event == 'scene_clear':
        _require_evidence(evidence, 'scene_id', 'observed_at_ns',
                          'scene_image_sha256', 'path_scene_sha256')
        state['scene_id'] = evidence['scene_id']
        state['scene_observed_at_ns'] = int(evidence['observed_at_ns'])
        state['path_scene_sha256'] = evidence['path_scene_sha256']
        state['phase'] = 'HOME'
    elif phase == 'HOME' and event == 'home_move_reached':
        _require_evidence(evidence, 'target_joints_rad', 'arrival_joints_rad',
                          'feedback_event', 'observed_at_ns')
        if evidence['feedback_event'] != 'arm_target_reached':
            raise PipelineTransitionError('HOME requires fresh arm arrival feedback')
        requested = _seven_joints(evidence['target_joints_rad'], 'HOME target')
        arrived = _seven_joints(evidence['arrival_joints_rad'], 'HOME arrival')
        if max(abs(a-b) for a, b in zip(requested, arrived)) > math.radians(1.):
            raise PipelineTransitionError('HOME arrival is more than 1 degree from target')
        if int(evidence['observed_at_ns']) < int(state['scene_observed_at_ns']):
            raise PipelineTransitionError('HOME feedback predates scene-clear observation')
        state['home_move_count'] += 1
        if evidence.get('final_home_target') is True:
            _require_evidence(evidence, 'reviewed_final_home_joints_rad',
                              'arm_status', 'ctrl_mode', 'joints_enabled')
            reviewed = _seven_joints(evidence['reviewed_final_home_joints_rad'],
                                     'Reviewed HOME target')
            if max(abs(a-b) for a, b in zip(requested, reviewed)) > math.radians(.05):
                raise PipelineTransitionError('Final HOME target differs from reviewed configuration')
            if (evidence['arm_status'] != 0 or evidence['ctrl_mode'] != 1
                    or evidence['joints_enabled'] != [True] * 7):
                raise PipelineTransitionError('Final HOME needs CAN/NORMAL/all seven axes enabled')
            state['home_arrival_time_ns'] = int(evidence['observed_at_ns'])
            state['home_arrival_joints_rad'] = list(arrived)
            state['home_source'] = 'reviewed_low_speed_move_j_sequence'
            state['phase'] = 'CAPTURE'
    elif phase == 'HOME' and event == 'user_home_designated':
        _require_evidence(evidence, 'user_instruction', 'first_joints_rad',
                          'second_joints_rad', 'observed_at_ns',
                          'scene_photo_sha256', 'arm_status', 'ctrl_mode',
                          'joints_enabled')
        if evidence['user_instruction'] != 'current_pose_as_home':
            raise PipelineTransitionError('Current HOME needs explicit user designation')
        first = _seven_joints(evidence['first_joints_rad'], 'First live HOME read')
        second = _seven_joints(evidence['second_joints_rad'], 'Second live HOME read')
        if (max(abs(a-b) for a, b in zip(first, second)) > math.radians(.1)
                or evidence['arm_status'] != 0
                or evidence['joints_enabled'] != [True]*7
                or int(evidence['observed_at_ns']) <= int(state['scene_observed_at_ns'])
                or not isinstance(evidence['scene_photo_sha256'], str)
                or len(evidence['scene_photo_sha256']) != 64):
            raise PipelineTransitionError('User HOME needs stationary NORMAL/all-seven live evidence')
        state['user_designated_home_joints_rad'] = list(second)
        state['home_source'] = 'user_designated_current_pose_read_only'
        if evidence['ctrl_mode'] == 1:
            state['home_arrival_time_ns'] = int(evidence['observed_at_ns'])
            state['home_arrival_joints_rad'] = list(second)
            state['phase'] = 'CAPTURE'
        elif evidence['ctrl_mode'] == 3:
            state['phase'] = 'WAIT_CONTROL_HANDOFF'
        else:
            raise PipelineTransitionError('Current HOME needs CAN or WEB mode identification')
    elif phase == 'WAIT_CONTROL_HANDOFF' and event == 'home_control_verified':
        _require_evidence(evidence, 'first_joints_rad', 'second_joints_rad',
                          'observed_at_ns', 'scene_photo_sha256',
                          'arm_status', 'ctrl_mode', 'joints_enabled')
        first = _seven_joints(evidence['first_joints_rad'], 'First CAN handoff read')
        second = _seven_joints(evidence['second_joints_rad'], 'Second CAN handoff read')
        designated = _seven_joints(state['user_designated_home_joints_rad'],
                            'User designated HOME')
        if (evidence['ctrl_mode'] != 1 or evidence['arm_status'] != 0
                or evidence['joints_enabled'] != [True]*7
                or max(abs(a-b) for a, b in zip(first, second)) > math.radians(.1)
                or max(abs(a-b) for a, b in zip(second, designated)) > math.radians(.1)
                or int(evidence['observed_at_ns']) <= int(state['scene_observed_at_ns'])
                or not isinstance(evidence['scene_photo_sha256'], str)
                or len(evidence['scene_photo_sha256']) != 64):
            raise PipelineTransitionError('CAN handoff changed the user designated HOME')
        state['home_arrival_time_ns'] = int(evidence['observed_at_ns'])
        state['home_arrival_joints_rad'] = list(second)
        state['phase'] = 'CAPTURE'
    elif phase == 'CAPTURE' and event == 'rgbd_captured':
        _require_evidence(evidence, 'frame_id', 'metadata_sha256',
                          'color_sha256', 'depth_sha256', 'captured_at_ns',
                          'target_red_roi_xyxy', 'target_red_roi_sha256',
                          'path_scene_sha256')
        if (state['home_arrival_time_ns'] is None or
                int(evidence['captured_at_ns']) <= state['home_arrival_time_ns']):
            raise PipelineTransitionError('Formal RGB-D capture must follow current HOME arrival')
        state['snapshot_frame_id'] = evidence['frame_id']
        bbox = evidence['target_red_roi_xyxy']
        if (not isinstance(bbox, list) or len(bbox) != 4
                or any(type(v) is not int for v in bbox)
                or not (0 <= bbox[0] < bbox[2] and 0 <= bbox[1] < bbox[3])):
            raise PipelineTransitionError('Formal red ROI needs valid xyxy image bounds')
        for name in ('target_red_roi_sha256', 'path_scene_sha256'):
            if not isinstance(evidence[name], str) or len(evidence[name]) != 64:
                raise PipelineTransitionError(f'{name} needs SHA-256')
        state['target_red_roi_xyxy'] = list(bbox)
        state['target_red_roi_sha256'] = evidence['target_red_roi_sha256']
        state['path_scene_sha256'] = evidence['path_scene_sha256']
        state['phase'] = 'LOCALIZE'
    elif phase == 'LOCALIZE' and event == 'visible_top_localized':
        _require_evidence(evidence, 'frame_id', 'top_center_camera_m',
                          'top_center_base_m', 'top_quality')
        if (evidence['frame_id'] != state['snapshot_frame_id']
                or evidence['top_quality'].get('valid') is not True):
            raise PipelineTransitionError('Measured top must be valid in the current RGB-D frame')
        state['phase'] = 'PLAN'
    elif phase == 'PLAN' and event == 'top_plan_ready':
        _require_evidence(evidence, 'frame_id', 'plan_sha256', 'execute_ready',
                          'reviewed_approach_sequence')
        if (evidence['frame_id'] != state['snapshot_frame_id']
                or evidence['execute_ready'] is not True):
            raise PipelineTransitionError('Plan must be ready and bound to the current RGB-D frame')
        approach = evidence['reviewed_approach_sequence']
        expected = ['clearance', 'pretop', 'guarded_hover']
        if not isinstance(approach, list) or approach not in (expected, ['prep', *expected]):
            raise PipelineTransitionError('Plan must name the exact reviewed top approach stages')
        state['plan_sha256'] = evidence['plan_sha256']
        state['reviewed_approach_sequence'] = list(approach)
        state['phase'] = 'APPROACH'
    elif phase == 'APPROACH' and event == 'approach_stage_reached':
        _require_evidence(evidence, 'stage', 'plan_sha256', 'feedback_event')
        if (evidence['plan_sha256'] != state['plan_sha256']
                or evidence['feedback_event'] != 'arm_target_reached'):
            raise PipelineTransitionError('Approach needs current-plan arrival feedback')
        stages = state['reviewed_approach_sequence']
        if (evidence.get('reviewed_stage_sequence') != stages
                or len(state['approach_completed']) >= len(stages)
                or evidence['stage'] != stages[len(state['approach_completed'])]):
            raise PipelineTransitionError('Approach stage is missing or out of order')
        state['approach_completed'].append(evidence['stage'])
        if len(state['approach_completed']) == len(stages):
            state['phase'] = 'CONTACT'
    elif phase == 'CONTACT' and event == 'probe_stage_reached':
        _require_evidence(evidence, 'stage', 'plan_sha256', 'feedback_event')
        if (len(state['probe_completed']) >= len(PROBE_STAGES)
                or evidence['plan_sha256'] != state['plan_sha256']
                or evidence['feedback_event'] != 'arm_target_reached'
                or evidence['stage'] != PROBE_STAGES[len(state['probe_completed'])]):
            raise PipelineTransitionError('Probe needs the next current-plan arm arrival')
        state['probe_completed'].append(evidence['stage'])
    elif phase == 'CONTACT' and event == 'physical_contact_observed':
        _require_evidence(evidence, 'observed_at_ns', 'scene_image_sha256', 'at_stage')
        latest = (state['probe_completed'][-1] if state['probe_completed']
                  else state['approach_completed'][-1])
        if evidence['at_stage'] != latest:
            raise PipelineTransitionError('Contact must refer to the current verified hover/probe')
        state['contact_observed'] = True
        state['phase'] = 'GRIP'
    elif phase == 'GRIP' and event == 'hand_close_reached':
        _require_evidence(evidence, 'plan_sha256', 'feedback_event', 'six_hand_positions')
        if (not state['contact_observed'] or evidence['plan_sha256'] != state['plan_sha256']
                or evidence['feedback_event'] != 'hand_target_reached'
                or len(evidence['six_hand_positions']) != 6):
            raise PipelineTransitionError('Grip needs observed contact and six-channel hand feedback')
        state['phase'] = 'VERIFY_GRIP'
    elif phase == 'VERIFY_GRIP' and event == 'cup_hold_observed':
        _require_evidence(evidence, 'observed_at_ns', 'scene_image_sha256',
                          'cup_pose_before_m', 'cup_pose_after_m')
        if evidence.get('held') is not True:
            raise PipelineTransitionError('Hand target alone does not prove a held cup')
        state['hold_observed'] = True
        state['hold_observed_at_ns'] = int(evidence['observed_at_ns'])
        state['phase'] = 'SHAKE'
    elif phase == 'SHAKE' and event == 'tiny_shake_stage_reached':
        _require_evidence(evidence, 'stage', 'plan_sha256', 'feedback_event',
                          'hold_observed_at_ns')
        if (not state['hold_observed']
                or int(evidence['hold_observed_at_ns']) != state['hold_observed_at_ns']
                or evidence['plan_sha256'] != state['plan_sha256']
                or evidence['feedback_event'] != 'arm_target_reached'
                or len(state['shake_completed']) >= len(SHAKE_STAGES)
                or evidence['stage'] != SHAKE_STAGES[len(state['shake_completed'])]):
            raise PipelineTransitionError('Shake needs a held cup and the next verified small step')
        state['shake_completed'].append(evidence['stage'])
        if len(state['shake_completed']) == len(SHAKE_STAGES):
            state['phase'] = 'REVEAL'
    elif phase == 'REVEAL' and event == 'reveal_stage_reached':
        _require_evidence(evidence, 'stage', 'plan_sha256', 'feedback_event',
                          'hold_observed_at_ns', 'observed_at_ns')
        if (not state['hold_observed']
                or int(evidence['hold_observed_at_ns']) != state['hold_observed_at_ns']
                or evidence['plan_sha256'] != state['plan_sha256']
                or evidence['feedback_event'] != 'arm_target_reached'
                or len(state['reveal_completed']) >= len(REVEAL_STAGES)
                or evidence['stage'] != REVEAL_STAGES[len(state['reveal_completed'])]):
            raise PipelineTransitionError('Reveal needs held cup and next verified lift or exit')
        state['reveal_completed'].append(evidence['stage'])
        state['reveal_last_arrival_time_ns'] = int(evidence['observed_at_ns'])
        if len(state['reveal_completed']) == len(REVEAL_STAGES):
            state['phase'] = 'VERIFY_REVEAL'
    elif phase == 'VERIFY_REVEAL' and event == 'cup_revealed':
        _require_evidence(evidence, 'plan_sha256', 'scene_image_sha256',
                          'dice_region_clear', 'hold_observed_at_ns', 'observed_at_ns')
        if (not state['hold_observed']
                or int(evidence['hold_observed_at_ns']) != state['hold_observed_at_ns']
                or evidence['plan_sha256'] != state['plan_sha256']
                or evidence['dice_region_clear'] is not True
                or state['reveal_completed'] != list(REVEAL_STAGES)
                or int(evidence['observed_at_ns']) <= state['reveal_last_arrival_time_ns']):
            raise PipelineTransitionError('Reveal needs post-exit observation of clear dice region')
        state['cup_revealed'] = True
        state['phase'] = 'DONE'
    else:
        raise PipelineTransitionError(f'{event} cannot advance {phase}')
    _append(state, event, evidence)
    return state
