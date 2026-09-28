"""Real configuration parsing with fake hardware; never sends CAN commands."""
import copy
import io
import json
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cup_grasp_demo.flow import green_hot_reload as hot
from cup_grasp_demo.flow import green_control as control
from cup_grasp_demo.flow.core import ROOT, digest
from cup_grasp_demo.flow.green_pipeline import Workflow
from scripts import action_registry


def save(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    cfg = json.loads((ROOT / 'configs/green_cup.json').read_text())
    strategy = tmp_path / 'strategy.json'
    shake = tmp_path / 'shake.json'
    shutil.copyfile(ROOT / cfg['green_cup']['strategy_file'], strategy)
    shutil.copyfile(ROOT / cfg['green_cup']['joint_test_config'], shake)
    cfg['green_cup'].update(strategy_file=str(strategy), joint_test_config=str(shake))
    config = tmp_path / 'config.json'
    save(config, cfg)
    gestures = tmp_path / 'gestures'
    shutil.copytree(action_registry.GESTURES_DIR, gestures)
    original_loader = action_registry.load_registry
    monkeypatch.setattr(action_registry, 'load_registry', lambda: original_loader(gestures))
    monkeypatch.setattr(hot, 'GESTURES_DIR', gestures)
    flow = Workflow(SimpleNamespace(config=config, session=tmp_path / 'run'))
    flow._sdk = Mock(cfg=flow.cfg)
    flow._vision = Mock(cfg=flow.cfg)
    refresh = hot.ParameterReloader(flow, io.StringIO())
    refresh()
    yield SimpleNamespace(flow=flow, refresh=refresh, cfg=cfg, config=config,
                          strategy=strategy, shake=shake, gestures=gestures)
    flow.close()


def test_no_change_keeps_connections_and_plan_cache(runtime):
    flow = runtime.flow
    sdk, vision, prepared = flow._sdk, flow._vision, flow._prepared
    assert runtime.refresh() is None
    assert flow._sdk is sdk and flow._vision is vision and flow._prepared is prepared
    sdk.close.assert_not_called()
    vision.close.assert_not_called()


def test_speed_offsets_and_shake_swap_together_without_reconnecting(runtime):
    flow = runtime.flow
    sdk, vision, prepared = flow._sdk, flow._vision, flow._prepared
    runtime.cfg['green_cup']['fast_speed_percent'] = 45
    runtime.cfg['green_cup']['place_offset_base_mm'] = [1, 2, 0]
    strategy = json.loads(runtime.strategy.read_text())
    strategy['contact_offset_base_mm'] = [1, 2, 3]
    shake = json.loads(runtime.shake.read_text())
    shake['cycles'] += 1
    save(runtime.config, runtime.cfg)
    save(runtime.strategy, strategy)
    save(runtime.shake, shake)
    _, _, changed = runtime.refresh()
    assert set(changed) == {str(runtime.config), str(runtime.strategy), str(runtime.shake)}
    assert flow.cfg['speed_percent'] == 45
    assert flow.g['place_offset_base_mm'] == [1, 2, 0]
    assert flow.g['contact_offset_base_mm'] == [1, 2, 3]
    assert flow._joint_recipe['cycles'] == shake['cycles']
    assert flow._prepared is not prepared
    assert flow._sdk is sdk and sdk.cfg is flow.cfg
    assert flow._vision is vision and vision.cfg is flow.cfg
    sdk.close.assert_not_called()
    vision.close.assert_not_called()


@pytest.mark.parametrize('failure', ['json', 'speed', 'gesture'])
def test_invalid_update_rejected_without_losing_last_good_snapshot(runtime, failure):
    flow = runtime.flow
    old_cfg, old_sdk, accepted = flow.cfg, flow._sdk, runtime.refresh.accepted
    if failure == 'json':
        runtime.config.write_text('{')
    elif failure == 'speed':
        runtime.cfg['green_cup']['fast_speed_percent'] = 101
        save(runtime.config, runtime.cfg)
    else:
        (runtime.gestures / 'broken.json').write_text('{')
    with pytest.raises(ValueError):
        runtime.refresh()
    assert flow.cfg is old_cfg and flow._sdk is old_sdk
    assert runtime.refresh.accepted is accepted
    old_sdk.call.assert_not_called()
    runtime.cfg['green_cup']['fast_speed_percent'] = 55
    save(runtime.config, runtime.cfg)
    (runtime.gestures / 'broken.json').unlink(missing_ok=True)
    runtime.refresh()
    assert flow.cfg['speed_percent'] == 55


def test_gesture_update_add_remove_and_frozen_execution_snapshot(runtime, monkeypatch):
    from scripts import result_feedback
    execute = Mock()
    monkeypatch.setattr(result_feedback, 'execute_recipe', execute)
    _, old_action = control.build_action_runtime(runtime.flow, io.StringIO())
    data = json.loads((runtime.gestures / 'rps.json').read_text())
    data['gestures']['rock']['speed_percent'] = 75
    save(runtime.gestures / 'rps.json', data)
    extra = dict(speed_percent=100, finger_duration_s=.5,
                 gestures={'test_wave': copy.deepcopy(data['gestures']['rock'])})
    save(runtime.gestures / 'extra.json', extra)
    names, new_action, _ = runtime.refresh()
    assert 'test_wave' in names
    old_action('rock')
    assert execute.call_args.args[0]['speed_percent'] == 100
    new_action('rock')
    assert execute.call_args.args[0]['speed_percent'] == 75
    assert execute.call_args.args[4] is runtime.flow._sdk
    (runtime.gestures / 'extra.json').unlink()
    assert 'test_wave' not in runtime.refresh()[0]


def test_update_during_grasp_applies_only_after_return_home(runtime, monkeypatch, tmp_path):
    records = []
    def perform(flow, phase):
        records.append((phase, flow.cfg['speed_percent'], flow._joint_recipe['cycles']))
    monkeypatch.setattr(Workflow, 'perform', perform)
    flow = runtime.flow
    out = io.StringIO()
    session = control.ControlSession(flow, tmp_path / 'state.json', io.StringIO(), out,
                                     io.StringIO(), refresh_parameters=runtime.refresh)
    session._handle(dict(command='advance', until='GRIP'))
    original_speed, original_cycles = records[-1][1:]
    runtime.cfg['green_cup']['fast_speed_percent'] = 50
    save(runtime.config, runtime.cfg)
    shake = json.loads(runtime.shake.read_text())
    shake['cycles'] += 1
    save(runtime.shake, shake)
    session._handle(dict(command='advance', until='RETURN_HOME'))
    assert all(speed == original_speed and cycles == original_cycles
               for _, speed, cycles in records)
    session._handle(dict(command='advance', until='HOME'))
    assert records[-1] == ('HOME', 50, shake['cycles'])


def test_rejected_reload_does_not_dispatch_action_or_recovery(runtime, tmp_path):
    action = Mock()
    session = control.ControlSession(runtime.flow, tmp_path / 'state.json', io.StringIO(),
                                     io.StringIO(), io.StringIO(), actions=['rock'],
                                     run_action=action, refresh_parameters=runtime.refresh)
    runtime.config.write_text('{')
    for request in (dict(command='action', name='rock'), dict(command='advance'),
                    dict(command='reload')):
        assert session._handle(request)
    action.assert_not_called()
    runtime.flow._sdk.call.assert_not_called()
    assert session.next_index == 0 and session.state['status'] == 'WAITING'
    events = [json.loads(x) for x in session.outgoing.getvalue().splitlines()]
    assert [e['code'] for e in events] == ['reload_failed'] * 3


def test_source_and_calibration_changes_still_block(runtime, tmp_path):
    protected = tmp_path / 'protected.py'
    protected.write_text('original')
    runtime.flow.hashes[str(protected)] = digest(protected)
    runtime.flow._file_stats[str(protected)] = runtime.flow.file_stamp(protected)
    protected.write_text('changed')
    with pytest.raises(ValueError, match='运行期间配置/程序发生变化'):
        runtime.refresh()
    runtime.flow._sdk.call.assert_not_called()


@pytest.mark.parametrize('setting', ['channel', 'calibration'])
def test_installation_changes_require_restart(runtime, tmp_path, setting):
    if setting == 'channel':
        runtime.cfg['channel'] = 'can1'
    else:
        path = tmp_path / 'new_calibration.json'
        shutil.copyfile(runtime.flow.cfg['calibration'], path)
        runtime.cfg['calibration'] = str(path)
    save(runtime.config, runtime.cfg)
    with pytest.raises(ValueError, match='重启'):
        runtime.refresh()
    runtime.flow._sdk.call.assert_not_called()


def test_vision_settings_rebuild_vision_but_keep_sdk(runtime):
    flow = runtime.flow
    sdk, vision = flow._sdk, flow._vision
    runtime.cfg['green_cup']['perception']['confidence'] = .4
    save(runtime.config, runtime.cfg)
    runtime.refresh()
    assert flow._vision is None  # Lazily reopened by next HOME, not by gestures.
    vision.close.assert_called_once()
    assert flow._sdk is sdk
    sdk.close.assert_not_called()


def test_partial_write_during_validation_is_not_accepted(runtime, monkeypatch):
    before = runtime.refresh.accepted
    original = control.build_action_runtime
    def racing_build(*args, **kwargs):
        result = original(*args, **kwargs)
        runtime.cfg['green_cup']['fast_speed_percent'] = 42
        save(runtime.config, runtime.cfg)
        return result
    monkeypatch.setattr(control, 'build_action_runtime', racing_build)
    with pytest.raises(ValueError, match='保存尚未完成'):
        runtime.refresh(force=True)
    assert runtime.refresh.accepted is before
    runtime.flow._sdk.call.assert_not_called()


def test_action_refresh_can_restore_unavailable_registry(tmp_path):
    action = Mock(return_value={'name': 'rock'})
    refresh = Mock(return_value=(['rock'], action, ['rps.json']))
    flow = SimpleNamespace()
    session = control.ControlSession(flow, tmp_path / 'state.json', io.StringIO(),
                                     io.StringIO(), io.StringIO(), refresh_parameters=refresh)
    assert session._handle(dict(command='action', name='rock'))
    action.assert_called_once_with('rock')


def test_shake_planner_reads_pinned_recipe_not_edited_file(runtime, monkeypatch):
    from cup_grasp_demo.flow import green_pipeline
    frozen = copy.deepcopy(runtime.flow._joint_recipe)
    runtime.shake.write_text('{')
    plan = Mock(side_effect=RuntimeError('stop before hardware or collision planning'))
    monkeypatch.setattr(green_pipeline, 'joint_plan', plan)
    with pytest.raises(RuntimeError, match='stop before hardware'):
        runtime.flow.build_shake({})
    assert plan.call_args.args[1] == frozen
