"""Exercise control boundaries without importing an SDK or opening hardware."""
import copy
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from calibration.web import service as module
from calibration.web.common import ROOT, check_pin, digest, pin_record, read, write
from calibration.web.service import Conflict, Service


@pytest.fixture
def service(tmp_path, monkeypatch):
    cfg = read(ROOT/'configs/calibration_workflow.json')
    cfg.update(state_file=str(tmp_path/'cli-state.json'), result=None, plan=None,
               registration=None, output_root=str(tmp_path/'cli-output'))
    write(tmp_path/'config.json', cfg)
    monkeypatch.setattr(module, 'occupied', lambda **kw: [])
    monkeypatch.setattr(module, 'device_lock', lambda channel: tmp_path/'device.lock')
    service = Service(tmp_path/'config.json', tmp_path/'web')
    yield service
    if service.active:
        service.active['process'].poll.return_value = 0
        write(service.active['job']/'outcome.json', {'status': 'cancelled'})
        service._tick()
    service.close()


@pytest.fixture
def process(monkeypatch):
    child = Mock(pid=123456789)
    child.poll.return_value = None
    spawn = Mock(return_value=child)
    monkeypatch.setattr(module.subprocess, 'Popen', spawn)
    return child, spawn


def body(**values):
    return {'prepared': True, 'motion_confirmed': True, **values}


def failed_result(service):
    result = service.data/'failed.json'
    write(result, {'quality_passed': False, 'all_sample_errors_m_deg': [[.012, 2.4]]})
    write(service.state_path, {'result': str(result)})
    return result


def test_pin_hash_no_cleartext_and_validation():
    record = pin_record('12345678')
    assert check_pin('12345678', record)
    assert not check_pin('12345679', record)
    assert '12345678' not in json.dumps(record)
    for value in ('123', 'abcdefgh', '１２３４５６７８'):
        with pytest.raises(ValueError):
            pin_record(value)


def test_readonly_blocks_all_jobs(service, process):
    service.readonly = True
    with pytest.raises(Conflict, match='只读'):
        service.start('first', body(), 'owner', {})
    process[1].assert_not_called()


def test_busy_is_checked_before_device_open(service, process, monkeypatch):
    monkeypatch.setattr(module, 'occupied', lambda **kw: [{'pid': 24}])
    with pytest.raises(Conflict, match='24'):
        service.start('first', body(), 'owner', {})
    process[1].assert_not_called()


def test_preparation_motion_confirmation_and_recovery(service, process):
    with pytest.raises(ValueError, match='准备'):
        service.start('first', {}, 'owner', {})
    with pytest.raises(ValueError, match='运动'):
        service.start('auto', {'prepared': True}, 'owner', {})
    service.set_recovery(True)
    with pytest.raises(Conflict, match='检查'):
        service.start('first', body(), 'owner', {})
    process[1].assert_not_called()


def test_one_owner_cannot_heartbeat_or_command_another_job(service, process):
    service.start('first', body(), 'one:tab1', {})
    write(service.active['job']/'stage.json', {'script': 'calibration/calibrate.py'})
    with pytest.raises(Conflict):
        service.heartbeat('one:tab2')
    with pytest.raises(Conflict):
        service.cancel('one:tab2')
    with pytest.raises(Conflict):
        service.command('one:tab2', 'sample')
    with pytest.raises(Conflict):
        service.start('inspect', body(), 'two:tab1', {})
    service.command('one:tab1', 'sample')
    with pytest.raises(Conflict):
        service.command('one:tab1', 'sample')
    service.heartbeat('one:tab1')


def test_cancellation_keeps_locks_until_exit_and_does_not_resume(service, process):
    service.start('first', body(), 'owner', {})
    active = service.active
    active['last_heartbeat'] = time.monotonic()-11
    service._tick()
    assert active['record']['status'] == 'stopping'
    assert (active['job']/'cancel.json').exists()
    service.heartbeat('owner')
    assert active['record']['status'] == 'stopping'
    assert not active['device'].closed
    with pytest.raises(Conflict):
        service.command('owner', 'sample')
    process[0].poll.return_value = 130
    service._tick()
    assert service.active is None
    assert active['device'].closed
    assert active['workflow_lock'].closed
    assert service.recovery_required


def test_quality_gate_binds_pin_reason_and_exact_result(service, process):
    path = failed_result(service)
    for session, data in [({}, body()),
                          ({'engineer_until': time.monotonic()+600}, body(reason='接受临时结果')),
                          ({'engineer_until': time.monotonic()+600}, body(reason='接受临时结果', approved_sha256='old', quality_confirmed=True))]:
        with pytest.raises(ValueError, match='工程师'):
            service.start('apply', data, 'owner', session)
    process[1].assert_not_called()
    service.start('apply', body(reason='现场验证临时精度', approved_sha256=digest(path), quality_confirmed=True),
                  'owner', {'engineer_until': time.monotonic()+600})
    request = read(service.active['job']/'request.json')
    assert request['approved_sha256'] == digest(path)
    assert read(service.active['job']/'config.json')['allow_provisional'] is True
    assert read(path)['quality_passed'] is False
    assert '现场验证临时精度' in (service.data/'audit.jsonl').read_text()


def test_good_result_never_inherits_global_provisional_permission(service, process):
    path = failed_result(service)
    write(path, {'quality_passed': True, 'thresholds': {'position_m': .015, 'angle_deg': 5.}, 'all_sample_errors_m_deg': [[.001, .2]], 'holdout_errors_m_deg': [[.001, .2]]})
    service.start('apply', body(), 'owner', {})
    assert read(service.active['job']/'config.json')['allow_provisional'] is False
    assert process[1].call_args.kwargs['start_new_session'] is True
    assert len(process[1].call_args.kwargs['pass_fds']) == 3


def test_unlock_rate_limit(service):
    write(service.data/'engineer.json', pin_record('12345678'))
    for _ in range(5):
        with pytest.raises(ValueError, match='PIN'):
            service.authenticate({}, '00000000')
    with pytest.raises(ValueError, match='频繁'):
        service.authenticate({}, '12345678')


def test_window_is_bound_to_new_teaching_camera_and_invalidates_plan(service):
    dataset = service.data/'datasets'/'first'
    dataset.mkdir(parents=True)
    write(dataset/'manifest.json', {'camera': {'width': 960, 'height': 720}, 'board': {'type': 'charuco'}})
    write(service.state_path, {'teaching_dataset': str(dataset), 'plan': 'old-plan'})
    with pytest.raises(ValueError):
        service.window([0, 0, 961, 720])
    with pytest.raises(ValueError):
        service.window([True, 0, 200, 300])
    service.window([20, 10, 940, 700])
    assert read(dataset/'board_window.json')['camera']['width'] == 960
    assert read(service.state_path)['plan'] is None
    write(service.state_path, {})
    with pytest.raises(ValueError, match='参考轨迹只读'):
        service.window([20, 10, 940, 700])


def test_crash_recovery_requires_inspection(tmp_path):
    config = ROOT/'configs/calibration_workflow.json'
    job = tmp_path/'jobs'/'20261012_000000_12345678'
    write(job/'record.json', {'action': 'auto', 'status': 'running'})
    service = Service(config, tmp_path)
    try:
        assert service.recovery_required
        assert read(job/'record.json')['status'] == 'interrupted'
    finally:
        service.close()


def test_image_traversal_rejected(service):
    with pytest.raises(ValueError):
        service.image_path('preview', '../../engineer.json')


def test_invalidated_state_never_falls_back_to_bundled_calibration(service):
    write(service.state_path, {'teaching_calibration': None, 'plan': None})
    with pytest.raises(ValueError, match='teaching_calibration'):
        service.selected('teaching_calibration')
    assert service.snapshot()['actions']['plan']['ready'] is False


def test_motion_exit_without_stationary_proof_requires_recovery(service, process, monkeypatch):
    monkeypatch.setattr(service, 'source', lambda action: None)
    service.start('auto', body(), 'owner', {})
    write(service.active['job']/'outcome.json', {'status': 'completed'})
    process[0].poll.return_value = 0
    service._tick()
    assert service.recovery_required


def test_approval_is_reused_across_steps_but_not_changed_result_or_threshold(service, process):
    path = failed_result(service)
    session = {'engineer_until': time.monotonic()+600}
    service.start('apply', body(reason='管理员接受临时精度', approved_sha256=digest(path), quality_confirmed=True), 'owner', session)
    write(service.active['job']/'outcome.json', {'status': 'completed'})
    process[0].poll.return_value = 0
    service._tick()
    assert service.source('apply')['approved']
    assert not service.source('apply')['approval_required']
    process[0].poll.return_value = None
    service.start('apply', body(), 'another-page', {})
    assert read(service.active['job']/'record.json')['provisional_reason'] == '管理员接受临时精度'
    write(service.active['job']/'outcome.json', {'status': 'completed'})
    process[0].poll.return_value = 0
    service._tick()
    service.config['quality_thresholds']['angle_deg'] = 4
    assert service.source('apply')['approval_required']
    service.config['quality_thresholds']['angle_deg'] = 5
    write(path, {'quality_passed': False, 'all_sample_errors_m_deg': [[.030, 8]]})
    assert service.source('apply')['approval_required']


def test_noncritical_preparation_does_not_request_approval(service, process):
    path = failed_result(service)
    write(service.state_path, {'teaching_calibration': str(path)})
    assert not service.source('plan')['approval_required']
    service.start('plan', body(), 'owner', {})
    assert read(service.active['job']/'config.json')['allow_provisional']
    assert not (service.data/'quality-approvals.json').exists()


def test_inspection_ready_requires_completed_camera_joints_and_stop_proof(service):
    job = service.data/'jobs'/'20261012_000000_12345678'
    write(job/'record.json', {'action': 'inspect', 'status': 'completed'})
    write(job/'frame.json', {'time': time.time()})
    write(job/'progress.json', {'joints_deg': [0]*7})
    assert not service.snapshot()['inspection_ready']
    write(job/'stop.json', {'confirmed': True})
    assert service.snapshot()['inspection_ready']
    service.set_recovery(True)
    assert not service.snapshot()['inspection_ready']
    service.set_recovery(False)
    write(job/'record.json', {'action': 'inspect', 'status': 'cancelled'})
    assert not service.snapshot()['inspection_ready']


def test_live_connection_requires_fresh_camera_and_joint_feedback(service, process):
    service.start('inspect', body(), 'owner', {})
    job = service.active['job']
    assert not service.snapshot()['inspection_connected']
    write(job/'frame.json', {'time': time.time()})
    write(job/'progress.json', {'time': time.time(), 'joints_deg': [0]*7})
    assert service.snapshot()['inspection_connected']
    assert not service.snapshot()['inspection_ready']
    write(job/'frame.json', {'time': time.time()-4})
    assert not service.snapshot()['inspection_connected']
    write(job/'frame.json', {'time': time.time()})
    write(job/'progress.json', {'time': time.time()-4, 'joints_deg': [0]*7})
    assert not service.snapshot()['inspection_connected']
    write(job/'progress.json', {'time': time.time(), 'joints_deg': [0]*6})
    assert not service.snapshot()['inspection_connected']
    write(job/'progress.json', {'time': time.time(), 'joints_deg': [0]*7})
    write(job/'camera_error.json', {'error': 'camera disconnected'})
    assert not service.snapshot()['inspection_connected']
    (job/'camera_error.json').unlink()
    service.cancel('owner')
    assert not service.snapshot()['inspection_connected']


def test_home_motion_confirmation_and_pending_home_blocks_registration(service, process):
    write(service.state_path, {'home_required':True})
    for action in ('register', 'apply', 'table'):
        with pytest.raises(Conflict, match='HOME'):
            service.start(action, body(), 'owner', {})
    with pytest.raises(ValueError, match='运动'):
        service.start('home', {'prepared':True}, 'owner', {})
    process[1].assert_not_called()
    service.start('home', body(), 'owner', {})
    assert service.active['record']['action'] == 'home'
    assert service.active['device'] is not None


def visibility_fixture(service):
    teaching = service.data/'datasets'/'teach'
    current = service.data/'datasets'/'auto'
    camera = {'serial':'test', 'width':960, 'height':720, 'crop_xywh':[220,0,960,720],
              'camera_matrix':[[1,0,0],[0,1,0],[0,0,1]]}
    for path in (teaching, current):
        write(path/'manifest.json', {'camera':camera})
    write(teaching/'board_window.json', {'window_xyxy':[15,10,849,703], 'camera':camera})
    write(service.state_path, {'teaching_dataset':str(teaching), 'dataset':str(current)})
    return teaching, current, camera


def test_visibility_prefers_teaching_image_and_keeps_exact_saved_coordinates(service):
    teaching, current, _ = visibility_fixture(service)
    (teaching/'sample_0000.png').write_bytes(b'image')
    (current/'sample_0000.png').write_bytes(b'image')
    v = service.visibility()
    assert v['available'] and v['window_xyxy'] == [15,10,849,703]
    assert (v['width'],v['height']) == (960,720)
    assert service.image_path('visibility') == teaching/'sample_0000.png'
    assert service.active is None


def test_visibility_fallback_requires_matching_camera_profile(service):
    teaching, current, camera = visibility_fixture(service)
    (current/'sample_0000.png').write_bytes(b'image')
    assert service.image_path('visibility') == current/'sample_0000.png'
    assert '非实时' in service.visibility()['image_label']
    for key, value in [('serial','different'), ('width',1280), ('crop_xywh',[0,0,960,720]),
                       ('camera_matrix',[[2,0,0],[0,2,0],[0,0,1]])]:
        write(current/'manifest.json', {'camera':{**camera,key:value}})
        assert service.visibility()['image_path'] is None
        with pytest.raises(ValueError, match='采样图'):
            service.image_path('visibility')


@pytest.mark.parametrize('box', [None, [0,0,961,720], [10,10,0,0], [True,0,900,700]])
def test_visibility_does_not_invent_bounds_for_invalid_windows(service, box):
    teaching, _, camera = visibility_fixture(service)
    write(teaching/'board_window.json', {'window_xyxy':box,'camera':camera})
    assert service.visibility()['available'] is False


def test_live_window_requires_matching_camera_not_only_resolution(service):
    _, _, camera = visibility_fixture(service)
    job = service.data/'jobs'/'20261012_140000_12345678'
    write(job/'record.json', {'id':job.name,'action':'inspect','status':'completed','started':1})
    def frame(profile):
        write(job/'frame.json', {'time':time.time(),'width':960,'height':720,'camera':profile})
        return service.snapshot()['visibility']['live_compatible']
    assert frame(camera)
    assert not frame({**camera,'crop_xywh':[0,0,960,720]})
    assert not frame({**camera,'serial':'other'})
    assert not frame(None)
