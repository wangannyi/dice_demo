from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import nero_revo2_demo as demo
from calibration.web import hand_home
from calibration.web.common import read, write


@pytest.mark.parametrize('mode', ['no_feedback', 'nonzero', 'stale', 'moving', 'cancel', 'send_error', 'slow_send'])
def test_zero_command_without_hand_feedback(tmp_path, monkeypatch, mode):
    clock = [100.]
    monkeypatch.setattr(hand_home.time, 'time', lambda: clock[0])
    monkeypatch.setattr(hand_home.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(hand_home.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    hand = Mock()
    hand.get_finger_pos.return_value = None if mode == 'no_feedback' else SimpleNamespace(
        timestamp=1. if mode == 'stale' else 100.,
        msg=SimpleNamespace(**dict.fromkeys(demo.FINGER_NAMES, 200)))
    hand.is_ok.return_value = False
    def send(**kwargs):
        if mode == 'send_error':
            raise RuntimeError('send failed')
        if mode == 'slow_send':
            clock[0] += .1
    hand.position_time_ctrl.side_effect = send
    robot = Mock()
    robot.has_comm_error.return_value = False
    robot.get_arm_status.return_value = SimpleNamespace(msg=SimpleNamespace(motion_status=0, arm_status=0, ctrl_mode=1))
    def joints():
        if mode == 'cancel' and hand.position_time_ctrl.called:
            raise KeyboardInterrupt
        return {'joints_rad': [.1 if mode == 'moving' and hand.position_time_ctrl.called else 0.]*7}
    arm = SimpleNamespace(robot=robot, _home_hand=hand, read_joints=joints)
    if mode in ['no_feedback', 'nonzero', 'stale']:
        proof = hand_home.reset_hand(arm, tmp_path)
        assert proof['command_sent'] and proof['status'] == 'completed'
        assert proof['confirmed'] is False and proof['verification'] == 'command_only'
        assert proof['completed_at'] - proof['commanded_at'] >= 1.
        hand.position_time_ctrl.assert_any_call(mode='time', **dict.fromkeys(demo.FINGER_NAMES, 100))
    else:
        with pytest.raises(KeyboardInterrupt if mode == 'cancel' else RuntimeError):
            hand_home.reset_hand(arm, tmp_path)
        assert read(tmp_path/'hand_home.json')['status'] == 'failed'
        assert read(tmp_path/'hand_home.json')['confirmed'] is False
    hand.position_time_ctrl.assert_any_call(mode='pos', **dict.fromkeys(demo.FINGER_NAMES, 0))
    hand.get_finger_pos.assert_not_called()
    hand.get_hand_status.assert_not_called()
    hand.is_ok.assert_not_called()
    robot.move_js.assert_not_called()


def test_arm_home_cannot_clear_pending_hand_check(tmp_path):
    import time
    from calibration.web.worker import reconcile_home, BrowserWorkflow
    from calibration.web.common import ROOT
    from home_start import load_home
    job = tmp_path/'job'
    job.mkdir()
    config = read(ROOT/'configs/calibration_workflow.json')
    config['state_file'] = str(tmp_path/'state.json')
    write(job/'config.json', config)
    write(job/'request.json', {'action': 'home'})
    write(config['state_file'], {'hand_home_required': True, 'home_required': True})
    reconcile_home(job, config, dict(confirmed=True, observed_at=time.time(),
                   sdk_timestamp=time.time(), stationary_window_s=.6,
                   joints_rad=load_home(ROOT/config['home'])))
    assert read(config['state_file'])['home_required'] is True
    flow = BrowserWorkflow(job)
    flow.save(home_required=False)
    assert read(config['state_file'])['home_required'] is True
    state = read(config['state_file'])
    state.update(hand_home_required=False, hand_home={'confirmed': False, 'command_sent': True, 'verification': 'command_only'})
    write(config['state_file'], state)
    flow.save(home_required=False)
    assert read(config['state_file'])['home_required'] is False
    assert read(config['state_file'])['hand_home'] == {'confirmed': False, 'command_sent': True, 'verification': 'command_only'}


@pytest.mark.parametrize('action', ['auto', 'home', 'restore', 'inspect'])
def test_home_adapter_changes_only_web_motion_actions(tmp_path, monkeypatch, action):
    import home_start, sensors, calibrate, auto_collect, preview
    from calibration.web.worker import install_adapters
    for obj, name in ((sensors,'RealSenseCamera'), (sensors,'NeroFeedback'),
                      (calibrate,'capture_sample'), (auto_collect,'capture_sample'),
                      (auto_collect,'AutoCollectionView'), (auto_collect,'ensure_can_control'),
                      (preview,'CollectionPreview')):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    order = []
    original = Mock(side_effect=lambda *a, **k: order.append('arm') or [0]*7)
    monkeypatch.setattr(home_start, 'move_home', original)
    reset = Mock(side_effect=lambda *a: order.append('hand') or {'confirmed': False, 'command_sent': True, 'verification': 'command_only'})
    monkeypatch.setattr(hand_home, 'reset_hand', reset)
    state = tmp_path/'state.json'
    install_adapters(tmp_path, {'action': action}, {'state_file': str(state)})
    arm = Mock()
    assert home_start.move_home(arm, [], 4, 6) == [0]*7
    assert order == (['arm'] if action == 'inspect' else ['arm', 'hand'])
    if action != 'inspect':
        assert read(state)['hand_home_required'] is False
        reset.side_effect = RuntimeError('hand timeout')
        with pytest.raises(RuntimeError, match='hand timeout'):
            home_start.move_home(arm, [], 4, 6)
        assert read(state)['home_required'] is True
        assert read(state)['hand_home_required'] is True
