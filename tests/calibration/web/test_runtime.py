"""Fault-injection tests for fresh feedback and separate camera publication."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from calibration.web import runtime
from calibration.web.common import ROOT, read, write
from calibration.web.runtime import BrowserPreview, LiveCamera, confirm_hold, confirm_stationary


class FakeClock:
    def __init__(self):
        self.value = 0.
    def monotonic(self):
        self.value += .1
        return self.value


def arm_with(q, stamps):
    arm = Mock()
    arm.read_joints.side_effect = [{'joints_rad': q(i), 'sdk_timestamp': s} for i, s in enumerate(stamps)]
    arm.robot.has_comm_error.return_value = False
    arm.robot.get_arm_status.return_value = SimpleNamespace(msg=SimpleNamespace(motion_status=0))
    return arm


def test_stopped_requires_fresh_stationary_feedback(monkeypatch):
    monkeypatch.setattr(runtime.time, 'monotonic', FakeClock().monotonic)
    arm = arm_with(lambda i: [0.]*7, range(20))
    proof = confirm_hold(arm)
    assert proof['confirmed']
    arm.robot.move_js.assert_called_once_with([0.]*7)
    assert proof['stationary_window_s'] >= .5


def test_repeated_or_moving_feedback_never_confirms(monkeypatch):
    monkeypatch.setattr(runtime.time, 'monotonic', FakeClock().monotonic)
    with pytest.raises(RuntimeError, match='时间戳'):
        confirm_stationary(arm_with(lambda i: [0.]*7, [1]*20))
    with pytest.raises(RuntimeError, match='静止'):
        confirm_stationary(arm_with(lambda i: [i*.01]*7, range(100)), timeout=1.)


def test_preview_records_each_frame_and_rejects_missing_board(tmp_path):
    (tmp_path/'commands').mkdir()
    write(tmp_path/'commands'/'001.json', {'command': 'sample'})
    camera = SimpleNamespace(K=np.eye(3), D=np.zeros(5), capture=lambda: np.zeros((20, 30, 3), np.uint8))
    detector = Mock()
    detector.detect.side_effect = ValueError('No markers')
    records = []
    def on_frame(*args):
        records.append(args)
        if len(records) == 2:
            write(tmp_path/'commands'/'002.json', {'command': 'finish'})
    view = BrowserPreview(tmp_path)
    assert view.read_command(camera, detector, 0, on_frame=on_frame) == 'q'
    assert len(records) == 2
    assert read(tmp_path/'progress.json')['last_accepted'] is False
    assert read(tmp_path/'detection.json')['detected'] is False


def test_camera_publishes_without_motion_loop(tmp_path):
    class Camera:
        K, D, info = np.eye(3), np.zeros(5), {'serial': 'fake'}
        def capture(self):
            time.sleep(.01)
            return np.full((20, 30, 3), 50, np.uint8)
        def close(self):
            self.closed = True
    base = Camera()
    camera = LiveCamera(base, tmp_path)
    try:
        assert camera.capture().shape == (20, 30, 3)
        deadline = time.monotonic()+2
        while not (tmp_path/'frame.json').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert read(tmp_path/'frame.json')['width'] == 30
        assert (tmp_path/'preview.jpg').stat().st_size > 0
    finally:
        camera.close()
    assert base.closed
    assert not camera.thread.is_alive()


# Fake devices exist only in this test subprocess; production has no simulated
# success mode or dependency on these fixtures.
FAKE_DEVICES = '''
import sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path.cwd()/'calibration'))
sys.path.insert(0, str(Path.cwd()/'nero_revo2_control'))
import sensors
class Camera:
    def __init__(self, *args, **kwargs):
        self.K=np.array([[100.,0,32],[0,100.,24],[0,0,1.]])
        self.D=np.zeros(5)
        self.info={'serial':'fake','width':64,'height':48}
    def capture(self):
        time.sleep(.03)
        return np.zeros((48,64,3),np.uint8)
    def close(self): pass
class Arm:
    def __init__(self, *args, **kwargs): self.robot=self; self.stamp=0
    def read_joints(self):
        time.sleep(.02)
        self.stamp+=1
        return dict(joints_rad=[0.]*7,joints_deg=[0.]*7,sdk_timestamp=self.stamp)
    def read(self): return np.eye(4)
    def has_comm_error(self): return False
    def get_arm_status(self): return SimpleNamespace(msg=SimpleNamespace(motion_status=0))
    def close(self): pass
sensors.RealSenseCamera=Camera
sensors.NeroFeedback=Arm
from calibration.web.worker import main
main()
'''


def make_job(tmp_path, action):
    job = tmp_path/'job'
    (job/'commands').mkdir(parents=True)
    config = read(ROOT/'configs/calibration_workflow.json')
    config.update(state_file=str(tmp_path/'state.json'), output_root=str(tmp_path/'datasets'),
                  measurement=None, hand_board='calibration/config/board_4x5_21p5mm.json')
    write(job/'config.json', config)
    write(job/'request.json', {'action': action, 'heartbeat_timeout': 1 if action == 'inspect' else 10})
    write(job/'heartbeat.json', {'monotonic': time.monotonic()})
    return job, config


def test_real_worker_heartbeat_loss_cleans_up_and_verifies_stop(tmp_path):
    job, _ = make_job(tmp_path, 'inspect')
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-c', FAKE_DEVICES, str(job)], cwd=ROOT,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    assert result.returncode == 0, result.stdout.decode()
    assert read(job/'outcome.json')['status'] == 'cancelled', result.stdout.decode()
    assert read(job/'stop.json')['confirmed'] is True
    assert (job/'cancel.json').exists()


def test_worker_uses_original_solver_without_hardware(tmp_path):
    from core import pose_matrix, inverse
    job, config = make_job(tmp_path, 'solve')
    dataset = tmp_path/'input'
    dataset.mkdir()
    write(dataset/'manifest.json', {'schema':1,'mode':'eye_to_hand','T_flange_tcp':np.eye(4).tolist(),
          'tcp':'flange','board':{},'camera':{'serial':'synthetic'}})
    rng = np.random.default_rng(4)
    bc=pose_matrix([.42,-.18,.7,.2,-.3,.4]); fm=pose_matrix([.02,.03,.11,-.4,.2,.1])
    for i in range(24):
        bf=pose_matrix([*rng.uniform(-.3,.3,3),*rng.uniform(-.8,.8,3)])
        write(dataset/f'sample_{i:04d}.json', {'T_base_flange':bf.tolist(),'T_camera_board':(inverse(bc)@bf@fm).tolist()})
    write(Path(config['state_file']), {'dataset':str(dataset)})
    env = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-m', 'calibration.web.worker', str(job)], cwd=ROOT,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    assert result.returncode == 0, result.stdout.decode()
    assert read(job/'outcome.json')['status'] == 'completed', result.stdout.decode()
    state=read(config['state_file'])
    solved=read(state['result'])
    assert solved['quality_passed'] is True
    np.testing.assert_allclose(solved['T_base_camera'], bc, atol=1e-8)
    assert not (job/'preview.jpg').exists()


@pytest.mark.parametrize('observation', ['missing', 'fixed_table', 'correct'])
def test_auto_attachment_gate_runs_before_can_control(tmp_path, monkeypatch, observation):
    import sensors
    import calibrate
    import preview
    import auto_collect
    from calibration.web.worker import install_adapters
    job = tmp_path/'job'
    job.mkdir()
    calibration = tmp_path/'calibration.json'
    ident = np.eye(4).tolist()
    write(calibration, {'T_base_camera': ident, 'T_flange_tcp': ident, 'T_tcp_board': ident})
    plan = tmp_path/'plan.json'
    write(plan, {'calibration': str(calibration), 'board': {}})
    class Camera:
        def __init__(self, *args, **kwargs):
            self.K, self.D, self.info = np.eye(3), np.zeros(5), {}
        def capture(self):
            time.sleep(.01)
            return np.zeros((20,30,3), np.uint8)
        def close(self): pass
    detector = Mock()
    observed = np.eye(4)
    if observation == 'missing':
        detector.detect.side_effect = ValueError('No board')
    else:
        if observation == 'fixed_table':
            observed[0,3] = .3
        detector.detect.return_value = observed, {'corners':12}, np.zeros((20,30,3), np.uint8)
    ready = Mock()
    # Register original attributes with monkeypatch so scoped worker adapters
    # cannot escape into the remaining legacy regression tests.
    for obj, name, value in [(sensors,'RealSenseCamera',Camera),
                             (sensors,'NeroFeedback',sensors.NeroFeedback),
                             (sensors,'CharucoDetector',lambda cfg: detector),
                             (calibrate,'capture_sample',calibrate.capture_sample),
                             (preview,'CollectionPreview',preview.CollectionPreview),
                             (auto_collect,'AutoCollectionView',auto_collect.AutoCollectionView),
                             (auto_collect,'ensure_can_control',ready)]:
        monkeypatch.setattr(obj, name, value)
    _, factory, cameras, _ = install_adapters(job, {'action':'auto'}, {'plan':str(plan)})
    factory()
    arm = SimpleNamespace(read=lambda:np.eye(4))
    try:
        if observation != 'correct':
            with pytest.raises(ValueError):
                auto_collect.ensure_can_control(arm)
            ready.assert_not_called()
        else:
            auto_collect.ensure_can_control(arm)
            ready.assert_called_once_with(arm)
            for camera in cameras:
                camera.close()
            # Return HOME must not repeat the pre-sampling board gate after
            # auto_collect has closed its camera; readiness is still checked.
            auto_collect.ensure_can_control(arm)
            assert ready.call_count == 2
            detector.detect.assert_called_once()
    finally:
        for camera in cameras:
            camera.close()


def test_web_auto_returns_home_only_after_successful_solve(tmp_path, monkeypatch):
    from calibration.web.worker import BrowserWorkflow
    from calibration.workflow import Workflow
    job, config = make_job(tmp_path, 'auto')
    flow = BrowserWorkflow(job)
    order = []
    def run(self, action, execute=False):
        order.append(('collect_and_solve', action))
        self.save(result='saved-result.json')
    monkeypatch.setattr(Workflow, 'run', run)
    monkeypatch.setattr(BrowserWorkflow, 'return_to_home', lambda self: order.append(('home', self.state['result'])))
    flow.run('auto', execute=True)
    assert order == [('collect_and_solve', 'auto'), ('home', 'saved-result.json')]
    assert read(config['state_file'])['home_required'] is True  # only confirmed HOME clears it
    for error in (KeyboardInterrupt(), RuntimeError('capture failed')):
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(Workflow, 'run', fail)
        order.clear()
        with pytest.raises(type(error)):
            flow.run('auto', execute=True)
        assert order == []
    with pytest.raises(ValueError, match='确认'):
        flow.run('home')


@pytest.mark.parametrize('feedback', ['home', 'elsewhere', 'stale'])
def test_home_completion_requires_arrival_and_fresh_stationary_feedback(tmp_path, monkeypatch, feedback):
    import home_start
    import sensors
    from calibration.web import worker
    job, config = make_job(tmp_path, 'home')
    flow = worker.BrowserWorkflow(job)
    target = home_start.load_home(ROOT/config['home'])
    move = Mock(return_value=target)
    monkeypatch.setattr(home_start, 'return_home', move)
    arm = Mock()
    monkeypatch.setattr(sensors, 'NeroFeedback', Mock(return_value=arm))
    proof = {'confirmed':True, 'joints_rad': target if feedback == 'home' else [v+.1 for v in target],
             'sdk_timestamp':100, 'stationary_window_s':.6}
    confirm = Mock(return_value=proof)
    if feedback == 'stale':
        confirm.side_effect = RuntimeError('stale timestamp')
    monkeypatch.setattr(worker, 'confirm_stationary', confirm)
    if feedback == 'home':
        flow.return_to_home()
        assert read(config['state_file'])['home_required'] is False
        assert read(job/'home_return.json')['status'] == 'completed'
        assert read(job/'home_return.json')['joint_error_deg'] == 0
    else:
        with pytest.raises(RuntimeError):
            flow.return_to_home()
        assert read(config['state_file'])['home_required'] is True
        assert read(job/'home_return.json')['status'] == 'running'
    arm.close.assert_called_once()
    move.assert_called_once()
