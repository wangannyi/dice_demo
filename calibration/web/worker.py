"""One process per web job. Legacy module adapters exist only in this process."""
import importlib
import signal
import sys
import threading
import time
from pathlib import Path

from calibration.web.common import HARDWARE, MOTION, ROOT, digest, event, read, resolve, write
from calibration.web.quality import evaluate
from calibration.web.runtime import BrowserAutoView, BrowserPreview, LiveCamera, confirm_hold, confirm_stationary
from calibration.workflow import Workflow

sys.path.insert(0, str(ROOT/'calibration'))
sys.path.insert(0, str(ROOT/'nero_revo2_control'))


class BrowserWorkflow(Workflow):
    def __init__(self, job):
        self.job = job
        self.request = read(job/'request.json')
        super().__init__(job/'config.json')
        self.producing = False
        self.source_override = {}

    def return_to_home(self):
        from home_start import load_home, return_home
        from sensors import NeroFeedback
        import math
        self.save(home_required=True)
        event(self.job/'stage.json', script='web_home_return')
        progress = read(self.job/'progress.json', {})
        event(self.job/'progress.json', **{k: v for k, v in progress.items() if k not in ('time', 'phase', 'message')},
              phase='RETURN_HOME', message='标定结果已保留，正在回 HOME；到位并静止后才能继续')
        event(self.job/'home_return.json', status='running')
        home = resolve(self.c['home'])
        target = load_home(home)
        motion = self.c['motion']
        return_home(home, self.c['channel'], motion['speed_percent'],
                    motion['speed_deg_s'], motion['acc_deg_s2'])
        arm = NeroFeedback(self.c['channel'])
        try:
            proof = confirm_stationary(arm)
            error = math.degrees(max(abs(a-b) for a, b in zip(proof['joints_rad'], target)))
            if error > .8:
                raise RuntimeError('未确认 HOME 到位；结果已保留，请重新回 HOME')
            evidence = dict(proof, home_file=str(home), home_sha256=digest(home), joint_error_deg=error)
            event(self.job/'home_return.json', status='completed', **evidence)
            self.save(home_required=False, home_return=evidence)
            event(self.job/'progress.json', **{k: v for k, v in progress.items() if k not in ('time', 'phase', 'message')},
                  phase='HOME_READY', message='已回 HOME 并确认静止，可继续固定板和桌面登记')
        finally:
            arm.close()

    def run(self, action, execute=False):
        if action in ('auto', 'home'):
            if not execute:
                raise ValueError('回 HOME 需要明确确认机械臂运动')
            self.save(home_required=True)
        if action == 'home':
            self.return_to_home()
            return
        super().run(action, execute=execute)
        # Never return automatically after cancellation or a failed capture/solve.
        if action == 'auto':
            self.return_to_home()

    def prepare_source(self):
        source = self.request.get('source_path')
        if not source:
            return
        original = read(source)
        updated = evaluate(original, self.c)
        if original == updated:
            return
        target = self.job/'evaluated-source.json'
        write(target, updated)
        event(self.job/'quality-assessment.json', source_path=source,
              source_sha256=digest(source), evaluated_sha256=digest(target),
              thresholds=self.c.get('quality_thresholds'))
        self.source_override[str(resolve(source))] = target
        if self.request['action'] in ('register', 'apply'):
            # Only the quality assessment changes; geometry and residuals are retained.
            super().save(result=target)

    def selected(self, key):
        # Newly taught data must supersede the bundled reference trajectory.
        value = self.state[key] if key in self.state else self.c.get(key)
        if not value:
            raise ValueError(f'缺少 {key}，请先完成前一步')
        path = resolve(value)
        return self.source_override.get(str(path), path)

    def quality(self, path):
        result = evaluate(read(path), self.c)
        if result.get('quality_passed') is True or self.producing:
            return
        original = self.request.get('source_path')
        derived = original and self.source_override.get(str(resolve(original))) == Path(path)
        if not (digest(path) == self.request.get('approved_sha256') or
                (derived and digest(original) == self.request.get('approved_sha256'))):
            raise ValueError('质量未通过：需要工程师针对当前结果重新确认')

    def solve(self, dataset, teaching=False):
        self.producing = True
        try:
            super().solve(dataset, teaching)
        finally:
            self.producing = False

    def save(self, **values):
        # Legacy save stringifies artifact paths; keep Web motion evidence typed.
        for key in ("home_required", "home_return"):
            if key in values:
                self.state[key] = values.pop(key)
        if 'teaching_dataset' in values:
            for key in ('plan', 'teaching_calibration', 'result', 'registration'):
                self.state[key] = None
        if 'result' in values and self.request['action'] != 'restore':
            self.state['registration'] = None
        super().save(**values)
        event(self.job/'artifacts.json', **self.state)

    def call(self, script, *args, quality_exit=False):
        # Invoke the same CLI entry points in this isolated worker. No shell, no
        # subprocess descendants that could survive a cancelled workflow parent.
        args = list(args)
        for flag in ('--calibration', '--registration', '--result'):
            if flag in args:
                i = args.index(flag)+1
                args[i] = self.source_override.get(str(resolve(args[i])), args[i])
        if script == 'calibration/auto_collect.py' and '--plan' in args and self.source_override:
            i = args.index('--plan')+1
            plan = read(args[i])
            if any(digest(p) != h for p, h in plan['source_sha256'].items()):
                raise ValueError('路线源文件已改变，请重新生成路线')
            replacement = self.source_override.get(str(resolve(plan['calibration'])))
            if replacement:
                plan['calibration'] = str(replacement)
                plan['source_sha256'][str(replacement)] = digest(replacement)
                plan['quality_passed'] = read(replacement)['quality_passed']
                path = self.job/'evaluated-plan.json'
                write(path, plan)
                args[i] = path
        event(self.job/'stage.json', script=script)
        print('步骤:', script, *args, flush=True)
        module = ('calibration.tools.' + Path(script).stem if '/tools/' in script
                  else Path(script).stem)
        try:
            code = importlib.import_module(module).main([str(a) for a in args])
        except SystemExit as exc:
            code = exc.code
        if code not in (None, 0) and not (quality_exit and code == 2):
            raise RuntimeError(f'{script} 失败（{code}），后续步骤未执行')


def install_adapters(job, request, config):
    import sensors
    import calibrate
    import preview
    from core import distance, inverse, matrix
    base_camera, base_arm = sensors.RealSenseCamera, sensors.NeroFeedback
    cameras = []
    stop_evidence = []
    motion_armed = False

    def camera_factory(*args, **kwargs):
        # restore opens the stream before HOME, and observe reuses that same stream.
        for camera in cameras:
            if not camera.closed:
                return camera
        camera = LiveCamera(base_camera(*args, **kwargs), job)
        cameras.append(camera)
        return camera

    class WebArm(base_arm):
        def close(self):
            try:
                if motion_armed and sys.exc_info()[0] is not None:
                    try:
                        proof = confirm_hold(self)
                        stop_evidence.append(proof)
                        event(job/'stop.json', **proof)
                    except Exception as exc:
                        event(job/'stop.json', confirmed=False, error=str(exc))
            finally:
                super().close()

    sensors.RealSenseCamera = camera_factory
    sensors.NeroFeedback = WebArm
    preview.CollectionPreview = lambda: BrowserPreview(job)

    original_capture = calibrate.capture_sample
    previous = []

    def checked_capture(*args, **kwargs):
        sample, image, vis = original_capture(*args, **kwargs)
        for old in previous:
            arm_change = distance(matrix(old['T_base_flange']), matrix(sample['T_base_flange']))
            board_change = distance(matrix(old['T_camera_board']), matrix(sample['T_camera_board']))
            if (arm_change[0] > .01 or arm_change[1] > 5) and board_change[0] < .0008 and board_change[1] < .3:
                raise ValueError('检测到的板没有随末端移动；请遮挡桌面固定板并检查末端板')
        previous.append(sample)
        return sample, image, vis

    calibrate.capture_sample = checked_capture
    import auto_collect
    auto_collect.AutoCollectionView = lambda box, total: BrowserAutoView(job, box, total)
    original_ready = auto_collect.ensure_can_control

    def guarded_ready(arm):
        nonlocal motion_armed
        if request['action'] == 'auto' and not motion_armed:
            plan = read(resolve(config['plan']))
            cal = read(plan['calibration'])
            camera = next(c for c in cameras if not c.closed)
            detector = sensors.CharucoDetector(plan['board'])
            pose, quality, _ = detector.detect(camera.capture(), camera.K, camera.D)
            expected = (inverse(matrix(cal['T_base_camera'])) @ arm.read()
                        @ matrix(cal['T_flange_tcp']) @ matrix(cal['T_tcp_board']))
            pos, angle = distance(expected, pose)
            if pos > .04 or angle > 15:
                raise ValueError('末端板位置与示教安装不一致；可能误识别桌面板。请检查后重新标定')
            event(job/'detection.json', detected=True, quality=quality)
        original_ready(arm)
        motion_armed = True

    auto_collect.ensure_can_control = guarded_ready
    return base_arm, camera_factory, cameras, stop_evidence


def inspect_devices(job, config, base_arm, camera_factory):
    import sensors
    board = read(resolve(config['hand_board']))
    camera = camera_factory(config['serial'], image_profile=board.get('image_profile'))
    detector = sensors.CharucoDetector(board)
    arm = base_arm(config['channel'])
    try:
        while True:
            joints = arm.read_joints()
            image = camera.capture()
            try:
                _, quality, _ = detector.detect(image, camera.K, camera.D)
                event(job/'detection.json', detected=True, quality=quality)
            except ValueError as exc:
                event(job/'detection.json', detected=False, reason=str(exc))
            event(job/'progress.json', phase='inspect', joints_deg=joints['joints_deg'],
                  message='设备反馈已连接；此步骤只读取，不驱动机械臂')
            commands = sorted((job/'commands').glob('*.json'))
            if commands:
                commands[0].unlink()
                return
    finally:
        arm.close()


def main():
    job = Path(sys.argv[1]).resolve()
    request, config = read(job/'request.json'), read(job/'config.json')
    action = request['action']
    cameras, stop_evidence = [], []
    base_arm = None
    outcome = {'status': 'failed', 'error': '作业未完成'}
    # TERM (service shutdown) and INT (owner cancellation) take the same cleanup path.
    def interrupt(signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    finished = threading.Event()
    def watch_owner():
        # Survives a crashed server. The worker keeps inherited device/service
        # locks until cleanup completes; a replacement server cannot race it.
        while not finished.wait(.25):
            heartbeat = read(job/'heartbeat.json', {})
            expired = (time.monotonic()-heartbeat.get('monotonic', 0)
                       > request.get('heartbeat_timeout', 10))
            if expired or (job/'cancel.json').exists():
                if expired:
                    event(job/'cancel.json', reason='控制页面或服务心跳丢失')
                import os
                os.kill(os.getpid(), signal.SIGINT)
                return
    threading.Thread(target=watch_owner, daemon=True).start()
    event(job/'ready.json', ready=True)
    try:
        if (job/'cancel.json').exists():
            raise KeyboardInterrupt
        base_arm, camera_factory, cameras, stop_evidence = install_adapters(job, request, config)
        w = BrowserWorkflow(job)
        if request.get('source_path'):
            if digest(request['source_path']) != request['source_sha256']:
                raise ValueError('源标定文件已改变，请重新检查和确认')
        w.prepare_source()
        # Make selected paths unambiguous to worker adapters and legacy workflow.
        for key in ('plan', 'registration', 'result', 'dataset', 'teaching_dataset', 'teaching_calibration'):
            if w.state.get(key):
                w.c[key] = w.state[key]
        config.update(w.c)
        if action == 'restore':
            board = read(resolve(config['reference_board']))
            camera_factory(config['serial'], image_profile=board.get('image_profile'))
        if action == 'inspect':
            inspect_devices(job, config, base_arm, camera_factory)
        else:
            # An inherited provisional restore retains quality=false. It is still
            # a produced result, never an implicit authorization to apply it.
            w.producing = action == 'restore'
            w.run(action, execute=action in MOTION)
        outcome = {'status': 'completed'}
    except KeyboardInterrupt:
        outcome = {'status': 'cancelled', 'error': '已取消作业；保留已保存数据，不自动续跑'}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        outcome = {'status': 'failed', 'error': str(exc)}
    finally:
        finished.set()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        cleanup_errors = []
        for camera in cameras:
            try:
                camera.close()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        if action in HARDWARE and base_arm is not None:
            arm = None
            try:
                arm = base_arm(config['channel'])
                proof = confirm_stationary(arm)
                event(job/'stop.json', **proof)
            except Exception as exc:
                event(job/'stop.json', confirmed=False, error=str(exc))
                outcome['needs_device_check'] = True
            finally:
                if arm is not None:
                    arm.close()
        if cleanup_errors:
            outcome.update(status='failed', error='设备释放异常：'+'; '.join(cleanup_errors), needs_device_check=True)
        event(job/'outcome.json', **outcome)


if __name__ == '__main__':
    main()
