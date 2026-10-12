"""Job ownership, quality approval, device leases and durable operation records."""
import collections
import copy
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from pathlib import Path

from calibration.web.common import (ACTIONS, HARDWARE, MOTION, ROOT, check_pin, device_lock,
                                    digest, event, lock_file, occupied, read, resolve,
                                    result_summary, write)
from calibration.workflow import validate
from calibration.web.quality import evaluate, approval_key

QUALITY_CHECKPOINTS = {'auto', 'restore', 'apply', 'table'}


class Conflict(ValueError):
    pass


class Service:
    def __init__(self, config, data, *, readonly=False, heartbeat_timeout=10.):
        self.config_path, self.data = Path(config).resolve(), Path(data).resolve()
        self.config = read(self.config_path)
        validate(self.config)
        self.readonly, self.timeout = readonly, heartbeat_timeout
        if not 5 <= heartbeat_timeout <= 30:
            raise ValueError('心跳超时应在 5–30 秒之间')
        self.data.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.data, 0o700)
        self.instance_lock = lock_file(self.data/'service.lock')
        self.mutex = threading.RLock()
        self.sessions = {}
        self.attempts = collections.deque()
        self.active = None
        self.closing = False
        self.state_path = self.data/'state.json'
        if not self.state_path.exists():
            # Reuse CLI provenance only when those files actually exist.
            state = read(resolve(self.config['state_file']), {})
            state = {k: str(resolve(v)) for k, v in state.items()
                     if isinstance(v, str) and resolve(v).exists()}
            write(self.state_path, state)
        self.recovery_required = read(self.data/'recovery.json', {}).get('required', False)
        for path in (self.data/'jobs').glob('*/record.json'):
            record = read(path)
            if record['status'] in ('running', 'stopping'):
                record.update(status='interrupted', error='服务中断；不得自动续跑', ended=time.time())
                write(path, record)
                if record['action'] in HARDWARE:
                    self.set_recovery(True)
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.monitor.start()

    def audit(self, kind, **values):
        import json
        with (self.data/'audit.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'kind': kind, **values}, ensure_ascii=False)+'\n')

    def set_recovery(self, value):
        self.recovery_required = value
        write(self.data/'recovery.json', {'required': value})

    def session(self, token=None):
        with self.mutex:
            now = time.monotonic()
            self.sessions = {k: v for k, v in self.sessions.items() if now-v['seen'] < 86400}
            if token not in self.sessions:
                token = secrets.token_urlsafe(32)
                self.sessions[token] = {'csrf': secrets.token_urlsafe(24), 'engineer_until': 0}
            self.sessions[token]['seen'] = now
            return token, self.sessions[token]

    def authenticate(self, session, pin):
        with self.mutex:
            now = time.monotonic()
            while self.attempts and now-self.attempts[0] > 60:
                self.attempts.popleft()
            if len(self.attempts) >= 5:
                raise ValueError('尝试过于频繁，请一分钟后重试')
            self.attempts.append(now)
            if not check_pin(pin, read(self.data/'engineer.json')):
                self.audit('unlock_failed')
                raise ValueError('PIN 不正确，或尚未在 K3 配置工程师 PIN')
            session['engineer_until'] = now+600
            self.audit('engineer_unlocked')

    def selected(self, key):
        state = read(self.state_path, {})
        value = state[key] if key in state else self.config.get(key)
        if not value:
            raise ValueError(f'缺少 {key}，请先完成对应步骤')
        return resolve(value)

    def source(self, action):
        if action in ('register', 'apply', 'table') and read(self.state_path, {}).get('home_required'):
            raise Conflict('请先回 HOME 并确认到位，再进行固定板或桌面登记')
        if action == 'plan':
            path = self.selected('teaching_calibration')
        elif action == 'auto':
            plan = read(self.selected('plan'))
            path = Path(plan['calibration'])
        elif action in ('apply', 'register'):
            path = self.selected('result')
        elif action == 'restore':
            path = self.selected('registration')
        elif action == 'table':
            path = resolve(read(resolve(self.config['pipeline_config']))['calibration'])
        else:
            return None
        data = read(path)
        if data is None:
            raise ValueError('找不到标定源文件，请重新检查前置步骤')
        evaluated = evaluate(data, self.config)
        quality = evaluated.get('source_calibration', evaluated)
        summary = result_summary(path, quality)
        if summary:
            summary.pop('data', None)
        key = approval_key(data, self.config)
        approved = read(self.data/'quality-approvals.json', {}).get(key)
        return {'path': str(path), 'sha256': digest(path), 'approval_key': key,
                'approved': bool(approved), 'approval': approved,
                'approval_required': action in QUALITY_CHECKPOINTS and not approved and not quality.get('quality_passed'),
                'quality_passed': quality.get('quality_passed') is True, 'summary': summary}

    def idle(self):
        processes = occupied()
        if processes:
            raise Conflict('设备被其他程序占用，请正常退出原程序后重新检测：' +
                           '、'.join(str(p['pid']) for p in processes))

    def start(self, action, body, owner, session):
        with self.mutex:
            if self.readonly:
                raise Conflict('当前为只读预览模式，不连接硬件或执行作业')
            if self.closing or self.active:
                raise Conflict('已有作业正在运行或释放设备，请等待结束')
            if action not in ACTIONS:
                raise ValueError('未知操作')
            if action in HARDWARE and body.get('prepared') is not True:
                raise ValueError('请先确认页面上的现场准备事项')
            if action in MOTION and body.get('motion_confirmed') is not True:
                raise ValueError('请明确确认机械臂运动，保持工作区无人且急停可及')
            if self.recovery_required and action in HARDWARE and action != 'inspect':
                raise Conflict('上次设备状态未确认，请先运行设备检查')
            source = self.source(action)
            approval = None
            new_approval = None
            reason = None
            if source and not source['quality_passed']:
                if source['approval_required']:
                    reason = body.get('reason', '')
                    if (session.get('engineer_until', 0) <= time.monotonic()
                        or body.get('approved_sha256') != source['sha256']
                        or not isinstance(reason, str) or not 5 <= len(reason.strip()) <= 500
                        or body.get('quality_confirmed') is not True):
                        raise ValueError('源标定未达标，需要工程师解锁、确认当前误差并填写至少 5 字的原因')
                    new_approval = {'reason': reason.strip(), 'time': time.time(),
                                    'source_sha256': source['sha256'], 'action': action}
                else:
                    reason = (source.get('approval') or {}).get('reason', '准备步骤：保留未达标标记，未授权运动或应用')
                approval = source['sha256']
            if action in ('solve',):
                self.selected('dataset')
            device = workflow_lock = None
            try:
                if action in HARDWARE:
                    self.idle()
                    device = lock_file(device_lock(self.config['channel']))
                    workflow_lock = lock_file(resolve(self.config['state_file']).with_suffix('.lock'))
                    self.idle()
                job_id = time.strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(4)
                job = self.data/'jobs'/job_id
                (job/'commands').mkdir(parents=True)
                request = {'action': action, 'approved_sha256': approval, 'heartbeat_timeout': self.timeout}
                if source:
                    request.update(source_path=source['path'], source_sha256=source['sha256'])
                config = copy.deepcopy(self.config)
                config.update(state_file=str(self.state_path), output_root=str(self.data/'datasets'),
                              allow_provisional=bool(approval), show=True)
                write(job/'config.json', config)
                write(job/'request.json', request)
                event(job/'heartbeat.json', monotonic=time.monotonic())
                record = {'id': job_id, 'action': action, 'status': 'running', 'started': time.time()}
                if approval:
                    record.update(provisional_reason=reason, approved_sha256=approval)
                write(job/'record.json', record)
                self.audit('job_started', job=job_id, action=action, source=source,
                           provisional_reason=reason,
                           pipeline_config=str(resolve(config['pipeline_config'])))
                output = (job/'worker.log').open('wb')
                try:
                    process = subprocess.Popen([sys.executable, '-u', '-m', 'calibration.web.worker', str(job)],
                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                        start_new_session=True, pass_fds=(self.instance_lock.fileno(),) +
                        ((device.fileno(), workflow_lock.fileno()) if device else ()))
                finally:
                    output.close()
                self.active = dict(process=process, job=job, record=record, owner=owner,
                                   last_heartbeat=time.monotonic(), device=device, workflow_lock=workflow_lock,
                                   cancel_at=None)
                if new_approval:
                    approvals = read(self.data/'quality-approvals.json', {})
                    approvals[source['approval_key']] = new_approval
                    write(self.data/'quality-approvals.json', approvals)
                    self.audit('quality_approved', approval_key=source['approval_key'], **new_approval)
                return record
            except BaseException:
                if device:
                    device.close()
                if workflow_lock:
                    workflow_lock.close()
                raise

    def require_owner(self, owner):
        if not self.active or self.active['owner'] != owner:
            raise Conflict('此页面不持有当前作业控制权；请回到启动作业的页面')
        return self.active

    def heartbeat(self, owner):
        with self.mutex:
            active = self.require_owner(owner)
            if active['cancel_at'] is None:
                active['last_heartbeat'] = time.monotonic()
                event(active['job']/'heartbeat.json', monotonic=active['last_heartbeat'])

    def cancel(self, owner=None, reason='用户取消'):
        with self.mutex:
            active = self.require_owner(owner) if owner else self.active
            if active and active['cancel_at'] is None:
                active['cancel_at'] = time.monotonic()
                active['record'].update(status='stopping', cancel_reason=reason)
                write(active['job']/'record.json', active['record'])
                event(active['job']/'cancel.json', reason=reason)
                # Only the worker watchdog delivers SIGINT, once. A second signal
                # from the server could interrupt its holding/cleanup sequence.
                self.audit('cancel_requested', job=active['record']['id'], reason=reason)

    def command(self, owner, command):
        with self.mutex:
            active = self.require_owner(owner)
            if active['cancel_at'] is not None:
                raise Conflict('正在停止，不能继续采样')
            action = active['record']['action']
            if command not in ('sample', 'finish') or action not in ('first', 'inspect'):
                raise ValueError('此作业不接受采样操作')
            if action == 'inspect' and command != 'finish':
                raise ValueError('设备检查不能保存标定样本')
            stage = read(active['job']/'stage.json', {}).get('script')
            if action == 'first' and stage != 'calibration/calibrate.py':
                raise Conflict('采样尚未开始或已进入求解阶段')
            if command == 'sample' and read(active['job']/'progress.json', {}).get('phase') == 'capturing':
                raise Conflict('正在采样，请保持静止，等待当前采样结束')
            pending = list((active['job']/'commands').glob('*.json'))
            if pending:
                raise Conflict('上一条操作尚未完成')
            write(active['job']/'commands'/f'{time.time_ns()}.json', {'command': command})

    def window(self, box):
        with self.mutex:
            if self.readonly or self.active:
                raise Conflict('请在作业结束后保存可见范围')
            dataset = self.selected('teaching_dataset')
            # Never rewrite the bundled historical trajectory used by the CLI.
            if not dataset.is_relative_to(self.data/'datasets'):
                raise ValueError('参考轨迹只读；首次安装采样后可为新数据绘制范围')
            manifest = read(dataset/'manifest.json')
            width, height = manifest['camera']['width'], manifest['camera']['height']
            if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                    or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
                raise ValueError('范围必须为图像内的整数坐标，右下角应大于左上角')
            write(dataset/'board_window.json', {'schema': 1, 'window_xyxy': box,
                                               'camera': manifest['camera'], 'board': manifest['board']})
            state = read(self.state_path, {})
            state['plan'] = None
            write(self.state_path, state)
            self.audit('window_saved', dataset=str(dataset), window_xyxy=box)

    def _tick(self):
        with self.mutex:
            a = self.active
            if not a:
                return
            code = a['process'].poll()
            if code is None:
                if time.monotonic()-a['last_heartbeat'] > self.timeout:
                    self.cancel(reason='控制页面心跳超时或关闭')
                if a['cancel_at'] and time.monotonic()-a['cancel_at'] > 12:
                    a['record']['error'] = '设备尚未确认释放；请现场检查，必要时使用硬件急停'
                    self.set_recovery(True)
                return
            outcome = read(a['job']/'outcome.json', {'status': 'failed', 'error': f'作业异常退出（{code}）',
                                                    'needs_device_check': a['record']['action'] in HARDWARE})
            if a['cancel_at'] and outcome['status'] == 'completed':
                outcome.update(status='cancelled', error='完成边界收到取消，请检查已产生的文件')
            if a['record']['action'] in MOTION and not read(a['job']/'stop.json', {}).get('confirmed'):
                outcome['needs_device_check'] = True
            if outcome.get('needs_device_check'):
                self.set_recovery(True)
            if a['record']['action'] == 'inspect' and outcome['status'] == 'completed':
                if read(a['job']/'stop.json', {}).get('confirmed'):
                    self.set_recovery(False)
            a['record'].update(**outcome, ended=time.time())
            write(a['job']/'record.json', a['record'])
            self.audit('job_finished', **a['record'])
            if a['device']:
                a['device'].close()
            if a.get('workflow_lock'):
                a['workflow_lock'].close()
            self.active = None

    def _monitor(self):
        while not self.closing or self.active:
            try:
                self._tick()
            except Exception as exc:
                print('作业监控错误:', exc, flush=True)
            time.sleep(.25)

    def visibility(self):
        """Read the saved teaching window; never invent a full-frame fallback."""
        try:
            dataset = self.selected('teaching_dataset')
            path = dataset/'board_window.json'
            window = read(path)
            manifest = read(dataset/'manifest.json')
            camera = manifest['camera']
            width, height = camera['width'], camera['height']
            box = window['window_xyxy']
            if (type(width) is not int or type(height) is not int or
                not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box) or
                not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
                raise ValueError('保存的可见范围坐标无效')
            profile_keys = ('serial', 'width', 'height', 'crop_xywh', 'source_resolution',
                            'camera_matrix', 'dist_coeffs')
            def matches(other):
                return isinstance(other, dict) and all(camera.get(k) == other.get(k) for k in profile_keys)
            if not matches(window.get('camera', {})):
                raise ValueError('可见范围与示教相机配置不一致')
            image = dataset/'sample_0000.png'
            label = '示教采样图（非实时画面）'
            if not image.is_file():
                image = None
                try:
                    latest = self.selected('dataset')
                    if matches(read(latest/'manifest.json')['camera']) and (latest/'sample_0000.png').is_file():
                        image = latest/'sample_0000.png'
                        label = '本次采样图（非实时画面）；范围来自已保存的示教数据'
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            return {'available': True, 'window_xyxy': box, 'width': width, 'height': height,
                    'source_dataset': str(dataset), 'source_name': dataset.name, 'camera': camera,
                    'image_path': str(image) if image else None,
                    'image_label': label if image else '没有匹配的采样图，按原图尺寸显示范围示意',
                    'version': digest(path) + (str(image.stat().st_mtime_ns) if image else '')}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {'available': False, 'reason': '尚无可展示的已保存范围：'+str(exc)}

    def snapshot(self, owner=None):
        with self.mutex:
            state = read(self.state_path, {})
            result_path = state['result'] if 'result' in state else self.config.get('result')
            try:
                result = result_summary(resolve(result_path), evaluate(read(resolve(result_path)), self.config)) if result_path else None
                pipeline = read(resolve(self.config['pipeline_config']))
                installed = resolve(pipeline['calibration'])
                applied = bool(result and installed.exists() and digest(installed) == result['sha256'])
                table_ready = applied and not pipeline['green_cup'].get('installation_requires_calibration', True)
            except (OSError, KeyError, ValueError):
                result, applied, table_ready = None, False, False
            records = sorted((self.data/'jobs').glob('*/record.json'), reverse=True)[:30]
            jobs = [read(p) for p in records]
            inspection = next((p.parent for p in records if read(p).get('action') == 'inspect'), None)
            inspected = bool(inspection and read(inspection/'record.json').get('status') == 'completed'
                             and read(inspection/'stop.json', {}).get('confirmed')
                             and read(inspection/'frame.json')
                             and len(read(inspection/'progress.json', {}).get('joints_deg', [])) == 7)
            current = self.active['job'] if self.active else (records[0].parent if records else None)
            detail = None
            if current:
                detail = {'id': current.name, **read(current/'record.json'),
                          **{key: read(current/(key+'.json')) for key in
                             ('progress', 'stage', 'frame', 'detection', 'stop', 'camera_error', 'home_return')}}
                detail['frame_age_s'] = (max(0., time.time()-detail['frame']['time'])
                                         if detail['frame'] else None)
                log = current/'worker.log'
                if log.exists():
                    with log.open('rb') as stream:
                        stream.seek(max(0, log.stat().st_size-12000))
                        detail['log'] = stream.read().decode(errors='replace')
            readiness = {}
            inspection_connected = bool(self.active and detail and detail['action'] == 'inspect'
                and detail['status'] == 'running' and not detail['camera_error']
                and detail['frame_age_s'] is not None and detail['frame_age_s'] <= 3
                and detail['progress'] and time.time()-detail['progress'].get('time', 0) <= 3
                and len(detail['progress'].get('joints_deg', [])) == 7)
            for action in ACTIONS:
                try:
                    source = self.source(action)
                    if action == 'plan':
                        dataset = self.selected('teaching_dataset')
                        if not (dataset/'board_window.json').exists():
                            raise ValueError('请先绘制并保存整块标定板的可见范围')
                    if action == 'solve':
                        self.selected('dataset')
                    readiness[action] = {'ready': True, 'source': source}
                except (ValueError, KeyError, OSError, TypeError) as exc:
                    readiness[action] = {'ready': False, 'reason': str(exc)}
            visibility = self.visibility()
            frame = (detail or {}).get('frame') or {}
            saved_camera, live_camera = visibility.get('camera'), frame.get('camera')
            keys = ('serial', 'width', 'height', 'crop_xywh', 'source_resolution',
                    'camera_matrix', 'dist_coeffs')
            visibility['live_compatible'] = bool(saved_camera and live_camera and
                all(saved_camera.get(k) == live_camera.get(k) for k in keys))
            return {'readonly': self.readonly, 'root': str(ROOT), 'service_pid': os.getpid(),
                    'pipeline_config': str(resolve(self.config['pipeline_config'])),
                    'busy': occupied(exclude=[self.active['process'].pid] if self.active else []),
                    'active': self.active is not None, 'owns_job': bool(self.active and self.active['owner'] == owner),
                    'recovery_required': self.recovery_required, 'job': detail, 'jobs': jobs,
                    'inspection_ready': inspected and not self.recovery_required,
                    'inspection_connected': inspection_connected,
                    'visibility': visibility, 'state': state, 'result': result, 'applied': applied, 'table_ready': table_ready,
                    'actions': readiness, 'heartbeat_timeout_s': self.timeout,
                    'pin_configured': (self.data/'engineer.json').exists()}

    def image_path(self, kind, job_id=None):
        if kind == 'visibility':
            path = self.visibility().get('image_path')
            if not path:
                raise ValueError('没有匹配的采样图')
            return Path(path)
        if kind == 'teaching':
            return self.selected('teaching_dataset')/'sample_0000.png'
        if kind not in ('preview', 'sample') or not re.fullmatch(r'[0-9]{8}_[0-9]{6}_[a-f0-9]{8}', job_id or ''):
            raise ValueError('无效图像请求')
        return self.data/'jobs'/job_id/(kind+'.jpg')

    def close(self):
        self.closing = True
        self.cancel(reason='服务关闭')
        self.monitor.join(timeout=15)
        # On a stuck native call, the inherited lock remains held by the worker.
        self.instance_lock.close()
