"""Silver-cup grasp/shake workflow with step, auto and quiet fast execution."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

from cup_grasp_demo.flow import debug as common, grasp_cli
from cup_grasp_demo.flow.core import cached_file_digests, cached_screen_geometry, digest, load_config, read_json
from cup_grasp_demo.flow.pipeline_home import execute_home
from cup_grasp_demo.flow.phase_timing import emit_line, timing_line


PHASES = ('HOME', 'CAPTURE', 'PLAN', 'APPROACH', 'GRIP')
LABELS = {'HOME': '先张手，再归位', 'CAPTURE': '定位杯子并读取最新配置',
          'PLAN': '计算到闭手准备点的轨迹', 'APPROACH': '复核杯位并运动到 READY',
          'GRIP': '拇指根先闭合，再闭合其余五路，停留原位',
          'SHAKE_PLAN': '只读计算摇晃轨迹和限值，不执行摇晃',
          'SHAKE': '按配置往返摇晃，回到中心后保持闭手'}
STATE_FILE = 'pipeline_state.json'


def save(path, value):
    """Commit an in-flight marker before hardware work; retain it across crashes."""
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Backend:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config(args.config)
        if getattr(args, 'until', None) in ('shake-plan', 'shake'):
            from cup_grasp_demo.flow.parameters import effective_shake_options
            effective_shake_options(self.cfg)
        if self.cfg.get('side_grasp', {}).get('strategy') != 'direct_close':
            raise ValueError('pipeline 当前使用 direct_close 银杯侧抓配置')
        if not self.cfg.get('home_open_hand'):
            raise ValueError('pipeline 要求 home_open_hand=true，先张手再归位')
        self.show = args.show and args.mode == 'step'
        self.prepared = {}
        self.scene = None
        self.latest_snapshot = None
        self.shake_prepared = None
        self.capture_proof = None

    def fingerprint(self):
        code = grasp_cli.implementation_hashes()
        for name in ('pipeline_runner.py', 'pipeline_home.py', 'home_execution.py', 'phase_timing.py'):
            code[name] = digest(common.HERE / name)
        if getattr(self.args, 'until', None) in ('shake-plan', 'shake'):
            for name in ('shake.py', 'shake_cli.py', 'shake_readback.py',
                         'shake_execution.py', 'shake_tracking.py', 'shake_camera.py',
                         'parameters.py', 'planar_shake.py', 'planar_shake_cli.py',
                         'planar_shake_execution.py', 'shake_study.py'):
                code[name] = digest(common.HERE / name)
        return dict(config=digest(self.args.config), code=code, sources=common.source_hashes(self.cfg))

    def artifacts(self):
        names = ('session.json', 'ready_plan.json', 'grip_plan.json')
        if getattr(self.args, 'until', None) in ('shake-plan', 'shake'):
            names += ('shake_plan.json',)
        return {name: digest(self.args.session / name)
                for name in names
                if (self.args.session / name).exists()}

    def plan(self, start, until, name):
        path = self.args.session / name
        fast = getattr(self.args, 'mode', 'step') == 'fast'
        prepared = {} if fast else None
        snapshot = None
        if fast and start == 'home':
            snapshot = read_json(self.args.session / 'after.json')
        elif fast:
            snapshot = getattr(self, 'latest_snapshot', None)
        if snapshot is not None:
            if not 0 <= time.time() - snapshot['observed_epoch_s'] <= 60:
                snapshot = None
        result = grasp_cli.plan(SimpleNamespace(session=self.args.session, start=start, until=until,
                                               output=path, joints_json=None),
                                prepared=prepared, cached_scene=getattr(self, 'scene', None) if fast else None,
                                snapshot=snapshot)
        if result:
            raise RuntimeError(f'规划被阻塞：{path}')
        if fast:
            self.prepared[str(path)] = prepared
            self.scene = prepared['scene']
        return path

    def move(self, path):
        # Only invoked after this run's explicit operator authorization.
        runs = self.args.session / 'runs'
        existing = set(runs.iterdir()) if runs.exists() else set()
        grasp_cli.execute(SimpleNamespace(plan=path, execute=True, show=self.show,
                                          fast=getattr(self.args, 'mode', 'step') == 'fast'),
                          confirm=lambda _: 'GRASP',
                          prepared=getattr(self, 'prepared', {}).get(str(path)),
                          capture_proof=getattr(self, 'capture_proof', None))
        created = [p / 'receipt.json' for p in runs.iterdir() if p not in existing
                   and (p / 'receipt.json').exists()]
        if len(created) != 1 or not read_json(created[0]).get('success'):
            raise RuntimeError('没有唯一成功动作收据，不能继续下一阶段')
        receipt = read_json(created[0])
        expected = 'READY' if read_json(path)['until_state'] == 'ready' else 'GRIP_COMMANDS_SENT'
        if receipt.get('completed_state') != expected:
            raise RuntimeError('动作收据没有确认所选终点，不能继续下一阶段')
        self.latest_snapshot = read_json(receipt['actual_json']).get('final_snapshot')
        return dict(receipt_path=str(created[0]), **receipt)

    def shake(self):
        from cup_grasp_demo.flow import shake_cli
        # Use the same live validation and stop handling as the standalone command.
        # A pipeline authorizes this stage at launch (or at the STEP prompt).
        path = self.args.session / 'shake_plan.json'
        result = shake_cli.execute(SimpleNamespace(plan=path, execute=True, show=self.show,
                                                   fast=getattr(self.args, 'mode', 'step') == 'fast'),
                                   confirm=lambda _: 'SHAKE', return_receipt=True,
                                   prepared=getattr(self, 'shake_prepared', None))
        if (not isinstance(result, dict) or result.get('success') is not True
                or result.get('returned_center') is not True
                or result.get('measured_wave', {}).get('tracking_verified') is not True):
            detail = (f"{result.get('error', '到位/轨迹反馈不足')}；记录：{result.get('actual_path')}"
                      if isinstance(result, dict) else '执行器没有返回动作收据')
            raise RuntimeError('摇晃未确认完成并回到中心，pipeline 停止：' + detail)
        return dict(actual_path=result['actual_path'], completed_state='SHAKE_COMPLETED',
                    success=True, returned_center=True, measured_wave=result['measured_wave'],
                    motion_elapsed_s=result.get('motion_elapsed_s'),
                    duration_completed=result.get('duration_completed'),
                    cup_retention_verified=result.get('cup_retention_verified', False),
                    dice_change_verified=result.get('dice_change_verified', False))

    def perform(self, phase):
        directory = self.args.session
        if phase == 'HOME':
            result = execute_home(directory, self.cfg, self.show,
                                  **({'fast': True} if self.args.mode == 'fast' else {}))
            if getattr(self.args, 'mode', 'step') == 'fast':
                self.latest_snapshot = read_json(result['actual_path']).get('final_snapshot')
            return result
        if phase == 'CAPTURE':
            fast = getattr(self.args, 'mode', 'step') == 'fast'
            prepared = {} if fast else None
            common.capture(SimpleNamespace(config=self.args.config, session=directory,
                                           replay=None, show=self.show), prepared=prepared,
                           snapshot=getattr(self, 'latest_snapshot', None) if fast else None)
            if prepared:
                self.scene = prepared['scene']
                self.capture_proof = prepared.get('capture_proof')
            session, _ = common.verify_session(directory)
            return dict(capture_id=session['capture_id'], session_path=str(directory / 'session.json'))
        if phase == 'PLAN':
            return dict(plan_path=str(self.plan('home', 'ready', 'ready_plan.json')))
        if phase == 'APPROACH':
            return self.move(directory / 'ready_plan.json')
        if phase == 'GRIP':
            return self.move(self.plan('ready', 'grip', 'grip_plan.json'))
        if phase == 'SHAKE_PLAN':
            from cup_grasp_demo.flow import shake_cli
            path = directory / 'shake_plan.json'
            self.shake_prepared = {} if getattr(self.args, 'mode', 'step') == 'fast' else None
            code = shake_cli.plan(SimpleNamespace(config=self.args.config, session=directory,
                                                   output=path, feedback_json=None), prepared=self.shake_prepared)
            if code:
                raise RuntimeError(f'摇晃规划超限或不可达；未执行摇晃，查看 {path}')
            return dict(plan_path=str(path), execution_enabled=False,
                        completed_state='SHAKE_PLANNED')
        if phase == 'SHAKE':
            return self.shake()
        raise ValueError(f'未知阶段：{phase}')


@contextmanager
def log_output(path):
    """Capture Python prints AND inherited child-process output; always restore FDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', buffering=1) as stream:
        sys.stdout.flush()
        sys.stderr.flush()
        saved = (os.dup(1), os.dup(2))
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            with redirect_stdout(stream), redirect_stderr(stream):
                yield
        finally:
            try:
                stream.flush()
            finally:
                for fd, original in zip((1, 2), saved):
                    os.dup2(original, fd)
                    os.close(original)


def run(args, *, backend_factory=Backend, ask=None):
    if getattr(args, 'until', None) is None:
        args.until = 'shake' if args.mode == 'fast' else 'grip'
    if args.mode != 'fast' or not args.execute or args.status:
        return _run(args, backend_factory=backend_factory, ask=ask)
    log = args.session / 'runs' / ('pipeline_fast_' + uuid.uuid4().hex) / 'pipeline.log'
    print(f'FAST 开始：运行到 {args.until}；日志：{log}', flush=True)
    try:
        # Duplicate the terminal before log_output redirects native FDs too.
        terminal = sys.stdout
        with os.fdopen(os.dup(1), 'w', buffering=1) as console, log_output(log), cached_screen_geometry(), cached_file_digests():
            def timing_output(message):
                emit_line(message)  # Keep timing in the full log as well.
                target = console if terminal is sys.__stdout__ else terminal
                print(message, file=target, flush=True)
            try:
                result = _run(args, backend_factory=backend_factory, ask=ask, log_path=log,
                              timing_output=timing_output)
            except BaseException:
                traceback.print_exc()
                raise
    except BaseException:
        print(f'FAST 停止；详细日志：{log}', file=sys.stderr, flush=True)
        raise
    state = read_json(args.session / STATE_FILE)
    print(f'FAST 完成：{state["completed_state"]}；状态：{args.session / STATE_FILE}', flush=True)
    return result


def _run(args, *, backend_factory=Backend, ask=None, log_path=None, timing_output=emit_line):
    ask = input if ask is None else ask
    path = args.session / STATE_FILE
    if args.status:
        print(json.dumps(read_json(path) if path.exists() else {'status': 'NOT_STARTED'},
                         indent=2, ensure_ascii=False))
        return 0
    phases = list(PHASES[:4] if args.until == 'ready' else PHASES)
    if args.until in ('shake-plan', 'shake'):
        phases.append('SHAKE_PLAN')
    if args.until == 'shake':
        phases.append('SHAKE')
    if args.mode == 'fast' and args.until == 'shake':
        phases.remove('SHAKE_PLAN')
        phases.insert(phases.index('GRIP'), 'SHAKE_PLAN')
    print('PIPELINE：' + ' → '.join(phases))
    if args.until == 'shake-plan':
        print('闭手后只计算摇晃轨迹；不发送摇晃指令，不自动抬杯或揭杯。')
    elif args.until == 'shake':
        print('完成闭手和摇晃后回到中心并保持闭手；不自动抬杯、揭杯或归位。')
    else:
        print('终点为闭手指令完成，不代表已抓牢；不抬杯、不摇晃、不揭杯。' if args.until == 'grip'
              else '终点为 READY，保持张手，不闭手。')
    if not args.execute:
        print('流程预览，不访问相机或机械臂；加 --execute 才运行。')
        return 0
    backend = backend_factory(args)
    fingerprint = backend.fingerprint()
    if args.resume:
        state = read_json(path)
        if (state.get('kind') != 'silver_cup_debug_pipeline' or state.get('status') != 'PAUSED'
                or state.get('phases') != phases):
            raise ValueError('只允许续跑正常 PAUSED 且终点相同的流程；失败/中断后检查现场并从头开始')
        if state['fingerprint'] != fingerprint or state['artifacts'] != backend.artifacts():
            raise ValueError('配置、程序或 RUN 已改变，不能续跑；请重新从 HOME 开始 pipeline')
    else:
        if path.exists() and read_json(path).get('status') == 'RUNNING':
            raise ValueError('上轮在动作中中断；检查实物后将该状态文件改名存档，再从头开始')
        state = dict(kind='silver_cup_debug_pipeline', run_id=uuid.uuid4().hex, status='PAUSED',
                     next_index=0, phases=phases, events=[], fingerprint=fingerprint,
                     artifacts=backend.artifacts(), physical_grip_verified=False)
    print('开始条件：张手空间与归位通路无障碍；相机/基座/桌面未移动。')
    print('允许从桌面上的抓杯状态开始：HOME 先原位张手再归位，归位后识别杯子。')
    print('HOME 检查桌面与关节路径，杯体避让由操作者确认；运行期间不操作 WEB 或其他控制程序。')
    history = args.session / 'runs' / ('pipeline_' + state['run_id'])
    history.mkdir(parents=True, exist_ok=True)
    if log_path is not None:
        state['log_path'] = str(log_path)
    state['mode'] = args.mode

    def persist():
        state['updated_epoch_s'] = time.time()
        save(path, state)
        save(history / STATE_FILE, state)

    persist()
    invocation_timings = []
    active_timing = None
    try:
        for index in range(state['next_index'], len(phases)):
            phase = phases[index]
            print(f'PIPELINE [{index + 1}/{len(phases)}] {phase}：{LABELS[phase]}', flush=True)
            if args.mode == 'step':
                answer = ask('Enter 执行本阶段；q 暂停退出：').strip().lower()
                while answer not in ('', 'q'):
                    answer = ask('请输入 Enter 或 q：').strip().lower()
                if answer == 'q':
                    state['status'] = 'PAUSED'
                    persist()
                    print(f'已暂停，下一阶段 {phase}；加 --resume 继续。')
                    return 0
            phase_started = time.perf_counter()
            active_timing = (phase, phase_started)
            if state['fingerprint'] != backend.fingerprint() or state['artifacts'] != backend.artifacts():
                raise ValueError('配置、程序或 RUN 已改变；重新从 HOME 开始以读取新参数')
            state.update(status='RUNNING', active_phase=phase)
            state['events'].append(dict(phase=phase, event='started', epoch_s=time.time()))
            persist()
            work_started = time.perf_counter()
            result = backend.perform(phase)
            elapsed = time.perf_counter() - phase_started
            state.setdefault('phase_timings_s', {})[phase] = elapsed
            state['events'].append(dict(phase=phase, event='completed', epoch_s=time.time(),
                                        duration_s=elapsed, precheck_s=work_started-phase_started,
                                        result=result))
            state.update(status='PAUSED', next_index=index + 1, artifacts=backend.artifacts())
            persist()
            invocation_timings.append(elapsed)
            active_timing = None
            timing_output(timing_line(phase, elapsed))
        state.update(status='COMPLETED', active_phase=None,
                     completed_state={'ready': 'READY', 'grip': 'GRIP_COMMANDS_SENT',
                                      'shake-plan': 'SHAKE_PLANNED',
                                      'shake': 'SHAKE_COMPLETED'}[args.until])
        persist()
        print(f'PIPELINE 完成：{state["completed_state"]}；状态：{path}')
        return 0
    except BaseException as exc:
        if active_timing is not None:
            phase, started = active_timing
            elapsed = time.perf_counter() - started
            invocation_timings.append(elapsed)
            state.setdefault('phase_timings_s', {})[phase] = elapsed
            state['events'].append(dict(phase=phase, event='failed', epoch_s=time.time(),
                                        duration_s=elapsed))
            timing_output(timing_line(phase, elapsed, 'failed'))
        state.update(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, EOFError)) else 'FAILED',
                     error=f'{type(exc).__name__}: {exc}')
        persist()
        raise
    finally:
        if invocation_timings:
            timing_output(f'[耗时汇总] 本次 PIPELINE：{sum(invocation_timings):.3f} s'
                          '（阶段合计，不含按键等待）')
