#!/usr/bin/env python3
"""Config-driven calibration commands; existing collectors remain authoritative."""
import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calibration.apply_result import atomic_bytes


def read(path):
    return json.loads(Path(path).read_text())


def resolve(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def validate(c):
    if c['schema'] != 1:
        raise ValueError('Unsupported workflow schema')
    for key in ('show', 'allow_provisional', 'capture_only_visibility'):
        if type(c[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if c['fps'] not in (6, 15, 30) or c['tcp'] not in ('flange', 'palm'):
        raise ValueError('Invalid fps or tcp')
    for key, value, low, high in (
        ('speed_percent', c['motion']['speed_percent'], 1, 100),
        ('speed_deg_s', c['motion']['speed_deg_s'], .01, 100),
        ('acc_deg_s2', c['motion']['acc_deg_s2'], .01, 100),
        ('reference_frames', c['reference_frames'], 10, 120),
        ('max_step_deg', c['max_step_deg'], .01, 10 if c['capture_only_visibility'] else 2),
        ('margin_px', c['margin_px'], 0, 50),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Invalid {key}')
    for key in ('speed_percent',):
        if type(c['motion'][key]) is not int:
            raise ValueError(f'{key} must be integer')
    if type(c['reference_frames']) is not int:
        raise ValueError('reference_frames must be integer')
    if c['measurement'] is not None:
        for key in ('width_mm', 'height_mm'):
            value = c['measurement'][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid measurement.{key}')
    for key in ('serial', 'channel', 'reference_id', 'hand_board', 'reference_board',
                'home', 'output_root', 'state_file', 'pipeline_config'):
        if not isinstance(c[key], str) or not c[key].strip():
            raise ValueError(f'{key} must be a nonempty string')


def require_idle():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, PermissionError):
            continue
        if any(a.endswith('/debug.py') or a == 'debug.py' for a in args) and 'pipeline' in args:
            raise ValueError(f'机械臂 CONTROL/FAST 仍在运行（PID {proc.name}），请先退出该会话')


class Workflow:
    def __init__(self, config, dry_run=False):
        self.c = read(config)
        validate(self.c)
        self.dry = dry_run
        self.state_path = resolve(self.c['state_file'])
        self.state = read(self.state_path) if self.state_path.exists() else {}

    def selected(self, key):
        value = self.c.get(key) or self.state.get(key)
        if not value:
            raise ValueError(f'缺少 {key}：请在配置中填写，或先完成对应步骤')
        return resolve(value)

    def new(self, label):
        return resolve(self.c['output_root']) / (datetime.now().strftime('%Y%m%d_%H%M%S_') + label + '_' + uuid4().hex[:6])

    def save(self, **values):
        if not self.dry:
            self.state.update({k: str(v) for k, v in values.items()})
            atomic_bytes(self.state_path, (json.dumps(self.state, ensure_ascii=False, indent=2)+'\n').encode())

    def call(self, script, *args, quality_exit=False):
        argv = [sys.executable, str(ROOT/script), *map(str, args)]
        print('运行: ' + ' '.join(argv), flush=True)
        if self.dry:
            return
        result = subprocess.run(argv, cwd=ROOT, check=False)
        if result.returncode and not (quality_exit and result.returncode == 2):
            raise RuntimeError(f'{script} 失败，退出码 {result.returncode}；未继续后续步骤')

    def quality(self, path):
        if not self.dry and not read(path).get('quality_passed'):
            if not self.c['allow_provisional']:
                raise ValueError(f'质量未通过：{path}。结果已保留；如明确接受临时精度，在配置中设置 allow_provisional=true')
            print('注意：使用质量未通过的临时标定，质量标记保持 false', flush=True)

    def solve(self, dataset, teaching=False):
        out = self.new('solve')
        if not self.dry:
            out.parent.mkdir(parents=True, exist_ok=True)
        m = self.c['measurement']
        if m:
            self.call('calibration/tools/reprocess_dimensions.py', '--dataset', dataset,
                      '--output', out, '--width-mm', m['width_mm'], '--height-mm', m['height_mm'], quality_exit=True)
        else:
            if not self.dry:
                out.mkdir()
            self.call('calibration/calibrate.py', 'solve', '--dataset', dataset,
                      '--output', out/'result.json', quality_exit=True)
        if not self.dry and not (out/'result.json').is_file():
            raise ValueError('求解没有产生结果文件')
        self.save(result=out/'result.json')
        if teaching:
            self.save(teaching_calibration=out/'result.json')
        self.quality(out/'result.json')

    def motion(self):
        c = self.c
        return ['--home', resolve(c['home']), '--channel', c['channel'],
                '--speed-percent', c['motion']['speed_percent'],
                '--smooth-speed-deg-s', c['motion']['speed_deg_s'],
                '--smooth-acc-deg-s2', c['motion']['acc_deg_s2']]

    def run(self, action, execute=False):
        c = self.c
        if not self.dry and action in ('first', 'auto', 'register', 'restore', 'apply', 'table'):
            require_idle()
        opt = ['--allow-provisional'] if c['allow_provisional'] else []
        if action in ('auto', 'restore') and not (execute or self.dry):
            raise ValueError('该操作会驱动机械臂，请使用 --execute；仅查看命令用 --dry-run')
        if action == 'status':
            print(json.dumps({'config': c, 'state': self.state}, ensure_ascii=False, indent=2))
        elif action == 'first':
            out = self.new('first')
            if not self.dry:
                out.parent.mkdir(parents=True, exist_ok=True)
            print('请固定末端手眼板并遮挡桌面板；手动示教，Enter 采样，q 结束。', flush=True)
            board = resolve(c['hand_board'])
            if not self.dry:
                board_config = read(board)
                board_config.setdefault('image_profile', {})['fps'] = c['fps']
                board = out.with_suffix('.board.json')
                atomic_bytes(board, (json.dumps(board_config, indent=2)+'\n').encode())
            self.call('calibration/calibrate.py', 'collect', '--serial', c['serial'], '--tcp', c['tcp'],
                      '--channel', c['channel'], '--board', board, '--dataset', out,
                      *(['--preview'] if c['show'] else []))
            self.save(dataset=out, teaching_dataset=out)
            self.solve(out, teaching=True)
        elif action in ('solve', 'window'):
            dataset = self.selected('dataset')
            if action == 'solve':
                self.solve(dataset)
            else:
                self.call('calibration/auto_collect.py', 'draw-window', '--dataset', dataset)
        elif action == 'plan':
            out = self.new('plan').with_suffix('.json')
            if not self.dry:
                out.parent.mkdir(parents=True, exist_ok=True)
            cal = self.selected('teaching_calibration')
            self.quality(cal)
            self.call('calibration/auto_collect.py', 'plan', '--dataset', self.selected('teaching_dataset'),
                      '--calibration', cal, '--output', out, '--max-step-deg', c['max_step_deg'],
                      '--margin-px', c['margin_px'], *opt,
                      *(['--capture-only-visibility'] if c['capture_only_visibility'] else []))
            self.save(plan=out)
        elif action == 'auto':
            plan = self.selected('plan')
            if not self.dry:
                self.quality(resolve(read(plan)['calibration']))
            out = self.new('auto')
            if not self.dry:
                out.parent.mkdir(parents=True, exist_ok=True)
            print('请固定末端手眼板并遮挡桌面板；HOME 后沿示教路线采样。', flush=True)
            self.call('calibration/auto_collect.py', 'run', '--plan', plan, '--output', out,
                      '--fps', c['fps'], *self.motion(), '--start-from', 'home',
                      *(['--show'] if c['show'] else []), '--execute')
            self.save(dataset=out)
            self.solve(out)
        elif action in ('register', 'restore'):
            out = self.new(action)
            if not self.dry:
                out.parent.mkdir(parents=True, exist_ok=True)
            common = ['--board', resolve(c['reference_board']), '--serial', c['serial'],
                      '--reference-id', c['reference_id'], '--frames', c['reference_frames'], '--output', out]
            print('请露出固定桌面板，保持基座和固定板不动。', flush=True)
            if action == 'restore':
                self.call('calibration/reference_board.py', 'restore-auto', '--registration', self.selected('registration'),
                          *common, *self.motion(), *opt, '--execute')
                self.save(result=out/'restored_calibration.json')
                self.quality(out/'restored_calibration.json')
            else:
                result = self.selected('result')
                self.quality(result)
                self.call('calibration/reference_board.py', 'observe', *common)
                self.call('calibration/reference_board.py', 'register', '--calibration', result,
                          '--observation', out/'observation.json', '--output', out/'registration.json', *opt)
                self.save(registration=out/'registration.json')
        elif action in ('apply', 'table'):
            config = resolve(c['pipeline_config'])
            if action == 'apply':
                result = self.selected('result')
                self.quality(result)
                if not self.dry:
                    pipeline = read(config)
                    backup = self.new('backup')
                    backup.mkdir(parents=True)
                    for label, path in [('pipeline_config.json', config),
                                        ('handeye_result.json', resolve(pipeline['calibration'])),
                                        ('home_table_scene.json', resolve(pipeline['green_cup']['home_table_scene']))]:
                        shutil.copy2(path, backup/label)
                    self.save(backup=backup)
                    print(f'备份：{backup}', flush=True)
                self.call('calibration/apply_result.py', '--result', result, '--config', config)
            out = self.new('table')
            self.call('scripts/table_capture.py', '--config', config, '--session', out)
            self.call('scripts/register_home_table.py', '--config', config, '--table-scene', out/'planar_table_scene.json')
            self.save(table=out/'planar_table_scene.json')
        print('完成' + ('（仅预览，未执行）' if self.dry else ''), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('first', 'window', 'solve', 'plan', 'auto', 'register', 'restore', 'apply', 'table', 'status'))
    parser.add_argument('--config', type=Path, default=ROOT/'configs/calibration_workflow.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        w = Workflow(args.config, args.dry_run)
        if args.dry_run or args.action == 'status':
            w.run(args.action, args.execute)
        else:
            w.state_path.parent.mkdir(parents=True, exist_ok=True)
            with w.state_path.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Reload state after taking the workflow lock.
                w = Workflow(args.config)
                w.run(args.action, args.execute)
        return 0
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f'标定停止：{exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('标定已中断；未继续后续步骤', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
