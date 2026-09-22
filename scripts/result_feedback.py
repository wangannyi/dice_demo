#!/usr/bin/env python3
"""Standalone dice feedback gestures; preview by default, no camera acquisition."""
import argparse
from copy import deepcopy
import json
import math
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_SYSTEM = ROOT / 'configs/green_cup.json'


def recipe_for(config, action):
    name = config.get('aliases', {'win': 'yeah', 'lose': 'thumbs-up'}).get(action, action)
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('动作名称仅允许字母、数字、下划线和连字符')
    if name not in config['gestures']:
        raise ValueError('未知动作：' + name + '；可选：' + ', '.join(config['gestures']))
    recipe = config['gestures'][name]
    speed = recipe.get('speed_percent', config['speed_percent'])
    duration = recipe.get('finger_duration_s', config['finger_duration_s'])
    hand_mode = recipe.get('finger_speed_mode', config.get('finger_speed_mode', 'timed'))
    max_wait = recipe.get('finger_max_wait_s', config.get('finger_max_wait_s', .65))
    if hand_mode not in ('timed', 'max'):
        raise ValueError('finger_speed_mode must be timed or max')
    if type(max_wait) not in (int, float) or not math.isfinite(max_wait) or not .65 <= max_wait <= 5:
        raise ValueError('finger_max_wait_s must be 0.65..5')
    execution = dict(config.get('execution', {}))
    execution.update(recipe.get('execution', {}))
    mode, delay = execution.get('mode', 'arm_then_hand'), execution.get('delay_s', 0.0)
    if mode not in ('arm_then_hand', 'hand_then_arm', 'together'):
        raise ValueError('execution.mode must be arm_then_hand, hand_then_arm or together')
    if type(delay) not in (int, float) or not math.isfinite(delay) or not 0 <= delay <= 30:
        raise ValueError('execution.delay_s must be 0..30')
    if type(speed) is not int or not 1 <= speed <= 100:
        raise ValueError('speed_percent must be 1..100')
    if type(duration) not in (int, float) or not math.isfinite(duration) or not .5 <= duration <= 2.55:
        raise ValueError('finger_duration_s must be 0.5..2.55')
    recipe = config['gestures'][name]
    joints, hand = recipe['joints_deg'], recipe['hand_0_100']
    if len(joints) != 7 or any(type(x) not in (int, float) or not math.isfinite(x) for x in joints):
        raise ValueError('joints_deg requires seven finite angles in degrees')
    if len(hand) != 6 or any(type(x) is not int or not 0 <= x <= 100 for x in hand):
        raise ValueError('hand_0_100 requires six integers in 0..100')
    from cup_grasp_demo.flow.feedback_sequence import sequence_values
    sequence = deepcopy(recipe.get('hand_sequence'))
    sequence_values(sequence, hand, max_wait if hand_mode == 'max' else duration)
    return dict(hand_sequence=sequence, gesture=name, joints_deg=list(joints), hand_0_100=list(hand),
                speed_percent=speed, finger_duration_s=duration,
                execution=dict(mode=mode, delay_s=delay),
                finger_speed_mode=hand_mode, finger_max_wait_s=max_wait)


def execute_recipe(recipe, cfg, scene, directory, client, planner):
    """One SDK connection; arm failure prevents the hand command."""
    from cup_grasp_demo.flow.core import write_json
    cfg = deepcopy(cfg)
    cfg['speed_percent'] = recipe['speed_percent']
    cfg['green_cup'].update(finger_duration_s=(recipe['finger_max_wait_s'] if recipe['finger_speed_mode'] == 'max' else recipe['finger_duration_s']),
                            finger_settle_s=0.0, read_hand_feedback=False,
                            require_hand_position=False,
                            grip_targets_0_100=list(recipe['hand_0_100']),
                            feedback_hand_max_speed=recipe['finger_speed_mode'] == 'max')
    def snapshot(name):
        report = client.call('snapshot', directory / (name + '.json'))
        if (report.get('arm_status') != 0 or report.get('motion_status', 0) != 0
                or report.get('joints_enabled') != [True] * 7
                or report.get('ctrl_mode') not in (1, 3)):
            raise RuntimeError('机械臂必须停稳、全部使能且无错误')
        return report['joints_rad']

    def send(plan, name):
        if plan['blockers']:
            raise ValueError(str(plan['blockers']))
        request = directory / (name + '_request.json')
        write_json(request, dict(execution_authorized=True,
                                authorized_epoch_s=time.time(), plan=plan, config=cfg))
        report = client.call('run', directory / (name + '_actual.json'), request)
        if not report.get('success'):
            raise RuntimeError(f'{name} failed: {report.get("error")}')
        return report

    start = snapshot('before')
    plan = planner(start, [[math.radians(x) for x in recipe['joints_deg']]], scene, cfg)
    if plan['blockers']:
        raise ValueError(str(plan['blockers']))
    mode = recipe['execution']['mode']
    delay = recipe['execution']['delay_s']
    def send_hand():
        from cup_grasp_demo.flow.feedback_sequence import sequence_values
        targets, interval = sequence_values(recipe.get('hand_sequence'), recipe['hand_0_100'], cfg['green_cup']['finger_duration_s'])
        for index, target in enumerate(targets):
            cfg['green_cup']['grip_targets_0_100'] = list(target)
            label = 'hand' if len(targets) == 1 else f'hand_{index:02d}'
            began = time.monotonic()
            send(dict(kind='green_hand_command', start_q_rad=snapshot('before_' + label),
                      target_0_100=target, blockers=[], stages=[]), label)
            if index < len(targets)-1:
                remaining = interval - (time.monotonic()-began)
                if remaining > 0:
                    time.sleep(remaining)
    if mode == 'together':
        plan.update(kind='feedback_together', target_0_100=recipe['hand_0_100'],
                    hand_delay_s=delay, hand_sequence=recipe.get('hand_sequence'))
        send(plan, 'together')
    elif mode == 'hand_then_arm':
        send_hand()
        if delay:
            time.sleep(delay)
        send(plan, 'arm')
    else:
        send(plan, 'arm')
        if delay:
            time.sleep(delay)
        send_hand()
    write_json(directory / 'receipt.json', dict(success=True, recipe=recipe,
               note='机械臂动作及手指指令/观察时间完成；手指姿态未实测确认；保持当前姿态'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', help='配置中的动作名或别名')
    parser.add_argument('--list', action='store_true', help='list configured actions without hardware')
    parser.add_argument('--gestures', type=Path, default=ROOT / 'configs/actions/result_feedback.json')
    parser.add_argument('--config', type=Path, default=DEFAULT_SYSTEM)
    parser.add_argument('--session', type=Path, default=ROOT / 'cup_grasp_demo/datasets/result_feedback')
    parser.add_argument('--execute', action='store_true', help='execute immediately without another prompt')
    args = parser.parse_args(argv)
    config = json.loads(args.gestures.read_text())
    if args.list:
        for name in config['gestures']:
            print(name)
        return 0
    if args.action is None:
        parser.error('请指定动作名，或使用 --list')
    recipe = recipe_for(config, args.action)
    print(json.dumps(recipe, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print('仅预览，未连接 CAN、相机或发送指令。加 --execute 执行。')
        return 0
    from cup_grasp_demo.flow.core import load_config, read_json, digest
    from cup_grasp_demo.flow.debug import new_run
    from cup_grasp_demo.flow.green_runtime import SDKClient
    from cup_grasp_demo.flow.green_cup_planning import arm_plan
    from cup_grasp_demo.flow.session_storage import session_lock
    cfg = load_config(args.config)
    table = read_json(ROOT / cfg['green_cup']['home_table_scene'])
    if table['calibration_sha256'] != digest(cfg['calibration']):
        raise ValueError('桌面记录与当前标定不一致，请更新桌面记录')
    with session_lock(args.session):
        directory = new_run(args.session, recipe['gesture'])
        client = SDKClient(cfg, directory)
        try:
            execute_recipe(recipe, cfg, table['scene'], directory, client, arm_plan)
        finally:
            client.close()
        print(f'动作完成，保持当前姿态；记录：{directory / "receipt.json"}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('反馈动作已中断', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'FEEDBACK STOP: {exc}', file=sys.stderr)
        raise SystemExit(2)
