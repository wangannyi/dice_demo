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
INTERACTIVE_ACTIONS = (
    ('win', '骰子：机械臂赢（V 手势）'),
    ('lose', '骰子：机械臂输（点赞）'),
    ('draw', '骰子：平局'),
    ('rock', '猜拳：石头'),
    ('paper', '猜拳：布'),
    ('scissors', '猜拳：剪刀'),
    ('home', '归位：HOME（六路手指张开）'),
    ('rps-ready', '猜拳：预备位置（六路手指张开）'),
)


def choose_action(registry, input_fn=input):
    """Choose one public game action; q/EOF leaves without hardware access."""
    registered = set(registry.names())
    actions = [(name, label) for name, label in INTERACTIVE_ACTIONS if name in registered]
    if not actions:
        raise ValueError('没有可供交互选择的游戏动作')
    print('\n请选择动作：')
    for index, (name, label) in enumerate(actions, 1):
        print(f'  {index}. {label}  [{name}]')
    print('  q. 退出')
    while True:
        try:
            value = input_fn('输入编号或动作名：').strip()
        except EOFError:
            return None
        if value.lower() in ('q', 'quit', 'exit'):
            return None
        if value.isdigit() and 1 <= int(value) <= len(actions):
            return actions[int(value)-1][0]
        if value in {name for name, _ in actions}:
            return value
        print('无效选择，请重新输入。')


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


def interactive_execute(registry, cfg, scene, session, client, planner, new_run_fn):
    """Reuse one initialized SDK connection until the operator explicitly exits."""
    completed = 0
    print('动作控制已初始化；完成动作后会返回菜单，输入 q 关闭连接并退出。', flush=True)
    while True:
        action = choose_action(registry)
        if action is None:
            print('常驻动作会话已退出。', flush=True)
            return completed
        recipe = registry.recipe(action)
        directory = new_run_fn(session, recipe['gesture'])
        print(f'执行动作：{action} [{recipe["gesture"]}]', flush=True)
        execute_recipe(recipe, cfg, scene, directory, client, planner)
        completed += 1
        print(f'动作完成，保持当前姿态；记录：{directory / "receipt.json"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', help='配置中的动作名或别名')
    parser.add_argument('--list', action='store_true', help='list configured actions without hardware')
    parser.add_argument('--gestures', type=Path, default=ROOT / 'configs/actions/gestures',
                        help='手势分组目录，或单个组文件路径（调试用）')
    parser.add_argument('--config', type=Path, default=DEFAULT_SYSTEM)
    parser.add_argument('--session', type=Path, default=ROOT / 'cup_grasp_demo/datasets/result_feedback')
    parser.add_argument('--execute', action='store_true', help='execute immediately without another prompt')
    args = parser.parse_args(argv)
    from scripts.action_registry import load_registry
    registry = load_registry(args.gestures)
    for message in registry.errors:
        print(f'[gestures] {message}', file=sys.stderr, flush=True)
    if args.list:
        for name in registry.names():
            print(name)
        return 0
    interactive = args.action is None
    if interactive and args.execute:
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
            controller_directory = new_run(args.session, 'feedback_control')
            client = SDKClient(cfg, controller_directory)
            try:
                interactive_execute(registry, cfg, table['scene'], args.session,
                                    client, arm_plan, new_run)
            finally:
                client.close()
        return 0
    action = args.action if args.action is not None else choose_action(registry)
    if action is None:
        print('已退出，未发送指令。')
        return 0
    recipe = registry.recipe(action)
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
