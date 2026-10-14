"""Translate terminal commands into the existing control protocol."""

import json


HELP = (
    "next / n：执行下一阶段",
    "run / g：连续执行到 RETURN_HOME，完成后可再次 run",
    "until <阶段>：顺序执行到指定阶段，例如 until grip",
    "home：空闲时张手并归位",
    "action <动作名>：空闲时执行手势，例如 action rock",
    "actions：列出可用动作；status：查看状态；query_pose：查询姿态",
    "reload：重载参数；refresh_perception：重新定位",
    "quit / q / close：释放设备并退出（执行中会排队，并非急停）",
    "help / ?：显示帮助；空行忽略；仍支持逐行 JSON",
)


def parse_command(line):
    """Keep JSON requests intact; shorthand uses the same guarded dispatcher.

    No generated IDs: the protocol already supports requests without an ID,
    so mixing terminal commands and application IDs cannot cause collisions.
    """
    text = line.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if text[0] in '{["':
            raise
    words = text.split()
    command = words[0].lower()
    if len(words) == 1:
        if command in ('next', 'n', 'advance'):
            return dict(command='advance')
        if command in ('run', 'g'):
            return dict(command='advance', until='RETURN_HOME')
        if command == 'home':
            return dict(command='action', name='home')
        if command in ('quit', 'q', 'close'):
            return dict(command='close')
        if command in ('help', '?'):
            return dict(command='help')
        if command in ('status', 'actions', 'reload', 'query_pose', 'refresh_perception'):
            return dict(command=command)
    if len(words) == 2:
        if command == 'until':
            return dict(command='advance', until=words[1].upper())
        if command == 'action':
            return dict(command='action', name=words[1])
    raise ValueError('未知命令或参数数量错误；输入 help 查看帮助')
