"""Reusable capture directories; exclusive CLI access and explicit invalidation."""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import shutil


CAPTURE_PENDING = '.capture_pending'
CAPTURE_FILES = ('session.json', 'before.json', 'before.log', 'after.json', 'after.log',
                 'target.png', 'target_tcp.json', 'cup_candidates.json', 'cup_candidates.png',
                 'yolo_seg.json', 'yolo_seg.png')


@contextmanager
def session_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.debug.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('此 RUN 正在采集、规划或执行；等待该命令结束后再重试') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def reset_capture(directory, replay=None):
    """Discard only generated capture outputs; retain runs and user files."""
    directory = Path(directory)
    rgbd = directory / 'rgbd'
    if replay is not None and (Path(replay) / 'rgbd').resolve() == rgbd.resolve():
        raise ValueError('回放源不能与待覆盖 RUN 的 rgbd 相同')
    directory.mkdir(parents=True, exist_ok=True)
    # This marker invalidates old plans even if validation/camera work fails.
    (directory / CAPTURE_PENDING).write_text('重新采集尚未成功；请重试 capture\n')
    for name in CAPTURE_FILES:
        (directory / name).unlink(missing_ok=True)
    if rgbd.is_symlink():
        rgbd.unlink()
    elif rgbd.exists():
        shutil.rmtree(rgbd)


def prepare_plan(output, kind):
    """A failed replan must never leave an older executable plan at this path."""
    output = Path(output)
    if output.exists():
        previous = json.loads(output.read_text())
        if not isinstance(previous, dict) or previous.get('kind') != kind:
            raise ValueError(f'输出文件不是同类型计划，不能覆盖：{output}')
        output.unlink()


def dispatch(args, handler):
    """Hold the RUN lock through confirmation and execution, not just writes."""
    directory = getattr(args, 'session', None)
    if getattr(args, 'command', None) in ('move', 'grasp', 'shake'):
        directory = Path(json.loads(args.plan.read_text())['session_path'])
    elif getattr(args, 'command', None) == 'home':
        directory = args.output
    if directory is None:
        return handler(args)
    with session_lock(directory):
        return handler(args)
