"""Exclusive CLI access for one session directory."""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path


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
