"""Small, hardware-independent helpers shared by the web server and worker."""
import fcntl
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from calibration.apply_result import atomic_bytes

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = {'first', 'plan', 'auto', 'solve', 'register', 'restore', 'apply', 'table', 'inspect', 'home'}
HARDWARE = {'first', 'auto', 'register', 'restore', 'apply', 'table', 'inspect', 'home'}
MOTION = {'auto', 'restore', 'home'}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def write(path, data):
    atomic_bytes(path, (json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode())


def resolve(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin_record(pin):
    if not pin.isascii() or not pin.isdigit() or not 8 <= len(pin) <= 20:
        raise ValueError('PIN 必须为 8–20 位数字')
    salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac('sha256', pin.encode(), bytes.fromhex(salt), 200000).hex()
    return {'salt': salt, 'hash': hashed}


def check_pin(pin, record):
    if not record or not isinstance(pin, str) or len(pin) > 20:
        return False
    hashed = hashlib.pbkdf2_hmac('sha256', pin.encode(), bytes.fromhex(record['salt']), 200000).hex()
    return secrets.compare_digest(hashed, record['hash'])


def lock_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def device_lock(channel):
    if not channel.isalnum() or len(channel) > 20:
        raise ValueError('无效的 CAN 通道')
    return Path('/tmp') / f'nero_{channel}_control.lock'


def occupied(exclude=()):
    """Read-only evidence; never kill an application or claim this is a hardware lock."""
    ignored = {os.getpid(), *exclude}
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in ignored:
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            executable = [Path(a).name for a in argv if a and not a.startswith('-')]
            legacy = ('debug.py' in executable and 'pipeline' in argv)
            collector = any(a in executable for a in ('calibrate.py', 'auto_collect.py',
                           'reference_board.py', 'workflow.py', 'visual_servo_probe.py'))
            if legacy or collector:
                found.append({'pid': int(proc.name), 'program': ' '.join(argv[:4])[:250]})
        except (OSError, PermissionError):
            continue
    return found


def result_summary(path, data=None):
    if not path:
        return None
    data = read(path) if data is None else data
    if data is None:
        return None
    errors = data.get('all_sample_errors_m_deg', []) + data.get('holdout_errors_m_deg', [])
    return {'path': str(path), 'sha256': digest(path), 'quality_passed': data.get('quality_passed') is True,
            'sample_count': data.get('sample_count', len(data.get('all_sample_errors_m_deg', []))),
            'max_position_mm': max((e[0]*1000 for e in errors), default=None),
            'max_angle_deg': max((e[1] for e in errors), default=None),
            'thresholds': data.get('thresholds', {}), 'data': data}


def event(path, **values):
    write(path, {'time': time.time(), **values})
