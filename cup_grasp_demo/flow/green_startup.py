"""Overlap read-only FAST startup with heavy CLI imports; never send motion."""
import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import uuid

_pending = None


class Startup:
    def __init__(self, config, session, raw, digest):
        from cup_grasp_demo.flow.green_runtime import SDKClient, VisionResources
        self.config, self.session, self.digest = config, session, digest
        self.started = time.perf_counter()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='fast-startup')
        self.claimed = set()
        directory = session/'runs'/('startup_'+uuid.uuid4().hex[:12])
        directory.mkdir(parents=True)
        self.futures = {
            'sdk': self.pool.submit(SDKClient, raw, directory),
            'vision': self.pool.submit(VisionResources, raw, directory/'unused_capture'),
        }
        self.closed = False

    def acquire(self, key):
        value = self.futures[key].result(timeout=35)
        self.claimed.add(key)
        return value

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
        for key, future in self.futures.items():
            if key not in self.claimed and not future.cancelled() and future.exception() is None:
                future.result().close()


def launch(argv):
    global _pending
    # Previews, STEP/AUTO, status, resume, and other tools keep their original lifecycle.
    if not argv or argv[0] != 'pipeline' or '--execute' not in argv or any(
            x in argv for x in ('--help', '-h', '--status', '--resume')):
        return
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument('--mode'); parser.add_argument('--config'); parser.add_argument('--session')
    try:
        args, _ = parser.parse_known_args(argv[1:])
        if args.mode != 'fast' or not args.config or not args.session:
            return
        config, session = Path(args.config).resolve(), Path(args.session).resolve()
        content = config.read_bytes(); raw = json.loads(content)
        green = raw.get('green_cup', {})
        if (raw.get('pipeline_strategy') != 'green_open_cup'
                or green.get('installation_requires_calibration', False)
                or not green.get('persistent_runtime', True)
                or not green.get('fast_parallel_startup', False)):
            return
    except (OSError, ValueError, argparse.ArgumentError):
        return  # Full CLI validation will report the original error.
    _pending = Startup(config, session, raw, hashlib.sha256(content).hexdigest())
    atexit.register(_pending.close)


def claim(config, session):
    global _pending
    value = _pending
    if value is None:
        return None
    if (Path(config).resolve() != value.config or Path(session).resolve() != value.session
            or hashlib.sha256(value.config.read_bytes()).hexdigest() != value.digest):
        value.close()
        _pending = None
        raise ValueError('FAST startup configuration changed')
    _pending = None
    return value
