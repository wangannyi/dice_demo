"""Bounded lifecycle for FAST SDK and camera resources; no background robot commands."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from cup_grasp_demo.calibration_debug.core import ROOT, digest, read_json


class SDKClient:
    def __init__(self, cfg, directory):
        self.cfg = cfg
        self.log = (directory / 'sdk_worker.log').open('x')
        self.process = subprocess.Popen([
            os.environ.get('DICE_SDK_PYTHON', '/home/test2/agilex-api-test/venv/bin/python'),
            str(Path(__file__).with_name('green_sdk_worker.py')), '--channel', cfg['channel']],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1)
        try:
            if self.receive(15).get('ready') is not True:
                raise RuntimeError('SDK worker did not become ready')
        except BaseException:
            self.close()
            raise

    def receive(self, timeout):
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError('Persistent SDK response timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('SDK worker exited; see sdk_worker.log')
        return json.loads(line)

    def call(self, command, output, request=None, *, on_dispatched=None):
        message = dict(command=command, output=str(output))
        if command == 'snapshot' and self.cfg.get('green_cup', {}).get('fast_cached_feedback', False):
            message['cached_snapshot'] = True
        stages = 1
        if request is not None:
            message.update(request=str(request), sha256=digest(request))
            plan = read_json(request)['plan']
            stages = max(1, len(plan.get('stages', [])))
        try:
            self.process.stdin.write(json.dumps(message) + '\n')
            self.process.stdin.flush()
            if on_dispatched is not None:
                on_dispatched()
            # Feedback overlap can intentionally defer hand startup by up to 30 s.
            hand_budget = 0
            if request is not None and plan.get('kind') == 'feedback_together':
                from cup_grasp_demo.calibration_debug.feedback_sequence import sequence_values
                request_cfg = read_json(request)['config']
                duration = request_cfg['green_cup']['finger_duration_s']
                targets, interval = sequence_values(plan.get('hand_sequence'), plan['target_0_100'], duration)
                hand_budget = 35 + (len(targets)-1)*interval
            answer = self.receive(stages * self.cfg['timeout_s'] + 15 + hand_budget + (plan.get('duration_s', 0) if command == 'shake' else 0))
            if answer.get('output') != str(output):
                raise RuntimeError('SDK receipt path mismatch')
            report = read_json(output)
            if answer.get('returncode') or not report.get('success'):
                # Let the workflow rebuild a stale shake plan once, only when
                # the executor explicitly confirms that no motion was attempted.
                if (command == 'shake'
                        and report.get('failure_code') == 'start_position_changed'
                        and report.get('motion_attempted') is False):
                    return report
                raise RuntimeError(f"{report.get('error')}; 日志：{output.with_suffix('.log')}")
            return report
        except BaseException:
            self.close()
            raise

    def close(self):
        proc = self.process
        if proc.poll() is None:
            try:
                proc.stdin.write('{"command":"close"}\n')
                proc.stdin.flush()
                proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for pipe in (proc.stdin, proc.stdout):
            pipe.close()
        self.log.close()


class VisionResources:
    def __init__(self, cfg, output):
        from dice_cup_localization.capture_rgbd import CaptureSession, argument_parser
        from cup_grasp_demo.calibration_debug.green_capture import capture_arguments
        self.cfg = cfg
        args = argument_parser().parse_args(['--output', str(output), '--serial', cfg['serial'],
                                            *capture_arguments(cfg)])
        args.unique_warmup = True
        args.fast_storage = cfg['green_cup'].get('fast_uncompressed_capture', False)
        args.warmup_frames = cfg['green_cup'].get('fast_camera_warmup_frames', 20)
        args.fresh_discard_frames = cfg['green_cup'].get('fast_camera_fresh_discard_frames', 2)
        if type(args.warmup_frames) is not int or not 1 <= args.warmup_frames <= 60:
            raise ValueError('fast_camera_warmup_frames must be 1..60')
        if type(args.fresh_discard_frames) is not int or not 0 <= args.fresh_discard_frames <= 5:
            raise ValueError('fast_camera_fresh_discard_frames must be 0..5')
        self.camera = CaptureSession(args)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='green-prepare')
        self.camera_future = self.pool.submit(self.camera.start)
        self.model_future = self.pool.submit(self.load_model)
        self.closed = False

    def load_model(self):
        from cup_grasp_demo.calibration_debug.green_yolo import configured_session
        p = self.cfg['green_cup']['perception']
        path = (ROOT / p['model']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Model must be inside dice_demo')
        return configured_session(p)

    def capture(self, output, frames):
        self.camera_future.result(timeout=15)
        self.model_future.result(timeout=30)
        self.camera.capture(output, frames, fresh=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Startup has bounded SDK frame waits; join before stopping that pipeline.
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.camera.close()
