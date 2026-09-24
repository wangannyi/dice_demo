"""Bounded lifecycle for FAST SDK and camera resources; no background robot commands."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib
ROOT = Path(__file__).resolve().parents[2]

from cup_grasp_demo.flow.green_rtsp import RtspStreamer, rtsp_settings


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


class SDKClient:
    def __init__(self, cfg, directory, *, worker_argv=None):
        self.cfg = cfg
        self.directory = Path(directory)
        self.log_path = (self.directory / 'sdk_worker.log').resolve()
        # Test seam: a fake worker argv replaces the real subprocess command.
        self._worker_argv = worker_argv
        self._log_index = 0
        self._spawn()

    def _spawn(self):
        """Start one SDK worker subprocess and wait for its ready line."""
        self.log = self.log_path.open('a')
        argv = self._worker_argv or [
            os.environ.get('DICE_SDK_PYTHON', '/usr/bin/python3'),
            str(Path(__file__).with_name('green_sdk_worker.py')), '--channel', self.cfg['channel'],
            '--evidence-output', str((self.directory / 'sdk_startup.json').resolve())]
        self._log_index += 1
        if self._log_index > 1:
            self.log.write(f'\n===== worker restart #{self._log_index - 1} =====\n')
            self.log.flush()
        self.process = subprocess.Popen(argv,
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1)
        try:
            answer = self.receive(15)
            if answer.get('ready') is not True:
                raise RuntimeError(f"SDK worker 启动失败：{answer.get('error', '未就绪')}；日志：{self.log_path}")
        except BaseException:
            self._terminate()
            raise

    def restart(self):
        """Replace a dead worker; the old process is gone, only pipes need closing."""
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except OSError:
                pass
        self.log.close()
        self._spawn()

    def _dead_worker_receipt(self, output):
        """Receipt of a dead worker; None unless the file parses. No motion checks here."""
        if not output.exists():
            return None
        try:
            return read_json(output)
        except (OSError, ValueError):
            return None

    def _zero_tx_failure(self, report):
        """A dead worker may be restarted only when its receipt proves nothing moved."""
        tx = report.get('tx') or {}
        return (report.get('success') is False
                and report.get('motion_attempted') is False
                and report.get('finger_commands_sent') in (None, 0)
                and tx.get('actual_tx_count') == 0
                and tx.get('transmission_outcome_uncertain') is False)

    def receive(self, timeout):
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError('Persistent SDK response timed out')
        line = self.process.stdout.readline()
        if not line:
            detail = ''
            try:
                with self.log_path.open('rb') as log:
                    log.seek(0, 2)
                    log.seek(max(0, log.tell()-4096))
                    lines = log.read().decode(errors='replace').strip().splitlines()
                    detail = ': '+lines[-1][:800] if lines else ''
            except OSError:
                pass
            raise RuntimeError(f'SDK worker exited{detail}；日志：{self.log_path}')
        return json.loads(line)

    def call(self, command, output, request=None, *, on_dispatched=None):
        try:
            return self._call_once(command, output, request, on_dispatched=on_dispatched)
        except RuntimeError as exc:
            # Worker-death recovery: only when the subprocess is confirmed dead.
            # A timeout never restarts (the worker may still be executing; a
            # second executor is unsafe). The receipt then decides:
            #   success=True      -> the command actually finished; return it
            #   zero-transmission -> restart once and resend
            #   anything else     -> surface the failure (human intervention)
            if 'SDK worker exited' not in str(exc):
                self.close()
                raise
            report = self._dead_worker_receipt(output)
            if report is not None and report.get('success') is True:
                # Command actually finished; recover the connection, never resend.
                self.restart()
                return report
            if report is None or not self._zero_tx_failure(report):
                self.close()
                raise
        # Receipt proved nothing moved and nothing was sent; one restart+retry.
        self.restart()
        report = self._call_once(command, output, request)
        report['worker_restarted'] = True
        return report

    def _call_once(self, command, output, request=None, *, on_dispatched=None):
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
                from cup_grasp_demo.flow.feedback_sequence import sequence_values
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
            self._terminate()
            raise

    def _terminate(self):
        """Stop the worker process and close its pipes; the log stays open."""
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
            try:
                pipe.close()
            except OSError:
                pass

    def close(self):
        self._terminate()
        self.log.close()


class VisionResources:
    def __init__(self, cfg, output):
        from vision.capture.realsense_session import CaptureSession, argument_parser
        from cup_grasp_demo.flow.green_capture import capture_arguments
        self.cfg = cfg
        from vision.capture.config import load_camera_config
        camera = load_camera_config()
        args = argument_parser().parse_args(['--output', str(output), '--serial', camera['serial'],
                                            *capture_arguments(cfg)])
        args.unique_warmup = True
        args.fast_storage = cfg['green_cup'].get('fast_uncompressed_capture', False)
        args.warmup_frames = camera['warmup_frames']
        args.fresh_discard_frames = camera['fresh_discard_frames']
        if type(args.warmup_frames) is not int or not 1 <= args.warmup_frames <= 60:
            raise ValueError('fast_camera_warmup_frames must be 1..60')
        if type(args.fresh_discard_frames) is not int or not 0 <= args.fresh_discard_frames <= 5:
            raise ValueError('fast_camera_fresh_discard_frames must be 0..5')
        self.camera = CaptureSession(args)
        # RTSP publishing (green_cup.rtsp): one reader thread owns the camera
        # and feeds cropped color frames to a gst-launch publish pipeline.
        self.streamer = None
        settings = rtsp_settings(cfg['green_cup'].get('rtsp', {}))
        if settings['enabled']:
            crop = self.camera.crop
            width, height = ((crop[2], crop[3]) if crop
                             else tuple(self.camera.color_resolution))
            self.streamer = RtspStreamer(settings['host'], settings['port'],
                                         settings['path'], width, height, args.fps)
            self.camera.set_stream_sink(self.streamer.submit)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='green-prepare')
        self.camera_future = self.pool.submit(self.camera.start)
        self.model_future = self.pool.submit(self.load_model)
        self.closed = False

    def load_model(self):
        from vision.inference.detector import configured_session
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
        if self.streamer is not None:
            self.streamer.close()
