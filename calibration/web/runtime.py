"""Adapters installed only inside a dedicated web worker, never in legacy CLI runs."""
import math
import os
import signal
import threading
import time

from calibration.apply_result import atomic_bytes
from calibration.web.common import event, read


def confirm_hold(arm, timeout=4.):
    """Send a fresh holding target, then require a stationary fresh-feedback window."""
    q = arm.read_joints()['joints_rad']
    arm.robot.move_js(q)
    return confirm_stationary(arm, timeout)


def confirm_stationary(arm, timeout=4.):
    baseline = None
    since = None
    last_stamp = None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sample = arm.read_joints()  # Requires a new SDK timestamp; cached feedback cannot pass.
        stamp = sample.get('sdk_timestamp')
        if stamp is None or stamp == last_stamp:
            raise RuntimeError('无法确认停止：关节反馈时间戳没有更新')
        last_stamp = stamp
        q = sample['joints_rad']
        if len(q) != 7 or not all(math.isfinite(v) for v in q):
            raise ValueError('关节反馈无效')
        status = arm.robot.get_arm_status()
        if arm.robot.has_comm_error() or status is None:
            raise RuntimeError('无法确认停止：机械臂通信异常')
        now = time.monotonic()
        if baseline is None or max(abs(a-b) for a, b in zip(q, baseline)) > math.radians(.1):
            baseline, since = q, now
        elif status.msg.motion_status == 0 and now - since >= .5:
            return {'confirmed': True, 'observed_at': time.time(), 'joints_rad': q,
                    'sdk_timestamp': sample['sdk_timestamp'], 'stationary_window_s': now-since}
    raise RuntimeError('未取得连续静止反馈；请现场检查，必要时使用硬件急停')


class LiveCamera:
    """One camera reader; JPEG publication is off the arm command path."""
    def __init__(self, base, job):
        self.base, self.job = base, job
        self.K, self.D, self.info = base.K, base.D, base.info
        self.condition = threading.Condition()
        self.frame = None
        self.counter = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._stream, name='calibration-video', daemon=True)
        self.thread.start()

    def _stream(self):
        import cv2
        last_write = 0.
        try:
            while not self.closed:
                frame = self.base.capture()
                with self.condition:
                    self.frame = frame.copy()
                    self.counter += 1
                    self.condition.notify_all()
                now = time.monotonic()
                if now-last_write >= .2:
                    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        atomic_bytes(self.job/'preview.jpg', encoded.tobytes())
                        event(self.job/'frame.json', width=frame.shape[1], height=frame.shape[0], camera=self.info)
                    last_write = now
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()
            if not self.closed:
                event(self.job/'camera_error.json', error=self.error)
                os.kill(os.getpid(), signal.SIGINT)

    def capture(self):
        with self.condition:
            previous = self.counter
            ready = self.condition.wait_for(
                lambda: self.counter != previous or self.error or self.closed, timeout=4.)
            if self.error or not ready or self.closed:
                raise RuntimeError(self.error or '相机没有新画面')
            return self.frame.copy()

    def close(self):
        if self.closed:
            return
        self.closed = True
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=4.)
        self.base.close()


class BrowserPreview:
    """The collector's existing preview callback keeps per-frame teaching feedback."""
    def __init__(self, job):
        self.job = job
        self.board_window = None

    def restore(self, image, count, quality):
        self.result(True, count, '已载入样本', quality, image)

    def result(self, accepted, count, message, quality=None, image=None):
        if image is not None:
            import cv2
            ok, encoded = cv2.imencode('.jpg', image)
            if ok:
                atomic_bytes(self.job/'sample.jpg', encoded.tobytes())
        event(self.job/'progress.json', phase='sampling', accepted=count,
              last_accepted=accepted, message=message, quality=quality)

    def read_command(self, camera, detector, count, on_frame=None):
        while True:
            image = camera.capture()
            try:
                board, quality, vis = detector.detect(image, camera.K, camera.D)
                detection = {'detected': True, 'quality': quality}
            except ValueError as exc:
                board = quality = None
                detection = {'detected': False, 'reason': str(exc)}
            if on_frame:
                on_frame(image, board, quality)
            event(self.job/'detection.json', **detection)
            commands = sorted((self.job/'commands').glob('*.json'))
            if not commands:
                continue
            command = read(commands[0])['command']
            commands[0].unlink()
            if command == 'finish':
                return 'q'
            if command == 'sample':
                if board is None:
                    self.result(False, count, '未识别到完整标定板，请调整姿态或遮挡桌面板')
                    continue
                event(self.job/'progress.json', phase='capturing', accepted=count,
                      message='正在采样，请保持机械臂和标定板静止')
                return ''

    def close(self):
        pass


class BrowserAutoView:
    def __init__(self, job, box, total):
        self.job, self.box, self.total = job, box, total

    def update(self, image, state, count=0, quality=None, board_outline=None):
        # Called at rest only. The independent camera reader publishes live motion video.
        event(self.job/'progress.json', phase=state, accepted=count, total=self.total, quality=quality)
        if quality:
            import cv2
            ok, encoded = cv2.imencode('.jpg', image)
            if ok:
                atomic_bytes(self.job/'sample.jpg', encoded.tobytes())

    def status(self, state, count):
        event(self.job/'progress.json', phase=state, accepted=count, total=self.total)

    def pump(self):
        pass

    def close(self):
        pass
