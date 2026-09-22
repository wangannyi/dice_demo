"""RTSP publishing through a gst-launch subprocess fed with raw BGR frames.

The publish pipeline mirrors dice-game's yolov8_segdetect streamer
(queue leaky -> videoconvert -> NV12 -> spacemith264enc -> h264parse ->
rtspclientsink).  Keeping gst-launch in a child process isolates the vendor
VPU library's stdout chatter, so the parent's JSON-lines stdout protocol
(control mode) stays clean.  The writer keeps only the newest frame (frames
are dropped, never queued) and the child is restarted with a backoff when it
dies; publishing failures never propagate to the capture pipeline.
"""

import subprocess
import threading
import time


def rtsp_settings(raw):
    """Validate and normalise the green_cup.rtsp config section."""
    if not isinstance(raw, dict):
        raise ValueError("green_cup.rtsp must be an object")
    unknown = set(raw) - {"enabled", "host", "port", "path"}
    if unknown:
        raise ValueError(f"Unknown green_cup.rtsp keys: {sorted(unknown)}")
    host = raw.get("host", "127.0.0.1")
    port = raw.get("port", 8554)
    path = raw.get("path", "/dice/seg")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("green_cup.rtsp.host must be a non-empty string")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("green_cup.rtsp.port must be an integer 1..65535")
    if not isinstance(path, str) or not path.startswith("/") or any(c.isspace() for c in path):
        raise ValueError("green_cup.rtsp.path must start with '/' and contain no whitespace")
    return dict(enabled=bool(raw.get("enabled", False)), host=host, port=port, path=path)


class RtspStreamer:
    """Latest-only BGR frame feeder for one gst-launch RTSP publish pipeline."""

    def __init__(self, host, port, path, width, height, fps, restart_backoff_s=5.0):
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("RTSP stream dimensions must be positive integers")
        if type(fps) is not int or fps <= 0:
            raise ValueError("RTSP stream fps must be a positive integer")
        self._url = f"rtsp://{host}:{port}{path}"
        self._width, self._height = width, height
        self._fps = fps
        # Frames are converted to NV12 in-process (see _to_nv12) and framed
        # by rawvideoparse (no pixel conversion).  GStreamer's videoconvert
        # silently emits zero pixels for BGR->NV12 when fed from fdsrc on
        # this board's riscv64 ORC build, and feeding NV12 straight past
        # fdsrc caps fails with a basesrc flow error (2026-09-22, verified).
        self._blocksize = width * height * 3 // 2
        self._backoff = restart_backoff_s
        self._proc = None
        self._stdin = None
        self._spawned = 0
        self._restart_not_before = 0.0
        self._slot_lock = threading.Lock()
        self._slot = None
        self._wakeup = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._writer_loop, name="green-rtsp", daemon=True)
        self.sent = 0
        self.dropped = 0
        self.last_error = ""
        self._thread.start()

    def submit(self, frame):
        """Publish the newest frame; an older pending frame is dropped."""
        if self._stopped.is_set():
            return
        with self._slot_lock:
            if self._slot is not None:
                self.dropped += 1
            self._slot = frame
        self._wakeup.set()

    def url(self):
        return self._url

    def command(self):
        w, h = self._width, self._height
        return [
            "gst-launch-1.0", "-q",
            "fdsrc", "fd=0", f"blocksize={self._blocksize}", "do-timestamp=true", "!",
            # rawvideoparse only frames the NV12 byte stream; no conversion.
            "rawvideoparse", "format=nv12", f"width={w}", f"height={h}",
            f"framerate={self._fps}/1", "!",
            "queue", "max-size-buffers=2", "leaky=downstream", "!",
            # "code-hight" is the vendor's actual property spelling.
            "spacemith264enc", f"coding-width={w}", f"code-hight={h}", "!",
            "h264parse", "config-interval=-1", "!",
            "video/x-h264,stream-format=byte-stream,alignment=au", "!",
            "rtspclientsink", f"location={self._url}", "protocols=tcp", "latency=0",
        ]

    @staticmethod
    def _to_nv12(frame):
        """BGR ndarray -> NV12 bytes (Y plane followed by interleaved UV)."""
        import cv2
        import numpy as np
        h, w = frame.shape[:2]
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        y = yuv[:h].reshape(-1)
        chroma = yuv[h:]
        half = chroma.shape[0] // 2
        u = chroma[:half].reshape(-1)
        v = chroma[half:].reshape(-1)
        uv = np.empty(u.size + v.size, dtype=np.uint8)
        uv[0::2] = u
        uv[1::2] = v
        return y.tobytes() + uv.tobytes()

    def _writer_loop(self):
        while not self._stopped.is_set():
            self._wakeup.wait(timeout=1.0)
            if self._stopped.is_set():
                break
            with self._slot_lock:
                frame, self._slot = self._slot, None
            self._wakeup.clear()
            if frame is None:
                continue
            if not self._ensure_running():
                self.dropped += 1
                continue
            try:
                payload = frame if isinstance(frame, bytes) else self._to_nv12(frame)
                self._stdin.write(payload)
                self._stdin.flush()
                self.sent += 1
            except (BrokenPipeError, OSError) as exc:
                self.last_error = str(exc)
                self._kill_child()

    def _ensure_running(self):
        if self._proc is not None and self._proc.poll() is None:
            return True
        self._kill_child()
        if time.monotonic() < self._restart_not_before:
            return False
        try:
            # stdout is devnull'd: the VPU encoder library chats on stdout and
            # must never reach this process' JSON-lines stdout; stderr passes
            # through so real GStreamer errors stay visible.
            self._proc = subprocess.Popen(self.command(), stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=None)
            self._stdin = self._proc.stdin
            self._spawned += 1
            return True
        except OSError as exc:
            self.last_error = str(exc)
            self._restart_not_before = time.monotonic() + self._backoff
            return False

    def _kill_child(self):
        proc, self._proc, self._stdin = self._proc, None, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for pipe in (proc.stdin,):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass

    def restarts(self):
        return max(0, self._spawned - 1)

    def close(self):
        self._stopped.set()
        self._wakeup.set()
        # Unblock a writer stuck in write(): killing the child turns the pipe
        # into an error and the loop exits on the stopped flag.
        self._kill_child()
        self._thread.join(timeout=3)
        self._kill_child()
