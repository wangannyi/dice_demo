"""Deadline-based target streaming and read-only feedback collection."""
import math
import threading
import time


class DeadlineClock:
    """Skip missed slots instead of sending a burst of overdue targets."""
    def __init__(self, rate_hz, start, *, monotonic=time.monotonic, sleep=time.sleep):
        self.period = 1.0 / rate_hz
        self.start = start
        self.deadline = start
        self.monotonic, self.sleep = monotonic, sleep
        self.skipped = 0
        self.max_lateness_s = 0.0

    def wait(self):
        now = self.monotonic()
        while now < self.deadline:
            self.sleep(self.deadline - now)
            now = self.monotonic()
        self.max_lateness_s = max(self.max_lateness_s, now - self.deadline)
        return now

    def advance(self):
        self.deadline += self.period
        now = self.monotonic()
        if self.deadline < now:
            missed = math.floor((now - self.deadline) / self.period) + 1
            self.skipped += missed
            self.deadline += missed * self.period


class _ReaderStopped(Exception):
    pass


class FeedbackReader:
    """Only this worker reads snapshots; only the calling thread sends targets."""
    def __init__(self, read, initial, *, joint_max_age_s=.1, state_max_age_s=.25):
        self.read = read
        self.row = initial
        self.joint_max_age_s = joint_max_age_s
        self.state_max_age_s = state_max_age_s
        self.error = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name='shake-feedback', daemon=True)

    def _sleep(self, duration):
        if self.stop.wait(duration):
            raise _ReaderStopped()

    def _run(self):
        previous = self.row
        try:
            while not self.stop.is_set():
                row = self.read(previous=previous, sleep=self._sleep,
                                joint_max_age_s=self.joint_max_age_s,
                                state_max_age_s=self.state_max_age_s)
                if self.stop.is_set():
                    break
                with self.lock:
                    self.row = row
                previous = row
        except _ReaderStopped:
            pass
        except BaseException as exc:
            with self.lock:
                self.error = exc

    def start(self):
        self.thread.start()
        return self

    def latest(self, now_epoch):
        with self.lock:
            row, error = self.row, self.error
        if error is not None:
            raise RuntimeError('Shake feedback reader failed: ' + str(error)) from error
        stamps = row['sdk_snapshot']['packet_timestamps_after_epoch_s'].values()
        if any(not 0 <= now_epoch - stamp <= self.joint_max_age_s for stamp in stamps):
            raise RuntimeError(
                f'Shake joint feedback exceeds {self.joint_max_age_s * 1000:g} ms freshness limit'
            )
        if any(not 0 <= now_epoch - stamp <= self.state_max_age_s for stamp in
               [row['status_timestamp_epoch_s'], *row['enable_feedback_timestamps_epoch_s']]):
            raise RuntimeError(
                f'Shake status/enable feedback exceeds {self.state_max_age_s * 1000:g} ms freshness limit'
            )
        return row

    def close(self):
        self.stop.set()
        # A snapshot may be inside its bounded 2 s read. Join before another reader.
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError('Shake feedback reader did not stop')


def stream_statistics(commands, rate_hz, clock):
    times = [row['elapsed_s'] for row in commands]
    intervals = [b-a for a,b in zip(times,times[1:])]
    return dict(requested_hz=rate_hz, command_count=len(times),
                achieved_hz=(len(intervals)/(times[-1]-times[0]) if intervals and times[-1]>times[0] else None),
                max_interval_ms=max(intervals, default=0)*1000,
                skipped_slots=clock.skipped, max_lateness_ms=clock.max_lateness_s*1000,
                timestamp_basis='SDK call completion; not CAN wire acknowledgement')
