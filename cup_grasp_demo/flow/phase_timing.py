"""Wall-clock stage timing without sleeps or changes to motion scheduling."""

from contextlib import contextmanager
import time


def timing_line(name, seconds, status="completed"):
    suffix = "" if status == "completed" else "（未完成）"
    return f"[耗时] {name}：{seconds:.3f} s{suffix}"


def emit_line(message):
    print(message, flush=True)


class PhaseTimer:
    def __init__(self, emit=emit_line, clock=None):
        self.records = []
        self.emit = emit
        self.clock = clock or time.perf_counter

    @contextmanager
    def phase(self, name):
        started = self.clock()
        status = "failed"
        try:
            yield
            status = "completed"
        finally:
            elapsed = self.clock() - started
            self.records.append(dict(phase=name, duration_s=elapsed, status=status))
            self.emit(timing_line(name, elapsed, status))

    def summary(self, name):
        elapsed = sum(row["duration_s"] for row in self.records)
        self.emit(f"[耗时汇总] {name}：{elapsed:.3f} s（阶段合计，不含确认等待）")
        return elapsed
