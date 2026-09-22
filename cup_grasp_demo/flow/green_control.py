"""JSON-lines control for one persistent green-cup pipeline process.

Only the command reader owns phase progression.  The SDK and camera are opened
once before ``ready`` and stay open while the reader waits for another command.
"""

import copy
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

from cup_grasp_demo.flow.core import cached_screen_geometry
from cup_grasp_demo.flow.green_pipeline import PHASES, Workflow
from cup_grasp_demo.flow.pipeline_runner import save
from cup_grasp_demo.flow.session_storage import session_lock


def emit(stream, event, **fields):
    stream.write(json.dumps(dict(event=event, **fields), ensure_ascii=False,
                            allow_nan=False) + "\n")
    stream.flush()


class ControlSession:
    """Advance a single state machine; reject skips and duplicate command IDs."""

    def __init__(self, flow, state_path, incoming, outgoing, diagnostic):
        self.flow = flow
        self.state_path = Path(state_path)
        self.incoming = incoming
        self.outgoing = outgoing
        self.diagnostic = diagnostic
        self.next_index = 0
        self.cycle = 1
        self.seen_ids = set()
        self.state = self._new_state()
        save(self.state_path, self.state)

    def _new_state(self):
        return dict(strategy="green_open_cup", mode="control", cycle=self.cycle,
                    status="WAITING", events=[], phase_timings_s={})

    def _next_phase(self):
        return PHASES[self.next_index] if self.next_index < len(PHASES) else None

    def _reply(self, event, request_id=None, **fields):
        emit(self.outgoing, event, id=request_id, cycle=self.cycle,
             status=self.state["status"], next_phase=self._next_phase(), **fields)

    def _reject(self, request_id, code, message):
        self._reply("rejected", request_id, code=code, message=message)

    def _advance(self, request_id, target_index):
        if self.next_index > target_index:
            self._reject(request_id, "already_completed", "目标阶段已经完成，不能重复执行")
            return True
        while self.next_index <= target_index:
            phase = PHASES[self.next_index]
            self.state["status"] = "RUNNING"
            self.state["active_phase"] = phase
            save(self.state_path, self.state)
            self._reply("phase_started", request_id, phase=phase)
            started = time.perf_counter()
            try:
                self.flow.unchanged()
                with redirect_stdout(self.diagnostic):
                    self.flow.perform(phase)
            except Exception as exc:
                self.state.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                                  receipts=getattr(self.flow, "receipts", {}),
                                  recovery_events=getattr(self.flow, "recovery_events", []))
                save(self.state_path, self.state)
                self._reply("failed", request_id, phase=phase,
                            error=self.state["error"], state_file=str(self.state_path))
                return False
            elapsed = time.perf_counter() - started
            self.state["events"].append(dict(phase=phase, status="completed"))
            self.state["phase_timings_s"][phase] = elapsed
            self.state["receipts"] = getattr(self.flow, "receipts", {})
            self.state["recovery_events"] = getattr(self.flow, "recovery_events", [])
            self.next_index += 1
            self.state["status"] = "COMPLETED" if self._next_phase() is None else "WAITING"
            save(self.state_path, self.state)
            self._reply("phase_completed", request_id, phase=phase, elapsed_s=elapsed)
        self._reply("command_completed", request_id, through=PHASES[target_index])
        return True

    def _handle(self, request):
        if not isinstance(request, dict):
            self._reject(None, "invalid_request", "每行必须是 JSON 对象")
            return True
        request_id = request.get("id")
        if (request_id is not None and
                (isinstance(request_id, bool) or not isinstance(request_id, (str, int))
                 or len(str(request_id)) > 64)):
            self._reject(None, "invalid_id", "id 必须是长度不超过 64 的字符串或整数")
            return True
        if request_id is not None:
            if request_id in self.seen_ids:
                self._reject(request_id, "duplicate_id", "此命令 id 已处理，不能重复派发")
                return True
            self.seen_ids.add(request_id)
        command = request.get("command")
        if command == "status":
            self._reply("status", request_id, completed_phases=[
                item["phase"] for item in self.state["events"]],
                state_file=str(self.state_path))
            return True
        if command == "close":
            if self.state["status"] != "COMPLETED":
                self.state["status"] = "PAUSED"
                save(self.state_path, self.state)
            self._reply("closed", request_id, state_file=str(self.state_path))
            return False
        if command == "new_cycle":
            if self.state["status"] != "COMPLETED":
                self._reject(request_id, "cycle_in_progress", "仅完成 RETURN_HOME 后可开始新一轮")
                return True
            self.cycle += 1
            self.next_index = 0
            self.flow.receipts = {}
            self.flow.held = None
            self.flow._snapshot_cache = None
            if hasattr(self.flow, "recovery_events"):
                self.flow.recovery_events = []
            self.state = self._new_state()
            save(self.state_path, self.state)
            self._reply("ready", request_id, phases=list(PHASES),
                        state_file=str(self.state_path))
            return True
        if command == "refresh_perception":
            if self.next_index not in (2, 3):
                self._reject(request_id, "invalid_phase", "仅 CAPTURE 后、APPROACH 前可重新定位")
                return True
            self.next_index = 1
            self._reply("perception_reset", request_id)
            return True
        if command == "advance":
            target = request.get("until")
            if target is None:
                target_index = self.next_index
            elif isinstance(target, str) and target.upper() in PHASES:
                target_index = PHASES.index(target.upper())
            else:
                self._reject(request_id, "invalid_phase", "until 必须是 pipeline 阶段名")
                return True
            if target_index >= len(PHASES):
                self._reject(request_id, "cycle_completed", "本轮已完成；发送 new_cycle 或 close")
                return True
            return self._advance(request_id, target_index)
        self._reject(request_id, "invalid_command",
                     "command 必须是 status、advance、refresh_perception、new_cycle 或 close")
        return True

    def serve(self):
        self._reply("ready", phases=list(PHASES), state_file=str(self.state_path))
        while True:
            line = self.incoming.readline()
            if not line:
                if self.state["status"] != "COMPLETED":
                    self.state["status"] = "PAUSED"
                    save(self.state_path, self.state)
                self._reply("closed", reason="stdin_eof", state_file=str(self.state_path))
                return 0
            if len(line) > 4096:
                self._reject(None, "invalid_request", "单条命令超过 4096 字符")
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                self._reject(None, "invalid_json", "每行必须是完整 JSON")
                continue
            if not self._handle(request):
                return 2 if self.state["status"] == "FAILED" else 0


def run(args, incoming=None, outgoing=None, diagnostic=None):
    """Own the device lock and resources until a close, EOF or phase failure."""
    incoming = incoming if incoming is not None else sys.stdin
    outgoing = outgoing if outgoing is not None else sys.stdout
    diagnostic = diagnostic if diagnostic is not None else sys.stderr
    if args.status:
        from cup_grasp_demo.flow.core import read_json
        emit(outgoing, "status", **read_json(args.session / "green_pipeline_state.json"))
        return 0
    if args.resume or args.show or args.until not in (None, "place"):
        emit(outgoing, "failed", error="control 模式不支持 --resume/--show/--until 非 place")
        return 2
    if not args.execute:
        emit(outgoing, "preview", phases=list(PHASES), execute=False)
        return 0
    runtime_args = copy.copy(args)
    # Run the phase machine with FAST parameters (fast_speed_percent 30,
    # approach/return 60, finger 0.25 s, trapezoid profile, no joint-snapshot
    # brackets around capture, background capture when already home).  The
    # persistent startup below still pre-connects SDK/camera/model like STEP.
    runtime_args.mode = "fast"
    runtime_args.show = False
    flow = None
    server = None
    try:
        with session_lock(args.session), cached_screen_geometry():
            try:
                with redirect_stdout(diagnostic):
                    flow = Workflow(runtime_args)
                    if flow.g.get("installation_requires_calibration", False):
                        raise ValueError("请先完成现场标定和桌面登记")
                    flow.prepare_step_runtime()
                server = ControlSession(flow, args.session / "green_pipeline_state.json",
                                        incoming, outgoing, diagnostic)
                return server.serve()
            finally:
                if flow is not None:
                    with redirect_stdout(diagnostic):
                        flow.close()
    except BaseException as exc:
        if server is not None and server.state["status"] != "FAILED":
            server.state.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            save(server.state_path, server.state)
        emit(outgoing, "failed", error=f"{type(exc).__name__}: {exc}",
             state_file=str(args.session / "green_pipeline_state.json"))
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
