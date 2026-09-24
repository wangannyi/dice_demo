"""JSON-lines control for one persistent green-cup pipeline process.

The resident process is a ready-state dispatcher over one SDK/CAN connection:
static actions (gestures, HOME) dispatch directly whenever the grasp flow is
idle, while the vision-coupled phases keep the advance protocol.  Rounds are
gone: finishing RETURN_HOME auto-resets to idle for the next run.
"""

import copy
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

from cup_grasp_demo.flow.core import cached_screen_geometry
from cup_grasp_demo.flow.green_pipeline import PHASES, Workflow
from cup_grasp_demo.flow.core import save
from cup_grasp_demo.flow.session_storage import session_lock


def emit(stream, event, **fields):
    stream.write(json.dumps(dict(event=event, **fields), ensure_ascii=False,
                            allow_nan=False) + "\n")
    stream.flush()


class ControlSession:
    """Advance a single state machine; reject skips and duplicate command IDs."""

    def __init__(self, flow, state_path, incoming, outgoing, diagnostic,
                 actions=(), run_action=None, reload_actions=None):
        self.flow = flow
        self.state_path = Path(state_path)
        self.incoming = incoming
        self.outgoing = outgoing
        self.diagnostic = diagnostic
        self.actions = tuple(actions)
        self.run_action = run_action
        self.reload_actions = reload_actions
        self.next_index = 0
        self.cycle = 1
        self.seen_ids = set()
        self.state = self._new_state()
        save(self.state_path, self.state)

    def _new_state(self):
        return dict(strategy="green_open_cup", mode="control", cycle=self.cycle,
                    status="WAITING", events=[], phase_timings_s={}, actions=[],
                    rounds_total=None, rounds_remaining=None)

    def _next_phase(self):
        return PHASES[self.next_index] if self.next_index < len(PHASES) else None

    def _reply(self, event, request_id=None, **fields):
        emit(self.outgoing, event, id=request_id, cycle=self.cycle,
             status=self.state["status"], next_phase=self._next_phase(), **fields)

    def _reject(self, request_id, code, message):
        self._reply("rejected", request_id, code=code, message=message)

    def _inter_round_request(self):
        """Non-blocking peek between rounds; returns a parsed request or None.

        Only streams with a real file descriptor (stdin, pipes) support
        select; in-memory test streams simply report "nothing to peek".
        """
        import select
        try:
            fileno = self.incoming.fileno()
        except (AttributeError, OSError, ValueError):
            return None
        ready, _, _ = select.select([fileno], [], [], 0)
        if not ready:
            return None
        line = self.incoming.readline()
        if not line:
            return {"command": "close", "_eof": True}
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            self._reject(None, "invalid_json", "每行必须是完整 JSON")
            return None
        return request if isinstance(request, dict) else None

    def _stop_rounds(self, request_id, reason):
        self.state["status"] = "WAITING"
        self.state["rounds_total"] = None
        self.state["rounds_remaining"] = None
        save(self.state_path, self.state)
        self._reply("rounds_stopped", request_id, reason=reason)

    def _advance(self, request_id, target_index, rounds=1):
        if self.next_index > target_index:
            self._reject(request_id, "already_completed", "目标阶段已经完成，不能重复执行")
            return True
        if rounds > 1:
            self.state["rounds_total"] = rounds
            self.state["rounds_remaining"] = rounds
            save(self.state_path, self.state)
        stop_reason = None
        completed_rounds = 0
        for round_index in range(rounds):
            if round_index:
                # Between rounds the arm is idle at HOME; peek for stop/close.
                while True:
                    inter = self._inter_round_request()
                    if inter is None:
                        break
                    command = inter.get("command")
                    inter_id = inter.get("id")
                    if inter_id is not None:
                        if inter_id in self.seen_ids:
                            self._reject(inter_id, "duplicate_id", "此命令 id 已处理，不能重复派发")
                            continue
                        self.seen_ids.add(inter_id)
                    if command == "stop":
                        self._stop_rounds(inter_id, "stop 命令")
                        stop_reason = "stop"
                        break
                    if command == "close":
                        self._reply("command_completed", request_id,
                                    through=PHASES[target_index],
                                    rounds_completed=completed_rounds)
                        self.state["status"] = "PAUSED"
                        self.state["rounds_total"] = None
                        self.state["rounds_remaining"] = None
                        save(self.state_path, self.state)
                        self._reply("closed", inter_id, state_file=str(self.state_path))
                        return False
                    self._reject(inter_id, "multi_round_busy",
                                 "连跑进行中，仅接受 stop/close；其他命令请连跑结束后再发")
                if stop_reason:
                    break
            if rounds > 1:
                self.state["rounds_remaining"] = rounds - round_index
                save(self.state_path, self.state)
                self._reply("round_started", request_id,
                            round=round_index + 1, rounds=rounds)
            if not self._advance_single(request_id, target_index):
                # Phase failure ends the process; no cross-round retry.
                return False
            completed_rounds += 1
        if rounds > 1:
            self.state["rounds_total"] = None
            self.state["rounds_remaining"] = None
            save(self.state_path, self.state)
            if stop_reason is None:
                self._reply("rounds_completed", request_id, rounds=rounds,
                            runs=self.cycle - 1)
        self._reply("command_completed", request_id, through=PHASES[target_index],
                    rounds_completed=completed_rounds)
        return True

    def _advance_single(self, request_id, target_index):
        if self.next_index > target_index:
            self._reject(request_id, "already_completed", "目标阶段已经完成，不能重复执行")
            return True
        finished_run = False
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
            self.state["status"] = "WAITING"
            save(self.state_path, self.state)
            self._reply("phase_completed", request_id, phase=phase, elapsed_s=elapsed)
            if self._next_phase() is None:
                finished_run = True
        if finished_run:
            self._reply("run_completed", request_id, runs=self.cycle)
            self._reset_for_next_run()
        # command_completed is emitted once by the outer _advance wrapper.
        return True

    def _reset_for_next_run(self):
        """RETURN_HOME finished: auto-reset to idle (no rounds, no new_cycle)."""
        self.cycle += 1
        self.next_index = 0
        self.flow.receipts = {}
        self.flow.held = None
        self.flow._snapshot_cache = None
        if hasattr(self.flow, "recovery_events"):
            self.flow.recovery_events = []
        self.state = self._new_state()
        save(self.state_path, self.state)

    def _run_action(self, request_id, name):
        self.state["status"] = "RUNNING"
        self.state["active_action"] = name
        save(self.state_path, self.state)
        self._reply("action_started", request_id, name=name)
        started = time.perf_counter()
        try:
            result = self.run_action(name)
        except ValueError as exc:
            # Unknown/invalid action name: nothing moved, reject and keep serving.
            self.state["status"] = "WAITING"
            self.state["active_action"] = None
            save(self.state_path, self.state)
            self._reject(request_id, "unknown_action", str(exc))
            return True
        except Exception as exc:
            self.state.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            save(self.state_path, self.state)
            self._reply("failed", request_id, phase=f"action:{name}",
                        error=self.state["error"], state_file=str(self.state_path))
            return False
        elapsed = time.perf_counter() - started
        self.state["status"] = "WAITING"
        self.state["active_action"] = None
        self.state["actions"].append(dict(name=result.get("name", name),
                                          elapsed_s=round(elapsed, 3),
                                          receipt=result.get("receipt")))
        save(self.state_path, self.state)
        self._reply("action_completed", request_id, name=result.get("name", name),
                    receipt=result.get("receipt"), elapsed_s=round(elapsed, 3))
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
            self.state["status"] = "PAUSED"
            save(self.state_path, self.state)
            self._reply("closed", request_id, state_file=str(self.state_path))
            return False
        if command == "new_cycle":
            self._reject(request_id, "removed",
                         "轮次已移除：流程完成自动复位，直接 advance 即可开始下一轮")
            return True
        if command == "actions":
            self._reply("actions", request_id, names=sorted(set(self.actions)))
            return True
        if command == "reload":
            if self.next_index != 0:
                self._reject(request_id, "flow_in_progress",
                             f"抓取流程进行中（下一阶段 {self._next_phase()}），跑完后再重载手势")
                return True
            if self.reload_actions is None:
                self._reject(request_id, "no_actions", "本会话未接入动作注册器")
                return True
            try:
                actions, run_action = self.reload_actions()
            except Exception as exc:
                # Keep the old table on any failure; swap is atomic below.
                self._reject(request_id, "reload_failed", f"{type(exc).__name__}: {exc}")
                return True
            self.actions = tuple(actions)
            self.run_action = run_action
            self._reply("actions_reloaded", request_id, names=sorted(set(self.actions)))
            return True
        if command == "action":
            name = request.get("name")
            if self.run_action is None:
                self._reject(request_id, "no_actions", "本会话未接入动作执行器")
                return True
            if not isinstance(name, str) or not name.strip():
                self._reject(request_id, "invalid_name", "name 必须是动作名字符串")
                return True
            if self.next_index != 0:
                self._reject(request_id, "flow_in_progress",
                             f"抓取流程进行中（下一阶段 {self._next_phase()}），跑完后再执行动作")
                return True
            return self._run_action(request_id, name.strip())
        if command == "refresh_perception":
            if self.next_index not in (2, 3):
                self._reject(request_id, "invalid_phase", "仅 CAPTURE 后、APPROACH 前可重新定位")
                return True
            self.next_index = 1
            self._reply("perception_reset", request_id)
            return True
        if command == "stop":
            self._reject(request_id, "not_in_multi_round",
                         "stop 仅在连跑（advance rounds>1）局间生效；当前不在连跑")
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
                self._reject(request_id, "invalid_phase", "until 超出阶段范围")
                return True
            rounds = request.get("rounds", 1)
            if (isinstance(rounds, bool) or not isinstance(rounds, int)
                    or not 1 <= rounds <= 99):
                self._reject(request_id, "invalid_rounds",
                             "rounds 必须是 1..99 的整数（有界连跑，不提供无限模式）")
                return True
            return self._advance(request_id, target_index, rounds=rounds)
        self._reject(request_id, "invalid_command",
                     "command 必须是 status、advance、refresh_perception、action、actions、reload 或 close")
        return True

    def serve(self):
        self._reply("ready", phases=list(PHASES), actions=sorted(set(self.actions)),
                    state_file=str(self.state_path))
        while True:
            line = self.incoming.readline()
            if not line:
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


def build_action_runtime(flow, diagnostic):
    """Wire gesture execution onto the resident SDK connection.

    Returns (names, run_action).  Raises on unusable config/table; the caller
    decides whether to degrade to a no-action session.
    """
    from scripts.action_registry import load_registry
    from scripts.result_feedback import execute_recipe
    from cup_grasp_demo.flow.core import ROOT, read_json, digest
    from cup_grasp_demo.flow.debug import new_run
    from cup_grasp_demo.flow.green_cup_planning import arm_plan

    registry = load_registry()
    for message in registry.errors:
        diagnostic.write(f"[actions] {message}\n")
    table = read_json(ROOT / flow.cfg["green_cup"]["home_table_scene"])
    if table.get("calibration_sha256") != digest(flow.cfg["calibration"]):
        raise ValueError("桌面记录与当前标定不一致，请更新桌面记录")

    def run_action(name):
        if name == "home":
            home = read_json(ROOT / flow.cfg["home"])
            # flow.cfg holds the raw JSON (strategy not merged); open_targets_0_100
            # lives in the strategy file since the strategy split and reaches the
            # validated view flow.g. Prefer g, fall back to raw config for mocks.
            green = getattr(flow, "g", None) or flow.cfg["green_cup"]
            mode = green.get("home_execution_mode", "arm_then_hand")
            if mode not in ("arm_then_hand", "together"):
                raise ValueError("home_execution_mode must be arm_then_hand or together")
            recipe = dict(gesture="home", joints_deg=list(home["joints_deg"]),
                          hand_0_100=list(green["open_targets_0_100"]),
                          speed_percent=green.get("fast_speed_percent", 30),
                          finger_duration_s=green.get("fast_finger_duration_s", 0.25),
                          execution=dict(mode=mode, delay_s=0.0),
                          finger_speed_mode="timed", finger_max_wait_s=0.65)
        else:
            recipe = registry.recipe(name)
        directory = new_run(Path(flow.root), "green_action_" + recipe["gesture"])
        started = time.perf_counter()
        execute_recipe(recipe, flow.cfg, table["scene"], directory,
                       flow._sdk, arm_plan)
        return dict(name=recipe["gesture"], receipt=str(directory / "receipt.json"),
                    elapsed_s=round(time.perf_counter() - started, 3))

    names = ["home", *registry.names()]
    return names, run_action


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
                try:
                    actions, run_action = build_action_runtime(flow, diagnostic)
                except Exception as exc:
                    diagnostic.write(f"[actions] 手势执行器不可用：{exc}\n")
                    actions, run_action = (), None
                server = ControlSession(flow, args.session / "green_pipeline_state.json",
                                        incoming, outgoing, diagnostic,
                                        actions=actions, run_action=run_action,
                                        reload_actions=lambda: build_action_runtime(flow, diagnostic))
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
