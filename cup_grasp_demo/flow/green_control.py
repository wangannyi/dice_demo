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
import queue
import sys
import threading
import time

from cup_grasp_demo.flow.core import cached_screen_geometry
from cup_grasp_demo.flow.green_pipeline import PHASES, Workflow
from cup_grasp_demo.flow.core import save
from cup_grasp_demo.flow.session_storage import session_lock


_EMIT_LOCK = threading.Lock()


def emit(stream, event, **fields):
    line = json.dumps(dict(event=event, **fields), ensure_ascii=False,
                      allow_nan=False) + "\n"
    # The reader thread answers probes while serve() executes commands;
    # the lock keeps the two writers from interleaving a single line.
    with _EMIT_LOCK:
        stream.write(line)
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
        self.queue = queue.Queue()
        self.executor_lock = threading.Lock()
        save(self.state_path, self.state)

    def _new_state(self):
        return dict(strategy="green_open_cup", mode="control", cycle=self.cycle,
                    status="WAITING", events=[], phase_timings_s={}, actions=[],
                    last_failure=None)

    def _next_phase(self):
        return PHASES[self.next_index] if self.next_index < len(PHASES) else None

    def _claim_request_id(self, request):
        """Validate and mark the command id seen; returns (id, code, message).

        code is None when the id is usable (and now recorded as seen); the
        reject then carries the offending id.  Owned by whichever thread
        dispatches the line, so ids stay unique across both dispatch paths.
        """
        request_id = request.get("id")
        if (request_id is not None and
                (isinstance(request_id, bool) or not isinstance(request_id, (str, int))
                 or len(str(request_id)) > 64)):
            return None, "invalid_id", "id 必须是长度不超过 64 的字符串或整数"
        if request_id is not None:
            if request_id in self.seen_ids:
                return request_id, "duplicate_id", "此命令 id 已处理，不能重复派发"
            self.seen_ids.add(request_id)
        return request_id, None, None

    def _recovery_enabled(self):
        """green_cup.failure_recovery（默认开）：硬失败自动归位而非退出。"""
        green = getattr(self.flow, "g", None)
        value = green.get("failure_recovery", True) if isinstance(green, dict) else True
        return bool(value)

    def _recover(self, request_id, error_text):
        """After a hard failure: home the arm and keep serving.

        The resident process must outlive motion failures.  Recovery runs on
        the serve thread behind the executor lock; probes keep answering
        (status instantly, query_pose as command_busy) and new motion commands
        queue behind recovery.  A failing recovery itself falls back to the
        legacy exit — an arm whose homing fails is not safe to keep commanding.
        """
        if not self._recovery_enabled() or self.run_action is None:
            return False
        try:
            if "SDK worker exited" in str(error_text):
                sdk = getattr(self.flow, "_sdk", None)
                if sdk is not None:
                    sdk.restart()
            self._reply("recovery_started", request_id, action="home",
                        cause=str(error_text)[:300])
            result = self.run_action("home")
            # Same reset as a finished run: the failed round is void, held-cup
            # state and caches must not leak into the next command.
            self._reset_for_next_run()
            self.state["last_failure"] = str(error_text)[:300]
            self.state["actions"].append(dict(
                name="home", elapsed_s=result.get("elapsed_s"),
                receipt=result.get("receipt"), note="recovery"))
            save(self.state_path, self.state)
            self._reply("recovered", request_id, action="home",
                        receipt=result.get("receipt"))
            return True
        except BaseException as error:
            self.state.update(status="FAILED",
                              error=f"恢复失败（home）: {type(error).__name__}: {error}",
                              last_failure=str(error_text)[:300])
            save(self.state_path, self.state)
            self._reply("failed", request_id, phase="recovery:home",
                        error=self.state["error"], state_file=str(self.state_path))
            return False

    def _query_pose(self, request_id):
        """Read-only pose probe for the patrol homing on the game side.

        One ``snapshot`` over the persistent SDK worker, then compare against
        ``home`` (the HOME-phase posture source).  Any failure — CAN down,
        worker dead, malformed receipt — is a ``rejected``, never a ``failed``:
        probing must not exit the session.
        """
        import math
        from cup_grasp_demo.flow.core import ROOT, read_json
        from cup_grasp_demo.flow.debug import new_run
        try:
            directory = new_run(Path(self.flow.root), "green_pose")
            report = self.flow._sdk.call("snapshot", directory / "actual.json")
            joints = report.get("joints_rad") if isinstance(report, dict) else None
            if not isinstance(report, dict) or report.get("success") is not True \
                    or not isinstance(joints, list) or len(joints) != 7:
                raise RuntimeError(
                    str(report.get("error", "snapshot 回执缺少 joints_rad"))[:200]
                    if isinstance(report, dict) else "snapshot 无回执")
            home = read_json(ROOT / self.flow.cfg["home"])
            green = getattr(self.flow, "g", None)
            tolerance = float(green["home_pose_tolerance_deg"]) if isinstance(green, dict) \
                and "home_pose_tolerance_deg" in green else 5.0
            deltas = [math.degrees(j - math.radians(h))
                      for j, h in zip(joints, home["joints_deg"])]
            at_home = max(abs(d) for d in deltas) <= tolerance
            self._reply("pose", request_id, joints_rad=[round(j, 6) for j in joints],
                        delta_deg=[round(d, 3) for d in deltas], at_home=at_home)
        except Exception as exc:
            self._reject(request_id, "pose_unavailable", f"{type(exc).__name__}: {exc}")
        return True

    def _reply(self, event, request_id=None, **fields):
        emit(self.outgoing, event, id=request_id, cycle=self.cycle,
             status=self.state["status"], next_phase=self._next_phase(), **fields)

    def _reject(self, request_id, code, message):
        self._reply("rejected", request_id, code=code, message=message)

    def _advance(self, request_id, target_index):
        if self.next_index > target_index:
            self._reject(request_id, "already_completed", "目标阶段已经完成，不能重复执行")
            return True
        if not self._advance_single(request_id, target_index):
            # Phase failure: recover by homing instead of exiting the session.
            return self._recover(request_id, self.state.get("error", ""))
        self._reply("command_completed", request_id, through=PHASES[target_index])
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
            error = f"{type(exc).__name__}: {exc}"
            self.state.update(status="FAILED", error=error, last_failure=error)
            save(self.state_path, self.state)
            self._reply("failed", request_id, phase=f"action:{name}",
                        error=error, state_file=str(self.state_path))
            # 硬失败不再必然退出：先尝试归位恢复，恢复失败才退出。
            return self._recover(request_id, error)
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

    def _handle(self, request, _id_validated=False):
        if not isinstance(request, dict):
            self._reject(None, "invalid_request", "每行必须是 JSON 对象")
            return True
        if _id_validated:
            # The reader thread already claimed the id before queueing.
            request_id = request.get("id")
        else:
            request_id, code, message = self._claim_request_id(request)
            if code is not None:
                self._reject(request_id, code, message)
                return True
        command = request.get("command")
        if command == "status":
            self._reply("status", request_id, completed_phases=[
                item["phase"] for item in self.state["events"]],
                state_file=str(self.state_path))
            return True
        if command == "query_pose":
            return self._query_pose(request_id)
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
            self._reject(request_id, "removed",
                         "连跑已移除：单轮执行无中断需求，等待 command_completed 或 close 即可")
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
            if "rounds" in request:
                self._reject(request_id, "removed",
                             "连跑已移除：一次 advance 一轮，循环由上层按 command_completed 驱动")
                return True
            return self._advance(request_id, target_index)
        self._reject(request_id, "invalid_command",
                     "command 必须是 status、advance、refresh_perception、action、actions、reload 或 close")
        return True

    def _reader_loop(self):
        """Single stdin reader: answer read-only probes at once, queue the rest.

        serve() blocks inside a motion command for seconds; status/actions/
        query_pose must not wait behind it.  Read-only answers read the state
        dict directly (single-field access and list copy are atomic under the
        GIL), and query_pose additionally try-acquires the executor lock: the
        SDK worker channel is single-connection RPC, so a probe must never
        interleave with an executing command — busy means answer now with
        command_busy instead of queueing behind the motion.
        """
        try:
            while True:
                line = self.incoming.readline()
                if not line:
                    self.queue.put(None)   # EOF sentinel: serve exits as PAUSED.
                    return
                if len(line) > 4096:
                    self._reject(None, "invalid_request", "单条命令超过 4096 字符")
                    continue
                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    self._reject(None, "invalid_json", "每行必须是完整 JSON")
                    continue
                if not isinstance(request, dict):
                    self._reject(None, "invalid_request", "每行必须是 JSON 对象")
                    continue
                request_id, code, message = self._claim_request_id(request)
                if code is not None:
                    self._reject(request_id, code, message)
                    continue
                command = request.get("command")
                if command == "status":
                    self._reply("status", request_id, completed_phases=[
                        item["phase"] for item in list(self.state["events"])],
                        state_file=str(self.state_path))
                elif command == "actions":
                    self._reply("actions", request_id, names=sorted(set(self.actions)))
                elif command == "query_pose":
                    if self.executor_lock.acquire(blocking=False):
                        try:
                            self._query_pose(request_id)
                        finally:
                            self.executor_lock.release()
                    else:
                        self._reject(request_id, "command_busy",
                                     "抓取流程或动作执行中，稍后再试 query_pose")
                else:
                    self.queue.put(request)
        except BaseException as error:
            try:
                self.diagnostic.write(f"[reader] {type(error).__name__}: {error}\n")
            except Exception:
                pass
            # Unblock serve(): without a sentinel it would wait forever.
            self.queue.put(None)

    def serve(self):
        self._reply("ready", phases=list(PHASES), actions=sorted(set(self.actions)),
                    state_file=str(self.state_path))
        reader = threading.Thread(target=self._reader_loop, daemon=True,
                                  name="control-reader")
        reader.start()
        while True:
            request = self.queue.get()
            if request is None:
                self.state["status"] = "PAUSED"
                save(self.state_path, self.state)
                self._reply("closed", reason="stdin_eof", state_file=str(self.state_path))
                return 0
            # One executor at a time: motion commands and the query_pose probe
            # share the single SDK worker channel; the reader only try-acquires.
            with self.executor_lock:
                if not self._handle(request, _id_validated=True):
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
        # home 不再有内建 recipe：统一走注册表（configs/actions/gestures/
        # result_feedback.json 的 home 手势），用户的调参（speed_percent 等）
        # 因此真正生效——旧内建分支硬编码 fast_speed_percent=30/timed，
        # 把 2026-09-24 的 home 100% 调机静默遮蔽了。
        recipe = registry.recipe(name)
        directory = new_run(Path(flow.root), "green_action_" + recipe["gesture"])
        started = time.perf_counter()
        execute_recipe(recipe, flow.cfg, table["scene"], directory,
                       flow._sdk, arm_plan)
        return dict(name=recipe["gesture"], receipt=str(directory / "receipt.json"),
                    elapsed_s=round(time.perf_counter() - started, 3))

    names = registry.names()
    if "home" not in names:
        diagnostic.write("[actions] 手势库缺少 home——归位（reset_home）将不可用\n")
    else:
        # home.json 仍是阶段机 HOME 阶段的姿态来源；两处维护同一组关节角，
        # 分叉时只警告不拒绝（分叉 = action home 与 HOME 阶段去到不同姿态）。
        home = read_json(ROOT / flow.cfg["home"])
        if list(home["joints_deg"]) != list(registry.recipe("home")["joints_deg"]):
            diagnostic.write(
                "[actions] home.json 与手势库 home 的 joints_deg 不一致，"
                "归位动作将使用手势库的值\n"
            )
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
