#!/usr/bin/env python3
"""数字控制台：拉起常驻 control 进程，用数字菜单下发阶段指令。

在 K3 上、仓库根目录运行：

    python3 scripts/control_console.py              # 真机：run.sh control --execute
    python3 scripts/control_console.py --simulate   # 演练：无硬件，复用真实命令语义，动作是假的

可选参数：
    --config PATH    覆盖 DICE_CONFIG（默认 configs/green_cup.json）
    --session PATH   覆盖 DICE_RUN（默认 cup_grasp_demo/datasets/green_current）

也可以不选数字，直接输入一行 JSON 命令（如 {"command":"advance","until":"LIFT"}）。
阶段执行中下发的命令会在常驻进程内排队，按顺序处理。
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parent.parent

# 固定快捷键（常用手势与归位）；home 由常驻进程代码生成，恒可用。
FIXED_SHORTCUTS = {"y": "yeah", "t": "thumbs-up", "e": "tie", "h": "home"}
GESTURE_HINTS = {"yeah": "机械臂赢", "thumbs-up": "机械臂输", "tie": "平局",
                 "rock": "石头", "paper": "布", "scissors": "剪刀"}
# 流程命令与固定手势键保留，动态手势不得占用。
RESERVED_KEYS = set("0123456789aglryteh")


def gesture_shortcuts(gestures):
    """固定键优先，其余手势用可用首字母；冲突的动作仅列名（a <名字> 执行）。"""
    shortcuts = {}
    for key, name in FIXED_SHORTCUTS.items():
        if name == "home" or name in gestures:
            shortcuts[key] = name
    assigned = set(shortcuts.values())
    for name in gestures:
        if name in assigned:
            continue
        initial = name[0].lower()
        if (len(initial) == 1 and initial.isalnum() and not initial.isdigit()
                and initial not in RESERVED_KEYS and initial not in shortcuts):
            shortcuts[initial] = name
            assigned.add(name)
    return shortcuts


def build_menu(shortcuts, gestures):
    entries = []
    for key, name in shortcuts.items():
        hint = GESTURE_HINTS.get(name)
        if name == "home":
            entries.append(f"{key}  home 归位（张手+回 HOME，跟随主流程配置）")
        else:
            entries.append(f"{key}  {name}（{hint}）" if hint else f"{key}  {name}")
    rows = ["    ".join(entries[i:i+3]) for i in range(0, len(entries), 3)]
    others = [name for name in gestures if name not in shortcuts.values()]
    lines = [
        "静态手势（空闲时随意调度、随意互切；抓取流程进行中会被拒，跑完自动回空闲）：",
        *["  " + row for row in rows],
    ]
    if others:
        lines.append("  其他手势（无快捷键，用 a <名字>）：" + "、".join(others))
    lines += [
        "  l  列出全部可用动作名（含别名）",
        "  r  reload 重载手势（改完 configs/actions/gestures/ 文件后即时生效，不断连）",
        "",
        "抓取流程（视觉联动复合任务）：",
        "  g  完整流程（一口气到 RETURN_HOME，结束自动回空闲）",
        "  2  单阶段推进（调试）      3  连续执行到 GRIP（闭手抓杯）",
        "  4  连续执行到 SHAKE（摇完停住）    5  连续执行到 RETURN_HOME",
        "  6  refresh_perception   退回 CAPTURE 重新识别（仅 CAPTURE 后、APPROACH 前可用）",
        "  1  status               8  close 释放设备并退出",
        "  9  仅退控制台（子进程收到 EOF 后释放设备，状态记 PAUSED）",
        "  0  显示本菜单",
        "也可直接输入一行 JSON 命令，如 {\"command\":\"action\",\"name\":\"yeah\"}。",
    ]
    return "\n".join(lines)


BASE_CHOICES = {
    "1": {"command": "status"},
    "2": {"command": "advance"},
    "3": {"command": "advance", "until": "GRIP"},
    "4": {"command": "advance", "until": "SHAKE"},
    "5": {"command": "advance", "until": "RETURN_HOME"},
    "g": {"command": "advance", "until": "RETURN_HOME"},
    "6": {"command": "refresh_perception"},
    "l": {"command": "actions"},
    "r": {"command": "reload"},
    "8": {"command": "close"},
}

HINTS = {
    "ready": "常驻就绪：预热完成，空闲态可发手势或开始抓取流程",
    "phase_started": "阶段开始",
    "phase_completed": "阶段完成",
    "command_completed": "命令完成",
    "run_completed": "抓取流程完成：自动复位回空闲，可发手势或直接再来一轮",
    "action_started": "静态动作开始",
    "action_completed": "静态动作完成（保持姿态）",
    "actions": "可用动作名",
    "actions_reloaded": "手势表已重新加载（拒载明细见常驻进程 stderr 日志）",
    "status": "当前状态",
    "perception_reset": "已退回 CAPTURE：下次 advance 会重新识别与规划",
    "rejected": "命令被拒绝",
    "failed": "执行失败，常驻进程将退出",
    "closed": "设备已释放，进程退出",
    "preview": "预览模式（未带 --execute），进程即将退出",
}


def describe(event):
    parts = []
    for key, label in (("phase", "阶段"), ("next_phase", "下一阶段"), ("status", "状态"),
                       ("run", "运行序号"), ("cycle", "运行序号"), ("code", "code"),
                       ("through", "推进到"), ("name", "动作"), ("receipt", "收据"),
                       ("actions", "动作")):
        if event.get(key) is not None:
            value = event[key]
            parts.append(f"{label}=" + ("、".join(value) if isinstance(value, list) else str(value)))
    if event.get("names") is not None:
        parts.append("可选=" + "、".join(event["names"]))
    if event.get("elapsed_s") is not None:
        parts.append(f"耗时={event['elapsed_s']:.2f}s")
    for key in ("message", "error"):
        if event.get(key):
            parts.append(str(event[key]))
    line = f"[{event.get('event', '?')}] {HINTS.get(event.get('event'), '')}".rstrip()
    if parts:
        line += "（" + "；".join(parts) + "）"
    return line


class SubprocessBackend:
    """真机模式：run.sh control --execute 作为子进程。"""

    def __init__(self, config, session):
        env = dict(os.environ)
        if config:
            env["DICE_CONFIG"] = str(Path(config).resolve())
        if session:
            env["DICE_RUN"] = str(Path(session).resolve())
        self.proc = subprocess.Popen(
            ["bash", str(ROOT / "run.sh"), "control", "--execute"],
            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1)

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def close_stdin(self):
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def lines(self):
        yield from self.proc.stdout

    def alive(self):
        return self.proc.poll() is None

    def wait(self):
        return self.proc.wait()


class SimulateBackend:
    """演练模式：进程内跑真实 ControlSession，Workflow 换成假动作。"""

    def __init__(self, actions):
        sys.path.insert(0, str(ROOT))
        from cup_grasp_demo.flow.green_control import ControlSession

        class FakeFlow:
            receipts = {}
            held = None
            _snapshot_cache = None
            recovery_events = []

            def unchanged(self):
                pass

            def perform(self, phase):
                time.sleep(0.4)

        def fake_run_action(name):
            if name not in actions:
                raise ValueError("未知动作：" + name + "；可选：" + ", ".join(actions))
            time.sleep(0.3)
            return dict(name=name, receipt="/tmp/sim_receipt.json", elapsed_s=0.3)

        self.tmp = tempfile.mkdtemp(prefix="dice-control-sim-")
        read_in, write_in = os.pipe()
        read_out, write_out = os.pipe()
        self._write_in = os.fdopen(write_in, "w", buffering=1)
        self._read_out = os.fdopen(read_out, "r", buffering=1)
        self.session = ControlSession(
            FakeFlow(), Path(self.tmp) / "green_pipeline_state.json",
            os.fdopen(read_in, "r", buffering=1),
            os.fdopen(write_out, "w", buffering=1), sys.stderr,
            actions=tuple(actions), run_action=fake_run_action,
            reload_actions=lambda: (tuple(actions), fake_run_action))
        self.thread = threading.Thread(target=self.session.serve, daemon=True)
        self.thread.start()

    def send(self, line):
        self._write_in.write(line + "\n")
        self._write_in.flush()

    def close_stdin(self):
        try:
            self._write_in.close()
        except (BrokenPipeError, OSError):
            pass

    def lines(self):
        yield from self._read_out

    def alive(self):
        return self.thread.is_alive()

    def wait(self):
        self.thread.join(timeout=10)
        return 0

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="dice_demo 常驻模式数字控制台")
    parser.add_argument("--simulate", action="store_true",
                        help="无硬件演练：真实命令语义，动作是假的")
    parser.add_argument("--config", help="覆盖 DICE_CONFIG")
    parser.add_argument("--session", help="覆盖 DICE_RUN")
    args = parser.parse_args()

    # 手势名单来自 configs/actions/gestures/ 目录注册（与常驻进程同一来源）；
    # home 由常驻进程代码生成，恒可用。
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.action_registry import load_registry
    registry = load_registry()
    for message in registry.errors:
        print(f"[gestures] {message}", file=sys.stderr)
    gestures = registry.gesture_names()
    gestures = list(dict.fromkeys(["home", *gestures]))
    shortcuts = gesture_shortcuts(gestures)
    menu = build_menu(shortcuts, gestures)
    choices = dict(BASE_CHOICES)
    for key, name in shortcuts.items():
        choices[key] = {"command": "action", "name": name}
    shortcut_keys = "/".join(shortcuts)

    if args.simulate:
        backend = SimulateBackend(["home", *registry.names()])
        print("演练模式（--simulate）：命令语义是真的，阶段动作是假的（每阶段约 0.4s）")
    else:
        backend = SubprocessBackend(args.config, args.session)
        print("真机模式：run.sh control --execute，等待 SDK/CAN 连接、相机预热、模型加载……")
    print(menu)

    terminal_events = {"closed", "failed", "preview"}
    stop = threading.Event()

    def reader():
        try:
            for raw in backend.lines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except ValueError:
                    print(f"← {raw}")
                    continue
                print(describe(event))
                if event.get("event") in terminal_events:
                    stop.set()
        finally:
            stop.set()

    threading.Thread(target=reader, daemon=True).start()

    seq = 0
    try:
        while not stop.is_set() and backend.alive():
            try:
                raw = input("指令> ").strip()
            except EOFError:
                break
            if not raw:
                continue
            if raw == "0":
                print(menu)
                continue
            if raw in choices:
                command = dict(choices[raw])
            elif raw.startswith("a ") and raw[2:].strip():
                command = {"command": "action", "name": raw[2:].strip()}
            elif raw.startswith("{"):
                try:
                    command = json.loads(raw)
                except ValueError:
                    print("JSON 解析失败；请输入菜单指令或一行完整 JSON")
                    continue
            else:
                print(f"无效指令；输入 0 显示菜单（手势快捷键 {shortcut_keys}、"
                      "a <名字> 任意动作、g 完整抓取）")
                continue
            seq += 1
            command.setdefault("id", f"c{seq}")
            try:
                backend.send(json.dumps(command, ensure_ascii=False))
            except (BrokenPipeError, OSError):
                print("常驻进程已退出")
                break
            if raw == "8":
                break
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C：关闭 stdin，常驻进程将在当前阶段结束后释放设备")
    finally:
        backend.close_stdin()
        code = backend.wait()
        if isinstance(backend, SimulateBackend):
            backend.cleanup()
    print(f"控制台结束，常驻进程退出码 {code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
