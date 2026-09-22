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

MENU = """\
数字指令：
  1  status               查询状态（不动硬件）
  2  advance              执行下一阶段
  3  advance→GRIP         连续执行到闭手抓杯
  4  advance→SHAKE        连续执行到摇完停住（可能仍持杯）
  5  advance→RETURN_HOME  放杯+张手+归位
  6  refresh_perception   退回 CAPTURE 重新识别（仅 CAPTURE 后、APPROACH 前可用）
  7  new_cycle            下一轮（仅 RETURN_HOME 完成后可用）
  8  close                释放设备并退出
  9  仅退控制台（子进程收到 EOF 后释放设备，状态记 PAUSED）
  0  显示本菜单
也可直接输入一行 JSON 命令，如 {"command":"advance","until":"LIFT"}。
"""

CHOICES = {
    "1": {"command": "status"},
    "2": {"command": "advance"},
    "3": {"command": "advance", "until": "GRIP"},
    "4": {"command": "advance", "until": "SHAKE"},
    "5": {"command": "advance", "until": "RETURN_HOME"},
    "6": {"command": "refresh_perception"},
    "7": {"command": "new_cycle"},
    "8": {"command": "close"},
}

HINTS = {
    "ready": "常驻就绪：预热完成，可以下发指令",
    "phase_started": "阶段开始",
    "phase_completed": "阶段完成",
    "command_completed": "命令完成",
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
                       ("cycle", "轮次"), ("code", "code"), ("through", "推进到")):
        if event.get(key) is not None:
            parts.append(f"{label}={event[key]}")
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

    def __init__(self):
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

        self.tmp = tempfile.mkdtemp(prefix="dice-control-sim-")
        read_in, write_in = os.pipe()
        read_out, write_out = os.pipe()
        self._write_in = os.fdopen(write_in, "w", buffering=1)
        self._read_out = os.fdopen(read_out, "r", buffering=1)
        self.session = ControlSession(
            FakeFlow(), Path(self.tmp) / "green_pipeline_state.json",
            os.fdopen(read_in, "r", buffering=1),
            os.fdopen(write_out, "w", buffering=1), sys.stderr)
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

    if args.simulate:
        backend = SimulateBackend()
        print("演练模式（--simulate）：命令语义是真的，阶段动作是假的（每阶段约 0.4s）")
    else:
        backend = SubprocessBackend(args.config, args.session)
        print("真机模式：run.sh control --execute，等待 SDK/CAN 连接、相机预热、模型加载……")
    print(MENU)

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
                print(MENU)
                continue
            if raw in CHOICES:
                command = dict(CHOICES[raw])
            elif raw.startswith("{"):
                try:
                    command = json.loads(raw)
                except ValueError:
                    print("JSON 解析失败；请输入数字 0-9 或一行完整 JSON")
                    continue
            else:
                print("无效指令；数字 0-9 或一行 JSON（0 显示菜单）")
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
