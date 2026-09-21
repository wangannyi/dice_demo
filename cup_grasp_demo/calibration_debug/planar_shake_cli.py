"""Plan/preview/run planar JS shaking, standalone or selected by the pipeline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cup_grasp_demo.calibration_debug import debug as common
from cup_grasp_demo.calibration_debug import planar_scene
from cup_grasp_demo.calibration_debug.core import Screen, digest, load_config, read_json, write_json
from cup_grasp_demo.calibration_debug.planar_shake import make_plan
from cup_grasp_demo.calibration_debug.phase_timing import PhaseTimer, emit_line, timing_line
from cup_grasp_demo.calibration_debug.parameters import effective_shake_options
from cup_grasp_demo.calibration_debug.session_storage import CAPTURE_PENDING, prepare_plan, session_lock

HERE = Path(__file__).resolve().parent
SOURCES = ("planar_shake.py", "planar_shake_cli.py", "planar_shake_execution.py",
           "shake.py", "shake_study.py", "shake_tracking.py", "shake_execution.py",
           "shake_readback.py", "parameters.py", "core.py", "joint_delivery.py", "phase_timing.py",
           "shake_cli.py", "pipeline_runner.py", "planar_scene.py")


def configured_trial(cfg, path):
    if path is not None:
        return read_json(path)
    selected = effective_shake_options(cfg)
    if selected.get("strategy") != "planar_js":
        raise ValueError("主配置未选择 shake.strategy=planar_js，请重新 shake-plan")
    return {key: value for key, value in selected.items() if key != "strategy"}


def summary(plan):
    print(json.dumps({k: plan.get(k) for k in ("backend", "parameters", "frequency_adaptation", "planning_timings_s", "total_stroke_mm",
        "peak_deviation", "joint_peak_acceleration_rad_s2", "table_screen", "planning_passed", "blockers")},
        indent=2, ensure_ascii=False))


def timed_command(operation, args, label):
    timer = PhaseTimer()
    try:
        return operation(args, timer)
    finally:
        if timer.records:
            total = timer.summary(label)
            if getattr(timer, "output", None) is not None:
                write_json(timer.output, dict(phases=timer.records, phase_total_s=total,
                                              excludes_confirmation_wait=True))


def plan_command(args, *, prepared=None):
    return timed_command(lambda args, timer: _plan_command(args, timer, prepared=prepared), args, "JS_PLAN")


def _plan_command(args, timer, *, prepared=None):
    planning_started = time.perf_counter()
    with timer.phase('INPUT_READBACK 配置、场景与关节读取'):
        table_scene = getattr(args, 'table_scene', None)
        cfg = load_config(args.config)
        offline = args.feedback_json is not None
        if table_scene is not None:
            session, frozen = planar_scene.verify(table_scene, args.config)
        else:
            if (args.session / CAPTURE_PENDING).exists() or not (args.session / 'session.json').exists():
                raise ValueError('抓杯场景未就绪；独立摇晃请先 table-capture，再 plan --table-scene "$RUN/planar_table_scene.json"')
            session = read_json(args.session / "session.json")
            if not offline:
                session, frozen = common.verify_session(args.session)
        if not offline and cfg["channel"] != frozen["channel"]:
            raise ValueError("CAN channel mismatch")
        run = common.new_run(args.session, "planar_js_plan")
        timer.output = run / "timings.json"
        feedback_path = args.feedback_json or run / "limits.json"
        if not offline:
            with (run / "readback.log").open("x") as log:
                subprocess.run([os.environ.get("DICE_SDK_PYTHON", "/home/test2/agilex-api-test/venv/bin/python"),
                                str(HERE / "shake_readback.py"), "--channel", cfg["channel"],
                                "--output", str(feedback_path)], cwd=ROOT, stdout=log, stderr=log, check=True)
        trial = configured_trial(cfg, args.trial_config)
        feedback = read_json(feedback_path)
    ik_started = time.perf_counter()
    with timer.phase('TRAJECTORY 轨迹计算'):
        result = make_plan(feedback, trial, session["T_flange_tcp"])
    geometry_started = time.perf_counter()
    with timer.phase('TABLE_CHECK 桌面检查'):
        table = Screen(table_only=True).check_table_batch([s["q_rad"] for s in result["samples"]], session["scene"], cfg)
    result["planning_timings_s"] = dict(
        inputs_and_readback_s=ik_started-planning_started,
        trajectory_s=geometry_started-ik_started,
        table_geometry_s=time.perf_counter()-geometry_started)
    result["table_screen"] = table
    result["blockers"].extend(table["blockers"])
    result["planning_passed"] = not result["blockers"]
    result["phase_timings"] = list(timer.records)
    with timer.phase('SAVE_PLAN 保存计划'):
        paths = [args.config, table_scene or args.session / "session.json", feedback_path]
        if args.trial_config is not None:
            paths.append(args.trial_config)
        paths += [HERE / name for name in SOURCES]
        paths += [ROOT / "rgb_hand_tracking" / name for name in
                  ("visual_servo_probe.py", "passive_pose_bridge.py", "finger_feedback_probe.py")]
        result.update(offline_only=offline, created_epoch_s=time.time(), session_path=str(args.session.resolve()),
                      config_path=str(args.config.resolve()),
                      table_scene_path=str(table_scene.resolve()) if table_scene is not None else None,
                      trial_config_path=str(args.trial_config.resolve()) if args.trial_config is not None else None,
                      configuration_source="trial_config" if args.trial_config is not None else "main_config.shake",
                      feedback_path=str(feedback_path.resolve()),
                      input_hashes={str(p.resolve()): digest(p) for p in paths})
        # main() invalidated the previous plan before any readback/IK operation.
        output = args.output or args.session / "planar_js_plan.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".pending.json")
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        temp.replace(output)
    if prepared is not None and result["planning_passed"]:
        prepared.update(plan_sha256=digest(output), table=table,
                        dependencies=dict(result["input_hashes"]))
    summary(result)
    print("计划：" + str(output))
    return 0 if result["planning_passed"] else 2


def run_command(args, *, return_receipt=False, prepared=None):
    return timed_command(lambda args, timer: _run_command(
        args, timer, return_receipt=return_receipt, prepared=prepared), args, "JS_RUN（含 SDK 子阶段）")


def _run_command(args, timer, *, return_receipt=False, prepared=None):
    preflight_started = time.perf_counter()
    with timer.phase('PLAN_CHECK 计划与文件校验'):
        plan = read_json(args.plan)
        summary(plan)
        if not args.execute:
            print("仅预览，未连接机械臂。")
            return 0
        if plan["offline_only"] or not plan["planning_passed"] or plan["blockers"]:
            raise ValueError("离线/未通过计划不可执行")
        if not 0 <= time.time()-plan["created_epoch_s"] <= 600:
            raise ValueError("计划过期，重新 plan")
        for path, expected in plan["input_hashes"].items():
            if digest(path) != expected:
                raise ValueError("计划依赖改变，重新 plan："+path)
    with timer.phase('REPLAN 场景校验与轨迹复算'):
        if plan.get('table_scene_path'):
            session, _ = planar_scene.verify(plan['table_scene_path'], plan['config_path'])
        else:
            session, _ = common.verify_session(Path(plan["session_path"]))
        cfg = load_config(Path(plan["config_path"]))
        trusted = (getattr(args, 'fast', False) and prepared
                   and prepared.get('plan_sha256') == digest(args.plan)
                   and prepared.get('dependencies') == plan['input_hashes'])
        if not trusted:
            trial_path = plan.get("trial_config_path")
            trial = configured_trial(cfg, Path(trial_path) if trial_path else None)
            recomputed = make_plan(read_json(plan["feedback_path"]), trial, session["T_flange_tcp"])
            for field in ("samples", "parameters", "joint_limits_rad", "joint_peak_velocity_rad_s",
                          "joint_peak_acceleration_rad_s2", "T_flange_virtual_tcp", "T_base_tcp_center",
                          "tcp_rpy_center_rad", "direction_base", "start_q_rad", "sample_hz"):
                if plan[field] != recomputed[field]:
                    raise ValueError("轨迹与重新计算结果不一致：" + field)
            if not recomputed["planning_passed"]:
                raise ValueError("重新计算的轨迹未通过")
    with timer.phase('TABLE_CHECK 执行前桌面检查'):
        table = (dict(prepared['table']) if trusted else Screen(table_only=True).check_table_batch(
            [s["q_rad"] for s in plan["samples"]], session["scene"], cfg))
        if table["blockers"]:
            raise ValueError("桌面检查未通过：" + str(table["blockers"]))
    preflight_s = time.perf_counter()-preflight_started
    print(f"执行前复核：{preflight_s:.3f} s（不含输入等待）")
    print("本次仅从当前姿态摇晃并回中心，不执行 HOME、抓杯、张手或揭杯。")
    print("状态：" + ("空手" if args.load == "empty" else "已抓牢杯子、杯口受桌面支撑"))
    print("确认相机/基座/桌面未移动、完整往返区域留空，且没有其他控制器。")
    print("普通跟随误差只记录；关节越界、平面失控、控制器故障及通信中断会停止。")
    print("已指定 --execute，检查通过后直接执行。")
    with timer.phase('PREPARE_REQUEST 保存执行请求'):
        run = common.new_run(Path(plan["session_path"]), "execute_planar_js")
        timer.output = run / "timings.json"
        request = dict(plan=plan, execution_authorized=True, authorized_epoch_s=time.time(),
                       scene_observed=True, load=args.load, channel=cfg["channel"],
                       preflight_elapsed_s=preflight_s,
                       table_screen=table, restore_speed_percent=cfg["speed_percent"],
                       input_hashes=dict(plan["input_hashes"], **{str(args.plan.resolve()): digest(args.plan)}))
        write_json(run / "request.json", request)
    with timer.phase('SDK_PROCESS 执行器总耗时（含下列子阶段）'):
        with (run / "actual.log").open("x") as log:
            child = subprocess.Popen([os.environ.get("DICE_SDK_PYTHON", "/home/test2/agilex-api-test/venv/bin/python"),
                str(HERE / "planar_shake_execution.py"), "--request", str(run / "request.json"),
                "--sha256", digest(run / "request.json"), "--output", str(run / "actual.json")],
                cwd=ROOT, stdout=log, stderr=log)
            try:
                child.wait(timeout=plan["parameters"]["duration_s"]+45)
            except BaseException:
                child.terminate()
                child.wait(timeout=12)
                raise
    print("记录：" + str(run / "actual.json"))
    if not (run / "actual.json").exists():
        raise RuntimeError("执行器无收据，查看 " + str(run / "actual.log"))
    report = read_json(run / "actual.json")
    report["actual_path"] = str(run / "actual.json")
    for row in report.get("phase_timings", []):
        emit_line(timing_line("SDK/" + row["phase"], row["duration_s"], row["status"]))
    print(json.dumps({k: report.get(k) for k in ("success", "error", "duration_completed", "motion_elapsed_s",
                                               "send_rate_hz", "measured_wave", "hold_error")}, indent=2, ensure_ascii=False))
    return report if return_receipt else (0 if report["success"] else 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="只读反馈并规划；不运动")
    p.add_argument("--config", type=Path, default=HERE / "index_joint_center/config.json")
    p.add_argument("--trial-config", type=Path, default=HERE / "planar_shake.json")
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--feedback-json", type=Path, help="离线反馈，仅计算，不可执行")
    p.add_argument("--output", type=Path)
    p.add_argument("--table-scene", type=Path, help="独立桌面数据，不依赖抓杯 capture")
    t = sub.add_parser("table-capture", help="仅采集桌面，无杯子识别或运动")
    t.add_argument("--config", type=Path, default=HERE / "index_joint_center/config.json")
    t.add_argument("--session", type=Path, required=True)
    r = sub.add_parser("run", help="预览或执行已规划实验")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--load", choices=("empty", "cup"), default="empty")
    r.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    directory = args.session if args.command in ("plan", "table-capture") else Path(read_json(args.plan)["session_path"])
    with session_lock(directory):
        if args.command == "table-capture":
            return planar_scene.capture(args)
        if args.command == "plan":
            output = args.output or args.session / "planar_js_plan.json"
            prepare_plan(output, "planar_js_shake_trial")
            return plan_command(args)
        return run_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
