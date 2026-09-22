"""Standalone joint laboratory: plan, explicit run, and read-only constraint comparison."""

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
from types import SimpleNamespace
from cup_grasp_demo.calibration_debug.core import (
    Screen,
    cached_screen_geometry,
    configured_tcp,
    digest,
    load_config,
    read_json,
    write_json,
)
from cup_grasp_demo.calibration_debug.joint_profile import KIND, make_plan, options, trajectory
from cup_grasp_demo.calibration_debug.session_storage import (
    session_lock,
    prepare_plan,
)
from cup_grasp_demo.calibration_debug.shake import Kinematics
from cup_grasp_demo.calibration_debug.shake_study import compare

HERE = Path(__file__).resolve().parent
SDK_PYTHON = "/usr/bin/python3"


def readback(args, cfg):
    if args.feedback_json:
        return args.feedback_json
    directory = common.new_run(args.session, "joint_readback")
    output = directory / "controller_limits.json"
    with (directory / "readback.log").open("x") as log:
        result = subprocess.run(
            [
                os.environ.get("DICE_SDK_PYTHON", SDK_PYTHON),
                str(HERE / "shake_readback.py"),
                "--channel",
                cfg["channel"],
                "--output",
                str(output),
            ],
            stdout=log,
            stderr=log,
            cwd=ROOT,
            timeout=30,
        )
    if result.returncode:
        detail = read_json(output).get("error", "") if output.exists() else "采集进程未生成结果"
        raise RuntimeError("只读采集失败：" + detail + "；日志：" + str(directory / "readback.log"))
    return output


def implementation_hashes():
    return {str(p): digest(p) for p in sorted(HERE.glob("*.py"))}


def summarize(plan):
    keys = (
        "parameters",
        "planning_timings_s",
        "duration_s",
        "reference_frequency_hz",
        "triangular_profile",
        "joint_peak_velocity_rad_s",
        "joint_peak_acceleration_rad_s2",
        "table_screen",
        "blockers",
        "warnings",
    )
    print(
        json.dumps(
            {k: plan[k] for k in keys if k in plan}, ensure_ascii=False, indent=2
        )
    )
    print(f"规划通过：{plan['planning_passed']}；单侧角幅；不包含抓杯动作；未发送运动")


def design(args):
    timings, last = {}, time.perf_counter()
    def mark(name):
        nonlocal last
        now = time.perf_counter()
        timings[name], last = now-last, now
    output = args.output or args.session / "joint_plan.json"
    prepare_plan(output, KIND)
    cfg = load_config(args.system_config)
    raw = options(read_json(args.config))
    trajectory(raw)  # Reject invalid timing before camera capture or CAN reads.
    mark("config_s")
    scene_run = common.new_run(args.session, "joint_table")
    table_scene = getattr(args, "table_scene", None)
    if table_scene is not None:
        saved, _ = planar_scene.verify(table_scene, args.system_config)
        scene = saved["scene"]
        quality = saved["calibration_quality_passed"]
    elif args.feedback_json:
        scene = read_json(args.session / "session.json")["scene"]
        quality = None
    else:
        # A motor test needs a fresh table plane, not a detected cup or HOME.
        from cup_grasp_demo.calibration_debug.pipeline_home import home_scene

        common.capture_rgbd(scene_run / "rgbd", cfg)
        scene, quality = home_scene(scene_run, cfg)
    scene_file = scene_run / "scene.json"
    write_json(
        scene_file,
        dict(
            scene=scene,
            calibration_quality_passed=quality,
            offline_only=bool(args.feedback_json),
        ),
    )
    mark("capture_and_table_s")
    feedback = readback(args, cfg)
    mark("readback_s")
    kin = Kinematics()
    plan = make_plan(read_json(feedback), raw, kin.model.limits_rad)
    mark("joint_trajectory_s")
    screen = Screen(table_only=True).check_table_batch(
        [s["q_rad"] for s in plan["samples"]], scene, cfg
    )
    mark("table_geometry_s")
    plan["blockers"] += screen["blockers"]
    plan["planning_passed"] = not plan["blockers"]
    plan["table_screen"] = screen
    plan["model_flange_checks"] = []
    for factor in (0, 1, -1):
        q = [a + factor * b for a, b in zip(plan["start_q_rad"], plan["amplitude_rad"])]
        plan["model_flange_checks"].append(
            dict(q_rad=q, T_base_flange=kin.forward(q)[0].tolist())
        )
    hashes = implementation_hashes()
    hashes.update(
        {
            str(p): digest(p)
            for p in [
                args.config,
                args.system_config,
                feedback,
                scene_file,
            ]
        }
    )
    hashes.update({str(ROOT / p): v for p, v in common.source_hashes(cfg).items()})
    hashes.update(
        {str(p): digest(p) for p in (scene_run / "rgbd").glob("*") if p.is_file()}
    )
    if table_scene is not None:
        hashes[str(table_scene)] = digest(table_scene)
        hashes.update(saved["input_hashes"])
    plan.update(
        table_scene_path=str(table_scene) if table_scene is not None else None,
        session_path=str(args.session),
        system_config=str(args.system_config),
        input_hashes=hashes,
        feedback_path=str(feedback),
        scene_path=str(scene_file),
        created_epoch_s=time.time(),
        offline_only=bool(args.feedback_json),
    )
    mark("fingerprints_and_finalize_s")
    plan["planning_timings_s"] = timings
    write_json(output, plan)
    summarize(plan)
    print("计划：" + str(output))
    return 0 if plan["planning_passed"] else 2


def study(args):
    output = args.output or args.session / "shake_study.json"
    prepare_plan(output, "shake_constraint_study")
    cfg = load_config(args.system_config)
    feedback = readback(args, cfg)
    result = compare(read_json(feedback), cfg, configured_tcp(cfg))
    result.update(
        feedback_path=str(feedback),
        feedback_sha256=digest(feedback),
        offline_only=bool(args.feedback_json),
        created_epoch_s=time.time(),
    )
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("只读比较：" + str(output))
    return 0


def execute(args):
    preflight_started = time.perf_counter()
    plan = read_json(args.plan)
    summarize(plan)
    if not args.execute:
        return 0 if plan["planning_passed"] else 2
    if (
        plan.get("kind") != KIND
        or not plan["planning_passed"]
        or plan["blockers"]
        or plan["offline_only"]
    ):
        raise ValueError("计划未通过或来自离线反馈，不能执行")
    for name, expected in plan["input_hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError("输入已修改，请重新 plan：" + name)
    cfg = load_config(Path(plan["system_config"]))
    if not 0 <= time.time() - plan["created_epoch_s"] <= cfg["plan_max_age_s"]:
        raise ValueError("计划过期，请重新 plan")
    if plan.get("table_scene_path"):
        planar_scene.verify(plan["table_scene_path"], plan["system_config"])
    directory = Path(plan["session_path"])
    scene = read_json(Path(plan["scene_path"]))["scene"]
    computed = make_plan(
        read_json(Path(plan["feedback_path"])),
        plan["parameters"],
        Kinematics().model.limits_rad,
    )
    for key in (
        "parameters",
        "samples",
        "segments",
        "start_q_rad",
        "amplitude_rad",
        "duration_s",
    ):
        if computed[key] != plan[key]:
            raise ValueError("计划与重新计算不一致：" + key)
    if not computed["planning_passed"]:
        raise ValueError("轨迹重新检查未通过")
    screen = Screen(table_only=True).check_table_batch(
        [s["q_rad"] for s in plan["samples"]], scene, cfg
    )
    if screen["blockers"]:
        raise ValueError("桌面检查未通过：" + str(screen["blockers"]))
    preflight_s = time.perf_counter() - preflight_started
    print(f"执行前复核：{preflight_s:.3f} s（不含输入等待）")
    print("本实验从当前关节位置往返并回中心；不执行 HOME、闭手或持杯摇晃。")
    print(
        "确认空手、整个往返区域无障碍、桌面/基座未移动，且没有其他控制程序。Ctrl-C 尝试原位保持。"
    )
    print("已指定 --execute，检查通过后直接执行。")
    run = common.new_run(directory, "execute_joint_test")
    hashes = dict(plan["input_hashes"])
    hashes[str(args.plan)] = digest(args.plan)
    request = dict(
        execution_authorized=True,
        authorized_epoch_s=time.time(),
        empty_hand_clear_path_confirmed=True,
        preflight_elapsed_s=preflight_s,
        plan=plan,
        table_screen=screen,
        channel=cfg["channel"],
        restore_speed_percent=cfg["speed_percent"],
        input_hashes=hashes,
    )
    request_file = run / "request.json"
    actual = run / "actual.json"
    write_json(request_file, request)
    with (run / "actual.log").open("x") as log:
        child = subprocess.Popen(
            [
                os.environ.get("DICE_SDK_PYTHON", SDK_PYTHON),
                str(HERE / "joint_execution.py"),
                "--request",
                str(request_file),
                "--sha256",
                digest(request_file),
                "--output",
                str(actual),
            ],
            stdout=log,
            stderr=log,
            cwd=ROOT,
        )
        try:
            child.wait(timeout=plan["duration_s"] + 40)
        except BaseException:
            child.terminate()
            child.wait(timeout=10)
            raise
    if not actual.exists():
        raise RuntimeError("执行器无收据：" + str(run / "actual.log"))
    report = read_json(actual)
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in (
                    "success", "error", "duration_completed", "tracking_summary", "measurement"
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("执行记录：" + str(actual))
    return 0 if report["success"] else 2


def configure_acceleration(args):
    cfg = load_config(args.system_config)
    raw = options(read_json(args.config))
    if raw['controller_acceleration_rad_s2'] is None:
        raise ValueError('请配置 controller_acceleration_rad_s2，七个值依次对应 J1..J7')
    directory = common.new_run(args.session, 'joint_acc_limits')
    command = [os.environ.get('DICE_SDK_PYTHON', SDK_PYTHON),
               str(HERE / 'joint_acc_limits.py'), '--config', str(args.config),
               '--sha256', digest(args.config), '--channel', cfg['channel'],
               '--output', str(directory)]
    if args.execute:
        command.append('--execute')
    result = subprocess.run(command, cwd=ROOT)
    print('参数记录：' + str(directory / 'actual.json'))
    print('原值备份：' + str(directory / 'before.json'))
    if result.returncode == 0 and args.execute:
        print('参数设置完成；请重新 plan 后再 run，旧计划不可复用。')
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--session", type=Path, required=True)
        p.add_argument(
            "--system-config",
            type=Path,
            default=HERE / "index_joint_center/config.json",
        )
        p.add_argument(
            "--feedback-json", type=Path, help="离线反馈：不可执行，只生成结果"
        )
        p.add_argument("--output", type=Path)
        if name == "plan":
            p.add_argument("--execute", action="store_true", help="规划通过后在同一进程直接执行")
            p.add_argument("--table-scene", type=Path, help="复用独立桌面数据，跳过相机采集")
            p.add_argument(
                "--config", type=Path, default=HERE / "joint_test_config.json"
            )
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser('acc-limits', help='读取或显式设置七轴控制器加速度上限，不运动')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--config', type=Path, default=HERE / 'joint_test_config.json')
    p.add_argument('--system-config', type=Path, default=HERE / 'index_joint_center/config.json')
    p.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    for k, v in vars(args).items():
        if isinstance(v, Path):
            setattr(args, k, v.resolve())
    try:
        if args.command == "run" and not args.plan.is_file():
            raise ValueError(
                "计划文件不存在；请先修正配置并重新 plan，规划成功后再 run。"
                "建议使用 plan ... && run ...，避免规划失败后继续执行。"
            )
        directory = (
            args.session
            if args.command != "run"
            else Path(read_json(args.plan)["session_path"])
        )
        with session_lock(directory):
            if args.command == "plan" and args.execute:
                with cached_screen_geometry():
                    result = design(args)
                    if result != 0:
                        return result
                    return execute(SimpleNamespace(
                        plan=args.output or args.session / "joint_plan.json", execute=True))
            return {"plan": design, "compare": study, "run": execute,
                    "acc-limits": configure_acceleration}[args.command](
                args
            )
    except (ValueError, RuntimeError, TimeoutError, OSError, KeyError) as error:
        print("JOINT TEST STOP: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
