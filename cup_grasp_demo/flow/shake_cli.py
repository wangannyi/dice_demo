"""Shake design and explicitly authorized standalone trial commands."""

import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from cup_grasp_demo.flow import debug as common
from cup_grasp_demo.flow.core import (
    digest,
    load_config,
    read_json,
    write_json,
)
from cup_grasp_demo.flow.parameters import effective_shake_options, shake_options
from cup_grasp_demo.flow.session_storage import (
    CAPTURE_PENDING,
    prepare_plan,
)
from cup_grasp_demo.flow.shake import KIND, make_plan


def summarize(result):
    keys = (
        "parameters",
        "total_stroke_mm",
        "nominal_cycles",
        "ramp_s_each_end",
        "direction_base",
        "cartesian_peak_velocity_m_s",
        "cartesian_peak_acceleration_m_s2",
        "joint_peak_acceleration_rad_s2",
        "uniform_retiming_reference",
        "blockers",
    )
    print(
        json.dumps(
            {key: result[key] for key in keys if key in result},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        f"规划通过：{result['planning_passed']}；execution_enabled=False；未发送运动或手指指令"
    )


def plan(args, *, prepared=None):
    output = args.output or args.session / "shake_plan.json"
    # Both strategies own this generated output; switching strategy invalidates
    # the previous plan before any new capture, readback or planning can fail.
    previous = read_json(output) if output.exists() else {'kind': KIND}
    prior_kind = previous.get('kind') if isinstance(previous, dict) else None
    if prior_kind not in (KIND, 'planar_js_shake_trial'):
        raise ValueError(f'输出文件不是摇晃计划，不能覆盖：{output}')
    prepare_plan(output, prior_kind)
    cfg = load_config(args.config)
    effective = effective_shake_options(cfg)
    if effective.get('strategy') == 'planar_js':
        from cup_grasp_demo.flow import planar_shake_cli
        return planar_shake_cli.plan_command(SimpleNamespace(
            config=args.config, trial_config=None, session=args.session, output=output,
            feedback_json=args.feedback_json), prepared=prepared)
    shake_options(cfg)
    if (args.session / CAPTURE_PENDING).exists():
        raise ValueError("本 RUN 最新采集尚未成功，请先完成 capture")
    offline = bool(args.feedback_json)
    if offline:
        session = read_json(args.session / "session.json")
        feedback_path = args.feedback_json
    else:
        session, frozen = common.verify_session(args.session)
        if cfg["channel"] != frozen["channel"]:
            raise ValueError("CAN channel differs from captured session")
        run = common.new_run(args.session, "shake_readback")
        feedback_path = run / "controller_limits.json"
        python = os.environ.get(
            "DICE_SDK_PYTHON", "/usr/bin/python3"
        )
        command = [
            python,
            str(common.HERE / "shake_readback.py"),
            "--channel",
            cfg["channel"],
            "--output",
            str(feedback_path),
        ]
        with (run / "readback.log").open("x") as stream:
            completed = subprocess.run(
                command, stdout=stream, stderr=stream, cwd=common.ROOT
            )
        if completed.returncode:
            raise RuntimeError(f"摇晃只读反馈采集失败，查看 {run / 'readback.log'}")
    feedback = read_json(feedback_path)
    result = make_plan(session, feedback, cfg)
    result.update(
        session_path=str(args.session),
        session_sha256=digest(args.session / "session.json"),
        feedback_path=str(feedback_path),
        feedback_sha256=digest(feedback_path),
        config_path=str(args.config),
        config_sha256=digest(args.config),
        created_epoch_s=time.time(),
        offline_only=offline,
        implementation_hashes={
            name: digest(common.HERE / name)
            for name in (
                "shake.py",
                "shake_cli.py",
                "shake_readback.py",
                "parameters.py",
            )
        },
    )
    write_json(output, result)
    if prepared is not None and result['planning_passed']:
        from cup_grasp_demo.flow.core import Screen
        table = Screen(table_only=True).check_table_batch([s['q_rad'] for s in result['samples']], session['scene'], cfg)
        if table['blockers']:
            raise ValueError('桌面模型检查失败：' + str(table['blockers']))
        prepared.update(plan_sha256=digest(output), table=table,
                        dependencies={str(common.HERE / name): digest(common.HERE / name)
                                      for name in ('core.py', 'shake_cli.py', 'shake.py', 'parameters.py')})
    summarize(result)
    print(f"计划：{output}")
    return 2 if result["blockers"] else 0


def execute(args, *, confirm=None, return_receipt=False, prepared=None):
    """Execute one bounded trial; optionally return the receipt to the pipeline."""
    import sys
    from cup_grasp_demo.flow.core import Screen

    source = read_json(args.plan)
    if source.get('kind') == 'planar_js_shake_trial':
        from cup_grasp_demo.flow import planar_shake_cli
        return planar_shake_cli.run_command(SimpleNamespace(
            plan=args.plan, execute=args.execute, load='cup',
            fast=getattr(args, 'fast', False)), return_receipt=return_receipt, prepared=prepared)
    summarize(source)
    if not args.execute:
        print("本次只查看计划；已夹住且杯口贴桌后，加 --execute 执行一次独立摇晃。")
        return 0
    if (
        source.get("offline_only")
        or not source["planning_passed"]
        or source["blockers"]
    ):
        raise ValueError("不能执行离线或未通过的摇晃计划")
    if not 0 <= time.time() - source["created_epoch_s"] <= 600:
        raise ValueError("摇晃计划已过期，请重新 shake-plan")
    directory = Path(source["session_path"])
    session, _ = common.verify_session(directory)
    cfg = load_config(Path(source["config_path"]))
    hashes = {
        str(args.plan): digest(args.plan),
        source["config_path"]: source["config_sha256"],
        str(directory / "session.json"): source["session_sha256"],
        source["feedback_path"]: source["feedback_sha256"],
    }
    for name, expected in source["implementation_hashes"].items():
        hashes[str(common.HERE / name)] = expected
    for name in (
        "shake_execution.py",
        "shake_tracking.py",
        "shake_camera.py",
        "core.py",
    ):
        hashes[str(common.HERE / name)] = digest(common.HERE / name)
    for name in (
        "visual_servo_probe.py",
        "passive_pose_bridge.py",
        "finger_feedback_probe.py",
    ):
        hashes[str(common.ROOT / "nero_revo2_control/bridges" / name)] = digest(
            common.ROOT / "nero_revo2_control/bridges" / name
        )
    for name, expected in session["source_hashes"].items():
        hashes[str(common.ROOT / name)] = expected
    for name, expected in hashes.items():
        if digest(name) != expected:
            raise ValueError("计划依赖已变更，请重新 shake-plan：" + name)
    trusted = (getattr(args, 'fast', False) and prepared
               and prepared.get('plan_sha256') == digest(args.plan)
               and all(digest(path) == value for path, value in prepared['dependencies'].items()))
    if not trusted:
        recomputed = make_plan(session, read_json(source["feedback_path"]), cfg)
        for name in ('parameters', 'samples', 'start_q_rad', 'direction_base', 'T_base_flange_center'):
            if source[name] != recomputed[name]:
                raise ValueError('保存的轨迹与当前参数重新计算结果不一致，请重新 shake-plan')
        if not recomputed['planning_passed']:
            raise ValueError('重新计算摇晃轨迹未通过')
    plan = dict(source, table_normal_base=session["scene"]["cup_normal_base"])
    run = common.new_run(directory, "execute_shake")
    table = (dict(prepared['table']) if trusted else Screen(table_only=True).check(
        [s['q_rad'] for s in plan['samples']], session['scene'], True, cfg))
    table["scope"] = (
        "sampled arm/adapter/open-hand model vs table; actual closed hand and table edges require scene observation"
    )
    write_json(run / "table_screen.json", table)
    if table["blockers"]:
        raise ValueError("桌面模型检查失败：" + str(table["blockers"]))
    if not getattr(args, 'fast', False):
        common.capture_rgbd(run / "before_rgbd", cfg)
        common.show(run / "before_rgbd/frame_000.png", args.show)
    print("本次动作：保持闭手与当前朝向，杯口贴桌往返，结束停在摇晃中心。")
    print("确认已夹住杯子、杯口落在桌面、整个往返区域无障碍，且无人操作其他控制器。")
    if confirm is not None and confirm("执行一次摇晃：").strip() != "SHAKE":
        print("已取消，未运动。")
        return 0
    video = run / "video"
    video_log = (run / "video.log").open("x")
    recorder = None
    actual = run / "actual.json"
    try:
        record_video = not getattr(args, 'fast', False) or cfg.get('fast_record_shake_video', False)
        if record_video:
            recorder = subprocess.Popen(
                [sys.executable, str(common.HERE / 'shake_camera.py'),
                 '--serial', cfg['serial'], '--output', str(video)],
                stdout=video_log, stderr=video_log, cwd=common.ROOT,
            )
            deadline = time.monotonic() + 15
            while not (video / 'ready').exists():
                if recorder.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('相机录像启动失败，未执行摇晃；查看 video.log')
                time.sleep(.05)
        request = dict(
            execution_authorized=True,
            authorized_epoch_s=time.time(),
            scene_observed=True,
            camera_process=dict(pid=recorder.pid, argv=recorder.args) if recorder else None,
            plan=plan,
            channel=cfg["channel"],
            table_screen=table,
            input_hashes=hashes,
            speed_percent=60,
            restore_speed_percent=cfg["speed_percent"],
        )
        write_json(run / "request.json", request)
        python = os.environ.get(
            "DICE_SDK_PYTHON", "/usr/bin/python3"
        )
        with (run / "actual.log").open("x") as log:
            child = subprocess.Popen(
                [
                    python,
                    str(common.HERE / "shake_execution.py"),
                    "--request",
                    str(run / "request.json"),
                    "--sha256",
                    digest(run / "request.json"),
                    "--output",
                    str(actual),
                ],
                stdout=log,
                stderr=log,
                cwd=common.ROOT,
            )
            try:
                child.wait(timeout=max(45, plan['parameters']['duration_s'] + 35))
            except BaseException:
                child.terminate()  # Runner traps SIGTERM and attempts a fresh measured hold.
                child.wait(timeout=10)
                raise
    finally:
        if recorder is not None:
            if video.exists():
                (video / "stop").touch()
            try:
                recorder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                recorder.terminate()
                recorder.wait(timeout=5)
        video_log.close()
    if not actual.exists():
        raise RuntimeError("SDK 执行器未生成收据：" + str(run / "actual.log"))
    report = read_json(actual)
    if not getattr(args, 'fast', False):
        common.capture_rgbd(run / "after_rgbd", cfg)
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "success",
                    "error",
                    "measured_wave",
                    "returned_center",
                    "hold_error",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if report.get("failure_hold"):
        print(
            "异常保持已确认：" + str(report["failure_hold"].get("hold_verified", False))
        )
    print("摇晃记录：" + str(actual))
    print("实物持杯与骰子变化需另行确认；未发送张手、抬杯、揭杯或 HOME 指令。")
    if return_receipt:
        return dict(report, actual_path=str(actual))
    return 0 if report["success"] else 2
