"""CLI handlers for operator-run side-grasp tests."""

import json
from pathlib import Path
import time

from cup_grasp_demo.calibration_debug import debug as common
from cup_grasp_demo.calibration_debug.core import digest, read_json, write_json
from cup_grasp_demo.calibration_debug.cup_recheck import verify_cup
from cup_grasp_demo.calibration_debug.grasp import make_grasp_plan, observed_scene
from cup_grasp_demo.calibration_debug.session_storage import prepare_plan
from cup_grasp_demo.side_grasp.preview_index import load_batch


def implementation_hashes():
    hashes = {name: digest(common.HERE / name) for name in
              ('session_storage.py', 'parameters.py', 'core.py', 'contact_geometry.py', 'cup_recheck.py', 'cup_selection.py', 'cup_perception.py', 'debug.py', 'grasp.py',
               'tcp_overlay.py', 'grasp_cli.py', 'grasp_execution.py', 'hardware.py', 'joint_delivery.py', 'direct_grasp.py')}
    hashes['side_grasp/preview_index.py'] = digest(common.ROOT / 'cup_grasp_demo/side_grasp/preview_index.py')
    for name in ('yolo_seg.py', 'geometry.py'):
        hashes['dice_cup_localization/' + name] = digest(common.ROOT / 'dice_cup_localization' / name)
    return hashes


def summarize(plan):
    keys = ('allow_hand_cup_contact', 'contact_warnings', 'approach', 'strategy', 'start_state', 'until_state', 'pregrasp_base_m', 'close_ready_base_m',
            'closing_gap_mm', 'contact_base_m', 'outward_base', 'blockers')
    print(json.dumps({key: plan[key] for key in keys if key in plan}, indent=2, ensure_ascii=False))
    states = list(dict.fromkeys(s['state'] for s in plan['stages']))
    print('状态：' + ' → '.join(states))
    print(f"分段：{len(plan['stages'])}；screen_passed={plan['screen_passed']}；未自动确认抓牢")


def plan(args, *, prepared=None, cached_scene=None, snapshot=None):
    prepare_plan(args.output, 'side_grasp_debug_plan')
    session, cfg = common.verify_session(args.session)
    run = common.new_run(args.session, 'grasp_plan')
    if snapshot is not None:
        # Same-process FAST handoff; execution still validates live start joints.
        write_json(run / 'snapshot.json', snapshot)
    elif args.joints_json:
        raw = args.joints_json.read_text()
        try:
            snapshot = json.loads(raw)
        except json.JSONDecodeError:
            snapshot = json.loads(raw.splitlines()[-1])
    else:
        snapshot = common.bridge('snapshot', run / 'snapshot.json', cfg)
    common.ready(snapshot)
    scene = cached_scene if cached_scene is not None else observed_scene(args.session, session, cfg)
    result = make_grasp_plan(session, snapshot['joints_rad'], cfg, scene, args.start, args.until)
    result.update(session_path=str(args.session), session_sha256=digest(args.session / 'session.json'),
                  created_epoch_s=time.time(), implementation_hashes=implementation_hashes(),
                  offline_only=bool(args.joints_json) or session['replay_only'])
    write_json(args.output, result)
    if prepared is not None:
        prepared.update(plan_sha256=digest(args.output), scene=scene)
    summarize(result)
    print(f'计划：{args.output}')
    return 2 if result['blockers'] else 0


def validate(plan, directory, cfg):
    if plan.get('kind') != 'side_grasp_debug_plan' or plan.get('offline_only'):
        raise ValueError('Only a live side-grasp plan can execute')
    if (plan['blockers'] or not plan['screen_passed']
            or plan['session_sha256'] != digest(directory / 'session.json')
            or plan['implementation_hashes'] != implementation_hashes()
            or not 0 <= time.time() - plan['created_epoch_s'] <= cfg['plan_max_age_s']):
        raise ValueError('抓取计划被阻塞、改变或过期，请重新 grasp-plan')


def reusable_capture(planned, cfg, proof, *, now=None):
    """Only same-process, same-session capture; never trust a persisted timestamp."""
    maximum = cfg.get('fast_capture_reuse_max_age_s', 0)
    if not proof or maximum <= 0 or planned['start_state'] != 'home':
        return False
    elapsed = (time.monotonic() if now is None else now) - proof['started_monotonic_s']
    return (proof['session_sha256'] == planned['session_sha256']
            and 0 <= elapsed <= maximum)


def execute(args, *, confirm=None, prepared=None, capture_proof=None):
    planned = read_json(args.plan)
    directory = Path(planned['session_path'])
    session, cfg = common.verify_session(directory)
    validate(planned, directory, cfg)
    fast = getattr(args, 'fast', False)
    if prepared is not None:
        if not fast or prepared.get('plan_sha256') != digest(args.plan):
            raise ValueError('同一进程已检查的抓取计划发生改变')
    else:
        scene = observed_scene(directory, session, cfg)
        rebuilt = make_grasp_plan(session, planned['start_q_rad'], cfg, scene,
                                  planned['start_state'], planned['until_state'])
        if json.dumps(rebuilt, sort_keys=True) != json.dumps({k: planned[k] for k in rebuilt}, sort_keys=True):
            raise ValueError('抓取路径或参数改变，请重新 grasp-plan')
    summarize(planned)
    if not args.execute:
        print('仅复核计划；加 --execute 才执行抓取测试')
        return 0
    if session['replay_only']:
        raise ValueError('Replay session cannot execute')
    run = common.new_run(directory, 'execute_grasp')
    before = None
    reuse = bool(fast and prepared is not None and reusable_capture(planned, cfg, capture_proof))
    if fast and cfg.get('fast_single_capture', False) and planned['start_state'] == 'home' and not reuse:
        raise ValueError('FAST 同轮杯位缺失、已改变或过期；重新运行 pipeline；本模式不重复拍照识别')
    if not fast:
        before = common.capture_with_feedback(run / 'rgbd', cfg)
    elif planned['start_state'] == 'home' and not reuse:
        # Keep the live cup recheck. TCP preview-only joint snapshots are omitted.
        common.capture_rgbd(run / 'rgbd', cfg)
    if not fast or (planned['start_state'] == 'home' and not reuse):
        common.check_overlay_camera(run / 'rgbd/frame_000.png', session)
    if reuse:
        preview = directory / 'target.png'
        write_json(run / 'cup_recheck.json', dict(
            method='same_process_recent_capture', independent_recapture=False,
            capture_age_s=time.monotonic() - capture_proof['started_monotonic_s'],
            maximum_age_s=cfg['fast_capture_reuse_max_age_s'],
            session_sha256=planned['session_sha256'], source=str(directory / 'session.json')))
    elif planned['start_state'] == 'home':
        meta, depth, image, _ = load_batch(run)
        common.camera_transform(meta, cfg)
        verify_cup(depth, image, meta, cfg, session['scene']['geometry'], run)
        preview = run / 'cup_recheck.png'
    else:
        preview = run / 'rgbd/frame_000.png'
        print('分步继续：手部可能遮挡杯子，请确认杯子仍在冻结位置；此时不声称自动复核了杯位。')
    endpoint = (planned['close_ready_base_m'] if planned.get('strategy') == 'direct_close' else
                planned['pregrasp_base_m'] if planned['until_state'] == 'pregrasp' else planned['contact_base_m'])
    if not fast:
        common.target_overlay(preview, run / 'before_target.png', session, endpoint, before)
        common.show(run / 'before_target.png', args.show)
    print('确认：相机/基座/桌面未移动；六路手指全张开；杯子保持采集位置；通路无障碍。')
    print('本次终点：' + planned['until_state'] + '；闭手后停留原位，不自动抬杯或摇晃。')
    if confirm is not None and confirm('执行所选抓取阶段：').strip() != 'GRASP':
        return 0
    validate(planned, directory, cfg)
    if reuse and not reusable_capture(planned, cfg, capture_proof):
        raise ValueError('同轮杯位在准备执行期间过期，请重新运行 pipeline')
    if ((not fast or (planned['start_state'] == 'home' and not reuse))
            and time.time() - (run / 'rgbd/frame_000.png').stat().st_mtime > 180):
        raise ValueError('图像已过期，请重新 grasp')
    write_json(run / 'request.json', dict(execution_authorized=True, authorized_epoch_s=time.time(),
                                         plan=planned, config=cfg))
    actual = common.bridge('run', run / 'actual.json', cfg, run / 'request.json')
    receipt = dict(success=True, completed_state=actual['last_state'],
                   physical_grip_verified=False, actual_json=str(run / 'actual.json'),
                   plan_sha256=digest(args.plan), session_sha256=planned['session_sha256'])
    write_json(run / 'receipt.json', receipt)
    print(f'阶段完成：{actual["last_state"]}；记录：{run / "receipt.json"}')
    if fast:
        return 0
    after = common.capture_with_feedback(run / 'after_rgbd', cfg)
    common.check_overlay_camera(run / 'after_rgbd/frame_000.png', session)
    hand_state = 'after_close_command' if planned['until_state'] == 'grip' else 'unknown'
    common.target_overlay(run / 'after_rgbd/frame_000.png', run / 'after_target.png',
                          session, endpoint, after, hand_state)
    common.show(run / 'after_target.png', args.show)
    return 0
