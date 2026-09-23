"""HOME -> green open-cup GRASP -> lifted joint SHAKE -> PLACE -> HOME."""

import json
import math
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
import cv2
import numpy as np
from cup_grasp_demo.flow import debug as common
from cup_grasp_demo.flow.core import (
    ROOT,
    Screen,
    cached_screen_geometry,
    digest,
    load_config,
    read_json,
    write_json,
)
from vision.geometry.cup_height import detect
from vision.inference.detector import infer, runtime_settings
from cup_grasp_demo.flow.green_cup_planning import (
    arm_plan,
    held_cup_clearance,
    offset_target,
    solve,
    vertical_targets,
)
from cup_grasp_demo.flow.core import save, log_output
from cup_grasp_demo.flow.session_storage import session_lock
from cup_grasp_demo.flow.shake import Kinematics
from cup_grasp_demo.flow.joint_profile import make_plan as joint_plan, trajectory as joint_trajectory
from vision.capture.frame_io import load_batch

PHASES = (
    "HOME",
    "CAPTURE",
    "PLAN",
    "APPROACH",
    "GRIP",
    "LIFT",
    "SHAKE",
    "LOWER",
    "OPEN",
    "RETURN_HOME",
)


class ReplanStart(RuntimeError):
    pass


def validate(cfg):
    from cup_grasp_demo.flow.green_capture import capture_arguments

    capture_arguments(cfg)
    g = cfg["green_cup"]
    # Merge grasp strategy fields (vision/strategy/) so validate can check them.
    if "strategy_file" in g:
        from vision.strategy.loader import load_strategy
        strategy = load_strategy(g["strategy_file"])
        for key, value in vars(strategy).items():
            g.setdefault(key, value)
    if g.get('fast_motion_profile', 'quintic') not in ('quintic', 'trapezoid'):
        raise ValueError('fast_motion_profile must be quintic or trapezoid')
    for key in ('fast_cached_feedback', 'fast_overlap_grip_preparation',
                'fast_uncompressed_capture', 'fast_analytic_rim_jacobian', 'fast_batched_ik_rotations',
                'fast_parallel_startup', 'fast_dogbox_ik', 'fast_minimize_lift_travel'):
        if type(g.get(key, False)) is not bool:
            raise ValueError(key + ' must be boolean')
    if g.get('precision_error_action', 'stop') not in ('stop', 'record'):
        raise ValueError('precision_error_action must be stop or record')
    hand_bound = g.get('hand_start_tolerance_deg', 0.5)
    if isinstance(hand_bound, bool) or not isinstance(hand_bound, (int, float)) or not math.isfinite(hand_bound) or not 0.1 <= hand_bound <= 5.0:
        raise ValueError('hand_start_tolerance_deg must be 0.1..5.0 degrees')
    delta = g.get('settled_joint_delta_deg', .05)
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or not .05 <= delta <= .5:
        raise ValueError('settled_joint_delta_deg must be 0.05..0.5')
    if type(g.get("direct_return_home", False)) is not bool:
        raise ValueError("direct_return_home must be boolean")
    if type(g.get("persistent_runtime", True)) is not bool:
        raise ValueError("persistent_runtime must be boolean")
    if g.get('table_plane_source', 'live_depth') not in ('live_depth', 'calibrated'):
        raise ValueError('table_plane_source must be live_depth or calibrated')
    if type(g.get("recovery_attempts", 1)) is not int or not 0 <= g.get("recovery_attempts", 1) <= 2:
        raise ValueError("recovery_attempts must be 0..2")
    lower_attempts = g.get("lower_recovery_attempts", g.get("recovery_attempts", 1))
    if type(lower_attempts) is not int or not 0 <= lower_attempts <= 2:
        raise ValueError("lower_recovery_attempts must be 0..2")
    release_tolerance = g.get("place_arrival_tolerance_mm", g["place_tolerance_mm"])
    if isinstance(release_tolerance, bool) or not isinstance(release_tolerance, (int, float)) or not math.isfinite(release_tolerance) or not 0 <= release_tolerance <= 10:
        raise ValueError("place_arrival_tolerance_mm must be 0..10")
    tolerance = g.get("shake_start_tolerance_deg", .5)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not .05 <= tolerance <= .5:
        raise ValueError("shake_start_tolerance_deg must be 0.05..0.5")
    freshness = g.get("shake_feedback_freshness_limit_s", .1)
    if (isinstance(freshness, bool) or not isinstance(freshness, (int, float))
            or not math.isfinite(freshness) or not .1 <= freshness <= .5):
        raise ValueError("shake_feedback_freshness_limit_s must be 0.1..0.5")
    if type(g.get("fast_speed_percent", 100)) is not int or not 1 <= g.get("fast_speed_percent", 100) <= 100:
        raise ValueError("fast_speed_percent must be 1..100")
    phase_speeds = g.get('fast_phase_speed_percent', {})
    if not isinstance(phase_speeds, dict) or set(phase_speeds) - {'approach', 'return_home'} or any(type(v) is not int or not 1 <= v <= 100 for v in phase_speeds.values()):
        raise ValueError('fast_phase_speed_percent: approach/return_home must be 1..100')
    fast_finger = g.get('fast_finger_duration_s', g['finger_duration_s'])
    if isinstance(fast_finger, bool) or not isinstance(fast_finger, (int, float)) or not math.isfinite(fast_finger) or not .25 <= fast_finger <= 2.55:
        raise ValueError('fast_finger_duration_s must be 0.25..2.55')
    from cup_grasp_demo.flow.green_hand_execution import open_feedback_reference
    open_feedback_reference(g)
    for name in ("open_targets_0_100", "grip_targets_0_100"):
        if len(g[name]) != 6 or any(
            type(v) is not int or not 0 <= v <= 100 for v in g[name]
        ):
            raise ValueError(name)
    if type(g.get("approach_via_above", True)) is not bool:
        raise ValueError("approach_via_above must be boolean")
    if g["open_targets_0_100"] != [0] * 6:
        raise ValueError("HOME/PLACE require open hand")
    g.setdefault("retreat_clearance_mm",
                 g["approach_clearance_mm"] if g.get("approach_via_above", True) else 50)
    g.setdefault("place_offset_base_mm", [0, 0, 0])
    for key, bounds in {
        "lift_mm": (10, 100),
        "retreat_clearance_mm": (20, 150),
        "approach_clearance_mm": (20 if g.get("approach_via_above", True) else 0, 150),
        "finger_duration_s": (0.5, 2.55),
        "finger_settle_s": (0, 2),
        "held_cup_margin_mm": (0, 20),
        "place_tolerance_mm": (0, 5),
    }.items():
        v = g[key]
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not bounds[0] <= v <= bounds[1]
        ):
            raise ValueError(key)
    for key in (
        "contact_offset_base_mm",
        "place_offset_base_mm",
        "tcp_offset_flange_mm",
        "wrist_reference_deg",
    ):
        if np.asarray(g[key]).shape != (3,) or not np.isfinite(g[key]).all():
            raise ValueError(key)
    if np.linalg.norm(g["contact_offset_base_mm"]) > 100:
        raise ValueError("Contact offset >100 mm")
    if np.linalg.norm(g["place_offset_base_mm"]) > 100:
        raise ValueError("Place offset >100 mm")
    p = g["perception"]
    runtime_settings(p)
    for key in ("confidence", "iou_threshold", "mask_threshold"):
        if (
            not isinstance(p[key], (int, float))
            or isinstance(p[key], bool)
            or not 0 < p[key] < 1
        ):
            raise ValueError("green_cup.perception." + key)
    for key in ("height_range_mm", "diameter_range_mm"):
        v = np.asarray(p[key], dtype=float)
        if v.shape != (2,) or not np.isfinite(v).all() or not 0 < v[0] < v[1]:
            raise ValueError(key)
    for key in ("rim_band_mm", "rim_rms_mm", "min_area_px", "min_arc_deg"):
        if (
            not isinstance(p[key], (int, float))
            or not math.isfinite(p[key])
            or p[key] <= 0
        ):
            raise ValueError(key)
    if p.get("geometry_method", "depth_band") not in ("depth_band", "image_rim_depth", "stereo_rim"):
        raise ValueError("Unknown green cup geometry_method")
    if g.get('table_plane_source') == 'calibrated' and p.get('geometry_method') != 'stereo_rim':
        raise ValueError('calibrated table plane currently requires stereo_rim')
    from cup_grasp_demo.flow.green_image_rim import options

    options(p)
    from vision.geometry.circle_rim import height_options
    height_options(p)
    if p.get("geometry_method") == "stereo_rim":
        from vision.geometry.circle_rim import quality_options
        quality_options(p.get("stereo_rim"))
    return g


class Workflow:
    def __init__(self, args):
        self.args = args
        self.cfg = load_config(args.config)
        self.g = validate(self.cfg)
        # validate() already merged the strategy JSON; tolerate mocks without it.
        if "strategy_file" in self.g:
            from vision.strategy.loader import load_strategy
            strategy = load_strategy(self.g["strategy_file"])
            for key, value in vars(strategy).items():
                self.g.setdefault(key, value)
        # Reject malformed shake recipes before HOME opens the hand or moves.
        # Runtime planning still rechecks the actual limits and held-cup path.
        joint_trajectory(read_json(ROOT / self.g['joint_test_config']))
        # FAST is the only green-cup mode; its tuning is applied unconditionally.
        self.cfg.setdefault('joint_delivery', {})['profile'] = self.g.get('fast_motion_profile', 'quintic')
        self.g['perception']['save_debug_images'] = False
        self.g['perception']['analytic_jacobian'] = self.g.get('fast_analytic_rim_jacobian', False)
        self.cfg["speed_percent"] = self.g.get("fast_speed_percent", 100)
        self.g["fast_completion"] = True
        self.g["require_arm_position"] = False
        self.g["finger_settle_s"] = 0.0
        self.g["finger_duration_s"] = self.g.get("fast_finger_duration_s", self.g["finger_duration_s"])
        self.g["read_hand_feedback"] = False
        if self.g['perception'].get('height_mode') == 'fixed' and self.g['perception'].get('geometry_method') == 'stereo_rim':
            self.g['perception']['frame_count'] = 1
        self._sdk = None
        self._vision = None
        self._prepared = None
        if self.g.get('persistent_runtime', True):
            from cup_grasp_demo.flow.green_prepared import PreparedRoutes
            self._prepared = PreparedRoutes()
        self._route_pool = None
        self._route_future = None
        self._snapshot_cache = None
        self.root = args.session
        self.kin = Kinematics()
        self.scene = None
        self.held = None
        self.receipts = {}
        self.tcp = np.asarray(
            read_json(self.cfg["tcp_candidate"])["T_flange_contact_candidate"]
        )
        self.tcp[:3, 3] += np.asarray(self.g["tcp_offset_flange_mm"]) / 1000
        self.reference = read_json(ROOT / self.g["reference"])
        self.deps = [
            args.config,
            Path(self.cfg["calibration"]),
            Path(self.cfg["tcp_candidate"]),
            Path(self.cfg["home"]),
            ROOT / self.g["reference"],
            ROOT / self.g["home_table_scene"],
            ROOT / self.g["joint_test_config"],
            ROOT / self.g["perception"]["model"],
        ]
        self.deps += list(common.HERE.glob("*.py"))
        self.hashes = {str(p.resolve()): digest(p) for p in self.deps}
        self._file_stats = {p: self.file_stamp(p) for p in self.hashes}
        from cup_grasp_demo.flow.green_startup import claim
        self._startup = claim(args.config, args.session)

    def bridge(self, command, output, cfg, request=None, *, on_dispatched=None):
        if not getattr(self, 'g', {}).get('persistent_runtime', hasattr(self, '_sdk')):
            return common.bridge(command, output, cfg, request)
        from cup_grasp_demo.flow.green_runtime import SDKClient
        if getattr(self, '_sdk', None) is None:
            self._sdk = (self._startup.acquire('sdk') if getattr(self, '_startup', None) is not None and 'sdk' not in self._startup.claimed
                         else SDKClient(cfg, common.new_run(self.root, 'green_sdk_session')))
        return self._sdk.call(command, output, request,
                              **({'on_dispatched': on_dispatched} if on_dispatched is not None else {}))

    def close_sdk(self):
        if getattr(self, '_sdk', None) is not None:
            self._sdk.close()
            self._sdk = None
        self._snapshot_cache = None

    def close(self):
        if getattr(self, '_startup', None) is not None:
            self._startup.close()
        for name in ('_capture_pool', '_shake_pool', '_geometry_pool'):
            pool = getattr(self, name, None)
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        if getattr(self, '_route_pool', None) is not None:
            self._route_pool.shutdown(wait=True, cancel_futures=True)
            self._route_pool = None
        try:
            self.close_sdk()
        finally:
            if getattr(self, '_vision', None) is not None:
                self._vision.close()
                self._vision = None

    def prepare_vision(self):
        if self.g.get('persistent_runtime', True) and getattr(self, '_vision', None) is None:
            from cup_grasp_demo.flow.green_runtime import VisionResources
            self._vision = (self._startup.acquire('vision') if getattr(self, '_startup', None) is not None
                            else VisionResources(self.cfg, self.root / 'unused_capture_path'))

    def prepare_step_runtime(self):
        """Connect once up front (CONTROL session startup); frames stay fresh."""
        if not self.g.get('persistent_runtime', True):
            return
        self.prepare_vision()
        try:
            from cup_grasp_demo.flow.green_runtime import SDKClient
            self._sdk = SDKClient(self.cfg, common.new_run(self.root, 'green_sdk_session'))
            self._vision.camera_future.result(timeout=15)
            self._vision.model_future.result(timeout=30)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def file_stamp(path):
        s = Path(path).stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def unchanged(self):
        for p, h in self.hashes.items():
            if (self.file_stamp(p) == self._file_stats.get(p)
                    and all(time.time_ns() - t >= 2_000_000_000 for t in self._file_stats[p][-2:])):
                continue
            if digest(p) != h:
                raise ValueError("运行期间配置/程序发生变化，请停止后重新开始：" + p)

    def snapshot(self):
        # Reuse only a just-returned physical SDK receipt; the executor still
        # rereads and validates the real starting joints before every motion.
        cached = getattr(self, "_snapshot_cache", None)
        if cached is not None and 0 <= time.time() - cached["observed_epoch_s"] <= 1.0:
            common.ready(cached)
            return list(cached["joints_rad"])
        run = common.new_run(self.root, "green_readback")
        snapshot = self.bridge("snapshot", run / "actual.json", self.cfg)
        common.ready(snapshot)
        self._snapshot_cache = snapshot
        return snapshot["joints_rad"]

    def issue(self, plan, label):
        for attempt in range(self.g.get('recovery_attempts', 1) + 1):
            try:
                return self._issue_once(plan, label)
            except ReplanStart:
                if (self.g.get('precision_error_action', 'stop') != 'record'
                        or attempt >= self.g.get('recovery_attempts', 1)):
                    raise
                # No arm/hand action was sent. Recheck the complete new path,
                # including held-cup clearance, before authorizing another request.
                start = self.snapshot()
                if plan.get('stages'):
                    context = plan['replan_context']
                    updated = arm_plan(start, [s['target_q_rad'] for s in plan['stages']],
                                       self.scene, self.cfg, **context)
                    updated['kind'] = plan['kind']
                    if 'target_0_100' in plan:
                        updated['target_0_100'] = plan['target_0_100']
                    plan = updated
                else:
                    plan = dict(plan, start_q_rad=start)
                self.record_recovery(label.upper(), '起点精度偏差：按新反馈重建请求并复核路径')

    def _issue_once(self, plan, label):
        if plan["blockers"]:
            raise ValueError(str(plan["blockers"]))
        self.unchanged()
        run = common.new_run(self.root, "green_" + label)
        write_json(run / "plan.json", plan)
        request = dict(
            execution_authorized=True,
            authorized_epoch_s=time.time(),
            plan=plan,
            config=self.cfg,
        )
        if label in self.g.get('fast_phase_speed_percent', {}):
            request['config'] = dict(self.cfg, speed_percent=self.g['fast_phase_speed_percent'][label])
        write_json(run / "request.json", request)
        try:
            actual = self.bridge(
                "run", run / "actual.json", self.cfg, run / "request.json",
                **({'on_dispatched': self.launch_following} if label == 'grip'
                   and getattr(self, '_pending_following', None) is not None else {})
            )
        except RuntimeError:
            receipt = run / 'actual.json'
            failed = read_json(receipt) if receipt.exists() else {}
            if (failed.get('failure_code') == 'start_position_changed'
                    and failed.get('motion_attempted') is False
                    and failed.get('finger_commands_sent') is False):
                self.close_sdk()
                raise ReplanStart('起点已改变，需要新反馈重新规划') from None
            raise
        if not actual.get("success"):
            self._snapshot_cache = None
            raise RuntimeError("阶段执行失败：" + label)
        self._snapshot_cache = actual.get("final_snapshot")
        self.receipts[label] = str((run / "actual.json").resolve())
        if label == 'grip':
            self._grip_finished_epoch_s = actual.get('final_snapshot', {}).get('observed_epoch_s')
        if label == 'lift':
            events = actual.get('joint_delivery_events', [])
            began = events[0].get('motion_started_epoch_s') if events else None
            self.record_handoff('GRIP_TO_LIFT', getattr(self, '_grip_finished_epoch_s', None), began)
            self._lift_finished_epoch_s = actual.get('final_snapshot', {}).get('observed_epoch_s')
        return actual

    def record_handoff(self, label, ended, began):
        if ended is None or began is None:
            return
        if not hasattr(self, 'handoff_timings'):
            self.handoff_timings = {}
        elapsed = began - ended
        self.handoff_timings[label] = dict(gap_s=elapsed, target_s=.5, target_met=0 <= elapsed <= .5)

    def hand(self, target, label):
        return self.issue(
            dict(
                kind="green_hand_command",
                target_0_100=target,
                start_q_rad=self.snapshot(),
                blockers=[],
                stages=[],
            ),
            label,
        )

    def move(self, targets, label, held=None, margin=0):
        future = getattr(self, '_following_futures', {}).pop(label, None)
        if future is not None:
            ready = future.result(timeout=self.cfg['timeout_s'])
            self._prepared.routes.update(ready.routes)
        start = self.snapshot()
        prepared = getattr(self, '_prepared', None)
        plan = prepared.take(label, start, targets, self.scene, held, margin,
                             start_tolerance_deg=min(.05, self.cfg['start_tolerance_deg'])) if prepared else None
        if plan is None:
            plan = arm_plan(start, targets, self.scene, self.cfg, held=held, cup_margin_mm=margin)
        return self.issue(plan, label)

    def table(self):
        record = read_json(ROOT / self.g["home_table_scene"])
        if record["calibration_sha256"] != digest(self.cfg["calibration"]):
            raise ValueError("标定已改变，请重新保存 HOME 桌面参数")
        self.table_scene = record["scene"]
        self.scene = self.table_scene

    def capture(self):
        if (self.g.get('table_plane_source', 'live_depth') == 'calibrated'
                and not hasattr(self, 'table_scene')):
            self.table()
        (self.root / 'green_capture_retry.json').unlink(missing_ok=True)
        for attempt in range(2):
            try:
                return self._capture_once()
            except ValueError as exc:
                from vision.geometry.circle_rim import RimEdgeQualityError
                transient = isinstance(exc, RimEdgeQualityError) or str(exc) in (
                    'table_plane_not_supported',
                    'Stereo rim requires one YOLO cup in the red workspace',
                )
                if attempt or getattr(self, '_vision', None) is None or not transient:
                    raise
                save(self.root / 'green_capture_retry.json', dict(
                    reason=str(exc), retry=1, action='capture_fresh_frame'))
                # Reuse the running stream: no restart, sleep or extra discarded frames.

    def _capture_once(self):
        run = common.new_run(self.root, "green_capture")
        # Invalidate before acquisition too: camera/model failure cannot expose an old target.
        save(
            self.root / "green_scene.json", dict(valid=False, source_run=str(run))
        )
        for name in (
            "green_rim_debug.png",
            "green_detection.png",
            "green_top_view.png",
            "green_mask.png",
        ):
            (self.root / name).unlink(missing_ok=True)
        try:
            if getattr(self, '_vision', None) is not None:
                self._vision.capture(run / 'rgbd', self.g['perception'].get('frame_count', 5))
            elif self.g['perception'].get('frame_count') == 1:
                common.capture_rgbd(run / "rgbd", self.cfg, frames=1)
            else:
                common.capture_rgbd(run / "rgbd", self.cfg)
            if self.g['perception'].get('frame_count') == 1:
                meta, depth, image, _ = load_batch(run, min_frames=1)
            else:
                meta, depth, image, _ = load_batch(run)
            camera, quality = common.camera_transform(meta, self.cfg)
            instances, model_info = infer(image, self.g["perception"])
        except Exception as exc:
            failure = dict(
                valid=False, source_run=str(run), error=f"{type(exc).__name__}: {exc}"
            )
            save(self.root / "green_rim_diagnostics.json", failure)
            write_json(run / "rim_diagnostics.json", failure)
            raise
        write_json(
            run / "yolo_seg.json", dict(model=model_info, candidates=len(instances))
        )
        from cup_grasp_demo.flow.green_image_rim import overlay

        diagnostic = {}
        try:
            if self.g["perception"].get("geometry_method") == "stereo_rim":
                from vision.geometry.circle_rim import detect_stereo, table_from_base_scene
                fixed_table = None
                if self.g.get('table_plane_source', 'live_depth') == 'calibrated':
                    fixed_table = table_from_base_scene(self.table_scene, camera)
                geo, mask, contour = detect_stereo(
                    run, depth, image, meta, self.g["perception"],
                    self.cfg["plane_tolerance_mm"], instances, diagnostic,
                    fixed_table=fixed_table)
            else:
                geo, mask, contour = detect(
                    depth, image, meta, self.g["perception"],
                    self.cfg["plane_tolerance_mm"], instances=instances,
                    diagnostics=diagnostic)
        except ValueError as exc:
            diagnostic["error"] = str(exc)
            from vision.geometry.circle_rim import RimEdgeQualityError
            if isinstance(exc, RimEdgeQualityError):
                diagnostic['edge_quality'] = exc.report
            raise
        finally:
            write_json(run / "rim_diagnostics.json", diagnostic)
            save(self.root / "green_rim_diagnostics.json", diagnostic)
            if diagnostic.get("error"):
                debug_image = overlay(image, diagnostic)
                cv2.imwrite(str(run / "rim_debug.png"), debug_image)
                cv2.imwrite(str(self.root / "green_rim_debug.png"), debug_image)
        convert = lambda p: (camera @ np.r_[p, 1])[:3]
        center = convert(geo["rim_center_camera_m"])
        normal = camera[:3, :3] @ geo["table_normal_camera"]
        support = center - geo["height_m"] * normal
        self.scene = dict(
            cup_support_base_m=support.tolist(), cup_normal_base=normal.tolist()
        )
        self.contact = center + np.asarray(self.g["contact_offset_base_mm"]) / 1000
        self.geometry = geo
        self.center = center
        self.normal = normal
        self.capture_time = time.time()
        save(
            self.root / "green_scene.json",
            dict(
                valid=True,
                geometry=geo,
                scene=self.scene,
                contact_base_m=self.contact.tolist(),
                T_flange_tcp=self.tcp.tolist(),
                calibration_quality_passed=quality,
                captured_epoch_s=self.capture_time,
                detection_backend="best_green YOLO + image rim + angularly checked depth",
                model=model_info,
                physical_tcp_verified=False,
            ),
        )

    def plan(self):
        plan_started = time.perf_counter()
        target = np.asarray(self.reference["T_base_flange"]).copy()
        target[:3, 3] = self.contact - target[:3, :3] @ self.tcp[:3, 3]
        seed = np.asarray(self.reference["joints_rad"])
        solver_options = ({'fast_fk': True} if self.g.get('fast_batched_ik_rotations', False) else {})
        if self.g.get('fast_dogbox_ik', False):
            solver_options['method'] = 'dogbox'
        self.grasp_q = solve(target, seed, self.g["wrist_reference_deg"], **solver_options)
        self.approach_targets = [self.grasp_q]
        if self.g.get("approach_via_above", True):
            above = target.copy()
            above[2, 3] += self.g["approach_clearance_mm"] / 1000
            self.above_q = solve(above, self.grasp_q, self.g["wrist_reference_deg"], **solver_options)
            self.approach_targets.insert(0, self.above_q)
        ik_finished = time.perf_counter()
        geometry_future = getattr(self, '_geometry_future', None)
        if geometry_future is not None:
            geometry_future.result(timeout=30)
        start_q = self.snapshot()
        readback_finished = time.perf_counter()
        route = arm_plan(start_q, self.approach_targets, self.scene, self.cfg)
        path_finished = time.perf_counter()
        if route["blockers"]:
            raise ValueError(str(route["blockers"]))
        if getattr(self, '_prepared', None) is not None:
            self._prepared.put('approach', route, self.scene)
        save(
            self.root / "green_grasp_plan.json",
            dict(
                planning_timings_s=dict(ik=ik_finished-plan_started,
                    readback=readback_finished-ik_finished,
                    path=path_finished-readback_finished),
                route=route,
                grasp_q_deg=np.degrees(self.grasp_q).tolist(),
                contact_base_m=self.contact.tolist(),
                wrist_preference_deg=self.g["wrist_reference_deg"],
            ),
        )

    def held_geometry(self, q):
        flange, _ = self.kin.forward(q)
        cup = np.eye(4)
        cup[:3, 2] = self.normal
        x = np.array([1.0, 0.0, 0.0])
        x -= self.normal * (x @ self.normal)
        x /= np.linalg.norm(x)
        cup[:3, 0] = x
        cup[:3, 1] = np.cross(self.normal, x)
        cup[:3, 3] = self.center - self.normal * self.geometry['height_m'] / 2
        return dict(T_flange_cup=np.linalg.inv(flange) @ cup,
                    radius_m=self.geometry['radius_m'], height_m=self.geometry['height_m'])

    def prepare_following(self):
        """Prepare from measured grasp pose while the SDK is closing fingers."""
        if getattr(self, '_prepared', None) is None or self.args.until in ('ready', 'grip'):
            return
        overlap = self.g.get('fast_overlap_grip_preparation', False)
        cached = getattr(self, '_snapshot_cache', None)
        if (overlap and cached is not None
                and 0 <= time.time() - cached.get('observed_epoch_s', 0) <= .25):
            # APPROACH has already awaited fresh idle feedback. Pure geometry
            # work may overlap GRIP; the SDK remains owned by the main thread.
            common.ready(cached)
            actual = list(cached['joints_rad'])
        else:
            self._snapshot_cache = None
            before = self.snapshot()
            for _ in range(5):
                self._snapshot_cache = None
                actual = self.snapshot()
                delta_deg = float(np.degrees(np.max(np.abs(np.asarray(actual) - before))))
                if delta_deg <= self.g.get('settled_joint_delta_deg', .05):
                    self.settled_joint_delta_deg = delta_deg
                    break
                before = actual
            else:
                raise RuntimeError('抓取前机械臂尚未停稳')
        self.place_q = actual
        self.held = self.held_geometry(self.place_q)
        start, held = list(self.place_q), self.held
        self._route_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='green-routes')
        self._following_futures = {name: Future() for name in ('lift', 'lower', 'return_home')}
        self._route_future = self._following_futures.pop('lift')
        futures = dict(self._following_futures, lift=self._route_future)

        def publish(label, route, vertical):
            from cup_grasp_demo.flow.green_prepared import PreparedRoutes
            ready = PreparedRoutes()
            ready.routes[label] = route
            ready.vertical.update(vertical)
            futures[label].set_result(ready)

        def prepare():
            try:
                self.build_following(start, held, publish=publish)
            except BaseException as exc:
                for future in futures.values():
                    if not future.done():
                        future.set_exception(exc)
        if overlap and self.g.get('persistent_runtime', True):
            # Start CPU work only after the hand request has reached the worker.
            # Otherwise it competes with request serialization before closure.
            context = copy_context()
            self._pending_following = lambda: self._route_pool.submit(context.run, prepare)
            return  # GRIP starts immediately; LIFT awaits only unfinished work.
        self._route_pool.submit(copy_context().run, prepare)
        # Complete lift preparation before closing, so GRIP -> LIFT performs no IK.
        ready = self._route_future.result(timeout=self.cfg['timeout_s'])
        target = ready.vertical['lift'][1][-1]
        self.start_shake_prepare(target)
        shake_ready = getattr(self, '_shake_future', None)
        if shake_ready is not None:
            shake_ready.result(timeout=self.cfg['timeout_s'])

    def launch_following(self):
        launch = getattr(self, '_pending_following', None)
        self._pending_following = None
        if launch is not None:
            launch()

    def build_following(self, start, held, publish=None):
        from cup_grasp_demo.flow.green_prepared import PreparedRoutes
        prepared = PreparedRoutes()
        fast_options = ({'fast_fk': True} if self.g.get('fast_batched_ik_rotations', False) else {})
        if self.g.get('fast_minimize_lift_travel', False):
            fast_options['minimize_travel'] = True
        lift = vertical_targets(start, self.tcp, self.g['lift_mm'] / 1000,
                                self.g['wrist_reference_deg'], single_target=True, **fast_options)
        place = offset_target(
            start,
            self.tcp,
            np.asarray(self.g.get('place_offset_base_mm', [0, 0, 0])) / 1000,
            self.g['wrist_reference_deg'],
            **fast_options,
        )
        retreat = [] if self.g.get("direct_return_home", False) else vertical_targets(
            start, self.tcp, self.g["retreat_clearance_mm"] / 1000,
            self.g["wrist_reference_deg"], single_target=True, **fast_options)
        prepared.put_vertical('lift', start, lift)
        prepared.put_vertical('place', start, [place])
        prepared.put_vertical('retreat', start, retreat)
        margin = -self.g['place_tolerance_mm']
        for label, source, targets, load, clearance in (
            ('lift', start, lift, held, margin),
            ('lower', lift[-1], [place], held, margin),
            ('return_home', place, [*retreat, self.home], None, 0),
        ):
            route = arm_plan(source, targets, self.scene, self.cfg, held=load, cup_margin_mm=clearance)
            if route['blockers']:
                raise ValueError(f"{label}: {route['blockers']}")
            prepared.put(label, route, self.scene, load, clearance)
            if publish is not None:
                publish(label, prepared.routes[label], dict(prepared.vertical) if label == 'lift' else {})
        return prepared

    def finish_following(self):
        future = getattr(self, '_route_future', None)
        if future is None:
            return
        prepared = future.result(timeout=self.cfg['timeout_s'])
        self._route_future = None
        # GRIP receipts can precede final arm settling: compare a fresh pose.
        if not self.g.get('fast_overlap_grip_preparation', False):
            self._snapshot_cache = None
        actual = self.snapshot()
        if (self.g.get('fast_overlap_grip_preparation', False)
                and 'lift' in prepared.vertical):
            from cup_grasp_demo.flow.green_prepared import lift_seed_close
            old, targets = prepared.vertical['lift']
            if lift_seed_close(old, actual, self.kin, self.tcp):
                # Reuse the IK endpoint, never an unchecked path from another start.
                self._following_futures = {}
                self._shake_future = None
                self.place_q = actual
                self.held = self.held_geometry(actual)
                self._prepared.routes.clear()
                self._prepared.put_vertical('lift', actual, targets)
                self.lift_endpoint_reused = True
                return
        if np.max(np.abs(np.asarray(actual) - self.place_q)) > np.radians(self.cfg['start_tolerance_deg']):
            # Changed branch/pose: discard prepared paths, retain fresh planning.
            self._following_futures = {}
            self._shake_future = None
            self.place_q = actual
            self.held = self.held_geometry(actual)
            self.record_recovery('GRIP', '实际起点变化，丢弃后台路径并按实时姿态规划')
            return
        self._prepared.routes.update(prepared.routes)
        self._prepared.vertical.update(prepared.vertical)

    def vertical(self, label, start, dz, *, single_target):
        prepared = getattr(self, '_prepared', None)
        targets = prepared.take_vertical(label, start, self.kin) if prepared else None
        return targets if targets is not None else vertical_targets(
            start, self.tcp, dz, self.g['wrist_reference_deg'], single_target=single_target,
            **({'minimize_travel': True} if self.g.get('fast_minimize_lift_travel', False) else {}),
            **({'fast_fk': True} if self.g.get('fast_batched_ik_rotations', False) else {}))

    def place_targets(self, start):
        offset_mm = np.asarray(self.g.get('place_offset_base_mm', [0, 0, 0]), dtype=float)
        if np.array_equal(offset_mm, np.zeros(3)):
            return [list(start)]
        prepared = getattr(self, '_prepared', None)
        targets = prepared.take_vertical('place', start, self.kin) if prepared else None
        if targets is not None:
            return targets
        options = {'fast_fk': True} if self.g.get('fast_batched_ik_rotations', False) else {}
        if self.g.get('fast_minimize_lift_travel', False):
            options['minimize_travel'] = True
        return [offset_target(
            start,
            self.tcp,
            offset_mm / 1000,
            self.g['wrist_reference_deg'],
            **options,
        )]

    def record_recovery(self, phase, reason):
        if not hasattr(self, 'recovery_events'):
            self.recovery_events = []
        self.recovery_events.append(dict(phase=phase, reason=reason, epoch_s=time.time()))
        print(f"{phase} 自动修正：{reason}", flush=True)

    def start_shake_prepare(self, target):
        if getattr(self.args, 'until', None) in ('ready', 'grip') or getattr(self, '_shake_future', None) is not None:
            return
        if not getattr(self, 'receipts', {}).get('approach'):
            return
        receipt = read_json(self.receipts['approach'])
        events = receipt.get('joint_delivery_events', [])
        limits = events[-1].get('live_limits') if events else None
        if not limits:
            return  # Legacy executor receipt: use fresh readback at SHAKE.
        self._shake_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='green-shake-plan')
        self._shake_future = self._shake_pool.submit(copy_context().run, self.build_shake,
            dict(success=True, limits=limits, q_after_rad=list(target)))

    def build_shake(self, feedback):
        p = joint_plan(
            feedback,
            read_json(ROOT / self.g["joint_test_config"]),
            self.kin.model.limits_rad,
        )
        qs = [s["q_rad"] for s in p["samples"]]
        table = Screen(table_only=True).check_table_batch(qs, self.scene, self.cfg)
        table["held_cup_min_mm"] = held_cup_clearance(
            qs,
            self.held["T_flange_cup"],
            self.held["radius_m"],
            self.held["height_m"],
            self.scene,
        )
        if table["held_cup_min_mm"] < self.g["held_cup_margin_mm"]:
            table["blockers"].append("持杯摇晃轨迹可能碰桌")
        p["blockers"] += table["blockers"]
        p["planning_passed"] = not p["blockers"]
        p["offline_only"] = False
        p["model_flange_checks"] = []
        for factor in (0, 1, -1):
            q = [a + factor * b for a, b in zip(p["start_q_rad"], p["amplitude_rad"])]
            p["model_flange_checks"].append(
                dict(q_rad=q, T_base_flange=self.kin.forward(q)[0].tolist())
            )
        if p['blockers']:
            raise ValueError(str(p['blockers']))
        return p, table

    def shake(self, attempt=0):
        persistent = self.g.get('persistent_runtime', True)
        if not persistent:
            self.close_sdk()
        run = common.new_run(self.root, "green_joint_shake")
        sdk = os.environ.get(
            "DICE_SDK_PYTHON", "/usr/bin/python3"
        )
        prepared = getattr(self, '_shake_future', None) if attempt == 0 else None
        if prepared is not None:
            p, table = prepared.result(timeout=self.cfg['timeout_s'])
            self._shake_future = None
        elif persistent:
            receipt = read_json(self.receipts['lift'])
            limits = receipt['joint_delivery_events'][-1]['live_limits']
            p, table = self.build_shake(dict(success=True, limits=limits, q_after_rad=self.snapshot()))
        else:
            with (run / "readback.log").open("x") as log:
                subprocess.run(
                    [
                        sdk,
                        str(common.HERE / "shake_readback.py"),
                        "--channel",
                        self.cfg["channel"],
                        "--output",
                        str(run / "limits.json"),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=log,
                    check=True,
                )
            feedback = read_json(run / "limits.json")
            p, table = self.build_shake(feedback)
        write_json(run / "plan.json", p)
        if p["blockers"]:
            raise ValueError(str(p["blockers"]))
        receipt = self.receipts["grip"]
        hashes = dict(self.hashes)
        hashes[receipt] = digest(receipt)
        hashes[str((run / "plan.json").resolve())] = digest(run / "plan.json")
        req = dict(
            execution_authorized=True,
            authorized_epoch_s=time.time(),
            load_context="green_cup_held",
            start_tolerance_deg=self.g.get("shake_start_tolerance_deg", .5),
            feedback_freshness_limit_s=self.g.get("shake_feedback_freshness_limit_s", .1),
            require_center_position=self.g.get("require_arm_position", True),
            grasp_receipt_path=receipt,
            grip_targets_0_100=self.g["grip_targets_0_100"],
            held_cup_margin_mm=self.g["held_cup_margin_mm"],
            plan=p,
            table_screen=table,
            input_hashes=hashes,
            channel=self.cfg["channel"],
            restore_speed_percent=self.cfg["speed_percent"],
        )
        write_json(run / "request.json", req)
        if persistent:
            report = self.bridge('shake', run / 'actual.json', self.cfg, run / 'request.json')
            self._snapshot_cache = None
        else:
            with (run / "actual.log").open("x") as log:
                subprocess.run(
                    [
                        sdk,
                        str(common.HERE / "joint_execution.py"),
                        "--request",
                        str(run / "request.json"),
                        "--sha256",
                        digest(run / "request.json"),
                        "--output",
                        str(run / "actual.json"),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=log,
                    check=False,
                )
            report = read_json(run / "actual.json")
        self.record_handoff('LIFT_TO_SHAKE', getattr(self, '_lift_finished_epoch_s', None), report.get('motion_started_epoch_s'))
        if not report.get("success") or not report.get("duration_completed"):
            if (report.get("failure_code") == "start_position_changed"
                    and report.get("motion_attempted") is False
                    and attempt < self.g.get("recovery_attempts", 1)):
                self.record_recovery("SHAKE", "起点变化，重新读取实时姿态并重新检查摇晃轨迹")
                return self.shake(attempt + 1)
            raise RuntimeError("摇晃未完成，保持闭手：" + str(report.get("error", "缺少完成记录")))
        # Verify return to the actual lift endpoint independently of a success label.
        if self.g.get("require_arm_position", True) and max(abs(a - b) for a, b in zip(self.snapshot(), self.lift_q)) > math.radians(
            0.5
        ):
            raise RuntimeError("摇晃未回抬杯起点，禁止下降")
        self.receipts["shake"] = str(run / "actual.json")

    def perform(self, phase):
        if phase == "HOME":
            self.prepare_vision()
            self.table()
            home = read_json(self.cfg["home"])
            self.home = np.radians(home["joints_deg"]).tolist()
            start = self.snapshot()
            already_home = np.max(np.abs(np.asarray(start) - self.home)) <= math.radians(.5)
            if already_home:
                # CONTROL cycles reach HOME repeatedly; reuse the pools.
                if getattr(self, '_geometry_pool', None) is None:
                    self._geometry_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='green-geometry')
                self._geometry_future = self._geometry_pool.submit(
                    copy_context().run, lambda: Screen(table_only=True))
                route = dict(start_q_rad=start, stages=[], blockers=[])
                if getattr(self, '_capture_pool', None) is None:
                    self._capture_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='green-capture')
                self._capture_future = self._capture_pool.submit(self.capture)
            else:
                route = arm_plan(start, [self.home], self.scene, self.cfg)
            route.update(kind='green_home_open', target_0_100=[0] * 6)
            self.issue(route, 'home')
        elif phase == "CAPTURE":
            future = getattr(self, '_capture_future', None)
            if future is None:
                self.capture()
            else:
                future.result(timeout=60)
                self._capture_future = None
        elif phase == "PLAN":
            self.plan()
        elif phase == "APPROACH":
            if time.time() - self.capture_time > self.cfg["plan_max_age_s"]:
                raise ValueError("杯位数据过期")
            approach_started = time.perf_counter()
            self.move(self.approach_targets, "approach")
            motion_finished = time.perf_counter()
            if getattr(self, '_prepared', None) is not None:
                self.prepare_following()
            save(self.root / "green_approach_timings.json", dict(
                movement_s=motion_finished-approach_started,
                following_preparation_s=time.perf_counter()-motion_finished,
                total_s=time.perf_counter()-approach_started))
        elif phase == "GRIP":
            self.hand(self.g["grip_targets_0_100"], "grip")
            if getattr(self, '_route_future', None) is None:
                self.place_q = self.snapshot()
                self.held = self.held_geometry(self.place_q)
        elif phase == "LIFT":
            self.finish_following()
            targets = self.vertical(
                "lift", self.place_q,
                self.g["lift_mm"] / 1000,
                single_target=True,
            )
            if hasattr(self, 'args'):
                self.start_shake_prepare(targets[-1])
            self.move(targets, "lift", self.held, -self.g["place_tolerance_mm"])
            self.lift_q = self.snapshot()
        elif phase == "SHAKE":
            self.shake()
        elif phase == "LOWER":
            # Replan from fresh feedback to the configured base-frame place
            # offset. Zero offset preserves the original grasp pose exactly.
            targets = self.place_targets(self.place_q)
            self.move(targets, "lower", self.held, -self.g["place_tolerance_mm"])
            before = self.kin.forward(targets[-1])[0] @ self.tcp
            lower_attempts = self.g.get("lower_recovery_attempts", self.g.get("recovery_attempts", 1))
            for attempt in range(lower_attempts + 1):
                after = self.kin.forward(self.snapshot())[0] @ self.tcp
                error_mm = np.linalg.norm(after[:3, 3] - before[:3, 3]) * 1000
                self.place_arrival_error_mm = float(error_mm)
                if error_mm <= self.g.get("place_arrival_tolerance_mm", self.g["place_tolerance_mm"]):
                    break
                if attempt == lower_attempts:
                    if self.g.get('precision_error_action', 'stop') == 'record':
                        self.record_recovery(
                            "LOWER",
                            f"放杯偏差 {error_mm:.2f} mm；record 模式继续张手归位",
                        )
                        break
                    raise RuntimeError(f"放杯修正后偏差仍为 {error_mm:.2f} mm，保持闭手")
                self.record_recovery("LOWER", f"放杯偏差 {error_mm:.2f} mm，重新规划到原放杯位置")
                self.move(targets, "lower_correct", self.held, -self.g["place_tolerance_mm"])
        elif phase == "OPEN":
            self.hand(self.g["open_targets_0_100"], "release")
        elif phase == "RETURN_HOME":
            if self.g.get("direct_return_home", False):
                self.move([self.home], "return_home")
                return
            q = self.snapshot()
            targets = self.vertical(
                "retreat", q,
                self.g["retreat_clearance_mm"] / 1000,
                single_target=True,
            )
            # One retreat endpoint and one HOME endpoint in one SDK session.
            self.move([*targets, self.home], "return_home")


def run(args):
    if args.status:
        print(
            json.dumps(
                read_json(args.session / "green_pipeline_state.json"),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.resume:
        raise ValueError("绿色杯流程暂不支持跨进程恢复；停止后核实持杯状态再开始")
    if getattr(args, 'mode', 'fast') != 'fast':
        raise ValueError("绿色杯流程仅支持 fast 模式（step/auto 已移除；CONTROL 会话内部同样走 fast 参数）")
    if not args.execute:
        print(
            "绿色杯：HOME → CAPTURE → GRASP → LIFT → SHAKE → LOWER → OPEN → HOME；加 --execute 执行"
        )
        return 0
    until = {
        "ready": "APPROACH",
        "grip": "GRIP",
        "shake": "SHAKE",
        "place": "RETURN_HOME",
    }.get(args.until)
    if until is None:
        raise ValueError("绿色杯 until 使用 ready/grip/shake/place")
    with session_lock(args.session), cached_screen_geometry():
        flow = Workflow(args)
        if getattr(flow, 'g', {}).get('installation_requires_calibration', False):
            raise ValueError('相机/基座重新安装：请先完成新手眼标定、桌面板登记及桌面参数更新，再清除 installation_requires_calibration')
        state = dict(
            strategy="green_open_cup", status="RUNNING", events=[], phase_timings_s={}
        )
        startup = getattr(flow, '_startup', None)
        if startup is not None:
            state['parallel_startup_elapsed_s'] = time.perf_counter() - startup.started
        path = args.session / "green_pipeline_state.json"
        try:
            for phase in PHASES:
                print("GREEN PIPELINE " + phase, flush=True)
                start = time.perf_counter()
                flow.unchanged()
                state["active_phase"] = phase
                save(path, state)
                log_run = common.new_run(
                    args.session, "green_phase_" + phase.lower()
                )
                with log_output(log_run / "phase.log"):
                    flow.perform(phase)
                elapsed = time.perf_counter() - start
                state["events"].append(dict(phase=phase, status="completed"))
                state["phase_timings_s"][phase] = elapsed
                if phase == 'CAPTURE' and startup is not None:
                    state['startup_to_capture_s'] = time.perf_counter() - startup.started
                state['reuse_events'] = flow._prepared.events if getattr(flow, '_prepared', None) else []
                state["handoff_timings"] = getattr(flow, 'handoff_timings', {})
                state['lift_endpoint_reused'] = getattr(flow, 'lift_endpoint_reused', False)
                state["receipts"] = flow.receipts
                state["recovery_events"] = getattr(flow, "recovery_events", [])
                save(path, state)
                print(f"[耗时] {phase}: {elapsed:.3f} s", flush=True)
                if phase == until:
                    break
            state["status"] = "COMPLETED"
            save(path, state)
            return 0
        except BaseException as exc:
            state.update(
                status="FAILED",
                error=f"{type(exc).__name__}: {exc}",
                receipts=flow.receipts,
                recovery_events=getattr(flow, "recovery_events", []),
            )
            save(path, state)
            raise
        finally:
            flow.close()
