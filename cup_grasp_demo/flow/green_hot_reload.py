"""Transactional parameter refresh at CONTROL's serialized idle boundary.

No watcher thread touches hardware or an in-flight plan. Configuration errors
reject the pending command; the previous validated snapshot remains intact.
"""
from pathlib import Path

from cup_grasp_demo.flow.core import ROOT, digest, read_json
from cup_grasp_demo.flow.green_pipeline import Workflow
from scripts.action_registry import GESTURES_DIR


def tuning_paths(config, green):
    paths = {Path(config).resolve(), (ROOT / green['joint_test_config']).resolve()}
    strategy = green.get('strategy_file')
    if strategy is not None:
        path = (ROOT / strategy if str(strategy).endswith('.json')
                else ROOT / 'vision/strategy' / f'{strategy}.json')
        paths.add(path.resolve())
    return paths


def vision_key(cfg):
    green = cfg['green_cup']
    return (green['perception'], green.get('rtsp', {}),
            green.get('fast_uncompressed_capture', False))


class ParameterReloader:
    def __init__(self, flow, diagnostic):
        self.flow = flow
        self.diagnostic = diagnostic
        self.accepted = None  # First command validates the complete on-disk set.
        flow._hot_reload_paths = {str(p) for p in tuning_paths(flow.args.config, flow.g)}
        # Camera/calibration/code are installation resources, not motion tuning.
        # Keep their existing change guard even when the main config is reloadable.
        camera = ROOT / 'vision/camera.json'
        self.camera_hash = digest(camera)

    def manifest(self):
        raw = read_json(self.flow.args.config)
        paths = tuning_paths(self.flow.args.config, raw['green_cup'])
        paths.update(p.resolve() for p in GESTURES_DIR.glob('*.json'))
        return {str(p): digest(p) for p in sorted(paths)}

    def __call__(self, *, force=False):
        from cup_grasp_demo.flow.green_control import build_action_runtime

        flow = self.flow
        flow.unchanged()  # Source/installation changes must never be re-baselined.
        if digest(ROOT / 'vision/camera.json') != self.camera_hash:
            raise ValueError('相机配置已变化，请停止应用，核对标定后重启')
        before = self.manifest()
        if not force and before == self.accepted:
            return None
        candidate = Workflow(flow.args)
        try:
            if candidate.g.get('installation_requires_calibration', False):
                raise ValueError('请先完成现场标定和桌面登记')
            for key in ('channel', 'pipeline_strategy'):
                if candidate.cfg.get(key) != flow.cfg.get(key):
                    raise ValueError(f'{key} 变化需要重启应用')
            for key in ('persistent_runtime', 'hot_reload'):
                if candidate.g.get(key, True) != flow.g.get(key, True):
                    raise ValueError(f'{key} 变化需要重启应用')
            candidate._hot_reload_paths = {
                str(p) for p in tuning_paths(candidate.args.config, candidate.g)}
            old_fixed = {p: h for p, h in flow.hashes.items()
                         if p not in flow._hot_reload_paths}
            new_fixed = {p: h for p, h in candidate.hashes.items()
                         if p not in candidate._hot_reload_paths}
            if old_fixed != new_fixed:
                raise ValueError('标定、HOME、桌面、参考姿态、模型或程序变化需要重启应用')
            actions, run_action = build_action_runtime(
                candidate, self.diagnostic, strict=True, runtime_flow=flow)
            if before != self.manifest():
                raise ValueError('参数保存尚未完成，请保存完毕后重试')
            # The full parser/registry read must correspond to the manifest.
            for p in candidate._hot_reload_paths:
                if candidate.hashes[p] != before[p]:
                    raise ValueError('参数保存尚未完成，请保存完毕后重试')
            candidate.unchanged()
            if digest(ROOT / 'vision/camera.json') != self.camera_hash:
                raise ValueError('相机配置已变化，请核对标定后重启')
        except BaseException:
            candidate.close()
            raise

        changed = sorted(p for p in set(before) | set(self.accepted or {})
                         if before.get(p) != (self.accepted or {}).get(p))
        # Drain old preplanning/capture tasks before sharing resources with a new
        # snapshot. No SDK/CAN reconnection for ordinary speed/offset/hand edits.
        sdk, vision = flow._sdk, flow._vision
        keep_vision = vision_key(flow.cfg) == vision_key(candidate.cfg)
        flow._sdk = None
        if keep_vision:
            flow._vision = None
        try:
            flow.close()
        except BaseException:
            flow._sdk = sdk
            if keep_vision:
                flow._vision = vision
            candidate.close()
            raise
        candidate._sdk = sdk
        if sdk is not None:
            sdk.cfg = candidate.cfg
        if keep_vision:
            candidate._vision = vision
            if vision is not None:
                vision.cfg = candidate.cfg
        # Preserve the public flow identity used by the dispatcher/finalizer.
        # All prepared routes, captured poses and per-cycle state are discarded.
        flow.__dict__.clear()
        flow.__dict__.update(candidate.__dict__)
        self.accepted = before
        self.diagnostic.write('[config] 参数热加载完成：' + ', '.join(changed) + '\n')
        self.diagnostic.flush()
        return actions, run_action, changed
