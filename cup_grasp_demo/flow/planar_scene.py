"""Independent table capture for shaking; never detects cups or controls joints."""
from pathlib import Path
import json
import time

from cup_grasp_demo.flow import debug as common
from cup_grasp_demo.flow.core import configured_tcp, digest, load_config, read_json


def capture(args):
    output = args.session / 'planar_table_scene.json'
    pending = output.with_suffix('.pending')
    pending.write_text('table capture incomplete\n')
    cfg = load_config(args.config)
    run = common.new_run(args.session, 'planar_table_capture')
    common.capture_rgbd(run / 'rgbd', cfg)
    scene, quality = home_scene(run, cfg)
    paths = [args.config, Path(cfg['calibration']), Path(cfg['tcp_candidate'])]
    paths += list((run / 'rgbd').glob('*'))
    record = dict(kind='planar_table_scene', captured_epoch_s=time.time(),
                  config_path=str(args.config.resolve()), scene=scene,
                  T_flange_tcp=configured_tcp(cfg).tolist(),
                  calibration_quality_passed=quality,
                  input_hashes={str(p.resolve()): digest(p) for p in paths if p.is_file()})
    temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(output)
    pending.unlink()
    print('桌面已保存（未识别杯子、未控制机械臂）：'+str(output))
    return 0


def verify(path, config_path):
    path = Path(path)
    if path.with_suffix('.pending').exists():
        raise ValueError('最新桌面采集未成功，请重新 table-capture')
    record = read_json(path)
    if record.get('kind') != 'planar_table_scene':
        raise ValueError('不是独立摇晃桌面数据')
    if Path(record['config_path']).resolve() != Path(config_path).resolve():
        raise ValueError('桌面配置不匹配，请重新 table-capture')
    for name, expected in record['input_hashes'].items():
        if digest(name) != expected:
            raise ValueError('桌面依赖已改变，请重新 table-capture：'+name)
    return record, load_config(config_path)


def home_scene(run, cfg):
    """Fit the red table without requiring a visible, unoccluded cup."""
    meta, depth, image, _ = load_batch(run)
    camera, quality = common.camera_transform(meta, cfg)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165))
           & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50))
    table, normal, fraction, rms = _plane(
        deproject(depth, red, meta['intrinsics'], meta['depth_scale_m']),
        Config(plane_tolerance_m=cfg['plane_tolerance_mm'] / 1000))
    scene = dict(cup_support_base_m=(camera[:3, :3] @ table + camera[:3, 3]).tolist(),
                 cup_normal_base=(camera[:3, :3] @ normal).tolist(),
                 cup_envelope_radius_m=0, geometry=dict(height_m=0),
                 table_fit=dict(inlier_fraction=fraction, rms_mm=rms * 1000))
    return scene, quality
