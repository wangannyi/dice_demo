#!/usr/bin/env python3
"""NERO eye-to-hand calibration. Commands never move/enable/reset the arm."""
import argparse
import json
import signal
import sys
import time
from pathlib import Path
import numpy as np
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from nero_calibration.core import matrix, pose_matrix, matrix_pose, inverse, tcp_transform, solve, distance

ROOT = Path(__file__).resolve().parent


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_image_new(path, image):
    import cv2
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError('Image encoding failed')
    with Path(path).open('xb') as f:
        f.write(encoded.tobytes())


def read_dataset(path):
    path = Path(path)
    if (path/'INVALIDATED.json').exists():
        raise ValueError('Dataset is invalidated; inspect INVALIDATED.json and start a new session')
    if (path/'AUTO_INCOMPLETE.json').exists():
        raise ValueError('Automatic collection is incomplete; inspect AUTO_INCOMPLETE.json')
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest.get('schema') != 1 or manifest.get('mode') != 'eye_to_hand':
        raise ValueError('Unsupported dataset schema/mode')
    matrix(manifest['T_flange_tcp'])
    samples = [json.loads(p.read_text()) for p in sorted(path.glob('sample_*.json'))]
    for s in samples:
        flange = matrix(s['T_base_flange'])
        matrix(s['T_camera_board'])
        if 'T_base_tcp' in s and not np.allclose(
                matrix(s['T_base_tcp']), flange @ matrix(manifest['T_flange_tcp']),
                atol=1e-6, rtol=0):
            raise ValueError('Saved T_base_tcp disagrees with flange and session TCP')
    return manifest, samples


def same_board_for_resume(previous, current):
    # Detection ROI does not change object geometry or camera pixel coordinates.
    # All other fields, including dimensions, image profile and quality gates,
    # remain exact-match requirements. Original manifest is never rewritten.
    return ({k: v for k, v in previous.items() if k not in ('image_roi_xyxy', 'image_exclude_rois_xyxy')} ==
            {k: v for k, v in current.items() if k not in ('image_roi_xyxy', 'image_exclude_rois_xyxy')})


def read_resume_dataset(path, board, tcp_name, tcp):
    """Validate a complete session before opening the camera or arm."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError('Resume requires an existing dataset directory')
    manifest, samples = read_dataset(path)
    if not same_board_for_resume(manifest.get('board', {}), board):
        raise ValueError('Resume board configuration disagrees with session manifest')
    if manifest.get('tcp') != tcp_name or not np.allclose(
            matrix(manifest['T_flange_tcp']), tcp, atol=1e-9, rtol=0):
        raise ValueError('Resume TCP disagrees with session manifest')
    if not isinstance(manifest.get('camera'), dict) or not manifest['camera'].get('serial'):
        raise ValueError('Resume session lacks camera identity')
    expected = {'sample_' + f'{i:04d}' + suffix
                for i in range(len(samples))
                for suffix in ('.json', '.png', '_detected.png')}
    actual = {p.name for p in path.iterdir() if p.name.startswith('sample_')}
    if actual != expected or any(not (path / name).is_file() for name in expected):
        raise ValueError('Resume sample numbering has gaps, missing images, or orphan files')
    for sample in samples:
        if 'T_base_tcp' not in sample:
            raise ValueError('Resume sample lacks T_base_tcp for the recorded session TCP')
    return manifest, samples


def capture_sample(arm, camera, detector, frame_check=None):
    from nero_calibration.sensors import assert_still
    poses = []
    # Require stationary fresh observations before and around image acquisition.
    for _ in range(8):
        poses.append(arm.read())
        time.sleep(.05)
    assert_still(poses)
    joints_before = arm.read_joints()
    # Discard queued frames collected before the current pose settled.
    discarded_joints = []
    for _ in range(5):
        frame = camera.capture()
        discarded_joints.append(arm.read_joints())
        if frame_check is not None:
            frame_check(frame)
    start = time.monotonic()
    observations = []
    for _ in range(3):
        poses.append(arm.read())
        image = camera.capture()
        frame_joints = arm.read_joints()
        board, quality, vis = detector.detect(image, camera.K, camera.D)
        if frame_check is not None:
            frame_check(image, board)
        poses.append(arm.read())
        observations.append((board, quality, image, vis, poses[-1], frame_joints))
    assert_still(poses)
    if any(p > .002 or a > 1. for p, a in
           [distance(observations[0][0], o[0]) for o in observations[1:]]):
        raise ValueError('Board observations unstable; sample rejected')
    board, quality, image, vis, flange, _ = min(observations, key=lambda o: o[1]['reprojection_rms_px'])
    joints_after = arm.read_joints()
    if np.max(np.abs(np.asarray(joints_after['joints_rad']) - joints_before['joints_rad'])) > np.radians(.2):
        raise ValueError('Joints moved during calibration sample')
    return {'T_base_flange': flange.tolist(), 'T_camera_board': board.tolist(),
            'joints_rad': joints_after['joints_rad'], 'joints_deg': joints_after['joints_deg'],
            'joint_observations': {'before': joints_before, 'after': joints_after},
            'frame_joint_observations': [o[5] for o in observations],
            'discarded_frame_joint_observations': discarded_joints,
            'joint_recording_source': 'fresh_sdk_feedback',
            'quality': quality, 'time_unix_s': time.time(),
            'capture_duration_s': time.monotonic()-start}, image, vis


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    def add_tcp(q):
        q.add_argument('--tcp', choices=['flange', 'palm', 'custom'], default='flange')
        q.add_argument('--tcp-config', type=Path)
    q = sub.add_parser('tcp', help='Print fixed transform without hardware')
    add_tcp(q)
    q.add_argument('--flange-pose', type=float, nargs=6)
    q.add_argument('--target-tcp-pose', type=float, nargs=6)
    q = sub.add_parser('feedback-check', help='Read current flange/TCP without moving')
    add_tcp(q)
    q.add_argument('--channel', default='can0')
    for command in ['camera-check', 'collect']:
        q = sub.add_parser(command)
        q.add_argument('--board', type=Path, default=ROOT/'config/board_perceptive.json')
        q.add_argument('--serial', help='RealSense serial; omit only with one connected camera')
        q.add_argument('--dataset', type=Path, required=True, help='Output directory')
        if command == 'collect':
            add_tcp(q)
            q.add_argument('--channel', default='can0')
            q.add_argument('--resume', action='store_true', help='Append to a validated existing session')
            q.add_argument('--preview', action='store_true', help='Open live OpenCV/X11 sampling window')
    q = sub.add_parser('solve')
    q.add_argument('--dataset', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True, help='New result JSON; never applies to robot')
    q.add_argument('--max-position-mm', type=float, default=5.)
    q.add_argument('--max-angle-deg', type=float, default=2.)
    args = p.parse_args(argv)
    if args.command == 'tcp':
        t = tcp_transform(args.tcp, args.tcp_config)
        out = {'tcp': args.tcp, 'T_flange_tcp': t.tolist(), 'offset_m_rad': matrix_pose(t)}
        if args.flange_pose is not None:
            out['tcp_pose_m_rad'] = matrix_pose(pose_matrix(args.flange_pose) @ t)
        if args.target_tcp_pose is not None:
            out['flange_target_m_rad'] = matrix_pose(pose_matrix(args.target_tcp_pose) @ inverse(t))
        print(json.dumps(out, indent=2))
        return 0
    if args.command == 'feedback-check':
        from nero_calibration.sensors import NeroFeedback, assert_still
        t = tcp_transform(args.tcp, args.tcp_config)
        arm = NeroFeedback(args.channel)
        try:
            poses = [arm.read() for _ in range(8)]
            assert_still(poses)
            print(json.dumps({'flange_pose_m_rad': matrix_pose(poses[-1]),
                              'tcp_pose_m_rad': matrix_pose(poses[-1] @ t),
                              'tcp': args.tcp}))
            return 0
        finally:
            arm.close()
    if args.command == 'solve':
        meta, samples = read_dataset(args.dataset)
        result = solve(samples, meta['T_flange_tcp'], args.max_position_mm/1000, args.max_angle_deg)
        result['camera'] = meta['camera']
        result['board'] = meta['board']
        result['tcp'] = meta['tcp']
        result['sample_count'] = len(samples)
        write_new(args.output, result)
        print(json.dumps({'result': str(args.output), 'quality_passed': result['quality_passed']}))
        return 0 if result['quality_passed'] else 2
    import cv2
    from nero_calibration.sensors import RealSenseCamera, NeroFeedback, CharucoDetector
    cfg = json.loads(args.board.read_text())
    detector = CharucoDetector(cfg)
    tcp = tcp_transform(args.tcp, args.tcp_config) if args.command == 'collect' else np.eye(4)
    resume = args.command == 'collect' and args.resume
    if resume:
        manifest, existing = read_resume_dataset(args.dataset, cfg, args.tcp, tcp)
        if args.serial is not None and args.serial != manifest['camera']['serial']:
            raise ValueError('Resume camera serial disagrees with session manifest')
    else:
        # Existing output paths are rejected unless collection explicitly resumes.
        args.dataset.mkdir(parents=True, exist_ok=False)
        existing = []
    camera = arm = preview = None
    previous_sigint_handler = None
    try:
        if args.command == 'collect' and args.preview:
            from nero_calibration.preview import CollectionPreview
            preview = CollectionPreview()
        camera = RealSenseCamera(args.serial, **({"image_profile": cfg["image_profile"]} if "image_profile" in cfg else {}))
        if resume:
            if camera.info != manifest['camera']:
                raise ValueError('Resume camera identity/intrinsics disagree with session manifest')
        else:
            write_new(args.dataset/'manifest.json', {'schema': 1, 'mode': 'eye_to_hand',
                      'camera': camera.info, 'board': cfg, 'opencv_version': cv2.__version__,
                      'tcp': getattr(args, 'tcp', 'flange'), 'T_flange_tcp': tcp.tolist(),
                      'sampling': 'stationary bracketed observations, not hardware timestamp synchronization',
                      'created_unix_s': time.time()})
        if args.command == 'camera-check':
            image = camera.capture()
            if not cv2.imwrite(str(args.dataset/'color.png'), image):
                raise RuntimeError('Image write failed')
            try:
                board, quality, vis = detector.detect(image, camera.K, camera.D)
                cv2.imwrite(str(args.dataset/'detected.png'), vis)
                write_new(args.dataset/'detection.json', {'T_camera_board': board.tolist(), 'quality': quality})
                print(json.dumps({'camera': camera.info, 'detection': quality}))
                return 0
            except ValueError as exc:
                print(json.dumps({'camera': camera.info, 'board_detected': False, 'reason': str(exc)}))
                return 2
        arm = NeroFeedback(args.channel)
        print('静止后 Enter 采样，q 退出。程序只读取反馈，不移动/使能/失能机械臂。', flush=True)
        count = len(existing)
        accepted = [matrix(sample['T_base_flange']) for sample in existing]
        if preview and existing:
            last_image = cv2.imread(str(args.dataset/f'sample_{count-1:04d}_detected.png'))
            preview.restore(last_image, count, existing[-1]['quality'])
        if preview and (args.dataset/'board_window.json').is_file():
            preview.board_window = json.loads((args.dataset/'board_window.json').read_text())['window_xyxy']
        def record_preview_frame(image, board_pose, quality):
            observation = {'time_unix_s': time.time(), 'joints': arm.read_joints(),
                           'T_camera_board': None if board_pose is None else board_pose.tolist(),
                           'quality': quality, 'image_width': image.shape[1],
                           'image_height': image.shape[0]}
            with (args.dataset/'teaching_frames.jsonl').open('a') as stream:
                stream.write(json.dumps(observation, allow_nan=False) + '\n')
        while True:
            try:
                command = (preview.read_command(camera, detector, count,
                                                on_frame=record_preview_frame) if preview
                           else input(f'[{count} samples] > ').strip().lower())
            except EOFError:
                break
            if command == 'q':
                break
            if command:
                continue
            sample = vis = None
            try:
                sample, image, vis = capture_sample(arm, camera, detector)
                f = matrix(sample['T_base_flange'])
                if any(pos < .005 and ang < 3 for pos, ang in (distance(f, t) for t in accepted)):
                    raise ValueError('Pose too similar to an accepted sample')
                sample['T_base_tcp'] = (f @ tcp).tolist()
                sample['sampling_board_config'] = cfg
                stem = f'sample_{count:04d}'
                write_image_new(args.dataset/(stem+'.png'), image)
                write_image_new(args.dataset/(stem+'_detected.png'), vis)
                write_new(args.dataset/(stem+'.json'), sample)
                from nero_calibration.pose_teaching import export_poses
                export_poses(args.dataset)
                accepted.append(f)
                count += 1
                print(json.dumps({'accepted': count, **sample['quality']}), flush=True)
                if preview:
                    preview.result(True, count, f'Saved {stem}.json', sample['quality'], vis)
            except ValueError as exc:
                print(f'样本未保存: {exc}', flush=True)
                if preview:
                    preview.result(False, count, str(exc),
                                   sample['quality'] if sample else None, vis)
        print(f'已保存 {count} 组。使用 solve 命令离线求解。')
        return 0
    except KeyboardInterrupt:
        if args.command != 'collect':
            raise
        # Ignore further Ctrl-C presses while the RealSense stream stops.
        previous_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        print('\n采样已中断；已保存的样本仍在数据目录中。', flush=True)
        return 130
    finally:
        if arm is not None or camera is not None or preview is not None:
            if previous_sigint_handler is None:
                previous_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                try:
                    if preview is not None:
                        preview.close()
                finally:
                    try:
                        if arm is not None:
                            arm.close()
                    finally:
                        if camera is not None:
                            camera.close()
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
