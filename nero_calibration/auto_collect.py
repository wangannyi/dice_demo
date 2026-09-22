#!/usr/bin/env python3
"""Teach a camera window, preview a joint replay, and collect hand-eye samples."""
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'nero_revo2_control'))
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from nero_calibration.calibrate import capture_sample, read_dataset, write_image_new, write_new
from nero_calibration.core import inverse, matrix
from kinematics import load_model
from nero_calibration.sensors import CharucoDetector, NeroFeedback, RealSenseCamera


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def board_corners(board, calibration):
    measured = calibration.get('board_measurement', {})
    width = measured.get('grid_width_mm', board['squares_x'] * board['square_length_m'] * 1000) / 1000
    height = measured.get('grid_height_mm', board['squares_y'] * board['square_length_m'] * 1000) / 1000
    return np.asarray([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]], dtype=np.float64)


def outline(t_camera_board, camera, corners):
    t = matrix(t_camera_board)
    points = (t[:3, :3] @ corners.T + t[:3, 3:4]).T
    if np.any(points[:, 2] <= 0):
        raise ValueError('Board projects behind camera')
    rv = cv2.Rodrigues(t[:3, :3])[0]
    projected = cv2.projectPoints(corners, rv, t[:3, 3],
                                  np.asarray(camera['camera_matrix'], dtype=float),
                                  np.asarray(camera['dist_coeffs'], dtype=float))[0]
    return projected.reshape(-1, 2)


def inside_window(points, box, margin=0):
    p = np.asarray(points, dtype=float)
    x0, y0, x1, y1 = box
    return bool(np.isfinite(p).all() and np.all(p[:, 0] >= x0 + margin)
                and np.all(p[:, 0] < x1 - margin) and np.all(p[:, 1] >= y0 + margin)
                and np.all(p[:, 1] < y1 - margin))


def validate_window(data, manifest):
    box = data['window_xyxy']
    width, height = manifest['camera']['width'], manifest['camera']['height']
    if (data['camera'] != manifest['camera'] or data['board'] != manifest['board']
            or len(box) != 4 or any(type(v) is not int for v in box)
            or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
        raise ValueError('Board window or camera/board identity does not match teaching session')
    return box


def draw_window(dataset):
    manifest, samples = read_dataset(dataset)
    if not samples:
        raise ValueError('Teach at least one sample before drawing a board window')
    image = cv2.imread(str(dataset / 'sample_0000.png'))
    if image is None:
        raise ValueError('First sample image is unavailable')
    x, y, w, h = cv2.selectROI('Draw area containing the full hand board at every pose',
                               image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise ValueError('No board window was selected')
    data = {'schema': 1, 'window_xyxy': [int(x), int(y), int(x+w), int(y+h)],
            'camera': manifest['camera'], 'board': manifest['board']}
    validate_window(data, manifest)
    target = dataset / 'board_window.json'
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'board_window': str(target), 'window_xyxy': data['window_xyxy']}))


def inputs(dataset, calibration_path):
    manifest, samples = read_dataset(dataset)
    calibration = json.loads(calibration_path.read_text())
    window_path = dataset / 'board_window.json'
    window = json.loads(window_path.read_text())
    box = validate_window(window, manifest)
    if len(samples) < 8 or any('joints_rad' not in s for s in samples):
        raise ValueError('Need at least eight taught samples with fresh seven-axis feedback')
    if calibration['camera'] != manifest['camera']:
        raise ValueError('Calibration camera/intrinsics differ from teaching session')
    if not np.allclose(matrix(calibration['T_flange_tcp']), matrix(manifest['T_flange_tcp'])):
        raise ValueError('Calibration and teaching TCP differ')
    return manifest, samples, calibration, box, window_path


def project_joint(q, model, manifest, calibration, corners):
    base_board = (matrix(model.fk(q)) @ matrix(manifest['T_flange_tcp'])
                  @ matrix(calibration['T_tcp_board']))
    return outline(inverse(matrix(calibration['T_base_camera'])) @ base_board,
                   manifest['camera'], corners)


def interpolate(a, b, max_step_deg):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (7,) or b.shape != (7,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid seven-axis waypoint')
    steps = max(1, math.ceil(math.degrees(float(np.max(np.abs(b-a)))) / max_step_deg))
    return [(a + (b-a) * (i/steps)).tolist() for i in range(1, steps+1)]


def check_path(start, targets, model, manifest, calibration, box, max_step_deg,
               margin_px, capture_indices=None):
    corners = board_corners(manifest['board'], calibration)
    waypoints = []
    previous = start
    if capture_indices is None:
        capture_indices = list(range(len(targets)))
    for sample_index, target in enumerate(targets):
        segment = interpolate(previous, target, max_step_deg)
        for i, q in enumerate(segment):
            if any(not joint.lower_rad + math.radians(2) < angle < joint.upper_rad - math.radians(2)
                   for joint, angle in zip(model.joints, q)):
                raise ValueError(f'Joint limit margin failed on segment {sample_index}')
            points = project_joint(q, model, manifest, calibration, corners)
            if not inside_window(points, box, margin_px):
                raise ValueError(f'Board leaves drawn window on segment {sample_index}, step {i}')
            waypoints.append({'q_rad': q, 'capture_sample_index': capture_indices[sample_index]
                              if i == len(segment)-1 else None})
        previous = target
    return waypoints


def taught_route(dataset, samples, manifest, calibration, box, margin_px):
    trace_path = dataset/'teaching_frames.jsonl'
    if not trace_path.is_file():
        raise ValueError('No per-frame teaching trace; repeat manual collect with --preview')
    trace = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    route = [samples[0]['joints_rad']]
    capture = [0]
    corners = board_corners(manifest['board'], calibration)
    for index, sample in enumerate(samples):
        frames = sample.get('frame_joint_observations')
        if not isinstance(frames, list) or len(frames) != 3:
            raise ValueError(f'Sample {index} lacks joints for all three captured frames')
        observed = outline(sample['T_camera_board'], manifest['camera'], corners)
        if not inside_window(observed, box, margin_px):
            raise ValueError(f'Hand board left drawn window at sample {index}')
    for index in range(1, len(samples)):
        start = samples[index-1]['time_unix_s']
        end = samples[index]['time_unix_s']
        if end <= start:
            raise ValueError('Teaching samples are not chronological')
        segment = [frame for frame in trace if start < frame['time_unix_s'] < end]
        if not segment:
            raise ValueError(f'No teaching frames between samples {index-1} and {index}')
        timed_joints = [(start, samples[index-1]['joints_rad'])]
        for frame in segment:
            if (frame['image_width'] != manifest['camera']['width']
                    or frame['image_height'] != manifest['camera']['height']
                    or frame['T_camera_board'] is None):
                raise ValueError(f'Hand board not detected throughout taught segment {index}')
            observed = outline(frame['T_camera_board'], manifest['camera'], corners)
            if not inside_window(observed, box, margin_px):
                raise ValueError(f'Hand board left drawn window during teaching segment {index}')
            route.append(frame['joints']['joints_rad'])
            capture.append(None)
            timed_joints.append((frame['time_unix_s'], frame['joints']['joints_rad']))
        timed_joints.append((end, samples[index]['joints_rad']))
        for (t0, q0), (t1, q1) in zip(timed_joints, timed_joints[1:]):
            if t1 <= t0:
                raise ValueError(f'Teaching frame timestamps are not increasing in segment {index}')
            if (t1-t0 > 2. and math.degrees(max(abs(a-b) for a, b in zip(q0, q1))) > .2):
                raise ValueError(f'Unrecorded arm movement in teaching segment {index}')
        route.append(samples[index]['joints_rad'])
        capture.append(index)
    return route, capture, trace_path


def make_plan(dataset, calibration_path, output, max_step_deg=2., margin_px=8., allow_provisional=False):
    manifest, samples, calibration, box, window_path = inputs(dataset, calibration_path)
    if not calibration.get('quality_passed') and not allow_provisional:
        raise ValueError('Calibration quality failed; pass --allow-provisional to use the accepted result')
    model = load_model()
    targets = [s['joints_rad'] for s in samples]
    route, capture, trace_path = taught_route(dataset, samples, manifest, calibration,
                                               box, margin_px)
    # Check the observed manual route and interpolations between its frames.
    path = check_path(targets[0], route, model, manifest, calibration, box,
                      max_step_deg, margin_px, capture)
    files = [dataset/'manifest.json', window_path, calibration_path,
             trace_path,
             *sorted(dataset.glob('sample_[0-9][0-9][0-9][0-9].json'))]
    plan = {'schema': 1, 'kind': 'handeye_auto_collection', 'source_dataset': str(dataset.resolve()),
            'calibration': str(calibration_path.resolve()), 'camera': manifest['camera'],
            'board': manifest['board'], 'window_xyxy': box, 'margin_px': margin_px,
            'max_step_deg': max_step_deg, 'targets_rad': targets, 'waypoints': path,
            'path_source': 'per_frame_teaching_trace',
            'source_sha256': {str(p.resolve()): digest(p) for p in files},
            'collision_checked': False, 'live_visibility_required': True,
            'quality_passed': bool(calibration.get('quality_passed'))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'plan': str(output), 'samples': len(targets), 'motion_steps': len(path),
                      'collision_checked': False, 'quality_passed': plan['quality_passed']}))
    return plan


def live_board(camera, detector, manifest, calibration, box, margin_px, image=None, pose=None):
    if image is None:
        image = camera.capture()
    if pose is None:
        pose, quality, _ = detector.detect(image, camera.K, camera.D)
    else:
        quality = None
    points = outline(pose, manifest['camera'], board_corners(manifest['board'], calibration))
    if not inside_window(points, box, margin_px):
        raise RuntimeError('Hand board left the drawn image window')
    return quality


def require_arm_ready(arm):
    robot = arm.robot
    status = robot.get_arm_status()
    if (status is None or status.msg.arm_status != 0 or status.msg.ctrl_mode != 1
            or robot.get_joints_enable_status_list() != [True]*7 or robot.has_comm_error()):
        raise RuntimeError('Require normal CAN control and seven enabled joints')


def move_and_watch(arm, camera, detector, manifest, calibration, box, margin_px, target):
    require_arm_ready(arm)
    robot = arm.robot
    last_q = arm.read_joints()['joints_rad']
    robot.move_js(target)
    deadline = time.monotonic() + 12.
    try:
        while time.monotonic() < deadline:
            live_board(camera, detector, manifest, calibration, box, margin_px)
            last_q = arm.read_joints()['joints_rad']
            if math.degrees(max(abs(a-b) for a, b in zip(last_q, target))) <= .5:
                return
        raise TimeoutError('Automatic calibration waypoint did not settle in 12 s')
    except BaseException:
        robot.move_js(last_q)
        raise


def run_plan(plan_path, output, channel, speed_percent, fps=None):
    plan = json.loads(plan_path.read_text())
    if plan.get('kind') != 'handeye_auto_collection':
        raise ValueError('Not a hand-eye automatic collection plan')
    if any(digest(path) != value for path, value in plan['source_sha256'].items()):
        raise ValueError('Teaching/calibration/window changed after planning')
    source = Path(plan['source_dataset'])
    calibration_path = Path(plan['calibration'])
    manifest, _, calibration, box, _ = inputs(source, calibration_path)
    if fps is not None:
        if fps not in (6, 15, 30):
            raise ValueError('Automatic calibration FPS must be 6, 15 or 30')
        manifest = copy.deepcopy(manifest)
        if 'fps' in manifest['camera']:
            manifest['camera']['fps'] = fps
        manifest['board'].setdefault('image_profile', {})['fps'] = fps
    if output.exists():
        raise ValueError('Automatic output directory already exists')
    camera = arm = None
    previous_auto_mode = None
    try:
        camera = RealSenseCamera(manifest['camera']['serial'],
                                 image_profile=manifest['board'].get('image_profile'))
        if camera.info != manifest['camera']:
            raise ValueError('Camera serial/intrinsics/crop changed after teaching')
        arm = NeroFeedback(channel)
        detector = CharucoDetector(manifest['board'])
        require_arm_ready(arm)
        live_board(camera, detector, manifest, calibration, box, plan['margin_px'])
        model = load_model()
        current = arm.read_joints()['joints_rad']
        # The current-to-first approach is checked at execution, since its start changes.
        approach = check_path(current, [plan['targets_rad'][0]], model, manifest, calibration,
                              box, plan['max_step_deg'], plan['margin_px'])
        output.mkdir(parents=True)
        write_new(output/'AUTO_INCOMPLETE.json', {'started_unix_s': time.time(),
                  'plan': str(plan_path.resolve()), 'reason': 'run has not completed all waypoints'})
        write_new(output/'manifest.json', {**manifest, 'created_unix_s': time.time(),
                  'sampling': 'automatic replay with image-window monitoring',
                  'source_dataset': str(source), 'auto_plan': str(plan_path.resolve())})
        robot = arm.robot
        robot.set_joint_limits_enabled(True)
        robot.set_speed_percent(speed_percent)
        previous_auto_mode = robot.get_auto_set_motion_mode_enabled()
        robot.set_auto_set_motion_mode_enabled(False)
        robot.set_motion_mode('js')
        path = approach + plan['waypoints'][1:]
        count = 0
        for waypoint in path:
            move_and_watch(arm, camera, detector, manifest, calibration, box,
                           plan['margin_px'], waypoint['q_rad'])
            index = waypoint['capture_sample_index']
            if index is None:
                continue
            def check_frame(image, pose=None):
                live_board(camera, detector, manifest, calibration, box,
                           plan['margin_px'], image=image, pose=pose)
            sample, image, vis = capture_sample(arm, camera, detector, frame_check=check_frame)
            points = outline(sample['T_camera_board'], manifest['camera'],
                             board_corners(manifest['board'], calibration))
            if not inside_window(points, box, plan['margin_px']):
                raise RuntimeError('Captured board left the drawn image window')
            sample['T_base_tcp'] = (matrix(sample['T_base_flange'])
                                    @ matrix(manifest['T_flange_tcp'])).tolist()
            sample['source_sample_index'] = index
            stem = f'sample_{count:04d}'
            write_image_new(output/(stem+'.png'), image)
            write_image_new(output/(stem+'_detected.png'), vis)
            write_new(output/(stem+'.json'), sample)
            count += 1
            print(json.dumps({'accepted': count, 'source_sample_index': index,
                              **sample['quality']}), flush=True)
        (output/'AUTO_INCOMPLETE.json').unlink()
        return count
    finally:
        if arm is not None:
            try:
                if previous_auto_mode is not None:
                    arm.robot.set_auto_set_motion_mode_enabled(previous_auto_mode)
            finally:
                arm.close()
        if camera is not None:
            camera.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('draw-window', help='Draw a full-board visibility window on first taught frame')
    q.add_argument('--dataset', type=Path, required=True)
    q = sub.add_parser('plan', help='Offline project the complete replay path into the drawn window')
    q.add_argument('--dataset', type=Path, required=True)
    q.add_argument('--calibration', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--max-step-deg', type=float, default=2.)
    q.add_argument('--margin-px', type=float, default=8.)
    q.add_argument('--allow-provisional', action='store_true')
    q = sub.add_parser('run', help='Move the arm and acquire a new dataset')
    q.add_argument('--plan', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--channel', default='can0')
    q.add_argument('--speed-percent', type=int, default=10)
    q.add_argument('--fps', type=int, choices=(6, 15, 30),
                   help='Override only the teaching frame rate; keep resolution and crop')
    q.add_argument('--execute', action='store_true', required=True)
    args = p.parse_args(argv)
    if args.command == 'draw-window':
        draw_window(args.dataset)
    elif args.command == 'plan':
        if not 0 < args.max_step_deg <= 2 or not 0 <= args.margin_px <= 50:
            raise ValueError('Invalid path step or image margin')
        make_plan(args.dataset, args.calibration, args.output, args.max_step_deg,
                  args.margin_px, args.allow_provisional)
    else:
        if not 1 <= args.speed_percent <= 20:
            raise ValueError('Automatic calibration speed must be 1..20%')
        run_plan(args.plan, args.output, args.channel, args.speed_percent, args.fps)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, TimeoutError, OSError, ImportError) as exc:
        print(f'AUTO CALIBRATION STOP: {exc}', file=sys.stderr)
        raise SystemExit(1)
