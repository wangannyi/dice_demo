#!/usr/bin/env python3
"""Teach a camera window, preview a joint replay, and collect hand-eye samples."""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'nero_revo2_control'))
from calibrate import capture_sample, read_dataset, write_image_new, write_new
from core import inverse, matrix
from kinematics import load_model
from reference_board import same_camera_geometry
from sensors import CharucoDetector, NeroFeedback, RealSenseCamera


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
    trace_path = dataset / 'teaching_frames.jsonl'
    missing = []
    if not window_path.is_file():
        missing.append(window_path.name)
    if not trace_path.is_file():
        missing.append(trace_path.name)
    if any(not isinstance(sample.get('frame_joint_observations'), list)
           or len(sample['frame_joint_observations']) != 3 for sample in samples):
        missing.append('sample_*.json: frame_joint_observations')
    if missing:
        raise ValueError('Teaching dataset cannot be replayed; missing '
                         + ', '.join(missing)
                         + '. Repeat manual collect --preview and draw-window.')
    window = json.loads(window_path.read_text())
    box = validate_window(window, manifest)
    if len(samples) < 8 or any('joints_rad' not in s for s in samples):
        raise ValueError('Need at least eight taught samples with fresh seven-axis feedback')
    if not same_camera_geometry(calibration['camera'], manifest['camera']):
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


def taught_route(dataset, samples, manifest, calibration, box, margin_px,
                 capture_only_visibility=False):
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
                    or frame['image_height'] != manifest['camera']['height']):
                raise ValueError(f'Teaching image geometry changed in segment {index}')
            if frame['T_camera_board'] is None and not capture_only_visibility:
                raise ValueError(f'Hand board not detected throughout taught segment {index}')
            if (frame['T_camera_board'] is not None and not capture_only_visibility
                    and not inside_window(outline(frame['T_camera_board'], manifest['camera'], corners),
                                          box, margin_px)):
                raise ValueError(f'Hand board left drawn window during teaching segment {index}')
            joints = frame.get('joints', {}).get('joints_rad')
            if joints is None or len(joints) != 7 or not np.isfinite(joints).all():
                raise ValueError(f'Invalid recorded joints in teaching segment {index}')
            route.append(joints)
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


def compress_recorded_route(route, capture, tolerance_deg=.1):
    """Keep taught turns and capture endpoints; bound joint-space shortcut error."""
    anchors = [i for i, index in enumerate(capture) if index is not None]
    selected = [anchors[0]]
    tolerance = math.radians(tolerance_deg)
    for start, end in zip(anchors, anchors[1:]):
        points = np.asarray(route[start:end+1], dtype=float)
        keep = {0, len(points)-1}
        stack = [(0, len(points)-1)]
        while stack:
            a, b = stack.pop()
            if b-a <= 1:
                continue
            direction = points[b]-points[a]
            denominator = float(direction @ direction)
            inner = points[a+1:b]
            if denominator < 1e-20:
                error = np.max(np.abs(inner-points[a]), axis=1)
            else:
                fraction = np.clip((inner-points[a]) @ direction / denominator, 0, 1)
                error = np.max(np.abs(inner-(points[a]+fraction[:, None]*direction)), axis=1)
            worst = int(np.argmax(error))
            if error[worst] > tolerance:
                middle = a+1+worst
                keep.add(middle)
                stack.extend([(a, middle), (middle, b)])
        selected.extend(start+i for i in sorted(keep) if i)
    return [route[i] for i in selected], [capture[i] for i in selected]


def clamp_recorded_route(route, model, max_correction_deg=.5):
    """Trim small measured overshoot to the SDK soft limit, never a large excursion."""
    lower = np.asarray([joint.lower_rad for joint in model.joints], dtype=float)
    upper = np.asarray([joint.upper_rad for joint in model.joints], dtype=float)
    clipped = []
    largest = 0.
    for point in route:
        q = np.asarray(point, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Invalid recorded seven-axis waypoint')
        limited = np.clip(q, lower+math.radians(.02), upper-math.radians(.02))
        correction = math.degrees(float(np.max(np.abs(limited-q))))
        if correction > max_correction_deg:
            raise ValueError(f'Recorded waypoint exceeds SDK joint limit by {correction:.2f}°')
        largest = max(largest, correction)
        clipped.append(limited.tolist())
    return clipped, largest


def make_plan(dataset, calibration_path, output, max_step_deg=2., margin_px=8.,
              allow_provisional=False, capture_only_visibility=False):
    manifest, samples, calibration, box, window_path = inputs(dataset, calibration_path)
    if not calibration.get('quality_passed') and not allow_provisional:
        raise ValueError('Calibration quality failed; pass --allow-provisional to use the accepted result')
    model = load_model()
    targets = [s['joints_rad'] for s in samples]
    route, capture, trace_path = taught_route(dataset, samples, manifest, calibration,
                                               box, margin_px, capture_only_visibility)
    if capture_only_visibility:
        route, capture = compress_recorded_route(route, capture)
        route, correction_deg = clamp_recorded_route(route, model)
        path = [{'q_rad': route[0], 'capture_sample_index': capture[0]}]
        for previous, target, index in zip(route, route[1:], capture[1:]):
            segment = interpolate(previous, target, max_step_deg)
            path.extend({'q_rad': q, 'capture_sample_index': index if i == len(segment)-1 else None}
                        for i, q in enumerate(segment))
        targets = [point['q_rad'] for point in path if point['capture_sample_index'] is not None]
    else:
        # Check the observed manual route and interpolations between its frames.
        path = check_path(targets[0], route, model, manifest, calibration, box,
                          max_step_deg, margin_px, capture)
        correction_deg = 0.
    files = [dataset/'manifest.json', window_path, calibration_path,
             trace_path,
             *sorted(dataset.glob('sample_[0-9][0-9][0-9][0-9].json'))]
    plan = {'schema': 1, 'kind': 'handeye_auto_collection', 'source_dataset': str(dataset.resolve()),
            'calibration': str(calibration_path.resolve()), 'camera': manifest['camera'],
            'board': manifest['board'], 'window_xyxy': box, 'margin_px': margin_px,
            'max_step_deg': max_step_deg, 'targets_rad': targets, 'waypoints': path,
            'path_source': 'per_frame_teaching_trace',
            'source_sha256': {str(p.resolve()): digest(p) for p in files},
            'collision_checked': False,
            'live_visibility_required': not capture_only_visibility,
            'capture_visibility_required': True,
            'visibility_policy': 'capture_only' if capture_only_visibility else 'continuous',
            'route_compression_max_error_deg': .1 if capture_only_visibility else 0.,
            'recorded_joint_limit_correction_deg': correction_deg,
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


class AutoCollectionView:
    """Optional X11 view; drawing and event pumping never read the camera during motion."""
    TITLE = 'Nero automatic hand-eye calibration'

    def __init__(self, box, total):
        if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
            raise RuntimeError('X11 DISPLAY is missing; log in with ssh -X before using --show')
        self.box = box
        self.total = total
        self.last_image = None
        self.enabled = True
        cv2.namedWindow(self.TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.TITLE, 960, 720)

    def update(self, image, state, count=0, quality=None, board_outline=None):
        if not self.enabled:
            return
        canvas = image.copy()
        x0, y0, x1, y1 = self.box
        cv2.rectangle(canvas, (x0, y0), (x1-1, y1-1), (255, 255, 0), 2)
        if board_outline is not None:
            cv2.polylines(canvas, [np.rint(board_outline).astype(np.int32)], True,
                          (0, 220, 0), 2)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 78), (0, 0, 0), -1)
        cv2.putText(canvas, f'{state} | saved {count}/{self.total}', (12, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
        detail = ('Ctrl+C to stop; cyan: allowed board area' if quality is None else
                  f'corners {quality["corners"]} | reprojection RMS '
                  f'{quality["reprojection_rms_px"]:.3f} px')
        cv2.putText(canvas, detail, (12, 59), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (180, 255, 180), 1, cv2.LINE_AA)
        self.last_image = image.copy()
        try:
            cv2.imshow(self.TITLE, canvas)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.enabled = False
            print(f'X11 preview closed; automatic calibration continues: {exc}', file=sys.stderr)

    def status(self, state, count):
        if self.last_image is not None:
            self.update(self.last_image, state, count)

    def pump(self):
        if self.enabled:
            try:
                cv2.waitKey(1)
            except cv2.error:
                self.enabled = False

    def close(self):
        if self.enabled:
            try:
                cv2.destroyWindow(self.TITLE)
            except cv2.error:
                pass


def require_arm_ready(arm):
    robot = arm.robot
    status = robot.get_arm_status()
    if (status is None or status.msg.arm_status != 0 or status.msg.ctrl_mode != 1
            or robot.get_joints_enable_status_list() != [True]*7 or robot.has_comm_error()):
        raise RuntimeError('Require normal CAN control and seven enabled joints')


def ensure_can_control(arm):
    """Take CAN ownership from a stationary WEB-controlled arm before replay."""
    robot = arm.robot
    status = robot.get_arm_status()
    if status is None:
        raise RuntimeError('No arm status before CAN control switch')
    state = status.msg
    if state.ctrl_mode == 3:
        if (state.arm_status != 0 or state.motion_status != 0
                or robot.get_joints_enable_status_list() != [True]*7
                or robot.has_comm_error()):
            raise RuntimeError('Cannot switch to CAN unless the arm is normal, stationary and enabled')
        robot.set_motion_mode('js')
        deadline = time.monotonic() + 2.
        while time.monotonic() < deadline:
            status = robot.get_arm_status()
            if status is not None and status.msg.ctrl_mode == 1:
                break
            time.sleep(.02)
        else:
            raise TimeoutError('CAN control switch did not receive confirmation within 2s')
    require_arm_ready(arm)


def move_and_watch(arm, camera, detector, manifest, calibration, box, margin_px, target,
                   watch_board=True, pump=None):
    require_arm_ready(arm)
    robot = arm.robot
    last_q = arm.read_joints()['joints_rad']
    robot.move_js(target)
    deadline = time.monotonic() + 30.
    best_error = math.inf
    last_progress = time.monotonic()
    resend_count = 0
    stable_q = None
    stable_since = None
    try:
        while time.monotonic() < deadline:
            if pump is not None:
                pump()
            if watch_board:
                live_board(camera, detector, manifest, calibration, box, margin_px)
            last_q = arm.read_joints()['joints_rad']
            now = time.monotonic()
            error = math.degrees(max(abs(a-b) for a, b in zip(last_q, target)))
            if error < best_error - .05:
                best_error, last_progress = error, now
            if error <= .8:
                if stable_q is None or math.degrees(max(abs(a-b) for a,b in zip(last_q, stable_q))) > .05:
                    stable_q, stable_since = list(last_q), now
                elif now-stable_since >= .3:
                    return
            else:
                stable_q = stable_since = None
                if now-last_progress >= 1.5 and resend_count < 2:
                    robot.move_js(target)
                    resend_count += 1
                    last_progress = now
        raise TimeoutError(f'Automatic calibration waypoint did not settle in 30 s; '
                           f'joint_error={error:.3f}°, resend_count={resend_count}')
    except BaseException:
        robot.move_js(last_q)
        raise


def smooth_route_profile(route, max_speed_deg_s=4., max_acc_deg_s2=6.):
    """Parameterize a taught joint polyline with zero speed at its capture endpoints."""
    points = np.asarray(route, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 7 or len(points) < 2
            or not np.isfinite(points).all() or not math.isfinite(max_speed_deg_s)
            or not math.isfinite(max_acc_deg_s2) or max_speed_deg_s <= 0
            or max_acc_deg_s2 <= 0):
        raise ValueError('Invalid smooth calibration route or speed limits')
    lengths = np.degrees(np.max(np.abs(np.diff(points, axis=0)), axis=1))
    cumulative = np.r_[0., np.cumsum(lengths)]
    distance = float(cumulative[-1])
    # Quintic position has peak normalized velocity 1.875 and acceleration 5.774.
    duration = max(.3, 1.875 * distance / max_speed_deg_s,
                   math.sqrt(5.774 * distance / max_acc_deg_s2))
    return points, cumulative, duration


def smooth_route_target(profile, elapsed):
    points, cumulative, duration = profile
    u = min(1., max(0., elapsed / duration))
    progress = 10*u**3 - 15*u**4 + 6*u**5
    distance = progress * cumulative[-1]
    if distance >= cumulative[-1]:
        return points[-1].tolist()
    i = min(len(points)-2, int(np.searchsorted(cumulative, distance, side='right')-1))
    span = cumulative[i+1] - cumulative[i]
    fraction = 0. if span < 1e-10 else (distance-cumulative[i])/span
    return (points[i] + fraction*(points[i+1]-points[i])).tolist()


def stream_smooth_route(arm, route, max_speed_deg_s=4., max_acc_deg_s2=6., hz=50,
                        pump=None):
    """Stream intermediate targets without stopping; caller settles at sample endpoint."""
    profile = smooth_route_profile(route, max_speed_deg_s, max_acc_deg_s2)
    robot = arm.robot
    # HighGUI/X11 event processing can block unpredictably. Service the window
    # while stationary, never inside the time-critical command stream.
    if pump is not None:
        pump()
    begin = previous_send = time.monotonic()
    last_tick = begin
    phase_elapsed = 0.
    last = list(route[0])
    last_feedback = begin
    feedback_stamp = None
    tick = 0
    try:
        while True:
            now = time.monotonic()
            if tick and now - previous_send > .1:
                raise RuntimeError('Automatic calibration command stream paused over 100 ms '
                                   f'({1000*(now-previous_send):.1f} ms)')
            # A late loop must not skip over intermediate joint targets. This
            # bounds each update while allowing the requested profile to take
            # longer if CAN or the preview window delays a send.
            phase_elapsed = min(profile[2], phase_elapsed +
                                min(max(0., now-last_tick), .4/max_speed_deg_s))
            last_tick = now
            elapsed = phase_elapsed
            target = smooth_route_target(profile, elapsed)
            if math.degrees(max(abs(a-b) for a,b in zip(target, last))) > .5:
                raise RuntimeError('Automatic calibration command jump exceeds 0.5°')
            if robot.has_comm_error():
                raise RuntimeError('CAN feedback error during automatic calibration motion')
            feedback = robot.get_joint_angles()
            if feedback is not None and feedback.timestamp != feedback_stamp:
                feedback_stamp = feedback.timestamp
                last_feedback = now
            if now-last_feedback > 1.:
                raise RuntimeError('Joint feedback stale during automatic calibration motion')
            robot.move_js(target)
            last, previous_send = target, time.monotonic()
            if elapsed >= profile[2]:
                return
            tick += 1
            time.sleep(max(0., begin + tick/hz - time.monotonic()))
    except BaseException:
        feedback = robot.get_joint_angles()
        robot.move_js(list(feedback.msg) if feedback is not None else last)
        raise


def capture_after_settling(arm, camera, detector, frame_check, attempts=3):
    """Retry only a sample rejected because the arm had not finished settling."""
    for attempt in range(attempts):
        try:
            return capture_sample(arm, camera, detector, frame_check=frame_check)
        except ValueError as exc:
            if (str(exc) not in ('Arm moved during capture; sample rejected',
                                 'Joints moved during calibration sample')
                    or attempt == attempts-1):
                raise


def run_plan(plan_path, output, channel, speed_percent, fps=None,
             smooth_speed_deg_s=4., smooth_acc_deg_s2=6., show=False,
             start_home=False, home_path=None):
    plan = json.loads(plan_path.read_text())
    if plan.get('kind') != 'handeye_auto_collection':
        raise ValueError('Not a hand-eye automatic collection plan')
    if any(digest(path) != value for path, value in plan['source_sha256'].items()):
        raise ValueError('Teaching/calibration/window changed after planning')
    from home_start import DEFAULT_HOME, joint_route, load_home, move_home
    home = load_home(home_path or DEFAULT_HOME) if start_home else None
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
    camera = arm = preview = None
    previous_auto_mode = None
    try:
        camera = RealSenseCamera(manifest['camera']['serial'],
                                 image_profile=manifest['board'].get('image_profile'))
        if camera.info != manifest['camera']:
            raise ValueError('Camera serial/intrinsics/crop changed after teaching')
        if show:
            preview = AutoCollectionView(box, len(plan['targets_rad']))
            preview.update(camera.capture(), 'READY', 0)
        arm = NeroFeedback(channel)
        detector = CharucoDetector(manifest['board'])
        capture_only = plan.get('visibility_policy') == 'capture_only'
        if not capture_only and not start_home:
            live_board(camera, detector, manifest, calibration, box, plan['margin_px'])
        ensure_can_control(arm)
        model = load_model()
        current = arm.read_joints()['joints_rad']
        home_route = None
        if start_home:
            # Validate both legs before sending the first HOME motion command.
            home_route = joint_route(current, home, model, plan['max_step_deg'])
            if capture_only:
                entry = joint_route(home, plan['waypoints'][0]['q_rad'], model,
                                    plan['max_step_deg'])
                approach = [{'q_rad': q, 'capture_sample_index': None} for q in entry[1:]]
            else:
                approach = check_path(home, [plan['targets_rad'][0]], model,
                                      manifest, calibration, box,
                                      plan['max_step_deg'], plan['margin_px'])
        elif capture_only:
            deviations = [math.degrees(max(abs(a-b) for a,b in zip(current,point['q_rad'])))
                          for point in plan['waypoints']]
            nearest = int(np.argmin(deviations))
            if deviations[nearest] > 2.:
                raise ValueError('Capture-only replay must start within 2° of a taught waypoint')
            # Retrace the manually taught route backwards from any waypoint to sample zero.
            approach = [{'q_rad': point['q_rad'], 'capture_sample_index': None}
                        for point in reversed(plan['waypoints'][:nearest])]
        else:
            # The current-to-first approach is checked at execution, since its start changes.
            approach = check_path(current, [plan['targets_rad'][0]], model,
                                  manifest, calibration, box,
                                  plan['max_step_deg'], plan['margin_px'])
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
        if start_home:
            actual = move_home(arm, home_route, smooth_speed_deg_s, smooth_acc_deg_s2, preview)
            # Check actual settled feedback as well as the nominal HOME.
            joint_route(actual, home, model, plan['max_step_deg'])
            write_new(output/'home_start.json', {'home_file': str(Path(home_path or DEFAULT_HOME).resolve()),
                                               'joints_rad': home, 'actual_joints_rad': actual,
                                               'collision_checked': False})
        path = approach + (plan['waypoints'] if capture_only else plan['waypoints'][1:])
        count = 0
        pending = []
        for waypoint in path:
            if capture_only:
                pending.append(waypoint['q_rad'])
                if waypoint['capture_sample_index'] is None:
                    continue
                current = arm.read_joints()['joints_rad']
                if preview is not None:
                    preview.status(f'MOVING to sample {count+1}', count)
                stream_smooth_route(arm, [current] + pending,
                                    smooth_speed_deg_s, smooth_acc_deg_s2,
                                    pump=None if preview is None else preview.pump)
                move_and_watch(arm, camera, detector, manifest, calibration, box,
                               plan['margin_px'], waypoint['q_rad'], watch_board=False,
                               pump=None if preview is None else preview.pump)
                pending = []
            else:
                move_and_watch(arm, camera, detector, manifest, calibration, box,
                               plan['margin_px'], waypoint['q_rad'],
                               pump=None if preview is None else preview.pump)
            index = waypoint['capture_sample_index']
            if index is None:
                continue
            def check_frame(image, pose=None):
                live_board(camera, detector, manifest, calibration, box,
                           plan['margin_px'], image=image, pose=pose)
            sample, image, vis = capture_after_settling(arm, camera, detector, check_frame)
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
            if preview is not None:
                preview.update(vis, 'ACCEPTED', count, sample['quality'], points)
            print(json.dumps({'accepted': count, 'source_sample_index': index,
                              **sample['quality']}), flush=True)
        (output/'AUTO_INCOMPLETE.json').unlink()
        return count
    finally:
        if preview is not None:
            preview.close()
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
    q.add_argument('--capture-only-visibility', action='store_true',
                   help='Replay taught joints when board left view in transit; require full board at each sample')
    q = sub.add_parser('run', help='Move the arm and acquire a new dataset')
    q.add_argument('--plan', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--channel', default='can0')
    q.add_argument('--speed-percent', type=int, default=15)
    q.add_argument('--smooth-speed-deg-s', type=float, default=4.)
    q.add_argument('--smooth-acc-deg-s2', type=float, default=6.)
    q.add_argument('--fps', type=int, choices=(6, 15, 30),
                   help='Override only the teaching frame rate; keep resolution and crop')
    q.add_argument('--show', action='store_true', help='Show each accepted calibration frame over X11')
    q.add_argument('--start-from', choices=('home', 'current'), default='home',
                   help='Default: settle at HOME before entering the taught route; current preserves legacy replay')
    q.add_argument('--home', type=Path, help='HOME JSON (default: configs/actions/home.json)')
    q.add_argument('--execute', action='store_true', required=True)
    args = p.parse_args(argv)
    if args.command == 'draw-window':
        draw_window(args.dataset)
    elif args.command == 'plan':
        largest_step = 10 if args.capture_only_visibility else 2
        if not 0 < args.max_step_deg <= largest_step or not 0 <= args.margin_px <= 50:
            raise ValueError('Invalid path step or image margin')
        make_plan(args.dataset, args.calibration, args.output, args.max_step_deg,
                  args.margin_px, args.allow_provisional, args.capture_only_visibility)
    else:
        if not 1 <= args.speed_percent <= 100:
            raise ValueError('SDK speed percent must be 1..100%')
        if (not math.isfinite(args.smooth_speed_deg_s) or args.smooth_speed_deg_s <= 0
                or not math.isfinite(args.smooth_acc_deg_s2)
                or args.smooth_acc_deg_s2 <= 0):
            raise ValueError('Smooth calibration speed/acceleration must be finite and positive')
        run_plan(args.plan, args.output, args.channel, args.speed_percent, args.fps,
                 args.smooth_speed_deg_s, args.smooth_acc_deg_s2, args.show,
                 start_home=args.start_from == 'home', home_path=args.home)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, TimeoutError, OSError, ImportError) as exc:
        print(f'AUTO CALIBRATION STOP: {exc}', file=sys.stderr)
        raise SystemExit(1)
