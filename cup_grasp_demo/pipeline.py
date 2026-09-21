"""Opt-in persistent single-step state machine for the green-cup NERO demo.

Every step is separately committed.  Dry runs never call CAN.  A real motion
first persists an in-flight PAUSED marker; after a crash, only a new scene and
live feedback can reset it.  Evidence of contact, hold, and revealed dice must
come from independent observation, not position feedback.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from cup_grasp_demo import execute, pipeline_state
from nero_revo2_control.kinematics import load_model


DEFAULT_CONFIG = Path(__file__).resolve().parent/'config'/'green_pipeline.json'
OBSERVATION_MAX_AGE_S = 15.
HOME_START_TOLERANCE_DEG = .5
MOTION_PHASES = ('HOME', 'APPROACH', 'CONTACT', 'GRIP', 'SHAKE', 'REVEAL')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_atomic(path, value, *, first=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         indent=2).encode('utf-8')+b'\n'
    if first:
        with path.open('xb') as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        return
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with temporary.open('xb') as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _locked_state(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name+'.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _load_config(path):
    path = Path(path)
    config = json.loads(path.read_bytes())
    if (config.get('schema') != 1 or
            config.get('kind') != 'nero_green_cup_pipeline_config'):
        raise ValueError('Invalid green-pipeline configuration')
    home = config.get('home_candidate') or {}
    waypoints = home.get('waypoints_deg')
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError('HOME candidate needs ordered single-step waypoints')
    initial = home.get('initial_joints_deg')
    for joints in [initial, *waypoints]:
        if (not isinstance(joints, list) or len(joints) != 7 or
                any(type(v) not in (int, float) or not math.isfinite(v)
                    for v in joints)):
            raise ValueError('HOME requires finite seven-joint targets')
    if any(abs(item[2]-home['selected_j3_deg']) > .001 for item in waypoints):
        raise ValueError('HOME must preserve the selected J3 angle')
    if home.get('speed_percent') != 1 or not 10 <= home.get('timeout_s', 0) <= 90:
        raise ValueError('HOME requires bounded 1% speed and 10..90 s timeout')
    # The shipped config sits in cup_grasp_demo/config; data paths are relative
    # to the package directory, so diagnostic pre-HOME frames cannot be reused.
    return config, _sha256(path), path.parent.parent


def _load_state(path, config_sha):
    state = json.loads(Path(path).read_bytes())
    if state.get('config_sha256') != config_sha:
        raise ValueError('State belongs to a different pipeline configuration')
    return state


def _relative(base, name, scene_id=None):
    if '{scene}' in str(name):
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError('Artifact path needs a current scene identity')
        token = hashlib.sha256(scene_id.encode('utf-8')).hexdigest()[:12]
        name = str(name).replace('{scene}', token)
    value = Path(name)
    return value if value.is_absolute() else (base/value).resolve()


def _age_ns(then, now_ns):
    if type(then) is not int or then <= 0:
        raise ValueError('Observation needs a positive host ns timestamp')
    age = (now_ns-then)/1e9
    if not 0 <= age <= OBSERVATION_MAX_AGE_S:
        raise ValueError('Scene/review observation is stale or in the future')
    return age


def _scene_evidence(evidence, *, now_ns, require_cup_still=True):
    _age_ns(evidence.get('observed_at_ns'), now_ns)
    image = evidence.get('path_scene_image_path', evidence.get('scene_image_path'))
    image_sha = evidence.get('path_scene_sha256', evidence.get('scene_image_sha256'))
    if not image or _sha256(image) != image_sha:
        raise ValueError('Fresh scene image and SHA-256 must match')
    if (evidence.get('people_clear') is not True or
            (require_cup_still and evidence.get('cup_still') is not True)):
        raise ValueError('People-clear and expected cup-state observations are required')
    if not isinstance(evidence.get('scene_id'), str) or not evidence['scene_id']:
        raise ValueError('Scene observation requires a nonempty identity')
    return {'scene_id': evidence['scene_id'],
            'observed_at_ns': evidence['observed_at_ns'],
            'scene_image_sha256': image_sha,
            'path_scene_sha256': image_sha}


def _home_review(config, state, evidence, index, *, now_ns):
    """Bind next move to fresh visual review; occluded cup location stays unknown."""
    home = config['home_candidate']
    expected = home['initial_joints_deg'] if index == 0 else home['waypoints_deg'][index-1]
    target = home['waypoints_deg'][index]
    _age_ns(evidence.get('observed_at_ns'), now_ns)
    scene_image = evidence.get('scene_image_path')
    if (not scene_image or evidence.get('scene_id') != state['scene_id'] or
            evidence.get('home_stage_index') != index or
            evidence.get('scene_image_sha256') != _sha256(scene_image) or
            evidence.get('people_clear') is not True or
            evidence.get('cup_still') is not True or
            evidence.get('path_obstacles_clear') is not True or
            evidence.get('path_reviewed') is not True or
            evidence.get('cup_geometry_available') is not False or
            evidence.get('cup_collision_unresolved') is not True):
        raise ValueError('HOME needs fresh unchanged scene, reviewed path, and honest unresolved cup geometry')
    if (evidence.get('start_joints_deg') != expected or
            evidence.get('target_joints_deg') != target):
        raise ValueError('HOME review needs the exact next configured start/target')
    table_z = evidence.get('observed_table_z_base_m')
    if (type(table_z) not in (int, float) or not math.isfinite(table_z)
            or not -.05 <= table_z <= .1 or
            evidence.get('table_plane_source') != 'prior_same_fixed_table_rgbd'):
        raise ValueError('HOME needs an observed fixed-table plane in arm base')
    if index < 6 and evidence.get('outward_j2_uplift_reviewed') is not True:
        raise ValueError('HOME outward J2 disengagement must be scene-reviewed')
    return expected, target, float(table_z)


def _hand_table_path(model, q_path, hand_positions, table_z, *, require_uplift):
    """Exact official Revo2 STL vertex table minimum on joint-linear samples."""
    from cup_grasp_demo.hand_geometry import RightRevo2Model
    from cup_grasp_demo.top_execute import hand_urdf_angles

    right_hand = RightRevo2Model()
    angles = hand_urdf_angles(hand_positions)
    heights = []
    for q in q_path:
        transforms = right_hand.link_transforms_from_flange(
            np.asarray(model.fk(q), dtype=float), joint_positions_rad=angles)
        floor = math.inf
        for link, (_, _, vertices, collision_tf, _) in right_hand.collision.items():
            transform = transforms[link] @ collision_tf
            z = vertices @ transform[2, :3]+transform[2, 3]
            floor = min(floor, float(np.min(z)) - table_z)
        heights.append(floor)
    minimum = min(heights)
    if minimum < .005:
        raise ValueError(f'HOME official Revo2 collision STL reaches table margin: {minimum:.3f} m')
    if require_uplift and (heights[-1] < heights[0]+.001 or
                           any(b < a-.002 for a, b in zip(heights[:-1], heights[1:]))):
        raise ValueError('HOME J2 step does not monotonically lift hand from table')
    return {'sample_count': len(heights),
            'min_exact_hand_mesh_table_m': minimum,
            'outward_uplift_model_passed': bool(require_uplift),
            'cup_collision_resolved': False,
            'model_sources_sha256': [item['sha256'] for item in
                                     right_hand.provenance['model_sources']]}


def _control_prefix(config, python):
    script = Path(__file__).resolve().parents[1]/'nero_revo2_control'/'nero_revo2_demo.py'
    return execute._prefix(python, script, config['channel'])


def _home_motion(config, state, evidence, *, python, runner, now_ns):
    index = state['home_move_count']
    waypoints = config['home_candidate']['waypoints_deg']
    if index >= len(waypoints):
        raise ValueError('All candidate HOME steps already completed')
    expected, target, table_z = _home_review(
        config, state, evidence, index, now_ns=now_ns)
    prefix = _control_prefix(config, python)
    status = execute.read_arm_status(prefix, runner=runner)
    live_q, sdk = execute.read_joint_limits(prefix, runner=runner)
    for axis, (angle, reference, goal, limits) in enumerate(zip(
            live_q, expected, target, sdk), 1):
        actual = math.degrees(angle)
        if abs(actual-reference) > HOME_START_TOLERANCE_DEG:
            raise ValueError(f'HOME J{axis} live angle differs from reviewed start')
        if not limits[0] <= goal <= limits[1]:
            raise ValueError(f'HOME J{axis} target exceeds current SDK limits')
    if status.get('joints_enabled') != [True]*7:
        raise ValueError('HOME needs seven enabled joints')
    model = load_model()
    path = model.check_joint_path(live_q, [math.radians(v) for v in target],
                                  max_sample_step_rad=math.radians(.5))
    if not path.joint_limits_passed or not path.continuity_passed:
        raise ValueError('HOME independent URDF joint path failed')
    hand_positions = execute.read_hand_status(prefix, runner=runner)
    screen = _hand_table_path(model, path.samples_rad, hand_positions,
                              table_z, require_uplift=index < 6)
    # The cup is occluded, so this review is a table/hand disengagement screen.
    # The scene must be watched during each low-speed single-axis step.
    command = [*prefix, 'move-j', '--joints-deg', *[str(v) for v in target],
               '--speed', '1', '--execute', '--timeout',
               str(config['home_candidate']['timeout_s'])]
    events = execute._run_control(command, timeout=config['home_candidate']['timeout_s']+20,
                                  runner=runner)
    arrival = next((event for event in events if event.get('event') == 'arm_target_reached'
                    and event.get('command') == 'move-j'), None)
    if arrival is None:
        raise RuntimeError('HOME move-j lacks fresh target-arrival feedback')
    settled = execute.read_arm_status(prefix, runner=runner)
    q_after, _ = execute.read_joint_limits(prefix, runner=runner)
    for axis, (actual, goal) in enumerate(zip(q_after, target), 1):
        if abs(math.degrees(actual)-goal) > HOME_START_TOLERANCE_DEG:
            raise RuntimeError(f'HOME J{axis} did not arrive at reviewed target')
    return {'target_joints_rad': [math.radians(v) for v in target],
            'arrival_joints_rad': list(q_after), 'feedback_event': 'arm_target_reached',
            'observed_at_ns': time.time_ns(),
            'final_home_target': index == len(waypoints)-1,
            'reviewed_final_home_joints_rad':
            [math.radians(v) for v in waypoints[-1]] if index == len(waypoints)-1 else None,
            'arm_status': 0, 'ctrl_mode': 1,
            'joints_enabled': settled.get('joints_enabled'),
            'scene_id': state['scene_id'],
            'home_stage_index': index,
            'arrival_flange_m_rad': settled.get('flange_m_rad'),
            'table_hand_path_screen': screen,
            'cup_collision_unresolved': True}


def _snapshot(config, base, state, evidence, *, camera_python, python, runner):
    user_home = state.get('home_source') == 'user_designated_current_pose_read_only'
    if state.get('home_arrival_time_ns') is None:
        raise ValueError('Formal RGB-D capture requires verified current HOME')
    if user_home:
        if state.get('home_move_count') != 0:
            raise ValueError('User-designated HOME cannot contain waypoint motion')
        home_joints = state.get('home_arrival_joints_rad')
        if not isinstance(home_joints, list) or len(home_joints) != 7:
            raise ValueError('User-designated HOME has no seven-joint arrival')
        expected = [math.degrees(value) for value in home_joints]
    else:
        if state['home_move_count'] != len(config['home_candidate']['waypoints_deg']):
            raise ValueError('Formal RGB-D capture requires all current HOME steps')
        expected = config['home_candidate']['waypoints_deg'][-1]
    _age_ns(evidence.get('observed_at_ns'), time.time_ns())
    if (evidence.get('scene_id') != state['scene_id'] or
            not evidence.get('scene_image_path') or
            _sha256(evidence['scene_image_path']) != evidence.get('scene_image_sha256') or
            evidence.get('people_clear') is not True or
            evidence.get('cup_still') is not True):
        raise ValueError('Formal RGB-D capture needs a fresh people-clear, cup-still scene image')
    prefix = _control_prefix(config, python)
    execute.read_arm_status(prefix, runner=runner)
    q_before, _ = execute.read_joint_limits(prefix, runner=runner)
    for actual, target in zip(q_before, expected):
        if abs(math.degrees(actual)-target) > HOME_START_TOLERANCE_DEG:
            raise ValueError('Seven-joint live pose changed after HOME arrival')
    directory = _relative(base, config['snapshot_dir'], state['scene_id'])
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Snapshot directory already contains a frame; use a new run')
    command = [camera_python, '-m', 'cup_grasp_demo.grasp', 'snapshot', '--dataset',
               str(directory), '--serial', config['camera_serial'],
               '--warmup-frames', str(config['snapshot_warmup_frames']),
               '--timeout-ms', str(config['snapshot_timeout_ms'])]
    if not 3000 <= config['snapshot_timeout_ms'] <= 10000:
        raise ValueError('Snapshot timeout must be 3000..10000 ms')
    result = runner(command, capture_output=True, text=True, timeout=45, check=False)
    if result.returncode:
        raise RuntimeError(f'RGB-D snapshot failed: {result.stderr.strip()}')
    metadata_path = directory/'metadata.json'
    metadata = json.loads(metadata_path.read_bytes())
    workspace = _relative(base, config['red_workspace'])
    roi, roi_sha = _red_mat_roi(directory/'color.png',
                                workspace_config_path=workspace)
    execute.read_arm_status(prefix, runner=runner)
    q_after, _ = execute.read_joint_limits(prefix, runner=runner)
    for actual, target in zip(q_after, expected):
        if abs(math.degrees(actual)-target) > HOME_START_TOLERANCE_DEG:
            raise ValueError('Seven-joint live pose changed during RGB-D capture')
    return {'frame_id': metadata['frame_id'],
            'metadata_sha256': _sha256(metadata_path),
            'color_sha256': _sha256(directory/'color.png'),
            'depth_sha256': _sha256(directory/'depth.npz'),
            'captured_at_ns': metadata['host_capture_time_ns'],
            'home_joints_at_capture_rad': list(q_after),
            'home_feedback_checked_at_ns': time.time_ns(),
            'target_red_roi_xyxy': roi,
            'target_red_roi_sha256': roi_sha,
            'red_workspace_config_sha256': _sha256(workspace),
            'path_scene_sha256': evidence.get('path_scene_sha256',
                                              evidence.get('scene_image_sha256')),
            'snapshot_dir': str(directory)}


def _red_mat_roi(color_path, reviewed_bbox=None, *, workspace_config_path=None):
    """Bind target pixels to the red mat, independent of changing blue mat."""
    import cv2

    image = cv2.imread(str(color_path))
    if image is None:
        raise ValueError('Missing formal RGB-D color image for red target ROI')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = ((cv2.inRange(hsv, (0, 80, 60), (12, 255, 255)) != 0) |
           (cv2.inRange(hsv, (168, 80, 60), (179, 255, 255)) != 0))
    height, width = image.shape[:2]
    workspace = None
    if workspace_config_path is not None:
        from dice_cup_localization.red_workspace import RedWorkspace
        workspace = RedWorkspace(workspace_config_path, image.shape)
        x0, y0 = workspace.vertices.min(axis=0)
        x1, y1 = workspace.vertices.max(axis=0)+1
        bbox = [int(x0), int(y0), int(x1), int(y1)]
    elif reviewed_bbox is None:
        count, _, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8))
        if count < 2:
            raise ValueError('Red mat has no connected region in formal color frame')
        winner = stats[1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
        x, y, w, h, area = [int(v) for v in winner]
        if area < 5000:
            raise ValueError('Red mat region is too small for cup-target ROI')
        bbox = [x, y, x+w, y+h]
    else:
        bbox = reviewed_bbox
    if (not isinstance(bbox, list) or len(bbox) != 4 or
            any(type(v) is not int for v in bbox)):
        raise ValueError('Red target ROI needs four integer color pixels')
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height and
            (x1-x0)*(y1-y0) >= 5000):
        raise ValueError('Red target ROI is outside image or too small')
    if workspace is None and float(red[y0:y1, x0:x1].mean()) < .2:
        raise ValueError('Reviewed target ROI does not mostly cover red mat')
    if workspace is not None:
        red_pixels = red & workspace.mask
        if int(red_pixels.sum()) < 5000:
            raise ValueError('Configured red workspace contains too few red pixels')
        payload = (workspace.sha256.encode('ascii')+
                   np.flatnonzero(red_pixels).astype('<i4').tobytes()+
                   np.ascontiguousarray(image[red_pixels]).tobytes())
    else:
        payload = (json.dumps(bbox).encode('utf-8')+
                   np.ascontiguousarray(image[y0:y1, x0:x1]).tobytes())
    sha = hashlib.sha256(payload).hexdigest()
    return bbox, sha


def _localize_yolo(config, base, state, evidence, *, camera_python, runner,
                   now_ns):
    """Use the existing exact cap=0 YOLO+RGB-D module on formal HOME frame."""
    _scene_evidence(evidence, now_ns=now_ns)
    if evidence['scene_id'] != state['scene_id']:
        raise ValueError('Cup scene changed since formal RGB-D capture')
    snapshot = _relative(base, config['snapshot_dir'], state['scene_id'])
    metadata = json.loads((snapshot/'metadata.json').read_bytes())
    if metadata['frame_id'] != state['snapshot_frame_id']:
        raise ValueError('Localization snapshot differs from current formal frame')
    bbox = state.get('target_red_roi_xyxy')
    target_sha = state.get('target_red_roi_sha256')
    if bbox is None or target_sha is None:
        raise ValueError('Formal frame lacks bound red-mat target ROI')
    workspace = _relative(base, config['red_workspace'])
    measured_bbox, measured_sha = _red_mat_roi(
        snapshot/'color.png', workspace_config_path=workspace)
    if measured_sha != target_sha:
        raise ValueError('Formal red target ROI changed since capture')
    model_name = config.get('yolo_model')
    if not model_name or config.get('yolo_model_profile') != 'dice_cap2':
        raise ValueError('No verified cap=0 YOLO model/profile configured')
    model = _relative(base, model_name)
    if _sha256(model) != 'c45d2b7fa61c45c5ef608cabbefd69fd9befe19243a14685e69b2ad645505ec0':
        raise ValueError('Green-cap model SHA differs from verified two-class weight')
    recognition_dir = _relative(base, config['recognition_dir'], state['scene_id'])
    if recognition_dir.exists():
        raise ValueError('Recognition directory already exists; do not reuse an old result')
    script = Path(__file__).resolve().parents[1]/'dice_cup_localization'/'recognize.py'
    command = [camera_python, str(script), '--snapshot', str(snapshot),
               '--model', str(model), '--model-profile', 'dice_cap2',
               '--candidate-classes', '0',
               '--red-workspace', str(workspace),
               '--output', str(recognition_dir)]
    ort = config.get('ort_package_dir')
    if ort:
        command.extend(('--ort-package-dir', str(_relative(base, ort))))
    center_gate = config.get('geometry_max_center_spread_mm')
    if center_gate is not None:
        if (type(center_gate) not in (int, float) or
                not math.isfinite(center_gate) or not 5. <= center_gate <= 6.):
            raise ValueError('Reviewed side-section center spread must be 5..6 mm')
        command.extend(('--max-center-spread-mm', str(center_gate)))
    result = runner(command, capture_output=True, text=True,
                    timeout=90, check=False)
    if result.returncode:
        raise RuntimeError('Verified YOLO cap localization failed: '
                           +result.stderr.strip())
    recognition = json.loads((recognition_dir/'recognition.json').read_bytes())
    geometry = json.loads((recognition_dir/'geometry.json').read_bytes())
    if (geometry.get('red_workspace_config_sha256') != _sha256(workspace)
            or recognition.get('red_workspace', {}).get('config_sha256') !=
            _sha256(workspace)):
        raise ValueError('YOLO target departed from verified red workspace')
    from cup_grasp_demo.yolo_top_adapter import _source_provenance
    top = _source_provenance(geometry, recognition, metadata)
    from nero_calibration.core import matrix
    transform = matrix(json.loads(_relative(base, config['calibration']).read_bytes())
                       ['T_base_camera'])
    point = np.asarray(top['center_m'], dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('YOLO visible-top center must be finite color_optical meters')
    base_center = transform[:3, :3] @ point+transform[:3, 3]
    return {'frame_id': metadata['frame_id'],
            'top_center_camera_m': point.tolist(),
            'top_center_base_m': base_center.tolist(),
            'top_quality': {'valid': True,
                            'definition': top['center_definition'],
                            'measurements': top['quality'],
                            'source_model_sha256': geometry['model_sha256']},
            'recognition_sha256': _sha256(recognition_dir/'recognition.json'),
            'geometry_sha256': _sha256(recognition_dir/'geometry.json')}


def _read_control_event(prefix, command, wanted, *, runner):
    events = execute._run_control([*prefix, command], timeout=15, runner=runner)
    item = next((entry for entry in events if entry.get('event') == wanted), None)
    if item is None:
        raise ValueError(f'No fresh read-only {wanted} feedback')
    return item


def _trace_file(root, recorded_path):
    candidate = Path(recorded_path)
    if candidate.is_file() or candidate.is_dir():
        return candidate
    local = root/candidate.name
    if local.is_file() or local.is_dir():
        return local
    raise ValueError(f'Historical trace asset missing: {candidate.name}')


def _historical_scene_evidence(item, trace_dir):
    if (item.get('people_clear') is not True or
            item.get('cup_still') is not True or
            item.get('path_obstacles_clear') is not True or
            not isinstance(item.get('scene_id'), str) or not item['scene_id']):
        raise ValueError('Existing HOME needs reviewed historical clear scene')
    photo = _trace_file(trace_dir, item['photo_path'])
    if _sha256(photo) != item.get('photo_sha256'):
        raise ValueError('Historical pre-HOME scene photo SHA differs')
    when = item.get('observed_at_ns')
    if type(when) is not int or when <= 0:
        raise ValueError('Historical pre-HOME photo needs positive host time')
    if abs(photo.stat().st_mtime_ns-when) > 30_000_000_000:
        raise ValueError('Historical pre-HOME photo/time differ by over 30 s')
    return {'scene_id': item['scene_id'], 'observed_at_ns': when,
            'scene_image_sha256': item['photo_sha256'],
            'path_scene_sha256': item['photo_sha256']}


def _official_home_transcript(item, trace_dir, *, expected_start, target,
                              source_thread_id):
    file = _trace_file(trace_dir, item['command_stdout_path'])
    if _sha256(file) != item.get('stdout_sha256'):
        raise ValueError('Existing HOME transcript SHA differs')
    transcribed = item.get('stdout_provenance') == \
        'post_run_transcription_from_Codex_tool_result'
    if transcribed:
        if not source_thread_id:
            raise ValueError('Transcribed HOME result needs source Codex task identity')
    else:
        if item.get('stdout_provenance') != 'original_command_stdout':
            raise ValueError('HOME transcript provenance must say original or transcribed')
    events = []
    for line in file.read_text(encoding='utf-8').splitlines():
        if line.strip():
            events.append(json.loads(line))
    plans = [entry for entry in events if entry.get('event') == 'arm_plan']
    reached = [entry for entry in events if entry.get('event') == 'arm_target_reached']
    if len(plans) != 1 or len(reached) != 1:
        raise ValueError('Each existing HOME transcript needs exactly one plan and one fresh arrival')
    plan, arrival = plans[0], reached[0]
    if (plan.get('command') != 'move-j' or plan.get('execute') is not True
            or plan.get('speed_percent') != 1 or
            arrival.get('command') != 'move-j' or
            not isinstance(arrival.get('fresh_samples'), int) or
            arrival['fresh_samples'] < 10):
        raise ValueError('Existing HOME transcript is not a 1% executed fresh move-j')
    if transcribed:
        actual_target = plan.get('requested_target_deg_transcribed')
        if (not isinstance(actual_target, list) or len(actual_target) != 7 or
                max(abs(float(a)-goal) for a, goal in zip(actual_target, target)) > .05):
            raise ValueError('Post-run transcribed HOME target differs from configured step')
    else:
        actual_start = plan.get('current_joints_rad')
        actual_target = plan.get('flange_target')
        if (plan.get('target_frame') != 'joint' or
                not isinstance(actual_start, list) or len(actual_start) != 7 or
                not isinstance(actual_target, list) or len(actual_target) != 7):
            raise ValueError('Original HOME plan lacks exact seven-joint start/target')
        for actual, reference, goal, planned in zip(actual_start, expected_start,
                                                   target, actual_target):
            if (abs(math.degrees(actual)-reference) > .5 or
                    abs(math.degrees(planned)-goal) > .05):
                raise ValueError('Original HOME plan differs from configured single-step path')
    return plan, arrival


def record_existing_home(state_path, config_path, manifest_path,
                         formal_snapshot_dir, *, python=sys.executable,
                         runner=subprocess.run):
    """Adopt already completed nine HOME moves and formal capture without CAN TX."""
    config, identity, base = _load_config(config_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    trace_dir = manifest_path.parent
    if (manifest.get('schema') != 1 or
            manifest.get('kind') != 'read_only_existing_home_evidence' or
            manifest.get('config_sha256') != identity):
        raise ValueError('Existing HOME manifest/config identity differs')
    with _locked_state(state_path):
        state = _load_state(state_path, identity)
        if state['phase'] != 'WAIT_SCENE_CLEAR' or state['events']:
            raise ValueError('Existing HOME import needs a fresh empty state')
        scene = _historical_scene_evidence(manifest['scene'], trace_dir)
        steps = manifest.get('steps')
        waypoints = config['home_candidate']['waypoints_deg']
        if not isinstance(steps, list) or len(steps) != len(waypoints):
            raise ValueError('Existing HOME trace needs every configured step')
        replay = pipeline_state.apply_event(state, 'scene_clear', scene)
        previous_time = scene['observed_at_ns']
        prior = config['home_candidate']['initial_joints_deg']
        for index, (item, target) in enumerate(zip(steps, waypoints)):
            if item.get('index') != index+1:
                raise ValueError('Existing HOME transcript order differs from configuration')
            _official_home_transcript(
                item, trace_dir, expected_start=prior, target=target,
                source_thread_id=manifest.get('source_thread_id'))
            photo = _trace_file(trace_dir, item['photo_path'])
            if _sha256(photo) != item.get('photo_sha256'):
                raise ValueError('Existing HOME stage scene photo SHA differs')
            when = item.get('observed_at_ns')
            if (type(when) is not int or when <= previous_time or
                    abs(photo.stat().st_mtime_ns-when) > 30_000_000_000):
                raise ValueError('Existing HOME stage photo/time/order differ')
            arrival_deg = item.get('arrival_joints_deg')
            if (not isinstance(arrival_deg, list) or len(arrival_deg) != 7 or
                    any(type(v) not in (int, float) or not math.isfinite(v)
                        for v in arrival_deg) or
                    max(abs(a-b) for a, b in zip(arrival_deg, target)) > .5):
                raise ValueError('Historical HOME arrival is not within 0.5° target')
            final = index == len(waypoints)-1
            evidence = {'target_joints_rad': [math.radians(v) for v in target],
                        'arrival_joints_rad': [math.radians(v) for v in arrival_deg],
                        'feedback_event': 'arm_target_reached',
                        'observed_at_ns': when, 'final_home_target': final,
                        'stdout_sha256': item['stdout_sha256'],
                        'photo_sha256': item['photo_sha256'],
                        'transcribed_from_tool_result':
                        item['stdout_provenance'] ==
                        'post_run_transcription_from_Codex_tool_result'}
            if final:
                evidence.update(reviewed_final_home_joints_rad=
                                [math.radians(v) for v in waypoints[-1]],
                                arm_status=0, ctrl_mode=1,
                                joints_enabled=[True]*7)
            replay = pipeline_state.apply_event(replay, 'home_move_reached', evidence)
            previous_time, prior = when, target
        # The final live state, read now, must corroborate the imported terminal
        # HOME. It cannot corroborate historical intermediates; their source is
        # clearly labeled original stdout or Codex-result transcription.
        prefix = _control_prefix(config, python)
        execute.read_arm_status(prefix, runner=runner)
        live_q, _ = execute.read_joint_limits(prefix, runner=runner)
        if max(abs(math.degrees(actual)-goal) for actual, goal in
               zip(live_q, waypoints[-1])) > .5:
            raise ValueError('Current live CAN joints do not match adopted final HOME')
        formal = _trace_file(trace_dir, formal_snapshot_dir or
                             manifest['formal_snapshot_dir'])
        metadata_path = formal/'metadata.json'
        metadata = json.loads(metadata_path.read_bytes())
        if (metadata.get('schema') != 1 or
                metadata.get('serial') != config['camera_serial'] or
                metadata.get('host_capture_time_ns', 0) <= previous_time or
                _sha256(formal/'color.png') != metadata.get('sha256_color') or
                _sha256(formal/'depth.npz') != metadata.get('sha256_depth')):
            raise ValueError('Existing formal RGB-D is not a valid post-HOME frame')
        workspace = _relative(base, config['red_workspace'])
        bbox, roi_sha = _red_mat_roi(formal/'color.png',
                                    workspace_config_path=workspace)
        capture_scene = manifest.get('capture_scene') or {
            'photo_path': manifest.get('capture_scene_photo_path'),
            'photo_sha256': manifest.get('capture_scene_photo_sha256'),
            'people_clear': manifest.get('capture_people_clear'),
            'path_obstacles_clear': manifest.get('capture_scene_path_obstacles_clear')}
        photo = _trace_file(trace_dir, capture_scene.get('photo_path', ''))
        if (not photo.is_file() or _sha256(photo) !=
                capture_scene.get('photo_sha256') or
                capture_scene.get('people_clear') is not True or
                capture_scene.get('path_obstacles_clear') is not True):
            raise ValueError('Formal RGB-D import needs a verified clear path scene photo')
        destination = _relative(base, config['snapshot_dir'], scene['scene_id'])
        if destination.exists():
            raise ValueError('Formal state snapshot destination already exists')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(formal, destination)
        captured = {'frame_id': metadata['frame_id'],
                    'metadata_sha256': _sha256(destination/'metadata.json'),
                    'color_sha256': _sha256(destination/'color.png'),
                    'depth_sha256': _sha256(destination/'depth.npz'),
                    'captured_at_ns': metadata['host_capture_time_ns'],
                    'target_red_roi_xyxy': bbox,
                    'target_red_roi_sha256': roi_sha,
                    'red_workspace_config_sha256': _sha256(workspace),
                    'path_scene_sha256': capture_scene['photo_sha256'],
                    'existing_formal_source_dir': str(formal)}
        replay = pipeline_state.apply_event(replay, 'rgbd_captured', captured)
        _write_atomic(state_path, replay)
    return {'event': 'existing_home_and_capture_recorded',
            'phase': replay['phase'], 'home_steps': len(steps),
            'frame_id': replay['snapshot_frame_id'],
            'CAN_command_sent': False,
            'historical_stdout_provenance':
            'transcribed_from_codex_tool_result' if any(
                item['stdout_provenance'] ==
                'post_run_transcription_from_Codex_tool_result' for item in steps)
            else 'original_command_stdout'}


def _user_home_scene(evidence, state, *, now_ns):
    """Require an externally reviewed red-cup and path scene, never inferred q."""
    observed = _scene_evidence(evidence, now_ns=now_ns)
    image = Path(evidence.get('path_scene_image_path',
                              evidence.get('scene_image_path')))
    if abs(image.stat().st_mtime_ns-observed['observed_at_ns']) > 5_000_000_000:
        raise ValueError('Current HOME external photo timestamp differs from review')
    if (evidence.get('red_cup_on_red_mat_observed') is not True
            or evidence.get('path_obstacles_clear') is not True):
        raise ValueError('Current HOME needs reviewed red cup on red mat and clear full arm path')
    if (state['scene_id'] is not None
            and observed['scene_id'] != state['scene_id']):
        raise ValueError('Current HOME scene identity changed; pause and reset')
    return observed


def _current_home_read(prefix, *, runner):
    """Read status and seven angles; do not call helpers that require CAN mode."""
    status = _read_control_event(prefix, 'status', 'arm_status', runner=runner)
    joints = _read_control_event(prefix, 'read-joints', 'read_joints', runner=runner)
    mode = joints.get('ctrl_mode')
    rendered = status.get('status')
    if (type(mode) is not int or mode not in (1, 3)
            or joints.get('arm_status') != 0
            or joints.get('joints_enabled') != [True]*7
            or status.get('joints_enabled') != [True]*7
            or not isinstance(rendered, str)
            or not re.search(r'arm_status:\s*NORMAL\(0x0\)', rendered)
            or not re.search(r'ctrl_mode:\s*\w+\(0x%x\)' % mode, rendered)):
        raise ValueError('Current HOME read needs NORMAL, matching CAN/WEB mode, all seven enabled')
    q = execute._joints_rad(joints.get('joints_rad'), 'current HOME q')
    return q, mode


def adopt_current_home(state_path, config_path, evidence, *, python=sys.executable,
                       runner=subprocess.run, now_ns=None,
                       sleep_fn=time.sleep, clock_fn=time.time_ns):
    """Designate live stationary pose as HOME; no move-j or other CAN TX."""
    config, identity, _ = _load_config(config_path)
    now_ns = clock_fn() if now_ns is None else int(now_ns)
    with _locked_state(state_path):
        state = _load_state(state_path, identity)
        phase = state['phase']
        if phase not in ('WAIT_SCENE_CLEAR', 'HOME', 'WAIT_CONTROL_HANDOFF'):
            raise ValueError('Current HOME adoption needs empty WAIT, HOME or WEB handoff')
        if (phase == 'HOME' and (state['home_move_count'] != 0
                                 or state.get('home_source') is not None)):
            raise ValueError('Current HOME cannot overwrite an already started waypoint HOME')
        observed = _user_home_scene(evidence, state, now_ns=now_ns)
        replay = state
        if phase == 'WAIT_SCENE_CLEAR':
            replay = pipeline_state.apply_event(replay, 'scene_clear', observed)
        prefix = _control_prefix(config, python)
        first, first_mode = _current_home_read(prefix, runner=runner)
        sleep_fn(.2)
        second, second_mode = _current_home_read(prefix, runner=runner)
        if (first_mode != second_mode or
                max(abs(a-b) for a, b in zip(first, second)) > math.radians(.1)):
            raise ValueError('Current HOME double feedback changed mode or more than 0.1 degree')
        if (phase == 'WAIT_CONTROL_HANDOFF' and
                max(abs(a-b) for a, b in zip(
                    second, state['user_designated_home_joints_rad'])) >
                math.radians(.1)):
            paused = pipeline_state.apply_event(state, 'pause', {
                'reason': 'Arm moved after user designated current HOME; fresh scene/reset required',
                'command_may_have_been_sent': False})
            _write_atomic(state_path, paused)
            raise ValueError('WEB/CAN handoff moved the user designated HOME; run reset')
        read_time_ns = clock_fn()
        if read_time_ns <= observed['observed_at_ns']:
            raise ValueError('Current HOME reads must follow external scene observation')
        home_evidence = {'first_joints_rad': list(first),
                         'second_joints_rad': list(second),
                         'observed_at_ns': read_time_ns,
                         'scene_photo_sha256': observed['path_scene_sha256'],
                         'arm_status': 0, 'ctrl_mode': second_mode,
                         'joints_enabled': [True]*7,
                         'read_only_commands': ['status', 'read-joints',
                                                'status', 'read-joints'],
                         'path_obstacles_clear': True,
                         'red_cup_on_red_mat_observed': True}
        if phase == 'WAIT_CONTROL_HANDOFF':
            if second_mode == 3:
                return {'event': 'user_home_control_handoff_pending',
                        'phase': phase, 'CAN_command_sent': False,
                        'current_joints_deg': [math.degrees(v) for v in second],
                        'user_designated_joints_deg': [math.degrees(v) for v in
                                                       state['user_designated_home_joints_rad']]}
            event = 'home_control_verified'
        else:
            event = 'user_home_designated'
            home_evidence['user_instruction'] = 'current_pose_as_home'
        next_state = pipeline_state.apply_event(replay, event, home_evidence)
        _write_atomic(state_path, next_state)
    return {'event': event, 'phase': next_state['phase'],
            'CAN_command_sent': False, 'home_source': next_state['home_source'],
            'current_joints_deg': [math.degrees(v) for v in second],
            'home_arrival_time_ns': next_state['home_arrival_time_ns']}


def _reviewed_plan_inputs(config, base, state, evidence):
    """Take a current-frame review without changing the HOME config hash."""
    dynamic = any(key in evidence for key in (
        'palm_orientation_path', 'reveal_exit_lane_path', 'pretop_height_m'))
    if dynamic:
        if (evidence.get('orientation_reviewed') is not True
                or evidence.get('exit_lane_reviewed') is not True
                or evidence.get('pretop_height_reviewed') is not True
                or evidence.get('snapshot_frame_id') != state['snapshot_frame_id']):
            raise ValueError('Current-frame orientation, exit lane and pretop height need explicit review')
        names = (('palm_orientation_path', 'palm_orientation_sha256'),
                 ('reveal_exit_lane_path', 'reveal_exit_lane_sha256'))
        resolved = []
        for path_key, hash_key in names:
            value = evidence.get(path_key)
            if not isinstance(value, str) or not value:
                raise ValueError(f'{path_key} needs an existing reviewed file')
            path = _relative(base, value)
            expected = evidence.get(hash_key)
            if (not isinstance(expected, str) or len(expected) != 64
                    or _sha256(path) != expected):
                raise ValueError(f'{path_key} SHA-256 differs from review')
            resolved.append(path)
        orientation, lane = resolved
        pretop = evidence.get('pretop_height_m')
    else:
        if (not config.get('palm_orientation')
                or config.get('orientation_reviewed') is not True
                or not config.get('reveal_exit_lane')
                or config.get('pretop_height_reviewed') is not True):
            raise ValueError('Top palm orientation, pretop height, and exit lane are not reviewed')
        orientation = _relative(base, config['palm_orientation'])
        lane = _relative(base, config['reveal_exit_lane'])
        pretop = config.get('pretop_height_m')
    if (type(pretop) not in (int, float) or not math.isfinite(pretop)
            or not .04 <= pretop <= .12):
        raise ValueError('Reviewed pretop height must be 40..120 mm')
    lane_data = json.loads(lane.read_bytes())
    if (not isinstance(lane_data, dict)
            or lane_data.get('reviewed') is not True
            or lane_data.get('source_frame_id') != state['snapshot_frame_id']):
        raise ValueError('Reveal lane does not belong to current formal RGB-D')
    snapshot = _relative(base, config['snapshot_dir'], state['scene_id'])
    for kind, name in (('color', 'color.png'), ('depth', 'depth.npz')):
        if lane_data.get(f'snapshot_sha256_{kind}') != _sha256(snapshot/name):
            raise ValueError(f'Reveal lane {kind} differs from formal RGB-D')
    if lane_data.get('source') == 'measured_cup_footprint_conservative_dice_roi':
        if (lane_data.get('dice_visibility') != 'hidden_under_inverted_cup'
                or lane_data.get('dice_directly_observed') is not False):
            raise ValueError('Hidden-dice proxy must declare that dice were not observed')
    elif lane_data.get('source') != 'measured_dice_roi_plus_reviewed_exit_lane':
        raise ValueError('Reveal lane lacks measured dice ROI or cup-footprint proxy')
    clearance = config.get('clearance_height_m', .08)
    if (('clearance_height_m' in config and
         config.get('clearance_height_reviewed') is not True)
            or type(clearance) not in (int, float)
            or not math.isfinite(clearance) or not .025 <= clearance <= .08):
        raise ValueError('Current HOME axial clearance needs reviewed 25..80 mm')
    return orientation, lane, float(pretop), float(clearance)


def _plan_from_yolo(config, base, state, state_path, evidence, *, python,
                    camera_python, runner, now_ns):
    """Existing source adapter + top planner, with saved same-frame CAN reads."""
    _scene_evidence(evidence, now_ns=now_ns)
    if evidence['scene_id'] != state['scene_id']:
        raise ValueError('Cup scene changed before top planning')
    orientation, lane, pretop, clearance = _reviewed_plan_inputs(
        config, base, state, evidence)
    snapshot_max_age = config.get('planning_snapshot_max_age_s', 10.)
    if (type(snapshot_max_age) not in (int, float)
            or not 10. <= snapshot_max_age <= 120.
            or (snapshot_max_age > 10. and config.get('fresh_target_revalidation') is not True)):
        raise ValueError('Extended planning age needs opt-in fresh target RGB-D revalidation')
    snapshot = _relative(base, config['snapshot_dir'], state['scene_id'])
    recognition = _relative(base, config['recognition_dir'], state['scene_id'])
    side = _relative(base, config['source_side_plan'], state['scene_id'])
    plan = _relative(base, config['top_plan'], state['scene_id'])
    if side.exists() or plan.exists():
        raise ValueError('Top plan/source already exist; stale plans may not be reused')
    prefix = _control_prefix(config, python)
    execute.read_arm_status(prefix, runner=runner)
    q, _ = execute.read_joint_limits(prefix, runner=runner)
    execute.read_hand_status(prefix, runner=runner)
    status = _read_control_event(prefix, 'status', 'arm_status', runner=runner)
    joints = _read_control_event(prefix, 'read-joints', 'read_joints', runner=runner)
    hand = _read_control_event(prefix, 'hand-status', 'hand_status', runner=runner)
    if config.get('require_retracted_thumb') is True:
        positions = hand.get('positions', hand.get('finger_positions'))
        if (not isinstance(positions, dict)
                or any(float(positions.get(name, 100)) > 3. for name in
                       ('thumb_tip', 'thumb_base'))):
            raise ValueError('Top approach requires both thumb actuators in open 0..3 position')
    read_time_ns = time.time_ns()
    status['host_read_time_ns'] = read_time_ns
    joints['host_read_time_ns'] = read_time_ns
    hand['host_read_time_ns'] = read_time_ns
    for actual, arrival in zip(q, state['home_arrival_joints_rad']):
        if abs(actual-arrival) > math.radians(.5):
            raise ValueError('Arm changed after HOME before planning')
    feedback_files = {config['arm_status_feedback']: status,
                      config['joint_feedback']: joints,
                      config['hand_feedback']: hand}
    for name, item in feedback_files.items():
        _write_atomic(_relative(base, name, state['scene_id']), item, first=True)
    adapter_cmd = [camera_python, '-m', 'cup_grasp_demo.yolo_top_adapter',
                   '--snapshot', str(snapshot), '--recognition', str(recognition),
                   '--calibration', str(_relative(base, config['calibration'])),
                   '--home-state', str(state_path),
                   '--arm-status', str(_relative(base, config['arm_status_feedback'], state['scene_id'])),
                   '--joint-feedback', str(_relative(base, config['joint_feedback'], state['scene_id'])),
                   '--arm-read-time-ns', str(read_time_ns),
                   '--channel', config['channel'], '--allow-provisional',
                   '--output', str(side)]
    adapter = runner(adapter_cmd, capture_output=True, text=True, timeout=30, check=False)
    if adapter.returncode:
        raise RuntimeError('Same-frame YOLO top-source adapter failed: '+adapter.stderr.strip())
    top_cmd = [camera_python, '-m', 'cup_grasp_demo.top_grasp',
               '--side-plan', str(side),
               '--joint-feedback', str(_relative(base, config['joint_feedback'], state['scene_id'])),
               '--hand-feedback', str(_relative(base, config['hand_feedback'], state['scene_id'])),
               '--palm-orientation', str(orientation),
               '--orientation-reviewed',
               '--pretop-height-m', str(pretop),
               '--clearance-height-m', str(clearance),
               '--snapshot-max-age-s', str(snapshot_max_age),
               '--reveal-exit-lane', str(lane),
               '--allow-provisional', '--output', str(plan)]
    if config.get('require_retracted_thumb') is True:
        top_cmd.insert(-2, '--require-retracted-thumb')
    result = runner(top_cmd, capture_output=True, text=True, timeout=45, check=False)
    if result.returncode:
        raise RuntimeError('Independent top planner failed: '+result.stderr.strip())
    top_plan = json.loads(plan.read_bytes())
    approach = top_plan['stage_sequence'][:top_plan['stage_sequence'].index('guarded_hover')+1]
    if (top_plan.get('snapshot_frame_id') != state['snapshot_frame_id']
            or top_plan.get('checks', {}).get('execute_ready') is not True):
        raise ValueError('Top plan is not executable or belongs to old RGB-D')
    return {'frame_id': state['snapshot_frame_id'],
            'plan_sha256': _sha256(plan), 'execute_ready': True,
            'reviewed_approach_sequence': approach,
            'source_side_plan_sha256': _sha256(side),
            'arm_read_time_ns': read_time_ns}


def _evidence_event(state, evidence):
    phase = state['phase']
    if phase == 'WAIT_SCENE_CLEAR':
        return 'scene_clear'
    if phase == 'CONTACT' and evidence.get('physical_contact_observed') is True:
        return 'physical_contact_observed'
    if phase == 'VERIFY_GRIP':
        return 'cup_hold_observed'
    if phase == 'VERIFY_REVEAL' and evidence.get('dice_region_clear') is True:
        return 'cup_revealed'
    return None


def _top_stage(state, config, base):
    phase = state['phase']
    if phase == 'APPROACH':
        sequence = state.get('reviewed_approach_sequence')
        if not isinstance(sequence, list) or not sequence:
            raise ValueError('APPROACH lacks exact reviewed plan stage sequence')
        position = len(state['approach_completed'])
        if position >= len(sequence):
            raise ValueError('Approach sequence is exhausted')
        return sequence[position], 'approach_stage_reached', sequence
    if phase == 'CONTACT':
        position = len(state['probe_completed'])
        if position >= len(pipeline_state.PROBE_STAGES):
            raise ValueError('No more probes; require physical contact observation or pause')
        return pipeline_state.PROBE_STAGES[position], 'probe_stage_reached', None
    if phase == 'GRIP':
        return 'close', 'hand_close_reached', None
    if phase == 'SHAKE':
        position = len(state['shake_completed'])
        return pipeline_state.SHAKE_STAGES[position], 'tiny_shake_stage_reached', None
    if phase == 'REVEAL':
        completed = state.get('reveal_completed')
        if not isinstance(completed, list):
            raise ValueError('REVEAL state lacks two-stage receipt tracking')
        sequence = ('reveal_lift', 'reveal_exit')
        if len(completed) >= len(sequence):
            raise ValueError('REVEAL motion complete; only independent visual check remains')
        return sequence[len(completed)], 'reveal_stage_reached', None
    raise ValueError(f'{phase} has no top execution stage')


def _top_scene_preflight(state, evidence, *, now_ns):
    """Reject stale/mismatched scene declarations before latching CAN uncertainty."""
    stage = _top_stage(state, None, None)[0]
    cup_moving_by_plan = stage in ('tiny_shake_b', 'reveal_lift', 'reveal_exit')
    _scene_evidence(evidence, now_ns=now_ns,
                    require_cup_still=not cup_moving_by_plan)
    if cup_moving_by_plan and evidence.get('held_cup_observed') is not True:
        raise ValueError('Current moving cup must be independently observed held')
    if evidence.get('path_obstacles_clear') is not True:
        raise ValueError('Current full-scene arm/cable/blue-mat path obstacles are not cleared')
    if (evidence.get('scene_id') != state['scene_id'] or
            evidence.get('frame_id') != state['snapshot_frame_id'] or
            evidence.get('plan_sha256') != state['plan_sha256'] or
            evidence.get('scene_observed') is not True):
        raise ValueError('Top stage needs current scene/RGB-D/plan review')
    if stage == 'close' and (state['contact_observed'] is not True or
                             evidence.get('contact_observed') is not True):
        raise ValueError('Close needs independent physical contact observation')
    if stage in (*pipeline_state.SHAKE_STAGES, 'reveal_lift', 'reveal_exit'):
        if state['hold_observed'] is not True or evidence.get('hold_observed') is not True:
            raise ValueError('Motion with cup needs independent hold observation')
    if stage in pipeline_state.SHAKE_STAGES and evidence.get('mouth_supported_observed') is not True:
        raise ValueError('Dice shake requires observed mouth/table support')
    if stage in ('reveal_lift', 'reveal_exit') and any(
            evidence.get(name) is not True for name in
            ('dice_roi_reviewed', 'exit_lane_reviewed', 'cup_volume_path_reviewed')):
        raise ValueError('REVEAL needs reviewed dice ROI, exit lane, and held-cup volume path')
    speed = evidence.get('speed_percent', 1)
    if (type(speed) is not int or not 1 <= speed <= 3 or
            (stage in ('guarded_hover', *pipeline_state.PROBE_STAGES, 'close') and speed != 1)):
        raise ValueError('Reviewed stage speed must be 1..3%; contact stays at 1%')
    return stage


def _top_motion(config, base, state, evidence, *, python, camera_python,
                runner, now_ns):
    stage, event, sequence = _top_stage(state, config, base)
    _top_scene_preflight(state, evidence, now_ns=now_ns)
    args = [camera_python, '-m', 'cup_grasp_demo.top_execute', stage,
            '--plan', str(_relative(base, config['top_plan'], state['scene_id'])),
            '--source-side-plan', str(_relative(base, config['source_side_plan'], state['scene_id'])),
            '--snapshot-dir', str(_relative(base, config['snapshot_dir'], state['scene_id'])),
            '--calibration', str(_relative(base, config['calibration'])),
            '--receipt', str(_relative(base, config['top_receipt'], state['scene_id'])),
            '--channel', config['channel'], '--python', python,
            '--allow-provisional', '--scene-observed', '--execute']
    speed = evidence.get('speed_percent', 1)
    if speed != 1:
        args.extend(('--speed-percent', str(speed)))
    if stage == state['reviewed_approach_sequence'][0] and config.get('fresh_target_revalidation') is True:
        revalidation_dir = _relative(base, config['first_stage_revalidation_dir'],
                                     state['scene_id'])
        args.extend(('--revalidation-dir', str(revalidation_dir),
                     '--camera-python', camera_python))
    if stage == 'close':
        if state['contact_observed'] is not True or evidence.get('contact_observed') is not True:
            raise ValueError('Close needs independent physical contact observation')
        args.append('--contact-observed')
    if stage in (*pipeline_state.SHAKE_STAGES, 'reveal_lift', 'reveal_exit'):
        if state['hold_observed'] is not True or evidence.get('hold_observed') is not True:
            raise ValueError('Motion with cup needs independent hold observation')
        args.append('--hold-observed')
    if stage in pipeline_state.SHAKE_STAGES:
        if evidence.get('mouth_supported_observed') is not True:
            raise ValueError('Dice shake requires observed mouth/table support')
        args.append('--mouth-supported-observed')
    if stage in ('reveal_lift', 'reveal_exit'):
        if (evidence.get('dice_roi_reviewed') is not True or
                evidence.get('exit_lane_reviewed') is not True or
                evidence.get('cup_volume_path_reviewed') is not True):
            raise ValueError('REVEAL needs reviewed dice ROI, exit lane, and held-cup volume path')
    result = runner(args, capture_output=True, text=True, timeout=180, check=False)
    if result.returncode:
        raise RuntimeError('Top stage failed; CAN command may have been sent: '
                           +result.stderr.strip())
    receipt_path = _relative(base, config['top_receipt'], state['scene_id'])
    receipt = json.loads(receipt_path.read_bytes())
    item = receipt['stages'][stage]
    if item.get('verified') is not True:
        raise RuntimeError('Top stage receipt lacks verified arrival')
    output = {'stage': stage, 'plan_sha256': state['plan_sha256'],
              'feedback_event': item['feedback_event'],
              'receipt_sha256': _sha256(receipt_path),
              'observed_at_ns': item['completed_at_ns']}
    if event == 'approach_stage_reached':
        output['reviewed_stage_sequence'] = list(sequence)
    elif event == 'hand_close_reached':
        live_hand = execute.read_hand_status(_control_prefix(config, python), runner=runner)
        if len(live_hand) != 6:
            raise ValueError('Closed hand feedback needs all six positions')
        output['six_hand_positions'] = live_hand
    elif event == 'tiny_shake_stage_reached':
        output['hold_observed_at_ns'] = state.get('hold_observed_at_ns')
    elif event == 'reveal_stage_reached':
        output['hold_observed_at_ns'] = state.get('hold_observed_at_ns')
    return event, output


def _step_action(config, base, state, state_path, evidence, *, python,
                 camera_python, runner, now_ns):
    phase = state['phase']
    event = _evidence_event(state, evidence)
    if event is not None:
        if event == 'scene_clear':
            evidence = _scene_evidence(evidence, now_ns=now_ns)
        if event in ('physical_contact_observed', 'cup_hold_observed',
                     'cup_revealed'):
            observed = _scene_evidence(
                evidence, now_ns=now_ns,
                require_cup_still=event != 'cup_revealed')
            if observed['scene_id'] != state['scene_id']:
                raise ValueError('Physical observation belongs to a changed scene')
            evidence['scene_image_sha256'] = observed['scene_image_sha256']
            if (evidence.get('target_snapshot_frame_id',
                             state['snapshot_frame_id']) !=
                    state['snapshot_frame_id']):
                raise ValueError('Physical observation belongs to a changed RGB-D target')
            if event == 'cup_revealed':
                if evidence.get('held_cup_observed') is not True:
                    raise ValueError('Reveal needs independent observation of held moved cup')
                evidence['hold_observed_at_ns'] = state.get('hold_observed_at_ns')
                if evidence.get('plan_sha256') != state['plan_sha256']:
                    raise ValueError('Reveal observation belongs to a changed plan')
        return event, evidence
    if phase == 'HOME':
        return 'home_move_reached', _home_motion(
            config, state, evidence, python=python, runner=runner, now_ns=now_ns)
    if phase == 'CAPTURE':
        return 'rgbd_captured', _snapshot(
            config, base, state, evidence, camera_python=camera_python,
            python=python, runner=runner)
    if phase == 'LOCALIZE':
        return 'visible_top_localized', _localize_yolo(
            config, base, state, evidence, camera_python=camera_python,
            runner=runner, now_ns=now_ns)
    if phase == 'PLAN':
        return 'top_plan_ready', _plan_from_yolo(
            config, base, state, state_path, evidence,
            python=python, camera_python=camera_python,
            runner=runner, now_ns=now_ns)
    if phase in ('APPROACH', 'CONTACT', 'GRIP', 'SHAKE', 'REVEAL'):
        return _top_motion(config, base, state, evidence,
                           python=python, camera_python=camera_python,
                           runner=runner, now_ns=now_ns)
    raise ValueError(f'No single-step action for {phase}')


def run_step(state_path, config_path, *, evidence=None, execute_motion=False,
             python=sys.executable, camera_python=sys.executable,
             runner=subprocess.run, now_ns=None):
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    evidence = {} if evidence is None else dict(evidence)
    config, identity, base = _load_config(config_path)
    with _locked_state(state_path):
        state = _load_state(state_path, identity)
        if state['phase'] in ('PAUSED', 'ERROR', 'DONE'):
            raise ValueError(f"{state['phase']} cannot step; inspect status/reset")
        phase = state['phase']
        if phase == 'WAIT_CONTROL_HANDOFF' and execute_motion:
            raise ValueError('WEB handoff needs read-only adopt-current-home verification')
        if not execute_motion:
            if phase == 'HOME':
                position = state['home_move_count']
                target = config['home_candidate']['waypoints_deg'][position]
                action = {'home_stage_index': position, 'target_joints_deg': target}
            elif phase in ('APPROACH', 'CONTACT', 'GRIP', 'SHAKE'):
                action = {'stage': _top_stage(state, config, base)[0]}
            else:
                action = {'required_evidence_or_adapter': phase}
            return {'event': 'pipeline_preview', 'phase': phase, 'action': action,
                    'state_advanced': False, 'CAN_command_sent': False}
        # Persist a conservative in-flight marker before any physical command.
        # If the process disappears after this write, the old plan cannot retry.
        action_is_motion = phase in MOTION_PHASES and _evidence_event(state, evidence) is None
        if phase in ('LOCALIZE', 'PLAN'):
            # A stale external review is replaceable input. Reject it before
            # a read-only adapter failure can pause the entire run.
            _scene_evidence(evidence, now_ns=now_ns)
        if action_is_motion and phase in ('APPROACH', 'CONTACT', 'GRIP', 'SHAKE', 'REVEAL'):
            _top_scene_preflight(state, evidence, now_ns=now_ns)
        if action_is_motion:
            in_flight = pipeline_state.apply_event(state, 'pause', {
                'reason': f'In-flight {phase} single stage; reconcile live scene and feedback',
                'command_may_have_been_sent': True})
            _write_atomic(state_path, in_flight)
        try:
            event, verified = _step_action(config, base, state, state_path, evidence,
                                           python=python, camera_python=camera_python,
                                           runner=runner, now_ns=now_ns)
            next_state = pipeline_state.apply_event(state, event, verified)
            _write_atomic(state_path, next_state)
            return {'event': 'pipeline_step_verified', 'phase_before': phase,
                    'phase_after': next_state['phase'], 'transition': event,
                    'evidence': verified, 'state_advanced': True}
        except Exception as error:
            if not action_is_motion:
                paused = pipeline_state.apply_event(state, 'pause', {
                    'reason': f'{phase} failed: {error}',
                    'command_may_have_been_sent': False})
                _write_atomic(state_path, paused)
            raise


def _load_evidence(path):
    return {} if path is None else json.loads(Path(path).read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--camera-python', default=sys.executable)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    sub.add_parser('status')
    existing = sub.add_parser('record-existing-home',
                              help='Read-only adopt nine completed API HOME steps and post-HOME RGB-D')
    existing.add_argument('--manifest', type=Path, required=True)
    existing.add_argument('--formal-snapshot', type=Path,
                          help='Defaults to manifest formal_snapshot_dir; never recaptures')
    current = sub.add_parser('adopt-current-home',
                             help='Read-only designate stationary current pose; await WEB to CAN handoff')
    current.add_argument('--evidence', type=Path, required=True,
                         help='Fresh external photo/red cup/path review JSON')
    step = sub.add_parser('step')
    step.add_argument('--evidence', type=Path)
    step.add_argument('--execute', action='store_true',
                      help='Commit one verified step; motion phases dispatch one API command')
    pause = sub.add_parser('pause')
    pause.add_argument('--reason', required=True)
    reset = sub.add_parser('reset')
    reset.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config, identity, _ = _load_config(args.config)
        with _locked_state(args.state):
            if args.command == 'init':
                state = pipeline_state.new_state(identity)
                _write_atomic(args.state, state, first=True)
                result = {'event': 'pipeline_initialized', 'state': str(args.state),
                          'phase': state['phase'], 'home_candidate_provisional':
                          config['home_candidate']['provisional']}
            elif args.command == 'status':
                result = _load_state(args.state, identity)
            elif args.command in ('record-existing-home', 'adopt-current-home'):
                # Read-only HOME import/designation owns its own lock and K3
                # feedback; dispatch after releasing this outer lock.
                result = None
            elif args.command == 'pause':
                state = _load_state(args.state, identity)
                state = pipeline_state.apply_event(state, 'pause', {
                    'reason': args.reason,
                    'command_may_have_been_sent':
                    state.get('command_may_have_been_sent', False)})
                _write_atomic(args.state, state)
                result = {'event': 'pipeline_paused', 'phase': state['phase'],
                          'reason': state['pause_reason']}
            elif args.command == 'reset':
                state = _load_state(args.state, identity)
                evidence = _load_evidence(args.evidence)
                if state['phase'] != 'PAUSED':
                    raise ValueError('Reset requires a paused run')
                _age_ns(evidence.get('observed_at_ns'), time.time_ns())
                if (evidence.get('people_clear') is not True or
                        evidence.get('cup_still') is not True or
                        _sha256(evidence['scene_image_path']) !=
                        evidence.get('scene_image_sha256') or
                        not isinstance(evidence.get('live_arm_status'), dict) or
                        not evidence.get('live_camera_frame_id')):
                    raise ValueError('Reset needs new scene image, live arm status and RGB-D identity')
                state = pipeline_state.apply_event(state, 'reset_after_scene_change', {
                    'new_scene_id': evidence['new_scene_id'],
                    'live_arm_status': evidence['live_arm_status'],
                    'live_camera_frame_id': evidence['live_camera_frame_id']})
                _write_atomic(args.state, state)
                result = {'event': 'pipeline_reset', 'phase': state['phase'],
                          'old_targets_invalidated': True}
        if args.command == 'step':
            result = run_step(args.state, args.config, evidence=_load_evidence(args.evidence),
                              execute_motion=args.execute, python=args.python,
                              camera_python=args.camera_python)
        elif args.command == 'record-existing-home':
            result = record_existing_home(
                args.state, args.config, args.manifest,
                args.formal_snapshot, python=args.python)
        elif args.command == 'adopt-current-home':
            result = adopt_current_home(
                args.state, args.config, _load_evidence(args.evidence),
                python=args.python)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(json.dumps({'event': 'pipeline_failed', 'error': str(error)},
                         ensure_ascii=False), file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
