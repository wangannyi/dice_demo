"""Bind a same-frame YOLO/RGB-D visible top to the opt-in palm planner.

This adapter is read-only.  It creates a side-plan *source record* for the
separate top planner; no side-grasp target is executed.  The visible top is a
side-axis/observed-plane intersection, not an independently measured rim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from cup_grasp_demo import planning
from cup_grasp_demo.grasp import _check_capture_calibration, load_snapshot
from nero_calibration.core import matrix, matrix_pose, pose_matrix
from nero_revo2_control.kinematics import load_model


MAX_POSE_AGE_S = 10.
MAX_JOINT_DISAGREEMENT_RAD = math.radians(.5)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _point(value, name):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f'{name} requires three finite meters')
    return point


def _seven(value, name):
    joints = np.asarray(value, dtype=float)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError(f'{name} requires seven finite radians')
    return joints


def _source_provenance(result, recognition, metadata):
    source = result.get('input_provenance') or {}
    if (source.get('kind') != 'cup_grasp_demo_snapshot'
            or source.get('sha256_color') != metadata['sha256_color']
            or source.get('sha256_depth') != metadata['sha256_depth']
            or result.get('frame_id') != metadata['frame_id']
            or recognition.get('frame_id') != metadata['frame_id']
            or result.get('model_sha256') != (recognition.get('model') or {}).get('sha256')):
        raise ValueError('YOLO result is not bound to the exact current RGB-D/model')
    if (not isinstance(recognition.get('selected_instance'), int)
            or recognition['selected_instance'] < 0):
        raise ValueError('YOLO did not select one green cup instance')
    if result.get('valid') is not True or not isinstance(result.get('geometry'), dict):
        raise ValueError(f"YOLO cup geometry invalid: {result.get('reason')}")
    top = result['geometry'].get('top_surface') or {}
    if top.get('valid') is not True:
        raise ValueError(f"Visible cup top invalid: {top.get('reason')}")
    quality = top.get('quality') or {}
    if (quality.get('rim_center_independently_measured') is not False
            or quality.get('coaxial_cup_assumption') is not True
            or top.get('center_definition') != 'side_axis_intersection_with_visible_top_plane'):
        raise ValueError('Visible top center definition/provenance changed')
    return top


def _red_workspace_provenance(result, recognition, metadata):
    """Bind the one trained green cap to measured red-mat table support."""
    workspace = recognition.get('red_workspace') or {}
    config_sha = workspace.get('config_sha256')
    selected = recognition.get('selected_instance')
    instances = recognition.get('instances') or []
    if (recognition.get('model_profile') != 'dice_cap2'
            or result.get('model_profile') != 'dice_cap2'
            or not isinstance(config_sha, str) or len(config_sha) != 64
            or result.get('red_workspace_config_sha256') != config_sha
            or not isinstance(selected, int) or selected < 0
            or selected >= len(instances)):
        raise ValueError('Exact trained cap/red workspace provenance is missing')
    instance = instances[selected]
    mask_evidence = instance.get('red_workspace') or {}
    quality = (result.get('geometry') or {}).get('quality') or {}
    if (instance.get('class_id') != 0
            or float(instance.get('green_fraction', 0)) < .35
            or mask_evidence.get('valid') is not True
            or float(mask_evidence.get('mask_inside_fraction', 0)) < .95
            or float(mask_evidence.get('red_context_fraction', 0)) < .2
            or quality.get('table_support_center_inside_red_workspace') is not True):
        raise ValueError('Selected green cap lacks red-mat mask/table support evidence')
    return {
        'selected_green_cap_in_red_mat': True,
        'table_support_center_inside_red_mat': True,
        'config_sha256': config_sha,
        'selected_instance': selected,
        'source_frame_id': metadata['frame_id'],
        'snapshot_sha256_color': metadata['sha256_color'],
        'mask_inside_fraction': mask_evidence['mask_inside_fraction'],
        'red_context_fraction': mask_evidence['red_context_fraction'],
    }


def build_source_record(
    snapshot_dir, recognition_dir, calibration_path, arm_status,
    joint_feedback, *, home_state, arm_read_time_ns, channel='can0',
    allow_provisional=False, now_ns=None,
):
    """Return a planner source after current HOME and exact same-frame checks."""
    snapshot_dir, recognition_dir = Path(snapshot_dir), Path(recognition_dir)
    _, _, metadata = load_snapshot(snapshot_dir)
    recognition = json.loads((recognition_dir/'recognition.json').read_bytes())
    result = json.loads((recognition_dir/'geometry.json').read_bytes())
    top = _source_provenance(result, recognition, metadata)
    red_workspace = _red_workspace_provenance(result, recognition, metadata)
    if (home_state.get('schema') != 1
            or home_state.get('kind') != 'nero_green_cup_state_machine'
            or home_state.get('phase') not in ('LOCALIZE', 'PLAN')
            or home_state.get('snapshot_frame_id') != metadata['frame_id']
            or home_state.get('home_arrival_time_ns') is None
            or int(home_state['home_arrival_time_ns']) >= int(metadata['host_capture_time_ns'])):
        raise ValueError('Formal YOLO source requires current HOME followed by its RGB-D capture')
    calibration = planning.load_calibration(
        calibration_path, expected_camera_serial=metadata['serial'],
        allow_provisional=allow_provisional)
    _check_capture_calibration(metadata, calibration)
    if calibration.get('tcp') != 'palm':
        raise ValueError('This YOLO top adapter requires palm TCP')
    if (arm_status.get('event') != 'arm_status'
            or joint_feedback.get('event') != 'read_joints'):
        raise ValueError('Require saved read-only arm_status and read_joints events')
    q_arm = _seven(arm_status.get('joints_rad'), 'Arm status joints')
    q_live = _seven(joint_feedback.get('joints_rad'), 'Joint feedback')
    if np.max(np.abs(q_arm-q_live)) > MAX_JOINT_DISAGREEMENT_RAD:
        raise ValueError('Arm status and joint feedback changed between reads')
    q_home = _seven(home_state.get('home_arrival_joints_rad'), 'Current HOME receipt')
    if np.max(np.abs(q_live-q_home)) > math.radians(1.):
        raise ValueError('Arm moved after HOME; recapture and replan')
    if (joint_feedback.get('arm_status') != 0 or joint_feedback.get('ctrl_mode') != 1
            or joint_feedback.get('joints_enabled') != [True]*7):
        raise ValueError('Current arm must be CAN/NORMAL/seven axes enabled')
    now_ns = time.time_ns() if now_ns is None else int(now_ns)
    arm_read_time_ns = int(arm_read_time_ns)
    if (arm_read_time_ns <= int(metadata['host_capture_time_ns'])
            or now_ns < arm_read_time_ns
            or (now_ns-arm_read_time_ns)/1e9 > MAX_POSE_AGE_S):
        raise ValueError('Current CAN pose is stale or predates RGB-D capture')
    flange = pose_matrix(arm_status.get('flange_m_rad'))
    fk = matrix(load_model().fk(q_live))
    if (np.linalg.norm(fk[:3, 3]-flange[:3, 3]) > .005
            or (np.trace(fk[:3, :3].T @ flange[:3, :3])-1)/2
            < math.cos(math.radians(2))):
        raise ValueError('Current flange feedback disagrees with seven-axis URDF FK')
    mapped_input = dict(result)
    mapped_input['camera_serial'] = metadata['serial']
    cup = planning.map_localization(mapped_input, calibration,
                                    camera_serial=metadata['serial'],
                                    allow_provisional=allow_provisional)
    top_camera = _point(top['center_m'], 'Visible top camera center')
    normal_camera = _point(top['normal'], 'Visible top camera normal')
    transform = matrix(calibration['T_base_camera'])
    top_base = transform[:3, :3] @ top_camera + transform[:3, 3]
    normal_base = transform[:3, :3] @ normal_camera
    if (float(normal_base @ np.asarray(cup['axis'])) < math.cos(math.radians(15))
            or float((top_base-np.asarray(cup['support_center_m'])) @
                     np.asarray(cup['axis'])) < .03):
        raise ValueError('Measured top plane/height disagrees with upright cup')
    cup['measured_top_surface'] = {
        'valid': True, 'center_camera_m': top_camera.tolist(),
        'center_base_m': top_base.tolist(),
        'normal_camera': normal_camera.tolist(),
        'normal_base': normal_base.tolist(),
        'quality': top['quality'], 'center_definition': top['center_definition'],
        'source_frame_id': metadata['frame_id'],
        'snapshot_sha256_color': metadata['sha256_color'],
        'snapshot_sha256_depth': metadata['sha256_depth'],
        'model_sha256': result['model_sha256'],
    }
    age_s = max(0., (now_ns-int(metadata['host_capture_time_ns']))/1e9)
    return {
        'schema': 1, 'kind': 'read_only_side_grasp_proposal', 'units': 'm_rad',
        'source_role': 'yolo_visible_top_source_only_no_side_motion',
        'snapshot_frame_id': metadata['frame_id'],
        'snapshot_sha256_color': metadata['sha256_color'],
        'snapshot_sha256_depth': metadata['sha256_depth'],
        'calibration_sha256': _sha(calibration_path),
        'localization': mapped_input,
        'cup_base': cup,
        'current_flange_pose_base_m_rad': matrix_pose(flange),
        'current_palm_pose_base_m_rad': matrix_pose(
            flange @ matrix(calibration['T_flange_tcp'])),
        'pose_source': {'kind': 'can_feedback_read_only', 'channel': channel,
                        'host_read_time_ns': arm_read_time_ns,
                        'home_arrival_time_ns': home_state['home_arrival_time_ns']},
        'home_check': {'confirmed': True,
                       'joint_feedback_deg': np.degrees(q_live).tolist(),
                       'joint_target_deg': np.degrees(q_home).tolist(),
                       'provenance': {'kind': 'can_joint_feedback_read_only'}},
        'checks': {
            'snapshot_age_s': age_s,
            'snapshot_fresh': age_s <= MAX_POSE_AGE_S,
            'pose_source_live_can': True, 'require_home': True,
            'home_confirmed': True, 'home_gate_passed': True,
            'calibration_quality_passed': calibration['quality_passed'],
            'provisional_opt_in': bool(allow_provisional),
            'measured_visible_top_valid': True,
            'execute_ready': False,
        },
        'recognition': {'selected_instance': recognition['selected_instance'],
                        'model_sha256': result['model_sha256'],
                        'roi_xyxy': result.get('roi_xyxy'),
                        'candidate_classes': result.get('candidate_classes'),
                        'red_workspace': red_workspace},
        'motion_sent': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--recognition', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--home-state', type=Path, required=True)
    parser.add_argument('--arm-status', type=Path, required=True)
    parser.add_argument('--joint-feedback', type=Path, required=True)
    parser.add_argument('--arm-read-time-ns', type=int, required=True)
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--allow-provisional', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = build_source_record(
            args.snapshot, args.recognition, args.calibration,
            json.loads(args.arm_status.read_bytes()),
            json.loads(args.joint_feedback.read_bytes()),
            home_state=json.loads(args.home_state.read_bytes()),
            arm_read_time_ns=args.arm_read_time_ns,
            channel=args.channel, allow_provisional=args.allow_provisional)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'event': 'yolo_top_source_saved', 'output': str(args.output),
                          'frame_id': record['snapshot_frame_id'],
                          'top_camera_m': record['cup_base']['measured_top_surface']['center_camera_m'],
                          'top_base_m': record['cup_base']['measured_top_surface']['center_base_m'],
                          'motion_sent': False}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'event': 'yolo_top_source_failed', 'error': str(exc),
                          'motion_sent': False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
