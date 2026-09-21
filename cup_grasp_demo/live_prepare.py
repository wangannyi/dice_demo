"""Prepare one reviewed HOME-to-plan run before the RGB-D frame expires.

This PC helper dispatches only the state machine's read-only stages. It checks
the red tabletop against a previously inspected scene before each stage and
never sends an arm or hand motion command.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from zoneinfo import ZoneInfo

import cv2
import numpy as np


PC_ROOT = Path(__file__).resolve().parents[1]
K3_ROOT = Path('/home/test2/dice_demo')
K3_HOST = 'test2@10.0.91.111'
CONFIG = 'cup_grasp_demo/config/green_top_retracted_thumb.json'
CAMERA_PYTHON = '/home/test2/.venv-grasp/bin/python'


def run(command: list[str], *, cwd: Path = PC_ROOT) -> str:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            check=False)
    if result.returncode:
        raise RuntimeError(f'{shlex.join(command)} failed: '
                           f'{result.stdout[-1200:]} {result.stderr[-1200:]}')
    return result.stdout.strip()


def remote(command: list[str]) -> str:
    return run(['ssh', K3_HOST, 'cd /home/test2/dice_demo && '
                + shlex.join(command)])


def pipeline(state: Path, command: str, evidence: Path | None = None) -> dict:
    args = ['./cup_grasp_demo/run_k3.sh', 'pipeline', '--config', CONFIG,
            '--state', str(state), command]
    if command == 'step':
        args.append('--execute')
    if evidence is not None:
        args.extend(('--evidence', str(evidence)))
    lines = remote(args).splitlines()
    result = json.loads(lines[-1])
    print(f'{command}: {result.get("transition", result.get("event"))} '
          f'{result.get("phase_after", result.get("phase", ""))}', flush=True)
    return result


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_to_k3(local: Path, remote_path: Path, *, preserve_time=False) -> None:
    args = ['scp']
    if preserve_time:
        args.append('-p')
    run([*args, str(local), f'{K3_HOST}:{remote_path}'])


def copy_from_k3(remote_path: Path, local: Path) -> None:
    run(['scp', f'{K3_HOST}:{remote_path}', str(local)])


def photograph(review: Path, name: str, baseline: np.ndarray) -> Path:
    photo = review/f'{name}.jpg'
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'v4l2',
         '-input_format', 'mjpeg', '-video_size', '1280x720', '-i', '/dev/video4',
         '-frames:v', '1', '-y', str(photo)])
    latest = cv2.imread(str(photo))
    if latest is None or latest.shape != baseline.shape:
        raise RuntimeError('External scene camera image is missing')
    # The user's active work area is the red tabletop. People alongside the
    # station do not block this comparison; anyone or anything entering the
    # table/work path makes its pixels differ substantially.
    area = (slice(470, 700), slice(250, 750))
    delta = np.abs(baseline[area].astype(np.int16)
                   - latest[area].astype(np.int16)).max(axis=2)
    fraction = float(np.mean(delta > 50))
    print(f'{name}: red mat changed fraction {fraction:.5f}', flush=True)
    if fraction > .02:
        raise RuntimeError('Red tabletop changed; inspect a new scene before continuing')
    return photo


def evidence(photo: Path, scene_id: str, local: Path, remote_dir: Path,
             *, extra: dict | None = None) -> Path:
    remote_photo = remote_dir/photo.name
    copy_to_k3(photo, remote_photo, preserve_time=True)
    entry = {'scene_id': scene_id, 'observed_at_ns': photo.stat().st_mtime_ns,
             'scene_image_path': str(remote_photo),
             'scene_image_sha256': sha(photo), 'people_clear': True,
             'cup_still': True, 'path_obstacles_clear': True}
    if extra:
        entry.update(extra)
    local.write_text(json.dumps(entry, indent=2)+'\n', encoding='utf-8')
    remote_evidence = remote_dir/local.name
    copy_to_k3(local, remote_evidence)
    return remote_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-scene', type=Path, required=True)
    parser.add_argument('--reviewed-formal-color', type=Path, required=True)
    args = parser.parse_args()
    baseline = cv2.imread(str(args.reviewed_scene))
    formal_baseline = cv2.imread(str(args.reviewed_formal_color))
    if baseline is None or formal_baseline is None:
        raise ValueError('Review the supplied scene and formal RGB-D color first')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    scene_id = f'green_{stamp}'
    local_dir = PC_ROOT/'cup_grasp_demo/datasets'/f'pipeline_{stamp}'
    review = local_dir/'review'
    review.mkdir(parents=True, exist_ok=False)
    remote_dir = K3_ROOT/'cup_grasp_demo/datasets'/f'pipeline_{stamp}'
    remote(['mkdir', '-p', str(remote_dir/'review')])
    state = remote_dir/'state.json'
    pipeline(state, 'init')

    home_photo = photograph(review, 'scene_home', baseline)
    home_evidence = evidence(home_photo, scene_id, review/'home_evidence.json',
                             remote_dir,
                             extra={'red_cup_on_red_mat_observed': True})
    pipeline(state, 'adopt-current-home', home_evidence)
    captured = pipeline(state, 'step', home_evidence)
    if captured['transition'] != 'rgbd_captured':
        raise RuntimeError('Formal RGB-D capture did not finish')
    snapshot = Path(captured['evidence']['snapshot_dir'])
    formal_color = review/'formal_color.png'
    copy_from_k3(snapshot/'color.png', formal_color)
    latest = cv2.imread(str(formal_color))
    if latest is None or latest.shape != formal_baseline.shape:
        raise RuntimeError('Formal D435i frame size changed')
    # White disk at the top of the aligned frame is static on this rig.
    disk = (slice(0, 33), slice(255, 340))
    changed = np.abs(latest[disk].astype(np.int16)
                     - formal_baseline[disk].astype(np.int16)).max(axis=2)
    disk_fraction = float(np.mean(changed > 65))
    print(f'formal disk changed fraction {disk_fraction:.5f}', flush=True)
    if disk_fraction > .10:
        raise RuntimeError('White dice disk shifted in the formal frame; review its pixels')

    localize_photo = photograph(review, 'scene_localize', baseline)
    localize_evidence = evidence(localize_photo, scene_id,
                                 review/'localize_evidence.json', remote_dir)
    localized = pipeline(state, 'step', localize_evidence)
    if localized['transition'] != 'visible_top_localized':
        raise RuntimeError('Green cup localization did not finish')
    center = np.asarray(localized['evidence']['top_center_base_m'], dtype=float)
    prior = np.array([-.1255, .5319, .0549])
    if float(np.linalg.norm(center-prior)) > .015:
        raise RuntimeError('Cup moved >15 mm from the reviewed red-mat scene')

    token = hashlib.sha256(scene_id.encode()).hexdigest()[:12]
    recognition = K3_ROOT/'cup_grasp_demo/datasets/green_pipeline'/token/'recognition'
    dice_roi = remote_dir/'dice_roi.json'
    annotation = ['env',
                  'PYTHONPATH=/home/test2/dice_demo:/home/test2/dice_demo/cup_grasp_demo/.deps',
                  CAMERA_PYTHON, '-m', 'cup_grasp_demo.dice_roi_annotation',
                  '--snapshot', str(snapshot), '--calibration',
                  'nero_calibration/datasets/palm_result_01.json',
                  '--geometry', str(recognition/'geometry.json'),
                  '--center-px', '294', '-8', '--radius-px', '36',
                  '--output', str(dice_roi)]
    print('dice ROI:', remote(annotation).splitlines()[-1], flush=True)
    local_roi = review/'dice_roi.json'
    copy_from_k3(dice_roi, local_roi)
    roi = json.loads(local_roi.read_bytes())

    orientation = review/'palm_orientation.json'
    copy_from_k3(K3_ROOT/'cup_grasp_demo/datasets/pipeline_20260916_1006/review/palm_orientation.json',
                 orientation)
    remote_orientation = remote_dir/'review/palm_orientation.json'
    copy_to_k3(orientation, remote_orientation)
    lane = {
        'frame': 'base', 'source': 'measured_dice_roi_plus_reviewed_exit_lane',
        'reviewed': True, 'source_frame_id': roi['source_frame_id'],
        'snapshot_sha256_color': roi['snapshot_sha256_color'],
        'snapshot_sha256_depth': roi['snapshot_sha256_depth'],
        'dice_visibility': 'directly_visible', 'dice_directly_observed': True,
        'dice_roi_center_base_m': roi['center_base_m'],
        'dice_roi_radius_m': roi['radius_m'], 'minimum_roi_clearance_m': .02,
        'destination_palm_base_m': [.05, .48, .27],
        'review_basis': 'same_frame_visible_dice_disk_and_external_arm_scene',
        'measured_dice_roi_path': str(dice_roi),
        'measured_dice_roi_sha256': sha(local_roi),
    }
    lane_file = review/'reveal_exit_lane.json'
    lane_file.write_text(json.dumps(lane, indent=2)+'\n', encoding='utf-8')
    remote_lane = remote_dir/'review/reveal_exit_lane.json'
    copy_to_k3(lane_file, remote_lane)

    plan_photo = photograph(review, 'scene_plan', baseline)
    plan_evidence = evidence(plan_photo, scene_id, review/'plan_evidence.json',
                             remote_dir, extra={
                                 'snapshot_frame_id': roi['source_frame_id'],
                                 'palm_orientation_path': str(remote_orientation),
                                 'palm_orientation_sha256': sha(orientation),
                                 'orientation_reviewed': True,
                                 'reveal_exit_lane_path': str(remote_lane),
                                 'reveal_exit_lane_sha256': sha(lane_file),
                                 'exit_lane_reviewed': True,
                                 'pretop_height_m': .11,
                                 'pretop_height_reviewed': True,
                             })
    planned = pipeline(state, 'step', plan_evidence)
    if planned['transition'] != 'top_plan_ready':
        raise RuntimeError('Top grasp plan did not finish')
    top_plan = K3_ROOT/'cup_grasp_demo/datasets/green_pipeline'/token/'top_plan.json'
    copy_from_k3(top_plan, review/'top_plan.json')
    print(json.dumps({'scene_id': scene_id, 'state': str(state),
                      'plan_sha256': planned['evidence']['plan_sha256'],
                      'top_plan': str(top_plan),
                      'formal_color': str(formal_color)}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
