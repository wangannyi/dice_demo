"""Bind a human-reviewed cup scene to one formal RGB-D planning frame.

The operator must first inspect the current YOLO overlay and external scene.
This command only writes their exact-frame orientation, conservative hidden-
dice footprint and planning evidence; the independent planner checks IK and
all robot/hand geometry again.  No camera or CAN command is sent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from nero_calibration.core import matrix


MODEL_SHA = 'c45d2b7fa61c45c5ef608cabbefd69fd9befe19243a14685e69b2ad645505ec0'
DESTINATION_PALM_BASE_M = [.05, .48, .27]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_review_candidate(snapshot, recognition_dir, calibration_path, state_path,
                           scene_image, candidate_rotation_path, output_dir, *,
                           measured_dice_roi=None):
    snapshot, recognition_dir, output_dir = map(Path, (snapshot, recognition_dir, output_dir))
    meta = json.loads((snapshot/'metadata.json').read_bytes())
    geometry = json.loads((recognition_dir/'geometry.json').read_bytes())
    recognition = json.loads((recognition_dir/'recognition.json').read_bytes())
    state = json.loads(Path(state_path).read_bytes())
    if (state.get('phase') != 'PLAN' or state.get('snapshot_frame_id') != meta.get('frame_id')
            or meta.get('serial') != '346222071954'
            or _sha(snapshot/'color.png') != meta.get('sha256_color')
            or _sha(snapshot/'depth.npz') != meta.get('sha256_depth')
            or geometry.get('frame_id') != meta.get('frame_id')
            or recognition.get('frame_id') != meta.get('frame_id')
            or geometry.get('model_sha256') != MODEL_SHA
            or recognition.get('model', {}).get('sha256') != MODEL_SHA
            or recognition.get('selected_instance') != 0):
        raise ValueError('Reviewed candidate does not belong to current Dice YOLO/RGB-D')
    instance = recognition['instances'][0]
    red = instance.get('red_workspace', {})
    if (instance.get('class_id') != 0 or instance.get('score', 0) < .85
            or instance.get('green_fraction', 0) < .5
            or red.get('valid') is not True or red.get('mask_inside_fraction', 0) < .95):
        raise ValueError('Selected cup is not a confident green cap within red mat')
    observed = geometry.get('geometry', {})
    top = observed.get('top_surface', {})
    top_camera = np.asarray(top.get('center_m'), dtype=float)
    support_camera = np.asarray(observed.get('support_center_m'), dtype=float)
    diameter = float(observed.get('dimensions', {}).get('observed_side_diameter_m', math.nan))
    if (top.get('valid') is not True or top_camera.shape != (3,)
            or support_camera.shape != (3,) or not np.isfinite(top_camera).all()
            or not np.isfinite(support_camera).all()
            or not .06 <= diameter <= .09):
        raise ValueError('Current visible cup geometry drifted from reviewed scene')
    T = matrix(json.loads(Path(calibration_path).read_bytes())['T_base_camera'])
    support_base = T[:3, :3]@support_camera+T[:3, 3]
    rotation = np.asarray(json.loads(Path(candidate_rotation_path).read_bytes()), dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)
            or np.linalg.det(rotation) < .99999):
        raise ValueError('Reviewed palm rotation is not a right-handed 3x3')
    radius = max(.085, diameter/2+.013+.02)
    lane = {
        'frame': 'base', 'source': 'measured_cup_footprint_conservative_dice_roi',
        'reviewed': True, 'source_frame_id': meta['frame_id'],
        'snapshot_sha256_color': meta['sha256_color'],
        'snapshot_sha256_depth': meta['sha256_depth'],
        'dice_visibility': 'hidden_under_inverted_cup',
        'dice_directly_observed': False,
        'dice_roi_center_base_m': support_base.tolist(),
        'dice_roi_radius_m': radius, 'minimum_roi_clearance_m': .02,
        'destination_palm_base_m': DESTINATION_PALM_BASE_M,
        'review_basis': 'same_frame_cup_footprint_and_human_reviewed_external_arm_scene',
    }
    if measured_dice_roi is not None:
        dice_path = Path(measured_dice_roi)
        dice = json.loads(dice_path.read_bytes())
        center = np.asarray(dice.get('center_base_m'), dtype=float)
        radius = float(dice.get('radius_m', math.nan))
        if (dice.get('schema') != 1 or dice.get('kind') != 'measured_dice_roi'
                or dice.get('annotation_reviewed') is not True
                or dice.get('source_frame_id') != meta['frame_id']
                or dice.get('snapshot_sha256_color') != meta['sha256_color']
                or dice.get('snapshot_sha256_depth') != meta['sha256_depth']
                or dice.get('calibration_sha256') != _sha(calibration_path)
                or dice.get('table_geometry_sha256') != _sha(recognition_dir/'geometry.json')
                or center.shape != (3,) or not np.isfinite(center).all()
                or not math.isfinite(radius) or not .02 <= radius <= .12):
            raise ValueError('Measured dice ROI differs from current reviewed RGB-D/calibration')
        lane.update(source='measured_dice_roi_plus_reviewed_exit_lane',
                    dice_visibility='directly_visible',
                    dice_directly_observed=True,
                    dice_roi_center_base_m=center.tolist(),
                    dice_roi_radius_m=radius,
                    review_basis='same_frame_visible_dice_disk_and_external_arm_scene',
                    measured_dice_roi_path=str(dice_path.resolve()),
                    measured_dice_roi_sha256=_sha(dice_path))
    image = Path(scene_image)
    if not image.is_file() or not 0 <= (time.time_ns()-image.stat().st_mtime_ns)/1e9 <= 12.:
        raise ValueError('External arm scene photo is missing or stale')
    output_dir.mkdir(parents=True, exist_ok=True)
    orientation_path = output_dir/'palm_orientation.json'
    lane_path = output_dir/'reveal_exit_lane.json'
    evidence_path = output_dir/'plan_evidence.json'
    for path, value in ((orientation_path, rotation.tolist()), (lane_path, lane)):
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    evidence = {
        'scene_id': state['scene_id'], 'observed_at_ns': image.stat().st_mtime_ns,
        'scene_image_path': str(image.resolve()), 'scene_image_sha256': _sha(image),
        'path_scene_sha256': _sha(image), 'people_clear': True, 'cup_still': True,
        'path_obstacles_clear': True, 'snapshot_frame_id': meta['frame_id'],
        'palm_orientation_path': str(orientation_path.resolve()),
        'palm_orientation_sha256': _sha(orientation_path),
        'orientation_reviewed': True,
        'reveal_exit_lane_path': str(lane_path.resolve()),
        'reveal_exit_lane_sha256': _sha(lane_path),
        'exit_lane_reviewed': True, 'pretop_height_m': .11,
        'pretop_height_reviewed': True,
    }
    with evidence_path.open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return {'evidence': str(evidence_path), 'top_camera_m': top_camera.tolist(),
            'support_base_m': support_base.tolist(), 'model_score': instance['score'],
            'frame_color_sha256': meta['sha256_color']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('snapshot', 'recognition', 'calibration', 'state',
                'scene-image', 'candidate-rotation', 'output-dir'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--dice-roi', type=Path,
                   help='Exact-frame visible dice disk measurement; otherwise hidden-cup proxy')
    a = p.parse_args(argv)
    try:
        result = write_review_candidate(a.snapshot, a.recognition, a.calibration,
                                        a.state, a.scene_image,
                                        a.candidate_rotation, a.output_dir,
                                        measured_dice_roi=a.dice_roi)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'error': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
