"""Run YOLO segmentation on a saved RGB-D frame and write non-executable planning evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from dice_cup_localization.geometry import Config, localize
from dice_cup_localization.red_workspace import RedWorkspace
from dice_cup_localization.yolo_seg import YoloSegmentor, select_green_cup

CAP_MODEL_SHA256 = 'c45d2b7fa61c45c5ef608cabbefd69fd9befe19243a14685e69b2ad645505ec0'
PERCEPTIVE_COCO_MODEL_SHA256 = '55e02f2a98581a134515d6a342b0342bc8d29a69c6d4d34f2421fba0fcde91d4'
PERCEPTIVE_COCO_CLASSES = (41, 75)


def geometry_config(max_center_spread_mm=None):
    """Keep the established 5 mm gate; allow a narrow explicit K3 opt-in."""
    if max_center_spread_mm is None:
        return Config()
    value = float(max_center_spread_mm)
    if not np.isfinite(value) or not 5.0 <= value <= 6.0:
        raise ValueError('Reviewed center spread gate must be 5..6 mm')
    return Config(max_center_spread_m=value/1000.)


def _green_roi(image):
    """Propose a crop from color; YOLO must still segment an allowed instance."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, (35, 55, 35), (90, 255, 255))
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count < 2:
        raise ValueError('No green area for ROI proposal')
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[index]
    if area < 150:
        raise ValueError('Green area too small for ROI proposal')
    padding = max(50, round(0.8 * max(w, h)))
    return [max(0, int(x)-padding), max(0, int(y)-padding),
            min(image.shape[1], int(x+w)+padding),
            min(image.shape[0], int(y+h)+padding)]


def load_source(frame=None, snapshot=None):
    """Load one saved RGB-D pair and preserve a snapshot's verified hashes."""
    if snapshot:
        color_path = snapshot/'color.png'
        depth_path = snapshot/'depth.npz'
        metadata = json.loads((snapshot/'metadata.json').read_text())
        if metadata.get('schema') != 1 or metadata.get('depth_registered_to') != 'color_optical':
            raise ValueError('Unsupported snapshot schema/frame')
        for path, key in ((color_path, 'sha256_color'), (depth_path, 'sha256_depth')):
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get(key):
                raise ValueError('Snapshot hash mismatch: '+key)
        image = cv2.imread(str(color_path))
        with np.load(depth_path, allow_pickle=False) as bundle:
            depth = bundle['depth_raw']
        source_info = {'kind': 'cup_grasp_demo_snapshot', 'metadata': str(snapshot/'metadata.json'),
                       'sha256_color': metadata['sha256_color'],
                       'sha256_depth': metadata['sha256_depth']}
    else:
        image = cv2.imread(str(frame)+'.png')
        metadata = json.loads(Path(str(frame)+'.json').read_text())
        with np.load(str(frame)+'.npz', allow_pickle=False) as bundle:
            depth = bundle['depth']
        source_info = {'kind': 'dice_cup_localization_frame', 'frame_prefix': str(frame)}
    if image is None:
        raise ValueError('Missing color frame')
    if depth.shape != image.shape[:2] or depth.dtype != np.uint16:
        raise ValueError('Expected registered Z16 depth at color resolution')
    if metadata.get('intrinsics', {}).get('frame') != 'color_optical':
        raise ValueError('Require color_optical intrinsics')
    return image, depth, metadata, source_info


def _select_red_candidate(image, instances, workspace, candidate_classes, kind):
    """Require one model instance, green color and red-mat table context."""
    select_green_cup(image, instances, candidate_classes=candidate_classes)
    for index, item in enumerate(instances):
        other_masks = [other['mask'] for j, other in enumerate(instances) if j != index]
        item['red_workspace'] = workspace.evaluate(image, item['mask'], other_masks)
    eligible = [i for i, item in enumerate(instances)
                if (item['class_id'] in candidate_classes and item['mask'].sum() >= 100
                    and item['green_fraction'] >= .35 and item['red_workspace']['valid'])]
    selected = eligible[0] if len(eligible) == 1 else None
    reason = None if selected is not None else (
        f'no_green_{kind}_in_red_workspace' if not eligible
        else f'multiple_green_{kind}_in_red_workspace')
    return selected, reason


def _select_red_cap(image, instances, workspace):
    """Preserve the trained cap=0 selection contract."""
    return _select_red_candidate(image, instances, workspace, (0,), 'cap')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--frame', type=Path, help='Prefix of png/npz/json files')
    source.add_argument('--snapshot', type=Path,
                        help='cup_grasp_demo color.png/depth.npz/metadata.json directory')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-profile', choices=('coco', 'dice_cap2', 'coco_red_fallback'),
                        default='coco', help='Explicit model/layout/class contract')
    parser.add_argument('--red-workspace', type=Path,
                        help='Red-mat polygon config for dice_cap2/coco_red_fallback')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X1','Y1','X2','Y2'),
                        help='Explicit image crop to rerun YOLO on a larger view of the target')
    parser.add_argument('--auto-green-roi', action='store_true',
                        help='Find a green crop, then require YOLO instance confirmation')
    parser.add_argument('--candidate-classes', type=int, nargs='+',
                        help='COCO labels accepted as target hypotheses; default cup=41')
    parser.add_argument('--ort-package-dir', type=Path,
                        help='Existing ORT parent path; process-local, no installation')
    parser.add_argument('--max-center-spread-mm', type=float,
                        help='Explicit reviewed side-section gate, 5..6 mm; default 5 mm')
    args = parser.parse_args()
    if args.ort_package_dir:
        sys.path.append(str(args.ort_package_dir))
    image, depth, metadata, source_info = load_source(args.frame, args.snapshot)
    segmentor = YoloSegmentor(args.model)
    if args.model_profile == 'dice_cap2':
        if (segmentor.layout != 'standard2_cap_ground'
                or segmentor.provenance['sha256'] != CAP_MODEL_SHA256):
            raise ValueError('dice_cap2 requires exact verified cap/ground best.q.onnx')
        candidate_classes = [0]
        if args.candidate_classes is not None and args.candidate_classes != [0]:
            raise ValueError('dice_cap2 target class must be cap=0 only')
        segmentor.provenance.update(
            source='Fitz8863/spacemitk3-dice_demo yolov8_seg',
            upstream_url='https://raw.githubusercontent.com/Fitz8863/'
                         'spacemitk3-dice_demo/yolov8_seg/models/best.q.onnx',
            branch_commit='f4489b20900a6b747e5a0a1e1c11adeeb154fb56')
        class_names = ['cap', 'ground']
    elif args.model_profile == 'coco_red_fallback':
        if (segmentor.layout != 'spacemit13_coco'
                or segmentor.provenance['sha256'] != PERCEPTIVE_COCO_MODEL_SHA256):
            raise ValueError('coco_red_fallback requires exact perceptive_grasp YOLOv8n-seg weight')
        candidate_classes = list(PERCEPTIVE_COCO_CLASSES)
        if (args.candidate_classes is not None
                and set(args.candidate_classes) != set(PERCEPTIVE_COCO_CLASSES)):
            raise ValueError('coco_red_fallback target classes must be COCO cup=41/vase=75')
        segmentor.provenance.update(
            source='perceptive_grasp YOLOv8-Seg config; COCO model',
            reference_config='/home/anny/workspaces/spacemit_robot/application/ros2/'
                             'linksee/perceptive_grasp/config/yolov8_seg.yaml')
        class_names = ['COCO cup=41', 'COCO vase=75']
    else:
        if segmentor.layout != 'spacemit13_coco':
            raise ValueError('COCO profile requires 13-output model')
        candidate_classes = args.candidate_classes or [41]
        class_names = ['COCO']
    if args.roi and args.auto_green_roi:
        raise ValueError('Choose explicit ROI or auto-green-roi')
    roi = args.roi or (_green_roi(image) if args.auto_green_roi else
                       [0, 0, image.shape[1], image.shape[0]])
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        raise ValueError('Invalid ROI')
    instances = segmentor.infer(image[y1:y2, x1:x2])
    for item in instances:
        full_mask = np.zeros(image.shape[:2], dtype=bool)
        full_mask[y1:y2, x1:x2] = item['mask']
        item['mask'] = full_mask
        item['bbox_xyxy'] = [item['bbox_xyxy'][0]+x1, item['bbox_xyxy'][1]+y1,
                             item['bbox_xyxy'][2]+x1, item['bbox_xyxy'][3]+y1]
    workspace = None
    if args.model_profile in ('dice_cap2', 'coco_red_fallback'):
        config_path = args.red_workspace or Path(__file__).with_name('config')/'red_mat_640x480.json'
        workspace = RedWorkspace(config_path, image.shape)
        if args.model_profile == 'dice_cap2':
            selected, reason = _select_red_cap(image, instances, workspace)
        else:
            selected, reason = _select_red_candidate(
                image, instances, workspace, PERCEPTIVE_COCO_CLASSES, 'coco_cup_vase')
    else:
        selected, reason = select_green_cup(image, instances,
                                             candidate_classes=tuple(candidate_classes))
    args.output.mkdir(parents=True, exist_ok=False)
    overlay = image.copy()
    for index, instance in enumerate(instances):
        color = np.array([0, 255, 0] if index == selected else [255, 120, 0])
        mask = instance['mask']
        overlay[mask] = (overlay[mask]*.5+color*.5).astype(np.uint8)
        x1, y1, x2, y2 = map(int, instance['bbox_xyxy'])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(map(int, color)), 2)
        cv2.putText(overlay, f"{index}:class{instance['class_id']} {instance['score']:.2f}",
                    (x1, max(15, y1)), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    if not cv2.imwrite(str(args.output/'overlay.png'), overlay):
        raise RuntimeError('Failed overlay write')
    report = {'frame_id': metadata['frame_id'], 'model': segmentor.provenance,
              'model_profile': args.model_profile, 'class_names': class_names,
              'source': source_info, 'roi_source': 'green_preselect_yolo_confirmed'
              if args.auto_green_roi else 'explicit' if args.roi else 'full_image',
              'selected_instance': selected, 'selection_reason': reason,
              'green_hsv_range': [[35, 55, 35], [90, 255, 255]], 'min_green_fraction': .35,
              'roi_xyxy': roi, 'candidate_classes': candidate_classes,
              'semantic_note': 'cap=0 is trained cap' if args.model_profile == 'dice_cap2'
              else 'COCO cup/vase are aliases, not trained dice-cup classes'
              if args.model_profile == 'coco_red_fallback'
              else 'COCO aliases are hypotheses, not a trained dice-cup class',
              'red_workspace': {'config_path': workspace.path, 'config_sha256': workspace.sha256,
                                'polygon_xy': workspace.vertices.tolist(),
                                'source': workspace.config['source']}
              if workspace else None,
              'instances': [{**{k: v for k, v in item.items() if k != 'mask'},
                             'mask_pixels': int(item['mask'].sum())} for item in instances]}
    config = geometry_config(args.max_center_spread_mm)
    result = {'valid': False, 'geometry': None, 'reason': reason}
    if selected is not None:
        mask = instances[selected]['mask']
        # Local surrounding pixels are table candidates, validated by plane RANSAC.
        ring = cv2.dilate(mask.astype(np.uint8), np.ones((81, 81), np.uint8)) != 0
        excluded = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) != 0
        for instance in instances:
            excluded |= instance['mask']
        table_mask = ring & ~excluded
        if workspace:
            table_mask &= workspace.mask
        mask_source = ('yolov8_seg_cap_red_workspace' if args.model_profile == 'dice_cap2'
                       else 'yolov8_seg_coco_cup_vase_red_workspace'
                       if args.model_profile == 'coco_red_fallback'
                       else 'yolov8_seg_coco_alias_green_filter')
        metadata.update(instance_id=f"{metadata['frame_id']}:cup:{selected}",
                        mask_source=mask_source)
        np.savez_compressed(args.output/'localization_input.npz', depth=depth,
                            object_mask=mask, table_mask=table_mask)
        (args.output/'localization_input.json').write_text(json.dumps(metadata, indent=2))
        result = localize(depth, mask, table_mask, metadata, cfg=config)
        if workspace and result['valid']:
            inside = workspace.contains_projected(
                result['geometry']['support_center_m'], metadata['intrinsics'])
            result['geometry']['quality']['table_support_center_inside_red_workspace'] = inside
            if not inside:
                result.update(valid=False, geometry=None,
                              reason='table_support_center_outside_red_workspace')
    result['input_provenance'] = source_info
    result['model_sha256'] = segmentor.provenance['sha256']
    result['roi_xyxy'] = roi
    result['candidate_classes'] = candidate_classes
    result['model_profile'] = args.model_profile
    result['max_center_spread_m'] = config.max_center_spread_m
    result['red_workspace_config_sha256'] = workspace.sha256 if workspace else None
    (args.output/'recognition.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.output/'geometry.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    blockers = ['No accepted and physically verified T_base_camera supplied',
                'No verified flange-to-grasp-TCP transform supplied',
                'No NERO seven-axis offline IK and collision-checked path supplied']
    if not result['valid']:
        blockers.insert(0, 'Cup geometry unavailable: '+str(result['reason']))
    elif not result['geometry']['top_surface']['valid']:
        blockers.insert(0, 'Visible cup top center unavailable: '
                        + str(result['geometry']['top_surface']['reason']))
    plan = {'executable': False, 'status': 'blocked', 'sequence': ['home', 'pregrasp', 'grasp'],
            'base_targets': None, 'tcp_offset': None, 'joint_path': None,
            'blockers': blockers, 'motion_sent': False, 'scope': 'reach_grasp_pose_only'}
    (args.output/'plan.json').write_text(json.dumps(plan, indent=2))
    print(json.dumps({'output': str(args.output), 'geometry_valid': result['valid'],
                      'selected_instance': selected, 'plan_executable': False}))
    return 0 if result['valid'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
