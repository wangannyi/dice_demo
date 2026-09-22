#!/usr/bin/env python3
"""Set one RGB-D profile for the green pipeline and both red-cloth boards.

Only configuration files are changed. Existing calibration datasets are never
rewritten, and changing image geometry marks the pipeline for recalibration.
"""

import argparse
import json
import os
from pathlib import Path
import stat
import tempfile


ROOT = Path(__file__).resolve().parents[1]
GREEN_FILES = ('vision/camera.json',)
CAMERA_FILE = 'vision/camera.json'
# 标定板参数文件已随标定工具分离到 ../biaoding/config/，不再由本脚本同步。
DEFAULT_COLOR = [1280, 720]
DEFAULT_DEPTH = [1280, 720]
DEFAULT_CROP = [220, 0, 960, 720]


def dimensions(value):
    try:
        width, height = (int(part) for part in value.lower().split('x'))
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError('resolution must be WIDTHxHEIGHT') from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError('resolution must be positive')
    return [width, height]


def rectangle(value):
    try:
        result = [int(part) for part in value.split(',')]
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError('rectangle must contain four comma-separated integers') from exc
    if len(result) != 4:
        raise argparse.ArgumentTypeError('rectangle must contain four comma-separated integers')
    return result


def check_profile(link, color, depth, fps, crop):
    if color not in ([640, 480], [1280, 720]):
        raise ValueError('Color resolution must be 640x480 or 1280x720')
    if depth not in ([640, 480], [848, 480], [1280, 720]):
        raise ValueError('Depth resolution must be 640x480, 848x480 or 1280x720')
    if fps not in (6, 15, 30):
        raise ValueError('FPS must be 6, 15 or 30')
    x, y, width, height = crop
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > color[0] or y + height > color[1]:
        raise ValueError('Crop must lie inside the source color image')
    if link == 'usb2' and not (
        (color == depth == [1280, 720] and fps == 6)
        or (color == depth == [640, 480] and fps in (6, 15))
    ):
        raise ValueError('This USB 2 profile is not validated for simultaneous RGB, depth and stereo IR; use 1280x720@6 or 640x480@6/15')


def check_roi(roi, crop):
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= crop[2] and 0 <= y1 < y2 <= crop[3]):
        raise ValueError('Board ROI must lie inside the cropped color image')


def updated_files(root, link, *, color=None, depth=None, fps=None, crop=None,
                  hand_roi=None, reference_roi=None):
    color = color or DEFAULT_COLOR.copy()
    depth = depth or DEFAULT_DEPTH.copy()
    fps = fps if fps is not None else (6 if link == 'usb2' else 15)
    crop = crop if crop is not None else (DEFAULT_CROP.copy() if color == DEFAULT_COLOR
                                         else [0, 0, *color])
    check_profile(link, color, depth, fps, crop)
    files = {name: json.loads((root / name).read_text())
             for name in GREEN_FILES}
    previous = files[GREEN_FILES[0]]
    spatial_change = (previous['color_resolution'] != color or
                      previous.get('crop_xywh') != crop)
    if spatial_change and (hand_roi is None or reference_roi is None):
        raise ValueError('Color resolution/crop changed: supply --hand-roi and --reference-roi, then recalibrate')
    for roi in (hand_roi, reference_roi):
        if roi is not None:
            check_roi(roi, crop)
    for name in GREEN_FILES:
        camera = files[name]
        camera['color_resolution'] = color.copy()
        camera['depth_resolution'] = depth.copy()
        camera['fps'] = fps
        camera['crop_xywh'] = crop.copy()
    return files, spatial_change


def apply_profile(root, link, *, color=None, depth=None, fps=None, crop=None,
                  hand_roi=None, reference_roi=None, dry_run=False):
    files, spatial_change = updated_files(root, link, color=color, depth=depth,
        fps=fps, crop=crop, hand_roi=hand_roi, reference_roi=reference_roi)
    if not dry_run:
        for name, data in files.items():
            path = root / name
            payload = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                             prefix=path.name + '.', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            try:
                os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    camera = files[GREEN_FILES[0]]
    return dict(link=link, camera=camera, spatial_change=spatial_change,
                recalibration_required=spatial_change, files=list(files), dry_run=dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('link', choices=('usb2', 'usb3'))
    parser.add_argument('--fps', type=int, help='Override profile FPS')
    parser.add_argument('--color-resolution', type=dimensions, metavar='WIDTHxHEIGHT')
    parser.add_argument('--depth-resolution', type=dimensions, metavar='WIDTHxHEIGHT')
    parser.add_argument('--crop', type=rectangle, metavar='X,Y,W,H')
    parser.add_argument('--hand-roi', type=rectangle, metavar='X1,Y1,X2,Y2')
    parser.add_argument('--reference-roi', type=rectangle, metavar='X1,Y1,X2,Y2')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(apply_profile(ROOT, args.link, color=args.color_resolution,
              depth=args.depth_resolution, fps=args.fps, crop=args.crop,
              hand_roi=args.hand_roi, reference_roi=args.reference_roi,
              dry_run=args.dry_run), ensure_ascii=False, indent=2))
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
