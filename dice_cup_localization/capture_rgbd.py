"""Capture a bounded RGB-D sequence only during an assigned camera window."""

import argparse
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nero_calibration.image_profile import profile_options, crop_intrinsics, crop_image


def intrinsics_dict(intr):
    return {'width': intr.width, 'height': intr.height, 'fx': intr.fx, 'fy': intr.fy,
            'cx': intr.ppx, 'cy': intr.ppy, 'dist_coeffs': list(intr.coeffs),
            'distortion_model': str(intr.model)}


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--frames', type=int, default=3, choices=range(1, 11))
    parser.add_argument('--depth-width', type=int, default=640, choices=(640, 848, 1280))
    parser.add_argument('--depth-height', type=int, default=480, choices=(480, 720))
    parser.add_argument('--color-resolution', type=int, nargs=2, default=[640, 480])
    parser.add_argument('--crop-xywh', type=int, nargs=4)
    parser.add_argument('--stereo', action='store_true', help='Save both calibrated IR streams')
    parser.add_argument('--fps', type=int, default=15, choices=(6, 15, 30))
    parser.add_argument('--warmup-frames', type=int, default=20, choices=range(1, 61))
    parser.add_argument('--fresh-discard-frames', type=int, default=2, choices=range(0, 6))
    return parser


def frame_numbers(frames, stereo=False):
    color, depth = frames.get_color_frame(), frames.get_depth_frame()
    if not color or not depth:
        raise RuntimeError('Missing RGB-D frame at freshness boundary')
    numbers = [color.get_frame_number(), depth.get_frame_number()]
    if stereo:
        for side in (1, 2):
            ir = frames.get_infrared_frame(side)
            if not ir:
                raise RuntimeError('Missing IR frame at freshness boundary')
            numbers.append(ir.get_frame_number())
    return tuple(numbers)


def next_frames(pipeline, boundary, stereo=False):
    deadline = time.monotonic() + 3.0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('RGB and depth frames did not advance within 3 seconds')
        frames = pipeline.wait_for_frames(max(1, int(remaining * 1000)))
        numbers = frame_numbers(frames, stereo)
        if boundary is None or all(new > old for new, old in zip(numbers, boundary)):
            return frames


class CaptureSession:
    """Reusable stream; callers choose when to save fresh frames."""
    def __init__(self, args):
        self.args = args
        self.color_resolution, _, self.crop = profile_options(dict(color_resolution=args.color_resolution, fps=args.fps, crop_xywh=args.crop_xywh))
        self.pipeline = None
        self.started = False
        self.timings = {}

    def start(self):
        import pyrealsense2 as rs
        import time
        args = self.args
        if (args.depth_width, args.depth_height) not in ((640, 480), (848, 480), (1280, 720)):
            raise ValueError('Unsupported depth width/height pair')
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(args.serial)
        cfg.enable_stream(rs.stream.color, *self.color_resolution, rs.format.bgr8, args.fps)
        cfg.enable_stream(rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps)
        if args.stereo:
            for side in (1, 2):
                cfg.enable_stream(rs.stream.infrared, side, args.depth_width, args.depth_height,
                                  rs.format.y8, args.fps)
        try:
            started = time.perf_counter()
            self.profile = self.pipeline.start(cfg)
            self.timings["stream_start_s"] = time.perf_counter() - started
            self.started = True
            self.scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
            self.alignment = rs.align(rs.stream.color)
            deadline = time.monotonic() + 10
            warmup_started = time.perf_counter()
            warmed = 0
            previous = None
            while warmed < getattr(args, "warmup_frames", 20):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Camera warmup exceeded 10 seconds')
                frames = self.pipeline.wait_for_frames(3000)
                # RealSense can reuse a slower stream in consecutive framesets.
                # FAST counts complete, advancing sets rather than API returns.
                if getattr(args, 'unique_warmup', False):
                    try:
                        current = frame_numbers(frames, args.stereo)
                    except RuntimeError:
                        continue
                    if previous is not None and not all(n > o for n, o in zip(current, previous)):
                        continue
                    previous = current
                warmed += 1
            self.timings["warmup_s"] = time.perf_counter() - warmup_started
        except BaseException:
            self.close()
            raise

    def capture(self, output, count, *, fresh=False):
        import pyrealsense2 as rs
        from copy import copy
        if not self.started:
            raise RuntimeError('Camera has not started')
        if type(count) is not int or not 1 <= count <= 10:
            raise ValueError('Capture count must be 1..10')
        args = copy(self.args)
        args.output = Path(output)
        args.output.mkdir(parents=True, exist_ok=False)
        pipeline, profile, scale, alignment = self.pipeline, self.profile, self.scale, self.alignment
        capture_started = time.perf_counter()
        boundary = None
        if fresh:
            # Drop the queued set. FAST uses the next advancing RGB/depth set,
            # rather than unconditionally discarding two more frame periods.
            queued = pipeline.poll_for_frames()
            if queued:
                boundary = frame_numbers(queued, args.stereo)
            for _ in range(getattr(args, 'fresh_discard_frames', 2)):
                boundary = frame_numbers(pipeline.wait_for_frames(3000), args.stereo)
        for index in range(count):
            wait_started = time.perf_counter()
            frames = next_frames(pipeline, boundary, args.stereo) if fresh else pipeline.wait_for_frames(3000)
            wait_s = time.perf_counter() - wait_started
            if fresh:
                boundary = frame_numbers(frames, args.stereo)
            raw_depth = frames.get_depth_frame()
            raw_color = frames.get_color_frame()
            aligned = alignment.process(frames)
            color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color or not depth or not raw_depth or not raw_color:
                raise RuntimeError('Missing RGB-D frame')
            cp = color.profile.as_video_stream_profile()
            dp = raw_depth.profile.as_video_stream_profile()
            ci = crop_intrinsics(intrinsics_dict(cp.get_intrinsics()), self.crop)
            ci['frame'] = 'color_optical'
            ext = dp.get_extrinsics_to(cp)
            prefix = args.output / f'frame_{index:03d}'
            image = crop_image(np.asanyarray(color.get_data()), self.crop)
            png_options = [cv2.IMWRITE_PNG_COMPRESSION, 0] if getattr(args, 'fast_storage', False) else []
            if not cv2.imwrite(str(prefix)+'.png', image, png_options):
                raise RuntimeError('Failed to save color')
            arrays = dict(depth=crop_image(np.asanyarray(depth.get_data()), self.crop),
                          native_depth=np.asanyarray(raw_depth.get_data()).copy())
            stereo = {}
            if args.stereo:
                for side, number in [('left', 1), ('right', 2)]:
                    frame = frames.get_infrared_frame(number)
                    if not frame:
                        raise RuntimeError('Missing IR frame')
                    ir_profile = frame.profile.as_video_stream_profile()
                    ir_ext = ir_profile.get_extrinsics_to(cp)
                    arrays['ir_'+side] = np.asanyarray(frame.get_data()).copy()
                    stereo[side] = dict(intrinsics=intrinsics_dict(ir_profile.get_intrinsics()),
                                        to_color_rotation=list(ir_ext.rotation),
                                        to_color_translation=list(ir_ext.translation))
                    stereo[side+'_frame_number'] = frame.get_frame_number()
                    stereo[side+'_timestamp_ms'] = frame.get_timestamp()
            saver = np.savez if getattr(args, 'fast_storage', False) else np.savez_compressed
            saver(str(prefix)+'.npz', **arrays)
            metadata = {'capture_profile': {'color_resolution': [ci['width'], ci['height']],
                        'source_color_resolution': self.color_resolution, 'crop_xywh': self.crop,
                        'depth_resolution': [dp.width(), dp.height()],
                        'color_fps': cp.fps(), 'depth_fps': dp.fps(),
                        'usb_type': profile.get_device().get_info(rs.camera_info.usb_type_descriptor)},
                        'serial': args.serial, 'timestamp_ms': color.get_timestamp(),
                        'timestamp_domain': str(color.get_frame_timestamp_domain()),
                        'depth_timestamp_ms': raw_depth.get_timestamp(),
                        'depth_timestamp_domain': str(raw_depth.get_frame_timestamp_domain()),
                        'color_frame_number': color.get_frame_number(),
                        'depth_frame_number': raw_depth.get_frame_number(),
                        'frame_id': f'{args.serial}:{color.get_frame_number()}',
                        'depth_registered_to': 'color_optical', 'depth_scale_m': scale,
                        'intrinsics': ci, 'native_depth_intrinsics': intrinsics_dict(dp.get_intrinsics()),
                        'depth_to_color': {'rotation_column_major': list(ext.rotation),
                                           'translation_m': list(ext.translation)},
                        'alignment': 'librealsense rs.align(color)', 'filters': []}
            if args.stereo:
                metadata['stereo'] = stereo
            metadata['capture_timing'] = dict(self.timings, fresh_wait_s=wait_s,
                compressed_storage=not getattr(args, 'fast_storage', False),
                elapsed_capture_s=time.perf_counter()-capture_started,
                warmup_frames=getattr(args, 'warmup_frames', 20),
                unique_warmup=getattr(args, 'unique_warmup', False),
                fresh_discard_frames=getattr(args, 'fresh_discard_frames', 2))
            Path(str(prefix)+'.json').write_text(json.dumps(metadata, indent=2))
            print(prefix, flush=True)

    def close(self):
        if self.started:
            self.started = False
            self.pipeline.stop()


def main():
    args = argument_parser().parse_args()
    camera = CaptureSession(args)
    try:
        camera.start()
        camera.capture(args.output, args.frames)
    finally:
        camera.close()


if __name__ == '__main__':
    main()
