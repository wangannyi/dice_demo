"""Bounded color-only video witness; no CAN access and no camera option writes."""

import argparse
import json
from pathlib import Path
import time


def main():
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    camera = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    started = False
    rows = []
    try:
        camera.start(cfg)
        started = True
        for _ in range(20):
            camera.wait_for_frames(3000)
        # Covers the 60 s configured wave plus bounded SDK setup/stop time.
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not (args.output / "stop").exists():
            frame = camera.wait_for_frames(3000).get_color_frame()
            if not frame:
                raise RuntimeError("Missing color frame")
            image = np.asanyarray(frame.get_data())
            name = f"frame_{len(rows):04d}.jpg"
            if not cv2.imwrite(
                str(args.output / name), image, [cv2.IMWRITE_JPEG_QUALITY, 90]
            ):
                raise RuntimeError("Could not save frame")
            rows.append(
                dict(
                    file=name,
                    epoch_s=time.time(),
                    camera_timestamp_ms=frame.get_timestamp(),
                )
            )
            if len(rows) == 1:
                (args.output / "ready").write_text(str(time.time()))
    finally:
        if started:
            camera.stop()
        (args.output / "frames.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
