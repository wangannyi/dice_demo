"""Camera configuration loader; the single source of truth for acquisition."""
import json
from pathlib import Path
import hashlib

CAMERA_CONFIG = Path(__file__).resolve().parents[1] / "camera.json"


def load_camera_config():
    """Read vision/camera.json; validate structure and calibration binding."""
    raw = json.loads(CAMERA_CONFIG.read_text())
    required = {"serial", "color_resolution", "depth_resolution", "fps",
                "crop_xywh", "warmup_frames", "fresh_discard_frames",
                "calibration_file"}
    unknown = set(raw) - required
    if unknown:
        raise ValueError(f"Unknown camera.json keys: {sorted(unknown)}")
    for key in ("color_resolution", "depth_resolution", "crop_xywh"):
        value = raw[key]
        if (not isinstance(value, list) or len(value) != (2 if "resolution" in key else 4)
                or any(type(x) is not int for x in value)):
            raise ValueError(f"camera.json {key} must be integer array")
    if raw["color_resolution"] not in ([640, 480], [1280, 720]):
        raise ValueError("camera.json color_resolution must be [640,480] or [1280,720]")
    if raw["depth_resolution"] not in ([640, 480], [848, 480], [1280, 720]):
        raise ValueError("camera.json depth_resolution must be 640x480/848x480/1280x720")
    if type(raw["fps"]) is not int or raw["fps"] not in (6, 15, 30):
        raise ValueError("camera.json fps must be 6/15/30")
    if not isinstance(raw["serial"], str) or not raw["serial"].strip():
        raise ValueError("camera.json serial must be non-empty string")
    for key in ("warmup_frames", "fresh_discard_frames"):
        value = raw[key]
        if type(value) is not int or not 0 <= value <= (60 if "warmup" in key else 5):
            raise ValueError(f"camera.json {key} out of range")
    return raw


def calibration_digest():
    """sha256 of the calibration file bound in camera.json."""
    raw = load_camera_config()
    calibration = Path(__file__).resolve().parents[2] / raw["calibration_file"]
    if not calibration.is_file():
        raise ValueError(f"camera.json calibration_file not found: {calibration}")
    return hashlib.sha256(calibration.read_bytes()).hexdigest()
