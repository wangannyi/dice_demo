"""Capture profile from vision/camera.json; keep the calibrated RGB stream unchanged."""


def capture_arguments(cfg):
    if cfg.get("pipeline_strategy") != "green_open_cup":
        return []
    from vision.capture.config import load_camera_config
    camera = load_camera_config()
    from cup_grasp_demo.flow.image_profile import profile_options
    color, _, crop = profile_options(dict(
        color_resolution=camera["color_resolution"],
        fps=camera["fps"], crop_xywh=camera["crop_xywh"]))
    resolution = camera["depth_resolution"]
    extra = ["--stereo"] if cfg["green_cup"].get("perception", {}).get("geometry_method") == "stereo_rim" else []
    extra += ['--color-resolution', *map(str, color)]
    if crop is not None:
        extra += ['--crop-xywh', *map(str, crop)]
    return extra + ["--depth-width", str(resolution[0]), "--depth-height", str(resolution[1]),
            "--fps", str(camera["fps"])]
