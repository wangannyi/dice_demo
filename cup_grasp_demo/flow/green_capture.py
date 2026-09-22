"""Green-only capture profile; keep the calibrated RGB stream unchanged."""


def capture_arguments(cfg):
    if cfg.get("pipeline_strategy") != "green_open_cup":
        return []
    profile = cfg.get("green_cup", {}).get("camera", {})
    if not isinstance(profile, dict) or set(profile) - {"depth_resolution", "fps", "color_resolution", "crop_xywh"}:
        raise ValueError("Invalid green_cup.camera option")
    from cup_grasp_demo.flow.image_profile import profile_options
    color, _, crop = profile_options(profile)
    resolution = profile.get("depth_resolution", [640, 480])
    fps = profile.get("fps", 15)
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(type(v) is not int for v in resolution)
        or resolution not in ([640, 480], [848, 480], [1280, 720])
        or type(fps) is not int
        or fps not in (6, 15, 30)
    ):
        raise ValueError("Invalid green depth resolution or fps; check device support")
    extra = ["--stereo"] if cfg["green_cup"].get("perception", {}).get("geometry_method") == "stereo_rim" else []
    if 'color_resolution' in profile:
        extra += ['--color-resolution', *map(str, color)]
    if crop is not None:
        extra += ['--crop-xywh', *map(str, crop)]
    return extra + ["--depth-width", str(resolution[0]), "--depth-height", str(resolution[1]),
            "--fps", str(fps)]
