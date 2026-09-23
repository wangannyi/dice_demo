"""Shared, explicitly requested arm HOME stage for calibration workflows."""
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_HOME = Path(__file__).resolve().parents[1] / 'configs/actions/home.json'


def load_home(path=DEFAULT_HOME):
    data = json.loads(Path(path).read_text())
    q = np.asarray(data['joints_rad'], dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('HOME must contain seven finite joints_rad')
    if 'joints_deg' in data:
        degrees = np.asarray(data['joints_deg'], dtype=float)
        if degrees.shape != (7,) or not np.allclose(np.degrees(q), degrees, atol=.01, rtol=0):
            raise ValueError('HOME joints_deg and joints_rad disagree')
    return q.tolist()


def joint_route(start, target, model, step_deg=2.):
    """Check the complete joint interpolation; this is not collision checking."""
    from auto_collect import interpolate
    if not math.isfinite(step_deg) or step_deg <= 0:
        raise ValueError('HOME approach step must be finite and positive')
    route = [list(start)] + interpolate(start, target, step_deg)
    if len(model.joints) != 7:
        raise ValueError('HOME requires a seven-axis arm model')
    for q in route:
        if any(not joint.lower_rad <= angle <= joint.upper_rad
               for joint, angle in zip(model.joints, q)):
            raise ValueError('HOME/approach exceeds joint limits')
    return route


def move_home(arm, route, speed_deg_s, acc_deg_s2, preview=None):
    from auto_collect import move_and_watch, require_arm_ready, stream_smooth_route
    require_arm_ready(arm)
    if preview is not None:
        preview.status('HOME', 0)
    pump = None if preview is None else preview.pump
    print(json.dumps({'phase': 'HOME', 'status': 'started'}), flush=True)
    stream_smooth_route(arm, route, speed_deg_s, acc_deg_s2, pump=pump)
    move_and_watch(arm, None, None, None, None, None, 0, route[-1],
                   watch_board=False, pump=pump)
    actual = np.asarray(arm.read_joints()['joints_rad'], dtype=float)
    if actual.shape != (7,) or not np.isfinite(actual).all():
        raise RuntimeError('HOME feedback must contain seven finite joint angles')
    actual = actual.tolist()
    error = math.degrees(max(abs(a-b) for a, b in zip(actual, route[-1])))
    if error > .8:
        raise RuntimeError('HOME feedback did not confirm arrival')
    print(json.dumps({'phase': 'HOME', 'status': 'completed', 'joint_error_deg': error}), flush=True)
    return actual


def return_home(home_path=DEFAULT_HOME, channel='can0', speed_percent=15,
                speed_deg_s=4., acc_deg_s2=6.):
    """Move only the arm, settle, then release CAN; never change the fingers."""
    from auto_collect import ensure_can_control
    from kinematics import load_model
    from sensors import NeroFeedback
    if (not 1 <= speed_percent <= 100 or not math.isfinite(speed_deg_s)
            or speed_deg_s <= 0 or not math.isfinite(acc_deg_s2) or acc_deg_s2 <= 0):
        raise ValueError('Invalid HOME motion speed/acceleration')
    home = load_home(home_path)
    model = load_model()
    joint_route(home, home, model)
    arm = NeroFeedback(channel)
    previous_auto_mode = None
    try:
        route = joint_route(arm.read_joints()['joints_rad'], home, model)
        ensure_can_control(arm)
        robot = arm.robot
        robot.set_joint_limits_enabled(True)
        robot.set_speed_percent(speed_percent)
        previous_auto_mode = robot.get_auto_set_motion_mode_enabled()
        robot.set_auto_set_motion_mode_enabled(False)
        robot.set_motion_mode('js')
        return move_home(arm, route, speed_deg_s, acc_deg_s2)
    finally:
        try:
            if previous_auto_mode is not None:
                arm.robot.set_auto_set_motion_mode_enabled(previous_auto_mode)
        finally:
            arm.close()
