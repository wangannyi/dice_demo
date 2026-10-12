"""Web HOME hand reset on the existing CAN connection, without waiting for hand feedback."""
import math
import time

from calibration.web.common import event


def reset_hand(arm, job):
    import nero_revo2_demo as demo
    robot = arm.robot
    target = dict.fromkeys(demo.FINGER_NAMES, 0)
    event(job/'hand_home.json', confirmed=False, status='running', target_0_100=[0]*6)
    command_sent = False
    try:
        baseline = arm.read_joints()['joints_rad']

        def monitor():
            status = robot.get_arm_status()
            if (robot.has_comm_error() or status is None or status.msg.motion_status != 0
                    or status.msg.arm_status != 0 or status.msg.ctrl_mode != 1):
                raise RuntimeError('灵巧手归零时机械臂控制状态异常')
            q = arm.read_joints()['joints_rad']
            if max(abs(a-b) for a, b in zip(q, baseline)) > math.radians(.2):
                raise RuntimeError('灵巧手归零时机械臂发生移动')

        monitor()
        if not hasattr(arm, '_home_hand'):
            arm._home_hand = robot.init_effector(robot.OPTIONS.EFFECTOR.REVO2)
        hand = arm._home_hand
        monitor()
        commanded_at, started = time.time(), time.monotonic()
        hand.position_time_ctrl(mode='pos', **target)
        hand.position_time_ctrl(mode='time', **dict.fromkeys(demo.FINGER_NAMES, 100))
        command_sent = True
        if time.monotonic()-started > .05:
            raise RuntimeError('灵巧手位置和时间指令间隔超过 50 ms')
        deadline = time.monotonic()+1.
        while time.monotonic() < deadline:
            monitor()
            time.sleep(.05)
        monitor()
        proof = dict(confirmed=False, status='completed', command_sent=True,
                     verification='command_only', target_0_100=[0]*6,
                     commanded_at=commanded_at, duration_s=1., completed_at=time.time())
        event(job/'hand_home.json', **proof)
        return proof
    except BaseException as exc:
        event(job/'hand_home.json', confirmed=False, status='failed', target_0_100=[0]*6,
              command_sent=command_sent, verification='command_only',
              error=str(exc) or type(exc).__name__)
        raise
