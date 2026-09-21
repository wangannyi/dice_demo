"""Collect all controller limits without fourteen serial polling waits."""
import time


def read_limits(robot, *, sleep=time.sleep, monotonic=time.monotonic):
    readers = [(getter, joint) for joint in range(1, 8)
               for getter in (robot.get_joint_angle_vel_limits, robot.get_joint_acc_limits)]
    values = [None] * len(readers)
    # Space query frames to avoid saturating the adapter's short transmit queue.
    for index, (getter, joint) in enumerate(readers):
        values[index] = getter(joint, timeout=0., min_interval=0.)
        sleep(.001)
    deadline = monotonic() + .5
    while any(value is None for value in values):
        for index, (getter, joint) in enumerate(readers):
            if values[index] is None:
                # SDK throttling retains each outstanding request until its reply.
                values[index] = getter(joint, timeout=0., min_interval=1.)
        if all(value is not None for value in values):
            break
        if monotonic() >= deadline:
            missing = [f'J{i//2+1} {"angle/velocity" if i%2 == 0 else "acceleration"}'
                       for i, value in enumerate(values) if value is None]
            raise TimeoutError('Missing live controller limits: ' + ', '.join(missing))
        sleep(.001)
    return list(zip(values[::2], values[1::2]))
