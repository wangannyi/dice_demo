"""Overlapping feedback gesture on the existing single SDK/CAN owner thread."""
import math
import time


class HandSchedule:
    """Start hand delay relative to the first actual arm stream command."""
    def __init__(self, delay, duration, send, *, clock=time.monotonic):
        self.delay, self.duration, self.send, self.clock = delay, duration, send, clock
        self.started = self.sent = None

    def tick(self):
        now = self.clock()
        if self.started is None:
            self.started = now
        if self.sent is None and now - self.started >= self.delay:
            self.send()
            self.sent = self.clock()

    def complete(self):
        return self.sent is not None and self.clock() - self.sent >= self.duration



class SequenceSchedule:
    """Dispatch at most one pose per tick; never burst missed transitions."""
    def __init__(self, delay, duration, targets, interval, send, *, clock=time.monotonic):
        self.delay, self.duration, self.targets, self.interval = delay, duration, targets, interval
        self.send, self.clock = send, clock
        self.started = self.sent = self.last_sent = None
        self.index = 0

    def tick(self):
        now = self.clock()
        if self.started is None:
            self.started = now
        due = self.started+self.delay if self.last_sent is None else self.last_sent+self.interval
        if self.index < len(self.targets) and now >= due:
            self.send(self.targets[self.index])
            self.last_sent = self.clock()
            if self.sent is None:
                self.sent = self.last_sent
            self.index += 1

    def complete(self):
        return (self.index == len(self.targets) and self.last_sent is not None
                and self.clock()-self.last_sent >= self.duration)


def execute(plan, cfg, robot, hand, demo, result, arm_step, motion_robot):
    target = plan['target_0_100']
    delay = plan['hand_delay_s']
    duration = cfg['green_cup']['finger_duration_s']
    maximum = cfg['green_cup'].get('feedback_hand_max_speed', False)
    if type(maximum) is not bool:
        raise ValueError('feedback_hand_max_speed must be boolean')
    if len(plan['stages']) != 1:
        raise ValueError('Feedback overlap requires one arm stage')
    if len(target) != 6 or any(type(x) is not int or not 0 <= x <= 100 for x in target):
        raise ValueError('Invalid feedback hand target')
    for value, low, high in ((delay, 0, 30), (duration, .65 if maximum else .5, 5 if maximum else 2.55)):
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid feedback timing')
    from cup_grasp_demo.calibration_debug.feedback_sequence import sequence_values
    targets, interval = sequence_values(plan.get('hand_sequence'), target, duration)
    demo.require_right_hand(hand, True)
    def monitor():
        _, _, status = demo.arm_snapshot(robot)
        if status.arm_status != 0 or status.ctrl_mode != 1 or robot.get_joints_enable_status_list() != [True]*7:
            raise RuntimeError('Feedback gesture lost healthy CAN control')

    def send(pose=target):
        # Callback runs in the arm streaming thread: no second CAN owner, no
        # blocking hand-position wait, and no interleaving position/time frames.
        result['finger_commands_sent'] = True
        started = time.monotonic()
        hand.position_time_ctrl(mode='pos', **dict(zip(demo.FINGER_NAMES, pose)))
        hand.position_time_ctrl(mode='time', **dict.fromkeys(demo.FINGER_NAMES, 0 if maximum else round(duration*100)))
        result.setdefault('hand_started_epoch_s', time.time())
        result.setdefault('hand_sequence_events', []).append(dict(target_0_100=list(pose), sent_epoch_s=time.time()))
        if time.monotonic()-started > .05:
            raise RuntimeError('Revo2 position/time commands exceeded 50 ms')

    schedule = (HandSchedule(delay, duration, send) if plan.get('hand_sequence') is None
                else SequenceSchedule(delay, duration, targets, interval, send))
    previous = motion_robot.on_motion_tick
    motion_robot.on_motion_tick = schedule.tick
    try:
        arm_step(plan['stages'][0], cfg['speed_percent'])
        if schedule.started is None:
            raise RuntimeError('No arm stream command; hand not dispatched')
        while not schedule.complete():
            monitor()
            schedule.tick()
            if not schedule.complete():
                time.sleep(.01)
        result['hand_command'] = dict(target_0_100=target, duration_s=duration,
            requested_delay_s=delay, actual_delay_s=schedule.sent-schedule.started,
            speed_mode='max' if maximum else 'timed', command_time_ticks=0 if maximum else round(duration*100),
            completion_basis='elapsed_command_duration_only',
            sequence_commands=len(targets), final_target_0_100=targets[-1])
    finally:
        motion_robot.on_motion_tick = previous
