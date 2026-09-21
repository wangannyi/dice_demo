"""Explicit Nero v1.20 acceleration configuration; no motion or mode commands."""

import argparse
import hashlib
import json
import math
import signal
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'rgb_hand_tracking')]
from cup_grasp_demo.calibration_debug.shake_readback import core, is_read_only_request, query_joint_limit
from cup_grasp_demo.calibration_debug.joint_profile import options


def acceleration_frame(joint, value):
    raw = round(value * 100)
    if type(joint) is not int or not 1 <= joint <= 7 or not 1 <= raw <= 500:
        raise ValueError('Invalid acceleration write')
    return bytes([joint, 0, 0xAE, raw >> 8, raw & 255, 0, 0, 0])


class AccelerationGuard(core.AuditedSendGuard):
    expected_write = None

    def install(self):
        if self.installed:
            return
        super().install()
        audited = self.bus_class.send

        def send(bus, message, *args, **kwargs):
            ident, data = message.arbitration_id, bytes(message.data)
            if not is_read_only_request(ident, data) and not (
                ident == 0x475 and self.expected_write is not None
                and data == self.expected_write
            ):
                raise RuntimeError('Acceleration guard rejected CAN ' + hex(ident))
            return audited(bus, message, *args, **kwargs)

        self.bus_class.send = send


def set_acceleration_v120(robot, joint, value):
    """Use public setter with documented 0.01 wire unit, scoped to this instance.

    Nero's inherited setter uses x10000 despite the 0x475 specification and
    v1.20 correction note specifying x100. Keep the API argument in rad/s²;
    normalize only the message field and independently read back afterwards.
    Never modify the installed SDK, feedback conversion or other commands.
    """
    factory = robot._MSG_JointConfig
    captured = []

    def corrected(*args, **kwargs):
        if args or set(kwargs) != {'joint_index', 'acc_param_config_is_effective_or_not', 'max_joint_acc'}:
            raise ValueError('Unexpected SDK joint configuration signature')
        if (kwargs['joint_index'] != joint
                or kwargs['acc_param_config_is_effective_or_not'] != 0xAE
                or kwargs['max_joint_acc'] not in (round(value * 100), round(value * 10000))):
            raise ValueError('Unexpected SDK acceleration encoding')
        captured.append(kwargs['max_joint_acc'])
        return factory(**dict(kwargs, max_joint_acc=round(value * 100)))

    robot._MSG_JointConfig = corrected
    try:
        acknowledged = robot.set_joint_acc_limits(joint_index=joint, max_joint_acc=value, timeout=1.0)
    finally:
        robot._MSG_JointConfig = factory
    return dict(sdk_return=acknowledged, sdk_encoded_values=captured,
                transmitted_value=round(value * 100), wire_unit_rad_s2=.01)


def read_accelerations(robot):
    values = []
    for i in range(1, 8):
        reply = query_joint_limit(robot.get_joint_acc_limits, i)
        if reply is None or not math.isfinite(reply.msg.max_joint_acc) or reply.msg.max_joint_acc <= 0:
            raise RuntimeError(f'J{i} 加速度限值查询失败；未完成原值备份，禁止写入')
        values.append(reply.msg.max_joint_acc)
    return values


def stationary(session, start=None):
    rows, evidence = core.stopped_window(session)
    for row in rows:
        if core.ready_blockers(row, take_can_control=True):
            raise RuntimeError('机械臂必须已使能、无故障并停稳')
        if start is not None and max(abs(a-b) for a,b in zip(row['q_rad'], start)) > math.radians(.1):
            raise RuntimeError('关节姿态改变，停止参数设置')
    return rows[-1], evidence


def apply_targets(robot, guard, targets, before, record, save, check_stationary):
    for joint, value in enumerate(targets, 1):
        check_stationary()
        item = dict(joint=joint, requested_rad_s2=value, before_rad_s2=before[joint-1],
                    write_attempted=False, verified=False)
        record['joints'].append(item)
        if abs(before[joint-1] - value) > .005:
            guard.expected_write = acceleration_frame(joint, value)
            item['write_attempted'] = True
            save()  # Persist intent before the SDK can send a parameter write.
            try:
                item.update(set_acceleration_v120(robot, joint, value))
            finally:
                guard.expected_write = None
        reply = query_joint_limit(robot.get_joint_acc_limits, joint)
        item['readback_rad_s2'] = None if reply is None else reply.msg.max_joint_acc
        item['verified'] = (reply is not None and math.isfinite(reply.msg.max_joint_acc)
                            and abs(reply.msg.max_joint_acc - value) <= .005)
        save()
        if not item['verified']:
            raise RuntimeError(f'J{joint} 读回 {item["readback_rad_s2"]} 与目标 {value} 不符；停止后续写入，查看原值备份')


def run(config, channel, directory, execute=False):
    targets = options(config)['controller_acceleration_rad_s2']
    if targets is None:
        raise ValueError('先配置 controller_acceleration_rad_s2，顺序为 J1..J7')
    report = dict(success=False, executed=execute, targets_rad_s2=targets, joints=[],
                  motion_commands_sent=0, control_mode_commands_sent=0,
                  parameter_writes_verified=False)
    output = directory / 'actual.json'
    def save():
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temp.replace(output)
    bus, factory = core.load_sdk_runtime(channel)
    guard = AccelerationGuard(bus)
    session = core.PassivePoseSession(bus, factory)
    session.guard = guard
    try:
        host_before = core.host_control_evidence(channel)
        report['runtime'] = session.start()
        conflicts = core.evidence_blockers(host_before, core.host_control_evidence(channel))
        if conflicts:
            raise RuntimeError('; '.join(conflicts))
        initial, report['stationarity_before'] = stationary(session)
        report['initial'] = initial
        guard.permit()
        report['firmware'] = session.robot.get_firmware(timeout=1, min_interval=0)
        if not isinstance(report['firmware'], dict) or report['firmware'].get('software_version') != '1.20':
            raise RuntimeError('本命令的单位兼容仅验证 Nero 固件 1.20；未写入')
        before = read_accelerations(session.robot)
        report['before_rad_s2'] = before
        backup = dict(firmware=report['firmware'], channel=channel,
                      controller_acceleration_rad_s2=before, recorded_epoch_s=time.time())
        (directory / 'before.json').write_text(json.dumps(backup, indent=2) + '\n')
        save()
        if execute:
            if initial['status']['ctrl_mode'] != 1:
                raise RuntimeError('当前不是 CAN 模式；本参数命令不自动切换控制权，未写入')
            apply_targets(session.robot, guard, targets, before, report, save,
                          lambda: stationary(session, initial['q_rad']))
            report['after_rad_s2'] = read_accelerations(session.robot)
            if any(abs(a-b) > .005 for a,b in zip(report['after_rad_s2'], targets)):
                raise RuntimeError('最终七轴读回不一致，禁止进行运动测试')
            stationary(session, initial['q_rad'])
            report['parameter_writes_verified'] = True
        report['success'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        guard.expected_write = None
        guard.allowed = False
        session.close()
        report['tx'] = guard.report()
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--channel', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise RuntimeError('停止参数设置，信号 ' + str(signum))
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    if not args.channel.isalnum():
        raise ValueError('Invalid channel')
    raw = args.config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError('配置已变化，未写入')
    if (args.output / 'actual.json').exists() or (args.output / 'before.json').exists():
        raise FileExistsError('参数收据目录已使用')
    args.output.mkdir(parents=True, exist_ok=True)
    with core.control_lock(Path('/tmp/nero_' + args.channel + '_control.lock')):
        report = run(json.loads(raw), args.channel, args.output, args.execute)
    print(json.dumps({k: report.get(k) for k in (
        'success', 'executed', 'targets_rad_s2', 'before_rad_s2', 'after_rad_s2',
        'parameter_writes_verified', 'error')}, ensure_ascii=False))
    return 0 if report['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
