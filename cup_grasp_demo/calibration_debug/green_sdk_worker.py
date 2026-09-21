"""One sequential SDK connection, owned by one FAST workflow until workflow exit."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'nero_revo2_control'),
                 str(ROOT / 'rgb_hand_tracking'), str(Path(__file__).parent)]


def interrupted(signum, frame):
    raise KeyboardInterrupt('SDK worker interrupted')


def retryable_shake_start(command, report):
    """Only a rejected, transmission-free start can reuse this connection."""
    return (command == 'shake'
            and report.get('failure_code') == 'start_position_changed'
            and report.get('motion_attempted') is False
            and report.get('finger_commands_sent') == 0
            and report.get('parameter_write_commands_sent') == 0
            and report.get('tx', {}).get('actual_tx_count') == 0
            and report.get('tx', {}).get('transmission_outcome_uncertain') is False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', required=True)
    args = parser.parse_args()
    if not args.channel.isalnum():
        raise ValueError('Invalid CAN channel')
    import nero_revo2_demo as demo
    from visual_servo_probe import control_lock, host_control_evidence, evidence_blockers
    from cup_grasp_demo.calibration_debug import hardware, joint_execution
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    robot = None
    with control_lock('/tmp/nero_' + args.channel + '_control.lock'):
        try:
            with redirect_stdout(sys.stderr):
                before = host_control_evidence(args.channel)
                robot = demo.create_robot(args.channel)
                hand = robot.init_effector(robot.OPTIONS.EFFECTOR.REVO2)
                robot.connect()
                conflicts = evidence_blockers(before, host_control_evidence(args.channel))
                if conflicts:
                    raise RuntimeError('; '.join(conflicts))
            print(json.dumps({'ready': True}), flush=True)
            for line in sys.stdin:
                message = json.loads(line)
                if message.get('command') == 'close':
                    break
                if message.get('command') not in ('snapshot', 'run', 'shake'):
                    raise ValueError('Unknown SDK worker request')
                output = Path(message['output'])
                argv = [message['command'], '--channel', args.channel, '--output', str(output)]
                if message['command'] == 'snapshot' and message.get('cached_snapshot', False):
                    argv.append('--cached-snapshot')
                if message['command'] in ('run', 'shake'):
                    argv += ['--request', message['request'], '--sha256', message['sha256']]
                # Keep stdout exclusively for the RPC protocol; retain original receipts.
                with output.with_suffix('.log').open('x') as log, redirect_stdout(log), redirect_stderr(log):
                    if message['command'] == 'shake':
                        raw = Path(message['request']).read_bytes()
                        if hashlib.sha256(raw).hexdigest() != message['sha256']:
                            raise ValueError('Shake request changed')
                        if output.exists():
                            raise FileExistsError(output)
                        report = joint_execution.run(json.loads(raw), connected=robot, connection_evidence=before)
                        output.write_text(json.dumps(report, allow_nan=False) + '\n')
                        code = 0 if report.get('success') else 2
                    else:
                        code = hardware.main(argv, connected=(robot, hand, before))
                print(json.dumps({'output': str(output), 'returncode': code}), flush=True)
                if code and not (message['command'] == 'shake' and retryable_shake_start('shake', report)):
                    break  # Failed motion must never leave a reusable executor.
        finally:
            if robot is not None:
                robot.disconnect()


if __name__ == '__main__':
    main()
