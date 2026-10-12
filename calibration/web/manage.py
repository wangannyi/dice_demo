"""Start/stop only this repository's standalone service, with PID-reuse protection."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from calibration.web.common import ROOT, read, write


def process_stamp(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('start', 'stop', 'status'))
    p.add_argument('--port', type=int, default=8091)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--readonly', action='store_true')
    p.add_argument('--data', type=Path, default=ROOT/'calibration/datasets/web')
    args = p.parse_args(argv)
    data = args.data.resolve()
    data.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(data, 0o700)
    path = data/'service-process.json'
    record = read(path, {})
    running = bool(record and record.get('start_ticks') is not None
                   and process_stamp(record['pid']) == record['start_ticks'])
    if args.action == 'status':
        print(json.dumps({'running': running, **record}, ensure_ascii=False))
        return 0
    if args.action == 'stop':
        if not running:
            print('本工具服务未运行；未停止任何其他程序')
            return 0
        os.kill(record['pid'], signal.SIGTERM)
        deadline = time.monotonic()+20
        while process_stamp(record['pid']) == record['start_ticks'] and time.monotonic() < deadline:
            time.sleep(.2)
        if process_stamp(record['pid']) == record['start_ticks']:
            raise RuntimeError('服务仍在清理，未强制结束。请检查设备和日志')
        print('本工具服务已退出；作业保留的数据不会删除')
        return 0
    if running:
        print(f"服务已运行，PID {record['pid']}，端口 {record['port']}")
        return 0
    log = data/'server.log'
    with log.open('ab') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'calibration.web.server',
                                   '--host', args.host, '--port', str(args.port), '--data', str(data),
                                   *(['--readonly'] if args.readonly else [])],
                                  cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': process.pid, 'start_ticks': process_stamp(process.pid), 'port': args.port, 'root': str(ROOT)}
    write(path, record)
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError(f'服务启动失败，请查看 {log}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/status', timeout=.5) as response:
                status = json.load(response)
            if status.get('root') != str(ROOT):
                raise RuntimeError('端口已被其他目录的服务占用')
            if status.get('service_pid') != process.pid:
                raise RuntimeError('端口已被另一服务进程占用，请检查服务状态')
            print(f'标定服务已启动：端口 {args.port}，PID {process.pid}；未连接任何设备')
            return 0
        except (OSError, ValueError):
            time.sleep(.2)
    raise RuntimeError(f'服务未就绪，请查看 {log}')


if __name__ == '__main__':
    raise SystemExit(main())
