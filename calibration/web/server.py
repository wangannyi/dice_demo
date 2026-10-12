"""Standalone LAN calibration service; no devices are opened on startup."""
import argparse
import getpass
import http.cookies
import ipaddress
import json
import mimetypes
import os
import re
import secrets
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from calibration.web.common import ROOT, pin_record, write
from calibration.web.service import Conflict, Service

STATIC = Path(__file__).parent/'static'


class Handler(BaseHTTPRequestHandler):
    server_version = 'CalibrationWeb/1'

    def log_message(self, fmt, *args):
        # Session credentials and image polling never belong in application logs.
        pass

    def host(self):
        raw = self.headers.get('Host', '')
        host = urlsplit('//'+raw).hostname
        if host != 'localhost':
            try:
                ipaddress.ip_address(host)
            except ValueError as exc:
                raise ValueError('请通过 K3 的 IP 地址访问') from exc
        return raw

    def send(self, code, payload, content_type='application/json; charset=utf-8', cookie=None):
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        elif isinstance(payload, str):
            payload = payload.encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'same-origin')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self'; "
                         "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self'")
        if cookie:
            self.send_header('Set-Cookie', f'calib_session={cookie}; HttpOnly; SameSite=Strict; Path=/')
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def context(self):
        cookie = http.cookies.SimpleCookie()
        cookie.load(self.headers.get('Cookie', ''))
        token = cookie.get('calib_session')
        token, session = self.server.service.session(token.value if token else None)
        tab = self.headers.get('X-Control-Tab', '')
        if tab and not re.fullmatch(r'[a-zA-Z0-9_-]{16,80}', tab):
            raise ValueError('无效的页面身份')
        return token, session, token+':'+tab

    def do_GET(self):
        try:
            self.host()
            path = urlsplit(self.path)
            if path.path in ('/', '/app.js', '/style.css'):
                file = STATIC/('index.html' if path.path == '/' else path.path[1:])
                self.send(200, file.read_bytes(), mimetypes.guess_type(file)[0]+'; charset=utf-8')
                return
            token, session, owner = self.context()
            if path.path == '/api/status':
                snapshot = self.server.service.snapshot(owner)
                snapshot.update(csrf=session['csrf'], engineer_unlocked=session['engineer_until'] > __import__('time').monotonic())
                self.send(200, snapshot, cookie=token)
            elif path.path == '/api/image':
                query = parse_qs(path.query)
                file = self.server.service.image_path(query.get('kind', ['preview'])[0], query.get('job', [None])[0])
                self.send(200, file.read_bytes(), mimetypes.guess_type(file)[0])
            elif path.path == '/api/result':
                file = self.server.service.selected('result')
                self.send(200, file.read_bytes())
            else:
                self.send(404, {'error': '未找到资源'})
        except FileNotFoundError:
            self.send(404, {'error': '尚无图像或结果'})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.send(400, {'error': str(exc)})

    def do_POST(self):
        try:
            host = self.host()
            # Require the exact origin and a non-simple header. There is no CORS.
            if self.headers.get('Origin') != 'http://'+host:
                raise PermissionError('请求来源不匹配')
            token, session, owner = self.context()
            if not secrets.compare_digest(self.headers.get('X-CSRF-Token', ''), session['csrf']):
                raise PermissionError('页面会话已失效，请刷新页面')
            if not self.headers.get('X-Control-Tab'):
                raise PermissionError('缺少页面控制身份')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8192:
                raise ValueError('请求大小无效')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError('请求必须为 JSON 对象')
            service = self.server.service
            path = urlsplit(self.path).path
            if path == '/api/start':
                result = service.start(body.get('action'), body, owner, session)
            elif path == '/api/heartbeat':
                service.heartbeat(owner)
                result = {'ok': True}
            elif path == '/api/cancel':
                service.cancel(owner)
                result = {'ok': True}
            elif path == '/api/command':
                service.command(owner, body.get('command'))
                result = {'ok': True}
            elif path == '/api/window':
                service.window(body.get('window_xyxy'))
                result = {'ok': True}
            elif path == '/api/unlock':
                service.authenticate(session, body.get('pin'))
                result = {'ok': True}
            elif path == '/api/lock':
                session['engineer_until'] = 0
                result = {'ok': True}
            else:
                self.send(404, {'error': '未知接口'})
                return
            self.send(200, result, cookie=token)
        except PermissionError as exc:
            self.send(403, {'error': str(exc)})
        except Conflict as exc:
            self.send(409, {'error': str(exc)})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.send(400, {'error': str(exc)})


class Server(ThreadingHTTPServer):
    daemon_threads = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/calibration_workflow.json')
    parser.add_argument('--data', type=Path, default=ROOT/'calibration/datasets/web')
    parser.add_argument('--readonly', action='store_true', help='UI preview only; jobs and writes disabled')
    parser.add_argument('--heartbeat-timeout', type=float, default=10.)
    parser.add_argument('--set-pin', action='store_true', help='Set engineer PIN from the K3 terminal')
    parser.add_argument('--init-pin', action='store_true', help='Create a random PIN in a private local file if missing')
    args = parser.parse_args(argv)
    args.data.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.data, 0o700)
    if args.set_pin or args.init_pin:
        target = args.data/'engineer.json'
        if args.set_pin:
            pin = getpass.getpass('新的工程师 PIN（8–20 位数字）: ')
            if pin != getpass.getpass('再次输入: '):
                raise ValueError('两次输入不一致')
            write(target, pin_record(pin))
            (args.data/'initial-pin.txt').unlink(missing_ok=True)
        elif not target.exists():
            pin = ''.join(secrets.choice('0123456789') for _ in range(8))
            write(target, pin_record(pin))
            file = args.data/'initial-pin.txt'
            file.write_text(pin+'\n')
            os.chmod(file, 0o600)
        os.chmod(target, 0o600)
        print('工程师 PIN 已配置；PIN 不输出到日志。')
        return
    service = Service(args.config, args.data, readonly=args.readonly, heartbeat_timeout=args.heartbeat_timeout)
    server = Server((args.host, args.port), Handler)
    server.service = service
    def shutdown(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f'标定服务 http://{args.host}:{args.port}，应用目录 {ROOT}；启动未连接任何设备', flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
