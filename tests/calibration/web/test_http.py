"""HTTP boundaries: CSRF, readonly controls, file isolation and PIN session scope."""
import http.client
import json
import threading

import pytest

from calibration.web.common import ROOT, pin_record, write
from calibration.web.server import Server, Handler
from calibration.web.service import Service


@pytest.fixture
def http_server(tmp_path):
    service = Service(ROOT/'configs/calibration_workflow.json', tmp_path, readonly=True)
    server = Server(('127.0.0.1', 0), Handler)
    server.service = service
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    service.close()


def request(server, path, method='GET', data=None, headers=None):
    conn = http.client.HTTPConnection(*server.server_address, timeout=3)
    conn.request(method, path, json.dumps(data) if data is not None else None, headers or {})
    response = conn.getresponse()
    payload = response.read()
    result = response.status, dict(response.getheaders()), payload
    conn.close()
    return result


def auth_headers(server):
    code, headers, body = request(server, '/api/status')
    assert code == 200
    session = json.loads(body)
    host = f'{server.server_address[0]}:{server.server_address[1]}'
    return {'Cookie': headers['Set-Cookie'].split(';')[0], 'Origin': 'http://'+host,
            'X-CSRF-Token': session['csrf'], 'X-Control-Tab': 'test_browser_tab_12345',
            'Content-Type': 'application/json'}


def test_status_and_static_start_no_devices(http_server):
    code, headers, body = request(http_server, '/')
    assert code == 200 and '手眼标定'.encode() in body
    assert "frame-ancestors 'none'" in headers['Content-Security-Policy']
    code, _, body = request(http_server, '/api/status')
    assert code == 200 and json.loads(body)['readonly']
    assert http_server.service.active is None


def test_post_requires_origin_csrf_and_tab(http_server):
    headers = auth_headers(http_server)
    for field in ('Origin', 'X-CSRF-Token', 'X-Control-Tab'):
        invalid = dict(headers)
        invalid.pop(field)
        code, _, _ = request(http_server, '/api/start', 'POST', {'action': 'first'}, invalid)
        assert code == 403
    code, _, _ = request(http_server, '/api/start', 'POST', {'action': 'first'}, headers)
    assert code == 409  # readonly, never starts a worker
    assert http_server.service.active is None


def test_no_arbitrary_file_routes_or_dns_rebinding(http_server):
    for path in ('/../../etc/passwd', '/api/image?kind=preview&job=../../engineer.json', '/engineer.json'):
        code, _, _ = request(http_server, path)
        assert code in (400, 404)
    code, _, _ = request(http_server, '/api/status', headers={'Host':'attacker.example'})
    assert code == 400


def test_engineer_unlock_is_session_scoped(http_server):
    write(http_server.service.data/'engineer.json', pin_record('12345678'))
    headers = auth_headers(http_server)
    code, _, _ = request(http_server, '/api/unlock', 'POST', {'pin':'12345678'}, headers)
    assert code == 200
    _, _, body = request(http_server, '/api/status', headers={'Cookie':headers['Cookie']})
    assert json.loads(body)['engineer_unlocked'] is True
    _, _, body = request(http_server, '/api/status')
    assert json.loads(body)['engineer_unlocked'] is False
    request(http_server, '/api/lock', 'POST', {}, headers)
    _, _, body = request(http_server, '/api/status', headers={'Cookie':headers['Cookie']})
    assert json.loads(body)['engineer_unlocked'] is False
