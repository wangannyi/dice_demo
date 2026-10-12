"""HTTP boundaries: CSRF, readonly controls, file isolation and PIN session scope."""
import http.client
import json
import threading
from urllib.parse import urlencode

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


def gallery_dataset(server):
    service = server.service
    path = service.data/'datasets'/'saved'
    path.mkdir(parents=True)
    write(path/'manifest.json', {'schema': 1, 'mode': 'eye_to_hand'})
    write(service.state_path, {'dataset': str(path)})
    write(path/'sample_0000.json', {'quality': {'corners': 12, 'reprojection_rms_px': .12}})
    (path/'sample_0000.png').write_bytes(b'raw image')
    (path/'sample_0000_detected.png').write_bytes(b'annotated image')
    return path


def test_gallery_supports_historical_samples_and_missing_images(http_server):
    code, _, body = request(http_server, '/api/samples')
    assert code == 200 and json.loads(body)['samples'] == []
    path = gallery_dataset(http_server)
    _, _, body = request(http_server, '/api/samples')
    listing = json.loads(body)
    assert len(listing['samples']) == 1
    frame = listing['samples'][0]['frames'][0]
    assert frame['id'] == 'selected'
    for kind, expected in [('raw', b'raw image'), ('detected', b'annotated image')]:
        code, headers, body = request(http_server, frame[kind])
        assert code == 200 and body == expected
        assert headers['Content-Type'] == 'image/png'
    code, _, body = request(http_server, frame['data'])
    assert code == 200 and json.loads(body)['quality']['corners'] == 12
    (path/'sample_0000_detected.png').unlink()
    _, _, body = request(http_server, '/api/samples')
    assert json.loads(body)['samples'][0]['frames'][0]['detected'] is None
    assert http_server.service.active is None


def test_gallery_frame_selection_and_file_boundaries(http_server):
    path = gallery_dataset(http_server)
    archive = http_server.service.data/'jobs/20261012_120000_aabbccdd/capture_frames/attempt_0000'
    archive.mkdir(parents=True)
    write(path/'sample_0000.json', {'frame_archive': str(archive), 'selected_frame_index': 1})
    for index in range(3):
        write(archive/f'frame_{index:03d}.json', {'frame_index': index, 'status': 'detected'})
        (archive/f'frame_{index:03d}.png').write_bytes(b'raw')
        (archive/f'frame_{index:03d}_detected.png').write_bytes(b'detection')
    _, _, body = request(http_server, '/api/samples')
    listing = json.loads(body)
    frames = listing['samples'][0]['frames']
    assert len(frames) == 4 and '被选中' in frames[2]['label']
    code, _, body = request(http_server, frames[2]['data'])
    assert code == 200 and json.loads(body)['frame_index'] == 1
    params = dict(sample='sample_0000', frame='selected', key=listing['key'], kind='data')
    for invalid in ({'sample': '../../engineer'}, {'frame': '../config'},
                    {'kind': 'config'}, {'key': 'stale-dataset-key'}):
        assert request(http_server, '/api/sample-file?'+urlencode(dict(params, **invalid)))[0] == 400
    secret = http_server.service.data/'engineer.json'
    write(secret, {'secret': 'not a sample'})
    (archive/'frame_000.json').unlink()
    (archive/'frame_000.json').symlink_to(secret)
    assert request(http_server, frames[1]['data'])[0] == 404
    write(path/'sample_0000.json', {'frame_archive': str(http_server.service.data)})
    assert request(http_server, frames[2]['data'])[0] == 400
