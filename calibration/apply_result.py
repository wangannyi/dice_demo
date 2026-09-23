#!/usr/bin/env python3
"""Install one hand-eye result and require a fresh table registration."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/green_cup.json'
DEFAULT_DESTINATION = ROOT / 'configs/calibration/handeye_result.json'


def load_result(path):
    data = json.loads(Path(path).read_text())
    if data.get('mode') != 'eye_to_hand':
        raise ValueError('Calibration result mode must be eye_to_hand')
    matrix = data.get('T_base_camera')
    if not (isinstance(matrix, list) and len(matrix) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in matrix)):
        raise ValueError('Calibration result must contain a 4x4 T_base_camera')
    return data


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        if path.exists():
            os.chmod(temporary, path.stat().st_mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def install(result, config=DEFAULT_CONFIG, destination=DEFAULT_DESTINATION):
    result, config, destination = map(Path, (result, config, destination))
    load_result(result)
    payload = result.read_bytes()
    cfg = json.loads(config.read_text())
    try:
        relative = destination.resolve().relative_to(ROOT)
    except ValueError as exc:
        raise ValueError('Calibration destination must be inside the repository') from exc
    cfg['calibration'] = str(relative)
    cfg['green_cup']['installation_requires_calibration'] = True
    atomic_bytes(destination, payload)
    atomic_bytes(config, (json.dumps(cfg, ensure_ascii=False, indent=2) + '\n').encode())
    return {
        'calibration': str(destination.resolve()),
        'sha256': hashlib.sha256(payload).hexdigest(),
        'quality_passed': load_result(destination).get('quality_passed'),
        'table_registration_required': True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--destination', type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(install(args.result, args.config, args.destination), ensure_ascii=False, indent=2))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
