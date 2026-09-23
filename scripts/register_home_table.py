#!/usr/bin/env python3
"""Bind a checked table capture to the active calibration and enable Pipeline."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cup_grasp_demo.flow.planar_scene import verify


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                     prefix=path.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    try:
        if path.exists():
            os.chmod(temporary, path.stat().st_mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def register(config, table_scene):
    config = Path(config)
    record, cfg = verify(Path(table_scene), config)
    output = ROOT / cfg['green_cup']['home_table_scene']
    value = {
        'scene': record['scene'],
        'calibration_sha256': hashlib.sha256(Path(cfg['calibration']).read_bytes()).hexdigest(),
        'source': str(Path(table_scene).resolve()),
        'requires_fixed_base_and_table': True,
    }
    if output.exists():
        output.with_suffix('.json.bak').write_bytes(output.read_bytes())
    atomic_json(output, value)
    cfg['green_cup']['installation_requires_calibration'] = False
    atomic_json(config, cfg)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--table-scene', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(register(args.config, args.table_scene))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
