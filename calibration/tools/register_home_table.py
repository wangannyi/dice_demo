#!/usr/bin/env python3
"""Bind a verified table capture to the active calibration and enable Pipeline."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from calibration.apply_result import atomic_bytes  # noqa: E402
from cup_grasp_demo.flow.planar_scene import verify  # noqa: E402


def register(config, table_scene):
    config = Path(config)
    record, checked = verify(Path(table_scene), config)
    output = ROOT / checked['green_cup']['home_table_scene']
    value = {
        'scene': record['scene'],
        'calibration_sha256': hashlib.sha256(Path(checked['calibration']).read_bytes()).hexdigest(),
        'source': str(Path(table_scene).resolve()),
        'requires_fixed_base_and_table': True,
    }
    # Preserve the user's relative paths and all unrelated configuration fields.
    cfg = json.loads(config.read_text())
    cfg['green_cup']['installation_requires_calibration'] = False
    table_payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    config_payload = (json.dumps(cfg, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    atomic_bytes(output, table_payload)
    # Leave the gate closed if writing the checked table failed.
    atomic_bytes(config, config_payload)
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
