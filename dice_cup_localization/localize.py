"""Offline CLI for same-frame masks and registered RGB-D depth."""

import argparse
import json
from pathlib import Path

import numpy as np

from geometry import localize


def main():
    """Read an NPZ bundle and write one geometry result; never open hardware."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path, help='NPZ: depth, object_mask, table_mask')
    parser.add_argument('metadata', type=Path, help='Acquisition and color intrinsics JSON')
    args = parser.parse_args()
    try:
        metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
        with np.load(args.bundle, allow_pickle=False) as bundle:
            result = localize(bundle['depth'], bundle['object_mask'], bundle['table_mask'], metadata)
        print(json.dumps(result, allow_nan=False))
        return 0 if result['valid'] else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'valid': False, 'geometry': None, 'reason': str(exc)}, allow_nan=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
