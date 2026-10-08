#!/usr/bin/env python3
"""Capture a checked red-cloth table plane; never command the arm or hand."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cup_grasp_demo.flow import planar_scene  # noqa: E402
from cup_grasp_demo.flow.core import load_config  # noqa: E402
from cup_grasp_demo.flow.session_storage import session_lock  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    args = parser.parse_args(argv)
    load_config(args.config)
    with session_lock(args.session):
        return planar_scene.capture(SimpleNamespace(config=args.config, session=args.session)) or 0


if __name__ == '__main__':
    raise SystemExit(main())
