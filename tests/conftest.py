"""Make every tests/ subdirectory importable by bare module name.

Test modules import each other across subdirectories (e.g. flow
tests reuse fixtures from nero_revo2_control tests).  Inserting all test
subdirectories up front keeps those imports deterministic instead of relying
on pytest's per-directory sys.path insertion order.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for directory in sorted(p for p in HERE.rglob("*") if p.is_dir()):
    absolute = str(directory)
    if absolute not in sys.path:
        sys.path.append(absolute)
