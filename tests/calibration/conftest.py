"""Import path for calibration package tests.

calibration/ modules import each other by bare module name (e.g.
``from core import ...``); that only resolves with the calibration
directory itself on sys.path.  Real runs get it from executing scripts
inside calibration/, while these tests import ``calibration.xxx`` as a
package -- append the source directory so both forms coexist.
"""
import sys
from pathlib import Path

CALIBRATION_DIR = Path(__file__).resolve().parents[2] / 'calibration'
if str(CALIBRATION_DIR) not in sys.path:
    sys.path.append(str(CALIBRATION_DIR))
