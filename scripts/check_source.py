"""Read-only source validation; does not import robot modules or open devices."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.deps', 'runtime', 'datasets', 'output', 'diagnostics',
        'dist', '__pycache__', 'build', 'CMakeFiles', 'artifacts',
        'spacemitk3-dice_demo', 'tools'}
FORBIDDEN_PATH_TEXT = ('/home/' + 'test2/', '/home/' + 'anny/', '.venv-' + 'grasp',
                       'agilex-' + 'api-test', '/usr/lib/' + 'python')
RUNTIME_TREES = {'scripts', 'calibration', 'cup_grasp_demo', 'vision',
                 'nero_revo2_control'}
# Byte-identical copies kept on purpose (calibration/ stays independently
# deliverable, the K3 model tree keeps its xacro material). Drift between the
# two halves silently forks behavior; the check fails loudly instead.
DUPLICATE_PAIRS = (
    ('calibration/core.py', 'cup_grasp_demo/flow/transforms.py'),
    ('calibration/image_profile.py', 'cup_grasp_demo/flow/image_profile.py'),
)


def check_duplicate_pairs(errors):
    for first, second in DUPLICATE_PAIRS:
        left, right = ROOT / first, ROOT / second
        for path in (left, right):
            if not path.is_file():
                errors.append(f'{path.relative_to(ROOT)}: missing duplicate-pair member')
        if left.is_file() and right.is_file():
            left_hash = hashlib.md5(left.read_bytes()).hexdigest()
            right_hash = hashlib.md5(right.read_bytes()).hexdigest()
            if left_hash != right_hash:
                errors.append(f'{first} and {second} drifted apart '
                              f'(md5 {left_hash[:12]} vs {right_hash[:12]}); '
                              'restore byte-identical copies or re-converge deliberately')


def main():
    errors = []
    count = 0
    check_duplicate_pairs(errors)
    for p in ROOT.rglob('*'):
        parts = p.relative_to(ROOT).parts
        if not p.is_file() or any(x in SKIP or x.startswith('.venv') for x in parts):
            continue
        if p.suffix not in ('.py', '.sh', '.json', '.md', '.txt', '.yaml'):
            continue
        # Optional kernel/vendor trees have independent build requirements.
        if parts[0] in ('agx_arm_ros', 'kernel_usbcan_20260921'):
            continue
        try:
            text = p.read_text()
            # Installation metadata and historical documentation retain their
            # evidence paths. Executable project code must be relocatable.
            if (p.suffix in ('.py', '.sh') and parts[0] in RUNTIME_TREES
                    and 'fixtures' not in parts):
                for marker in FORBIDDEN_PATH_TEXT:
                    if marker in text:
                        raise ValueError(f'machine-specific path is forbidden: {marker}')
            if p.suffix == '.py':
                ast.parse(text, filename=str(p))
            elif p.suffix == '.json':
                json.loads(text)
            elif p.suffix == '.sh':
                subprocess.run(['bash', '-n', str(p)], check=True, capture_output=True)
            count += 1
        except Exception as exc:
            errors.append(f'{p.relative_to(ROOT)}: {exc}')
    print(f'Checked {count} source/config/document files; {len(errors)} failures')
    for error in errors:
        print(error, file=sys.stderr)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
