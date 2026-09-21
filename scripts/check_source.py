"""Read-only source validation; does not import robot modules or open devices."""
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.deps', 'runtime', 'datasets', 'output', 'diagnostics',
        'dist', '__pycache__', 'build', 'CMakeFiles', 'artifacts',
        'spacemitk3-dice_demo', 'tools'}


def main():
    errors = []
    count = 0
    for p in ROOT.rglob('*'):
        parts = p.relative_to(ROOT).parts
        if not p.is_file() or any(x in SKIP or x.startswith('.venv') for x in parts):
            continue
        if p.suffix not in ('.py', '.sh', '.json'):
            continue
        # Optional kernel/vendor trees have independent build requirements.
        if parts[0] in ('agx_arm_ros', 'kernel_usbcan_20260921'):
            continue
        try:
            text = p.read_text()
            if p.suffix == '.py':
                ast.parse(text, filename=str(p))
            elif p.suffix == '.json':
                json.loads(text)
            else:
                subprocess.run(['bash', '-n', str(p)], check=True, capture_output=True)
            count += 1
        except Exception as exc:
            errors.append(f'{p.relative_to(ROOT)}: {exc}')
    print(f'Checked {count} Python/JSON/Shell files; {len(errors)} failures')
    for error in errors:
        print(error, file=sys.stderr)
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
