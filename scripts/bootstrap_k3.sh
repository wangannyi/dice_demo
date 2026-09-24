#!/usr/bin/env bash
# Install the K3 runtime into the system Python; no virtualenv is created.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_DIR="$ROOT/third_party/wheels/k3-cp314"
MANIFEST="$WHEEL_DIR/MANIFEST.json"

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "Run with sudo: sudo bash scripts/bootstrap_k3.sh" >&2
  exit 2
fi

/usr/bin/python3 - "$MANIFEST" "$WHEEL_DIR" <<'PY'
import hashlib
import json
import platform
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
wheel_dir = Path(sys.argv[2])
data = json.loads(manifest_path.read_text())
if platform.machine() != data['architecture']:
    raise SystemExit(f"Unsupported architecture: {platform.machine()} != {data['architecture']}")
if sys.implementation.cache_tag != 'cpython-' + data['python'][2:]:
    raise SystemExit(f"Unsupported Python ABI: {sys.implementation.cache_tag} != {data['python']}")
wheel = wheel_dir / data['wheel']
if not wheel.is_file():
    raise SystemExit(f"Missing bundled wheel: {wheel}")
payload = wheel.read_bytes()
if len(payload) != data['size_bytes']:
    raise SystemExit('Bundled wheel size does not match MANIFEST.json')
actual = hashlib.sha256(payload).hexdigest()
if actual != data['sha256']:
    raise SystemExit(f"Bundled wheel checksum mismatch: {actual}")
print(f"Verified {wheel.name}: {actual}")
PY

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  python3-numpy python3-scipy python3-opencv python3-pip \
  python3-can python3-wrapt python3-packaging python3-typing-extensions \
  spacemit-onnxruntime python3-spacemit-ort

WHEEL="$WHEEL_DIR/$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["wheel"])' "$MANIFEST")"
SYSTEM_SITE="$(/usr/bin/python3 - <<'PY'
import sys

paths = [
    path for path in sys.path
    if path.startswith('/usr/local/lib/python')
    and path.endswith(('dist-packages', 'site-packages'))
]
if not paths:
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    paths = [f'/usr/local/lib/python{version}/dist-packages']
print(paths[0])
PY
)"
install -d "$SYSTEM_SITE"
# Remove files created by the former prefix-based pip command.  On
# Bianbu that command duplicated the `local` component and produced an
# unimportable /usr/local/local/... installation.
PYTHON_VERSION="$(/usr/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
LEGACY_SITE="/usr/local/local/lib/python${PYTHON_VERSION}/dist-packages"
rm -rf \
  "$LEGACY_SITE/pyrealsense2" \
  "$LEGACY_SITE/pyrealsense2.libs" \
  "$LEGACY_SITE/pyrealsense2-"*.dist-info
/usr/bin/python3 -m pip install --no-index --no-deps --target "$SYSTEM_SITE" \
  --break-system-packages --root-user-action=ignore --upgrade "$WHEEL"

env -i PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 DICE_ROOT="$ROOT" \
  /usr/bin/python3 - <<'PY'
import importlib
from pathlib import Path
for name in ('numpy', 'scipy', 'cv2', 'pyrealsense2', 'onnxruntime',
             'can', 'wrapt', 'packaging', 'typing_extensions'):
    module = importlib.import_module(name)
    print(name, getattr(module, '__version__', 'installed'), Path(module.__file__).resolve())
import cv2
if not hasattr(cv2, 'aruco'):
    raise SystemExit('OpenCV ArUco module is unavailable')
PY

PYTHONNOUSERSITE=1 DICE_PYTHON=/usr/bin/python3 \
  bash "$ROOT/scripts/check_environment.sh"
echo "K3 system Python installation completed."
echo "Activate it in this shell with: source '$ROOT/scripts/env.sh' --system"
