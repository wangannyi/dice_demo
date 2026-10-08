"""Install and select a checksummed, application-local K3 inference runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def read_lock(root=ROOT):
    return json.loads((root / 'configs/k3_runtime.lock.json').read_text())


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify(prefix, lock):
    if json.loads((prefix / 'installed-lock.json').read_text()) != lock:
        raise ValueError('Installed K3 runtime does not match the dependency lock')
    for relative, expected in lock['libraries'].items():
        if digest(prefix / relative) != expected:
            raise ValueError(f'K3 runtime checksum mismatch: {relative}')
    for name in ('onnxruntime', 'spacemit_ort'):
        if not (prefix / lock['python_path'] / name / '__init__.py').is_file():
            raise ValueError(f'Missing locked Python package: {name}')
    return prefix


def runtime_root(root=ROOT, native=False):
    lock = read_lock(root)
    if platform.machine() != lock['architecture']:
        return None
    if not native and sys.implementation.cache_tag != lock['python_abi']:
        raise ValueError(f"Locked runtime requires {lock['python_abi']}")
    prefix = root / 'runtime' / lock['id']
    try:
        return verify(prefix, lock)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f'Locked K3 runtime unavailable: {exc}. '
            f'Run: python3 {root / "scripts/k3_runtime.py"} install'
        ) from exc


def environment(root=ROOT, native=False, current=None):
    prefix = runtime_root(root, native=native)
    if prefix is None:
        return {}
    current = os.environ if current is None else current
    lock = read_lock(root)
    paths = [prefix / lock['native_path']]
    values = {}
    if not native:
        # Python and native ORT packages ship distinct cores. Never mix them.
        try:
            from scripts.runtime_library_path import prepare_library_paths
        except ModuleNotFoundError:
            from runtime_library_path import prepare_library_paths
        packages = prefix / lock['python_path']
        values['DICE_LOCKED_RUNTIME'] = str(prefix)
        values['DICE_LOCKED_PYTHON'] = str(packages)
        cache = Path(tempfile.gettempdir()) / f'dice-ort-libraries-{os.getuid()}'
        paths = prepare_library_paths(packages, cache, prefix / lock['native_path'])
        values['PYTHONPATH'] = str(packages) + (':' + current['PYTHONPATH'] if current.get('PYTHONPATH') else '')
    previous = current.get('LD_LIBRARY_PATH', '')
    values['LD_LIBRARY_PATH'] = ':'.join(map(str, paths)) + (':' + previous if previous else '')
    # This survives a parent application's loader-path overrides. Replace only
    # TCM preloads, retaining unrelated deployment instrumentation.
    preloads = current.get('LD_PRELOAD', '').replace(':', ' ').split()
    preloads = [p for p in preloads if not Path(p).name.startswith('libspine_tcm.so')]
    values['LD_PRELOAD'] = ' '.join([str(prefix / lock['tcm']), *preloads])
    return values


def install(root=ROOT, package_cache=Path('/var/cache/apt/archives')):
    lock = read_lock(root)
    if platform.machine() != lock['architecture'] or sys.implementation.cache_tag != lock['python_abi']:
        raise ValueError(f"Requires {lock['architecture']} / {lock['python_abi']}")
    parent = root / 'runtime'
    parent.mkdir(exist_ok=True)
    target = parent / lock['id']
    if target.exists():
        return verify(target, lock)
    with tempfile.TemporaryDirectory(prefix='.install-', dir=parent) as temporary:
        work = Path(temporary)
        staging = work / 'payload'
        staging.mkdir()
        for package in lock['packages']:
            archive = package_cache / package['file']
            if not archive.is_file():
                download = work / package['name']
                download.mkdir()
                subprocess.run(['apt-get', 'download', f"{package['name']}={package['version']}"], cwd=download, check=True)
                candidates = list(download.glob('*.deb'))
                if len(candidates) != 1:
                    raise ValueError(f"Expected exactly one archive for {package['name']}")
                archive = candidates[0]
            if digest(archive) != package['sha256']:
                raise ValueError(f'Package checksum mismatch: {archive}')
            subprocess.run(['dpkg-deb', '-x', str(archive), str(staging)], check=True)
        (staging / 'installed-lock.json').write_text(json.dumps(lock, indent=2) + '\n')
        verify(staging, lock)
        # mkdtemp is 0700; the installed payload must remain readable when
        # bootstrap was invoked with sudo and gameplay uses a normal user.
        staging.chmod(0o755)
        staging.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('install', 'path', 'environment', 'shell'))
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'install':
            print(install())
        elif args.command == 'path':
            print(runtime_root(native=args.native) or '')
        elif args.command == 'environment':
            print(json.dumps(environment(native=args.native)))
        else:
            values = environment()
            # env.sh selects and deduplicates loader paths after choosing its interpreter.
            values.pop('LD_LIBRARY_PATH', None)
            print('unset DICE_LOCKED_RUNTIME DICE_LOCKED_PYTHON')
            for key, value in values.items():
                print(f'export {key}={shlex.quote(value)}')
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'K3 dependency lock: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
