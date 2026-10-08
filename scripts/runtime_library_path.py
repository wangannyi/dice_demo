"""Select matching K3 native libraries for the configured vision Python."""
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import tempfile


def prepare_library_paths(package_dir, cache_dir, system_lib=Path('/usr/lib')):
    """Alias ORT's generic sonames to the core shipped with its Python binding."""
    capi = package_dir / 'onnxruntime/capi'
    provider = package_dir / 'spacemit_ort'
    cores = sorted({p.resolve() for p in capi.glob('libonnxruntime.so*') if p.is_file()})
    if len(cores) != 1:
        raise ValueError(f'Expected one ONNX Runtime core in {capi}, found {len(cores)}')
    core = cores[0]
    stat = core.stat()
    identity = f'{core}:{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}'
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    aliases = cache_dir / key
    aliases.mkdir(parents=True, exist_ok=True, mode=0o700)
    targets = {'libonnxruntime.so': core, 'libonnxruntime.so.1': core}
    shared = capi / 'libonnxruntime_providers_shared.so'
    if shared.is_file():
        targets[shared.name] = shared.resolve()
    for name, target in targets.items():
        link = aliases / name
        if link.is_symlink() and link.resolve() == target:
            continue
        temporary = aliases / f'.{name}.{os.getpid()}'
        temporary.symlink_to(target)
        temporary.replace(link)
    return [aliases, capi.resolve(), provider.resolve(), system_lib]


def configured_library_paths():
    # Other hosts retain their existing loader search path.
    if platform.machine() != 'riscv64':
        return []
    ort = importlib.util.find_spec('onnxruntime')
    ep = importlib.util.find_spec('spacemit_ort')
    if ort is None or ep is None:
        return []
    ort_packages = Path(ort.origin).resolve().parent.parent
    ep_packages = Path(ep.origin).resolve().parent.parent
    if ort_packages != ep_packages:
        raise ValueError('onnxruntime and spacemit_ort must use the same Python package directory')
    cache = Path(tempfile.gettempdir()) / f'dice-ort-libraries-{os.getuid()}'
    return prepare_library_paths(ort_packages, cache)


if __name__ == '__main__':
    print(':'.join(map(str, configured_library_paths())))
