"""Relocation and interpreter-selection checks for scripts/env.sh."""
import os
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import k3_runtime, runtime_library_path

ROOT = Path(__file__).resolve().parents[2]


class EnvironmentScriptTests(unittest.TestCase):
    def source(self, extra=None, arguments=''):
        env = {'PATH': os.environ['PATH'], 'HOME': '/home/should-not-be-used'}
        env.update(extra or {})
        command = (
            f'source "{ROOT}/scripts/env.sh" {arguments} && '
            'printf "%s\\n" "$DICE_ROOT" "$DICE_PYTHON" "$DICE_VISION_PYTHON" '
            '"$DICE_SDK_PYTHON" "$NERO_SDK_DIR" "$PYTHONPATH" "$LD_LIBRARY_PATH"'
        )
        result = subprocess.run(['bash', '-c', command], env=env, text=True,
                                capture_output=True, check=True)
        return result.stdout.splitlines()

    def test_system_python_and_repo_sdk_are_defaults(self):
        values = self.source()
        expected_python = subprocess.run(['bash', '-c', 'command -v python3'],
                                         env={'PATH': os.environ['PATH']}, text=True,
                                         capture_output=True, check=True).stdout.strip()
        self.assertEqual(values[0], str(ROOT))
        self.assertEqual(values[1:4], [expected_python] * 3)
        self.assertEqual(values[4], str(ROOT / 'third_party/pyAgxArm'))
        self.assertNotIn('/home/should-not-be-used', '\n'.join(values))
        expected_paths = [str(ROOT), str(ROOT / 'third_party/pyAgxArm')]
        locked = k3_runtime.runtime_root()
        if locked is not None:
            expected_paths.insert(0, str(locked / k3_runtime.read_lock()['python_path']))
        self.assertEqual(values[5].split(':')[:len(expected_paths)], expected_paths)

    def test_explicit_interpreter_and_extra_packages_are_honored(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / 'python3'
            python.write_text('#!/bin/sh\n')
            python.chmod(0o755)
            values = self.source({'DICE_PYTHON': str(python),
                                  'DICE_PYTHON_EXTRA': '/opt/deployment/python'})
            self.assertEqual(values[1:4], [str(python)] * 3)
            self.assertIn('/opt/deployment/python', values[5].split(':'))

    def test_system_mode_discards_stale_virtualenv_overrides(self):
        venv = '/home/old/.venv-' + 'grasp'
        sdk = '/home/old/agilex-' + 'api-test/pyAgxArm'
        stale = venv + '/bin/python'
        values = self.source({
            'DICE_PYTHON': stale,
            'DICE_VISION_PYTHON': stale,
            'DICE_SDK_PYTHON': stale,
            'CALIB_PYTHON': stale,
            'NERO_SDK_DIR': sdk,
            'DICE_PYTHON_EXTRA': venv + '/site-packages',
            'PYTHONPATH': venv + '/site-packages',
        }, '--system')
        self.assertEqual(values[1:4], ['/usr/bin/python3'] * 3)
        self.assertEqual(values[4], str(ROOT / 'third_party/pyAgxArm'))
        self.assertNotIn('/home/old', values[5])

    def test_native_libraries_precede_inherited_paths_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / 'python3'
            python.write_text('#!/bin/sh\nprintf "%s\\n" "/opt/dice-native:/usr/lib"\n')
            python.chmod(0o755)
            values = self.source({
                'DICE_VISION_PYTHON': str(python),
                'LD_LIBRARY_PATH': '/usr/local/lib:/usr/lib:/opt/custom:/usr/local/lib',
            })
            self.assertEqual(values[6], '/opt/dice-native:/usr/lib:/usr/local/lib:/opt/custom')

    def test_native_selection_failure_stops_sourcing(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / 'python3'
            python.write_text('#!/bin/sh\nexit 7\n')
            python.chmod(0o755)
            with self.assertRaises(subprocess.CalledProcessError) as context:
                self.source({'DICE_VISION_PYTHON': str(python)})
            self.assertIn('Unable to select matching K3 inference libraries', context.exception.stderr)

    def test_bundled_k3_realsense_wheel_matches_manifest(self):
        directory = ROOT / 'third_party/wheels/k3-cp314'
        manifest = json.loads((directory / 'MANIFEST.json').read_text())
        wheel = directory / manifest['wheel']
        self.assertEqual(manifest['architecture'], 'riscv64')
        self.assertEqual(manifest['python'], 'cp314')
        self.assertEqual(wheel.stat().st_size, manifest['size_bytes'])
        self.assertEqual(hashlib.sha256(wheel.read_bytes()).hexdigest(), manifest['sha256'])
        self.assertTrue((directory / manifest['license_file']).is_file())

    def test_bootstrap_installs_realsense_into_an_importable_system_path(self):
        script = (ROOT / 'scripts/bootstrap_k3.sh').read_text()
        self.assertIn("path for path in sys.path", script)
        self.assertIn('--target "$SYSTEM_SITE"', script)
        self.assertNotIn('--prefix /usr/local', script)
        self.assertIn('/usr/local/local/lib/python${PYTHON_VERSION}/dist-packages', script)

    @unittest.skipIf(os.geteuid() == 0, 'non-root behavior only')
    def test_bootstrap_requires_root_before_installing(self):
        result = subprocess.run(['bash', str(ROOT / 'scripts/bootstrap_k3.sh')],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Run with sudo', result.stderr)


class RuntimeLibraryPathTests(unittest.TestCase):
    def package(self, directory):
        packages = Path(directory) / 'python'
        capi = packages / 'onnxruntime/capi'
        capi.mkdir(parents=True)
        (packages / 'spacemit_ort').mkdir()
        core = capi / 'libonnxruntime.so.1.24.2+spacemit.a1'
        core.write_bytes(b'python-binding-runtime')
        (capi / 'libonnxruntime_providers_shared.so').write_bytes(b'matching-shared-provider')
        return packages, core

    def test_generic_sonames_resolve_to_selected_python_core(self):
        with tempfile.TemporaryDirectory() as directory:
            packages, core = self.package(directory)
            paths = runtime_library_path.prepare_library_paths(packages, Path(directory) / 'cache')
            self.assertEqual((paths[0] / 'libonnxruntime.so').resolve(), core)
            self.assertEqual((paths[0] / 'libonnxruntime.so.1').resolve(), core)
            shared = 'libonnxruntime_providers_shared.so'
            self.assertEqual((paths[0] / shared).resolve(), core.parent / shared)
            self.assertEqual(paths[1:], [core.parent, packages / 'spacemit_ort', Path('/usr/lib')])
            self.assertEqual(paths, runtime_library_path.prepare_library_paths(packages, Path(directory) / 'cache'))

    def test_upgrade_selects_a_new_alias_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            packages, core = self.package(directory)
            cache = Path(directory) / 'cache'
            first = runtime_library_path.prepare_library_paths(packages, cache)
            core.write_bytes(b'upgraded-python-binding-runtime')
            second = runtime_library_path.prepare_library_paths(packages, cache)
            self.assertNotEqual(first[0], second[0])
            self.assertEqual((second[0] / 'libonnxruntime.so').resolve(), core)

    def test_ambiguous_or_missing_core_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            packages, core = self.package(directory)
            other = core.parent / 'libonnxruntime.so.legacy'
            other.write_bytes(b'legacy')
            with self.assertRaisesRegex(ValueError, 'found 2'):
                runtime_library_path.prepare_library_paths(packages, Path(directory) / 'cache')
            other.unlink()
            core.unlink()
            with self.assertRaisesRegex(ValueError, 'found 0'):
                runtime_library_path.prepare_library_paths(packages, Path(directory) / 'cache')

    def test_other_hosts_do_not_change_loader_paths(self):
        with patch.object(runtime_library_path.platform, 'machine', return_value='x86_64'), \
                patch.object(runtime_library_path.importlib.util, 'find_spec') as lookup:
            self.assertEqual(runtime_library_path.configured_library_paths(), [])
            lookup.assert_not_called()

    def test_mixed_python_package_directories_are_rejected(self):
        class Spec:
            def __init__(self, origin):
                self.origin = origin

        with patch.object(runtime_library_path.platform, 'machine', return_value='riscv64'), \
                patch.object(runtime_library_path.importlib.util, 'find_spec', side_effect=[
                    Spec('/opt/new/onnxruntime/__init__.py'),
                    Spec('/opt/old/spacemit_ort/__init__.py'),
                ]):
            with self.assertRaisesRegex(ValueError, 'same Python package directory'):
                runtime_library_path.configured_library_paths()


if __name__ == '__main__':
    unittest.main()
