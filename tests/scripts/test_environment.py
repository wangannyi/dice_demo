"""Relocation and interpreter-selection checks for scripts/env.sh."""
import os
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class EnvironmentScriptTests(unittest.TestCase):
    def source(self, extra=None, arguments=''):
        env = {'PATH': os.environ['PATH'], 'HOME': '/home/should-not-be-used'}
        env.update(extra or {})
        command = (
            f'source "{ROOT}/scripts/env.sh" {arguments}; '
            'printf "%s\\n" "$DICE_ROOT" "$DICE_PYTHON" "$DICE_VISION_PYTHON" '
            '"$DICE_SDK_PYTHON" "$NERO_SDK_DIR" "$PYTHONPATH"'
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
        self.assertTrue(values[5].startswith(f'{ROOT}:{ROOT}/third_party/pyAgxArm'))

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


if __name__ == '__main__':
    unittest.main()
