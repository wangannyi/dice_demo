"""Dependency locking checks without devices or package installation."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import k3_runtime


class RuntimeLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'configs').mkdir()
        self.lock = dict(id='test-runtime', architecture='riscv64',
                         python_abi=sys.implementation.cache_tag,
                         python_path='packages', native_path='native',
                         tcm='native/libspine_tcm.so.3.0.0', packages=[],
                         libraries={'native/libspine_tcm.so.3.0.0': hashlib.sha256(b'old-tcm').hexdigest()})
        (self.root / 'configs/k3_runtime.lock.json').write_text(json.dumps(self.lock))
        self.prefix = self.root / 'runtime/test-runtime'
        self.prefix.mkdir(parents=True)
        for name in ('onnxruntime', 'spacemit_ort'):
            directory = self.prefix / 'packages' / name
            directory.mkdir(parents=True)
            (directory / '__init__.py').touch()
        (self.prefix / 'native').mkdir()
        (self.prefix / self.lock['tcm']).write_bytes(b'old-tcm')
        (self.prefix / 'installed-lock.json').write_text(json.dumps(self.lock))

    def test_native_scope_preserves_parent_environment_and_unrelated_preloads(self):
        parent = {'LD_LIBRARY_PATH': '/opt/system-new',
                  'LD_PRELOAD': '/opt/new/libspine_tcm.so.3:/opt/instrument.so',
                  'PYTHONPATH': '/opt/tts'}
        original = dict(parent)
        with patch.object(k3_runtime.platform, 'machine', return_value='riscv64'):
            result = k3_runtime.environment(self.root, native=True, current=parent)
        self.assertEqual(result['LD_LIBRARY_PATH'], str(self.prefix/'native')+':/opt/system-new')
        self.assertEqual(result['LD_PRELOAD'], str(self.prefix/self.lock['tcm'])+' /opt/instrument.so')
        self.assertNotIn('PYTHONPATH', result)
        self.assertEqual(parent, original)

    def test_missing_or_changed_runtime_never_falls_back_to_system(self):
        with patch.object(k3_runtime.platform, 'machine', return_value='riscv64'):
            (self.prefix / self.lock['tcm']).write_bytes(b'new-tcm')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                k3_runtime.environment(self.root, native=True)
            (self.prefix / self.lock['tcm']).unlink()
            with self.assertRaisesRegex(ValueError, 'Locked K3 runtime unavailable'):
                k3_runtime.environment(self.root, native=True)

    def test_changed_lock_requires_a_matching_install(self):
        changed = dict(self.lock, packages=[{'version': 'new'}])
        (self.root/'configs/k3_runtime.lock.json').write_text(json.dumps(changed))
        with patch.object(k3_runtime.platform, 'machine', return_value='riscv64'):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                k3_runtime.runtime_root(self.root)

    def test_non_k3_keeps_existing_environment(self):
        with patch.object(k3_runtime.platform, 'machine', return_value='x86_64'):
            self.assertEqual(k3_runtime.environment(self.root), {})

    def test_corrupted_archive_is_rejected_before_extraction(self):
        lock = dict(self.lock, id='not-installed', packages=[dict(
            name='spacemit-onnxruntime', version='2.0.6', file='runtime.deb',
            sha256=hashlib.sha256(b'correct').hexdigest())])
        (self.root/'configs/k3_runtime.lock.json').write_text(json.dumps(lock))
        cache = self.root/'cache'
        cache.mkdir()
        (cache/'runtime.deb').write_bytes(b'incorrect')
        with patch.object(k3_runtime.platform, 'machine', return_value='riscv64'), \
                patch.object(k3_runtime.subprocess, 'run') as execute:
            with self.assertRaisesRegex(ValueError, 'Package checksum mismatch'):
                k3_runtime.install(self.root, cache)
            execute.assert_not_called()
        self.assertFalse((self.root/'runtime/not-installed').exists())


if __name__ == '__main__':
    unittest.main()
