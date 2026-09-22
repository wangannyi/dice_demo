"""FAST preloading never moves hardware and preserves other entry lifecycles."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from cup_grasp_demo.calibration_debug import green_startup as s
from cup_grasp_demo.calibration_debug import green_runtime as r


class StartupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root/'cfg.json'
        self.raw = dict(pipeline_strategy='green_open_cup', green_cup=dict(fast_parallel_startup=True))
        self.config.write_text(json.dumps(self.raw))
        self.argv = ['pipeline', '--config', str(self.config), '--session', str(self.root/'run'),
                     '--mode', 'fast', '--execute']
        s._pending = None

    def tearDown(self):
        if s._pending is not None:
            s._pending.close()
        s._pending = None
        self.temp.cleanup()

    def test_preview_step_auto_and_other_commands_do_not_start(self):
        cases = [self.argv[:-1], ['green-detect', *self.argv[1:]],
                 self.argv+['--status'], self.argv+['--resume'], self.argv+['--help']]
        cases += [[('step' if x == 'fast' else x) for x in self.argv],
                  [('auto' if x == 'fast' else x) for x in self.argv]]
        with patch.object(s, 'Startup') as factory:
            for args in cases:
                s.launch(args)
            factory.assert_not_called()

    def test_disabled_or_uncalibrated_does_not_start(self):
        with patch.object(s, 'Startup') as factory:
            for values in ({'fast_parallel_startup':False},
                           {'fast_parallel_startup':True,'installation_requires_calibration':True},
                           {'fast_parallel_startup':True,'persistent_runtime':False}):
                self.raw['green_cup'] = values
                self.config.write_text(json.dumps(self.raw))
                s.launch(self.argv)
            factory.assert_not_called()

    def test_claim_transfers_ownership_and_closes_unused_resources(self):
        sdk, vision = Mock(), Mock()
        with patch.object(r,'SDKClient',return_value=sdk), patch.object(r,'VisionResources',return_value=vision), \
             patch.object(s.atexit, 'register'):
            s.launch(self.argv)
            warm = s.claim(self.config, self.root/'run')
            self.assertIsNone(s._pending)
            self.assertIs(warm.acquire('sdk'), sdk)
            warm.futures['vision'].result(timeout=1)
            warm.close(); warm.close()
            sdk.close.assert_not_called()
            vision.close.assert_called_once()

    def test_config_change_rejects_preloaded_resources(self):
        sdk, vision = Mock(), Mock()
        with patch.object(r,'SDKClient',return_value=sdk), patch.object(r,'VisionResources',return_value=vision), \
             patch.object(s.atexit,'register'):
            s.launch(self.argv)
            s._pending.futures['sdk'].result(timeout=1)
            s._pending.futures['vision'].result(timeout=1)
            self.config.write_text('{}')
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                s.claim(self.config,self.root/'run')
            sdk.close.assert_called_once(); vision.close.assert_called_once()

    def test_failed_startup_closes_other_resource(self):
        vision = Mock()
        with patch.object(r,'SDKClient',side_effect=RuntimeError('CAN failure')), \
             patch.object(r,'VisionResources',return_value=vision), patch.object(s.atexit,'register'):
            s.launch(self.argv)
            warm=s.claim(self.config,self.root/'run')
            with self.assertRaisesRegex(RuntimeError,'CAN failure'):
                warm.acquire('sdk')
            # Ensure the successful peer has finished before cleanup assertion.
            warm.futures['vision'].result(timeout=1)
            warm.close(); vision.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
