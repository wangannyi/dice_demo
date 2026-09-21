import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

from camera_profile import CameraBrightnessProfile


ROOT = Path(__file__).resolve().parent
PROFILE = json.loads((ROOT/'config/usb7_marker_rgb_profile.json').read_text())
CONFIG = {'device': '/dev/video7', 'width': 1280, 'height': 720, 'format': 'MJPG',
          'backend': 'v4l2_rgb', 'intrinsics_file': 'usb7_intrinsics_20260916.json'}


class HardwareMock:
    def __init__(self):
        self.brightness = 17
        self.calls = []
        self.get_count = 0
        self.fail_get = None
        self.fail_set = set()
        self.clamp_requested = None
        self.fail_audit = False
        self.bad_get_output = None

    def __call__(self, argv, **options):
        self.calls.append((argv.copy(), options.copy()))
        if options != {'capture_output': True, 'text': True, 'check': False, 'timeout': 5}:
            raise AssertionError('Unexpected runner options')
        if argv[:3] != ['v4l2-ctl', '--device', '/dev/video7']:
            raise AssertionError('Unexpected camera command')
        flag = argv[3]
        if flag == '--get-ctrl':
            self.get_count += 1
            if argv[4] != 'brightness':
                raise AssertionError('Unexpected control read')
            if self.get_count == self.fail_get:
                raise subprocess.TimeoutExpired(argv, 5)
            stdout = (self.bad_get_output if self.bad_get_output is not None
                      else 'brightness: '+str(self.brightness)+'\n')
            return SimpleNamespace(returncode=0, stdout=stdout, stderr='')
        if flag == '--set-ctrl':
            if not argv[4].startswith('brightness='):
                raise AssertionError('Only brightness may be written')
            target = int(argv[4].split('=')[1])
            # Even a failed write changes the mock hardware, reproducing the
            # partial-success case that requires a finally restoration.
            self.brightness = target
            if target == 128 and self.clamp_requested is not None:
                self.brightness = self.clamp_requested
            if target in self.fail_set:
                return SimpleNamespace(returncode=1, stdout='', stderr='setter failed')
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        if flag == '--all':
            return SimpleNamespace(returncode=int(self.fail_audit),
                                   stdout='brightness: '+str(self.brightness)
                                   +'\nzoom_absolute: 100\nexposure_auto: 3\n',
                                   stderr='audit failed' if self.fail_audit else '')
        raise AssertionError('Unexpected v4l2 control command')


class CameraProfileTests(unittest.TestCase):
    def setUp(self):
        self.hardware = HardwareMock()

    def profile(self, profile=None, config=None):
        return CameraBrightnessProfile('/dev/video7', PROFILE if profile is None else profile,
                                       CONFIG if config is None else config, runner=self.hardware)

    def test_constructing_valid_profile_reads_or_writes_no_hardware(self):
        profile = self.profile()
        self.assertEqual(self.hardware.calls, [])
        self.assertFalse(profile.record['active'])
        self.assertEqual(profile.record['apply_status'], 'not_started')
        json.dumps(profile.record, allow_nan=False)

    def test_explicit_apply_warmup_end_checks_and_exact_restoration(self):
        profile = self.profile()
        record = profile.apply()
        self.assertIs(record, profile.record)
        self.assertEqual(record['before_brightness'], 17)
        self.assertEqual(record['requested_controls'], {'brightness': 128})
        self.assertEqual(record['apply_readback_brightness'], 128)
        self.assertTrue(record['active'])
        self.assertIn('brightness: 17', record['v4l2_before']['stdout'])
        self.assertIn('brightness: 128', record['v4l2_active']['stdout'])
        profile.check_active()  # Camera warmup completed.
        profile.check_active()  # Capture completed.
        self.assertEqual(len(record['checks']), 2)
        self.assertTrue(all(check['valid'] and check['readback_brightness'] == 128
                            for check in record['checks']))
        profile.restore()
        self.assertEqual(self.hardware.brightness, 17)
        self.assertEqual(record['restore_readback_brightness'], 17)
        self.assertEqual(record['restore_status'], 'restored')
        self.assertIn('brightness: 17', record['v4l2_after']['stdout'])
        self.assertFalse(record['active'])
        self.assertFalse(record['restore_required'])
        self.assertLessEqual(record['apply_started_monotonic_s'], record['apply_finished_monotonic_s'])
        self.assertLessEqual(record['restore_started_monotonic_s'],
                             record['restore_finished_monotonic_s'])
        writes = [argv[4] for argv, _ in self.hardware.calls if argv[3] == '--set-ctrl']
        self.assertEqual(writes, ['brightness=128', 'brightness=17'])
        json.dumps(record, allow_nan=False)

    def test_clamped_apply_readback_raises_but_finally_can_restore(self):
        profile = self.profile()
        self.hardware.clamp_requested = 127
        try:
            with self.assertRaisesRegex(RuntimeError, 'does not match requested'):
                profile.apply()
            self.assertTrue(profile.record['restore_available'])
            self.assertTrue(profile.record['restore_required'])
            self.assertEqual(profile.record['apply_readback_brightness'], 127)
            self.assertEqual(profile.record['apply_status'], 'failed')
        finally:
            profile.restore()
        self.assertEqual(self.hardware.brightness, 17)

    def test_failed_setter_that_changed_hardware_is_still_restored(self):
        profile = self.profile()
        self.hardware.fail_set.add(128)
        try:
            with self.assertRaisesRegex(RuntimeError, 'setter failed'):
                profile.apply()
            self.assertEqual(self.hardware.brightness, 128)
            self.assertTrue(profile.restore_required)
        finally:
            profile.restore()
        self.assertEqual(self.hardware.brightness, 17)
        self.assertEqual(profile.record['restore_status'], 'restored')

    def test_readback_timeout_after_successful_setter_can_restore(self):
        profile = self.profile()
        self.hardware.fail_get = 2
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                profile.apply()
            self.assertEqual(self.hardware.brightness, 128)
        finally:
            profile.restore()
        self.assertEqual(self.hardware.brightness, 17)
        self.assertEqual(profile.record['restore_attempts'], 1)
        self.assertTrue(any('TimeoutExpired' in command['error']
                            for command in profile.record['commands'] if command['error']))

    def test_active_control_drift_fails_end_check_and_restores_original(self):
        profile = self.profile()
        profile.apply()
        profile.check_active()
        self.hardware.brightness = 90
        with self.assertRaisesRegex(RuntimeError, 'Active brightness changed'):
            profile.check_active()
        self.assertFalse(profile.record['active'])
        self.assertEqual(profile.record['checks'][-1]['readback_brightness'], 90)
        self.assertFalse(profile.record['checks'][-1]['valid'])
        profile.restore()
        self.assertEqual(self.hardware.brightness, 17)

    def test_restore_failure_is_reported_and_retry_can_recover(self):
        profile = self.profile()
        profile.apply()
        self.hardware.fail_set.add(17)
        with self.assertRaisesRegex(RuntimeError, 'setter failed'):
            profile.restore()
        self.assertEqual(profile.record['restore_status'], 'failed')
        self.assertTrue(profile.record['restore_required'])
        self.assertEqual(profile.record['errors'][-1]['phase'], 'restore')
        self.hardware.fail_set.clear()
        profile.restore()
        self.assertEqual(profile.record['restore_attempts'], 2)
        self.assertEqual(profile.record['restore_status'], 'restored')

    def test_wrong_restore_readback_is_not_silently_accepted(self):
        profile = self.profile()
        profile.apply()
        self.hardware.bad_get_output = 'brightness: 16\n'
        with self.assertRaisesRegex(RuntimeError, 'does not match original'):
            profile.restore()
        self.assertEqual(profile.record['restore_readback_brightness'], 16)
        self.assertEqual(profile.record['restore_status'], 'failed')
        self.assertTrue(profile.restore_required)
        self.hardware.bad_get_output = None
        profile.restore()

    def test_duplicate_restore_is_idempotent_without_additional_queries_or_writes(self):
        profile = self.profile()
        profile.apply()
        profile.restore()
        calls = len(self.hardware.calls)
        end = profile.record['restore_finished_monotonic_s']
        self.assertIs(profile.restore(), profile.record)
        self.assertEqual(len(self.hardware.calls), calls)
        self.assertEqual(profile.record['restore_attempts'], 1)
        self.assertEqual(profile.record['restore_finished_monotonic_s'], end)

    def test_original_read_failure_never_writes_unrestorable_controls(self):
        profile = self.profile()
        self.hardware.fail_get = 1
        with self.assertRaises(subprocess.TimeoutExpired):
            profile.apply()
        calls = len(self.hardware.calls)
        profile.restore()
        self.assertEqual(len(self.hardware.calls), calls)
        self.assertFalse(profile.record['restore_available'])
        self.assertFalse(profile.record['restore_required'])
        self.assertFalse(any(argv[3] == '--set-ctrl' for argv, _ in self.hardware.calls))

    def test_failed_before_audit_performs_no_write_and_retains_audit_failure(self):
        profile = self.profile()
        self.hardware.fail_audit = True
        with self.assertRaisesRegex(RuntimeError, 'audit failed'):
            profile.apply()
        self.assertEqual(profile.record['v4l2_before']['returncode'], 1)
        self.assertEqual(profile.record['v4l2_before']['stderr'], 'audit failed')
        self.assertFalse(profile.restore_required)
        profile.restore()
        self.assertFalse(any(argv[3] == '--set-ctrl' for argv, _ in self.hardware.calls))

    def test_illegal_profile_never_queries_or_writes_hardware(self):
        mutations = [
            lambda p: p.update(schema=True),
            lambda p: p.update(kind='unrelated'),
            lambda p: p.update(camera=[]),
            lambda p: p['camera'].update(device='/dev/video8'),
            lambda p: p['camera']['mode'].update(width=True),
            lambda p: p['camera']['mode'].update(width=1280.),
            lambda p: p['camera']['mode'].update(width=640),
            lambda p: p['camera']['mode'].update(height=0),
            lambda p: p['camera']['mode'].update(format=123),
            lambda p: p['camera']['mode'].update(format='YUYV'),
            lambda p: p['camera']['mode'].update(zoom=100),
            lambda p: p['controls'].update(exposure=10),
            lambda p: p['controls'].update(zoom_absolute=100),
            lambda p: p['controls'].update(lens=1),
            lambda p: p.update(controls={}),
        ]
        for mutate in mutations:
            profile = copy.deepcopy(PROFILE)
            mutate(profile)
            with self.assertRaises(ValueError):
                self.profile(profile)
        for invalid in (True, False, 0, 256, 128., '128', None):
            profile = copy.deepcopy(PROFILE)
            profile['controls']['brightness'] = invalid
            with self.assertRaises(ValueError):
                self.profile(profile)
        self.assertEqual(self.hardware.calls, [])

    def test_invalid_camera_configuration_is_rejected_before_control_reads(self):
        for change in ({'device': '/dev/video8'}, {'width': True}, {'width': 1280.},
                       {'height': 360}, {'format': 'YUYV'}):
            with self.assertRaises(ValueError):
                self.profile(config={**CONFIG, **change})
        self.assertEqual(self.hardware.calls, [])

    def test_active_check_and_reapply_cannot_bypass_profile_lifecycle(self):
        profile = self.profile()
        with self.assertRaises(RuntimeError):
            profile.check_active()
        profile.apply()
        calls = len(self.hardware.calls)
        with self.assertRaises(RuntimeError):
            profile.apply()
        self.assertEqual(len(self.hardware.calls), calls)
        profile.restore()
        with self.assertRaises(RuntimeError):
            profile.check_active()


if __name__ == '__main__':
    unittest.main()
