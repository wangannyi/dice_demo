"""Explicit, reversible brightness-only profile for live USB RGB capture.

Constructing a profile validates data and performs no hardware operation.
Callers explicitly apply it and always restore it in finally, including when
apply fails. No lens, zoom, exposure, camera stream, or SDK API is touched.
"""
import re
import subprocess
import time


class CameraBrightnessProfile:
    def __init__(self, device, profile, camera_config, runner=subprocess.run):
        if (not isinstance(profile, dict) or type(profile.get('schema')) is not int
                or profile['schema'] != 1
                or profile.get('kind') != 'usb_marker_rgb_acquisition_profile'):
            raise ValueError('Invalid USB RGB acquisition profile schema or kind')
        camera = profile.get('camera')
        if not isinstance(camera, dict) or not isinstance(camera_config, dict):
            raise ValueError('Profile and configuration must contain camera dictionaries')
        mode = camera.get('mode')
        if not isinstance(mode, dict) or set(mode) != {'width', 'height', 'format'}:
            raise ValueError('Profile camera.mode requires width, height and format')
        if (not isinstance(device, str) or not device.strip()
                or not isinstance(camera.get('device'), str)
                or not isinstance(camera_config.get('device'), str)
                or camera['device'] != device or camera_config['device'] != device):
            raise ValueError('Profile camera device must match the selected configuration')
        for key in ('width', 'height'):
            expected, actual = mode.get(key), camera_config.get(key)
            if (type(expected) is not int or expected <= 0
                    or type(actual) is not int or actual <= 0 or actual != expected):
                raise ValueError('Profile camera mode mismatch: '+key)
        if (not isinstance(mode.get('format'), str) or not mode['format'].strip()
                or not isinstance(camera_config.get('format'), str)
                or mode['format'] != camera_config['format']):
            raise ValueError('Profile camera mode mismatch: format')
        controls = profile.get('controls')
        if not isinstance(controls, dict) or set(controls) != {'brightness'}:
            raise ValueError('Only brightness may be controlled by this profile')
        target = controls['brightness']
        if type(target) is not int or not 1 <= target <= 255:
            raise ValueError('Profile brightness must be an integer from 1 through 255')
        self.device = device
        self.target = target
        self.runner = runner
        self.original = None
        self.restore_required = False
        self.record = {
            'schema': 1, 'kind': 'usb_marker_rgb_acquisition_profile_record',
            'device': device, 'camera_mode': dict(mode),
            'requested_controls': {'brightness': target},
            'before_brightness': None, 'apply_readback_brightness': None,
            'readback_brightness': None, 'restore_readback_brightness': None,
            'state': 'created', 'active': False, 'apply_status': 'not_started',
            'restore_status': 'not_needed', 'restore_required': False,
            'restore_available': False, 'restore_attempts': 0,
            'apply_started_monotonic_s': None, 'apply_finished_monotonic_s': None,
            'active_since_monotonic_s': None, 'active_verified_monotonic_s': None,
            'restore_started_monotonic_s': None, 'restore_finished_monotonic_s': None,
            'v4l2_before': None, 'v4l2_active': None, 'v4l2_after': None,
            'checks': [], 'commands': [], 'errors': []}

    def _run(self, flag, value=None):
        argv = ['v4l2-ctl', '--device', self.device, flag]
        if value is not None:
            argv.append(value)
        command = {'argv': argv, 'started_monotonic_s': time.monotonic(),
                   'finished_monotonic_s': None, 'returncode': None,
                   'stdout': None, 'stderr': None, 'error': None}
        self.record['commands'].append(command)
        try:
            result = self.runner(argv, capture_output=True, text=True, check=False, timeout=5)
            if (type(result.returncode) is not int
                    or not isinstance(result.stdout, str) or not isinstance(result.stderr, str)):
                raise RuntimeError('Invalid v4l2-ctl command result')
            command.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if result.returncode != 0:
                raise RuntimeError('v4l2-ctl failed: '+result.stderr.strip())
            return command
        except BaseException as exc:
            command['error'] = type(exc).__name__+': '+str(exc)
            raise
        finally:
            command['finished_monotonic_s'] = time.monotonic()

    def _brightness(self):
        command = self._run('--get-ctrl', 'brightness')
        match = re.fullmatch(r'\s*brightness:\s*(-?\d+)\s*', command['stdout'])
        if match is None:
            raise RuntimeError('Could not parse exact brightness readback')
        return int(match.group(1))

    def _audit(self, key):
        # Retain even a failed --all command, including its stderr and timing.
        try:
            command = self._run('--all')
        finally:
            self.record[key] = self.record['commands'][-1]
        return command

    def _error(self, phase, exc):
        self.record['errors'].append({'phase': phase, 'error': type(exc).__name__+': '+str(exc),
                                      'monotonic_s': time.monotonic()})

    def apply(self):
        if self.record['apply_status'] != 'not_started':
            raise RuntimeError('Brightness profile apply may only be attempted once')
        self.record['apply_started_monotonic_s'] = time.monotonic()
        self.record['apply_status'] = 'applying'
        try:
            self.original = self._brightness()
            self.record['before_brightness'] = self.original
            self.record['restore_available'] = True
            self._audit('v4l2_before')
            # Mark restoration necessary before the setter: a failed command
            # can still have changed hardware before reporting its failure.
            self.restore_required = True
            self.record.update(restore_required=True, restore_status='pending')
            self._run('--set-ctrl', 'brightness='+str(self.target))
            readback = self._brightness()
            self.record.update(apply_readback_brightness=readback, readback_brightness=readback)
            self._audit('v4l2_active')
            if readback != self.target:
                raise RuntimeError('Applied brightness readback does not match requested value')
            self.record.update(state='active', active=True, apply_status='applied',
                               active_since_monotonic_s=time.monotonic())
            return self.record
        except BaseException as exc:
            self.record.update(state='apply_failed', active=False, apply_status='failed')
            self._error('apply', exc)
            raise
        finally:
            self.record['apply_finished_monotonic_s'] = time.monotonic()

    def check_active(self):
        if self.record['apply_status'] != 'applied' or not self.restore_required:
            raise RuntimeError('Brightness profile is not applied')
        check = {'started_monotonic_s': time.monotonic(), 'finished_monotonic_s': None,
                 'readback_brightness': None, 'valid': False, 'v4l2_active': None, 'error': None}
        self.record['checks'].append(check)
        try:
            readback = self._brightness()
            self.record['readback_brightness'] = check['readback_brightness'] = readback
            check['v4l2_active'] = self._audit('v4l2_active')
            if readback != self.target:
                raise RuntimeError('Active brightness changed from requested value')
            self.record.update(active=True, state='active',
                               active_verified_monotonic_s=time.monotonic())
            check['valid'] = True
            return self.record
        except BaseException as exc:
            check['error'] = type(exc).__name__+': '+str(exc)
            self.record.update(active=False, state='active_check_failed')
            self._error('check_active', exc)
            raise
        finally:
            check['finished_monotonic_s'] = time.monotonic()

    def restore(self):
        if not self.restore_required:
            return self.record
        if self.original is None:
            raise RuntimeError('Original brightness is unknown; restoration unavailable')
        self.record.update(restore_status='restoring', active=False,
                           restore_started_monotonic_s=time.monotonic())
        self.record['restore_attempts'] += 1
        try:
            self._run('--set-ctrl', 'brightness='+str(self.original))
            readback = self._brightness()
            self.record['restore_readback_brightness'] = readback
            self._audit('v4l2_after')
            if readback != self.original:
                raise RuntimeError('Restored brightness readback does not match original value')
            self.restore_required = False
            self.record.update(state='restored', restore_status='restored', restore_required=False)
            return self.record
        except BaseException as exc:
            self.record.update(state='restore_failed', restore_status='failed')
            self._error('restore', exc)
            raise
        finally:
            self.record['restore_finished_monotonic_s'] = time.monotonic()
