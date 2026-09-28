"""Firmware validation is cached only within a connected driver session."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from nero_revo2_control import nero_revo2_demo as demo


def robot(replies):
    return demo.bind_firmware_session(SimpleNamespace(
        connect=Mock(), disconnect=Mock(),
        get_firmware=Mock(side_effect=replies)))


def test_cache_and_reconnect():
    arm = robot([{'software_version': '1.20'}] * 3)
    arm.connect()
    demo.require_firmware(arm)
    demo.require_firmware(arm)
    assert arm.get_firmware.call_count == 1
    arm.disconnect()
    assert arm._demo_firmware_validated is False
    arm.connect()
    demo.require_firmware(arm)
    assert arm.get_firmware.call_count == 2
    arm.connect()  # A connect call alone also invalidates the old session.
    demo.require_firmware(arm)
    assert arm.get_firmware.call_count == 3


def test_missing_reply_retries_then_caches():
    arm = robot([None, None, {'software_version': '1.20'}])
    demo.require_firmware(arm)
    demo.require_firmware(arm)
    assert arm.get_firmware.call_count == 3


def test_timeout_is_not_version_mismatch_or_cached():
    arm = robot([None] * 3 + [{'software_version': '1.20'}])
    with pytest.raises(TimeoutError, match='3 attempts'):
        demo.require_firmware(arm)
    assert arm._demo_firmware_validated is False
    demo.require_firmware(arm)
    assert arm.get_firmware.call_count == 4


def test_wrong_version_is_not_retried_or_cached():
    arm = robot([{'software_version': '1.19'}])
    with pytest.raises(RuntimeError, match='expects Nero firmware'):
        demo.require_firmware(arm)
    assert arm.get_firmware.call_count == 1
    assert arm._demo_firmware_validated is False


def test_cached_version_does_not_skip_live_state_checks():
    arm = robot([{'software_version': '1.20'}])
    demo.require_firmware(arm)
    with pytest.raises(RuntimeError, match='not NORMAL'):
        demo.require_arm_ready(arm, [0] * 7, SimpleNamespace(arm_status=2))
    assert arm.get_firmware.call_count == 1


def test_failed_reconnect_invalidates_cache():
    arm = robot([{'software_version': '1.20'}])
    demo.require_firmware(arm)
    original = arm.connect.__wrapped__
    original.side_effect = RuntimeError('connection failed')
    with pytest.raises(RuntimeError, match='connection failed'):
        arm.connect()
    assert arm._demo_firmware_validated is False
