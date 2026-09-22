"""Bounded-cost feedback counters with complete CAN audit retained at the end."""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cup_grasp_demo.calibration_debug.joint_execution import JointGuard
from cup_grasp_demo.calibration_debug import shake_execution as shared


class CountingHistory(list):
    reads = 0

    def __getitem__(self, index):
        self.reads += 1
        return super().__getitem__(index)


class JointAuditTest(unittest.TestCase):
    def row(self, **fields):
        return dict(allowed=True, send_returned_successfully=True, **fields)

    def test_cost_depends_on_new_rows_not_total_history(self):
        guard = JointGuard(object)
        guard.history = CountingHistory([self.row() for _ in range(16000)])
        with patch.object(
            shared.core, "deepcopy", side_effect=AssertionError("hot-path deepcopy")
        ):
            self.assertEqual(guard.report()["actual_tx_count"], 16000)
            guard.history.reads = 0
            for _ in range(100):
                self.assertEqual(guard.report()["tx_attempts"], 16000)
                self.assertEqual(guard.report()["actual_tx_count"], 16000)
            self.assertEqual(guard.history.reads, 0)
            guard.history.append(self.row())
            self.assertEqual(guard.report()["actual_tx_count"], 16001)
            self.assertEqual(guard.history.reads, 1)

    def test_pending_and_failed_sends_count_once_even_out_of_order(self):
        guard = JointGuard(object)
        pending = dict(allowed=True, send_returned_successfully=False)
        guard.history = [
            self.row(),
            pending,
            self.row(),
            dict(allowed=False, send_returned_successfully=False),
        ]
        guard.denied = 1
        first = guard.report()
        self.assertEqual(first["actual_tx_count"], 2)
        self.assertEqual(first["denied_tx_attempts"], 1)
        self.assertEqual(guard.report()["actual_tx_count"], 2)
        pending["send_returned_successfully"] = True
        self.assertEqual(guard.report()["actual_tx_count"], 3)
        guard.history.append(
            dict(
                allowed=True,
                send_returned_successfully=False,
                error="native failure",
                transmission_outcome_uncertain=True,
            )
        )
        current = guard.report()
        self.assertEqual(current["tx_attempts"], 5)
        self.assertEqual(current["actual_tx_count"], 3)
        self.assertTrue(current["transmission_outcome_uncertain"])
        full = guard.report(include_history=True)
        self.assertEqual(full["actual_tx_count"], current["actual_tx_count"])
        self.assertEqual(full["history"], guard.history)
        self.assertIsNot(full["history"], guard.history)
        self.assertTrue(full["transmission_outcome_uncertain"])

    def test_real_guard_hooks_keep_allowlist_and_full_audit(self):
        class Bus:
            def send(self, message):
                if message.fail:
                    raise OSError("simulated native transmit failure")
                return True

        original = Bus.send
        guard = JointGuard(Bus)
        guard.install()
        guard.permit()
        guard.motion_allowed = True
        msg = SimpleNamespace(
            arbitration_id=0x155, data=bytes(8), is_extended_id=False, fail=False
        )
        try:
            Bus().send(msg)
            self.assertEqual(guard.report()["actual_tx_count"], 1)
            bad = deepcopy(msg)
            bad.arbitration_id = 0x471
            with self.assertRaisesRegex(RuntimeError, "guard rejected"):
                Bus().send(bad)
            msg.fail = True
            with self.assertRaises(OSError):
                Bus().send(msg)
            compact = guard.report()
            full = guard.report(include_history=True)
            self.assertEqual(compact["actual_tx_count"], 1)
            self.assertEqual(compact["tx_attempts"], 2)
            self.assertTrue(compact["transmission_outcome_uncertain"])
            for key in compact:
                if key != "history_omitted":
                    self.assertEqual(compact[key], full[key])
            self.assertEqual(len(full["history"]), 2)
        finally:
            guard.restore()
        self.assertIs(Bus.send, original)

    def test_not_permitted_still_denies_transmit(self):
        class Bus:
            def send(self, message):
                raise AssertionError("must not reach hardware")

        guard = JointGuard(Bus)
        guard.install()
        guard.motion_allowed = True
        msg = SimpleNamespace(arbitration_id=0x155, data=bytes(8), is_extended_id=False)
        try:
            with self.assertRaises(shared.core.PassiveTransmitForbidden):
                Bus().send(msg)
            self.assertEqual(guard.report()["actual_tx_count"], 0)
            self.assertEqual(guard.report()["denied_tx_attempts"], 1)
        finally:
            guard.restore()


if __name__ == "__main__":
    unittest.main()
