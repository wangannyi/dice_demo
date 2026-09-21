"""Do not mistake a verified log reader for a controller; retain conflict checks."""

from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cup_grasp_demo.calibration_debug import shake_execution as sdk
from cup_grasp_demo.calibration_debug import test_joint_lab as fixture


class ControlProcessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proc = self.root / "proc"
        self.log = self.root / "cup_grasp_demo/datasets/run/pipeline.log"
        self.log.parent.mkdir(parents=True)
        self.log.touch()
        self.row = self.process(123, ["tail", "-f", str(self.log)], "/usr/bin/tail")

    def process(self, pid, argv, executable):
        p = self.proc / str(pid)
        p.mkdir(parents=True, exist_ok=True)
        (p / "cmdline").write_bytes(b"\0".join(v.encode() for v in argv) + b"\0")
        (p / "exe").unlink(missing_ok=True)
        (p / "exe").symlink_to(executable)
        return dict(pid=pid, ppid=1, args=" ".join(argv))

    def collect(self, rows):
        evidence = dict(
            candidate_control_processes=deepcopy(rows),
            receiver_rows=[dict(line="can0 000 00000000")],
            errors=[],
        )
        with (
            patch.object(sdk, "ROOT", self.root),
            patch.object(
                sdk.core, "host_control_evidence", return_value=deepcopy(evidence)
            ),
        ):
            return sdk.control_evidence("can0", {}, proc_root=self.proc), evidence

    def test_verified_system_tail_is_recorded_and_other_candidates_remain(self):
        controller = self.process(
            456, ["python3", "nero_control.py"], "/usr/bin/python3"
        )
        result, original = self.collect([self.row, controller])
        self.assertEqual(result["candidate_control_processes"], [controller])
        self.assertEqual(result["verified_log_readers"][0]["pid"], 123)
        self.assertEqual(result["receiver_rows"], original["receiver_rows"])
        self.assertEqual(result["errors"], original["errors"])

    def test_spoofed_name_shell_command_or_stale_pid_stays_blocked(self):
        for executable, argv in [
            ("/usr/bin/python3", ["tail", "-f", str(self.log)]),
            ("/usr/bin/sh", ["sh", "-c", "tail -f " + str(self.log)]),
            ("/usr/bin/tail", ["tail", "-f", str(self.log), "extra"]),
        ]:
            with self.subTest(argv=argv):
                row = self.process(123, argv, executable)
                result, _ = self.collect([row])
                self.assertEqual(result["candidate_control_processes"], [row])
                self.assertEqual(result["verified_log_readers"], [])
        self.process(123, ["tail", "-f", str(self.log)], "/usr/bin/tail")
        changed = dict(self.row, args="tail -f /different/cup_grasp.log")
        result, _ = self.collect([changed])
        self.assertEqual(result["candidate_control_processes"], [changed])
        (self.proc / "123/exe").unlink()
        result, _ = self.collect([self.row])
        self.assertEqual(result["candidate_control_processes"], [self.row])

    def test_only_regular_project_logs_are_excluded(self):
        outside = self.root / "outside.log"
        outside.touch()
        escaped = self.log.parent / "escaped.log"
        escaped.symlink_to(outside)
        for path in [outside, escaped, self.log.parent, Path("relative.log")]:
            with self.subTest(path=path):
                row = self.process(123, ["tail", "-f", str(path)], "/usr/bin/tail")
                result, _ = self.collect([row])
                self.assertEqual(result["candidate_control_processes"], [row])

    def test_no_candidate_keeps_existing_evidence(self):
        result, original = self.collect([])
        self.assertEqual(result, dict(original, verified_log_readers=[]))

    def test_real_conflict_reports_pid_and_can_receiver_checks_still_apply(self):
        baseline = dict(errors=[], receiver_rows=[], candidate_control_processes=[])
        isolated = dict(
            baseline,
            channel="can0",
            receiver_rows=[
                dict(list="all", line="can0 000 00000000"),
                dict(list="err", line="can0 000 1fffffff"),
            ],
        )
        self.assertEqual(sdk.control_conflicts(baseline, isolated), [])
        conflict = dict(isolated, candidate_control_processes=[self.row])
        errors = sdk.control_conflicts(baseline, conflict)
        self.assertIn("Other candidate controller processes are present", errors)
        self.assertIn("PID 123: tail -f ", errors[-1])
        occupied = dict(baseline, receiver_rows=[dict(line="other socket")])
        self.assertIn(
            "CAN receivers existed before this SDK connection",
            sdk.control_conflicts(occupied, isolated),
        )

    def test_executor_preserves_evidence_and_sends_nothing_on_conflict(self):
        evidence = dict(candidate_control_processes=[self.row])
        report, _, robot = fixture.ExecutionTest().simulate(
            host_evidence=evidence,
            host_conflicts=["Other candidate controller processes are present"],
        )
        self.assertFalse(report["success"])
        self.assertFalse(report["motion_attempted"])
        self.assertIn("PID 123", report["error"])
        self.assertEqual(report["host_control_before_connect"], evidence)
        self.assertEqual(report["host_control_before_motion"], evidence)
        robot.move_js.assert_not_called()
        robot.set_speed_percent.assert_not_called()

    def test_live_tail_reproduces_old_match_and_is_verified_as_reader(self):
        # This child only reads an empty log; no SDK, CAN or movement is used.
        with tempfile.TemporaryDirectory(
            dir=sdk.ROOT / "cup_grasp_demo/datasets"
        ) as tmp:
            log = Path(tmp) / "pipeline.log"
            log.touch()
            child = subprocess.Popen(
                ["/usr/bin/tail", "-f", str(log)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                raw = sdk.core.host_control_evidence("can0")
                self.assertIn(
                    child.pid, [r["pid"] for r in raw["candidate_control_processes"]]
                )
                checked = sdk.control_evidence("can0", {})
                self.assertNotIn(
                    child.pid,
                    [r["pid"] for r in checked["candidate_control_processes"]],
                )
                self.assertIn(
                    child.pid, [r["pid"] for r in checked["verified_log_readers"]]
                )
            finally:
                child.terminate()
                child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
