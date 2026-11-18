"""Exercise the latency-report CLI using only small executor profile fixtures."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from automation.bench.__main__ import main


class LatencyReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def write_jobs(self, jobs):
        (self.directory / "wall_clock_profile.json").write_text(
            json.dumps(dict(version=1, clock="host_monotonic_ns", jobs=jobs)), encoding="utf-8")

    def report(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["latency-report", str(self.directory)]), 0)
        return out.getvalue()

    def test_intervals_and_coverage(self):
        base = 2**61
        self.write_jobs([
            dict(thread="MainThread", submit_ns=base, start_ns=base + 1, finish_ns=base + 25_000_001),
            dict(thread="worker", submit_ns=base, start_ns=base, finish_ns=None),
            dict(thread=None, submit_ns=base, start_ns=None, finish_ns=None),
        ])
        text = self.report()
        self.assertIn("1\tMainThread\t0.000001\t25.000000\t25.000001", text)
        self.assertIn("2\tworker\t0.000000\tUNKNOWN\tUNKNOWN", text)
        self.assertIn("3\tUNKNOWN\tUNKNOWN\tUNKNOWN\tUNKNOWN", text)
        self.assertIn("submit_to_start_ms  0.000001  (2/3)", text)
        self.assertIn("start_to_finish_ms  25.000000  (1/3)", text)
        self.assertIn("submit_to_finish_ms  25.000001  (1/3)", text)

    def test_empty_profile(self):
        self.write_jobs([])
        text = self.report()
        self.assertIn("Submitted jobs: 0", text)
        self.assertEqual(text.count("UNKNOWN  (0/0)"), 3)

    def test_missing_sidecar_and_invalid_order(self):
        for issue in ("missing", "inconsistent timestamps"):
            with self.subTest(issue=issue):
                if issue != "missing":
                    self.write_jobs([dict(thread=None, submit_ns=15, start_ns=None, finish_ns=16)])
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                    main(["latency-report", str(self.directory)])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(issue, errors.getvalue())


if __name__ == "__main__":
    unittest.main()
