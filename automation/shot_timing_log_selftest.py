"""Per-shot timing log: off by default, records jobs + finish, analysis summarises it.

  python3 -m automation.shot_timing_log_selftest
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class ShotTimingLog(unittest.TestCase):
    def test_off_by_default_and_roundtrip(self):
        from src.engine import shot_timing_log as stl
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.jsonl"
            os.environ.pop("SKJUTBANA_SHOT_TIMING_LOG", None)
            stl._PATH = None
            os.environ["SKJUTBANA_SHOT_TIMING_LOG_PATH"] = str(path)
            res = types.SimpleNamespace(shot_id=3, frame_ts=time.time(), peak_ts=time.time(),
                                        submitted_mono=time.monotonic(), queue_ms=5.0, worker_ms=300.0,
                                        stages_ms={"extract": 120.0}, candidates=[{}, {}], error="")
            stl.record_job(res)
            stl.record_finish(3, "matched", time.time() - 1.0, 1000.0)
            self.assertFalse(path.exists())          # disabled: nothing written
            os.environ["SKJUTBANA_SHOT_TIMING_LOG"] = "1"
            try:
                peak = time.time() - 0.5
                stl.record_job(res)
                ev = types.SimpleNamespace(audio_trigger={"timing": {"worker_queue_ms": 5.0, "flag": True}})
                stl.record_finish(3, "matched", peak, 900.0, ev)
                rows = [json.loads(l) for l in path.read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["n_jobs"], 1)
                self.assertEqual(rows[0]["jobs"][0]["candidates"], 2)
                self.assertNotIn("flag", rows[0]["timing"])
                from automation.bench.shot_timing import main
                buf = io.StringIO()
                with redirect_stdout(buf):
                    self.assertEqual(main([str(path)]), 0)
                self.assertIn("hit_ms", buf.getvalue())
                self.assertIn("extract", buf.getvalue())
            finally:
                os.environ.pop("SKJUTBANA_SHOT_TIMING_LOG", None)
                os.environ.pop("SKJUTBANA_SHOT_TIMING_LOG_PATH", None)
                stl._PATH = None


if __name__ == "__main__":
    unittest.main()
