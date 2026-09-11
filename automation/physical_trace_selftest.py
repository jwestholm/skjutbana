"""Synthetic, camera-free tests for fail-open physical trace capture."""
import json
import queue
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from automation.physical_trace_export import export
from src.engine.physical_trace import PhysicalTraceRecorder, TRACE_SCHEMA


class TraceTests(unittest.TestCase):
    def scanner(self):
        pre = np.zeros((4, 5), dtype=np.uint8)
        post = np.ones((4, 5), dtype=np.uint8)
        return SimpleNamespace(
            frame_history=[SimpleNamespace(timestamp=1.0, gray=pre)], pre_shot_snapshot=pre,
            pre_shot_snapshot_ts=0.9, audio_events=[],
            last_candidates=[{"camera_x": 2.0, "camera_y": 3.0, "score": 4.0}],
            last_stable_tracks=[], last_threshold_value=5.0, last_change_threshold_value=3.0,
            last_vote_threshold_value=0.0, last_window_debug={"raw_blobs_total": 1},
            debug_frames={"change_map": post, "camera_gray": post},
            last_best_candidate={"camera_x": 2.0, "camera_y": 3.0},
            last_trace_pipeline={"rejected_blobs": [{"reason": "area"}], "quota_limited": False},
        )

    def _shot(self, recorder, scanner, shot_id, state, emitted=False, rescue=False):
        event = SimpleNamespace(shot_id=shot_id, peak_ts=2.0, state="pending",
                                emitted=False, confidence=0.0, note="")
        recorder.start_from_scanner(scanner, event, {"physical_trace_capture_enabled": True,
                                                       "physical_trace_session_id": "synthetic"})
        scanner.frame_history = [SimpleNamespace(timestamp=1.0, gray=np.zeros((4, 5), dtype=np.uint8)),
                                 SimpleNamespace(timestamp=2.1, gray=np.ones((4, 5), dtype=np.uint8)),
                                 SimpleNamespace(timestamp=2.2, gray=np.full((4, 5), 2, dtype=np.uint8))]
        scanner.audio_events = [event]
        scanner.last_window_debug = {"rescue_used": rescue}
        recorder.observe_scanner(scanner, settings={"physical_trace_capture_enabled": True})
        event.state, event.emitted = state, emitted
        event.note = "timeout" if state == "missed" else "emitted"
        recorder.observe_scanner(scanner, settings={"physical_trace_capture_enabled": True})
        recorder.attach_ground_truth(shot_id, (2, 3))

    def test_synthetic_end_to_end_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder, scanner = PhysicalTraceRecorder(root, enabled=True), self.scanner()
            self._shot(recorder, scanner, 1, "matched", emitted=True)
            self._shot(recorder, scanner, 2, "missed")
            self._shot(recorder, scanner, 3, "matched", emitted=True, rescue=True)
            self.assertTrue(recorder.flush())
            trace = json.loads((root / "shots/shot_00000001/trace.json").read_text())
            self.assertEqual(trace["schema_version"], TRACE_SCHEMA)
            self.assertEqual([f["kind"] for f in trace["frames"]], ["pre_snapshot", "pre_history", "post", "post"])
            self.assertEqual(trace["stages"][0]["pipeline"]["rejected_blobs"][0]["reason"], "area")
            self.assertEqual(trace["outcome"]["status"], "matched")
            timeout_trace = json.loads((root / "shots/shot_00000002/trace.json").read_text())
            rescue_trace = json.loads((root / "shots/shot_00000003/trace.json").read_text())
            self.assertTrue(timeout_trace["outcome"]["timeout"])
            self.assertTrue(rescue_trace["outcome"]["rescue_used"])
            gt = json.loads((root / "shots/shot_00000001/ground_truth.json").read_text())
            self.assertEqual(gt["camera_x"], 2)
            output = root / "evaluation_trace.json"
            export(root, output)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["mode"], "live_path_replay")
            self.assertEqual(payload["shots"][0]["ground_truth"]["camera_x"], 2)
            report_dir = root / "report"
            subprocess.run(["python3", "-m", "automation.evaluate_pipeline", "--trace", str(output),
                            "--dataset-id", "synthetic-physical-trace", "--output", str(report_dir)],
                           check=True, capture_output=True, text=True)
            report = json.loads((report_dir / "report.json").read_text())
            self.assertEqual(report["total_shots"], 3)
            self.assertEqual(report["mode"], "live_path_replay")
            self.assertEqual(report["rescue"]["used"], 1)
            self.assertEqual(report["tolerances_camera_px"]["20"]["stages"]["raw"]["unavailable"], 3)

    def test_disabled_enabled_fail_open_and_ordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scanner, event = self.scanner(), SimpleNamespace(shot_id=9, peak_ts=1.0, state="pending")
            disabled = PhysicalTraceRecorder(root / "disabled", enabled=False)
            before = list(scanner.last_candidates)
            disabled.start_from_scanner(scanner, event)
            disabled.observe_scanner(scanner, event)
            self.assertEqual(before, scanner.last_candidates)
            self.assertFalse(list((root / "disabled").rglob("*")))
            enabled = PhysicalTraceRecorder(root / "enabled", enabled=True)
            enabled.start_from_scanner(scanner, event, {"physical_trace_capture_enabled": True})
            scanner.audio_events = [event]
            enabled.observe_scanner(scanner, settings={"physical_trace_capture_enabled": True})
            self.assertEqual(before, scanner.last_candidates)
            self.assertTrue(enabled.flush())
            frames = json.loads((root / "enabled/shots/shot_00000009/trace.json").read_text())["frames"]
            self.assertEqual([f["timestamp"] for f in frames], [0.9, 1.0])

    def test_missing_data_rejection_corruption_and_queue_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = PhysicalTraceRecorder(root, enabled=True)
            recorder._queue = queue.Queue(maxsize=1)
            job = {"kind": "json", "path": root / "a.json", "value": {"x": 1}, "meta": {}, "shot_id": None}
            self.assertTrue(recorder._enqueue(job))
            self.assertFalse(recorder._enqueue({**job, "path": root / "b.json"}))
            self.assertGreaterEqual(recorder._error_count, 1)
            writer_error = PhysicalTraceRecorder(root / "writer", enabled=True)
            (root / "writer").mkdir(parents=True, exist_ok=True)
            bad_parent = root / "writer" / "not-a-directory"
            bad_parent.write_text("x")
            self.assertTrue(writer_error._enqueue({"kind": "array", "path": bad_parent / "frame.npy", "value": np.ones((2, 2)), "meta": {}, "shot_id": None}))
            self.assertTrue(writer_error.flush())
            self.assertGreaterEqual(writer_error._error_count, 1)
            (root / "shots/shot_00000001").mkdir(parents=True)
            (root / "shots/shot_00000001/trace.json").write_text("{broken")
            with self.assertRaises(ValueError):
                export(root, root / "bad-export.json")

    def test_provenance_and_performance_sanity(self):
        with tempfile.TemporaryDirectory() as directory:
            scanner, event = self.scanner(), SimpleNamespace(shot_id=4, peak_ts=1.0, state="pending")
            off = PhysicalTraceRecorder(Path(directory) / "off", enabled=False)
            start = time.perf_counter()
            for _ in range(200):
                off.observe_scanner(scanner, event)
            disabled_ms = (time.perf_counter() - start) * 1000
            on = PhysicalTraceRecorder(Path(directory) / "on", enabled=True)
            on.start_from_scanner(scanner, event, {"physical_trace_capture_enabled": True})
            start = time.perf_counter()
            for _ in range(200):
                on.observe_scanner(scanner, event, {"physical_trace_capture_enabled": True})
            enabled_ms = (time.perf_counter() - start) * 1000
            self.assertLess(disabled_ms, 1000)
            self.assertLess(enabled_ms, 5000)
            self.assertIn("git_commit", on._provenance({}))
            self.assertIn("git_branch", on._provenance({}))
            self.assertIn("python_version", on._provenance({}))
            print(f"PERF_SANITY disabled_ms={disabled_ms:.2f} enabled_ms={enabled_ms:.2f}")
            self.assertTrue(on.flush())


if __name__ == "__main__":
    unittest.main()
