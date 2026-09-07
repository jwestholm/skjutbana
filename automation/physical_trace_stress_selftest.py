"""Large-array asynchronous physical trace persistence stress tests."""
import json
import tempfile
import time
import statistics
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from automation.physical_trace_export import export
from src.engine.physical_trace import PhysicalTraceRecorder


class TraceStressTests(unittest.TestCase):
    def scanner(self, frame):
        return SimpleNamespace(
            frame_history=[], pre_shot_snapshot=frame, pre_shot_snapshot_ts=0.0,
            audio_events=[], last_candidates=[{"camera_x": 100.0, "camera_y": 200.0, "score": 3.0}],
            last_stable_tracks=[], last_window_debug={"rescue_used": False},
            debug_frames={"change_map": frame[::4, ::4], "combined_map": frame[::4, ::4]},
            last_best_candidate={"camera_x": 100.0, "camera_y": 200.0},
            last_trace_pipeline={"rejected_blobs": [{"reason": "area"}]},
        )

    def test_ten_shot_large_frame_session_has_no_dangling_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = np.zeros((2160, 3840), dtype=np.uint8)
            recorder = PhysicalTraceRecorder(root, enabled=True)
            capture_start = time.perf_counter()
            for shot_id in range(1, 11):
                scanner = self.scanner(frame)
                event = SimpleNamespace(shot_id=shot_id, peak_ts=10.0, state="pending", emitted=False, note="")
                scanner.frame_history = [SimpleNamespace(timestamp=1.0, gray=frame), SimpleNamespace(timestamp=2.0, gray=frame)]
                recorder.start_from_scanner(scanner, event, {"physical_trace_capture_enabled": True, "physical_trace_session_id": "stress"})
                scanner.audio_events = [event]
                for n in range(10):
                    scanner.frame_history.append(SimpleNamespace(timestamp=11.0 + n, gray=frame))
                    recorder.observe_scanner(scanner, settings={"physical_trace_capture_enabled": True})
                event.state = "missed" if shot_id % 3 == 0 else "matched"
                event.emitted = event.state == "matched"
                scanner.last_window_debug = {"rescue_used": shot_id == 2}
                recorder.observe_scanner(scanner, settings={"physical_trace_capture_enabled": True})
                recorder.attach_ground_truth(shot_id, (100.0, 200.0))
            enqueue_ms = (time.perf_counter() - capture_start) * 1000.0
            flush_start = time.perf_counter()
            shutdown_report = recorder.shutdown(timeout=60.0)
            writer_ms = (time.perf_counter() - flush_start) * 1000.0
            stats = recorder.stats()
            self.assertTrue(shutdown_report["drained"])
            self.assertEqual(stats["writer_errors"], 0)
            self.assertEqual(stats["queue_drops"], 0)
            self.assertGreater(stats["queue_high_water"], 0)
            samples = recorder.capture_samples()
            latencies = sorted(float(s["total_ms"]) for s in samples)
            copy_latencies = [float(s["copy_ms"]) for s in samples]
            enqueue_latencies = [float(s["enqueue_ms"]) for s in samples]
            p50 = latencies[len(latencies) // 2] if latencies else 0.0
            p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0
            print(f"REALISTIC_PERF session_enqueue_ms={enqueue_ms:.1f} avg_capture_ms={statistics.mean(latencies):.3f} avg_copy_ms={statistics.mean(copy_latencies):.3f} avg_queue_insert_ms={statistics.mean(enqueue_latencies):.3f} p50_capture_ms={p50:.3f} p95_capture_ms={p95:.3f} max_capture_ms={max(latencies):.3f} writer_flush_ms={writer_ms:.1f} queue_high_water={stats['queue_high_water']} queue_high_water_bytes={stats['queue_high_water_bytes']} persisted_items={stats['persisted_items']}")
            for shot_dir in sorted((root / "shots").glob("shot_*")):
                trace_path = shot_dir / "trace.json"
                trace = json.loads(trace_path.read_text())
                completeness = trace["completeness"]
                self.assertTrue(completeness["trace_complete"])
                self.assertEqual(completeness["pre_frames_expected"], completeness["pre_frames_persisted"])
                self.assertEqual(completeness["post_frames_expected"], completeness["post_frames_persisted"])
                self.assertEqual(completeness["evidence_artifacts_expected"], completeness["evidence_artifacts_persisted"])
                for frame_ref in trace["frames"]:
                    self.assertTrue((shot_dir / frame_ref["path"]).exists())
                for stage in trace["stages"]:
                    for ref in stage["evidence_maps"].values():
                        self.assertTrue((shot_dir / ref["path"]).exists())
            output = root / "evaluation_trace.json"
            export(root, output)
            exported = json.loads(output.read_text())
            self.assertEqual(len(exported["shots"]), 10)
            self.assertTrue(all(shot["trace_complete"] for shot in exported["shots"]))

    def test_failed_artifact_is_not_referenced_and_is_fail_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = np.empty((8, 8), dtype=object)
            frame.fill(object())
            recorder = PhysicalTraceRecorder(root, enabled=True)
            scanner = self.scanner(np.zeros((8, 8), dtype=np.uint8))
            event = SimpleNamespace(shot_id=1, peak_ts=1.0, state="pending")
            recorder.start_from_scanner(scanner, event, {"physical_trace_capture_enabled": True})
            recorder._save_frame(1, "post", 2.0, frame)
            event.state = "missed"
            recorder.finish(1, scanner, event)
            self.assertTrue(recorder.flush())
            trace = json.loads((root / "shots/shot_00000001/trace.json").read_text())
            self.assertEqual([frame["kind"] for frame in trace["frames"]], ["pre_snapshot"])
            self.assertFalse(trace["completeness"]["trace_complete"])
            self.assertTrue(trace["completeness"]["errors"])
            self.assertTrue(trace["completeness"]["errors"][0]["exception_type"])
            self.assertEqual(trace["completeness"]["post_frames_persisted"], 0)
            self.assertFalse(any(frame["kind"] == "post" for frame in trace["frames"]))
            self.assertTrue(all((root / "shots/shot_00000001" / frame["path"]).exists() for frame in trace["frames"]))

    def test_slow_writer_is_bounded_and_fail_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = np.zeros((256, 256), dtype=np.uint8)
            recorder = PhysicalTraceRecorder(root, enabled=True, queue_maxsize=8, writer_delay_s=0.02)
            scanner = self.scanner(frame)
            event = SimpleNamespace(shot_id=1, peak_ts=1.0, state="pending", emitted=False)
            scanner.frame_history = [SimpleNamespace(timestamp=0.5, gray=frame)]
            recorder.start_from_scanner(scanner, event, {"physical_trace_capture_enabled": True})
            scanner.audio_events = [event]
            original_candidates = list(scanner.last_candidates)
            for n in range(20):
                scanner.frame_history.append(SimpleNamespace(timestamp=2.0 + n, gray=frame))
                recorder.observe_scanner(scanner, settings={"physical_trace_capture_enabled": True})
            event.state = "missed"
            recorder.observe_scanner(scanner, settings={"physical_trace_capture_enabled": True})
            self.assertTrue(recorder.shutdown(timeout=10.0)["drained"])
            self.assertEqual(original_candidates, scanner.last_candidates)
            stats = recorder.stats()
            self.assertLessEqual(stats["queue_high_water"], 8)
            self.assertGreater(stats["queue_drops"], 0)
            trace = json.loads((root / "shots/shot_00000001/trace.json").read_text())
            self.assertFalse(trace["completeness"]["trace_complete"])
            self.assertGreater(trace["completeness"]["queue_drops"], 0)
            for ref in trace["frames"]:
                self.assertTrue((root / "shots/shot_00000001" / ref["path"]).exists())


if __name__ == "__main__":
    unittest.main()
