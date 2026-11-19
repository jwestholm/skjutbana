"""Process teardown exercises with fake acquisition and temporary files only."""
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from src.engine.audio.audio_peak_detector import AudioPeakDetector
from src.engine.camera.camera_manager import CameraManager
from src.engine.physical_trace import PhysicalTraceRecorder
from src.engine.source_recording import SourceRecorder


class ShutdownResourceTests(unittest.TestCase):
    def test_late_audio_process_is_reaped_before_exit(self):
        detector = AudioPeakDetector()
        entered, resume = threading.Event(), threading.Event()
        child = Mock()
        child.wait.side_effect = [subprocess.TimeoutExpired("fixture", 0.25), 0]

        def delayed_spawn(*args, **kwargs):
            entered.set()
            if not resume.wait(3):
                raise AssertionError("test did not release the fake backend launch")
            return child

        self.addCleanup(resume.set)
        with patch.object(subprocess, "Popen", side_effect=delayed_spawn):
            detector.running = True
            detector.thread = threading.Thread(target=detector._thread_main)
            detector.thread.start()
            self.assertTrue(entered.wait(3))
            # Drive the race deterministically: Popen returns during the join.
            wait = detector.thread.join
            with patch.object(detector.thread, "join", side_effect=lambda: (resume.set(), wait(3))):
                detector.stop_for_exit()
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_count, 2)
        child.stdout.close.assert_called_once()
        self.assertIsNone(detector.proc)
        self.assertFalse(detector.start())

    def test_existing_audio_child_is_reaped(self):
        detector = AudioPeakDetector()
        child = detector.proc = Mock()
        detector.stop_for_exit()
        child.terminate.assert_called_once()
        child.wait.assert_called_once()
        child.stdout.close.assert_called_once()
        child.kill.assert_not_called()

    def test_stalled_camera_reader_keeps_ownership_of_handle(self):
        camera = CameraManager()
        camera.cap = Mock()
        camera._reader_thread = Mock()
        camera._reader_thread.is_alive.return_value = True
        with self.assertRaises(TimeoutError):
            camera.stop_for_exit()
        camera.cap.release.assert_not_called()
        self.assertFalse(camera.running)

    def test_source_writer_drains_before_session_footer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rec = SourceRecorder(root, repo_root=root, start_min_free_gb=0, queue_max_items=1)
            self.assertTrue(rec.start())
            rec.record_audio(b"sample", {"sequence": 1})
            rec.stop_for_exit()
            self.assertFalse(rec._thread.is_alive())
            self.assertEqual((rec.directory / "audio.pcm").read_bytes(), b"sample")
            footer = json.loads((rec.directory / "session_end.json").read_text())
            self.assertEqual(footer["reason"], "application_exit")
            self.assertEqual(footer["counts"]["unwritten_at_stop"], 0)

    def test_trace_writers_join_after_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = PhysicalTraceRecorder(root, repo_root=root, writer_workers=4, queue_maxsize=1)
            report = recorder.stop_for_exit()
            self.assertTrue(report["drained"])
            self.assertTrue(report["finalized"])
            self.assertFalse(any(w.is_alive() for w in recorder._workers))
            self.assertFalse(recorder._provenance_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
