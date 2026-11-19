"""Exercise the real shutdown supervisor with synthetic child processes."""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch
from contextlib import redirect_stderr
import io
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from src.engine.process_shutdown import PIPE_ENV, install_child_handlers, supervise


def fixture(directory: Path, mode: str):
    if PIPE_ENV not in os.environ:
        return supervise([sys.executable, "-m", "automation.process_shutdown_selftest",
                          "--fixture", str(directory), mode], grace_s=0.6)
    guard = install_child_handlers()
    if mode not in ("clean", "duplicate"):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(1)
        pool.submit(threading.Event().wait)
        # Retains inherited stdout/stderr. communicate() reaches EOF only if
        # this descendant also exits when the stuck executor is force-stopped.
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid_file = directory / "pid.tmp"
    pid_file.write_text(str(os.getpid()))
    pid_file.replace(directory / "pid")
    if mode == "direct-stall":
        while True:
            time.sleep(1.0)
    if mode != "normal-quit":
        while not guard.requested:
            time.sleep(0.01)
    if mode == "duplicate":
        time.sleep(0.1)
    guard.begin_cleanup()
    (directory / "cleanup").touch()
    if mode == "hang":
        pool.shutdown(wait=True)
    # normal-quit deliberately leaves the executor to Python's exit hook.
    return 0


class ShutdownTests(unittest.TestCase):
    def exercise(self, sig, mode, expected):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            env = dict(os.environ)
            env.pop(PIPE_ENV, None)
            command = [sys.executable, "-m", "automation.process_shutdown_selftest",
                       "--fixture", tmp, mode]
            proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 5
                while not (directory / "pid").exists() and time.monotonic() < deadline:
                    if proc.poll() is not None:
                        self.fail(f"fixture exited during startup: {proc.communicate()}")
                    time.sleep(0.01)
                self.assertTrue((directory / "pid").is_file())
                if sig is not None:
                    if mode == "direct-stall":
                        os.kill(int((directory / "pid").read_text()), sig)
                    else:
                        proc.send_signal(sig)
                        if mode == "duplicate":
                            time.sleep(0.02)
                            proc.send_signal(sig)
                _, stderr = proc.communicate(timeout=4)
                self.assertEqual(proc.returncode, expected, stderr)
                self.assertEqual((directory / "cleanup").exists(), mode != "direct-stall")
            finally:
                if proc.poll() is None:
                    try:
                        os.killpg(int((directory / "pid").read_text()), signal.SIGKILL)
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                    proc.kill()
                proc.wait(timeout=2)

    def test_clean_signals(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(sig=sig):
                self.exercise(sig, "clean", 0)

    def test_forwarded_duplicate_sigint_is_graceful(self):
        self.exercise(signal.SIGINT, "duplicate", 0)

    def test_stuck_executor_and_owned_subprocess(self):
        self.exercise(signal.SIGINT, "hang", 124)

    def test_signal_to_engine_pid_starts_supervisor_deadline(self):
        self.exercise(signal.SIGTERM, "direct-stall", 124)

    def test_bounded_interpreter_exit_after_normal_quit(self):
        self.exercise(None, "normal-quit", 124)

    def test_executor_shutdown_is_idempotent(self):
        from src.engine.shot_async_v2224 import AIShadowWorkerV2224, AsyncDetectorV2224

        for worker_class in (AsyncDetectorV2224, AIShadowWorkerV2224):
            for cancel_first in (False, True):
                with self.subTest(worker=worker_class.__name__, cancel_first=cancel_first):
                    worker = worker_class()
                    with patch.object(worker._executor, "shutdown", wraps=worker._executor.shutdown) as stop:
                        if cancel_first:
                            worker.shutdown(wait=False)
                            worker.shutdown(wait=False)
                        worker.shutdown()
                        worker.shutdown()
                    expected = [call(wait=False, cancel_futures=True)] if cancel_first else []
                    self.assertEqual(stop.call_args_list, expected + [call(wait=True, cancel_futures=True)])

    def test_application_cleanup_uses_existing_services(self):
        from src.engine import app

        for shared in (True, False):
            with self.subTest(shared_trace_recorder=shared):
                events = []

                def mark(name):
                    def call():
                        events.append(name)
                        if name == "camera":
                            raise RuntimeError("reader still busy")
                    return call

                audio = SimpleNamespace(stop_for_exit=mark("audio"))
                camera = SimpleNamespace(stop_for_exit=mark("camera"))
                source = SimpleNamespace(stop_for_exit=mark("source"))
                trace = SimpleNamespace(stop_for_exit=mark("trace"))
                secondary = trace if shared else SimpleNamespace(stop_for_exit=mark("object trace"))
                engine = SimpleNamespace(communication_server=SimpleNamespace(stop=mark("server")),
                                         scene=SimpleNamespace(on_exit=mark("scene")))
                modules = {
                    "src.engine.source_recording": SimpleNamespace(_RECORDER=source),
                    "src.engine.shot_async_v2224": SimpleNamespace(_shutdown_workers=mark("workers")),
                    "src.engine.physical_trace": SimpleNamespace(_RECORDER=trace, _OBJECT_RECORDER=secondary),
                }
                with patch.dict(sys.modules, modules), patch.multiple(
                    app, camera_manager=camera, audio_peak_detector=audio,
                    led_service=SimpleNamespace(stop=mark("LED")),
                    hit_scanner=SimpleNamespace(disable=mark("scanner")),
                    pygame=SimpleNamespace(quit=mark("pygame")),
                ), redirect_stderr(io.StringIO()) as errors:
                    app.App.close_for_exit(engine)
                self.assertIn("reader still busy", errors.getvalue())
                self.assertEqual(events, ["server", "audio", "camera", "source", "scene", "LED",
                                          "workers", "scanner", "trace"] +
                                 ([] if shared else ["object trace"]) + ["pygame"])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fixture"]:
        raise SystemExit(fixture(Path(sys.argv[2]), sys.argv[3]))
    unittest.main()
