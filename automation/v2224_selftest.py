from __future__ import annotations

from collections import deque
import importlib.util
from pathlib import Path
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "engine" / "shot_async_v2224.py"


def check(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def load_module():
    spec = importlib.util.spec_from_file_location("v2224_selftest_module", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load shot_async_v2224.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_async_detector(mod) -> None:
    worker = mod.AsyncDetectorV2224()

    def slow_detector(scanner, gray, frame_ts):
        time.sleep(0.08)
        scanner.debug_frames["worker_marker"] = np.array([[7]], dtype=np.uint8)
        scanner.last_window_debug = {"test": 1.0}
        scanner.last_threshold_value = 12.0
        return [{"camera_x": 12.0, "camera_y": 34.0, "timestamp": float(frame_ts), "score": 9.0}]

    worker.set_sync_detector(slow_detector)
    event = types.SimpleNamespace(shot_id=1, peak_ts=10.0, state="pending")
    scanner = types.SimpleNamespace(
        audio_events=deque([event], maxlen=128),
        frame_history=deque([], maxlen=360),
        known_holes=[],
        debug_frames={},
        last_candidates=[],
        last_window_debug={},
        last_threshold_value=0.0,
        last_change_threshold_value=0.0,
        last_vote_threshold_value=0.0,
        last_stable_tracks=[],
        _active_tracks={},
    )
    gray = np.zeros((16, 16), dtype=np.uint8)
    t0 = time.perf_counter()
    worker.submit_if_needed(scanner, gray, 10.10)
    submit_ms = (time.perf_counter() - t0) * 1000.0
    check(submit_ms < 40.0, "CV submission returns without waiting for slow detector")
    check(worker.has_pending(1), "CV job is pending on worker")
    time.sleep(0.11)
    ready = worker.take_ready_for_shot(1)
    check(len(ready) == 1, "CV worker returns one shot-scoped result")
    check(ready[0].worker_ms >= 60.0, "slow detector work occurred on worker")
    check(ready[0].candidates[0]["camera_x"] == 12.0, "candidate data survives worker boundary")
    worker.apply_result(scanner, ready[0])
    check(scanner.last_candidates[0]["camera_y"] == 34.0, "worker result integrates back into scanner state")
    worker.shutdown()




def test_two_shot_harvest(mod) -> None:
    worker = mod.AsyncDetectorV2224()

    def detector(scanner, gray, frame_ts):
        time.sleep(0.025)
        sid = int(scanner.audio_events[0].shot_id)
        return [{"camera_x": float(sid), "camera_y": float(sid), "timestamp": float(frame_ts), "score": 5.0}]

    worker.set_sync_detector(detector)

    def scanner_for(sid: int, peak: float):
        ev = types.SimpleNamespace(shot_id=sid, peak_ts=peak, state="pending")
        return types.SimpleNamespace(
            audio_events=deque([ev], maxlen=128), frame_history=deque([], maxlen=360),
            known_holes=[], debug_frames={}, last_candidates=[], last_window_debug={},
            last_threshold_value=0.0, last_change_threshold_value=0.0, last_vote_threshold_value=0.0,
            last_stable_tracks=[], _active_tracks={},
        )

    worker.submit_if_needed(scanner_for(1, 30.0), np.zeros((8, 8), dtype=np.uint8), 30.10)
    worker.submit_if_needed(scanner_for(2, 30.5), np.zeros((8, 8), dtype=np.uint8), 30.60)
    time.sleep(0.09)
    ready = worker.take_ready_for_active({1, 2})
    check([item.shot_id for item in ready] == [1, 2], "two pending shots are harvested in timestamp order")
    worker.shutdown()

def test_ai_shadow_worker(mod) -> None:
    worker = mod.AIShadowWorkerV2224()
    calls: list[tuple] = []

    def observe(runtime, scanner, event=None):
        time.sleep(0.04)
        calls.append(("observe", int(event.shot_id)))

    def choose(runtime, x, y, shot_id=None):
        calls.append(("choose", int(shot_id)))
        return {"apply": False}

    def mark(runtime, shot_id, state="finished"):
        calls.append(("mark", int(shot_id), str(state)))

    worker.configure(observe, choose, mark)
    event = types.SimpleNamespace(shot_id=4, peak_ts=20.0, state="pending", matched_track_id=None)
    scanner = types.SimpleNamespace(
        audio_events=deque([event], maxlen=128),
        frame_history=deque([], maxlen=360),
        known_holes=[],
        debug_frames={},
        last_candidates=[{"camera_x": 1.0, "camera_y": 2.0, "timestamp": 20.1}],
        last_best_candidate={"camera_x": 1.0, "camera_y": 2.0},
        _active_tracks={},
        _last_frame_ts=20.1,
    )
    runtime = types.SimpleNamespace(settings={"mode": "advisory"})

    empty_scanner = types.SimpleNamespace(
        audio_events=deque([event], maxlen=128), frame_history=deque([], maxlen=360),
        known_holes=[], debug_frames={}, last_candidates=[], last_best_candidate=None,
        _active_tracks={}, _last_frame_ts=20.05,
    )
    worker.submit(runtime, empty_scanner, explicit_event=None)
    time.sleep(0.01)
    check(not calls, "post-update AI shadow ignores empty candidate frames")

    t0 = time.perf_counter()
    worker.submit(runtime, scanner, explicit_event=event)
    submit_ms = (time.perf_counter() - t0) * 1000.0
    worker.record_finish(runtime, 4, "matched")
    check(submit_ms < 30.0, "AI shadow submission is non-blocking")
    time.sleep(0.09)
    check(("observe", 4) in calls, "AI observation runs on shadow worker")
    check(("choose", 4) in calls, "advisory chooser runs on shadow worker")
    check(("mark", 4, "matched") in calls, "terminal AI state is applied after shadow observation")
    worker.shutdown()


def test_source_contract() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    checks = {
        "single CV worker": 'ThreadPoolExecutor(max_workers=1, thread_name_prefix="shot-cv-v2224")' in source,
        "single AI shadow worker": 'ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-shadow-v2224")' in source,
        "advisory passthrough reason": 'v2224_async_shadow_passthrough' in source,
        "main loop always renders": 'pygame.display.flip()' in source,
        "scene simulation can freeze independently": 'freeze_simulation = controller.should_defer_scene_work' in source,
        "main installs V2.22.3 first": 'install_v2223_runtime(App)' in main_source,
        "main installs V2.22.4 second": 'install_v2224_runtime(App)' in main_source,
        "main entrypoint owns async policy": main_source.index('install_v2223_runtime(App)') < main_source.index('install_v2224_runtime(App)') < main_source.index('App().run()'),
    }
    for label, cond in checks.items():
        check(cond, label)


def main() -> None:
    print("V2.22.4 ASYNC SHOT PIPELINE SELFTEST")
    print("===================================")
    check(MODULE_PATH.exists(), "shot_async_v2224.py exists")
    mod = load_module()
    check(mod.SCHEMA_VERSION == "2.22.4", "schema is 2.22.4")
    test_source_contract()
    test_async_detector(mod)
    test_two_shot_harvest(mod)
    test_ai_shadow_worker(mod)
    print("\nAll V2.22.4 selftests passed.")


if __name__ == "__main__":
    main()
