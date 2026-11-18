"""H10 SKJUTBANA_LATE_REPROPOSE_S unit checks (default OFF).

Never opens camera, microphone or a game window. Run:
    python3 -m automation.late_repropose_selftest
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

fast = importlib.import_module("src.engine.shot_fast_v2225")
async_mod = importlib.import_module("src.engine.shot_async_v2224")
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
from src.engine.camera.hit_scanner import HitScanner
from src.engine.shot_async_v2224 import AsyncDetectorV2224

FLAG = "SKJUTBANA_LATE_REPROPOSE_S"


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def env(value=None):
    clean = {k: v for k, v in os.environ.items() if k not in (FLAG, "SKJUTBANA_CAUSAL_OWNERSHIP")}
    if value is not None:
        clean[FLAG] = value
    return patch.dict(os.environ, clean, clear=True)


def event(sid, peak, state="pending"):
    return SimpleNamespace(shot_id=sid, peak_ts=peak, state=state, lifecycle_state="PENDING",
                           audio_trigger={}, note="")


def scanner_with(*events):
    from collections import deque
    return SimpleNamespace(
        audio_events=deque(events, maxlen=128), frame_history=deque(maxlen=360), known_holes=[],
        debug_frames={}, last_candidates=[], last_trace_pipeline=None, last_window_debug={},
        last_stable_tracks=[], _active_tracks={}, association_lead_s=0.08, association_lag_s=1.5,
        last_threshold_value=0.0, last_change_threshold_value=0.0, last_vote_threshold_value=0.0,
        physical_trace_capture_enabled=False)


class Harness:
    """Real AsyncDetectorV2224 (own shot-cv-v2224 thread) with a recording fake detector."""

    def __init__(self):
        self.det = AsyncDetectorV2224()
        self.calls = []

        def sync_detect(clone, gray, frame_ts):
            sids = [int(ev.shot_id) for ev in clone.audio_events]
            self.calls.append(dict(
                frame_ts=float(frame_ts), clone_sids=sids, thread=threading.current_thread().name,
                late_tls=fast._late_job_shot_id(),
                rescue_view=fast.rescue_router_v2225.requested(sids[0]) if sids else False))
            return [{"camera_x": 100.0, "camera_y": 50.0, "score": 9.0}]

        self.det.set_sync_detector(sync_detect)
        self._ctx = patch.object(async_mod, "async_detector_v2224", self.det)

    def __enter__(self):
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self.det.wait_for_idle(3.0)
        self._ctx.__exit__(*exc)
        self.det.shutdown()


def installed_detect(previous_detect):
    """detect_v2225 wrapped around a recording previous (V2.22.4) detect."""
    with patch.object(HitScanner, "_detect_frame_candidates", previous_detect), \
            patch.object(HitScanner, "_resolve_audio_events", lambda self, now: 0), \
            patch.object(HitScanner, "_v2225_local_confirmation_patch", False, create=True):
        fast._install_local_confirmation_patch()
        return HitScanner._detect_frame_candidates


GRAY = np.zeros((4, 4), np.uint8)


def test_flag_parse():
    for raw, expected in ((None, 0.0), ("0.25", 0.25), ("abc", 0.0), ("-1", 0.0), ("0", 0.0), ("5", 0.0)):
        with env(raw):
            check(fast.late_repropose_s() == expected, f"flag {raw!r} parses to {expected}")


def test_flag_unset_is_identical():
    prev_calls = []
    detect = installed_detect(lambda self, gray, ts: prev_calls.append(ts) or [])
    ev = event(1, 10.0)
    sc = scanner_with(ev)
    frames = [10.05, 10.12, 10.30, 10.45, 10.60]
    with env(None), Harness() as h:
        outs = [detect(sc, GRAY, ts) for ts in frames]
        h.det.wait_for_idle(2.0)
        check(not h.calls and not h.det._futures and not h.det._ready, "flag unset: no extra worker job")
    check(prev_calls == frames and outs == [[]] * len(frames), "flag unset: previous detect sees every frame unchanged")
    check("v26_late_repropose" not in ev.audio_trigger and not hasattr(ev, "_v26_late_repropose_done"),
          "flag unset: no diagnostic or event marker written")
    check(fast._late_job_shot_id() == 0 and not fast.rescue_router_v2225.requested(1),
          "flag unset: rescue router view unchanged")


def test_one_late_job_per_shot():
    prev_calls = []
    detect = installed_detect(lambda self, gray, ts: prev_calls.append(ts) or [])
    ev = event(3, 10.0)
    sc = scanner_with(ev)
    frames = [10.05, 10.12, 10.20, 10.27, 10.31, 10.45, 10.80]
    with env("0.25"), Harness() as h:
        for ts in frames:
            detect(sc, GRAY, ts)
        h.det.wait_for_idle(2.0)
        check(len(h.calls) == 1, "flag set: exactly one extra extraction per shot")
        call = h.calls[0]
        check(call["frame_ts"] == 10.27 and call["frame_ts"] >= 10.0 + 0.25,
              "late job runs on the first frame >= peak + value")
        check(call["clone_sids"] == [3] and call["late_tls"] == 3 and call["rescue_view"],
              "late job is owned by its shot and sees the FULL rescue view inside the worker only")
        check(call["thread"].startswith("shot-cv-v2224"), "late job runs on the single shot-cv-v2224 worker")
        check(not fast.rescue_router_v2225.requested(3) and not fast.rescue_router_v2225.was_consumed(3),
              "main-thread rescue router state untouched")
        ready = h.det.take_ready_for_active({3})
        check(len(ready) == 1 and getattr(ready[0], "v26_late_repropose", False) and ready[0].shot_id == 3,
              "late result is delivered once, marked late, for the producer shot")
        sc.last_window_debug = {}
        AsyncDetectorV2224.apply_result(sc, ready[0])
        fast._record_late_result(sc, ready[0])
        diag = ev.audio_trigger["v26_late_repropose"]
        check(diag["status"] == "delivered" and diag["frame_ts"] == 10.27 and diag["n_candidates"] == 1,
              "diagnostic records frame_ts and n_candidates")
        check(ready[0].candidates[0]["v2224_producer_shot_id"] == 3, "late candidates carry producer-shot ownership")
    check(prev_calls == frames, "normal FAST/confirm path still runs on every frame")


def test_next_peak_cutoff_and_other_shot():
    detect = installed_detect(lambda self, gray, ts: [])
    ev1, ev2 = event(1, 10.0), event(2, 10.15)
    sc = scanner_with(ev1, ev2)
    with env("0.25"), Harness() as h:
        for ts in (10.10, 10.20, 10.26, 10.33, 10.41, 10.50, 10.90):
            detect(sc, GRAY, ts)
        h.det.wait_for_idle(2.0)
        check([c["clone_sids"] for c in h.calls] == [[2]], "only shot 2 gets a late job")
        check(h.calls[0]["frame_ts"] == 10.41 and h.calls[0]["late_tls"] == 2,
              "shot 2's job uses its own first frame >= its peak + value")
    check(ev1.audio_trigger["v26_late_repropose"]["status"] == "skipped_next_peak_cutoff",
          "shot 1 frame at/after next peak cutoff is not used")


def test_terminal_shot_never_submitted():
    detect = installed_detect(lambda self, gray, ts: [])
    ev = event(4, 10.0, state="matched")
    sc = scanner_with(ev)
    with env("0.25"), Harness() as h:
        detect(sc, GRAY, 10.3)
        h.det.wait_for_idle(1.0)
        check(not h.calls, "already emitted shot gets no late job")


def test_extract_routing():
    full_calls, fast_calls = [], []
    with patch.object(CandidateGeneratorV2, "_extract_candidates",
                      lambda self, *a, **k: full_calls.append(k["scanner"]) or ["FULL"]), \
            patch.object(CandidateGeneratorV2, "_v2225_fast_extract_patch", False, create=True), \
            patch.object(CandidateGeneratorV2, "_v2225_full_extract_original", None, create=True), \
            patch.object(fast, "fast_extract_candidates_v2225",
                         lambda self, *a, **k: fast_calls.append(k["scanner"]) or ["FAST"]):
        fast._install_fast_extractor_patch()
        extract = CandidateGeneratorV2._extract_candidates
        sc = scanner_with(event(7, 10.0))
        out = {}

        def worker(late):
            if late:
                fast._LATE_TLS.shot_id = 7
            try:
                out[late] = extract(object(), scanner=sc)
            finally:
                fast._LATE_TLS.shot_id = 0

        fast.rescue_router_v2225.request(7)
        try:
            for late in (True, False):
                t = threading.Thread(target=worker, args=(late,), name="shot-cv-v2224_test")
                t.start()
                t.join()
            check(out[True] == ["FULL"], "late job uses the FULL high-recall extractor")
            check(out[False] == ["FULL"] and fast.rescue_router_v2225.was_consumed(7),
                  "late job leaves the rescue request for the normal rescue job")
        finally:
            fast.rescue_router_v2225.clear(7)
        t = threading.Thread(target=worker, args=(False,), name="shot-cv-v2224_test")
        t.start()
        t.join()
        check(out[False] == ["FAST"] and len(full_calls) == 2, "normal job without request stays FAST")


def test_late_candidates_join_confirm_pool():
    sc = scanner_with(event(9, 10.0))
    sc._v2225_pre_by_shot = {}
    fast.local_confirm_manager_v2225.start(9, 10.15, [{"camera_x": 1.0, "camera_y": 1.0}], None)
    late_c = [{"camera_x": 5.0, "camera_y": 6.0, "pre_shot_change": 7.0}]

    def previous(self, gray, ts):
        self._v2225_result_shot_id, self._v2225_result_frame_ts = 9, 10.27
        self._v26_late_result_shot_id = 9
        return late_c

    detect = installed_detect(previous)
    try:
        with env("0.25"), Harness():
            fast.local_confirm_manager_v2225.get(9).phase = "rescue_queued"
            out = detect(sc, GRAY, 10.20)
        st = fast.local_confirm_manager_v2225.get(9)
        check(out == late_c, "late candidates are returned to tracking")
        check(len(st.candidates) == 2 and st.phase == "rescue_queued" and not fast.rescue_router_v2225.requested(9),
              "late result joins the shot's pool without requesting a rescue")
    finally:
        fast.local_confirm_manager_v2225.clear(9)


def main():
    test_flag_parse()
    test_flag_unset_is_identical()
    test_one_late_job_per_shot()
    test_next_peak_cutoff_and_other_shot()
    test_terminal_shot_never_submitted()
    test_extract_routing()
    test_late_candidates_join_confirm_pool()
    print("PASS late_repropose_selftest")


if __name__ == "__main__":
    main()
