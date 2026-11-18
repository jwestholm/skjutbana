"""Deterministic reliability checks; never opens camera, microphone or a game window."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import inspect
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

audio_module = importlib.import_module("src.engine.audio.audio_peak_detector")
scanner_module = importlib.import_module("src.engine.camera.hit_scanner")
controller_module = importlib.import_module("src.engine.shot_critical_v2223")

from src.engine.audio.audio_peak_detector import AudioPeakDetector, AudioPeakEvent
from src.engine.camera.hit_scanner import AudioShotEvent, HitScanner, HoleTrack, select_pre_background_frame_v1
from src.engine.input.object_hit_v2223 import ObjectHitRegistryV2223
from src.engine.shot_fast_v2225 import _drain_completed_before_timeout, _install_local_confirmation_patch
from src.engine.shot_async_v2224 import AsyncDetectorV2224


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


@dataclass
class WorkerResult:
    shot_id: int
    frame_ts: float
    finished_mono: float
    candidates: list


class ReadyQueue:
    def __init__(self, rows):
        self.rows = list(rows)
        self.applied = []

    def take_ready_for_active(self, active, keep_shot_ids=frozenset()):
        selected = [row for row in self.rows if row.shot_id in active]
        self.rows = [row for row in self.rows if row.shot_id in keep_shot_ids]
        return selected

    def apply_result(self, scanner, result):
        self.applied.append(result.shot_id)


class FakeScanner:
    association_lead_s = 0.08
    association_lag_s = 1.5

    def __init__(self, event):
        self.audio_events = [event]
        self.updated = []

    def _update_tracks(self, candidates, frame_ts):
        self.updated.append((list(candidates), frame_ts))


def test_deadline_drain():
    event = SimpleNamespace(shot_id=4, peak_ts=10.0, state="pending",
                            lifecycle_state="PROCESSING", decision_deadline_monotonic=20.0,
                            note="")
    result = WorkerResult(4, 11.0, 19.9, [{"camera_x": 12.0, "camera_y": 13.0}])
    scanner = FakeScanner(event)
    queue = ReadyQueue([result])
    consumed = []

    def consume(_scanner, candidates):
        consumed.extend(candidates)
        return candidates

    counts = _drain_completed_before_timeout(scanner, queue, consume)
    check(counts == {"ready": 1, "integrated": 1, "late_or_stale": 0, "orphaned": 0},
          "pre-deadline completed result is integrated at timeout boundary")
    check(queue.applied == [4] and scanner.updated[0][1] == 11.0,
          "worker shot identity and source frame timestamp are preserved")
    check(consumed[0]["camera_x"] == 12.0 and consumed[0]["camera_y"] == 13.0,
          "candidate XY is unchanged")
    check(_drain_completed_before_timeout(scanner, queue, consume)["ready"] == 0,
          "ready result is never integrated twice")

    late = WorkerResult(4, 11.0, 20.001, [{"camera_x": 99.0, "camera_y": 99.0}])
    queue = ReadyQueue([late])
    scanner = FakeScanner(event)
    counts = _drain_completed_before_timeout(scanner, queue, consume)
    check(counts["late_or_stale"] == 1 and not scanner.updated,
          "post-deadline worker result is explicitly rejected")
    check(event.note == "worker_result_after_original_deadline_discarded",
          "late result rejection is recorded on its owning event")

    stale = WorkerResult(4, 11.5001, 19.0, [{"camera_x": 1.0, "camera_y": 2.0}])
    event.note = ""
    queue = ReadyQueue([stale])
    scanner = FakeScanner(event)
    counts = _drain_completed_before_timeout(scanner, queue, consume)
    check(counts["late_or_stale"] == 1 and not scanner.updated,
          "frame beyond original association cutoff cannot be integrated")

    early = WorkerResult(4, 9.9199, 19.0, [{"camera_x": 1.0, "camera_y": 2.0}])
    event.peak_ts = 10.0
    queue = ReadyQueue([early])
    scanner = FakeScanner(event)
    counts = _drain_completed_before_timeout(scanner, queue, consume)
    check(counts["late_or_stale"] == 1 and not scanner.updated,
          "frame before the existing association lead cannot be integrated")


def test_audio_queue_timing():
    detector = AudioPeakDetector()
    rows = []
    detector.subscribe(lambda ev: rows.append(ev.acquisition))
    for seq in range(2):
        ev = AudioPeakEvent(timestamp=100.0 + seq, peak=.8, rms=.1,
                            acquisition={"event_enqueued_monotonic": 1.0 + seq,
                                         "pending_depth_at_enqueue": seq + 1})
        detector._pending_dispatch.append(ev)
    with patch.object(audio_module.time, "monotonic",
                      side_effect=[2.0, 2.001, 2.002, 2.1, 2.101, 2.102]):
        detector.update()
    check(len(rows) == 2 and [r["dispatch_batch_index"] for r in rows] == [0, 1],
          "audio dispatch preserves FIFO batch order and indexes")
    check([r["dispatch_batch_size"] for r in rows] == [2, 2],
          "audio trace records batch depth")
    check(abs(rows[0]["dispatch_queue_age_ms"] - 1000.0) < 1e-6
          and abs(rows[1]["dispatch_queue_age_ms"] - 100.0) < 1e-6,
          "audio queue age uses monotonic timestamps")
    check(abs(rows[0]["callback_duration_ms"] - 1.0) < 1e-6
          and abs(rows[1]["callback_duration_ms"] - 1.0) < 1e-6,
          "subscriber duration is measured without changing dispatch ownership")


def test_lifecycle_and_pre_provenance():
    scanner = HitScanner()
    scanner.enabled = True
    scanner.state = scanner.STATE_ACTIVE
    with patch.object(scanner_module.camera_manager, "get_ring_snapshot", return_value=[]):
        scanner._on_audio_peak(AudioPeakEvent(timestamp=100.0, peak=.8, rms=.1,
            acquisition={"dispatch_monotonic": 5.0, "event_enqueued_monotonic": 4.9}))
    event = scanner.audio_events[-1]
    check(event.lifecycle_state == "PENDING" and event.decision_deadline_monotonic >= event.created_monotonic,
          "shot begins pending with bounded monotonic deadline")
    check(event.audio_trigger["pre_source"] == "UNAVAILABLE" and event.audio_trigger["pre_timestamp"] is None,
          "missing PRE is not represented as a fabricated frame timestamp")
    scanner.disable()
    check(event.lifecycle_state == "CANCELLED", "disable cancels pending lifecycle explicitly")


def test_executor_submit_failure_is_contained():
    class FailingExecutor:
        def submit(self, *_args, **_kwargs):
            raise RuntimeError("synthetic executor saturation")

    import threading
    import numpy as np
    detector = AsyncDetectorV2224.__new__(AsyncDetectorV2224)
    detector._shutdown = False
    detector._sync_detect = lambda *_args: []
    detector._max_queued_per_shot = 1
    detector._frame_spacing_s = 0.055
    detector._submitted_frame_ts = {}
    detector._futures = {}
    detector._lock = threading.RLock()
    detector._executor = FailingExecutor()
    event = SimpleNamespace(shot_id=8, peak_ts=10.0, lifecycle_state="PENDING",
                            audio_trigger={"timing": {}}, note="")
    scanner = SimpleNamespace(audio_events=[event])
    detector._target_event = lambda _scanner, _ts: event
    detector._pending_for_shot = lambda _sid: 0
    detector._clone_scanner = lambda *_args: object()
    detector.submit_if_needed(scanner, np.zeros((2, 2), dtype="uint8"), 10.1)
    check(event.lifecycle_state == "PENDING" and not detector._submitted_frame_ts,
          "executor submit failure leaves event retryable and unthrottled")
    check(event.note == "worker_submit_failed"
          and "synthetic executor saturation" in event.audio_trigger["timing"]["worker_submit_error"],
          "executor submit fault is recorded without escaping the frame update")
    check(not detector._futures, "failed submit does not create a phantom worker job")


def test_executor_submit_failure_then_recovery_is_consumed_once():
    from concurrent.futures import Future
    import threading
    import numpy as np

    class FailOnceExecutor:
        def __init__(self):
            self.calls = 0

        def submit(self, fn, *args):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("one-shot saturation")
            future = Future()
            future.set_result(fn(*args))
            return future

    event = SimpleNamespace(shot_id=18, peak_ts=10.0, lifecycle_state="PENDING",
                            state="pending", audio_trigger={"timing": {}}, note="")
    scanner = SimpleNamespace(audio_events=[event], last_candidates=[], debug_frames={},
                              last_window_debug={}, last_threshold_value=0.0,
                              last_change_threshold_value=0.0, last_vote_threshold_value=0.0)
    consumed = []
    scanner._update_tracks = lambda candidates, frame_ts: consumed.append((candidates, frame_ts))
    detector = AsyncDetectorV2224.__new__(AsyncDetectorV2224)
    detector._shutdown = False
    detector._sync_detect = lambda *_args: [{"camera_x": 22.0, "camera_y": 31.0}]
    detector._max_queued_per_shot = 1
    detector._frame_spacing_s = 0.055
    detector._submitted_frame_ts = {}
    detector._futures = {}
    detector._ready = __import__("collections").deque()
    detector._last_applied_frame_ts = {}
    detector._lock = threading.RLock()
    detector._executor = FailOnceExecutor()
    detector._target_event = lambda _scanner, _ts: event
    detector._pending_for_shot = lambda _sid: 0
    detector._clone_scanner = lambda *_args: SimpleNamespace(debug_frames={},
        last_window_debug={}, last_threshold_value=0.0, last_change_threshold_value=0.0,
        last_vote_threshold_value=0.0, physical_trace_capture_enabled=False,
        last_trace_pipeline=None)

    detector.submit_if_needed(scanner, np.zeros((2, 2), dtype="uint8"), 10.1)
    check(event.lifecycle_state == "PENDING" and not detector._futures,
          "first transient submission fault leaves shot retryable")
    detector.submit_if_needed(scanner, np.zeros((2, 2), dtype="uint8"), 10.2)
    check(detector._executor.calls == 2 and len(detector._futures) == 1
          and event.lifecycle_state == "PROCESSING",
          "next frame submits exactly one worker job after recovery")
    ready = detector.take_ready_for_active({18})
    check(len(ready) == 1 and ready[0].shot_id == 18 and ready[0].frame_ts == 10.2,
          "recovered job is harvested once with its owning shot and frame")
    AsyncDetectorV2224.apply_result(scanner, ready[0])
    scanner._update_tracks(ready[0].candidates, ready[0].frame_ts)
    check(len(consumed) == 1 and consumed[0][0][0]["camera_x"] == 22.0
          and consumed[0][0][0]["v2224_producer_shot_id"] == 18
          and consumed[0][1] == 10.2,
          "recovered candidate is consumed once without XY or ownership loss")
    check(detector.take_ready_for_active({18}) == [] and not detector._futures,
          "subsequent drain cannot consume the recovered result twice")


def test_geometry_provenance_fields():
    registry = ObjectHitRegistryV2223()
    with patch.object(registry, "_load_viewport_xywh", return_value=None), patch.object(
            registry, "_camera_transformers", return_value=("none", None, None)):
        first = registry.snapshot(1, 10.0, pose_provenance="CURRENT_SCENE_BEFORE_AUDIO_CALLBACK_EVENT_TIME_UNVERIFIED",
                                  pose_capture_monotonic=20.0, queued_audio_batch_size=3)
        second = registry.snapshot(2, 10.1, pose_provenance="CURRENT_SCENE_AFTER_AUDIO_CALLBACK_EVENT_TIME_UNAVAILABLE",
                                   pose_capture_monotonic=20.1, queued_audio_batch_size=3)
    check(first.pose_provenance.endswith("EVENT_TIME_UNVERIFIED") and first.queued_audio_batch_size == 3,
          "first batched pose is identified as current-at-dispatch, not event-time truth")
    check(second.pose_provenance.endswith("EVENT_TIME_UNAVAILABLE") and second.peak_ts == 10.1,
          "later batch snapshot preserves event identity and marks event pose unavailable")
    context = registry.game_context(2)
    check(context["pose_provenance"] == second.pose_provenance and context["pose_capture_monotonic"] == 20.1,
          "geometry provenance survives the existing game-context API")


def test_batched_pose_provenance():
    from src.engine.shot_critical_v2223 import ShotCriticalControllerV2223

    class Detector:
        last_peak_ts = 10.2
        def pending_events_snapshot(self):
            return (SimpleNamespace(timestamp=10.0), SimpleNamespace(timestamp=10.2))

    class Scanner:
        enabled = True
        _next_shot_id = 4
        audio_events = [
            SimpleNamespace(shot_id=4, peak_ts=10.0,
                audio_trigger={"diagnostics": {"acquisition": {"dispatch_monotonic": 100.1}}}),
            SimpleNamespace(shot_id=5, peak_ts=10.2,
                audio_trigger={"diagnostics": {"acquisition": {"dispatch_monotonic": 100.2}}}),
        ]

    snapshots = {}
    calls = []
    def snapshot(sid, peak, **kwargs):
        snap = SimpleNamespace(shot_id=sid, peak_ts=peak, regions=(), pose_capture_monotonic=100.0,
            event_dispatch_monotonic=None, pose_age_at_dispatch_ms=None,
            pose_provenance=kwargs["pose_provenance"], queued_audio_batch_size=kwargs["queued_audio_batch_size"])
        calls.append((sid, kwargs))
        snapshots[sid] = snap
        return snap
    controller = ShotCriticalControllerV2223()
    controller._set_wait_cursor = lambda _app: None
    peak = controller.begin_pending_audio(None, Detector())
    with patch.object(controller_module.object_hit_registry_v2223, "snapshot", side_effect=snapshot), \
         patch.object(controller_module.object_hit_registry_v2223, "snapshot_for_shot",
                      side_effect=lambda sid: snapshots.get(sid)), \
         patch.object(controller_module.object_hit_registry_v2223, "begin_object_query", return_value=None):
        controller.prepare_object_snapshot(Scanner, SimpleNamespace(), peak)
        controller.discover_scanner_events(Scanner, SimpleNamespace())
    check(calls[0][0] == 4 and calls[0][1]["queued_audio_batch_size"] == 2,
          "pre-dispatch frozen pose records the actual queued batch size")
    check(calls[0][1]["pose_provenance"].endswith("EVENT_TIME_UNVERIFIED"),
          "first batched pose is not falsely certified as event-time geometry")
    check(calls[1][0] == 5 and calls[1][1]["queued_audio_batch_size"] == 2
          and calls[1][1]["pose_provenance"].endswith("EVENT_TIME_UNAVAILABLE"),
          "later batched event gets explicit unavailable event-pose provenance")


def test_camera_age_and_emit_parity():
    frames = [SimpleNamespace(timestamp=99.40, acquisition={"sequence": 10,
                "read_complete_monotonic": 8.0, "published_monotonic": 8.001,
                "exposure_monotonic": None}),
              SimpleNamespace(timestamp=100.01, acquisition={"sequence": 11,
                "read_complete_monotonic": 8.6, "published_monotonic": 8.601,
                "exposure_monotonic": None})]
    selected = select_pre_background_frame_v1(frames, 100.0)
    check(selected is frames[0] and selected.acquisition["exposure_monotonic"] is None,
          "PRE selector uses receive-time cutoff and leaves exposure time unknown")
    check(frames[1].timestamp > 100.0 and frames[1].acquisition["sequence"] == 11,
          "frame sequence and timestamp retain ownership for delayed consumption")

    outputs = []
    for trace_on in (False, True):
        scanner = HitScanner()
        event = AudioShotEvent(shot_id=7, peak_ts=50.0, created_at=50.1,
            audio_trigger={"timing": {"trace_enabled": True}} if trace_on else {})
        track = HoleTrack(track_id=3, camera_x=123.25, camera_y=456.5, created_at=50.0,
            first_seen_ts=50.1, last_seen_ts=50.3, hits=3, best_score=8.0)
        with patch.object(scanner_module.hit_input, "push_camera_hit", side_effect=lambda x, y: outputs.append((x, y))):
            emitted = scanner._emit_track_result(track, event)
        outputs.append((emitted, event.state, event.lifecycle_state))
    check(outputs[0] == outputs[2] == (123.25, 456.5),
          "trace metadata leaves emitted camera XY unchanged")
    check(outputs[1] == outputs[3] == (True, "matched", "COMPLETED"),
          "trace metadata leaves shot decision and lifecycle unchanged")


def test_deadline_drain_is_opt_in():
    source = inspect.getsource(_install_local_confirmation_patch)
    check('SKJUTBANA_DRAIN_READY_BEFORE_TIMEOUT' in source
          and '_drain_completed_before_timeout' in source,
          "deadline-boundary drain is guarded by an explicit opt-in flag")


def test_deadline_drain_respects_deadline_rescue_and_other_shots():
    import time as _time
    from src.engine.shot_fast_v2225 import rescue_router_v2225
    from src.engine.shot_async_v2224 import DetectorJobResultV2224

    def consume(_scanner, candidates):
        return candidates

    # Not yet at deadline: result must remain for the normal detect/rescue path.
    future = _time.monotonic() + 100.0
    event = SimpleNamespace(shot_id=71, peak_ts=10.0, state="pending", lifecycle_state="PROCESSING",
                            decision_deadline_monotonic=future, note="")
    queue = ReadyQueue([WorkerResult(71, 10.2, 1.0, [])])
    counts = _drain_completed_before_timeout(FakeScanner(event), queue, consume)
    check(counts["ready"] == 0 and len(queue.rows) == 1,
          "drain leaves results of shots before their deadline untouched")

    # Shot waiting for FULL rescue is never drained.
    event.decision_deadline_monotonic = 1.0
    rescue_router_v2225.reset()
    rescue_router_v2225.request(71)
    try:
        counts = _drain_completed_before_timeout(FakeScanner(event), queue, consume)
        check(counts["ready"] == 0 and len(queue.rows) == 1,
              "drain never consumes a shot waiting for FULL rescue")
    finally:
        rescue_router_v2225.reset()

    # Real detector queue: draining a due shot must not discard another pending shot's result.
    detector = AsyncDetectorV2224.__new__(AsyncDetectorV2224)
    import threading
    from collections import deque
    detector._lock = threading.Lock()
    detector._ready = deque()
    detector._last_applied_frame_ts = {}
    detector.harvest = lambda: None
    due = SimpleNamespace(shot_id=81, peak_ts=10.0, state="pending", lifecycle_state="PROCESSING",
                          decision_deadline_monotonic=2.0, note="")
    other = SimpleNamespace(shot_id=82, peak_ts=10.3, state="pending", lifecycle_state="PROCESSING",
                            decision_deadline_monotonic=future, note="")
    for sid, ts in ((81, 10.1), (82, 10.4)):
        item = SimpleNamespace(shot_id=sid, frame_ts=ts, finished_mono=1.0, candidates=[])
        detector._ready.append(item)
    scanner = FakeScanner(due)
    scanner.audio_events = [due, other]
    detector.apply_result = lambda _s, _r: None
    counts = _drain_completed_before_timeout(scanner, detector, consume)
    check(counts["integrated"] == 1 and [int(i.shot_id) for i in detector._ready] == [82],
          "drain of a due shot keeps the other pending shot's result queued")


def test_min_post_age_gate():
    import os as _os
    from src.engine.shot_async_v2224 import _min_post_age_s
    with patch.dict(_os.environ, {}, clear=False):
        _os.environ.pop("SKJUTBANA_MIN_POST_AGE_S", None)
        check(_min_post_age_s() == 0.0, "causal post-age gate is off by default")
    for raw, expected in (("0.11", 0.11), ("abc", 0.0), ("-1", 0.0), ("5", 0.0)):
        with patch.dict(_os.environ, {"SKJUTBANA_MIN_POST_AGE_S": raw}):
            check(_min_post_age_s() == expected, f"post-age gate parses {raw!r} as {expected}")
    detector = AsyncDetectorV2224.__new__(AsyncDetectorV2224)
    detector._shutdown = False
    detector._sync_detect = lambda *a: []
    event = SimpleNamespace(shot_id=5, peak_ts=100.0, lifecycle_state="PENDING", audio_trigger={})
    detector._target_event = lambda scanner, frame_ts: event
    submitted = []
    detector._pending_for_shot = lambda sid: 0
    detector._max_queued_per_shot = 2
    detector._frame_spacing_s = 0.0
    detector._submitted_frame_ts = {}
    detector._clone_scanner = lambda scanner, ev: scanner
    detector._executor = SimpleNamespace(submit=lambda *a, **k: submitted.append(a[3]) or object())
    import threading
    detector._lock = threading.Lock()
    detector._futures = {}
    scanner = SimpleNamespace(audio_events=[event])
    with patch.dict(_os.environ, {"SKJUTBANA_MIN_POST_AGE_S": "0.11"}):
        detector.submit_if_needed(scanner, np.zeros((2, 2), np.uint8), 100.07)
        check(submitted == [], "frame received before peak+min_post_age is not submitted")
        detector.submit_if_needed(scanner, np.zeros((2, 2), np.uint8), 100.12)
        check(submitted == [100.12], "first causally possible frame is submitted")


def test_novelty_and_causal_ownership_gates():
    import os as _os
    from src.engine.camera.hit_scanner_v2221 import _best_novel_track_for_event
    from src.engine.shot_fast_v2225 import _causally_confirmable, _stamp_origin
    static = SimpleNamespace(track_id=1, last_candidate={"pre_shot_change": 0.2})
    fresh = SimpleNamespace(track_id=2, last_candidate={"v2225_confirm_darkening": 12.0})
    unmeasured = SimpleNamespace(track_id=3, last_candidate={})
    seen = []
    scanner = SimpleNamespace(_active_tracks={1: static, 2: fresh, 3: unmeasured})
    scanner._best_track_for_event = lambda event: seen.append(sorted(scanner._active_tracks)) or None
    original = scanner._active_tracks
    with patch.dict(_os.environ, {"SKJUTBANA_REQUIRE_NOVELTY": "3"}):
        _best_novel_track_for_event(scanner, object())
    check(seen[-1] == [2] and scanner._active_tracks is original and sorted(original) == [1, 2, 3],
          "novelty gate selects over changed tracks only and restores the live mapping")
    with patch.dict(_os.environ, {}, clear=False):
        _os.environ.pop("SKJUTBANA_REQUIRE_NOVELTY", None)
        _best_novel_track_for_event(scanner, object())
    check(seen[-1] == [1, 2, 3], "novelty gate is off by default")

    state = SimpleNamespace(shot_id=1, candidates=[
        _stamp_origin({"camera_x": 1, "camera_y": 1, "pre_shot_change": 9.0}, 10.15),
        _stamp_origin({"camera_x": 5, "camera_y": 5, "pre_shot_change": 0.5}, 10.15)])
    sc = SimpleNamespace(audio_events=[SimpleNamespace(shot_id=1, peak_ts=10.0), SimpleNamespace(shot_id=2, peak_ts=10.2)])
    with patch.dict(_os.environ, {"SKJUTBANA_CAUSAL_OWNERSHIP": "1", "SKJUTBANA_MIN_POST_AGE_S": "0.11"}):
        check(len(_causally_confirmable(sc, state, 10.25)) == 2, "before the newer shot's causal boundary all proposals confirm")
        kept = _causally_confirmable(sc, state, 10.40)
        check([c["camera_x"] for c in kept] == [1], "after the boundary only proposals with originating change confirm")
    _os.environ.pop("SKJUTBANA_CAUSAL_OWNERSHIP", None)
    check(len(_causally_confirmable(sc, state, 10.40)) == 2, "causal ownership is off by default")


def test_pre_average_gate():
    import os as _os
    from src.engine.shot_critical_v2223 import _averaged_pre_v2223
    frames = [SimpleNamespace(timestamp=9.0 + 0.04 * i, frame_bgr=np.full((2, 2, 3), 10 * i, np.uint8)) for i in range(10)]
    best = frames[6]                      # t = 9.24
    scanner = SimpleNamespace(audio_events=[SimpleNamespace(peak_ts=9.05)])   # previous shot
    _os.environ.pop("SKJUTBANA_PRE_AVERAGE_FRAMES", None)
    check(_averaged_pre_v2223(scanner, frames, best, 9.6) is None, "PRE averaging is off by default")
    with patch.dict(_os.environ, {"SKJUTBANA_PRE_AVERAGE_FRAMES": "8", "SKJUTBANA_MIN_POST_AGE_S": "0.11"}):
        out = _averaged_pre_v2223(scanner, frames, best, 9.6)
    # eligible: t >= 9.16 (prev 9.05 + 0.11) and <= 9.24 -> frames 4,5,6 (values 40,50,60)
    check(out is not None and int(out[0, 0]) == 50 and scanner._pre_shot_average_frames == 3,
          "PRE average ends at the selected frame and never predates the previous shot's hole")


def test_local_contrast_preference():
    import os as _os
    from src.engine.camera.hit_scanner_v2221 import _best_novel_track_for_event
    mk = lambda tid, sid, dark, ring: SimpleNamespace(track_id=tid, last_candidate={
        "v2224_producer_shot_id": sid, "v2225_confirm_darkening": dark, "v2225_confirm_ring_abs": ring})
    texture = mk(1, 7, 6.0, 5.8)        # big change incl. surroundings
    hole = mk(2, 7, 12.0, 3.0)          # compact local darkening
    other_shot = mk(3, 8, 30.0, 0.0)    # belongs to another shot: must not set the bar
    legacy = SimpleNamespace(track_id=4, last_candidate={"score": 9.0})
    seen = []
    scanner = SimpleNamespace(_active_tracks={1: texture, 2: hole, 3: other_shot, 4: legacy})
    scanner._best_track_for_event = lambda event: seen.append(sorted(scanner._active_tracks)) or None
    original = scanner._active_tracks
    with patch.dict(_os.environ, {"SKJUTBANA_PREFER_LOCAL_CONTRAST": "3"}):
        _best_novel_track_for_event(scanner, SimpleNamespace(shot_id=7))
        check(seen[-1] == [2] and scanner._active_tracks is original,
              "local-contrast preference keeps only this shot's compact darkening")
        _best_novel_track_for_event(scanner, SimpleNamespace(shot_id=9))
        check(seen[-1] == [1, 2, 3, 4], "without own measured tracks the installed selector is unchanged")


def main():
    test_deadline_drain()
    test_min_post_age_gate()
    test_novelty_and_causal_ownership_gates()
    test_pre_average_gate()
    test_local_contrast_preference()
    test_deadline_drain_respects_deadline_rescue_and_other_shots()
    test_audio_queue_timing()
    test_lifecycle_and_pre_provenance()
    test_executor_submit_failure_is_contained()
    test_executor_submit_failure_then_recovery_is_consumed_once()
    test_geometry_provenance_fields()
    test_batched_pose_provenance()
    test_camera_age_and_emit_parity()
    test_deadline_drain_is_opt_in()
    print("All shot-critical reliability selftests passed.")


if __name__ == "__main__":
    main()
