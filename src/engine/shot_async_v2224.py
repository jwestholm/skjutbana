"""V2.22.4 asynchronous shot pipeline.

Goals
-----
* never run the ~0.9 s physical detector on the Pygame/main thread;
* keep V2.22.3 audio acknowledgement as the first engine decision;
* keep ordinary rendering/event handling alive while a physical shot is being
  analysed (scene *simulation* may stay frozen so the projected image is stable);
* in off/train_only/advisory, remove AIRuntime candidate/evidence work from the
  synchronous HitScanner emission path and perform it on a dedicated shadow
  worker instead;
* profile the expensive V2 detector stages without changing their algorithms.

Authority is intentionally conservative.  The global HitScanner remains the hit
authority.  AI authority modes keep the historical synchronous AI semantics;
only off/train_only/advisory are moved off the critical path in this version.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import copy
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

SCHEMA_VERSION = "2.22.4"
PATCH_REVISION = "r2"
_INSTALLED = False

_PROFILE_TLS = threading.local()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _runtime_settings() -> dict[str, Any]:
    try:
        from src.engine.ai.runtime import get_ai_runtime
        settings = getattr(get_ai_runtime(), "settings", {})
        return settings if isinstance(settings, dict) else {}
    except Exception:
        return {}


def _setting_bool(name: str, default: bool) -> bool:
    return bool(_runtime_settings().get(name, default))


def _setting_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(_runtime_settings().get(name, default))
    except Exception:
        value = int(default)
    return max(lo, min(hi, value))


def _setting_float(name: str, default: float, lo: float, hi: float) -> float:
    value = _finite(_runtime_settings().get(name, default), default)
    return max(lo, min(hi, value))


@dataclass
class DetectorJobResultV2224:
    shot_id: int
    frame_ts: float
    peak_ts: float
    submitted_mono: float
    started_mono: float
    finished_mono: float
    candidates: list[dict[str, Any]]
    debug_frames: dict[str, Any]
    window_debug: dict[str, Any]
    threshold: float
    change_threshold: float
    vote_threshold: float
    stages_ms: dict[str, float] = field(default_factory=dict)
    error: str = ""

    @property
    def queue_ms(self) -> float:
        return max(0.0, (self.started_mono - self.submitted_mono) * 1000.0)

    @property
    def worker_ms(self) -> float:
        return max(0.0, (self.finished_mono - self.started_mono) * 1000.0)


class DetectorStageProfilerV2224:
    """Low-risk timing wrappers around CandidateGeneratorV2 stages.

    The profiler is thread-local and is only active inside V2.22.4 detector
    jobs.  It records wall time; no detector output is changed.
    """

    METHOD_NAMES = (
        "generate",
        "_collect_pre_frames",
        "_build_reference_and_noise",
        "_register_current",
        "_extract_candidates",
        "_update_candidate_bank",
        "_merge_hybrid",
    )

    @classmethod
    def install(cls) -> None:
        try:
            from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
        except Exception:
            return
        if getattr(CandidateGeneratorV2, "_v2224_stage_profiler", False):
            return

        for name in cls.METHOD_NAMES:
            original = getattr(CandidateGeneratorV2, name, None)
            if not callable(original):
                continue
            marker = f"_v2224_profile_original_{name}"
            if hasattr(CandidateGeneratorV2, marker):
                continue
            setattr(CandidateGeneratorV2, marker, original)

            def make_wrapper(fn: Callable[..., Any], stage_name: str):
                def wrapped(self, *args, **kwargs):
                    active = getattr(_PROFILE_TLS, "stages", None)
                    if active is None:
                        return fn(self, *args, **kwargs)
                    t0 = time.perf_counter()
                    try:
                        return fn(self, *args, **kwargs)
                    finally:
                        elapsed = (time.perf_counter() - t0) * 1000.0
                        active[stage_name] = float(active.get(stage_name, 0.0)) + elapsed
                wrapped.__name__ = getattr(fn, "__name__", stage_name)
                wrapped.__doc__ = getattr(fn, "__doc__", None)
                return wrapped

            setattr(CandidateGeneratorV2, name, make_wrapper(original, name))

        CandidateGeneratorV2._v2224_stage_profiler = True


class AsyncDetectorV2224:
    """Single-owner detector worker.

    CandidateGeneratorV2 keeps per-shot persistence/bank state in one engine
    instance.  V2.22.4 therefore deliberately uses ONE CV worker by default;
    parallel calls into that mutable engine would trade latency for races.
    The win here is main-thread responsiveness and overlap with rendering/AI,
    not speculative concurrent mutation of the detector state.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shot-cv-v2224")
        self._futures: dict[Future, tuple[int, float]] = {}
        self._ready: deque[DetectorJobResultV2224] = deque()
        self._submitted_frame_ts: dict[int, float] = {}
        self._last_applied_frame_ts: dict[int, float] = {}
        self._sync_detect: Callable[..., Any] | None = None
        self._shutdown = False
        # Submission-path configuration is cached at install/startup.  Do not
        # perform lazy AIRuntime imports or settings lookups on the first PANG.
        self._max_queued_per_shot = 1
        self._frame_spacing_s = 0.055

    def set_sync_detector(self, fn: Callable[..., Any]) -> None:
        self._sync_detect = fn

    def configure(self, *, max_queued_per_shot: int = 1, frame_spacing_ms: float = 55.0) -> None:
        self._max_queued_per_shot = max(1, min(4, int(max_queued_per_shot)))
        self._frame_spacing_s = max(0.010, min(0.500, float(frame_spacing_ms) / 1000.0))

    def configure_from_runtime(self) -> None:
        self.configure(
            max_queued_per_shot=_setting_int("async_detector_max_queued_per_shot_v2224", 1, 1, 4),
            frame_spacing_ms=_setting_float("async_detector_frame_spacing_ms_v2224", 55.0, 10.0, 500.0),
        )

    def warmup(self, timeout_s: float = 1.0) -> None:
        """Start the executor thread during application startup, not on PANG."""
        if self._shutdown:
            return
        future = self._executor.submit(lambda: None)
        future.result(timeout=max(0.05, float(timeout_s)))

    def wait_for_idle(self, timeout_s: float = 3.0) -> None:
        deadline = time.perf_counter() + max(0.0, float(timeout_s))
        with self._lock:
            futures = list(self._futures)
        for future in futures:
            remaining = max(0.0, deadline - time.perf_counter())
            if remaining <= 0.0:
                break
            try:
                future.result(timeout=remaining)
            except Exception:
                pass
        self.harvest()

    def reset(self) -> None:
        # CandidateGeneratorV2 reset/HitScanner.disable must not race a live
        # detector call. Scene changes may therefore wait for the current CV
        # job, while normal gameplay never does.
        self.wait_for_idle(timeout_s=3.0)
        with self._lock:
            self._ready.clear()
            self._futures.clear()
            self._submitted_frame_ts.clear()
            self._last_applied_frame_ts.clear()

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=True)
        except Exception:
            pass

    def _pending_for_shot(self, shot_id: int) -> int:
        with self._lock:
            return sum(1 for _future, (sid, _ts) in self._futures.items() if sid == int(shot_id))

    @staticmethod
    def _target_event(scanner: Any, frame_ts: float):
        pending = [ev for ev in list(getattr(scanner, "audio_events", []) or []) if str(getattr(ev, "state", "")) == "pending"]
        if not pending:
            return None
        eligible = [ev for ev in pending if float(getattr(ev, "peak_ts", frame_ts)) <= float(frame_ts) + 0.02]
        pool = eligible if eligible else pending
        return min(pool, key=lambda ev: abs(float(frame_ts) - float(getattr(ev, "peak_ts", frame_ts))))

    @staticmethod
    def _clone_scanner(scanner: Any, event: Any) -> Any:
        clone = copy.copy(scanner)
        # Detection reads these values but must not mutate live containers.
        clone.audio_events = deque([copy.copy(event)], maxlen=getattr(scanner.audio_events, "maxlen", 128))
        clone.frame_history = deque(list(getattr(scanner, "frame_history", []) or []), maxlen=getattr(scanner.frame_history, "maxlen", 360))
        clone.known_holes = [dict(h) for h in list(getattr(scanner, "known_holes", []) or [])]
        clone.debug_frames = {}
        clone.last_candidates = []
        clone.last_window_debug = dict(getattr(scanner, "last_window_debug", {}) or {})
        clone.last_stable_tracks = []
        clone._active_tracks = {}
        clone._v2224_worker_clone = True
        return clone

    def submit_if_needed(self, scanner: Any, gray: np.ndarray, frame_ts: float) -> None:
        if self._shutdown or self._sync_detect is None:
            return
        event = self._target_event(scanner, frame_ts)
        if event is None:
            return
        sid = int(getattr(event, "shot_id", 0) or 0)
        if sid <= 0:
            return

        max_queued = self._max_queued_per_shot
        spacing_s = self._frame_spacing_s
        last_ts = float(self._submitted_frame_ts.get(sid, 0.0) or 0.0)
        if last_ts > 0.0 and float(frame_ts) - last_ts < spacing_s:
            return
        if self._pending_for_shot(sid) >= max_queued:
            return

        clone = self._clone_scanner(scanner, event)
        gray_job = np.asarray(gray).copy()  # immutable job ownership
        submitted = time.perf_counter()
        peak_ts = float(getattr(event, "peak_ts", frame_ts))
        self._submitted_frame_ts[sid] = float(frame_ts)
        future = self._executor.submit(
            self._run_job,
            clone,
            gray_job,
            float(frame_ts),
            sid,
            peak_ts,
            submitted,
        )
        with self._lock:
            self._futures[future] = (sid, float(frame_ts))

    def _run_job(
        self,
        clone: Any,
        gray: np.ndarray,
        frame_ts: float,
        shot_id: int,
        peak_ts: float,
        submitted_mono: float,
    ) -> DetectorJobResultV2224:
        started = time.perf_counter()
        stages: dict[str, float] = {}
        _PROFILE_TLS.stages = stages
        error = ""
        candidates: list[dict[str, Any]] = []
        try:
            candidates = [dict(c) for c in list(self._sync_detect(clone, gray, frame_ts) or [])]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            _PROFILE_TLS.stages = None
        finished = time.perf_counter()
        return DetectorJobResultV2224(
            shot_id=int(shot_id),
            frame_ts=float(frame_ts),
            peak_ts=float(peak_ts),
            submitted_mono=float(submitted_mono),
            started_mono=float(started),
            finished_mono=float(finished),
            candidates=candidates,
            debug_frames=dict(getattr(clone, "debug_frames", {}) or {}),
            window_debug=dict(getattr(clone, "last_window_debug", {}) or {}),
            threshold=_finite(getattr(clone, "last_threshold_value", 0.0)),
            change_threshold=_finite(getattr(clone, "last_change_threshold_value", 0.0)),
            vote_threshold=_finite(getattr(clone, "last_vote_threshold_value", 0.0)),
            stages_ms=dict(stages),
            error=error,
        )

    def harvest(self) -> None:
        done: list[Future] = []
        with self._lock:
            for future in list(self._futures):
                if future.done():
                    done.append(future)
        for future in done:
            try:
                result = future.result()
            except Exception as exc:
                sid, frame_ts = self._futures.get(future, (0, 0.0))
                now = time.perf_counter()
                result = DetectorJobResultV2224(
                    shot_id=int(sid), frame_ts=float(frame_ts), peak_ts=0.0,
                    submitted_mono=now, started_mono=now, finished_mono=now,
                    candidates=[], debug_frames={}, window_debug={},
                    threshold=0.0, change_threshold=0.0, vote_threshold=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            with self._lock:
                self._futures.pop(future, None)
                self._ready.append(result)

    def take_ready_for_shot(self, shot_id: int) -> list[DetectorJobResultV2224]:
        self.harvest()
        selected: list[DetectorJobResultV2224] = []
        keep: deque[DetectorJobResultV2224] = deque()
        with self._lock:
            while self._ready:
                item = self._ready.popleft()
                if int(item.shot_id) == int(shot_id):
                    selected.append(item)
                else:
                    keep.append(item)
            self._ready = keep
        selected.sort(key=lambda item: item.frame_ts)
        last_applied = float(self._last_applied_frame_ts.get(int(shot_id), 0.0) or 0.0)
        selected = [item for item in selected if item.frame_ts > last_applied + 1e-6]
        if selected:
            self._last_applied_frame_ts[int(shot_id)] = selected[-1].frame_ts
        return selected

    def take_ready_for_active(self, active_shot_ids: set[int]) -> list[DetectorJobResultV2224]:
        """Return completed results for *all* currently pending shots.

        A newer PANG may arrive while shot N is still on the worker. Once shot
        N+1 exists, selecting only the event nearest the current frame would
        strand N's completed result forever. Harvest all active shot ids and
        integrate them in camera timestamp order instead. Results for shots that
        are no longer pending are discarded.
        """
        self.harvest()
        active = {int(sid) for sid in active_shot_ids if int(sid) > 0}
        selected: list[DetectorJobResultV2224] = []
        with self._lock:
            while self._ready:
                item = self._ready.popleft()
                sid = int(item.shot_id)
                if sid not in active:
                    continue
                last_applied = float(self._last_applied_frame_ts.get(sid, 0.0) or 0.0)
                if item.frame_ts <= last_applied + 1e-6:
                    continue
                selected.append(item)
            selected.sort(key=lambda item: item.frame_ts)
            for item in selected:
                self._last_applied_frame_ts[int(item.shot_id)] = float(item.frame_ts)
        return selected

    def has_pending(self, shot_id: int | None = None) -> bool:
        self.harvest()
        with self._lock:
            if shot_id is None:
                return bool(self._futures)
            return any(sid == int(shot_id) for sid, _ts in self._futures.values())

    @staticmethod
    def apply_result(scanner: Any, result: DetectorJobResultV2224) -> None:
        scanner.last_candidates = [dict(c) for c in result.candidates]
        if result.debug_frames:
            # Preserve reference/debug entries created elsewhere, update only
            # maps produced by this detector job.
            merged = dict(getattr(scanner, "debug_frames", {}) or {})
            merged.update(result.debug_frames)
            scanner.debug_frames = merged
        scanner.last_window_debug = dict(result.window_debug)
        scanner.last_threshold_value = float(result.threshold)
        scanner.last_change_threshold_value = float(result.change_threshold)
        scanner.last_vote_threshold_value = float(result.vote_threshold)
        scanner._v2224_result_frame_ts = float(result.frame_ts)

        stages = result.stages_ms
        v2_ms = _finite(stages.get("generate", 0.0))
        other_ms = max(0.0, result.worker_ms - v2_ms)
        if _setting_bool("async_detector_log_v2224", True):
            parts = [
                f"[V2.22.4 CV] shot={result.shot_id}",
                f"age={(result.frame_ts - result.peak_ts) * 1000.0:.0f}ms",
                f"queue={result.queue_ms:.1f}ms",
                f"worker={result.worker_ms:.1f}ms",
                f"v2={v2_ms:.1f}ms",
                f"other={other_ms:.1f}ms",
                f"cand={len(result.candidates)}",
            ]
            if stages:
                for key, label in (
                    ("_collect_pre_frames", "pre"),
                    ("_build_reference_and_noise", "ref"),
                    ("_register_current", "reg"),
                    ("_extract_candidates", "extract"),
                    ("_update_candidate_bank", "bank"),
                    ("_merge_hybrid", "merge"),
                ):
                    if key in stages:
                        parts.append(f"{label}={stages[key]:.1f}ms")
            if result.error:
                parts.append(f"ERROR={result.error}")
            print(" ".join(parts))


async_detector_v2224 = AsyncDetectorV2224()


@dataclass
class AIShadowTaskV2224:
    shot_id: int
    scanner_snapshot: Any
    event_snapshot: Any
    final_state: str
    default_x: float
    default_y: float
    queued_mono: float
    key: tuple[Any, ...]


class AIShadowWorkerV2224:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-shadow-v2224")
        self._lock = threading.RLock()
        self._keys: set[tuple[Any, ...]] = set()
        self._original_observe = None
        self._original_choose = None
        self._original_mark = None
        self._finish_states: dict[int, str] = {}
        self._finish_enqueued: set[int] = set()
        self._shutdown = False

    def configure(self, observe, choose, mark) -> None:
        self._original_observe = observe
        self._original_choose = choose
        self._original_mark = mark

    def warmup(self, timeout_s: float = 1.0) -> None:
        """Start the shadow executor during application startup."""
        if self._shutdown:
            return
        future = self._executor.submit(lambda: None)
        future.result(timeout=max(0.05, float(timeout_s)))

    def record_finish(self, runtime: Any, shot_id: int, state: str) -> None:
        sid = int(shot_id or 0)
        if sid <= 0:
            return
        with self._lock:
            self._finish_states[sid] = str(state or "finished")
            if sid in self._finish_enqueued or self._shutdown:
                return
            self._finish_enqueued.add(sid)
        # The emission hook calls observe() before mark().  Submitting this to
        # the same one-worker executor therefore gives FIFO ordering: build the
        # shot context/evidence first, then apply its terminal state.  It also
        # closes the race where a very fast shadow observation could finish just
        # before the main thread records the HIT state.
        self._executor.submit(self._apply_finish, runtime, sid)

    def _apply_finish(self, runtime: Any, shot_id: int) -> None:
        try:
            with self._lock:
                state = self._finish_states.pop(int(shot_id), "finished")
            if self._original_mark is not None:
                self._original_mark(runtime, int(shot_id), state)
        except Exception as exc:
            if _setting_bool("async_ai_shadow_log_v2224", True):
                print(f"[V2.22.4 AI-SHADOW] shot={shot_id} finish_error={type(exc).__name__}: {exc}")
        finally:
            with self._lock:
                self._finish_enqueued.discard(int(shot_id))

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            self._executor.shutdown(wait=True)
        except Exception:
            pass

    @staticmethod
    def _target_event(scanner: Any, explicit_event: Any = None):
        if explicit_event is not None:
            return explicit_event
        pending = [ev for ev in list(getattr(scanner, "audio_events", []) or []) if str(getattr(ev, "state", "")) == "pending"]
        if not pending:
            return None
        return max(pending, key=lambda ev: (float(getattr(ev, "peak_ts", 0.0)), int(getattr(ev, "shot_id", 0))))

    @staticmethod
    def _default_xy(scanner: Any, event: Any) -> tuple[float, float]:
        matched = int(getattr(event, "matched_track_id", 0) or 0)
        tracks = getattr(scanner, "_active_tracks", {}) or {}
        track = tracks.get(matched) if matched else None
        if track is not None:
            return _finite(getattr(track, "camera_x", 0.0)), _finite(getattr(track, "camera_y", 0.0))
        best = getattr(scanner, "last_best_candidate", None)
        if isinstance(best, Mapping):
            return _finite(best.get("camera_x", 0.0)), _finite(best.get("camera_y", 0.0))
        candidates = list(getattr(scanner, "last_candidates", []) or [])
        if candidates:
            return _finite(candidates[0].get("camera_x", 0.0)), _finite(candidates[0].get("camera_y", 0.0))
        return 0.0, 0.0

    @staticmethod
    def _snapshot(scanner: Any, event: Any) -> tuple[Any, Any, str]:
        snap = copy.copy(scanner)
        ev = copy.copy(event)
        final_state = str(getattr(ev, "state", "pending") or "pending")
        ev.state = "pending"
        snap.audio_events = deque([ev], maxlen=getattr(scanner.audio_events, "maxlen", 128))
        snap.last_candidates = [dict(c) for c in list(getattr(scanner, "last_candidates", []) or [])]
        snap.debug_frames = dict(getattr(scanner, "debug_frames", {}) or {})
        snap.frame_history = deque(list(getattr(scanner, "frame_history", []) or []), maxlen=getattr(scanner.frame_history, "maxlen", 360))
        snap.known_holes = [dict(h) for h in list(getattr(scanner, "known_holes", []) or [])]
        snap._active_tracks = {k: copy.copy(v) for k, v in dict(getattr(scanner, "_active_tracks", {}) or {}).items()}
        return snap, ev, final_state

    def submit(self, runtime: Any, scanner: Any, explicit_event: Any = None) -> None:
        if self._shutdown or self._original_observe is None:
            return
        event = self._target_event(scanner, explicit_event)
        if event is None:
            return
        sid = int(getattr(event, "shot_id", 0) or 0)
        if sid <= 0:
            return
        candidates_now = list(getattr(scanner, "last_candidates", []) or [])
        # The historical bootstrap calls observe_scanner() after *every* scanner
        # update. While async CV is still pending there is nothing useful for AI
        # to analyse; queueing an empty 4K snapshot every render frame would be
        # catastrophic. Explicit emission observation is always accepted.
        if explicit_event is None and not candidates_now:
            return
        frame_ts = _finite(getattr(scanner, "_last_frame_ts", 0.0), 0.0)
        candidate_timestamps = [
            _finite(c.get("timestamp", 0.0), 0.0) for c in candidates_now
            if _finite(c.get("timestamp", 0.0), 0.0) > 0.0
        ]
        cand_ts = max(candidate_timestamps, default=_finite(getattr(scanner, "_v2224_result_frame_ts", 0.0), 0.0))
        if candidates_now:
            top = max(candidates_now, key=lambda c: _finite(c.get("score", 0.0), 0.0))
            fingerprint = (
                round(_finite(top.get("camera_x", 0.0)), 1),
                round(_finite(top.get("camera_y", 0.0)), 1),
                round(_finite(top.get("score", 0.0)), 2),
            )
        else:
            fingerprint = (0.0, 0.0, 0.0)
        key = (sid, round(cand_ts, 4), len(candidates_now), fingerprint)
        with self._lock:
            if key in self._keys:
                return
            max_keys = _setting_int("async_ai_shadow_history_v2224", 64, 8, 512)
            self._keys.add(key)
            if len(self._keys) > max_keys:
                # Set ordering is irrelevant; only dedupe recent work.
                self._keys = set(list(self._keys)[-max_keys:])

        snap, ev_snap, final_state = self._snapshot(scanner, event)
        default_x, default_y = self._default_xy(scanner, event)
        task = AIShadowTaskV2224(
            shot_id=sid,
            scanner_snapshot=snap,
            event_snapshot=ev_snap,
            final_state=final_state,
            default_x=default_x,
            default_y=default_y,
            queued_mono=time.perf_counter(),
            key=key,
        )
        self._executor.submit(self._run, runtime, task)

    def _run(self, runtime: Any, task: AIShadowTaskV2224) -> None:
        started = time.perf_counter()
        error = ""
        try:
            self._original_observe(runtime, task.scanner_snapshot, event=task.event_snapshot)
            mode = str(getattr(runtime, "settings", {}).get("mode", "train_only") or "train_only").lower()
            if mode == "advisory" and self._original_choose is not None:
                self._original_choose(runtime, task.default_x, task.default_y, shot_id=task.shot_id)
            # A non-pending snapshot can occur in post-update shadow capture;
            # honour it as a fallback.  Normal emission terminal state is queued
            # by patched_mark() behind this observation on the same executor.
            if task.final_state != "pending" and self._original_mark is not None:
                self._original_mark(runtime, task.shot_id, task.final_state)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter() - started) * 1000.0
        queue_ms = max(0.0, (started - task.queued_mono) * 1000.0)
        if _setting_bool("async_ai_shadow_log_v2224", True):
            suffix = f" error={error}" if error else ""
            print(f"[V2.22.4 AI-SHADOW] shot={task.shot_id} queue={queue_ms:.1f}ms compute={elapsed:.1f}ms{suffix}")


ai_shadow_worker_v2224 = AIShadowWorkerV2224()


def _install_async_detector_patch() -> None:
    from importlib import import_module
    hs_module = import_module("src.engine.camera.hit_scanner")
    HitScanner = hs_module.HitScanner
    if getattr(HitScanner, "_v2224_async_detector_patch", False):
        return

    sync_detect = HitScanner._detect_frame_candidates
    original_update_tracks = HitScanner._update_tracks
    original_disable = HitScanner.disable
    async_detector_v2224.set_sync_detector(sync_detect)

    def patched_detect(self, gray: np.ndarray, frame_ts: float):
        if not _setting_bool("async_detector_enabled_v2224", True):
            return sync_detect(self, gray, frame_ts)

        pending_events = [
            ev for ev in list(getattr(self, "audio_events", []) or [])
            if str(getattr(ev, "state", "")) == "pending"
        ]
        if not pending_events:
            return sync_detect(self, gray, frame_ts)
        active_ids = {int(getattr(ev, "shot_id", 0) or 0) for ev in pending_events}
        event = async_detector_v2224._target_event(self, frame_ts)
        if event is None:
            return sync_detect(self, gray, frame_ts)

        # Harvest ALL pending shot ids. A second PANG can arrive while the first
        # job is running; shot 1's result must still be integrated even though
        # the current frame is temporally nearer to shot 2.
        ready = async_detector_v2224.take_ready_for_active(active_ids)

        if not ready:
            # Only queue work when no completed verdict is waiting. In
            # particular, do not pre-queue a second expensive frame before the
            # first result has had a chance to confirm/emit the shot. This
            # prevents shot N from monopolising the single worker when a new
            # audio peak N+1 arrives.
            async_detector_v2224.submit_if_needed(self, gray, frame_ts)
            self._v2224_async_waiting = True
            self._v2224_result_frame_ts = 0.0
            return []

        # If more than one sequential detector result became ready while the
        # main thread rendered, apply older frames directly, then let the normal
        # HitScanner update apply the newest one. This preserves ordinary track
        # confirmation semantics without blocking the game loop.
        for result in ready[:-1]:
            async_detector_v2224.apply_result(self, result)
            original_update_tracks(self, result.candidates, result.frame_ts)

        result = ready[-1]
        async_detector_v2224.apply_result(self, result)
        self._v2224_async_waiting = False
        return result.candidates

    def patched_update_tracks(self, candidates, frame_ts: float):
        if not candidates and bool(getattr(self, "_v2224_async_waiting", False)):
            # No detector verdict yet is not a negative camera frame. Do not age
            # real tracks merely because the worker is busy.
            return None
        override_ts = _finite(getattr(self, "_v2224_result_frame_ts", 0.0), 0.0)
        if override_ts > 0.0 and candidates:
            self._v2224_result_frame_ts = 0.0
            return original_update_tracks(self, candidates, override_ts)
        return original_update_tracks(self, candidates, frame_ts)

    def patched_disable(self):
        # Wait only on explicit disable/scene transition, never in the normal
        # shot path. This prevents CandidateGeneratorV2 runtime-state reset from
        # racing the one worker that owns that mutable engine.
        async_detector_v2224.reset()
        try:
            return original_disable(self)
        finally:
            self._v2224_async_waiting = False
            self._v2224_result_frame_ts = 0.0

    HitScanner._detect_frame_candidates = patched_detect
    HitScanner._update_tracks = patched_update_tracks
    HitScanner.disable = patched_disable
    HitScanner._v2224_async_detector_patch = True


def _install_async_ai_shadow_patch() -> None:
    try:
        import src.engine.ai.runtime as runtime_module
        AIRuntime = runtime_module.AIRuntime
    except Exception:
        return
    if getattr(AIRuntime, "_v2224_async_ai_shadow_patch", False):
        return

    original_observe = AIRuntime.observe_scanner
    original_choose = AIRuntime.choose_for_emission
    original_mark = AIRuntime.mark_shot_finished
    ai_shadow_worker_v2224.configure(original_observe, original_choose, original_mark)

    async_modes = {"off", "train_only", "advisory"}

    def _async_mode(runtime) -> bool:
        mode = str(getattr(runtime, "settings", {}).get("mode", "train_only") or "train_only").strip().lower()
        enabled = bool(getattr(runtime, "settings", {}).get("enabled", True))
        return enabled and mode in async_modes and _setting_bool("async_ai_shadow_enabled_v2224", True)

    def patched_observe(self, scanner, event=None):
        if _async_mode(self):
            # Snapshot only; heavy patch extraction/ranking runs elsewhere.
            ai_shadow_worker_v2224.submit(self, scanner, explicit_event=event)
            return None
        # Authority modes retain the historical synchronous behaviour until a
        # bounded-deadline authority protocol is tested on physical holdout.
        return original_observe(self, scanner, event=event)

    def patched_choose(self, default_x: float, default_y: float, shot_id=None):
        if _async_mode(self):
            # Critical-path guarantee: advisory/training may NEVER delay a hit.
            # The shadow worker runs the captured historical chooser later for
            # diagnostics/resolver comparison, while gameplay gets passthrough.
            return {
                "apply": False,
                "camera_x": float(default_x),
                "camera_y": float(default_y),
                "confidence": 0.0,
                "reason": "v2224_async_shadow_passthrough",
                "shot_id": shot_id,
            }
        return original_choose(self, default_x, default_y, shot_id=shot_id)

    def patched_mark(self, shot_id: int, state: str = "finished"):
        if _async_mode(self):
            # Bootstrap calls mark immediately after emission.  Do not race the
            # worker's context construction; record terminal state and let the
            # same worker finish the context after observation/diagnostics.
            ai_shadow_worker_v2224.record_finish(self, shot_id, state)
            return None
        return original_mark(self, shot_id, state)

    AIRuntime.observe_scanner = patched_observe
    AIRuntime.choose_for_emission = patched_choose
    AIRuntime.mark_shot_finished = patched_mark
    AIRuntime._v2224_async_ai_shadow_patch = True

    # Already-created singleton automatically uses class method lookup.


def _install_settings_defaults() -> None:
    try:
        import src.engine.ai.runtime as runtime_module
        defaults = {
            "async_detector_enabled_v2224": True,
            "async_detector_max_queued_per_shot_v2224": 1,
            "async_detector_frame_spacing_ms_v2224": 55.0,
            "async_detector_log_v2224": True,
            "async_ai_shadow_enabled_v2224": True,
            "async_ai_shadow_log_v2224": True,
            "async_ai_shadow_history_v2224": 64,
        }
        runtime_module.DEFAULT_SETTINGS.update(defaults)
        existing = getattr(runtime_module, "_RUNTIME", None)
        if existing is not None:
            for key, value in defaults.items():
                getattr(existing, "settings", {}).setdefault(key, value)
    except Exception:
        pass


def _shutdown_workers() -> None:
    async_detector_v2224.shutdown()
    ai_shadow_worker_v2224.shutdown()


def install_v2224_runtime(AppClass: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    _install_settings_defaults()
    # Pay executor-thread startup and settings/import cost now, while the game
    # is starting.  The first real PANG must only snapshot + enqueue work.
    async_detector_v2224.configure_from_runtime()
    async_detector_v2224.warmup()
    ai_shadow_worker_v2224.warmup()
    DetectorStageProfilerV2224.install()
    _install_async_detector_patch()
    _install_async_ai_shadow_patch()

    if getattr(AppClass, "_v2224_async_shot_patch", False):
        _INSTALLED = True
        return

    def run_v2224(self) -> None:
        import pygame
        from config import FPS
        from src.engine.app import AUTOMATION_EVENT
        from src.engine.audio.audio_peak_detector import audio_peak_detector
        from src.engine.camera.camera_manager import camera_manager
        from src.engine.camera.hit_scanner import hit_scanner
        from src.engine.output.led_service import led_service
        from src.engine.shot_critical_v2223 import shot_critical_controller_v2223 as controller

        controller.last_seen_peak_ts = float(getattr(audio_peak_detector, "last_peak_ts", 0.0) or 0.0)

        try:
            while self.running:
                # A queued PANG remains the first engine decision. CV itself is
                # now a worker job, so the main thread stays available for the
                # next PANG and for rendering/window events.
                # Bypass frame pacing only when an *unacknowledged* audio peak
                # is already waiting. Once a shot is dispatched, keep normal FPS
                # pacing while its CV worker runs; an uncapped render loop would
                # waste a CPU core and can make perception slower without helping
                # audio latency.
                audio_pending = controller.pending_audio(audio_peak_detector)
                dt = self.clock.tick(0 if audio_pending else FPS) / 1000.0

                new_peak = controller.begin_pending_audio(self, audio_peak_detector)
                controller.prepare_object_snapshot(hit_scanner, self.scene, new_peak)

                t0 = time.perf_counter()
                audio_peak_detector.update()
                audio_ms = (time.perf_counter() - t0) * 1000.0
                controller.discover_scanner_events(hit_scanner, self.scene)

                t0 = time.perf_counter()
                camera_manager.update()
                camera_ms = (time.perf_counter() - t0) * 1000.0

                # HitScanner.update is now cheap on the main thread: frame
                # bookkeeping + result integration. _detect_frame_candidates()
                # submits/harvests the real CV work asynchronously.
                t0 = time.perf_counter()
                hit_scanner.update(dt)
                scanner_main_ms = (time.perf_counter() - t0) * 1000.0
                controller.discover_scanner_events(hit_scanner, self.scene)
                controller.note_stage(
                    audio_dispatch_ms=audio_ms if new_peak is not None else 0.0,
                    camera_update_ms=camera_ms,
                    scanner_update_ms=scanner_main_ms,
                )
                controller.enrich_spatial_context(hit_scanner)
                controller.evaluate_object_shadow(hit_scanner)
                controller.update_finished(hit_scanner)

                # Always service OS/Pygame events while CV is running. This is
                # the user-visible difference from V2.22.3: no one-second frozen
                # game thread. Scene *simulation* is still frozen for a real
                # physical shot unless the scene explicitly requires updates,
                # keeping the projected frame stable for camera evidence.
                self._post_automation_events()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.quit()
                        break
                    if event.type == AUTOMATION_EVENT:
                        self._handle_automation_event(event)
                        continue
                    switch = self.scene.handle_event(event)
                    if switch:
                        self._switch_to(switch.new_scene)
                        break

                if not self.running:
                    break

                freeze_simulation = controller.should_defer_scene_work(self.scene, hit_scanner)
                if not freeze_simulation:
                    switch = self.scene.update(dt)
                    if switch:
                        self._switch_to(switch.new_scene)

                self._update_window_caption()
                self.scene.render(self.screen)
                pygame.display.flip()
                controller.mark_visible_after_flip()

        finally:
            _shutdown_workers()
            try:
                self.scene.on_exit()
            except Exception:
                pass
            try:
                hit_scanner.disable()
            except Exception:
                pass
            try:
                led_service.stop()
            except Exception:
                pass
            try:
                audio_peak_detector.stop()
            except Exception:
                pass
            try:
                camera_manager.stop()
            except Exception:
                pass
            try:
                self.communication_server.stop()
            except Exception:
                pass
            pygame.quit()

    AppClass.run = run_v2224
    AppClass._v2224_async_shot_patch = True
    _INSTALLED = True
    print("[V2.22.4-r2] async CV + off-critical AI shadow pipeline installed")


__all__ = [
    "SCHEMA_VERSION",
    "DetectorJobResultV2224",
    "DetectorStageProfilerV2224",
    "AsyncDetectorV2224",
    "AIShadowWorkerV2224",
    "async_detector_v2224",
    "ai_shadow_worker_v2224",
    "install_v2224_runtime",
]
