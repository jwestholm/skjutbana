"""Deterministic offline host for the LIVE shot path (dedicated process only).

Runs the installed runtime (``install_v2223_runtime`` … ``install_v253_runtime``)
and calls the same ``shot_path_step`` the live loop calls, while replacing only
*sources and time*:

* camera frames  -> ``camera_manager.inject_frame`` with virtual receive times
* PCM blocks     -> ``audio_peak_detector._deliver_pcm_block`` with virtual receipt
* clocks         -> one replay-owned virtual clock (wall and monotonic kept apart)
* CV worker      -> a virtual executor honouring the Future contract; a job is
                    ``done()`` only after a modelled compute time on one worker
* AI shadow      -> disabled (diagnostics only); physical trace writing disabled

Never use this inside the game process: it patches ``time`` process-wide.
Timing fields produced under replay are VIRTUAL, never latency measurements.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import threading
import time as _time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

_REAL = dict(time=_time.time, monotonic=_time.monotonic, perf_counter=_time.perf_counter, sleep=_time.sleep)


class VirtualClock:
    """Virtual seconds since start; wall and monotonic are separate domains."""

    def __init__(self, wall0: float = 1_800_000_000.0, mono0: float = 100_000.0) -> None:
        self.t = 0.0
        self.wall0 = float(wall0)
        self.mono0 = float(mono0)
        self._frozen: float | None = None

    def now(self) -> float:
        return self._frozen if self._frozen is not None else self.t

    def wall(self) -> float:
        return self.wall0 + self.now()

    def mono(self) -> float:
        return self.mono0 + self.now()

    def advance_to(self, t: float) -> None:
        if t < self.t - 1e-12:
            raise ValueError(f"Virtual time cannot go backwards ({t} < {self.t})")
        self.t = float(t)

    @contextlib.contextmanager
    def frozen_at(self, t: float):
        previous = self._frozen
        self._frozen = float(t)
        try:
            yield
        finally:
            self._frozen = previous

    @contextlib.contextmanager
    def installed(self):
        """Patch the ``time`` module process-wide for the replay duration."""
        _time.time = self.wall
        _time.monotonic = self.mono
        _time.perf_counter = self.mono
        _time.sleep = lambda s: None  # nothing in the replayed path may block on real time
        try:
            yield self
        finally:
            _time.time = _REAL["time"]
            _time.monotonic = _REAL["monotonic"]
            _time.perf_counter = _REAL["perf_counter"]
            _time.sleep = _REAL["sleep"]


@dataclass
class _Job:
    fn: Callable[..., Any]
    args: tuple
    kwargs: dict
    future: Future
    start: float
    finish: float


class VirtualExecutor:
    """Single-worker executor on virtual time.

    Jobs queue FIFO like a 1-thread pool. Job N starts at max(submit, finish of
    N-1) and finishes ``cost_s`` later. The function runs when virtual time
    reaches its finish, with the clock frozen at its start; returned dataclasses
    with ``finished_mono`` get the modelled finish time.
    """

    def __init__(self, clock: VirtualClock, cost_s: float | Callable[[], float] = 0.15) -> None:
        self.clock = clock
        self.cost_s = cost_s
        self._jobs: list[_Job] = []
        self._worker_free_at = 0.0
        self.executed = 0

    def _cost(self) -> float:
        return float(self.cost_s() if callable(self.cost_s) else self.cost_s)

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        start = max(self.clock.now(), self._worker_free_at)
        finish = start + self._cost()
        self._worker_free_at = finish
        self._jobs.append(_Job(fn, args, kwargs, future, start, finish))
        return future

    def next_finish(self) -> float | None:
        return min((job.finish for job in self._jobs), default=None)

    def run_due(self) -> int:
        """Execute every job whose modelled finish is <= now (FIFO)."""
        ran = 0
        while self._jobs and self._jobs[0].finish <= self.clock.now() + 1e-12:
            job = self._jobs.pop(0)
            if not job.future.set_running_or_notify_cancel():
                continue
            # Live-path fidelity (2026-11-18): several live stages are gated on the CV worker
            # thread name ("shot-cv-v2224": V2.22.5 FAST extract, V2.24.1 object-local, V2.25.1
            # region proposal, V2.25.2 freshness, V2.25.3 novelty).  Running jobs on MainThread
            # silently replayed the FULL/legacy path instead.  SKJUTBANA_REPLAY_LEGACY_THREAD=1
            # restores the old behaviour for comparison with earlier baselines.
            thread = threading.current_thread()
            old_name = thread.name
            if os.environ.get("SKJUTBANA_REPLAY_LEGACY_THREAD") != "1":
                thread.name = "shot-cv-v2224-virtual"
            try:
                with self.clock.frozen_at(job.start):
                    result = job.fn(*job.args, **job.kwargs)
                if dataclasses.is_dataclass(result) and hasattr(result, "finished_mono"):
                    result = dataclasses.replace(result, finished_mono=self.clock.mono0 + job.finish)
                job.future.set_result(result)
            except BaseException as exc:  # mirrors ThreadPoolExecutor semantics
                job.future.set_exception(exc)
            finally:
                thread.name = old_name
            ran += 1
            self.executed += 1
        return ran

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        if cancel_futures:
            for job in self._jobs:
                job.future.cancel()
            self._jobs.clear()


class NullExecutor:
    """Disabled AI-shadow worker: accepts work, never runs it."""

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        future.cancel()
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        return None


class _ReplayScene:
    """Physical SHOT_ONLY-like scene: no regions, no synthetic training flags."""
    auto_training_enabled = False
    single_synth_round_active = False
    synthetic_trigger_pending = False
    synthetic_reveal_pending = False
    shot_critical_updates_required = False
    wants_hit_scanning = True

    def get_hit_regions(self):
        return ()


class _ReplayApp:
    """Stub App class handed to the installers (their App.run patch is unused)."""

    def __init__(self) -> None:
        self.scene = _ReplayScene()
        self.running = True


@dataclass
class Emission:
    camera_x: float
    camera_y: float
    wall: float
    virtual_t: float


@dataclass
class VirtualRuntime:
    """Owns the installed live runtime inside one replay process."""

    clock: VirtualClock = field(default_factory=VirtualClock)
    worker_cost_s: float = 0.15
    step_interval_s: float = 1.0 / 60.0
    emissions: list[Emission] = field(default_factory=list)
    _installed: bool = False

    def install(self) -> None:
        if self._installed:
            return
        os.environ["SKJUTBANA_PHYSICAL_TRACE_DISABLE"] = "1"
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        # Live import order: `from src.engine.app import App` imports the loading
        # scene, whose module import runs apply_bootstrap() (V2.22 resolver,
        # V2.22.1 crop/resolve, V2.22.2 cleanup, AI/trace scanner hook) BEFORE
        # main.py's v2223..v253 installers wrap on top. Mirror that exactly.
        import src.engine.app  # noqa: F401  (same side effects as main.py's import)
        from src.engine.ai.bootstrap import apply_bootstrap
        apply_bootstrap()
        from src.engine.shot_critical_v2223 import install_v2223_runtime
        from src.engine.shot_async_v2224 import install_v2224_runtime, async_detector_v2224, ai_shadow_worker_v2224
        from src.engine.shot_fast_v2225 import install_v2225_runtime
        from src.engine.shot_track_v2226 import install_v2226_runtime
        from src.engine.shot_object_local_v241 import install_v241_runtime
        from src.engine.shot_object_local_v243 import install_v243_runtime
        from src.engine.shot_object_local_v244 import install_v244_runtime
        from src.engine.shot_context_v250 import install_v250_runtime
        from src.engine.shot_region_proposal_v251 import install_v251_runtime
        from src.engine.shot_region_freshness_v252 import install_v252_runtime
        from src.engine.shot_cross_thread_novelty_v253 import install_v253_runtime
        # Same order as main.py.
        for installer in (install_v2223_runtime, install_v2224_runtime, install_v2225_runtime,
                          install_v2226_runtime, install_v241_runtime, install_v243_runtime,
                          install_v244_runtime, install_v250_runtime, install_v251_runtime,
                          install_v252_runtime, install_v253_runtime):
            installer(_ReplayApp)
        # Memory: the live ring holds 300 4K BGR frames (~7.5 GB). Replay needs
        # only a few seconds of history (PRE lookback ~0.35 s, object queries
        # < 1 s), so cap it; several replay processes can then run in parallel.
        from collections import deque
        from src.engine.camera.camera_manager import camera_manager
        ring_frames = int(os.environ.get("SKJUTBANA_REPLAY_RING_FRAMES", "100"))
        with camera_manager._lock:
            camera_manager._ring = deque(camera_manager._ring, maxlen=max(25, ring_frames))
        self.executor = VirtualExecutor(self.clock, self.worker_cost_s)
        async_detector_v2224._executor = self.executor
        ai_shadow_worker_v2224._executor = NullExecutor()

        from src.engine.input.hit_input import hit_input
        original_push = hit_input.push_camera_hit

        def capture_push(x, y, *args, **kwargs):
            self.emissions.append(Emission(float(x), float(y), self.clock.wall(), self.clock.now()))
            return original_push(x, y, *args, **kwargs)

        hit_input.push_camera_hit = capture_push
        self.app = _ReplayApp()
        self._installed = True

    def enable_candidate_export(self) -> list:
        """Log every candidate entering tracking (pass-through wrapper).

        Training/analysis only. The wrapper changes no inputs or outputs; it is
        installed only when exporting, so bench_fidelity_selftest (default
        runtime) still proves live parity.
        """
        from src.engine.camera.hit_scanner import HitScanner
        sink: list = []
        original = HitScanner._update_tracks
        clock = self.clock

        def logging_update_tracks(scanner, candidates, frame_ts):
            if getattr(scanner, "_detector_v2_ground_truth", None) is not None:
                raise RuntimeError("ground truth reached the detector during export (leakage)")
            for c in list(candidates or []):
                sink.append(dict(frame_ts=float(frame_ts), virtual_t=clock.now(), candidate=dict(c)))
            return original(scanner, candidates, frame_ts)

        HitScanner._update_tracks = logging_update_tracks
        self.candidate_log = sink
        return sink

    def start_scanner(self) -> None:
        from src.engine.camera.camera_manager import camera_manager
        from src.engine.camera.hit_scanner import hit_scanner
        camera_manager.running = True  # injected source; no device, no reader thread
        hit_scanner.shot_diag_enabled = False
        hit_scanner.enable()

    def step(self, dt: float | None = None) -> dict[str, Any]:
        from src.engine.shot_async_v2224 import shot_path_step
        self.executor.run_due()
        return shot_path_step(self.app, self.step_interval_s if dt is None else dt)


def use_recorded_settings(main: dict | None, ai: dict | None, directory) -> None:
    """Point settings loaders at copies of a recorded session's settings.

    Must run before runtime modules are imported/installed. Writes only into
    ``directory`` (a replay-owned temp dir); live content/ files are untouched.
    """
    import json
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if main is not None:
        path = directory / "settings.json"
        path.write_text(json.dumps(main, indent=2), encoding="utf-8")
        import config
        config.SETTINGS_PATH = str(path)
    if ai is not None:
        path = directory / "ai_settings.json"
        path.write_text(json.dumps(ai, indent=2), encoding="utf-8")
        import src.engine.ai.settings as ai_settings
        ai_settings.AI_SETTINGS_PATH = path


def make_pcm_receipt(clock: VirtualClock, samples: int, sample_rate: int,
                     delivery_latency_s: float) -> dict[str, Any]:
    """Receipt dict in the exact shape `_read_pcm_block` produces (virtual times)."""
    mono = clock.mono()
    return dict(read_start_monotonic=mono - 1e-4, read_complete_monotonic=mono,
                receive_wall_ts=clock.wall(), read_calls=1, complete_block=True,
                virtual_delivery_latency_s=float(delivery_latency_s))


__all__ = ["VirtualClock", "VirtualExecutor", "NullExecutor", "VirtualRuntime", "Emission", "make_pcm_receipt"]
