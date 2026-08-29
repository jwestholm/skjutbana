"""V2.22.6 frame-unique tracking + audio near-miss telemetry.

Installed by ``main.py`` after V2.22.5.

This patch deliberately does NOT retune the ranker.  It fixes two semantics
exposed by physical V2.22.5 testing:

1. ``HoleTrack.hits`` must mean observations on different camera frames, not
   multiple nearby candidates from the same frame.  Same-frame candidate
   agreement is retained as separate evidence instead of pretending to be
   temporal persistence.
2. A physical sound that almost satisfies AudioPeakDetector must no longer
   disappear silently.  Strong rejected transients are logged with the exact
   rejection gates while trigger behaviour remains unchanged.

The intended normal path after this patch is therefore:

    one global FAST proposal frame -> hits=1
    later LOCAL-CONFIRM camera frame -> hits=2 -> ready after event age gate

rather than one proposal frame producing hits=2..N by itself.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Any

import numpy as np

SCHEMA_VERSION = "2.22.6"
PATCH_REVISION = "r1"
_INSTALLED = False


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
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


def _setting_float(name: str, default: float, lo: float, hi: float) -> float:
    value = _finite(_runtime_settings().get(name, default), default)
    return max(float(lo), min(float(hi), value))


@dataclass
class TrackConfigV2226:
    frame_epsilon_s: float = 1e-5
    track_log: bool = True


@dataclass
class AudioTelemetryConfigV2226:
    enabled: bool = True
    near_ratio: float = 0.65
    min_log_interval_s: float = 0.15
    absolute_floor: float = 0.025
    log_triggers: bool = True


_TRACK_CONFIG = TrackConfigV2226()
_AUDIO_CONFIG = AudioTelemetryConfigV2226()


def _load_config() -> None:
    global _TRACK_CONFIG, _AUDIO_CONFIG
    _TRACK_CONFIG = TrackConfigV2226(
        frame_epsilon_s=_setting_float("track_frame_epsilon_s_v2226", 1e-5, 1e-8, 0.002),
        track_log=_setting_bool("track_frame_unique_log_v2226", True),
    )
    _AUDIO_CONFIG = AudioTelemetryConfigV2226(
        enabled=_setting_bool("audio_near_miss_enabled_v2226", True),
        near_ratio=_setting_float("audio_near_miss_ratio_v2226", 0.65, 0.20, 0.98),
        min_log_interval_s=_setting_float("audio_near_miss_min_interval_s_v2226", 0.15, 0.02, 2.0),
        absolute_floor=_setting_float("audio_near_miss_absolute_floor_v2226", 0.025, 0.001, 0.50),
        log_triggers=_setting_bool("audio_raw_trigger_log_v2226", True),
    )


def _install_settings_defaults() -> None:
    defaults = {
        "track_frame_epsilon_s_v2226": 1e-5,
        "track_frame_unique_log_v2226": True,
        "audio_near_miss_enabled_v2226": True,
        "audio_near_miss_ratio_v2226": 0.65,
        "audio_near_miss_min_interval_s_v2226": 0.15,
        "audio_near_miss_absolute_floor_v2226": 0.025,
        "audio_raw_trigger_log_v2226": True,
    }
    try:
        import src.engine.ai.runtime as runtime_module
        runtime_module.DEFAULT_SETTINGS.update(defaults)
        existing = getattr(runtime_module, "_RUNTIME", None)
        if existing is not None:
            for key, value in defaults.items():
                getattr(existing, "settings", {}).setdefault(key, value)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Frame-unique tracking
# ---------------------------------------------------------------------------


def _track_same_frame_support(track: Any, candidate: dict[str, Any], frame_ts: float) -> None:
    """Record same-frame agreement without incrementing temporal hits."""
    support = int(getattr(track, "v2226_same_frame_support", 1) or 1) + 1
    track.v2226_same_frame_support = support
    track.v2226_last_support_frame_ts = float(frame_ts)
    score = _finite(candidate.get("score", 0.0))
    track.best_score = float(max(_finite(getattr(track, "best_score", 0.0)), score))
    track.missed_frames = 0

    # Keep the best/current authoritative XY; do not average duplicate proposals
    # from one image into a potentially non-physical point between holes.
    last = dict(getattr(track, "last_candidate", {}) or {})
    last["v2226_same_frame_support"] = float(support)
    last["v2226_frame_unique_observation"] = 0.0
    last["v2226_support_score_max"] = float(track.best_score)
    track.last_candidate = last


def update_tracks_frame_unique_v2226(
    scanner: Any,
    candidates: list[dict[str, Any]],
    frame_ts: float,
    *,
    config: TrackConfigV2226 | None = None,
) -> dict[str, float]:
    """Update HitScanner tracks with at most one temporal hit per track/frame.

    Nearby candidates in one image are *same-frame support*.  They may improve
    score/provenance but cannot increment ``HoleTrack.hits``.  A later camera
    frame is required for another temporal observation.
    """
    from src.engine.camera.hit_scanner import HoleTrack

    cfg = config or _TRACK_CONFIG
    frame_ts = float(frame_ts)
    active = getattr(scanner, "_active_tracks", {})

    for track in active.values():
        track.missed_frames += 1
        # Same-frame support is diagnostic and should describe the current
        # observation frame, not accumulate indefinitely.
        if abs(_finite(getattr(track, "v2226_last_support_frame_ts", -1.0), -1.0) - frame_ts) > cfg.frame_epsilon_s:
            track.v2226_same_frame_support = 0

    ordered = sorted(
        (dict(c) for c in list(candidates or [])),
        key=lambda c: _finite(c.get("score", 0.0)),
        reverse=True,
    )

    observed_this_call: set[int] = set()
    new_tracks = 0
    temporal_matches = 0
    same_frame_support = 0
    max_support = 1

    for candidate in ordered:
        cx = _finite(candidate.get("camera_x", 0.0))
        cy = _finite(candidate.get("camera_y", 0.0))
        score = _finite(candidate.get("score", 0.0))

        best_track = None
        best_dist = float("inf")
        for track in active.values():
            dist = float(np.hypot(_finite(track.camera_x) - cx, _finite(track.camera_y) - cy))
            if dist <= float(getattr(scanner, "track_merge_radius_px", 12.0)) and dist < best_dist:
                best_track = track
                best_dist = dist

        if best_track is None:
            enriched = dict(candidate)
            enriched["v2226_same_frame_support"] = 1.0
            enriched["v2226_frame_unique_observation"] = 1.0
            track = HoleTrack(
                track_id=int(getattr(scanner, "_next_track_id", 1)),
                camera_x=float(cx),
                camera_y=float(cy),
                created_at=frame_ts,
                first_seen_ts=frame_ts,
                last_seen_ts=frame_ts,
                hits=1,
                best_score=float(score),
                emitted=False,
                missed_frames=0,
                state="tentative",
                last_candidate=enriched,
            )
            track.v2226_same_frame_support = 1
            track.v2226_last_support_frame_ts = frame_ts
            track.v2226_unique_frame_hits = 1
            active[track.track_id] = track
            scanner._next_track_id = track.track_id + 1
            observed_this_call.add(track.track_id)
            new_tracks += 1
            continue

        same_physical_frame = abs(_finite(getattr(best_track, "last_seen_ts", 0.0)) - frame_ts) <= cfg.frame_epsilon_s
        if best_track.track_id in observed_this_call or same_physical_frame:
            _track_same_frame_support(best_track, candidate, frame_ts)
            support = int(getattr(best_track, "v2226_same_frame_support", 1) or 1)
            max_support = max(max_support, support)
            same_frame_support += 1
            observed_this_call.add(best_track.track_id)
            continue

        # First observation of this track on a genuinely later camera frame.
        alpha = 0.35
        best_track.camera_x = float((1.0 - alpha) * _finite(best_track.camera_x) + alpha * cx)
        best_track.camera_y = float((1.0 - alpha) * _finite(best_track.camera_y) + alpha * cy)
        best_track.last_seen_ts = frame_ts
        best_track.hits += 1
        best_track.v2226_unique_frame_hits = int(best_track.hits)
        best_track.v2226_same_frame_support = 1
        best_track.v2226_last_support_frame_ts = frame_ts
        best_track.best_score = float(max(_finite(best_track.best_score), score))
        enriched = dict(candidate)
        enriched["v2226_same_frame_support"] = 1.0
        enriched["v2226_frame_unique_observation"] = 1.0
        best_track.last_candidate = enriched
        best_track.missed_frames = 0
        if best_track.hits >= int(getattr(scanner, "track_confirm_frames", 3)):
            best_track.state = "stable"
        observed_this_call.add(best_track.track_id)
        temporal_matches += 1

    scanner._drop_dead_tracks(frame_ts)

    diag = {
        "raw_candidates": float(len(ordered)),
        "new_tracks": float(new_tracks),
        "temporal_matches": float(temporal_matches),
        "same_frame_support": float(same_frame_support),
        "max_same_frame_support": float(max_support),
    }
    try:
        scanner.last_window_debug["v2226_raw_candidates"] = diag["raw_candidates"]
        scanner.last_window_debug["v2226_new_tracks"] = diag["new_tracks"]
        scanner.last_window_debug["v2226_temporal_matches"] = diag["temporal_matches"]
        scanner.last_window_debug["v2226_same_frame_support"] = diag["same_frame_support"]
        scanner.last_window_debug["v2226_max_same_frame_support"] = diag["max_same_frame_support"]
    except Exception:
        pass
    return diag


def _install_frame_unique_tracking_patch() -> None:
    from src.engine.camera.hit_scanner import HitScanner
    if getattr(HitScanner, "_v2226_frame_unique_tracking_patch", False):
        return

    previous_update_tracks = HitScanner._update_tracks
    HitScanner._v2226_update_tracks_original = previous_update_tracks

    def update_tracks_v2226(self, candidates, frame_ts):
        diag = update_tracks_frame_unique_v2226(self, candidates, frame_ts)
        if _TRACK_CONFIG.track_log and list(candidates or []) and bool(getattr(self, "_has_open_events", lambda: False)()):
            print(
                f"[V2.22.6 TRACK] frame={float(frame_ts):.3f} "
                f"raw={int(diag['raw_candidates'])} new={int(diag['new_tracks'])} "
                f"temporal={int(diag['temporal_matches'])} "
                f"same_frame={int(diag['same_frame_support'])} "
                f"max_support={int(diag['max_same_frame_support'])}"
            )

    HitScanner._update_tracks = update_tracks_v2226
    HitScanner._v2226_frame_unique_tracking_patch = True


# ---------------------------------------------------------------------------
# Audio raw/near-miss telemetry.  Trigger decision is intentionally copied from
# the existing AudioPeakDetector without changing its thresholds or cooldown.
# ---------------------------------------------------------------------------


def _audio_reject_reasons(
    *, peak: float, rms: float, noise_floor: float, min_abs: float,
    peak_ratio: float, crest_factor: float, cooldown_ok: bool,
) -> tuple[list[str], float, float, float]:
    dyn_threshold = float(noise_floor) * float(peak_ratio)
    crest_threshold = float(rms) * float(crest_factor)
    reasons: list[str] = []
    if peak < min_abs:
        reasons.append("abs")
    if peak < dyn_threshold:
        reasons.append("noise")
    if peak < max(min_abs, crest_threshold):
        reasons.append("crest")
    if not cooldown_ok:
        reasons.append("cooldown")
    required = max(min_abs, dyn_threshold, crest_threshold)
    return reasons, dyn_threshold, crest_threshold, required


def _install_audio_telemetry_patch() -> None:
    from src.engine.audio.audio_peak_detector import AudioPeakDetector, AudioPeakEvent
    if getattr(AudioPeakDetector, "_v2226_audio_telemetry_patch", False):
        return

    previous_process = AudioPeakDetector._process_chunk
    AudioPeakDetector._v2226_process_chunk_original = previous_process

    def process_chunk_v2226(self, chunk: bytes) -> None:
        data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        if data.size == 0:
            return
        data /= 32768.0
        abs_data = np.abs(data)
        peak = float(np.max(abs_data))
        rms = float(np.sqrt(np.mean(np.square(data))))
        self.last_peak_value = peak
        self.last_rms = rms

        # Preserve baseline behaviour exactly: update noise before evaluating
        # the current chunk, then use the same absolute/dynamic/crest/cooldown
        # trigger logic as the historical detector.
        alpha = 0.03
        self.noise_floor = (1.0 - alpha) * self.noise_floor + alpha * rms
        with self._lock:
            self._sample_history.extend(float(x) for x in data.tolist())

        now = time.time()
        trigger_threshold = max(self.min_abs_peak, self.noise_floor * self.peak_ratio)
        crest_ok = peak >= max(self.min_abs_peak, rms * self.crest_factor_required)
        cooldown_ok = (now - self.last_peak_ts) >= self.cooldown_s
        is_peak = peak >= trigger_threshold and crest_ok and cooldown_ok

        reasons, dyn_threshold, crest_threshold, required = _audio_reject_reasons(
            peak=peak,
            rms=rms,
            noise_floor=self.noise_floor,
            min_abs=self.min_abs_peak,
            peak_ratio=self.peak_ratio,
            crest_factor=self.crest_factor_required,
            cooldown_ok=cooldown_ok,
        )
        crest_ratio = peak / max(rms, 1e-9)

        if is_peak:
            event_ts = self._estimate_event_timestamp(
                samples=data,
                chunk_end_ts=now,
                trigger_threshold=max(self.min_abs_peak, trigger_threshold * 0.85),
            )
            self.last_peak_ts = event_ts
            ev = AudioPeakEvent(timestamp=event_ts, peak=peak, rms=rms)
            with self._lock:
                self._events.append(ev)
                self._pending_dispatch.append(ev)
            self.v2226_last_audio_decision = {
                "kind": "trigger",
                "timestamp": float(event_ts),
                "peak": peak,
                "rms": rms,
                "noise": float(self.noise_floor),
                "required": float(required),
                "crest_ratio": float(crest_ratio),
            }
            if _AUDIO_CONFIG.log_triggers:
                print(
                    f"[V2.22.6 AUDIO-RAW] TRIGGER peak={peak:.3f} rms={rms:.3f} "
                    f"noise={self.noise_floor:.3f} abs={self.min_abs_peak:.3f} "
                    f"dyn={dyn_threshold:.3f} crest={crest_ratio:.2f}/{self.crest_factor_required:.2f}"
                )
            return

        if not _AUDIO_CONFIG.enabled:
            return

        near_gate = max(_AUDIO_CONFIG.absolute_floor, float(required) * _AUDIO_CONFIG.near_ratio)
        if peak < near_gate:
            return
        last_log = _finite(getattr(self, "v2226_last_near_log_ts", 0.0))
        if now - last_log < _AUDIO_CONFIG.min_log_interval_s:
            return
        self.v2226_last_near_log_ts = now
        decision = {
            "kind": "near_miss",
            "timestamp": float(now),
            "peak": peak,
            "rms": rms,
            "noise": float(self.noise_floor),
            "abs_threshold": float(self.min_abs_peak),
            "dynamic_threshold": float(dyn_threshold),
            "crest_required": float(crest_threshold),
            "crest_ratio": float(crest_ratio),
            "required": float(required),
            "reasons": tuple(reasons),
        }
        self.v2226_last_audio_decision = decision
        with self._lock:
            history = getattr(self, "v2226_near_misses", None)
            if not isinstance(history, deque):
                history = deque(maxlen=64)
                self.v2226_near_misses = history
            history.append(dict(decision))
        reject = ",".join(reasons) if reasons else "unknown"
        print(
            f"[V2.22.6 AUDIO-RAW] NEAR-MISS peak={peak:.3f} rms={rms:.3f} "
            f"noise={self.noise_floor:.3f} abs={self.min_abs_peak:.3f} "
            f"dyn={dyn_threshold:.3f} crest={crest_ratio:.2f}/{self.crest_factor_required:.2f} "
            f"required={required:.3f} reject={reject}"
        )

    previous_status = AudioPeakDetector.get_status_lines

    def status_v2226(self):
        lines = list(previous_status(self))
        decision = getattr(self, "v2226_last_audio_decision", None)
        if isinstance(decision, dict):
            if decision.get("kind") == "near_miss":
                reasons = ",".join(decision.get("reasons", ())) or "unknown"
                lines.append(
                    f"Audio near-miss: peak {float(decision.get('peak', 0.0)):.3f} "
                    f"required {float(decision.get('required', 0.0)):.3f} ({reasons})"
                )
            elif decision.get("kind") == "trigger":
                lines.append(
                    f"Audio raw trigger: peak {float(decision.get('peak', 0.0)):.3f} "
                    f"required {float(decision.get('required', 0.0)):.3f}"
                )
        return lines

    AudioPeakDetector._process_chunk = process_chunk_v2226
    AudioPeakDetector.get_status_lines = status_v2226
    AudioPeakDetector._v2226_audio_telemetry_patch = True


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def install_v2226_runtime(AppClass: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_settings_defaults()
    _load_config()
    _install_frame_unique_tracking_patch()
    _install_audio_telemetry_patch()
    AppClass._v2226_frame_unique_tracking_patch = True
    _INSTALLED = True
    print("[V2.22.6] frame-unique tracking + audio near-miss telemetry installed")


__all__ = [
    "SCHEMA_VERSION",
    "PATCH_REVISION",
    "TrackConfigV2226",
    "AudioTelemetryConfigV2226",
    "update_tracks_frame_unique_v2226",
    "install_v2226_runtime",
]
