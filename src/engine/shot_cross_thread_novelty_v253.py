"""V2.25.3 cross-thread readiness + cross-shot physical novelty authority.

Physical V2.25.2 testing exposed two separate problems:

1. CandidateGeneratorV2 runs in the V2.22.4 CV worker and marked a *worker scanner*
   instance as registered-ready, while HitScanner authority runs on the main scanner
   instance.  The main thread therefore waited until the V2.25.2 fail-open despite a
   registered frame already being logged.
2. The registered freshness gate was intentionally permissive and, on the worn test
   board, nearly every region candidate passed it.  Stable projector/board hotspots
   therefore remained competitive from shot to shot.

V2.25.3 keeps V2.25.2 as a high-recall physical gate, but adds:

* a process-local, lock-protected shot bridge shared by worker and main scanner;
* cross-shot recurrence telemetry in canonical full-camera coordinates;
* a soft novelty score that favours a newly appearing physical location over a
  repeatedly selected hotspot;
* re-hit recovery when the registered PRE->POST signature strengthens materially;
* direct V2.22.5 confirmation before V2.25.x balancing, so a spatially novel
  candidate cannot be discarded by an older absolute-score balance first;
* one global V2.22.5 FULL rescue rather than an unregistered local fail-open.

No gameplay role, owner, score, health, material, damage or projectile semantic is
used.  Candidate XY is never snapped or rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Iterable, Sequence

import numpy as np

from src.engine.input.object_hit_v2223 import object_hit_registry_v2223
from src.engine.shot_fast_v2225 import rescue_router_v2225
from src.engine.shot_region_proposal_v251 import _finite, _shot_id_from_scanner

SCHEMA_VERSION = "2.25.3"
PATCH_REVISION = "r1"
_INSTALLED = False


@dataclass(frozen=True)
class CandidateSignatureV253:
    shot_id: int
    peak_ts: float
    group: str
    full_x: float
    full_y: float
    physical_score: float
    center_abs: float
    compact_abs: float
    dark_compact: float
    authority_source: str


@dataclass
class ShotRecordV253:
    shot_id: int
    peak_ts: float
    ready_frame_ts: float = 0.0
    confirmed: tuple[CandidateSignatureV253, ...] = ()
    updated_at: float = field(default_factory=time.time)


class CrossThreadShotBridgeV253:
    """Small process-local bridge between the single CV worker and main scanner.

    Shot id alone is not trusted forever: peak timestamp is retained so a reset/reuse
    of ids cannot inherit old readiness.  All state is diagnostic/physical only.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[int, ShotRecordV253] = {}
        self._last_peak_ts = 0.0
        self._max_shot_id = 0

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._last_peak_ts = 0.0
            self._max_shot_id = 0

    def _maybe_new_generation(self, shot_id: int, peak_ts: float) -> None:
        # A substantially newer peak with a lower/reused id means the scanner was
        # reset. Delayed worker results are normally within <2 s and therefore do
        # not satisfy this conservative reset condition.
        if (
            self._records
            and shot_id < self._max_shot_id
            and peak_ts > self._last_peak_ts + 2.0
        ):
            self._records.clear()
            self._max_shot_id = 0

    def mark_ready(self, shot_id: int, frame_ts: float, peak_ts: float) -> bool:
        sid = int(shot_id)
        if sid <= 0:
            return False
        now = time.time()
        with self._lock:
            self._maybe_new_generation(sid, peak_ts)
            rec = self._records.get(sid)
            if rec is None or (peak_ts > 0 and abs(rec.peak_ts - peak_ts) > 0.20):
                rec = ShotRecordV253(shot_id=sid, peak_ts=float(peak_ts), ready_frame_ts=float(frame_ts))
                self._records[sid] = rec
            else:
                rec.ready_frame_ts = max(float(rec.ready_frame_ts), float(frame_ts))
                rec.updated_at = now
            self._last_peak_ts = max(self._last_peak_ts, float(peak_ts))
            self._max_shot_id = max(self._max_shot_id, sid)
            self._prune_locked(now)
            return True

    def is_ready(self, shot_id: int, peak_ts: float = 0.0) -> bool:
        sid = int(shot_id)
        now = time.time()
        with self._lock:
            rec = self._records.get(sid)
            if rec is None or rec.ready_frame_ts <= 0.0:
                return False
            if now - rec.updated_at > 12.0:
                return False
            if peak_ts > 0.0 and rec.peak_ts > 0.0 and abs(float(peak_ts) - rec.peak_ts) > 0.20:
                return False
            return True

    def store_confirmed(self, shot_id: int, peak_ts: float, candidates: Sequence[dict[str, Any]]) -> None:
        sid = int(shot_id)
        if sid <= 0:
            return
        sigs: list[CandidateSignatureV253] = []
        for cand in candidates:
            group = str(cand.get("v251_region_group", ""))
            if not group:
                continue
            sigs.append(CandidateSignatureV253(
                shot_id=sid,
                peak_ts=float(peak_ts),
                group=group,
                full_x=_finite(cand.get("v253_full_camera_x", cand.get("camera_x", 0.0))),
                full_y=_finite(cand.get("v253_full_camera_y", cand.get("camera_y", 0.0))),
                physical_score=_finite(cand.get("v252_physical_score", 0.0)),
                center_abs=_finite(cand.get("v252_center_abs", 0.0)),
                compact_abs=_finite(cand.get("v252_compact_abs", 0.0)),
                dark_compact=_finite(cand.get("v252_dark_compact", 0.0)),
                authority_source=str(cand.get("v252_authority_source", "")),
            ))
        now = time.time()
        with self._lock:
            self._maybe_new_generation(sid, peak_ts)
            rec = self._records.get(sid)
            if rec is None:
                rec = ShotRecordV253(shot_id=sid, peak_ts=float(peak_ts))
                self._records[sid] = rec
            # Preserve the broad registered-confirmed pool from the first round;
            # later persistence rounds are usually smaller and must not erase
            # recurrence evidence. Merge by physical camera position.
            merged = list(rec.confirmed)
            for sig in sigs:
                replace_at = None
                for i, old in enumerate(merged):
                    if _distance(sig.full_x, sig.full_y, old.full_x, old.full_y) <= 3.0:
                        replace_at = i
                        break
                if replace_at is None:
                    merged.append(sig)
                elif sig.physical_score >= merged[replace_at].physical_score:
                    merged[replace_at] = sig
            merged.sort(key=lambda x: x.physical_score, reverse=True)
            rec.confirmed = tuple(merged[:64])
            rec.updated_at = now
            self._last_peak_ts = max(self._last_peak_ts, float(peak_ts))
            self._max_shot_id = max(self._max_shot_id, sid)
            self._prune_locked(now)

    def history(self, shot_id: int, peak_ts: float, previous_shots: int = 3) -> tuple[CandidateSignatureV253, ...]:
        sid = int(shot_id)
        with self._lock:
            records = [
                rec for other_sid, rec in self._records.items()
                if other_sid != sid
                and rec.confirmed
                and (peak_ts <= 0.0 or rec.peak_ts <= peak_ts + 0.05)
            ]
            records.sort(key=lambda rec: (rec.peak_ts, rec.shot_id), reverse=True)
            # unique previous shot contexts only
            records = records[:max(1, int(previous_shots))]
            return tuple(sig for rec in records for sig in rec.confirmed)

    def _prune_locked(self, now: float) -> None:
        stale = [sid for sid, rec in self._records.items() if now - rec.updated_at > 45.0]
        for sid in stale:
            self._records.pop(sid, None)
        if len(self._records) > 16:
            keep = sorted(self._records.values(), key=lambda r: r.updated_at, reverse=True)[:12]
            self._records = {r.shot_id: r for r in keep}


shot_bridge_v253 = CrossThreadShotBridgeV253()


def _runtime_settings() -> dict[str, Any]:
    try:
        from src.engine.ai.runtime import get_ai_runtime
        value = getattr(get_ai_runtime(), "settings", {})
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _setting_bool(name: str, default: bool) -> bool:
    return bool(_runtime_settings().get(name, default))


def _setting_float(name: str, default: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, _finite(_runtime_settings().get(name, default), default)))


def _setting_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(_runtime_settings().get(name, default))
    except Exception:
        value = int(default)
    return max(lo, min(hi, value))


def _enabled() -> bool:
    return _setting_bool("cross_shot_novelty_enabled_v253", True)


def _log_enabled() -> bool:
    return _setting_bool("cross_shot_novelty_log_v253", True)


def _history_shots() -> int:
    return _setting_int("cross_shot_novelty_history_shots_v253", 3, 1, 6)


def _recurrence_soft_px() -> float:
    return _setting_float("cross_shot_novelty_recurrence_soft_px_v253", 18.0, 6.0, 42.0)


def _recurrence_hard_px() -> float:
    return _setting_float("cross_shot_novelty_recurrence_hard_px_v253", 6.0, 2.0, 18.0)


def _authority_wait_s() -> float:
    return _setting_float("cross_thread_authority_wait_s_v253", 1.65, 0.8, 3.0)


def _rescue_request_s() -> float:
    return _setting_float("cross_thread_rescue_request_s_v253", 1.80, 1.0, 3.0)


def _peak_ts_from_scanner(scanner: Any, shot_id: int) -> float:
    sid = int(shot_id)
    for ev in list(getattr(scanner, "audio_events", []) or []):
        if int(getattr(ev, "shot_id", 0) or 0) == sid:
            return _finite(getattr(ev, "peak_ts", 0.0))
    snap = object_hit_registry_v2223.snapshot_for_shot(sid) if sid > 0 else None
    return _finite(getattr(snap, "peak_ts", 0.0)) if snap is not None else 0.0


def _candidate_full_camera_xy(scanner: Any, candidate: dict[str, Any]) -> tuple[float, float]:
    """Convert current CandidateGenerator working-plane XY to canonical camera XY.

    V2.22.1 rebases candidates after CandidateGenerator returns. V2.25.x runs inside
    CandidateGenerator, so recurrence history must do the equivalent mapping itself.
    """
    x = _finite(candidate.get("camera_x", 0.0))
    y = _finite(candidate.get("camera_y", 0.0))
    diag = getattr(scanner, "_v244_roi_diag", None)
    if not isinstance(diag, dict):
        return x, y
    crop = diag.get("crop")
    scale = diag.get("scale")
    if not (isinstance(crop, (tuple, list)) and len(crop) >= 2 and isinstance(scale, (tuple, list)) and len(scale) >= 2):
        return x, y
    sx = _finite(scale[0], 1.0) or 1.0
    sy = _finite(scale[1], 1.0) or 1.0
    return x / sx + _finite(crop[0]), y / sy + _finite(crop[1])


def _distance(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
    return math.hypot(float(a_x) - float(b_x), float(a_y) - float(b_y))


def _nearest_history(
    full_x: float,
    full_y: float,
    group: str,
    history: Sequence[CandidateSignatureV253],
) -> tuple[CandidateSignatureV253 | None, float, float]:
    nearest_global: CandidateSignatureV253 | None = None
    nearest_group: CandidateSignatureV253 | None = None
    dg = float("inf")
    ds = float("inf")
    for old in history:
        d = _distance(full_x, full_y, old.full_x, old.full_y)
        if d < dg:
            dg, nearest_global = d, old
        if old.group == group and d < ds:
            ds, nearest_group = d, old
    # Physical recurrence is screen/camera based, not semantic. Use the nearest
    # global hotspot for the penalty, while retaining same-group distance as telemetry.
    return nearest_global, dg, ds


def _signature_gain(candidate: dict[str, Any], old: CandidateSignatureV253 | None) -> float:
    if old is None:
        return 0.0
    center = max(0.0, _finite(candidate.get("v252_center_abs", 0.0)) - old.center_abs) / max(10.0, abs(old.center_abs) * 0.20)
    compact = max(0.0, _finite(candidate.get("v252_compact_abs", 0.0)) - old.compact_abs) / max(3.0, abs(old.compact_abs) * 0.45)
    dark = max(0.0, _finite(candidate.get("v252_dark_compact", 0.0)) - old.dark_compact) / max(2.0, abs(old.dark_compact) * 0.45)
    return min(2.0, 0.30 * center + 0.45 * compact + 0.25 * dark)


def _distance_novelty(distance_px: float, gain: float) -> float:
    if not math.isfinite(distance_px):
        return 1.0
    hard = _recurrence_hard_px()
    soft = max(hard + 1.0, _recurrence_soft_px())
    if distance_px >= soft:
        base = 1.0
    elif distance_px <= hard:
        base = 0.05 + 0.15 * (distance_px / max(1.0, hard))
    else:
        t = (distance_px - hard) / max(1.0, soft - hard)
        base = 0.20 + 0.80 * t
    # Re-hit/hole-in-hole recovery: recurrence is never a hard exclusion. A
    # materially stronger registered signature can recover most of the penalty.
    return min(1.25, base + 0.62 * min(1.5, max(0.0, gain)))


def _robust_group_excess(candidates: Sequence[dict[str, Any]]) -> dict[int, float]:
    by_group: dict[str, list[tuple[int, float]]] = {}
    for idx, cand in enumerate(candidates):
        group = str(cand.get("v251_region_group", ""))
        by_group.setdefault(group, []).append((idx, _finite(cand.get("v252_physical_score", 0.0))))
    out: dict[int, float] = {}
    for values in by_group.values():
        arr = np.asarray([v for _i, v in values], dtype=np.float32)
        med = float(np.median(arr)) if arr.size else 0.0
        mad = float(np.median(np.abs(arr - med))) if arr.size else 0.0
        scale = max(0.75, 1.4826 * mad)
        for idx, value in values:
            out[idx] = max(0.0, min(3.0, (value - med) / scale))
    return out


def annotate_cross_shot_novelty_v253(
    scanner: Any,
    shot_id: int,
    peak_ts: float,
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate exact candidate coordinates; never rewrite camera_x/camera_y."""
    original_xy = [(_finite(c.get("camera_x")), _finite(c.get("camera_y"))) for c in candidates]
    history = shot_bridge_v253.history(int(shot_id), float(peak_ts), _history_shots())
    excess = _robust_group_excess(candidates)
    group_counts: dict[str, int] = {}
    for c in candidates:
        group = str(c.get("v251_region_group", ""))
        group_counts[group] = group_counts.get(group, 0) + 1

    out: list[dict[str, Any]] = []
    for idx, original in enumerate(candidates):
        cand = dict(original)
        group = str(cand.get("v251_region_group", ""))
        fx, fy = _candidate_full_camera_xy(scanner, cand)
        old, nearest, same_group = _nearest_history(fx, fy, group, history)
        gain = _signature_gain(cand, old)
        novelty = _distance_novelty(nearest, gain)
        count = max(1, group_counts.get(group, 1))
        sparsity = 1.0 / math.sqrt(float(count))
        center = max(0.0, _finite(cand.get("v252_center_abs", 0.0)))
        compact = max(0.0, _finite(cand.get("v252_compact_abs", 0.0)))
        compact_ratio = min(1.0, compact / max(1.0, center))
        registered_bonus = 0.20 if str(cand.get("v252_authority_source", "")) == "region_registered" else 0.0
        score = (
            3.80 * novelty
            + 0.75 * excess.get(idx, 0.0)
            + 3.00 * sparsity
            + 1.20 * compact_ratio
            + registered_bonus
        )
        # Strong but soft suppression of exact recurring hotspots with no stronger
        # physical signature. The candidate remains legal and can recover via gain.
        hard = _recurrence_hard_px()
        soft = _recurrence_soft_px()
        if math.isfinite(nearest) and gain < 0.18:
            if nearest <= hard:
                score *= 0.35
            elif nearest <= max(hard + 2.0, soft * 0.60):
                score *= 0.58
            elif nearest < soft:
                score *= 0.82

        cand["v253_peak_ts"] = float(peak_ts)
        cand["v253_full_camera_x"] = float(fx)
        cand["v253_full_camera_y"] = float(fy)
        cand["v253_history_distance_px"] = float(nearest if math.isfinite(nearest) else 9999.0)
        cand["v253_same_group_history_distance_px"] = float(same_group if math.isfinite(same_group) else 9999.0)
        cand["v253_signature_gain"] = float(gain)
        cand["v253_distance_novelty"] = float(novelty)
        cand["v253_group_excess"] = float(excess.get(idx, 0.0))
        cand["v253_group_count"] = float(count)
        cand["v253_sparsity"] = float(sparsity)
        cand["v253_novelty_score"] = float(max(0.0, score))
        cand["v253_authority_ok"] = 1.0 if _finite(cand.get("v252_fresh_physical", 0.0)) > 0.5 else 0.0
        out.append(cand)

    # Contract test at runtime too: annotation is metadata only.
    assert original_xy == [(_finite(c.get("camera_x")), _finite(c.get("camera_y"))) for c in out]
    return out


def _confirm_strength_v253(candidate: dict[str, Any]) -> float:
    center = max(0.0, _finite(candidate.get("v2225_confirm_center_abs", 0.0)))
    compact = max(0.0, _finite(candidate.get("v2225_confirm_compact", 0.0)))
    peak = max(0.0, _finite(candidate.get("v2225_confirm_peak_abs", 0.0)))
    dark = max(0.0, _finite(candidate.get("v2225_confirm_darkening", 0.0)))
    return center + 1.8 * compact + 0.12 * peak + 0.35 * dark


def balance_confirmed_v253(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    values = [dict(c) for c in candidates if _finite(c.get("v253_authority_ok", 0.0)) > 0.5]
    if not values:
        return []
    by_group: dict[str, list[dict[str, Any]]] = {}
    for c in values:
        by_group.setdefault(str(c.get("v251_region_group", "")), []).append(c)
    for group, group_values in by_group.items():
        strengths = np.asarray([_confirm_strength_v253(c) for c in group_values], dtype=np.float32)
        med = float(np.median(strengths)) if strengths.size else 0.0
        mad = float(np.median(np.abs(strengths - med))) if strengths.size else 0.0
        scale = max(0.75, 1.4826 * mad)
        for c in group_values:
            s = _confirm_strength_v253(c)
            confirm_excess = max(0.0, min(3.0, (s - med) / scale))
            ratio = max(0.0, _finite(c.get("v2225_confirm_compact", 0.0))) / max(1.0, _finite(c.get("v2225_confirm_center_abs", 0.0)))
            c["v253_confirm_excess"] = float(confirm_excess)
            c["v253_confirm_score"] = float(
                _finite(c.get("v253_novelty_score", 0.0))
                + 0.90 * confirm_excess
                + 0.80 * min(1.5, ratio)
            )
        group_values.sort(key=lambda c: (_finite(c.get("v253_confirm_score")), _finite(c.get("v253_novelty_score"))), reverse=True)

    selected: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    for group in sorted(by_group):
        vals = by_group[group]
        if vals:
            selected.append(vals[0])
            extras.extend(vals[1:2])
    extras.sort(key=lambda c: _finite(c.get("v253_confirm_score", 0.0)), reverse=True)
    limit = max(len(selected), 8)
    selected.extend(extras[:max(0, limit - len(selected))])
    selected.sort(key=lambda c: _finite(c.get("v253_confirm_score", 0.0)), reverse=True)
    return selected[:max(8, len(by_group))]


def _patch_v252_ready_bridge() -> None:
    import src.engine.shot_region_freshness_v252 as v252

    def mark_ready_shared(scanner: Any, shot_id: int, frame_ts: float) -> None:
        peak_ts = _peak_ts_from_scanner(scanner, int(shot_id))
        first = not shot_bridge_v253.is_ready(int(shot_id), peak_ts)
        shot_bridge_v253.mark_ready(int(shot_id), float(frame_ts), peak_ts)
        # Retain local telemetry for tools that inspect worker scanner state.
        try:
            scanner._v253_shared_ready = int(shot_id)
        except Exception:
            pass
        if first and _log_enabled():
            print(
                f"[V2.25.3 READY-BRIDGE] shot={int(shot_id)} source=worker "
                f"frame={float(frame_ts):.3f} peak={peak_ts:.3f}"
            )

    def is_ready_shared(scanner: Any, shot_id: int) -> bool:
        peak_ts = _peak_ts_from_scanner(scanner, int(shot_id))
        return shot_bridge_v253.is_ready(int(shot_id), peak_ts)

    # V2.25.2 wrappers resolve these module globals at call time, so replacing
    # them here fixes existing closures without re-installing CandidateGenerator.
    v252._mark_ready = mark_ready_shared
    v252._is_ready = is_ready_shared


def _install_candidate_novelty_patch() -> None:
    from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2

    if getattr(CandidateGeneratorV2, "_v253_cross_shot_novelty_patch", False):
        return
    previous_generate = CandidateGeneratorV2.generate

    def generate_v253(self, scanner, gray, frame_ts, legacy_candidates):
        result = previous_generate(self, scanner, gray, frame_ts, legacy_candidates)
        if not _enabled() or not threading.current_thread().name.startswith("shot-cv-v2224"):
            return result
        sid = _shot_id_from_scanner(scanner)
        if sid <= 0 or rescue_router_v2225.requested(sid):
            return result
        ready = bool(getattr(result, "telemetry", {}).get("v252_registered_ready", False))
        if not ready:
            return result
        peak_ts = _peak_ts_from_scanner(scanner, sid)
        annotated = annotate_cross_shot_novelty_v253(scanner, sid, peak_ts, list(result.candidates))
        result.candidates = annotated
        try:
            result.telemetry = dict(result.telemetry or {})
            result.telemetry["v253_shared_ready"] = True
            result.telemetry["v253_history_candidates"] = len(shot_bridge_v253.history(sid, peak_ts, _history_shots()))
            scanner.last_window_debug["v253_shared_ready"] = 1.0
        except Exception:
            pass
        if _log_enabled():
            by_group: dict[str, list[dict[str, Any]]] = {}
            for c in annotated:
                group = str(c.get("v251_region_group", ""))
                if group:
                    by_group.setdefault(group, []).append(c)
            for group in sorted(by_group):
                vals = by_group[group]
                best = max(vals, key=lambda c: _finite(c.get("v253_novelty_score", 0.0)))
                d = _finite(best.get("v253_history_distance_px", 9999.0), 9999.0)
                status = "new" if d >= _recurrence_soft_px() else "recur"
                print(
                    f"[V2.25.3 NOVELTY] shot={sid} object={group} n={len(vals)} "
                    f"best={_finite(best.get('v253_novelty_score', 0.0)):.2f} {status}=1 "
                    f"dist={d:.1f}px gain={_finite(best.get('v253_signature_gain', 0.0)):.2f} "
                    f"src={best.get('v252_authority_source', '')} "
                    f"camera=({_finite(best.get('v253_full_camera_x')):.1f},{_finite(best.get('v253_full_camera_y')):.1f})"
                )
        return result

    CandidateGeneratorV2.generate = generate_v253
    CandidateGeneratorV2._v253_cross_shot_novelty_patch = True
    CandidateGeneratorV2._v253_previous_generate = previous_generate


def _install_confirmation_patch() -> None:
    import src.engine.shot_fast_v2225 as fast

    if getattr(fast, "_v253_novelty_confirm_patch", False):
        return
    previous_confirm = fast.local_confirm_candidates_v2225
    # This is the original V2.22.5 physical confirmation stored when V2.25.1
    # installed. Bypassing V2.25.1/2 balancing here prevents them dropping a
    # spatially novel candidate before V2.25.3 can compare it.
    base_confirm = getattr(fast, "_v251_previous_local_confirm", previous_confirm)

    def confirm_v253(pre_gray, current_gray, candidates, *, frame_ts, config=None):
        annotated = [c for c in candidates if _finite(c.get("v253_novelty_score", 0.0)) > 0.0]
        if not _enabled() or not annotated:
            return previous_confirm(pre_gray, current_gray, candidates, frame_ts=frame_ts, config=config)
        confirmed, diag = base_confirm(pre_gray, current_gray, candidates, frame_ts=frame_ts, config=config)
        registered = [
            c for c in confirmed
            if _finite(c.get("v252_fresh_physical", 0.0)) > 0.5
            and _finite(c.get("v253_authority_ok", 0.0)) > 0.5
        ]
        balanced = balance_confirmed_v253(registered)
        sid = int(_finite((annotated[0] if annotated else {}).get("v251_shot_id", 0.0)))
        peak_ts = _finite((annotated[0] if annotated else {}).get("v253_peak_ts", 0.0))
        if peak_ts <= 0.0:
            # CandidateGenerator sets shot id but not event object; the bridge's
            # existing ready record already carries the true peak. Store 0 only
            # as a fallback, history ordering still uses shot id.
            peak_ts = time.time()
        shot_bridge_v253.store_confirmed(sid, peak_ts, registered)
        if _log_enabled():
            best = max((_finite(c.get("v253_confirm_score", 0.0)) for c in balanced), default=0.0)
            print(
                f"[V2.25.3 CONFIRM] shot={sid} tested={int(diag.get('tested', 0))} "
                f"v2225={len(confirmed)} registered={len(registered)} balanced={len(balanced)} best={best:.2f}"
            )
        out_diag = dict(diag)
        out_diag["v253_v2225_confirmed"] = float(len(confirmed))
        out_diag["v253_registered_confirmed"] = float(len(registered))
        out_diag["v253_after_balance"] = float(len(balanced))
        return balanced, out_diag

    fast.local_confirm_candidates_v2225 = confirm_v253
    fast._v253_novelty_confirm_patch = True
    fast._v253_previous_local_confirm = previous_confirm


def _install_authority_selector_patch() -> None:
    from src.engine.camera.hit_scanner import HitScanner

    if getattr(HitScanner, "_v253_shared_novelty_selector", False):
        return
    previous_best = HitScanner._best_track_for_event
    base_physical_best = getattr(HitScanner, "_v251_previous_best_track", previous_best)

    def best_track_v253(self, event):
        sid = int(getattr(event, "shot_id", 0) or 0)
        snap = object_hit_registry_v2223.snapshot_for_shot(sid) if sid > 0 else None
        if not _enabled() or snap is None or not tuple(getattr(snap, "camera_regions", ()) or ()):
            return previous_best(self, event)
        if rescue_router_v2225.was_consumed(sid):
            return base_physical_best(self, event)

        peak_ts = _finite(getattr(event, "peak_ts", 0.0))
        age = max(0.0, time.time() - (peak_ts or time.time()))
        gate_log = getattr(self, "_v253_gate_log", None)
        if not isinstance(gate_log, set):
            gate_log = set()
            self._v253_gate_log = gate_log

        if not shot_bridge_v253.is_ready(sid, peak_ts):
            if age < _authority_wait_s():
                if _log_enabled() and (sid, "wait") not in gate_log:
                    gate_log.add((sid, "wait"))
                    print(f"[V2.25.3 AUTHORITY-WAIT] shot={sid} age={age*1000.0:.0f}ms reason=worker_not_ready")
                return None
            if rescue_router_v2225.request(sid):
                if _log_enabled():
                    print(f"[V2.25.3 RESCUE-REQUEST] shot={sid} age={age*1000.0:.0f}ms reason=no_shared_registered_frame")
                return None
            if age < 3.5:
                return None
            return base_physical_best(self, event)

        eligible: list[tuple[Any, dict[str, Any], float]] = []
        for track in getattr(self, "_active_tracks", {}).values():
            onset_dt = float(track.first_seen_ts) - float(event.peak_ts)
            if onset_dt < -float(self.association_lead_s) or onset_dt > float(self.association_lag_s):
                continue
            if track.emitted and event.matched_track_id != track.track_id:
                continue
            cand = getattr(track, "last_candidate", {}) or {}
            score = _finite(cand.get("v253_confirm_score", 0.0))
            if _finite(cand.get("v253_authority_ok", 0.0)) <= 0.5 or score <= 0.0:
                continue
            eligible.append((track, cand, onset_dt))

        if not eligible:
            if age >= _rescue_request_s() and rescue_router_v2225.request(sid):
                if _log_enabled() and (sid, "rescue") not in gate_log:
                    gate_log.add((sid, "rescue"))
                    print(f"[V2.25.3 RESCUE-REQUEST] shot={sid} age={age*1000.0:.0f}ms reason=no_novel_confirmed_track")
            return None

        best, cand, onset = max(
            eligible,
            key=lambda item: (
                _finite(item[1].get("v253_confirm_score", 0.0)),
                _finite(item[1].get("v253_distance_novelty", 0.0)),
                _finite(item[1].get("v253_group_excess", 0.0)),
                -abs(float(item[2])),
            ),
        )
        try:
            self.last_event_debug["v253_selector"] = "shared_registered_novelty"
            self.last_event_debug["v253_confirm_score"] = _finite(cand.get("v253_confirm_score", 0.0))
            self.last_event_debug["v253_history_distance_px"] = _finite(cand.get("v253_history_distance_px", 9999.0), 9999.0)
        except Exception:
            pass
        key = (sid, "selected", int(getattr(best, "track_id", 0) or 0))
        if _log_enabled() and key not in gate_log:
            gate_log.add(key)
            print(
                f"[V2.25.3 AUTHORITY] shot={sid} tracks={len(eligible)} "
                f"group={cand.get('v251_region_group', '')} src={cand.get('v252_authority_source', '')} "
                f"score={_finite(cand.get('v253_confirm_score', 0.0)):.2f} "
                f"dist={_finite(cand.get('v253_history_distance_px', 9999.0), 9999.0):.1f}px "
                f"xy=({_finite(cand.get('camera_x')):.1f},{_finite(cand.get('camera_y')):.1f})"
            )
        return best

    HitScanner._best_track_for_event = best_track_v253
    HitScanner._v253_shared_novelty_selector = True
    HitScanner._v253_previous_best_track = previous_best


def _install_settings_defaults() -> None:
    defaults = {
        "cross_shot_novelty_enabled_v253": True,
        "cross_shot_novelty_log_v253": True,
        "cross_shot_novelty_history_shots_v253": 3,
        "cross_shot_novelty_recurrence_soft_px_v253": 18.0,
        "cross_shot_novelty_recurrence_hard_px_v253": 6.0,
        "cross_thread_authority_wait_s_v253": 1.65,
        "cross_thread_rescue_request_s_v253": 1.80,
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


def install_v253_runtime(AppClass: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_settings_defaults()
    _patch_v252_ready_bridge()
    _install_candidate_novelty_patch()
    _install_confirmation_patch()
    _install_authority_selector_patch()
    AppClass._v253_cross_thread_novelty_patch = True
    _INSTALLED = True
    print(
        "[V2.25.3] shared worker/main readiness + cross-shot physical novelty authority installed "
        f"(history={_history_shots()} shots, recur={_recurrence_hard_px():.0f}/{_recurrence_soft_px():.0f}px, global rescue preserved)"
    )


__all__ = [
    "SCHEMA_VERSION",
    "PATCH_REVISION",
    "CandidateSignatureV253",
    "ShotRecordV253",
    "CrossThreadShotBridgeV253",
    "shot_bridge_v253",
    "annotate_cross_shot_novelty_v253",
    "balance_confirmed_v253",
    "install_v253_runtime",
]
