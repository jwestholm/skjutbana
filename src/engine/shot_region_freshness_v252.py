"""V2.25.2 registered freshness authority for object-context shots.

V2.25.1 successfully partitioned the frozen GameObject search space and bounded
candidate competition, but physical testing exposed two authority leaks:

* an old/legacy/banked candidate inside a region could still become the final
  track even when that region produced no registered V2.25.1 proposal; and
* CandidateGeneratorV2's short ``waiting_post_peak`` path could seed/confirm a
  legacy track and emit it before a registered PRE->POST V2 frame had run.

V2.25.2 keeps those early/legacy candidates for recall, but changes authority:

    early legacy proposals ---------------------------+
                                                       |
    V2 registered PRE -> registered POST evidence ----+--> revalidate exact XY
                                                       |      |
    V2.25.1 per-region proposals ----------------------+      v
                                                  registered-fresh candidates
                                                             |
                                                V2.22.5 persistence confirmation
                                                             |
                                                    V2.25.2 authority selector
                                                             |
                                                         physical Hit XY

Object roles, owner, score, damage and projectile semantics never participate in
this decision.  Candidate coordinates are never moved.  The V2.22.5 FULL rescue
remains the original global physical path and bypasses this authority gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Iterable, Sequence

import numpy as np

from src.engine.input.object_hit_v2223 import object_hit_registry_v2223
from src.engine.shot_fast_v2225 import rescue_router_v2225
from src.engine.shot_region_proposal_v251 import (
    RegionGroupV251,
    _camera_regions_to_work_groups,
    _finite,
    _region_for_candidate,
    _shot_id_from_scanner,
)

SCHEMA_VERSION = "2.25.2"
PATCH_REVISION = "r1"
_INSTALLED = False


@dataclass(frozen=True)
class RegisteredMetricsV252:
    center_abs: float
    ring_abs: float
    compact_abs: float
    peak_abs: float
    center_z: float
    center_dark: float
    ring_dark: float
    dark_compact: float
    dark_fraction: float
    pre_noise: float
    best_dx: int = 0
    best_dy: int = 0


@dataclass
class RegisteredFrameV252:
    shot_id: int
    frame_ts: float
    bbox: tuple[int, int, int, int]
    absdiff: np.ndarray
    zscore: np.ndarray
    darkening: np.ndarray
    temporal_noise: np.ndarray | None
    groups: tuple[RegionGroupV251, ...]


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
    return _setting_bool("registered_freshness_enabled_v252", True)


def _log_enabled() -> bool:
    return _setting_bool("registered_freshness_log_v252", True)


def _search_radius() -> int:
    return _setting_int("registered_freshness_search_radius_v252", 3, 0, 5)


def _authority_fail_open_s() -> float:
    return _setting_float("registered_freshness_fail_open_s_v252", 1.10, 0.50, 1.80)


def _fresh_min_abs() -> float:
    return _setting_float("registered_freshness_min_center_abs_v252", 1.60, 0.2, 20.0)


def _fresh_min_z() -> float:
    return _setting_float("registered_freshness_min_center_z_v252", 1.30, 0.2, 12.0)


def _fresh_min_compact() -> float:
    return _setting_float("registered_freshness_min_compact_v252", 0.30, -2.0, 20.0)


def _fresh_min_dark_compact() -> float:
    return _setting_float("registered_freshness_min_dark_compact_v252", 0.24, -2.0, 20.0)


def _fresh_min_dark_fraction() -> float:
    return _setting_float("registered_freshness_min_dark_fraction_v252", 0.42, 0.0, 1.0)


def _per_group_confirmed() -> int:
    return _setting_int("registered_freshness_confirmed_per_group_v252", 2, 1, 6)


def _confirmed_total() -> int:
    return _setting_int("registered_freshness_confirmed_total_v252", 8, 2, 32)


def _disc_ring_metrics(
    absdiff: np.ndarray,
    zscore: np.ndarray,
    darkening: np.ndarray,
    temporal_noise: np.ndarray | None,
    px: int,
    py: int,
) -> RegisteredMetricsV252 | None:
    h, w = absdiff.shape[:2]
    radius = 8
    if px < 0 or py < 0 or px >= w or py >= h:
        return None
    x0, x1 = max(0, px - radius), min(w, px + radius + 1)
    y0, y1 = max(0, py - radius), min(h, py + radius + 1)
    if x1 - x0 < 7 or y1 - y0 < 7:
        return None
    a = np.asarray(absdiff[y0:y1, x0:x1], dtype=np.float32)
    z = np.asarray(zscore[y0:y1, x0:x1], dtype=np.float32)
    d = np.asarray(darkening[y0:y1, x0:x1], dtype=np.float32)
    yy, xx = np.ogrid[:a.shape[0], :a.shape[1]]
    cx, cy = px - x0, py - y0
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    center = d2 <= 4.0
    ring = (d2 >= 16.0) & (d2 <= 64.0)
    if not np.any(center):
        return None
    center_abs = float(np.mean(a[center]))
    ring_abs = float(np.mean(a[ring])) if np.any(ring) else 0.0
    center_z = float(np.mean(z[center]))
    center_dark = float(np.mean(d[center]))
    ring_dark = float(np.mean(d[ring])) if np.any(ring) else 0.0
    if temporal_noise is not None and isinstance(temporal_noise, np.ndarray) and temporal_noise.shape == absdiff.shape:
        n = np.asarray(temporal_noise[y0:y1, x0:x1], dtype=np.float32)
        pre_noise = float(np.median(n[center]))
    else:
        pre_noise = 0.0
    compact_abs = center_abs - ring_abs
    dark_compact = center_dark - ring_dark
    dark_fraction = center_dark / max(0.50, center_abs)
    return RegisteredMetricsV252(
        center_abs=center_abs,
        ring_abs=ring_abs,
        compact_abs=compact_abs,
        peak_abs=float(np.max(a[center])),
        center_z=center_z,
        center_dark=center_dark,
        ring_dark=ring_dark,
        dark_compact=dark_compact,
        dark_fraction=max(0.0, min(1.5, dark_fraction)),
        pre_noise=max(0.0, pre_noise),
    )


def _metric_quality(metrics: RegisteredMetricsV252, region_evidence: float = 0.0) -> float:
    """Pure physical score; deliberately contains no object/game semantics."""
    compact = max(0.0, metrics.compact_abs)
    compact_ratio = compact / max(0.75, metrics.center_abs)
    dark_compact = max(0.0, metrics.dark_compact)
    base = (
        1.55 * min(12.0, max(0.0, metrics.center_z))
        + 0.72 * min(18.0, compact)
        + 4.2 * min(1.2, compact_ratio)
        + 0.34 * min(18.0, dark_compact)
        + 1.35 * min(1.0, max(0.0, metrics.dark_fraction))
        + 0.18 * min(14.0, max(0.0, region_evidence))
    )
    # A location that was already unstable in the immediate PRE stack is less
    # convincing, but this is only a soft physical penalty. Moving targets and
    # projected video must remain hittable.
    penalty = 0.16 * min(20.0, max(0.0, metrics.pre_noise))
    return max(0.0, float(base - penalty))


def _fresh_gate(metrics: RegisteredMetricsV252) -> bool:
    temporal_ok = (
        metrics.center_abs >= _fresh_min_abs()
        and (metrics.center_z >= _fresh_min_z() or metrics.center_abs >= 3.4)
    )
    shape_ok = (
        metrics.compact_abs >= _fresh_min_compact()
        or metrics.dark_compact >= _fresh_min_dark_compact()
        or (
            metrics.dark_fraction >= _fresh_min_dark_fraction()
            and metrics.dark_compact >= 0.12
            and metrics.center_abs >= max(1.8, _fresh_min_abs())
        )
    )
    return bool(temporal_ok and shape_ok)


def _best_metrics_for_candidate(candidate: dict[str, Any], ctx: RegisteredFrameV252) -> RegisteredMetricsV252 | None:
    bx0, by0, _bx1, _by1 = ctx.bbox
    base_x = int(round(_finite(candidate.get("camera_x")) - bx0))
    base_y = int(round(_finite(candidate.get("camera_y")) - by0))
    radius = _search_radius()
    best: tuple[float, RegisteredMetricsV252] | None = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            metrics = _disc_ring_metrics(
                ctx.absdiff, ctx.zscore, ctx.darkening, ctx.temporal_noise,
                base_x + dx, base_y + dy,
            )
            if metrics is None:
                continue
            quality = _metric_quality(metrics, _finite(candidate.get("v251_region_evidence", 0.0)))
            # Search is evidence-only. The returned candidate XY remains exactly
            # the original observed point; dx/dy are telemetry only.
            metrics = RegisteredMetricsV252(**{
                **metrics.__dict__, "best_dx": int(dx), "best_dy": int(dy)
            })
            if best is None or quality > best[0]:
                best = (quality, metrics)
    return best[1] if best is not None else None


def _annotate_candidate(candidate: dict[str, Any], ctx: RegisteredFrameV252) -> dict[str, Any]:
    out = dict(candidate)
    group_name = str(out.get("v251_region_group", ""))
    group = next((g for g in ctx.groups if g.group_id == group_name), None)
    if group is None:
        group = _region_for_candidate(ctx.groups, out)
    if group is None:
        out["v252_evidence_ready"] = 1.0
        out["v252_fresh_physical"] = 0.0
        out["v252_authority_source"] = "outside_frozen_regions"
        out["v252_authority_score"] = 0.0
        return out

    out.setdefault("v251_shot_id", int(ctx.shot_id))
    out.setdefault("v251_region_group", group.group_id)
    out.setdefault("v251_region_objects", "|".join(group.object_ids))
    out.setdefault("v251_region_roles", "|".join(group.roles))
    metrics = _best_metrics_for_candidate(out, ctx)
    out["v252_evidence_ready"] = 1.0
    if metrics is None:
        out["v252_fresh_physical"] = 0.0
        out["v252_authority_source"] = "diagnostic_only"
        out["v252_authority_score"] = 0.0
        return out

    out["v252_center_abs"] = float(metrics.center_abs)
    out["v252_ring_abs"] = float(metrics.ring_abs)
    out["v252_compact_abs"] = float(metrics.compact_abs)
    out["v252_peak_abs"] = float(metrics.peak_abs)
    out["v252_center_z"] = float(metrics.center_z)
    out["v252_center_dark"] = float(metrics.center_dark)
    out["v252_ring_dark"] = float(metrics.ring_dark)
    out["v252_dark_compact"] = float(metrics.dark_compact)
    out["v252_dark_fraction"] = float(metrics.dark_fraction)
    out["v252_pre_noise"] = float(metrics.pre_noise)
    out["v252_evidence_best_dx"] = float(metrics.best_dx)
    out["v252_evidence_best_dy"] = float(metrics.best_dy)
    quality = _metric_quality(metrics, _finite(out.get("v251_region_evidence", 0.0)))
    fresh = _fresh_gate(metrics)
    registered = _finite(out.get("v251_registered_region_proposal", 0.0)) > 0.5
    out["v252_fresh_physical"] = 1.0 if fresh else 0.0
    out["v252_authority_source"] = (
        "region_registered" if fresh and registered
        else "legacy_revalidated" if fresh
        else "diagnostic_only"
    )
    out["v252_physical_score"] = float(quality)
    out["v252_authority_score"] = float(quality)
    return out


def _mad(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    arr = np.asarray(values, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, max(0.50, 1.4826 * mad)


def _normalise_groups(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add within-region excess and a soft physical density penalty."""
    out = [dict(c) for c in candidates]
    groups: dict[str, list[dict[str, Any]]] = {}
    for cand in out:
        group = str(cand.get("v251_region_group", ""))
        if group:
            groups.setdefault(group, []).append(cand)
    for group, values in groups.items():
        scores = [_finite(c.get("v252_physical_score", 0.0)) for c in values]
        med, scale = _mad(scores)
        density = len(values)
        density_weight = 1.0 / (1.0 + 0.075 * max(0, density - 1))
        for cand in values:
            score = _finite(cand.get("v252_physical_score", 0.0))
            excess = max(0.0, (score - med) / scale)
            registered_bonus = 0.30 if str(cand.get("v252_authority_source", "")) == "region_registered" else 0.0
            authority = score * density_weight + 1.20 * min(6.0, excess) + registered_bonus
            cand["v252_group_count"] = float(density)
            cand["v252_group_median"] = float(med)
            cand["v252_group_excess"] = float(excess)
            cand["v252_density_weight"] = float(density_weight)
            cand["v252_authority_score"] = float(authority if _finite(cand.get("v252_fresh_physical", 0.0)) > 0.5 else 0.0)
    return out


def _balance_fresh_confirmed(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for original in candidates:
        cand = dict(original)
        if _finite(cand.get("v252_fresh_physical", 0.0)) <= 0.5:
            continue
        group = str(cand.get("v251_region_group", ""))
        if not group:
            continue
        old_bonus = (
            0.10 * max(0.0, _finite(cand.get("v2225_confirm_center_abs", 0.0)))
            + 0.18 * max(0.0, _finite(cand.get("v2225_confirm_compact", 0.0)))
        )
        cand["v252_confirm_score"] = float(_finite(cand.get("v252_authority_score", 0.0)) + min(2.0, old_bonus))
        groups.setdefault(group, []).append(cand)
    if not groups:
        return []
    selected: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    per_group = _per_group_confirmed()
    for group in sorted(groups):
        values = sorted(
            groups[group],
            key=lambda c: (_finite(c.get("v252_confirm_score", 0.0)), _finite(c.get("score", 0.0))),
            reverse=True,
        )
        selected.append(values[0])
        extras.extend(values[1:per_group])
    total = max(len(selected), _confirmed_total())
    extras.sort(key=lambda c: _finite(c.get("v252_confirm_score", 0.0)), reverse=True)
    selected.extend(extras[:max(0, total - len(selected))])
    selected.sort(key=lambda c: _finite(c.get("v252_confirm_score", 0.0)), reverse=True)
    return selected


def _mark_ready(scanner: Any, shot_id: int, frame_ts: float) -> None:
    ready = getattr(scanner, "_v252_registered_ready_shots", None)
    if not isinstance(ready, dict):
        ready = {}
        scanner._v252_registered_ready_shots = ready
    ready[int(shot_id)] = float(frame_ts)
    if len(ready) > 32:
        keep = sorted(ready.items(), key=lambda kv: kv[1])[-24:]
        scanner._v252_registered_ready_shots = dict(keep)


def _is_ready(scanner: Any, shot_id: int) -> bool:
    ready = getattr(scanner, "_v252_registered_ready_shots", {})
    return isinstance(ready, dict) and int(shot_id) in ready


def _install_settings_defaults() -> None:
    defaults = {
        "registered_freshness_enabled_v252": True,
        "registered_freshness_log_v252": True,
        "registered_freshness_search_radius_v252": 3,
        "registered_freshness_fail_open_s_v252": 1.10,
        "registered_freshness_min_center_abs_v252": 1.60,
        "registered_freshness_min_center_z_v252": 1.30,
        "registered_freshness_min_compact_v252": 0.30,
        "registered_freshness_min_dark_compact_v252": 0.24,
        "registered_freshness_min_dark_fraction_v252": 0.42,
        "registered_freshness_confirmed_per_group_v252": 2,
        "registered_freshness_confirmed_total_v252": 8,
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


def _install_registered_evidence_patch() -> None:
    from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2

    if getattr(CandidateGeneratorV2, "_v252_registered_freshness_patch", False):
        return
    previous_extract = CandidateGeneratorV2._extract_candidates
    previous_generate = CandidateGeneratorV2.generate

    def extract_v252(self, *args, **kwargs):
        scanner = kwargs.get("scanner")
        sid = _shot_id_from_scanner(scanner)
        live = threading.current_thread().name.startswith("shot-cv-v2224")
        rescue_before = bool(sid > 0 and rescue_router_v2225.requested(sid))
        result = previous_extract(self, *args, **kwargs)
        if not _enabled() or not live or sid <= 0 or rescue_before:
            return result
        arrays = (kwargs.get("absdiff"), kwargs.get("zscore"), kwargs.get("darkening"))
        bbox = kwargs.get("bbox")
        if not all(isinstance(x, np.ndarray) for x in arrays) or not (isinstance(bbox, tuple) and len(bbox) == 4):
            return result
        groups = _camera_regions_to_work_groups(scanner, sid)
        if not groups:
            return result
        model = getattr(self, "_shot_models", {}).get(sid, {})
        noise = model.get("temporal_noise") if isinstance(model, dict) else None
        if not isinstance(noise, np.ndarray) or noise.shape != arrays[0].shape:
            noise = None
        ctx = RegisteredFrameV252(
            shot_id=int(sid), frame_ts=_finite(kwargs.get("frame_ts", 0.0)),
            bbox=tuple(int(x) for x in bbox),
            absdiff=np.asarray(arrays[0]), zscore=np.asarray(arrays[1]),
            darkening=np.asarray(arrays[2]), temporal_noise=noise,
            groups=tuple(groups),
        )
        self._v252_last_registered_frame = ctx
        annotated = [_annotate_candidate(dict(c), ctx) for c in result]
        return _normalise_groups(annotated)

    def generate_v252(self, scanner, gray, frame_ts, legacy_candidates):
        sid = _shot_id_from_scanner(scanner)
        live = threading.current_thread().name.startswith("shot-cv-v2224")
        rescue_before = bool(sid > 0 and rescue_router_v2225.requested(sid))
        result = previous_generate(self, scanner, gray, frame_ts, legacy_candidates)
        if not _enabled() or not live or sid <= 0 or rescue_before:
            return result
        ctx = getattr(self, "_v252_last_registered_frame", None)
        if not isinstance(ctx, RegisteredFrameV252) or ctx.shot_id != sid:
            # This is normally CandidateGeneratorV2's waiting_post_peak path.
            try:
                result.telemetry = dict(result.telemetry or {})
                result.telemetry["v252_registered_ready"] = False
            except Exception:
                pass
            return result
        # Do not accidentally reuse a previous camera frame when generate() took
        # a non-registered early path in this call.
        if abs(float(ctx.frame_ts) - float(frame_ts)) > 0.12:
            return result

        annotated = [_annotate_candidate(dict(c), ctx) for c in list(result.candidates)]
        annotated = _normalise_groups(annotated)
        result.candidates = annotated
        _mark_ready(scanner, sid, frame_ts)
        fresh = [c for c in annotated if _finite(c.get("v252_fresh_physical", 0.0)) > 0.5]
        try:
            result.telemetry = dict(result.telemetry or {})
            result.telemetry["v252_registered_ready"] = True
            result.telemetry["v252_fresh_count"] = len(fresh)
            scanner.last_window_debug["v252_registered_ready"] = 1.0
            scanner.last_window_debug["v252_fresh_count"] = float(len(fresh))
        except Exception:
            pass

        if _log_enabled():
            by_group: dict[str, list[dict[str, Any]]] = {}
            for cand in annotated:
                group = str(cand.get("v251_region_group", ""))
                if group:
                    by_group.setdefault(group, []).append(cand)
            for group in sorted(by_group):
                values = by_group[group]
                fresh_values = [c for c in values if _finite(c.get("v252_fresh_physical", 0.0)) > 0.5]
                best = max(fresh_values or values, key=lambda c: _finite(c.get("v252_authority_score", c.get("v252_physical_score", 0.0))))
                print(
                    f"[V2.25.2 FRESHNESS] shot={sid} object={group} n={len(values)} fresh={len(fresh_values)} "
                    f"best={_finite(best.get('v252_authority_score', 0.0)):.2f} "
                    f"src={best.get('v252_authority_source', 'none')} "
                    f"center={_finite(best.get('v252_center_abs', 0.0)):.2f} ring={_finite(best.get('v252_ring_abs', 0.0)):.2f} "
                    f"compact={_finite(best.get('v252_compact_abs', 0.0)):.2f} z={_finite(best.get('v252_center_z', 0.0)):.2f} "
                    f"noise={_finite(best.get('v252_pre_noise', 0.0)):.2f} "
                    f"xy=({_finite(best.get('camera_x')):.1f},{_finite(best.get('camera_y')):.1f})"
                )
            print(f"[V2.25.2 REGISTERED-READY] shot={sid} candidates={len(annotated)} fresh={len(fresh)}")
        return result

    CandidateGeneratorV2._extract_candidates = extract_v252
    CandidateGeneratorV2.generate = generate_v252
    CandidateGeneratorV2._v252_registered_freshness_patch = True
    CandidateGeneratorV2._v252_previous_extract = previous_extract
    CandidateGeneratorV2._v252_previous_generate = previous_generate


def _install_confirmation_patch() -> None:
    import src.engine.shot_fast_v2225 as fast

    if getattr(fast, "_v252_registered_confirm_patch", False):
        return
    previous_confirm = fast.local_confirm_candidates_v2225

    def confirm_v252(pre_gray, current_gray, candidates, *, frame_ts, config=None):
        confirmed, diag = previous_confirm(
            pre_gray, current_gray, candidates, frame_ts=frame_ts, config=config
        )
        ready_candidates = [c for c in candidates if _finite(c.get("v252_evidence_ready", 0.0)) > 0.5]
        if not _enabled() or not ready_candidates:
            return confirmed, diag
        # V2.22.5 remains a second-frame persistence check, but its permissive
        # gate is no longer sufficient for object-context authority. Keep only
        # candidates that ALSO passed the registered immediate PRE->POST gate.
        fresh = [c for c in confirmed if _finite(c.get("v252_fresh_physical", 0.0)) > 0.5]
        balanced = _balance_fresh_confirmed(fresh)
        sid = int(_finite((ready_candidates[0] if ready_candidates else {}).get("v251_shot_id", 0.0)))
        if _log_enabled():
            best = max((_finite(c.get("v252_confirm_score", 0.0)) for c in balanced), default=0.0)
            print(
                f"[V2.25.2 REGISTERED-CONFIRM] shot={sid} tested={int(diag.get('tested', 0))} "
                f"old={len(confirmed)} registered_fresh={len(fresh)} balanced={len(balanced)} best={best:.2f}"
            )
        out_diag = dict(diag)
        out_diag["v252_old_confirmed"] = float(len(confirmed))
        out_diag["v252_registered_fresh"] = float(len(fresh))
        out_diag["v252_after_balance"] = float(len(balanced))
        return balanced, out_diag

    fast.local_confirm_candidates_v2225 = confirm_v252
    fast._v252_registered_confirm_patch = True
    fast._v252_previous_local_confirm = previous_confirm


def _install_authority_selector_patch() -> None:
    from src.engine.camera.hit_scanner import HitScanner

    if getattr(HitScanner, "_v252_registered_authority_selector", False):
        return
    previous_best = HitScanner._best_track_for_event
    # V2.25.1 stores the selector that existed before its local region authority.
    # The explicit FULL rescue must bypass BOTH V2.25.2 and V2.25.1 local
    # selectors so unannotated global-rescue tracks can actually win.
    base_physical_best = getattr(HitScanner, "_v251_previous_best_track", previous_best)

    def best_track_v252(self, event):
        sid = int(getattr(event, "shot_id", 0) or 0)
        snap = object_hit_registry_v2223.snapshot_for_shot(sid) if sid > 0 else None
        if not _enabled() or snap is None or not tuple(getattr(snap, "camera_regions", ()) or ()):
            return previous_best(self, event)

        # Once the explicit global rescue has actually run, V2.22.5 owns the
        # fallback authority exactly as before V2.25.x.
        if rescue_router_v2225.was_consumed(sid):
            return base_physical_best(self, event)

        age = max(0.0, time.time() - float(getattr(event, "peak_ts", time.time()) or time.time()))
        ready = _is_ready(self, sid)
        gate_log = getattr(self, "_v252_gate_log", None)
        if not isinstance(gate_log, set):
            gate_log = set()
            self._v252_gate_log = gate_log
        if not ready:
            if age < _authority_fail_open_s():
                if _log_enabled() and (sid, "wait") not in gate_log:
                    gate_log.add((sid, "wait"))
                    print(f"[V2.25.2 EARLY-GATE] shot={sid} age={age*1000.0:.0f}ms reason=await_registered_frame")
                return None
            if _log_enabled() and (sid, "failopen") not in gate_log:
                gate_log.add((sid, "failopen"))
                print(f"[V2.25.2 EARLY-GATE] shot={sid} age={age*1000.0:.0f}ms fail_open=1 reason=no_registered_frame")
            return base_physical_best(self, event)

        eligible: list[tuple[Any, dict[str, Any], float]] = []
        for track in getattr(self, "_active_tracks", {}).values():
            onset_dt = float(track.first_seen_ts) - float(event.peak_ts)
            if onset_dt < -float(self.association_lead_s) or onset_dt > float(self.association_lag_s):
                continue
            if track.emitted and event.matched_track_id != track.track_id:
                continue
            cand = getattr(track, "last_candidate", {}) or {}
            if _finite(cand.get("v252_fresh_physical", 0.0)) <= 0.5:
                continue
            score = _finite(cand.get("v252_confirm_score", cand.get("v252_authority_score", 0.0)))
            if score <= 0.0:
                continue
            eligible.append((track, cand, onset_dt))

        if not eligible:
            # Registered evidence has run but no authority-worthy track exists.
            # Keep the event pending so V2.22.5 can queue/consume its one global
            # full rescue. Bounded fail-open prevents a pathological deadlock.
            if age < _authority_fail_open_s():
                return None
            if _log_enabled() and (sid, "fresh_failopen") not in gate_log:
                gate_log.add((sid, "fresh_failopen"))
                print(f"[V2.25.2 AUTHORITY] shot={sid} fresh_tracks=0 age={age*1000.0:.0f}ms fail_open=1")
            return base_physical_best(self, event)

        best, cand, onset = max(
            eligible,
            key=lambda item: (
                _finite(item[1].get("v252_confirm_score", item[1].get("v252_authority_score", 0.0))),
                _finite(item[1].get("v252_group_excess", 0.0)),
                _finite(item[1].get("v252_physical_score", 0.0)),
                -abs(float(item[2])),
            ),
        )
        try:
            self.last_event_debug["v252_selector"] = "registered_fresh_physical"
            self.last_event_debug["v252_authority_score"] = _finite(cand.get("v252_authority_score", 0.0))
            self.last_event_debug["v252_source"] = str(cand.get("v252_authority_source", ""))
        except Exception:
            pass
        if _log_enabled() and (sid, "selected", int(getattr(best, "track_id", 0) or 0)) not in gate_log:
            gate_log.add((sid, "selected", int(getattr(best, "track_id", 0) or 0)))
            print(
                f"[V2.25.2 AUTHORITY] shot={sid} fresh_tracks={len(eligible)} "
                f"group={cand.get('v251_region_group', '')} src={cand.get('v252_authority_source', '')} "
                f"score={_finite(cand.get('v252_confirm_score', cand.get('v252_authority_score', 0.0))):.2f} "
                f"xy=({_finite(cand.get('camera_x')):.1f},{_finite(cand.get('camera_y')):.1f})"
            )
        return best

    HitScanner._best_track_for_event = best_track_v252
    HitScanner._v252_registered_authority_selector = True
    HitScanner._v252_previous_best_track = previous_best


def install_v252_runtime(AppClass: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_settings_defaults()
    _install_registered_evidence_patch()
    _install_confirmation_patch()
    _install_authority_selector_patch()
    AppClass._v252_registered_freshness_patch = True
    _INSTALLED = True
    print(
        "[V2.25.2] registered PRE->POST freshness authority + early-legacy gate installed "
        f"(search=±{_search_radius()}px, fail-open={_authority_fail_open_s():.2f}s, global rescue preserved)"
    )


__all__ = [
    "SCHEMA_VERSION",
    "PATCH_REVISION",
    "RegisteredMetricsV252",
    "RegisteredFrameV252",
    "_metric_quality",
    "_fresh_gate",
    "_normalise_groups",
    "_balance_fresh_confirmed",
    "install_v252_runtime",
]
