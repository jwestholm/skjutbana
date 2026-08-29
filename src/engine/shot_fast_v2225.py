"""V2.22.5 fast proposal + local confirmation.

This module is an additive runtime patch installed by ``main.py`` after
V2.22.3 and V2.22.4.

Design goals
------------
* keep V2.22.4's async worker / responsive game thread;
* run at most one expensive global proposal pass for the normal hit path;
* replace repeated full-frame/ROI persistence passes with cheap local PRE->POST
  confirmation around the already proposed candidates;
* use a sparse local-max FAST extractor in the live V2 worker, while retaining
  the existing high-recall extractor as an explicit rescue path;
* never hard-reject a re-hit/hole-in-hole merely because a known hole is nearby;
* ignore audio peaks while HitScanner is not ACTIVE (startup/calibration noise);
* discard/cancel queued work for shots that are already terminal.

The full V2 detector remains available.  Offline/F2 code and AI authority modes
are not switched to this FAST extractor by this patch: the fast extractor is
only selected for the V2.22.4 live CV worker.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

SCHEMA_VERSION = "2.22.5"
PATCH_REVISION = "r1"
_INSTALLED = False


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _odd(value: int, minimum: int = 1) -> int:
    value = max(int(minimum), int(value))
    return value if value % 2 else value + 1


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


# ---------------------------------------------------------------------------
# Runtime configuration cached at installation.  No lazy settings import is
# allowed to become part of the per-candidate local confirmation loop.
# ---------------------------------------------------------------------------


@dataclass
class FastConfigV2225:
    fast_extract_enabled: bool = True
    fast_max_raw_primary: int = 520
    fast_max_raw_rescue: int = 300
    fast_max_candidates: int = 180
    fast_rescue_if_primary_below: int = 90
    fast_temporal_rescue_enabled: bool = True
    fast_log: bool = True

    local_confirm_enabled: bool = True
    local_confirm_max_candidates: int = 200
    local_patch_radius_px: int = 11
    local_search_radius_px: int = 4
    local_min_center_abs: float = 1.65
    local_min_compact: float = 0.45
    local_min_peak_abs: float = 3.4
    local_min_darkening: float = 1.30
    local_min_frame_gap_ms: float = 18.0
    local_max_rounds: int = 2
    local_log: bool = True

    rescue_enabled: bool = True
    rescue_timeout_s: float = 3.5


_CONFIG = FastConfigV2225()


def _load_config_from_runtime() -> None:
    global _CONFIG
    _CONFIG = FastConfigV2225(
        fast_extract_enabled=_setting_bool("fast_extract_enabled_v2225", True),
        fast_max_raw_primary=_setting_int("fast_extract_max_raw_primary_v2225", 520, 80, 4000),
        fast_max_raw_rescue=_setting_int("fast_extract_max_raw_rescue_v2225", 300, 40, 3000),
        fast_max_candidates=_setting_int("fast_extract_max_candidates_v2225", 180, 20, 400),
        fast_rescue_if_primary_below=_setting_int("fast_extract_temporal_rescue_below_v2225", 90, 0, 400),
        fast_temporal_rescue_enabled=_setting_bool("fast_extract_temporal_rescue_v2225", True),
        fast_log=_setting_bool("fast_extract_log_v2225", True),
        local_confirm_enabled=_setting_bool("local_confirm_enabled_v2225", True),
        local_confirm_max_candidates=_setting_int("local_confirm_max_candidates_v2225", 200, 8, 400),
        local_patch_radius_px=_setting_int("local_confirm_patch_radius_px_v2225", 11, 6, 24),
        local_search_radius_px=_setting_int("local_confirm_search_radius_px_v2225", 4, 0, 8),
        local_min_center_abs=_setting_float("local_confirm_min_center_abs_v2225", 1.65, 0.1, 30.0),
        local_min_compact=_setting_float("local_confirm_min_compact_v2225", 0.45, -5.0, 30.0),
        local_min_peak_abs=_setting_float("local_confirm_min_peak_abs_v2225", 3.4, 0.2, 100.0),
        local_min_darkening=_setting_float("local_confirm_min_darkening_v2225", 1.30, 0.0, 30.0),
        local_min_frame_gap_ms=_setting_float("local_confirm_min_frame_gap_ms_v2225", 18.0, 0.0, 300.0),
        local_max_rounds=_setting_int("local_confirm_max_rounds_v2225", 2, 1, 4),
        local_log=_setting_bool("local_confirm_log_v2225", True),
        rescue_enabled=_setting_bool("full_rescue_enabled_v2225", True),
        rescue_timeout_s=_setting_float("full_rescue_timeout_s_v2225", 3.5, 2.0, 8.0),
    )


# ---------------------------------------------------------------------------
# Full-rescue routing.  The existing high-recall extractor is not deleted; a
# shot can request exactly one full rescue if local confirmation cannot verify
# any FAST/legacy proposal.
# ---------------------------------------------------------------------------


class RescueRouterV2225:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._requested: set[int] = set()
        self._consumed: set[int] = set()

    def request(self, shot_id: int) -> bool:
        sid = int(shot_id or 0)
        if sid <= 0:
            return False
        with self._lock:
            if sid in self._requested or sid in self._consumed:
                return False
            self._requested.add(sid)
            return True

    def consume(self, shot_id: int) -> bool:
        sid = int(shot_id or 0)
        with self._lock:
            if sid not in self._requested:
                return False
            self._requested.discard(sid)
            self._consumed.add(sid)
            return True

    def requested(self, shot_id: int) -> bool:
        with self._lock:
            return int(shot_id or 0) in self._requested

    def was_consumed(self, shot_id: int) -> bool:
        with self._lock:
            return int(shot_id or 0) in self._consumed

    def clear(self, shot_id: int) -> None:
        sid = int(shot_id or 0)
        with self._lock:
            self._requested.discard(sid)
            self._consumed.discard(sid)

    def reset(self) -> None:
        with self._lock:
            self._requested.clear()
            self._consumed.clear()


rescue_router_v2225 = RescueRouterV2225()


def _shot_id_from_scanner(scanner: Any) -> int:
    pending = [
        ev for ev in list(getattr(scanner, "audio_events", []) or [])
        if str(getattr(ev, "state", "")) == "pending"
    ]
    if not pending:
        return 0
    ev = min(pending, key=lambda item: float(getattr(item, "peak_ts", 0.0) or 0.0))
    return int(getattr(ev, "shot_id", 0) or 0)


# ---------------------------------------------------------------------------
# Sparse FAST V2 extractor
# ---------------------------------------------------------------------------


def _top_sparse_peaks(
    score_map: np.ndarray,
    mask: np.ndarray,
    *,
    kernel: int,
    limit: int,
) -> list[tuple[float, int, int]]:
    """Return strongest sparse local maxima without connected components.

    Plateaus may yield adjacent equal maxima.  That is intentional here: exact
    plateau de-duplication is deferred to the cheap coordinate/grid NMS below,
    avoiding a megapixel connectedComponentsWithStats pass.
    """
    if score_map.size == 0 or not np.any(mask):
        return []
    k = _odd(kernel, 3)
    dilated = cv2.dilate(
        score_map,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
    )
    peak_mask = mask & (score_map >= (dilated - 1e-6))
    ys, xs = np.nonzero(peak_mask)
    if ys.size == 0:
        return []
    scores = score_map[ys, xs].astype(np.float32, copy=False)
    take = min(int(limit), int(scores.size))
    if take <= 0:
        return []
    if scores.size > take:
        idx = np.argpartition(scores, -take)[-take:]
        ys = ys[idx]
        xs = xs[idx]
        scores = scores[idx]
    order = np.argsort(scores)[::-1]
    return [(float(scores[i]), int(xs[i]), int(ys[i])) for i in order]


def _grid_nms_and_quota(
    peaks: list[tuple[float, int, int, set[str]]],
    *,
    shape: tuple[int, int],
    nms_radius: float,
    cols: int,
    rows: int,
    per_tile: int,
    global_extra: int,
    max_candidates: int,
) -> list[tuple[float, int, int, set[str]]]:
    if not peaks:
        return []
    h, w = int(shape[0]), int(shape[1])
    cell = max(1.0, float(nms_radius))
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    chosen: list[tuple[float, int, int, set[str]]] = []
    tile_counts: dict[tuple[int, int], int] = defaultdict(int)
    tile_w = max(1.0, w / float(max(1, cols)))
    tile_h = max(1.0, h / float(max(1, rows)))
    radius_sq = float(nms_radius) * float(nms_radius)

    def far_enough(px: int, py: int) -> bool:
        gx = int(px / cell)
        gy = int(py / cell)
        for yy in range(gy - 1, gy + 2):
            for xx in range(gx - 1, gx + 2):
                for cx, cy in buckets.get((xx, yy), ()):  # tiny list
                    if float(px - cx) ** 2 + float(py - cy) ** 2 < radius_sq:
                        return False
        return True

    def add(peak: tuple[float, int, int, set[str]]) -> bool:
        _score, px, py, _sources = peak
        if not far_enough(px, py):
            return False
        chosen.append(peak)
        buckets[(int(px / cell), int(py / cell))].append((px, py))
        return True

    # Pass 1: spatial coverage, matching the intent of the full extractor.
    for peak in peaks:
        _score, px, py, _sources = peak
        tile = (
            min(max(1, cols) - 1, int(px / tile_w)),
            min(max(1, rows) - 1, int(py / tile_h)),
        )
        if tile_counts[tile] >= per_tile:
            continue
        if add(peak):
            tile_counts[tile] += 1
        if len(chosen) >= max_candidates:
            break

    # Pass 2: strongest global extras.
    extras = 0
    if len(chosen) < max_candidates and global_extra > 0:
        for peak in peaks:
            if extras >= global_extra or len(chosen) >= max_candidates:
                break
            if add(peak):
                extras += 1
    return chosen


def fast_extract_candidates_v2225(
    generator: Any,
    *,
    scanner: Any,
    saliency: np.ndarray,
    absdiff: np.ndarray,
    darkening: np.ndarray,
    dog: np.ndarray,
    zscore: np.ndarray,
    valid: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_ts: float,
    threshold: float,
    cfg: dict[str, Any],
) -> list[dict[str, float]]:
    """Fast live extractor preserving the full extractor's candidate schema."""
    started = time.perf_counter()
    x0, y0, _, _ = bbox
    min_change = max(0.0, _finite(cfg.get("min_temporal_change", 1.8), 1.8))
    min_z = max(0.0, _finite(cfg.get("min_zscore", 1.5), 1.5))
    strong_change = max(min_change, _finite(cfg.get("strong_temporal_change", 4.0), 4.0))
    primary_evidence = (absdiff >= strong_change) | ((absdiff >= min_change) & (zscore >= min_z))
    primary_mask = valid & primary_evidence & (saliency >= float(threshold))

    primary_kernel = _odd(_safe_int(cfg.get("local_max_kernel", 3), 3), 3)
    primary = _top_sparse_peaks(
        saliency,
        primary_mask,
        kernel=primary_kernel,
        limit=_CONFIG.fast_max_raw_primary,
    )

    # coordinate -> [score, sources]
    merged: dict[tuple[int, int], tuple[float, set[str]]] = {}
    for score, px, py in primary:
        merged[(px, py)] = (score, {"primary"})

    rescue_count = 0
    if (
        _CONFIG.fast_temporal_rescue_enabled
        and len(primary) < _CONFIG.fast_rescue_if_primary_below
        and np.any(valid)
    ):
        # Keep one independent temporal rescue, but deliberately skip the full
        # extractor's connected-component blob rescue and second saliency CC
        # path. This is the main live latency saving.
        temporal_map = (
            absdiff * (1.0 + 0.55 * np.clip(zscore, 0.0, 6.0))
            + 0.35 * np.maximum(dog, 0.0)
        ).astype(np.float32)
        values = temporal_map[valid]
        if values.size > 120_000:
            stride = max(1, values.size // 120_000)
            values = values[::stride]
        med = float(np.median(values)) if values.size else 0.0
        mad = float(np.median(np.abs(values - med))) if values.size else 0.0
        sigma = max(0.0, _finite(cfg.get("rescue_temporal_robust_sigma", 1.75), 1.75))
        minimum = max(0.0, _finite(cfg.get("rescue_temporal_min_score", 4.2), 4.2))
        temporal_threshold = max(minimum, med + sigma * 1.4826 * mad)
        rescue_change = max(0.0, _finite(cfg.get("rescue_min_temporal_change", 1.8), 1.8))
        rescue_z = max(0.0, _finite(cfg.get("rescue_min_zscore", 1.25), 1.25))
        rescue_strong = max(rescue_change, _finite(cfg.get("rescue_strong_temporal_change", 4.0), 4.0))
        rescue_evidence = (absdiff >= rescue_strong) | ((absdiff >= rescue_change) & (zscore >= rescue_z))
        temporal_mask = valid & rescue_evidence & (temporal_map >= temporal_threshold)
        rescue = _top_sparse_peaks(
            temporal_map,
            temporal_mask,
            kernel=_odd(_safe_int(cfg.get("rescue_local_max_kernel", 3), 3), 3),
            limit=_CONFIG.fast_max_raw_rescue,
        )
        rescue_count = len(rescue)
        for score, px, py in rescue:
            key = (px, py)
            old = merged.get(key)
            if old is None:
                merged[key] = (score, {"rescue_temporal"})
            else:
                merged[key] = (max(float(old[0]), score), set(old[1]) | {"rescue_temporal"})

    if not merged:
        scanner.last_window_debug["v2225_fast_extract_ms"] = (time.perf_counter() - started) * 1000.0
        scanner.last_window_debug["v2225_fast_primary"] = float(len(primary))
        scanner.last_window_debug["v2225_fast_rescue"] = float(rescue_count)
        scanner.last_window_debug["v2225_fast_output"] = 0.0
        return []

    raw = [(score, px, py, sources) for (px, py), (score, sources) in merged.items()]
    raw.sort(key=lambda item: item[0], reverse=True)

    # Refine only a bounded sparse list.  The full extractor refines every CC
    # maximum/rescue component before quota/NMS; doing it after top-K pruning is
    # much cheaper and still bounded to a few pixels.
    refine_limit = max(_CONFIG.fast_max_candidates * 2, 220)
    refined_map: dict[tuple[int, int], tuple[float, set[str], float]] = {}
    for score, px, py, sources in raw[:refine_limit]:
        try:
            rx, ry, shift = generator._refine_peak(
                px=px, py=py, absdiff=absdiff, zscore=zscore,
                dog=dog, valid=valid, cfg=cfg,
            )
        except Exception:
            rx, ry, shift = px, py, 0.0
        key = (int(rx), int(ry))
        old = refined_map.get(key)
        if old is None:
            refined_map[key] = (float(score), set(sources), float(shift))
        else:
            refined_map[key] = (
                max(float(old[0]), float(score)),
                set(old[1]) | set(sources),
                min(float(old[2]), float(shift)),
            )

    refined = [
        (score, px, py, sources, shift)
        for (px, py), (score, sources, shift) in refined_map.items()
    ]
    refined.sort(key=lambda item: item[0], reverse=True)

    simple_peaks = [(s, x, y, src) for s, x, y, src, _shift in refined]
    chosen_simple = _grid_nms_and_quota(
        simple_peaks,
        shape=saliency.shape,
        nms_radius=max(0.5, _finite(cfg.get("nms_radius_px", 3.5), 3.5)),
        cols=max(1, _safe_int(cfg.get("tile_columns", 8), 8)),
        rows=max(1, _safe_int(cfg.get("tile_rows", 6), 6)),
        per_tile=max(1, _safe_int(cfg.get("per_tile_candidates", 7), 7)),
        global_extra=max(0, _safe_int(cfg.get("global_extra_candidates", 100), 100)),
        max_candidates=min(
            max(1, _safe_int(cfg.get("max_v2_candidates", 220), 220)),
            _CONFIG.fast_max_candidates,
        ),
    )
    shift_by_xy = {(x, y): shift for _s, x, y, _src, shift in refined}

    candidates: list[dict[str, float]] = []
    for peak_saliency, px, py, sources in chosen_simple:
        try:
            features = generator._candidate_features(
                px=px, py=py, saliency=saliency, absdiff=absdiff,
                darkening=darkening, dog=dog, zscore=zscore,
            )
        except Exception:
            continue
        candidate: dict[str, float] = {
            "camera_x": float(px + x0),
            "camera_y": float(py + y0),
            "area": float(features.get("area", 1.0)),
            "radius": float(features.get("radius", 1.0)),
            "circularity": float(features.get("circularity", 0.75)),
            "score": float(features.get("score", 3.6)),
            "center_darkening": float(features.get("center_change", 0.0)),
            "local_contrast_gain": float(features.get("local_contrast", 0.0)),
            "blackhat_value": float(features.get("dog_value", 0.0)),
            "change_value": float(features.get("center_change", 0.0)),
            "pre_shot_change": float(features.get("center_change", 0.0)),
            "timestamp": float(frame_ts),
            "detector_v2": 1.0,
            "detector_v1": 0.0,
            "v2_saliency": float(peak_saliency),
            "v2_zscore": float(features.get("zscore", 0.0)),
            "v2_absdiff": float(features.get("absdiff", 0.0)),
            "v2_darkening": float(features.get("darkening", 0.0)),
            "v2_dog": float(features.get("dog_value", 0.0)),
            "v2_primary_peak": 1.0 if "primary" in sources else 0.0,
            "v2_rescue_saliency": 0.0,
            "v2_rescue_temporal": 1.0 if "rescue_temporal" in sources else 0.0,
            "v2_rescue_blob": 0.0,
            "v2_refine_shift_px": float(shift_by_xy.get((px, py), 0.0)),
            "v2225_fast_extract": 1.0,
        }
        try:
            generator._apply_known_hole_penalty(scanner, candidate)
        except Exception:
            pass
        candidates.append(candidate)

    candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
    elapsed = (time.perf_counter() - started) * 1000.0
    scanner.last_window_debug["v2225_fast_extract_ms"] = float(elapsed)
    scanner.last_window_debug["v2225_fast_primary"] = float(len(primary))
    scanner.last_window_debug["v2225_fast_rescue"] = float(rescue_count)
    scanner.last_window_debug["v2225_fast_output"] = float(len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Local confirmation
# ---------------------------------------------------------------------------


@dataclass
class ConfirmMetricsV2225:
    center_abs: float
    ring_abs: float
    compact: float
    peak_abs: float
    darkening: float
    best_dx: int
    best_dy: int


@dataclass
class ShotConfirmStateV2225:
    shot_id: int
    first_result_ts: float
    candidates: list[dict[str, Any]]
    pre_gray: np.ndarray | None
    rounds: int = 0
    rescue_round: bool = False
    phase: str = "await_confirm"
    created_mono: float = field(default_factory=time.perf_counter)


def _candidate_metrics_local(
    pre: np.ndarray,
    current: np.ndarray,
    camera_x: float,
    camera_y: float,
    *,
    patch_radius: int,
    search_radius: int,
) -> ConfirmMetricsV2225 | None:
    if pre.shape != current.shape or pre.ndim != 2 or current.ndim != 2:
        return None
    h, w = pre.shape
    cx = int(round(float(camera_x)))
    cy = int(round(float(camera_y)))
    margin = patch_radius + search_radius + 2
    x0 = max(0, cx - margin)
    x1 = min(w, cx + margin + 1)
    y0 = max(0, cy - margin)
    y1 = min(h, cy + margin + 1)
    if x1 - x0 < 9 or y1 - y0 < 9:
        return None

    p = pre[y0:y1, x0:x1].astype(np.float32)
    q = current[y0:y1, x0:x1].astype(np.float32)
    diff = np.abs(p - q)
    # 3x3 mean cheaply locates the compact persistent change within a few px of
    # the proposal without moving the authoritative candidate coordinate.
    smooth = cv2.blur(diff, (3, 3))
    lx = cx - x0
    ly = cy - y0
    sx0 = max(0, lx - search_radius)
    sx1 = min(smooth.shape[1], lx + search_radius + 1)
    sy0 = max(0, ly - search_radius)
    sy1 = min(smooth.shape[0], ly + search_radius + 1)
    search = smooth[sy0:sy1, sx0:sx1]
    if search.size == 0:
        return None
    flat = int(np.argmax(search))
    dy, dx = np.unravel_index(flat, search.shape)
    bx = int(sx0 + dx)
    by = int(sy0 + dy)

    yy, xx = np.ogrid[:diff.shape[0], :diff.shape[1]]
    d2 = (xx - bx) ** 2 + (yy - by) ** 2
    center = d2 <= 4.0               # r <= 2
    ring = (d2 >= 16.0) & (d2 <= 64.0)  # r 4..8
    if not np.any(center):
        return None
    center_abs = float(np.mean(diff[center]))
    ring_abs = float(np.mean(diff[ring])) if np.any(ring) else 0.0
    compact = float(center_abs - ring_abs)
    peak_abs = float(np.max(diff[center]))
    dark = np.maximum(p - q, 0.0)
    darkening = float(np.mean(dark[center]))
    return ConfirmMetricsV2225(
        center_abs=center_abs,
        ring_abs=ring_abs,
        compact=compact,
        peak_abs=peak_abs,
        darkening=darkening,
        best_dx=int(bx - lx),
        best_dy=int(by - ly),
    )


def local_confirm_candidates_v2225(
    pre_gray: np.ndarray | None,
    current_gray: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    frame_ts: float,
    config: FastConfigV2225 | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    cfg = config or _CONFIG
    started = time.perf_counter()
    if pre_gray is None or not isinstance(pre_gray, np.ndarray):
        return [], {"tested": 0.0, "confirmed": 0.0, "ms": 0.0, "reason": 1.0}

    ordered = sorted(
        (dict(c) for c in candidates),
        key=lambda c: (
            _finite(c.get("score", 0.0)),
            _finite(c.get("pre_shot_change", 0.0)),
        ),
        reverse=True,
    )[: cfg.local_confirm_max_candidates]

    confirmed: list[dict[str, Any]] = []
    for candidate in ordered:
        metrics = _candidate_metrics_local(
            pre_gray,
            current_gray,
            _finite(candidate.get("camera_x", 0.0)),
            _finite(candidate.get("camera_y", 0.0)),
            patch_radius=cfg.local_patch_radius_px,
            search_radius=cfg.local_search_radius_px,
        )
        if metrics is None:
            continue

        # A valid new/re-hit change is compact relative to its ring OR has a
        # sufficiently strong centre/peak.  Old holes with no new temporal
        # change have centre_abs ~= ring_abs ~= 0 and fail naturally.
        temporal_ok = metrics.center_abs >= cfg.local_min_center_abs
        compact_ok = metrics.compact >= cfg.local_min_compact
        strong_peak = metrics.peak_abs >= cfg.local_min_peak_abs
        dark_ok = metrics.darkening >= cfg.local_min_darkening
        ok = temporal_ok and (compact_ok or strong_peak or dark_ok)
        if not ok:
            continue

        out = dict(candidate)
        out["timestamp"] = float(frame_ts)
        out["v2225_local_confirm"] = 1.0
        out["v2225_confirm_center_abs"] = float(metrics.center_abs)
        out["v2225_confirm_ring_abs"] = float(metrics.ring_abs)
        out["v2225_confirm_compact"] = float(metrics.compact)
        out["v2225_confirm_peak_abs"] = float(metrics.peak_abs)
        out["v2225_confirm_darkening"] = float(metrics.darkening)
        out["v2225_confirm_best_dx"] = float(metrics.best_dx)
        out["v2225_confirm_best_dy"] = float(metrics.best_dy)
        # Confirmation is evidence for the *same* physical location; do not
        # interpolate/move XY. Give tracking/ranking a bounded score bonus only.
        bonus = max(0.0, min(5.0, 0.35 * metrics.center_abs + 0.60 * max(0.0, metrics.compact)))
        out["score"] = float(_finite(out.get("score", 0.0)) + bonus)
        confirmed.append(out)

    confirmed.sort(key=lambda c: _finite(c.get("score", 0.0)), reverse=True)
    elapsed = (time.perf_counter() - started) * 1000.0
    return confirmed, {
        "tested": float(len(ordered)),
        "confirmed": float(len(confirmed)),
        "ms": float(elapsed),
    }


class LocalConfirmManagerV2225:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.states: dict[int, ShotConfirmStateV2225] = {}

    def start(self, shot_id: int, frame_ts: float, candidates: list[dict[str, Any]], pre_gray: np.ndarray | None, *, rescue_round: bool = False) -> None:
        sid = int(shot_id or 0)
        if sid <= 0:
            return
        with self._lock:
            self.states[sid] = ShotConfirmStateV2225(
                shot_id=sid,
                first_result_ts=float(frame_ts),
                candidates=[dict(c) for c in candidates],
                pre_gray=pre_gray,
                rounds=0,
                rescue_round=bool(rescue_round),
                phase="await_confirm",
            )

    def get(self, shot_id: int) -> ShotConfirmStateV2225 | None:
        with self._lock:
            return self.states.get(int(shot_id or 0))

    def active_waiting(self, scanner: Any, frame_ts: float) -> ShotConfirmStateV2225 | None:
        pending_ids = {
            int(getattr(ev, "shot_id", 0) or 0)
            for ev in list(getattr(scanner, "audio_events", []) or [])
            if str(getattr(ev, "state", "")) == "pending"
        }
        gap_s = _CONFIG.local_min_frame_gap_ms / 1000.0
        with self._lock:
            waiting = [
                st for sid, st in self.states.items()
                if sid in pending_ids
                and st.phase == "await_confirm"
                and float(frame_ts) >= st.first_result_ts + gap_s
            ]
        if not waiting:
            return None
        waiting.sort(key=lambda st: st.first_result_ts)
        return waiting[0]

    def mark_rescue_queued(self, shot_id: int) -> None:
        with self._lock:
            st = self.states.get(int(shot_id or 0))
            if st is not None:
                st.phase = "rescue_queued"

    def mark_exhausted(self, shot_id: int) -> None:
        with self._lock:
            st = self.states.get(int(shot_id or 0))
            if st is not None:
                st.phase = "exhausted"

    def clear(self, shot_id: int) -> None:
        with self._lock:
            self.states.pop(int(shot_id or 0), None)
        rescue_router_v2225.clear(shot_id)

    def reset(self) -> None:
        with self._lock:
            self.states.clear()
        rescue_router_v2225.reset()


local_confirm_manager_v2225 = LocalConfirmManagerV2225()


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------


def _install_settings_defaults() -> None:
    defaults = {
        "fast_extract_enabled_v2225": True,
        "fast_extract_max_raw_primary_v2225": 520,
        "fast_extract_max_raw_rescue_v2225": 300,
        "fast_extract_max_candidates_v2225": 180,
        "fast_extract_temporal_rescue_below_v2225": 90,
        "fast_extract_temporal_rescue_v2225": True,
        "fast_extract_log_v2225": True,
        "local_confirm_enabled_v2225": True,
        "local_confirm_max_candidates_v2225": 200,
        "local_confirm_patch_radius_px_v2225": 11,
        "local_confirm_search_radius_px_v2225": 4,
        "local_confirm_min_center_abs_v2225": 1.65,
        "local_confirm_min_compact_v2225": 0.45,
        "local_confirm_min_peak_abs_v2225": 3.4,
        "local_confirm_min_darkening_v2225": 1.30,
        "local_confirm_min_frame_gap_ms_v2225": 18.0,
        "local_confirm_max_rounds_v2225": 2,
        "local_confirm_log_v2225": True,
        "full_rescue_enabled_v2225": True,
        "full_rescue_timeout_s_v2225": 3.5,
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


def _install_fast_extractor_patch() -> None:
    from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
    if getattr(CandidateGeneratorV2, "_v2225_fast_extract_patch", False):
        return
    previous_extract = CandidateGeneratorV2._extract_candidates
    CandidateGeneratorV2._v2225_full_extract_original = previous_extract

    def extract_v2225(self, *args, **kwargs):
        scanner = kwargs.get("scanner")
        sid = _shot_id_from_scanner(scanner) if scanner is not None else 0
        live_worker = threading.current_thread().name.startswith("shot-cv-v2224")
        if not live_worker or not _CONFIG.fast_extract_enabled:
            return previous_extract(self, *args, **kwargs)
        if sid > 0 and rescue_router_v2225.consume(sid):
            if _CONFIG.fast_log:
                print(f"[V2.22.5 FULL-RESCUE] shot={sid} using high-recall extractor")
            result = previous_extract(self, *args, **kwargs)
            try:
                scanner.last_window_debug["v2225_extract_mode"] = 2.0
            except Exception:
                pass
            return result
        result = fast_extract_candidates_v2225(self, *args, **kwargs)
        try:
            scanner.last_window_debug["v2225_extract_mode"] = 1.0
        except Exception:
            pass
        return result

    CandidateGeneratorV2._extract_candidates = extract_v2225
    CandidateGeneratorV2._v2225_fast_extract_patch = True


def _install_pre_by_shot_patch() -> None:
    from src.engine.camera.hit_scanner import HitScanner
    if getattr(HitScanner, "_v2225_pre_by_shot_patch", False):
        return
    previous_on_peak = HitScanner._on_audio_peak
    previous_disable = HitScanner.disable

    def on_peak_v2225(self, ev):
        expected_sid = int(getattr(self, "_next_shot_id", 0) or 0)
        result = previous_on_peak(self, ev)
        if expected_sid > 0:
            if not hasattr(self, "_v2225_pre_by_shot"):
                self._v2225_pre_by_shot = {}
            pre = getattr(self, "pre_shot_snapshot", None)
            if isinstance(pre, np.ndarray):
                # V2.22.3 creates a fresh array for each event. Keep that object
                # as the immutable shot-local PRE; no extra 4K copy required.
                self._v2225_pre_by_shot[expected_sid] = pre
            # Bound memory even if diagnostics retain old AudioShotEvents.
            keep = sorted(self._v2225_pre_by_shot)[-6:]
            self._v2225_pre_by_shot = {sid: self._v2225_pre_by_shot[sid] for sid in keep}
        return result

    def disable_v2225(self):
        try:
            return previous_disable(self)
        finally:
            self._v2225_pre_by_shot = {}
            local_confirm_manager_v2225.reset()

    HitScanner._on_audio_peak = on_peak_v2225
    HitScanner.disable = disable_v2225
    HitScanner._v2225_pre_by_shot_patch = True


def _install_result_marker_patch() -> None:
    from src.engine.shot_async_v2224 import AsyncDetectorV2224
    if getattr(AsyncDetectorV2224, "_v2225_result_marker_patch", False):
        return
    previous_apply = AsyncDetectorV2224.apply_result

    def apply_result_v2225(scanner: Any, result: Any) -> None:
        previous_apply(scanner, result)
        scanner._v2225_result_shot_id = int(getattr(result, "shot_id", 0) or 0)
        scanner._v2225_result_frame_ts = float(getattr(result, "frame_ts", 0.0) or 0.0)
        debug = dict(getattr(result, "window_debug", {}) or {})
        if _CONFIG.fast_log and "v2225_fast_extract_ms" in debug:
            print(
                f"[V2.22.5 FAST-EXTRACT] shot={scanner._v2225_result_shot_id} "
                f"extract={_finite(debug.get('v2225_fast_extract_ms')):.1f}ms "
                f"primary={int(_finite(debug.get('v2225_fast_primary')))} "
                f"temporal={int(_finite(debug.get('v2225_fast_rescue')))} "
                f"v2out={int(_finite(debug.get('v2225_fast_output')))}"
            )

    AsyncDetectorV2224.apply_result = staticmethod(apply_result_v2225)
    AsyncDetectorV2224._v2225_result_marker_patch = True


def _cancel_queued_for_shot(shot_id: int) -> int:
    try:
        from src.engine.shot_async_v2224 import async_detector_v2224
    except Exception:
        return 0
    sid = int(shot_id or 0)
    cancelled = 0
    with async_detector_v2224._lock:
        for future, (fsid, _fts) in list(async_detector_v2224._futures.items()):
            if int(fsid) != sid:
                continue
            if future.cancel():
                async_detector_v2224._futures.pop(future, None)
                cancelled += 1
        # Drop already-completed/ready results for terminal shots.
        async_detector_v2224._ready = type(async_detector_v2224._ready)(
            item for item in async_detector_v2224._ready if int(getattr(item, "shot_id", 0)) != sid
        )
        async_detector_v2224._submitted_frame_ts.pop(sid, None)
        async_detector_v2224._last_applied_frame_ts.pop(sid, None)
    return cancelled


def _install_local_confirmation_patch() -> None:
    from src.engine.camera.hit_scanner import HitScanner
    if getattr(HitScanner, "_v2225_local_confirmation_patch", False):
        return
    previous_detect = HitScanner._detect_frame_candidates  # V2.22.4 async wrapper
    previous_resolve = HitScanner._resolve_audio_events

    def detect_v2225(self, gray: np.ndarray, frame_ts: float):
        if not _CONFIG.local_confirm_enabled:
            return previous_detect(self, gray, frame_ts)

        # Prefer local confirmation for a shot whose first global result has
        # already seeded ordinary HitScanner tracks. This bypasses V2.22.4's
        # second full detector submission entirely.
        state = local_confirm_manager_v2225.active_waiting(self, frame_ts)
        if state is not None:
            state.rounds += 1
            confirmed, diag = local_confirm_candidates_v2225(
                state.pre_gray,
                gray,
                state.candidates,
                frame_ts=float(frame_ts),
            )
            if _CONFIG.local_log:
                print(
                    f"[V2.22.5 LOCAL-CONFIRM] shot={state.shot_id} round={state.rounds} "
                    f"tested={int(diag.get('tested', 0))} confirmed={len(confirmed)} "
                    f"time={diag.get('ms', 0.0):.1f}ms rescue={int(state.rescue_round)}"
                )
            if confirmed:
                # One confirmed second observation is enough for the existing
                # HitScanner's >=300ms / >=2-hits readiness rule. If a custom
                # scanner needs three hits, a second local round remains legal.
                if state.rounds >= _CONFIG.local_max_rounds:
                    state.phase = "confirmed"
                else:
                    state.first_result_ts = float(frame_ts)
                    state.candidates = [dict(c) for c in confirmed]
                self._v2224_async_waiting = False
                self._v2225_local_confirm_shot_id = state.shot_id
                return confirmed

            # FAST/legacy proposal could not be verified. One full high-recall
            # rescue is allowed; after a rescue-round failure we stop burning
            # whole-ROI CV and let normal timeout report MISS.
            if (
                _CONFIG.rescue_enabled
                and not state.rescue_round
                and rescue_router_v2225.request(state.shot_id)
            ):
                state.phase = "rescue_queued"
                if _CONFIG.local_log:
                    print(f"[V2.22.5 LOCAL-CONFIRM] shot={state.shot_id} no local proof -> queue FULL rescue")
                # Fall through to V2.22.4; the next worker call consumes the
                # rescue request and invokes the original high-recall extractor.
            else:
                state.phase = "exhausted"
                self._v2224_async_waiting = False
                return []

        candidates = previous_detect(self, gray, frame_ts)
        result_sid = int(getattr(self, "_v2225_result_shot_id", 0) or 0)
        result_ts = float(getattr(self, "_v2225_result_frame_ts", 0.0) or 0.0)
        if result_sid <= 0 or result_ts <= 0.0:
            return candidates

        # Consume marker once; it identifies a completed worker result even if
        # that result contained zero candidates.
        self._v2225_result_shot_id = 0
        self._v2225_result_frame_ts = 0.0

        pending_ids = {
            int(getattr(ev, "shot_id", 0) or 0)
            for ev in list(getattr(self, "audio_events", []) or [])
            if str(getattr(ev, "state", "")) == "pending"
        }
        if result_sid not in pending_ids:
            return []

        old_state = local_confirm_manager_v2225.get(result_sid)
        rescue_round = bool(old_state is not None and old_state.phase == "rescue_queued") or rescue_router_v2225.was_consumed(result_sid)
        pre_by_shot = getattr(self, "_v2225_pre_by_shot", {}) or {}
        pre = pre_by_shot.get(result_sid)
        if pre is None:
            pre = getattr(self, "pre_shot_snapshot", None)

        if candidates:
            local_confirm_manager_v2225.start(
                result_sid,
                result_ts,
                list(candidates),
                pre,
                rescue_round=rescue_round,
            )
        else:
            # Completed FAST result with zero proposals: do not run the same fast
            # global path forever. Move immediately to the one permitted rescue.
            self._v2224_result_frame_ts = 0.0
            if _CONFIG.rescue_enabled and not rescue_round and rescue_router_v2225.request(result_sid):
                local_confirm_manager_v2225.start(result_sid, result_ts, [], pre, rescue_round=False)
                local_confirm_manager_v2225.mark_rescue_queued(result_sid)
                if _CONFIG.local_log:
                    print(f"[V2.22.5 FAST] shot={result_sid} zero proposals -> queue FULL rescue")
            else:
                local_confirm_manager_v2225.mark_exhausted(result_sid)
        return candidates

    def resolve_v2225(self, now_ts: float):
        # Only shots explicitly waiting for full rescue get a longer timeout.
        old_timeout = float(getattr(self, "event_timeout_s", 2.0))
        needs_rescue_time = any(
            st.phase == "rescue_queued" and str(getattr(ev, "state", "")) == "pending"
            for st in list(local_confirm_manager_v2225.states.values())
            for ev in list(getattr(self, "audio_events", []) or [])
            if int(getattr(ev, "shot_id", 0) or 0) == st.shot_id
        )
        if needs_rescue_time:
            self.event_timeout_s = max(old_timeout, _CONFIG.rescue_timeout_s)
        try:
            result = previous_resolve(self, now_ts)
        finally:
            self.event_timeout_s = old_timeout

        # Terminal shots invalidate local state and queued-but-not-started CV.
        for ev in list(getattr(self, "audio_events", []) or []):
            state = str(getattr(ev, "state", "pending") or "pending")
            if state == "pending":
                continue
            sid = int(getattr(ev, "shot_id", 0) or 0)
            if local_confirm_manager_v2225.get(sid) is not None or rescue_router_v2225.requested(sid) or rescue_router_v2225.was_consumed(sid):
                cancelled = _cancel_queued_for_shot(sid)
                local_confirm_manager_v2225.clear(sid)
                pre = getattr(self, "_v2225_pre_by_shot", None)
                if isinstance(pre, dict):
                    pre.pop(sid, None)
                if _CONFIG.local_log and cancelled:
                    print(f"[V2.22.5 CANCEL] shot={sid} cancelled={cancelled} queued CV job(s)")
        return result

    HitScanner._detect_frame_candidates = detect_v2225
    HitScanner._resolve_audio_events = resolve_v2225
    HitScanner._v2225_local_confirmation_patch = True


def _install_startup_audio_gate() -> None:
    """Consume peaks while scanner is OFF/ARMING so calibration noise is not PANG."""
    try:
        from src.engine.shot_critical_v2223 import ShotCriticalControllerV2223
    except Exception:
        return
    if getattr(ShotCriticalControllerV2223, "_v2225_audio_gate_patch", False):
        return
    previous_pending = ShotCriticalControllerV2223.pending_audio
    previous_begin = ShotCriticalControllerV2223.begin_pending_audio

    def pending_v2225(self, detector: Any) -> bool:
        try:
            from src.engine.camera.hit_scanner import hit_scanner
            active = bool(getattr(hit_scanner, "enabled", False)) and str(getattr(hit_scanner, "state", "")) == str(getattr(hit_scanner, "STATE_ACTIVE", "active"))
            if not active:
                # Consume startup/calibration peak so it cannot become pending
                # retroactively when the scanner later arms.
                self.last_seen_peak_ts = max(
                    float(getattr(self, "last_seen_peak_ts", 0.0) or 0.0),
                    float(getattr(detector, "last_peak_ts", 0.0) or 0.0),
                )
                return False
        except Exception:
            pass
        return previous_pending(self, detector)

    def begin_v2225(self, app: Any, detector: Any):
        # AI Training starts its scanner around the same time as auto-calibration.
        # Do not turn marker/reference capture sounds into a delayed shot just
        # because the scanner has already reached STATE_ACTIVE.
        try:
            from src.engine.camera.hit_scanner import hit_scanner
            scene = getattr(app, "scene", None)
            scene_name = type(scene).__name__ if scene is not None else ""
            calibrating = getattr(scene, "_auto_cal_phase", None) is not None
            missing_reference = getattr(hit_scanner, "scene_reference_gray", None) is None
            if scene_name == "AITrainingScene" and (calibrating or missing_reference):
                peak_ts = float(getattr(detector, "last_peak_ts", 0.0) or 0.0)
                self.last_seen_peak_ts = max(float(getattr(self, "last_seen_peak_ts", 0.0) or 0.0), peak_ts)
                return None
        except Exception:
            pass
        return previous_begin(self, app, detector)

    ShotCriticalControllerV2223.pending_audio = pending_v2225
    ShotCriticalControllerV2223.begin_pending_audio = begin_v2225
    ShotCriticalControllerV2223._v2225_audio_gate_patch = True


def install_v2225_runtime(AppClass: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_settings_defaults()
    _load_config_from_runtime()
    _install_fast_extractor_patch()
    _install_pre_by_shot_patch()
    _install_result_marker_patch()
    _install_local_confirmation_patch()
    _install_startup_audio_gate()
    AppClass._v2225_fast_proposal_patch = True
    _INSTALLED = True
    print("[V2.22.5] fast proposal + local confirmation installed")


__all__ = [
    "SCHEMA_VERSION",
    "PATCH_REVISION",
    "FastConfigV2225",
    "ConfirmMetricsV2225",
    "ShotConfirmStateV2225",
    "fast_extract_candidates_v2225",
    "local_confirm_candidates_v2225",
    "local_confirm_manager_v2225",
    "rescue_router_v2225",
    "install_v2225_runtime",
]
