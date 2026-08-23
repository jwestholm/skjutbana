from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.engine.camera.detector_v24_extension import _annotate_candidate, _patch_prior, _ensure_diag_record

CONFIG_PATH = Path("content/ai/detector_v25.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "gt_local_probe_enabled": True,
    "gt_local_probe_radius_px": 48,
    "gt_local_probe_min_absdiff": 1.0,
    "gt_local_probe_min_zscore": 0.55,
    "gt_local_probe_refine_radius_px": 5,
    "tile_refine_enabled": True,
    "tile_refine_radius_px": 7,
    "tile_refine_max_candidates": 72,
    "tile_refine_reserved_slots": 56,
    "tile_refine_min_shift_px": 1.5,
    "tile_refine_max_shift_px": 11.0,
    "tile_refine_nms_radius_px": 2.0,
    "shadow_accumulator_enabled": True,
    "shadow_accumulator_match_radius_px": 12.0,
    "shadow_accumulator_dedupe_radius_px": 4.0,
    "shadow_accumulator_max_clusters": 700,
    "shadow_accumulator_gt_radius_px": 42.0,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


class V25Config:
    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = Path(path)
        self.values = dict(DEFAULT_CONFIG)
        self._mtime: float | None = None
        self._last_check = 0.0
        self.reload(force=True)

    def reload(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_check < 1.0:
            return
        self._last_check = now
        try:
            mtime = self.path.stat().st_mtime
        except Exception:
            mtime = None
        if not force and mtime == self._mtime:
            return
        self._mtime = mtime
        values = dict(DEFAULT_CONFIG)
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    values.update(loaded)
        except Exception as exc:
            print(f"[DETECTOR-V2.5] config load failed, defaults kept: {exc}")
        self.values = values

    def snapshot(self) -> dict[str, Any]:
        self.reload()
        return dict(self.values)


def _temporal_map(context: dict[str, Any]) -> np.ndarray | None:
    try:
        absdiff = context["absdiff"].astype(np.float32, copy=False)
        zscore = context["zscore"].astype(np.float32, copy=False)
        dog = context.get("dog")
        if not isinstance(dog, np.ndarray):
            dog = np.zeros_like(absdiff)
        temporal = (
            absdiff * (0.90 + 0.50 * np.clip(zscore, 0.0, 7.0))
            + 0.30 * np.maximum(dog.astype(np.float32, copy=False), 0.0)
        ).astype(np.float32)
        valid = context.get("valid")
        if isinstance(valid, np.ndarray) and valid.shape == temporal.shape:
            temporal = temporal.copy()
            temporal[~valid.astype(bool)] = 0.0
        return temporal
    except Exception:
        return None


def _weighted_refine(
    temporal: np.ndarray,
    absdiff: np.ndarray,
    zscore: np.ndarray,
    px: int,
    py: int,
    radius: int,
) -> tuple[float, float]:
    h, w = temporal.shape
    radius = max(2, int(radius))
    x0, x1 = max(0, px - radius), min(w, px + radius + 1)
    y0, y1 = max(0, py - radius), min(h, py + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return float(px), float(py)

    t = temporal[y0:y1, x0:x1].astype(np.float32, copy=False)
    a = absdiff[y0:y1, x0:x1].astype(np.float32, copy=False)
    z = zscore[y0:y1, x0:x1].astype(np.float32, copy=False)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    circle = rr <= float(radius)
    values = t[circle]
    if values.size == 0:
        return float(px), float(py)

    threshold = max(float(np.percentile(values, 68.0)), 0.58 * float(np.max(values)))
    mask = circle & (t >= threshold) & ((a >= 1.0) | (z >= 0.55))
    if not np.any(mask):
        return float(px), float(py)

    spatial = np.exp(-(rr * rr) / max(8.0, 2.0 * radius * radius)).astype(np.float32)
    weights = np.where(mask, np.maximum(t - threshold * 0.65, 0.0) * spatial, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-6:
        return float(px), float(py)
    return (
        float(np.sum(weights * xx) / total),
        float(np.sum(weights * yy) / total),
    )


def _gt_local_probe(
    context: dict[str, Any], gt_x: float, gt_y: float, cfg: dict[str, Any]
) -> dict[str, Any] | None:
    """Benchmark-only probe. Never enters the candidate/ranking path."""
    if not bool(cfg.get("gt_local_probe_enabled", True)):
        return None
    bbox = context.get("bbox")
    absdiff = context.get("absdiff")
    zscore = context.get("zscore")
    if not isinstance(bbox, tuple) or len(bbox) != 4:
        return None
    if not isinstance(absdiff, np.ndarray) or not isinstance(zscore, np.ndarray):
        return None
    temporal = _temporal_map(context)
    if temporal is None:
        return None

    x0, y0, _x1, _y1 = bbox
    gx, gy = gt_x - float(x0), gt_y - float(y0)
    h, w = temporal.shape
    radius = max(8, _safe_int(cfg.get("gt_local_probe_radius_px", 48), 48))
    ix0, ix1 = max(0, int(gx - radius)), min(w, int(gx + radius + 1))
    iy0, iy1 = max(0, int(gy - radius)), min(h, int(gy + radius + 1))
    if ix1 <= ix0 or iy1 <= iy0:
        return None

    local = temporal[iy0:iy1, ix0:ix1]
    local_abs = absdiff[iy0:iy1, ix0:ix1]
    local_z = zscore[iy0:iy1, ix0:ix1]
    yy, xx = np.mgrid[iy0:iy1, ix0:ix1]
    circle = ((xx - gx) ** 2 + (yy - gy) ** 2) <= radius * radius
    valid = context.get("valid")
    if isinstance(valid, np.ndarray) and valid.shape == temporal.shape:
        circle &= valid[iy0:iy1, ix0:ix1].astype(bool)

    min_abs = _safe_float(cfg.get("gt_local_probe_min_absdiff", 1.0), 1.0)
    min_z = _safe_float(cfg.get("gt_local_probe_min_zscore", 0.55), 0.55)
    dilated = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = circle & (local >= dilated - 1e-6) & ((local_abs >= min_abs) | (local_z >= min_z))
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"found": False, "search_radius_px": radius}

    scores = local[ys, xs]
    best = int(np.argmax(scores))
    px, py = int(xs[best] + ix0), int(ys[best] + iy0)
    rr = max(2, _safe_int(cfg.get("gt_local_probe_refine_radius_px", 5), 5))
    rx, ry = _weighted_refine(temporal, absdiff, zscore, px, py, rr)
    camera_x, camera_y = rx + float(x0), ry + float(y0)
    dx, dy = camera_x - gt_x, camera_y - gt_y
    sx = max(0, min(w - 1, int(round(rx))))
    sy = max(0, min(h - 1, int(round(ry))))
    return {
        "found": True,
        "camera_x": camera_x,
        "camera_y": camera_y,
        "dx": dx,
        "dy": dy,
        "distance_px": math.hypot(dx, dy),
        "temporal_score": float(temporal[sy, sx]),
        "absdiff": float(absdiff[sy, sx]),
        "zscore": float(zscore[sy, sx]),
        "search_radius_px": radius,
    }


def _refined_tile_candidates(
    engine: Any,
    context: dict[str, Any],
    tiles: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(cfg.get("tile_refine_enabled", True)) or not tiles:
        return []
    bbox = context.get("bbox")
    absdiff, zscore = context.get("absdiff"), context.get("zscore")
    if not isinstance(bbox, tuple) or len(bbox) != 4:
        return []
    if not isinstance(absdiff, np.ndarray) or not isinstance(zscore, np.ndarray):
        return []
    temporal = _temporal_map(context)
    if temporal is None:
        return []

    x0, y0, _x1, _y1 = bbox
    radius = max(2, _safe_int(cfg.get("tile_refine_radius_px", 7), 7))
    min_shift = max(0.0, _safe_float(cfg.get("tile_refine_min_shift_px", 1.5), 1.5))
    max_shift = max(min_shift, _safe_float(cfg.get("tile_refine_max_shift_px", 11.0), 11.0))
    limit = max(1, _safe_int(cfg.get("tile_refine_max_candidates", 72), 72))
    refined: list[dict[str, Any]] = []

    ordered = sorted(
        tiles,
        key=lambda c: (_safe_float(c.get("v24_patch_prior")), _safe_float(c.get("score"))),
        reverse=True,
    )
    for source in ordered:
        px = int(round(_safe_float(source.get("camera_x")) - x0))
        py = int(round(_safe_float(source.get("camera_y")) - y0))
        if px < 0 or py < 0 or px >= temporal.shape[1] or py >= temporal.shape[0]:
            continue
        rx, ry = _weighted_refine(temporal, absdiff, zscore, px, py, radius)
        shift = math.hypot(rx - px, ry - py)
        if shift < min_shift or shift > max_shift:
            continue
        candidate = dict(source)
        candidate.update({
            "camera_x": float(rx + x0),
            "camera_y": float(ry + y0),
            # Keep V2.4 provenance clean: this is a NEW V2.5 hypothesis
            # derived from a V2.4 tile seed, not the original tile peak.
            "v24_tile_probe": 0.0,
            "v25_source_v24_tile": 1.0,
            "v25_refined_tile": 1.0,
            "v25_refine_shift_px": float(shift),
        })
        try:
            _annotate_candidate(candidate, context=context, cfg=cfg)
        except Exception:
            pass
        refined.append(candidate)
        if len(refined) >= limit:
            break
    return refined


def _merge_additive(current: list[dict[str, Any]], extras: list[dict[str, Any]], *, reserve: int, radius: float) -> list[dict[str, Any]]:
    result = [dict(c) for c in current]
    added = 0
    for candidate in sorted(
        extras,
        key=lambda c: (_safe_float(c.get("v24_patch_prior", _patch_prior(c))), _safe_float(c.get("score"))),
        reverse=True,
    ):
        if added >= max(0, reserve):
            break
        cx, cy = _safe_float(candidate.get("camera_x")), _safe_float(candidate.get("camera_y"))
        if any(math.hypot(_safe_float(e.get("camera_x")) - cx, _safe_float(e.get("camera_y")) - cy) < radius for e in result):
            continue
        result.append(dict(candidate))
        added += 1
    return result


def _nearest_vector(candidates: list[dict[str, Any]], gt_x: float, gt_y: float) -> dict[str, float] | None:
    if not candidates:
        return None
    c = min(candidates, key=lambda item: math.hypot(_safe_float(item.get("camera_x")) - gt_x, _safe_float(item.get("camera_y")) - gt_y))
    dx, dy = _safe_float(c.get("camera_x")) - gt_x, _safe_float(c.get("camera_y")) - gt_y
    return {"dx": dx, "dy": dy, "distance_px": math.hypot(dx, dy), "camera_x": _safe_float(c.get("camera_x")), "camera_y": _safe_float(c.get("camera_y"))}


def _keep_best_vector(block: dict[str, Any], key: str, value: dict[str, float] | None) -> None:
    if not isinstance(value, dict):
        return
    old = block.get(key)
    if not isinstance(old, dict) or _safe_float(value.get("distance_px"), 1e9) < _safe_float(old.get("distance_px"), 1e9):
        block[key] = dict(value)


def _shadow_accumulate(engine: Any, shot_id: int, candidates: list[dict[str, Any]], frame_ts: float, cfg: dict[str, Any]) -> None:
    if shot_id <= 0 or not bool(cfg.get("shadow_accumulator_enabled", True)):
        return
    states = getattr(engine, "_v25_shadow_accumulators", None)
    if not isinstance(states, dict):
        states = {}
        engine._v25_shadow_accumulators = states
    state = states.setdefault(shot_id, {"frame": 0, "frames_with_candidates": 0, "observations": 0, "clusters": []})
    state["frame"] = int(state.get("frame", 0)) + 1
    frame = int(state["frame"])

    observations = [dict(c) for c in candidates if _safe_float(c.get("shot_accumulator_carried")) <= 0.5 and _safe_float(c.get("candidate_bank_carried", c.get("v2_bank_carried", 0.0))) <= 0.5]
    if observations:
        state["frames_with_candidates"] = int(state.get("frames_with_candidates", 0)) + 1
    observations.sort(key=lambda c: (_safe_float(c.get("v24_patch_prior", _patch_prior(c))), _safe_float(c.get("score"))), reverse=True)

    dedupe = max(1.0, _safe_float(cfg.get("shadow_accumulator_dedupe_radius_px", 4.0), 4.0))
    unique: list[dict[str, Any]] = []
    for c in observations:
        cx, cy = _safe_float(c.get("camera_x")), _safe_float(c.get("camera_y"))
        if any(math.hypot(_safe_float(e.get("camera_x")) - cx, _safe_float(e.get("camera_y")) - cy) < dedupe for e in unique):
            continue
        unique.append(c)
    state["observations"] = int(state.get("observations", 0)) + len(unique)

    clusters = state["clusters"]
    radius = max(2.0, _safe_float(cfg.get("shadow_accumulator_match_radius_px", 12.0), 12.0))
    used: set[int] = set()
    for c in unique:
        cx, cy = _safe_float(c.get("camera_x")), _safe_float(c.get("camera_y"))
        best_i, best_d = -1, 1e9
        for i, cluster in enumerate(clusters):
            if i in used or int(cluster.get("last_frame", -1)) == frame:
                continue
            d = math.hypot(_safe_float(cluster.get("mean_x")) - cx, _safe_float(cluster.get("mean_y")) - cy)
            if d <= radius and d < best_d:
                best_i, best_d = i, d
        if best_i < 0:
            clusters.append({
                "mean_x": cx, "mean_y": cy, "hits": 1,
                "first_frame": frame, "last_frame": frame,
                "first_ts": frame_ts, "last_ts": frame_ts, "m2": 0.0,
                "v1_hits": int(_safe_float(c.get("detector_v1")) > 0.5),
                "v2_hits": int(_safe_float(c.get("detector_v2")) > 0.5),
                "tile_hits": int(_safe_float(c.get("v24_tile_probe")) > 0.5),
                "refined_hits": int(_safe_float(c.get("v25_refined_tile")) > 0.5),
            })
            best_i = len(clusters) - 1
        else:
            cluster = clusters[best_i]
            n = max(1, int(cluster.get("hits", 1)))
            nn = n + 1
            mx, my = _safe_float(cluster.get("mean_x")), _safe_float(cluster.get("mean_y"))
            nx, ny = mx + (cx - mx) / nn, my + (cy - my) / nn
            cluster["m2"] = max(0.0, _safe_float(cluster.get("m2")) + (cx - mx) * (cx - nx) + (cy - my) * (cy - ny))
            cluster.update({"mean_x": nx, "mean_y": ny, "hits": nn, "last_frame": frame, "last_ts": frame_ts})
            for ck, sk in (("v1_hits", "detector_v1"), ("v2_hits", "detector_v2"), ("tile_hits", "v24_tile_probe"), ("refined_hits", "v25_refined_tile")):
                cluster[ck] = int(cluster.get(ck, 0)) + int(_safe_float(c.get(sk)) > 0.5)
        used.add(best_i)

    max_clusters = max(50, _safe_int(cfg.get("shadow_accumulator_max_clusters", 700), 700))
    if len(clusters) > max_clusters:
        clusters.sort(key=lambda c: (int(c.get("hits", 1)), int(c.get("last_frame", 0))), reverse=True)
        del clusters[max_clusters:]


def _shadow_summary(engine: Any, shot_id: int, gt_x: float, gt_y: float, cfg: dict[str, Any]) -> dict[str, Any]:
    states = getattr(engine, "_v25_shadow_accumulators", {})
    state = states.get(shot_id, {}) if isinstance(states, dict) else {}
    clusters = state.get("clusters", []) if isinstance(state, dict) else []
    if not isinstance(clusters, list):
        clusters = []
    nearest, nearest_d = None, 1e9
    for c in clusters:
        d = math.hypot(_safe_float(c.get("mean_x")) - gt_x, _safe_float(c.get("mean_y")) - gt_y)
        if d < nearest_d:
            nearest, nearest_d = c, d
    gt_cluster = None
    gt_radius = max(1.0, _safe_float(cfg.get("shadow_accumulator_gt_radius_px", 42.0), 42.0))
    if nearest is not None and nearest_d <= gt_radius:
        hits = max(1, int(nearest.get("hits", 1)))
        gt_cluster = {
            "distance_px": nearest_d,
            "dx": _safe_float(nearest.get("mean_x")) - gt_x,
            "dy": _safe_float(nearest.get("mean_y")) - gt_y,
            "hits": hits,
            "frame_span": int(nearest.get("last_frame", 1)) - int(nearest.get("first_frame", 1)) + 1,
            "time_span_s": max(0.0, _safe_float(nearest.get("last_ts")) - _safe_float(nearest.get("first_ts"))),
            "jitter_px": math.sqrt(max(0.0, _safe_float(nearest.get("m2"))) / max(1, hits - 1)),
            "v1_hits": int(nearest.get("v1_hits", 0)),
            "v2_hits": int(nearest.get("v2_hits", 0)),
            "tile_hits": int(nearest.get("tile_hits", 0)),
            "refined_hits": int(nearest.get("refined_hits", 0)),
        }
    return {
        "frames": int(state.get("frame", 0)) if isinstance(state, dict) else 0,
        "frames_with_candidates": int(state.get("frames_with_candidates", 0)) if isinstance(state, dict) else 0,
        "observations": int(state.get("observations", 0)) if isinstance(state, dict) else 0,
        "clusters_created": len(clusters),
        "clusters_hits_ge_2": sum(1 for c in clusters if int(c.get("hits", 1)) >= 2),
        "clusters_hits_ge_3": sum(1 for c in clusters if int(c.get("hits", 1)) >= 3),
        "clusters_hits_ge_4": sum(1 for c in clusters if int(c.get("hits", 1)) >= 4),
        "gt_cluster": gt_cluster,
    }


def patch_candidate_generator_v25(cls: type) -> None:
    if bool(getattr(cls, "_detector_v25_extension_installed", False)):
        return
    original_init = cls.__init__
    original_extract = cls._extract_candidates
    original_generate = cls.generate
    original_record = cls.record_funnel_evaluation
    original_empty = cls.record_empty_evaluation
    original_reset = cls.reset_runtime_state

    def init_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._v25_config = V25Config()
        self._v25_last_refined_tiles = []
        self._v25_shadow_accumulators = {}

    def extract_wrapped(self: Any, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        candidates = original_extract(self, *args, **kwargs)
        try:
            cfg = self._v25_config.snapshot()
            if not bool(cfg.get("enabled", True)):
                return candidates
            context = getattr(self, "_v24_last_context", None)
            tiles = [dict(c) for c in getattr(self, "_v24_last_tile_probes", []) or []]
            self._v25_last_refined_tiles = _refined_tile_candidates(self, context, tiles, cfg) if isinstance(context, dict) else []
            scanner = kwargs.get("scanner")
            gt = getattr(scanner, "_detector_v2_ground_truth", None) if scanner is not None else None
            if isinstance(context, dict) and isinstance(gt, dict):
                shot_id = _safe_int(gt.get("shot_id"))
                record = getattr(self, "_diagnostics", {}).get(shot_id)
                if not isinstance(record, dict) and scanner is not None and shot_id > 0:
                    try:
                        record = _ensure_diag_record(self, scanner, shot_id)
                    except Exception:
                        record = None
                if isinstance(record, dict):
                    v25 = record.setdefault("v25", {})
                    probe = _gt_local_probe(context, _safe_float(gt.get("camera_x")), _safe_float(gt.get("camera_y")), cfg)
                    if isinstance(probe, dict) and probe.get("found"):
                        old = v25.get("gt_local_probe")
                        if not isinstance(old, dict) or _safe_float(probe.get("temporal_score")) > _safe_float(old.get("temporal_score")):
                            v25["gt_local_probe"] = probe
                    vectors = v25.setdefault("best_vectors", {})
                    gx, gy = _safe_float(gt.get("camera_x")), _safe_float(gt.get("camera_y"))
                    _keep_best_vector(vectors, "v24_tile", _nearest_vector(tiles, gx, gy))
                    _keep_best_vector(vectors, "v25_refined_tile", _nearest_vector(self._v25_last_refined_tiles, gx, gy))
        except Exception as exc:
            self._v25_last_refined_tiles = []
            scanner = kwargs.get("scanner")
            if bool(getattr(scanner, "shot_diag_enabled", False)):
                print(f"[DETECTOR-V2.5] extraction diagnostics fallback: {exc}")
        return candidates

    def generate_wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_generate(self, *args, **kwargs)
        try:
            cfg = self._v25_config.snapshot()
            if not bool(cfg.get("enabled", True)):
                return result
            current = [dict(c) for c in list(result.candidates)]
            refined = [dict(c) for c in getattr(self, "_v25_last_refined_tiles", []) or []]
            current = _merge_additive(current, refined, reserve=max(0, _safe_int(cfg.get("tile_refine_reserved_slots", 56), 56)), radius=max(0.5, _safe_float(cfg.get("tile_refine_nms_radius_px", 2.0), 2.0)))
            telemetry = dict(getattr(result, "telemetry", {}) or {})
            telemetry.update({"schema_version_v25": "2.5", "v25_refined_tile_count": len(refined), "v25_final_count": len(current)})
            shot_id = _safe_int(telemetry.get("shot_id"))
            _shadow_accumulate(self, shot_id, current, float(kwargs.get("frame_ts", time.time())), cfg)
            scanner = kwargs.get("scanner")
            gt = getattr(scanner, "_detector_v2_ground_truth", None) if scanner is not None else None
            if isinstance(gt, dict) and _safe_int(gt.get("shot_id")) == shot_id:
                record = getattr(self, "_diagnostics", {}).get(shot_id)
                if isinstance(record, dict):
                    v25 = record.setdefault("v25", {})
                    vectors = v25.setdefault("best_vectors", {})
                    _keep_best_vector(vectors, "v25_final", _nearest_vector(current, _safe_float(gt.get("camera_x")), _safe_float(gt.get("camera_y"))))
                    v25["max_refined_tile_count"] = max(int(v25.get("max_refined_tile_count", 0)), len(refined))
                    v25["max_final_count"] = max(int(v25.get("max_final_count", 0)), len(current))
            return type(result)(candidates=current, telemetry=telemetry)
        except Exception as exc:
            scanner = kwargs.get("scanner")
            if bool(getattr(scanner, "shot_diag_enabled", False)):
                print(f"[DETECTOR-V2.5] generate fallback: {exc}")
            return result

    def record_v25(self: Any, scanner: Any, gt_xy: Any, raw_hotspots: Any) -> None:
        if scanner is None:
            return
        gt = getattr(scanner, "_detector_v2_ground_truth", None)
        if not isinstance(gt, dict):
            return
        if gt_xy is None:
            gt_xy = (_safe_float(gt.get("camera_x")), _safe_float(gt.get("camera_y")))
        shot_id = _safe_int(gt.get("shot_id"))
        record = getattr(self, "_diagnostics", {}).get(shot_id)
        if not isinstance(record, dict):
            return
        gx, gy = float(gt_xy[0]), float(gt_xy[1])
        cfg = self._v25_config.snapshot()
        v25 = record.setdefault("v25", {})
        v25["shadow_accumulator"] = _shadow_summary(self, shot_id, gx, gy, cfg)
        raw = [dict(c) for c in list(raw_hotspots or [])]
        refined = [c for c in raw if _safe_float(c.get("v25_refined_tile")) > 0.5]
        funnel = record.setdefault("evaluation_funnel", {})
        vec = _nearest_vector(refined, gx, gy)
        funnel["raw_v25_refined_tile_count"] = len(refined)
        funnel["raw_v25_refined_tile_nearest_px"] = vec.get("distance_px") if isinstance(vec, dict) else None
        try:
            from src.engine.ai.runtime import get_ai_runtime
            runtime = get_ai_runtime()
            actual = [dict(c) for c in getattr(runtime, "_v24_last_rank_pool", []) or []]
            shadow = [dict(c) for c in getattr(runtime, "_v25_shadow_rank_pool", []) or []]
            def rank_of(pool: list[dict[str, Any]]) -> tuple[int | None, float | None]:
                if not pool:
                    return None, None
                i = min(range(len(pool)), key=lambda j: math.hypot(_safe_float(pool[j].get("camera_x")) - gx, _safe_float(pool[j].get("camera_y")) - gy))
                d = math.hypot(_safe_float(pool[i].get("camera_x")) - gx, _safe_float(pool[i].get("camera_y")) - gy)
                return i + 1, d
            base_rank, base_d = rank_of(actual)
            shadow_rank, shadow_d = rank_of(shadow)
            funnel["v25_shadow_ranking"] = {"shadow_mode": True, "base_pool_count": len(actual), "shadow_pool_count": len(shadow), "base_gt_rank": base_rank, "base_gt_distance_px": base_d, "v4_shadow_gt_rank": shadow_rank, "v4_shadow_gt_distance_px": shadow_d}
        except Exception:
            pass

    def record_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        original_record(self, *args, **kwargs)
        try:
            record_v25(self, kwargs.get("scanner"), kwargs.get("gt_xy"), kwargs.get("raw_hotspots", []))
        except Exception:
            pass

    def empty_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        original_empty(self, *args, **kwargs)
        try:
            record_v25(self, kwargs.get("scanner"), kwargs.get("gt_xy"), [])
        except Exception:
            pass

    def reset_wrapped(self: Any) -> None:
        original_reset(self)
        self._v25_last_refined_tiles = []
        self._v25_shadow_accumulators = {}

    cls.__init__ = init_wrapped
    cls._extract_candidates = extract_wrapped
    cls.generate = generate_wrapped
    cls.record_funnel_evaluation = record_wrapped
    cls.record_empty_evaluation = empty_wrapped
    cls.reset_runtime_state = reset_wrapped
    cls._detector_v25_extension_installed = True


def apply_detector_v25_extension() -> None:
    from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
    patch_candidate_generator_v25(CandidateGeneratorV2)
    print("[DETECTOR-V2.5] localisation + refined-tile + shadow accumulator installed")


__all__ = [
    "apply_detector_v25_extension",
    "patch_candidate_generator_v25",
    "_gt_local_probe",
    "_weighted_refine",
    "_shadow_accumulate",
    "_shadow_summary",
]
