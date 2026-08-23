from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


CONFIG_PATH = Path("content/ai/detector_v24.json")
BENCHMARK_CONTROL_PATH = Path("content/ai/benchmark_control.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "tile_probe_enabled": True,
    "tile_probe_columns": 10,
    "tile_probe_rows": 8,
    "tile_probe_per_tile": 3,
    "tile_probe_max_candidates": 160,
    "tile_probe_local_sigma": 1.05,
    "tile_probe_min_absdiff": 2.2,
    "tile_probe_min_zscore": 0.95,
    "tile_probe_strong_absdiff": 3.8,
    "tile_probe_nms_radius_px": 3.5,
    "tile_probe_reserved_slots": 100,
    "shot_accumulator_enabled": True,
    "shot_accumulator_match_radius_px": 7.0,
    "shot_accumulator_max_age_s": 0.70,
    "shot_accumulator_single_v1_max_age_s": 0.32,
    "shot_accumulator_single_patch_max_age_s": 0.18,
    "shot_accumulator_single_patch_min_prior": 0.62,
    "shot_accumulator_single_patch_min_absdiff": 3.2,
    "shot_accumulator_confirm_hits": 2,
    "shot_accumulator_confirm_span_s": 0.014,
    "shot_accumulator_min_v1_score": 4.2,
    "shot_accumulator_min_patch_prior": 0.38,
    "shot_accumulator_carried_slots": 64,
    "shot_accumulator_max_clusters": 500,
    "shot_accumulator_output_limit": 320,
    "patch_radius_px": 8,
    "diagnostics_enabled": True,
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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class V24Config:
    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = Path(path)
        self.values = dict(DEFAULT_CONFIG)
        self._last_mtime: float | None = None
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
        if not force and mtime == self._last_mtime:
            return
        self._last_mtime = mtime
        values = dict(DEFAULT_CONFIG)
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    values.update(loaded)
        except Exception as exc:
            print(f"[DETECTOR-V2.4] config load failed, defaults kept: {exc}")
        self.values = values

    def get(self, key: str, default: Any = None) -> Any:
        self.reload()
        return self.values.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        self.reload()
        return dict(self.values)


def _patch_descriptor(
    *,
    px: int,
    py: int,
    absdiff: np.ndarray,
    darkening: np.ndarray,
    zscore: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Describe the *shape* of the local temporal change around a candidate.

    V2.3 showed that scalar detector strength could not distinguish the true
    candidate from artifacts. These descriptors instead ask whether the signal
    is compact, centred, roughly isotropic and locally stronger than its outer
    surroundings. Bright-ring/dark-core behaviour is retained as a soft feature,
    not a mandatory rule, because synthetic hole kinds vary.
    """

    h, w = absdiff.shape
    radius = max(5, _safe_int(cfg.get("patch_radius_px", 8), 8))
    x0 = max(0, px - radius)
    x1 = min(w, px + radius + 1)
    y0 = max(0, py - radius)
    y1 = min(h, py + radius + 1)

    if x1 <= x0 or y1 <= y0:
        return {}

    patch_abs = absdiff[y0:y1, x0:x1].astype(np.float32, copy=False)
    patch_dark = darkening[y0:y1, x0:x1].astype(np.float32, copy=False)
    patch_z = zscore[y0:y1, x0:x1].astype(np.float32, copy=False)
    patch_bright = np.maximum(patch_abs - patch_dark, 0.0)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx.astype(np.float32) - float(px)
    dy = yy.astype(np.float32) - float(py)
    distance = np.sqrt(dx * dx + dy * dy)

    core = distance <= 2.25
    inner = distance <= 4.25
    ring = (distance >= 2.75) & (distance <= 6.25)
    outer = (distance >= 6.0) & (distance <= float(radius))
    full = distance <= float(radius)

    def mean_where(array: np.ndarray, mask: np.ndarray) -> float:
        values = array[mask]
        return float(np.mean(values)) if values.size else 0.0

    core_abs = mean_where(patch_abs, core)
    ring_abs = mean_where(patch_abs, ring)
    outer_abs = mean_where(patch_abs, outer)
    core_z = mean_where(patch_z, core)
    ring_z = mean_where(patch_z, ring)
    core_dark = mean_where(patch_dark, core)
    ring_bright = mean_where(patch_bright, ring)

    full_weights = np.where(full, np.maximum(patch_abs, 0.0), 0.0).astype(np.float32)
    total_energy = float(np.sum(full_weights))
    inner_energy = float(np.sum(np.where(inner, full_weights, 0.0)))
    compactness = inner_energy / max(1e-6, total_energy)

    if total_energy > 1e-6:
        centroid_x = float(np.sum(full_weights * dx) / total_energy)
        centroid_y = float(np.sum(full_weights * dy) / total_energy)
        centroid_offset = math.hypot(centroid_x, centroid_y)
        centeredness = math.exp(-centroid_offset / 3.0)

        rel_x = dx - centroid_x
        rel_y = dy - centroid_y
        cov_xx = float(np.sum(full_weights * rel_x * rel_x) / total_energy)
        cov_yy = float(np.sum(full_weights * rel_y * rel_y) / total_energy)
        cov_xy = float(np.sum(full_weights * rel_x * rel_y) / total_energy)
        trace = cov_xx + cov_yy
        determinant = max(0.0, cov_xx * cov_yy - cov_xy * cov_xy)
        disc = max(0.0, trace * trace * 0.25 - determinant)
        root = math.sqrt(disc)
        eig_max = max(1e-6, trace * 0.5 + root)
        eig_min = max(0.0, trace * 0.5 - root)
        isotropy = _clip01(eig_min / eig_max)
    else:
        centroid_offset = float(radius)
        centeredness = 0.0
        isotropy = 0.0

    # Bounded relative contrast. 0.5 is roughly neutral, 1.0 strongly favours a
    # centre that changes more than the outer background.
    contrast = (core_abs - outer_abs) / max(1.0, core_abs + outer_abs)
    core_to_outer = _clip01(0.5 + 0.5 * contrast)

    ringness = _clip01(ring_abs / max(1e-6, core_abs + ring_abs + outer_abs))
    bipolar = _clip01(
        math.tanh(max(0.0, core_dark) / 4.0)
        * math.tanh(max(0.0, ring_bright) / 4.0)
    )
    local_snr = _clip01(math.tanh(max(core_z, ring_z) / 3.0))

    # Tiny normalized patch grids let the linear V4 ranker learn actual local
    # image SHAPE rather than only a handful of scalar summaries. 5x5 keeps the
    # model tiny (75 grid values across abs/signed/z) while preserving enough
    # structure to distinguish a compact centre/ring from long edges/corners.
    grid_size = 5
    full_values = patch_abs[full]
    if full_values.size:
        scale = max(1.0, float(np.percentile(full_values, 90.0)))
    else:
        scale = 1.0
    abs_norm = np.clip(patch_abs / scale, 0.0, 1.5).astype(np.float32)
    signed_norm = np.clip((patch_dark - patch_bright) / scale, -1.0, 1.0).astype(np.float32)
    z_norm = np.clip(patch_z / 5.0, 0.0, 1.5).astype(np.float32)

    abs_grid = cv2.resize(abs_norm, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    signed_grid = cv2.resize(signed_norm, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    z_grid = cv2.resize(z_norm, (grid_size, grid_size), interpolation=cv2.INTER_AREA)

    grid_features: dict[str, float] = {}
    for gy in range(grid_size):
        for gx in range(grid_size):
            grid_features[f"v24_patch_abs_g{gy}{gx}"] = float(abs_grid[gy, gx])
            grid_features[f"v24_patch_signed_g{gy}{gx}"] = float(signed_grid[gy, gx])
            grid_features[f"v24_patch_z_g{gy}{gx}"] = float(z_grid[gy, gx])

    result = {
        "v24_patch_core_abs": core_abs,
        "v24_patch_ring_abs": ring_abs,
        "v24_patch_outer_abs": outer_abs,
        "v24_patch_core_z": core_z,
        "v24_patch_ring_z": ring_z,
        "v24_patch_core_dark": core_dark,
        "v24_patch_ring_bright": ring_bright,
        "v24_patch_core_to_outer": core_to_outer,
        "v24_patch_compactness": _clip01(compactness),
        "v24_patch_centeredness": _clip01(centeredness),
        "v24_patch_isotropy": _clip01(isotropy),
        "v24_patch_bipolar": bipolar,
        "v24_patch_local_snr": local_snr,
        "v24_patch_ringness": ringness,
        "v24_patch_centroid_offset_px": float(centroid_offset),
    }
    result.update(grid_features)
    return result


def _patch_prior(candidate: dict[str, Any]) -> float:
    return _clip01(
        0.24 * _safe_float(candidate.get("v24_patch_local_snr", 0.0))
        + 0.20 * _safe_float(candidate.get("v24_patch_centeredness", 0.0))
        + 0.17 * _safe_float(candidate.get("v24_patch_compactness", 0.0))
        + 0.14 * _safe_float(candidate.get("v24_patch_core_to_outer", 0.0))
        + 0.11 * _safe_float(candidate.get("v24_patch_isotropy", 0.0))
        + 0.08 * _safe_float(candidate.get("v24_patch_bipolar", 0.0))
        + 0.06 * _safe_float(candidate.get("detector_agreement", 0.0))
    )


def _annotate_candidate(
    candidate: dict[str, Any],
    *,
    context: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    bbox = context.get("bbox")
    absdiff = context.get("absdiff")
    darkening = context.get("darkening")
    zscore = context.get("zscore")
    if (
        not isinstance(bbox, tuple)
        or len(bbox) != 4
        or not isinstance(absdiff, np.ndarray)
        or not isinstance(darkening, np.ndarray)
        or not isinstance(zscore, np.ndarray)
    ):
        return

    x0, y0, _x1, _y1 = bbox
    px = int(round(_safe_float(candidate.get("camera_x", 0.0)))) - int(x0)
    py = int(round(_safe_float(candidate.get("camera_y", 0.0)))) - int(y0)
    if py < 0 or px < 0 or py >= absdiff.shape[0] or px >= absdiff.shape[1]:
        return

    candidate.update(
        _patch_descriptor(
            px=px,
            py=py,
            absdiff=absdiff,
            darkening=darkening,
            zscore=zscore,
            cfg=cfg,
        )
    )
    candidate["v24_patch_prior"] = _patch_prior(candidate)


def _tile_probe_candidates(
    engine: Any,
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
    cfg: dict[str, Any],
) -> list[dict[str, float]]:
    """Direct local temporal sampler independent of the global saliency gate.

    V2.3 produced ~243 labelled rounds with strong GT signal but no peak. This
    path intentionally asks a simpler question per image tile: "what changed
    most *locally* here?" A true 4-gray-level hole no longer has to beat the
    strongest artifact elsewhere in the projector image.
    """

    if not bool(cfg.get("tile_probe_enabled", True)) or not np.any(valid):
        return []

    temporal = (
        absdiff * (0.85 + 0.48 * np.clip(zscore, 0.0, 7.0))
        + 0.28 * np.maximum(dog, 0.0)
    ).astype(np.float32)
    temporal[~valid] = 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(temporal, kernel)
    local_max = temporal >= (dilated - 1e-6)

    cols = max(1, _safe_int(cfg.get("tile_probe_columns", 8), 8))
    rows = max(1, _safe_int(cfg.get("tile_probe_rows", 6), 6))
    per_tile = max(1, _safe_int(cfg.get("tile_probe_per_tile", 2), 2))
    max_total = max(1, _safe_int(cfg.get("tile_probe_max_candidates", 80), 80))
    local_sigma = max(0.0, _safe_float(cfg.get("tile_probe_local_sigma", 1.15), 1.15))
    min_abs = max(0.0, _safe_float(cfg.get("tile_probe_min_absdiff", 2.4), 2.4))
    min_z = max(0.0, _safe_float(cfg.get("tile_probe_min_zscore", 1.05), 1.05))
    strong_abs = max(min_abs, _safe_float(cfg.get("tile_probe_strong_absdiff", 4.0), 4.0))
    nms_radius = max(1.0, _safe_float(cfg.get("tile_probe_nms_radius_px", 4.0), 4.0))

    h, w = temporal.shape
    x0_full, y0_full, _x1_full, _y1_full = bbox
    found: list[tuple[float, int, int]] = []

    for tile_y in range(rows):
        sy0 = int(round(tile_y * h / rows))
        sy1 = int(round((tile_y + 1) * h / rows))
        if sy1 <= sy0:
            continue
        for tile_x in range(cols):
            sx0 = int(round(tile_x * w / cols))
            sx1 = int(round((tile_x + 1) * w / cols))
            if sx1 <= sx0:
                continue

            tile_valid = valid[sy0:sy1, sx0:sx1]
            if not np.any(tile_valid):
                continue
            tile_values = temporal[sy0:sy1, sx0:sx1][tile_valid]
            median = float(np.median(tile_values))
            mad = float(np.median(np.abs(tile_values - median)))
            threshold = median + local_sigma * 1.4826 * mad

            tile_mask = (
                tile_valid
                & local_max[sy0:sy1, sx0:sx1]
                & (temporal[sy0:sy1, sx0:sx1] >= threshold)
                & (
                    (absdiff[sy0:sy1, sx0:sx1] >= strong_abs)
                    | (
                        (absdiff[sy0:sy1, sx0:sx1] >= min_abs)
                        & (zscore[sy0:sy1, sx0:sx1] >= min_z)
                    )
                )
            )
            ys, xs = np.nonzero(tile_mask)
            if len(xs) == 0:
                continue
            scores = temporal[sy0:sy1, sx0:sx1][ys, xs]
            order = np.argsort(scores)[::-1][:per_tile]
            for idx in order:
                px = int(xs[int(idx)] + sx0)
                py = int(ys[int(idx)] + sy0)
                found.append((float(temporal[py, px]), px, py))

    found.sort(key=lambda item: item[0], reverse=True)
    chosen: list[tuple[float, int, int]] = []
    for item in found:
        score, px, py = item
        if any(math.hypot(px - cx, py - cy) < nms_radius for _s, cx, cy in chosen):
            continue
        chosen.append(item)
        if len(chosen) >= max_total:
            break

    candidates: list[dict[str, float]] = []
    for temporal_score, px, py in chosen:
        try:
            features = engine._candidate_features(
                px=px,
                py=py,
                saliency=saliency,
                absdiff=absdiff,
                darkening=darkening,
                dog=dog,
                zscore=zscore,
            )
        except Exception:
            continue

        candidate: dict[str, float] = {
            "camera_x": float(px + x0_full),
            "camera_y": float(py + y0_full),
            "area": float(features.get("area", 1.0)),
            "radius": float(features.get("radius", 1.0)),
            "circularity": float(features.get("circularity", 0.5)),
            "score": float(max(3.8, min(30.0, 0.30 * temporal_score + 0.70 * features.get("score", 4.0)))),
            "center_darkening": float(features.get("center_change", 0.0)),
            "local_contrast_gain": float(features.get("local_contrast", 0.0)),
            "blackhat_value": float(features.get("dog_value", 0.0)),
            "change_value": float(features.get("center_change", 0.0)),
            "pre_shot_change": float(features.get("center_change", 0.0)),
            "timestamp": float(frame_ts),
            "detector_v1": 0.0,
            "detector_v2": 1.0,
            "v24_tile_probe": 1.0,
            "v2_saliency": float(saliency[py, px]),
            "v2_zscore": float(zscore[py, px]),
            "v2_absdiff": float(absdiff[py, px]),
            "v2_darkening": float(darkening[py, px]),
            "v2_dog": float(dog[py, px]),
        }
        candidate.update(
            _patch_descriptor(
                px=px,
                py=py,
                absdiff=absdiff,
                darkening=darkening,
                zscore=zscore,
                cfg=cfg,
            )
        )
        candidate["v24_patch_prior"] = _patch_prior(candidate)
        try:
            engine._apply_known_hole_penalty(scanner, candidate)
        except Exception:
            pass
        candidates.append(candidate)

    return candidates


def _merge_reserved(
    current: list[dict[str, Any]],
    extras: list[dict[str, Any]],
    *,
    reserve: int,
    radius: float,
) -> list[dict[str, Any]]:
    result = [dict(candidate) for candidate in current]
    extras_sorted = sorted(
        extras,
        key=lambda candidate: (
            _safe_float(candidate.get("v24_patch_prior", 0.0)),
            _safe_float(candidate.get("score", 0.0)),
        ),
        reverse=True,
    )
    added = 0
    for candidate in extras_sorted:
        if added >= max(0, int(reserve)):
            break
        cx = _safe_float(candidate.get("camera_x", 0.0))
        cy = _safe_float(candidate.get("camera_y", 0.0))
        if any(
            math.hypot(
                _safe_float(existing.get("camera_x", 0.0)) - cx,
                _safe_float(existing.get("camera_y", 0.0)) - cy,
            ) < radius
            for existing in result
        ):
            continue
        result.append(dict(candidate))
        added += 1
    return result


def _cluster_quality(candidate: dict[str, Any]) -> float:
    score = math.tanh(max(0.0, _safe_float(candidate.get("score", 0.0))) / 15.0)
    prior = _safe_float(candidate.get("v24_patch_prior", _patch_prior(candidate)))
    agreement = _clip01(_safe_float(candidate.get("detector_agreement", 0.0)))
    temporal = math.tanh(
        max(
            _safe_float(candidate.get("v2_absdiff", 0.0)),
            _safe_float(candidate.get("center_darkening", 0.0)),
            _safe_float(candidate.get("change_value", 0.0)),
        )
        / 6.0
    )
    return _clip01(0.42 * prior + 0.24 * score + 0.20 * temporal + 0.14 * agreement)


def _update_shot_accumulator(
    engine: Any,
    *,
    shot_id: int,
    candidates: list[dict[str, Any]],
    frame_ts: float,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if shot_id <= 0 or not bool(cfg.get("shot_accumulator_enabled", True)):
        return candidates

    states = getattr(engine, "_v24_shot_accumulators", None)
    if not isinstance(states, dict):
        states = {}
        engine._v24_shot_accumulators = states
    clusters: list[dict[str, Any]] = states.setdefault(shot_id, [])

    match_radius = max(2.0, _safe_float(cfg.get("shot_accumulator_match_radius_px", 7.0), 7.0))
    max_clusters = max(20, _safe_int(cfg.get("shot_accumulator_max_clusters", 420), 420))

    # Only *new camera observations* count as accumulator hits. A candidate that
    # was itself carried by an older bank must not create artificial persistence.
    observations = [
        dict(candidate)
        for candidate in candidates
        if _safe_float(candidate.get("candidate_bank_carried", candidate.get("v2_bank_carried", 0.0))) <= 0.5
        and _safe_float(candidate.get("shot_accumulator_carried", 0.0)) <= 0.5
    ]
    observations.sort(key=_cluster_quality, reverse=True)

    # Spatial de-duplication inside one frame.
    deduped: list[dict[str, Any]] = []
    for candidate in observations:
        cx = _safe_float(candidate.get("camera_x", 0.0))
        cy = _safe_float(candidate.get("camera_y", 0.0))
        if any(
            math.hypot(
                _safe_float(existing.get("camera_x", 0.0)) - cx,
                _safe_float(existing.get("camera_y", 0.0)) - cy,
            ) < 3.0
            for existing in deduped
        ):
            continue
        deduped.append(candidate)

    matched_clusters: set[int] = set()
    candidate_cluster: dict[int, dict[str, Any]] = {}

    for candidate in deduped:
        cx = _safe_float(candidate.get("camera_x", 0.0))
        cy = _safe_float(candidate.get("camera_y", 0.0))
        quality = _cluster_quality(candidate)

        best_index = -1
        best_distance = float("inf")
        for index, cluster in enumerate(clusters):
            if index in matched_clusters:
                continue
            distance = math.hypot(float(cluster["x"]) - cx, float(cluster["y"]) - cy)
            if distance <= match_radius and distance < best_distance:
                best_index = index
                best_distance = distance

        if best_index < 0:
            cluster = {
                "x": cx,
                "y": cy,
                "mean_x": cx,
                "mean_y": cy,
                "m2": 0.0,
                "hits": 1,
                "first_ts": float(frame_ts),
                "last_ts": float(frame_ts),
                "best": dict(candidate),
                "best_quality": quality,
                "v1_hits": int(_safe_float(candidate.get("detector_v1", 0.0)) > 0.5),
                "v2_hits": int(_safe_float(candidate.get("detector_v2", 0.0)) > 0.5),
                "tile_hits": int(_safe_float(candidate.get("v24_tile_probe", 0.0)) > 0.5),
            }
            clusters.append(cluster)
            best_index = len(clusters) - 1
        else:
            cluster = clusters[best_index]
            old_hits = max(1, int(cluster.get("hits", 1)))
            new_hits = old_hits + 1
            old_x = float(cluster.get("mean_x", cluster.get("x", cx)))
            old_y = float(cluster.get("mean_y", cluster.get("y", cy)))
            new_x = old_x + (cx - old_x) / float(new_hits)
            new_y = old_y + (cy - old_y) / float(new_hits)
            old_m2 = float(cluster.get("m2", 0.0))
            displacement_sq = (cx - old_x) * (cx - new_x) + (cy - old_y) * (cy - new_y)
            cluster["m2"] = max(0.0, old_m2 + displacement_sq)
            cluster["hits"] = new_hits
            cluster["mean_x"] = new_x
            cluster["mean_y"] = new_y
            cluster["x"] = new_x
            cluster["y"] = new_y
            cluster["last_ts"] = float(frame_ts)
            cluster["v1_hits"] = int(cluster.get("v1_hits", 0)) + int(
                _safe_float(candidate.get("detector_v1", 0.0)) > 0.5
            )
            cluster["v2_hits"] = int(cluster.get("v2_hits", 0)) + int(
                _safe_float(candidate.get("detector_v2", 0.0)) > 0.5
            )
            cluster["tile_hits"] = int(cluster.get("tile_hits", 0)) + int(
                _safe_float(candidate.get("v24_tile_probe", 0.0)) > 0.5
            )
            if quality > float(cluster.get("best_quality", 0.0)):
                cluster["best"] = dict(candidate)
                cluster["best_quality"] = quality

        matched_clusters.add(best_index)
        candidate_cluster[id(candidate)] = clusters[best_index]

    # Prune stale / low-value hypotheses.
    max_age = max(0.1, _safe_float(cfg.get("shot_accumulator_max_age_s", 0.70), 0.70))
    clusters[:] = [
        cluster
        for cluster in clusters
        if frame_ts - float(cluster.get("last_ts", frame_ts)) <= max_age
    ]
    if len(clusters) > max_clusters:
        clusters.sort(
            key=lambda cluster: (
                int(cluster.get("hits", 1)),
                float(cluster.get("best_quality", 0.0)),
                float(cluster.get("last_ts", 0.0)),
            ),
            reverse=True,
        )
        del clusters[max_clusters:]

    # Annotate current candidates by nearest current cluster. This makes temporal
    # stability visible to the ranker even when no carry is needed.
    for candidate in candidates:
        if _safe_float(candidate.get("shot_accumulator_carried", 0.0)) > 0.5:
            continue
        cx = _safe_float(candidate.get("camera_x", 0.0))
        cy = _safe_float(candidate.get("camera_y", 0.0))
        best = None
        best_dist = float("inf")
        for cluster in clusters:
            distance = math.hypot(float(cluster["x"]) - cx, float(cluster["y"]) - cy)
            if distance <= match_radius and distance < best_dist:
                best = cluster
                best_dist = distance
        if best is None:
            continue
        hits = max(1, int(best.get("hits", 1)))
        spread = math.sqrt(max(0.0, float(best.get("m2", 0.0))) / float(max(1, hits - 1)))
        stability = math.exp(-spread / 4.0)
        span = max(0.0, float(best.get("last_ts", frame_ts)) - float(best.get("first_ts", frame_ts)))
        candidate["shot_accumulator_hits"] = float(hits)
        candidate["shot_accumulator_span_s"] = span
        candidate["shot_accumulator_stability"] = _clip01(stability)
        candidate["shot_accumulator_confirmed"] = 1.0 if hits >= 2 else 0.0

    confirm_hits = max(2, _safe_int(cfg.get("shot_accumulator_confirm_hits", 2), 2))
    confirm_span = max(0.0, _safe_float(cfg.get("shot_accumulator_confirm_span_s", 0.014), 0.014))
    single_v1_age = max(0.05, _safe_float(cfg.get("shot_accumulator_single_v1_max_age_s", 0.32), 0.32))
    single_patch_age = max(0.03, _safe_float(cfg.get("shot_accumulator_single_patch_max_age_s", 0.18), 0.18))
    single_patch_prior = _clip01(_safe_float(cfg.get("shot_accumulator_single_patch_min_prior", 0.62), 0.62))
    single_patch_abs = max(0.0, _safe_float(cfg.get("shot_accumulator_single_patch_min_absdiff", 3.2), 3.2))
    min_v1_score = max(0.0, _safe_float(cfg.get("shot_accumulator_min_v1_score", 4.2), 4.2))
    min_prior = _clip01(_safe_float(cfg.get("shot_accumulator_min_patch_prior", 0.38), 0.38))
    carry_slots = max(0, _safe_int(cfg.get("shot_accumulator_carried_slots", 48), 48))
    output_limit = max(len(candidates), _safe_int(cfg.get("shot_accumulator_output_limit", 288), 288))

    carry_options: list[tuple[float, dict[str, Any]]] = []
    for cluster in clusters:
        hits = max(1, int(cluster.get("hits", 1)))
        age = max(0.0, frame_ts - float(cluster.get("last_ts", frame_ts)))
        span = max(0.0, float(cluster.get("last_ts", frame_ts)) - float(cluster.get("first_ts", frame_ts)))
        best_candidate = dict(cluster.get("best", {}))
        if not best_candidate:
            continue
        best_score = _safe_float(best_candidate.get("score", 0.0))
        prior = _safe_float(best_candidate.get("v24_patch_prior", _patch_prior(best_candidate)))
        v1_hits = int(cluster.get("v1_hits", 0))
        repeated = hits >= confirm_hits and span >= confirm_span
        strong_single_v1 = (
            hits == 1
            and v1_hits >= 1
            and age <= single_v1_age
            and (best_score >= min_v1_score or prior >= min_prior)
        )
        # A compact, locally hole-like temporal patch may only be visible in one
        # camera frame. Carry such a hypothesis very briefly even if it came
        # from V2/tile rather than V1. The high patch-prior + absdiff gate keeps
        # this stricter than ordinary candidate generation.
        strong_single_patch = (
            hits == 1
            and age <= single_patch_age
            and prior >= single_patch_prior
            and _safe_float(best_candidate.get("v2_absdiff", best_candidate.get("v24_patch_core_abs", 0.0))) >= single_patch_abs
        )
        if not repeated and not strong_single_v1 and not strong_single_patch:
            continue

        cx = float(cluster.get("x", 0.0))
        cy = float(cluster.get("y", 0.0))
        if any(
            math.hypot(
                _safe_float(candidate.get("camera_x", 0.0)) - cx,
                _safe_float(candidate.get("camera_y", 0.0)) - cy,
            ) < 4.5
            for candidate in candidates
        ):
            continue

        spread = math.sqrt(max(0.0, float(cluster.get("m2", 0.0))) / float(max(1, hits - 1)))
        stability = math.exp(-spread / 4.0)
        confidence = (
            1.25 * min(4, hits)
            + 1.30 * float(cluster.get("best_quality", 0.0))
            + 0.50 * min(3, v1_hits)
            + 0.25 * min(3, int(cluster.get("v2_hits", 0)))
            + 0.20 * min(3, int(cluster.get("tile_hits", 0)))
            + 0.45 * stability
            - 0.80 * age
        )
        carry_options.append((confidence, cluster))

    carry_options.sort(key=lambda item: item[0], reverse=True)
    output = [dict(candidate) for candidate in candidates]
    for _confidence, cluster in carry_options[:carry_slots]:
        if len(output) >= output_limit:
            break
        best = dict(cluster.get("best", {}))
        hits = max(1, int(cluster.get("hits", 1)))
        spread = math.sqrt(max(0.0, float(cluster.get("m2", 0.0))) / float(max(1, hits - 1)))
        stability = math.exp(-spread / 4.0)
        best["camera_x"] = float(cluster.get("x", best.get("camera_x", 0.0)))
        best["camera_y"] = float(cluster.get("y", best.get("camera_y", 0.0)))
        best["shot_accumulator_hits"] = float(hits)
        best["shot_accumulator_span_s"] = max(
            0.0,
            float(cluster.get("last_ts", frame_ts)) - float(cluster.get("first_ts", frame_ts)),
        )
        best["shot_accumulator_stability"] = _clip01(stability)
        best["shot_accumulator_confirmed"] = 1.0 if hits >= confirm_hits else 0.0
        best["shot_accumulator_carried"] = 1.0
        best["score"] = max(
            3.8,
            0.90 * _safe_float(best.get("score", 4.0)) + min(1.5, 0.25 * max(0, hits - 1)),
        )
        output.append(best)

    return output


def _attach_benchmark_seed(record: dict[str, Any]) -> None:
    gt = record.get("ground_truth")
    if not isinstance(gt, dict) or gt.get("benchmark_seed") is not None:
        return
    try:
        if not BENCHMARK_CONTROL_PATH.exists():
            return
        payload = json.loads(BENCHMARK_CONTROL_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and bool(payload.get("enabled", False)):
            gt["benchmark_seed"] = int(payload.get("seed"))
    except Exception:
        pass


def _ensure_diag_record(engine: Any, scanner: Any, shot_id: int) -> dict[str, Any] | None:
    if shot_id <= 0:
        return None
    records = getattr(engine, "_diagnostics", None)
    if not isinstance(records, dict):
        return None
    record = records.get(shot_id)
    if isinstance(record, dict):
        _attach_benchmark_seed(record)
        return record

    gt = getattr(scanner, "_detector_v2_ground_truth", None)
    gt_copy = dict(gt) if isinstance(gt, dict) and int(gt.get("shot_id", -1)) == shot_id else None
    record = {
        "schema_version": "2.4",
        "runtime_session_id": getattr(engine, "_runtime_session_id", "unknown"),
        "shot_id": shot_id,
        "created_at": time.time(),
        "git_commit": getattr(engine, "_git_commit", None),
        "frames_seen": 0,
        "max_counts": {"legacy": 0, "v2": 0, "merged": 0},
        "signal_max": {"absdiff": 0.0, "zscore": 0.0, "saliency": 0.0},
        "registration": {"applied_frames": 0, "best_response": 0.0, "max_abs_dx": 0.0, "max_abs_dy": 0.0},
        "ground_truth": gt_copy,
        "nearest_candidate_distance_px": {"legacy": None, "v2_frame": None, "v2": None, "merged": None},
        "gt_signal_max": {"absdiff": None, "zscore": None, "saliency": None},
        "v24": {},
    }
    records[shot_id] = record
    _attach_benchmark_seed(record)
    return record


def _nearest(candidates: list[dict[str, Any]], gt_x: float, gt_y: float) -> float | None:
    if not candidates:
        return None
    return min(
        math.hypot(
            _safe_float(candidate.get("camera_x", 0.0)) - gt_x,
            _safe_float(candidate.get("camera_y", 0.0)) - gt_y,
        )
        for candidate in candidates
    )


def build_ground_truth_patch_candidate(engine: Any, gt_xy: tuple[float, float]) -> dict[str, Any] | None:
    """Build a TRAINING-ONLY descriptor exactly at synthetic camera-space GT.

    This candidate is never injected into detection/ranking. It exists solely
    so Ranker V4 can learn the visual shape at the true point instead of
    treating an arbitrary candidate within the loose 42 px evaluation radius as
    a positive label.
    """
    context = getattr(engine, "_v24_last_context", None)
    cfg_obj = getattr(engine, "_v24_config", None)
    if not isinstance(context, dict) or cfg_obj is None:
        return None
    try:
        cfg = cfg_obj.snapshot()
        bbox = context.get("bbox")
        absdiff = context.get("absdiff")
        darkening = context.get("darkening")
        zscore = context.get("zscore")
        dog = context.get("dog")
        saliency = context.get("saliency")
        if (
            not isinstance(bbox, tuple)
            or len(bbox) != 4
            or not isinstance(absdiff, np.ndarray)
            or not isinstance(darkening, np.ndarray)
            or not isinstance(zscore, np.ndarray)
        ):
            return None
        x0, y0, _x1, _y1 = bbox
        px = int(round(float(gt_xy[0]))) - int(x0)
        py = int(round(float(gt_xy[1]))) - int(y0)
        if py < 0 or px < 0 or py >= absdiff.shape[0] or px >= absdiff.shape[1]:
            return None
        candidate: dict[str, Any] = {
            "camera_x": float(gt_xy[0]),
            "camera_y": float(gt_xy[1]),
            "training_ground_truth_patch": 1.0,
            "v2_absdiff": float(absdiff[py, px]),
            "v2_zscore": float(zscore[py, px]),
            "v2_dog": float(dog[py, px]) if isinstance(dog, np.ndarray) else 0.0,
            "v2_saliency": float(saliency[py, px]) if isinstance(saliency, np.ndarray) else 0.0,
        }
        candidate.update(
            _patch_descriptor(
                px=px,
                py=py,
                absdiff=absdiff,
                darkening=darkening,
                zscore=zscore,
                cfg=cfg,
            )
        )
        candidate["v24_patch_prior"] = _patch_prior(candidate)
        return candidate
    except Exception:
        return None


def patch_candidate_generator_class(candidate_generator_cls: type) -> None:
    if bool(getattr(candidate_generator_cls, "_detector_v24_extension_installed", False)):
        return

    original_init = candidate_generator_cls.__init__
    original_extract = candidate_generator_cls._extract_candidates
    original_generate = candidate_generator_cls.generate
    original_reset = candidate_generator_cls.reset_runtime_state
    original_flush = candidate_generator_cls.flush_resolved_shots
    original_record_funnel = candidate_generator_cls.record_funnel_evaluation
    original_record_empty = candidate_generator_cls.record_empty_evaluation

    def init_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._v24_config = V24Config()
        self._v24_last_context = None
        self._v24_last_tile_probes = []
        self._v24_shot_accumulators = {}

    def extract_wrapped(self: Any, *args: Any, **kwargs: Any) -> list[dict[str, float]]:
        candidates = original_extract(self, *args, **kwargs)
        try:
            cfg = self._v24_config.snapshot()
            if not bool(cfg.get("enabled", True)):
                return candidates
            required = ("scanner", "saliency", "absdiff", "darkening", "dog", "zscore", "valid", "bbox", "frame_ts")
            if not all(key in kwargs for key in required):
                return candidates
            context = {
                "bbox": tuple(kwargs["bbox"]),
                "absdiff": kwargs["absdiff"],
                "darkening": kwargs["darkening"],
                "zscore": kwargs["zscore"],
                "dog": kwargs["dog"],
                "saliency": kwargs["saliency"],
                "valid": kwargs["valid"],
                "frame_ts": float(kwargs["frame_ts"]),
            }
            self._v24_last_context = context
            for candidate in candidates:
                _annotate_candidate(candidate, context=context, cfg=cfg)
            self._v24_last_tile_probes = _tile_probe_candidates(
                self,
                scanner=kwargs["scanner"],
                saliency=kwargs["saliency"],
                absdiff=kwargs["absdiff"],
                darkening=kwargs["darkening"],
                dog=kwargs["dog"],
                zscore=kwargs["zscore"],
                valid=kwargs["valid"],
                bbox=tuple(kwargs["bbox"]),
                frame_ts=float(kwargs["frame_ts"]),
                cfg=cfg,
            )
        except Exception as exc:
            self._v24_last_tile_probes = []
            if bool(getattr(kwargs.get("scanner"), "shot_diag_enabled", False)):
                print(f"[DETECTOR-V2.4] extraction extension fallback: {exc}")
        return candidates

    def generate_wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_generate(self, *args, **kwargs)
        try:
            cfg = self._v24_config.snapshot()
            if not bool(cfg.get("enabled", True)):
                return result

            scanner = kwargs.get("scanner")
            frame_ts = float(kwargs.get("frame_ts", time.time()))
            current = [dict(candidate) for candidate in list(result.candidates)]
            context = getattr(self, "_v24_last_context", None)
            if isinstance(context, dict) and abs(float(context.get("frame_ts", -1.0)) - frame_ts) < 1e-6:
                for candidate in current:
                    _annotate_candidate(candidate, context=context, cfg=cfg)

            tile_probes = [dict(candidate) for candidate in getattr(self, "_v24_last_tile_probes", [])]
            reserve = max(0, _safe_int(cfg.get("tile_probe_reserved_slots", 40), 40))
            nms_radius = max(1.0, _safe_float(cfg.get("tile_probe_nms_radius_px", 4.0), 4.0))
            current = _merge_reserved(current, tile_probes, reserve=reserve, radius=nms_radius)

            telemetry = dict(getattr(result, "telemetry", {}) or {})
            shot_id = _safe_int(telemetry.get("shot_id", 0), 0)
            current = _update_shot_accumulator(
                self,
                shot_id=shot_id,
                candidates=current,
                frame_ts=frame_ts,
                cfg=cfg,
            )

            telemetry["schema_version_v24"] = "2.4"
            telemetry["v24_tile_probe_count"] = len(tile_probes)
            telemetry["v24_final_count"] = len(current)
            telemetry["v24_accumulator_clusters"] = len(
                getattr(self, "_v24_shot_accumulators", {}).get(shot_id, [])
            ) if shot_id > 0 else 0

            # Record BEST/EVER V2.4 paths separately from the legacy V2 fields.
            if scanner is not None and shot_id > 0:
                record = _ensure_diag_record(self, scanner, shot_id)
                gt = getattr(scanner, "_detector_v2_ground_truth", None)
                if isinstance(record, dict) and isinstance(gt, dict) and int(gt.get("shot_id", -1)) == shot_id:
                    gt_x = _safe_float(gt.get("camera_x", 0.0))
                    gt_y = _safe_float(gt.get("camera_y", 0.0))
                    v24 = record.setdefault("v24", {})
                    for key, candidates_for_key in (
                        ("tile_probe_nearest_px", tile_probes),
                        ("final_nearest_px", current),
                        (
                            "accumulator_nearest_px",
                            [c for c in current if _safe_float(c.get("shot_accumulator_carried", 0.0)) > 0.5],
                        ),
                    ):
                        value = _nearest(candidates_for_key, gt_x, gt_y)
                        if value is None:
                            continue
                        previous = v24.get(key)
                        v24[key] = value if previous is None else min(float(previous), value)
                    v24["max_tile_probe_count"] = max(
                        int(v24.get("max_tile_probe_count", 0)), len(tile_probes)
                    )
                    v24["max_final_count"] = max(int(v24.get("max_final_count", 0)), len(current))
                    v24["max_accumulator_count"] = max(
                        int(v24.get("max_accumulator_count", 0)),
                        sum(1 for c in current if _safe_float(c.get("shot_accumulator_carried", 0.0)) > 0.5),
                    )

            return type(result)(candidates=current, telemetry=telemetry)
        except Exception as exc:
            scanner = kwargs.get("scanner")
            if bool(getattr(scanner, "shot_diag_enabled", False)):
                print(f"[DETECTOR-V2.4] generate extension fallback: {exc}")
            return result

    def record_funnel_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        scanner = kwargs.get("scanner")
        gt_xy = kwargs.get("gt_xy")
        if scanner is not None and gt_xy is not None:
            gt = getattr(scanner, "_detector_v2_ground_truth", None)
            shot_id = _safe_int(gt.get("shot_id", 0), 0) if isinstance(gt, dict) else 0
            _ensure_diag_record(self, scanner, shot_id)

        original_record_funnel(self, *args, **kwargs)

        try:
            if scanner is None or gt_xy is None:
                return
            gt = getattr(scanner, "_detector_v2_ground_truth", None)
            if not isinstance(gt, dict):
                return
            shot_id = _safe_int(gt.get("shot_id", 0), 0)
            record = getattr(self, "_diagnostics", {}).get(shot_id)
            if not isinstance(record, dict):
                return
            funnel = record.setdefault("evaluation_funnel", {})
            raw = [dict(c) for c in kwargs.get("raw_hotspots", [])]
            gt_x, gt_y = float(gt_xy[0]), float(gt_xy[1])

            accumulator = [c for c in raw if _safe_float(c.get("shot_accumulator_carried", 0.0)) > 0.5]
            tile = [c for c in raw if _safe_float(c.get("v24_tile_probe", 0.0)) > 0.5]
            funnel["raw_v24_tile_count"] = len(tile)
            funnel["raw_v24_tile_nearest_px"] = _nearest(tile, gt_x, gt_y)
            funnel["raw_v24_accumulator_count"] = len(accumulator)
            funnel["raw_v24_accumulator_nearest_px"] = _nearest(accumulator, gt_x, gt_y)

            try:
                from src.engine.ai.runtime import get_ai_runtime

                runtime = get_ai_runtime()
                pool = [dict(c) for c in getattr(runtime, "_v24_last_rank_pool", []) or []]
                if pool:
                    gt_candidate = min(
                        pool,
                        key=lambda c: math.hypot(
                            _safe_float(c.get("camera_x", 0.0)) - gt_x,
                            _safe_float(c.get("camera_y", 0.0)) - gt_y,
                        ),
                    )
                    gt_distance = math.hypot(
                        _safe_float(gt_candidate.get("camera_x", 0.0)) - gt_x,
                        _safe_float(gt_candidate.get("camera_y", 0.0)) - gt_y,
                    )
                    selected = pool[0]
                    keys = [
                        "v24_combined_score",
                        "v24_patch_prior",
                        "ranker_v4_score",
                        "ranker_v4_raw",
                        "ranker_v4_weight",
                        "v24_patch_core_to_outer",
                        "v24_patch_compactness",
                        "v24_patch_centeredness",
                        "v24_patch_isotropy",
                        "v24_patch_bipolar",
                        "v24_patch_local_snr",
                        "shot_accumulator_hits",
                        "shot_accumulator_stability",
                        "v24_tile_probe",
                    ]
                    funnel["v24_ranking"] = {
                        "pool_count": len(pool),
                        "gt_distance_px": gt_distance,
                        "gt_rank": int(gt_candidate.get("rank", 0) or 0),
                        "gt": {key: _safe_float(gt_candidate.get(key, 0.0)) for key in keys},
                        "selected": {key: _safe_float(selected.get(key, 0.0)) for key in keys},
                    }
                    funnel["v24_ranking"]["selected_minus_gt"] = {
                        key: _safe_float(selected.get(key, 0.0)) - _safe_float(gt_candidate.get(key, 0.0))
                        for key in keys
                    }
                model = getattr(runtime, "_ranker_v4", None)
                if model is not None:
                    record["ranker_v4_summary"] = model.summary()
            except Exception:
                pass
        except Exception:
            pass

    def record_empty_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        scanner = kwargs.get("scanner")
        if scanner is not None:
            gt = getattr(scanner, "_detector_v2_ground_truth", None)
            shot_id = _safe_int(gt.get("shot_id", 0), 0) if isinstance(gt, dict) else 0
            _ensure_diag_record(self, scanner, shot_id)
        original_record_empty(self, *args, **kwargs)

    def flush_wrapped(self: Any, scanner: Any) -> None:
        original_flush(self, scanner)
        try:
            states = getattr(self, "_v24_shot_accumulators", {})
            written = set(getattr(self, "_diagnostics_written", set()) or set())
            for shot_id in list(states.keys()):
                if shot_id in written:
                    states.pop(shot_id, None)
        except Exception:
            pass

    def force_finalize_benchmark_diagnostics(self: Any, scanner: Any) -> dict[str, int]:
        """Write every labelled diagnostic still buffered at F2 run completion.

        The core detector normally waits for scanner resolution before writing a
        JSONL record. A very fast synthetic run can leave the last few labelled
        shots buffered when the scene immediately advances to the next run.
        Benchmark data is more valuable complete than delayed, so the automation
        scene calls this after all 100 rounds have already been evaluated.
        """
        records = getattr(self, "_diagnostics", None)
        written = getattr(self, "_diagnostics_written", None)
        if not isinstance(records, dict) or not isinstance(written, set):
            return {"finalized": 0, "missing_evaluation": 0}

        finalized = 0
        missing_evaluation = 0
        for shot_id, record in list(records.items()):
            if shot_id in written or not isinstance(record, dict):
                continue
            if not isinstance(record.get("ground_truth"), dict):
                continue

            if not isinstance(record.get("evaluation_funnel"), dict):
                missing_evaluation += 1
                record["benchmark_integrity"] = "missing_evaluation"
                record["evaluation_funnel"] = {
                    "captured_at": time.time(),
                    "raw_count": 0,
                    "raw_nearest_px": None,
                    "ranked_count": 0,
                    "ranked_nearest_px": None,
                    "selected_nearest_px": None,
                    "integrity_placeholder": True,
                }
            else:
                record["benchmark_integrity"] = "complete"

            if not isinstance(record.get("resolved"), dict):
                record["resolved"] = {
                    "state": "benchmark_completed",
                    "emitted": False,
                    "confidence": 0.0,
                    "note": "forced at completed synthetic F2 report",
                    "matched_track_id": None,
                    "matched_hole_id": None,
                }
                record["resolved_at"] = time.time()

            try:
                self._finalize_diagnostic_record(scanner, int(shot_id), record)
                finalized += 1
            except Exception:
                pass

        return {
            "finalized": finalized,
            "missing_evaluation": missing_evaluation,
        }

    def reset_wrapped(self: Any) -> None:
        original_reset(self)
        self._v24_last_context = None
        self._v24_last_tile_probes = []
        self._v24_shot_accumulators = {}

    candidate_generator_cls.__init__ = init_wrapped
    candidate_generator_cls._extract_candidates = extract_wrapped
    candidate_generator_cls.generate = generate_wrapped
    candidate_generator_cls.record_funnel_evaluation = record_funnel_wrapped
    candidate_generator_cls.record_empty_evaluation = record_empty_wrapped
    candidate_generator_cls.flush_resolved_shots = flush_wrapped
    candidate_generator_cls.force_finalize_benchmark_diagnostics = force_finalize_benchmark_diagnostics
    candidate_generator_cls.reset_runtime_state = reset_wrapped
    candidate_generator_cls._detector_v24_extension_installed = True


def apply_detector_v24_extension() -> None:
    from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2

    patch_candidate_generator_class(CandidateGeneratorV2)
    print("[DETECTOR-V2.4] local-tile + shot-accumulator extension installed")


__all__ = [
    "apply_detector_v24_extension",
    "patch_candidate_generator_class",
    "_patch_descriptor",
    "build_ground_truth_patch_candidate",
]
