from __future__ import annotations

"""V2.21.3 anchored temporal consensus proposals.

V2.21.2 proved two things on the first honest 30-shot full-frame
projector/camera session:

* global full-frame top-N proposals are dominated by room/projector nuisance;
* local PRE->POST evidence around an existing V1/V2 anchor is useful and rescued
  11/30 shots at <=20 px.

This module keeps the useful anchor prior, but fixes two weaknesses in V2.21.2:

1. saturated/flat local-max plateaus are collapsed into connected components
   instead of selecting arbitrary equal-valued pixels;
2. evidence is clustered across multiple temporal maps and compact components
   are preferred over long horizontal/vertical bands.

Ground truth is never accepted by any proposal function in this module.
Everything remains offline/shadow-only.
"""

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class TemporalConsensusConfigV2213:
    search_radius_px: int = 54
    local_max_kernel: int = 3
    threshold_percentile: float = 92.0
    minimum_local_contrast: float = 0.018
    components_per_source: int = 5
    top_per_anchor: int = 2
    component_max_area: int = 220
    component_min_compactness: float = 0.12
    cluster_radius_px: float = 5.0
    final_nms_radius_px: float = 4.0
    proposal_limit: int = 1200
    distance_prior_weight: float = 0.08
    source_names: tuple[str, ...] = (
        "blackhat_gain",
        "tophat_gain",
        "persistent_abs",
        "gradient_gain",
        "persistent_dark",
    )
    source_weights: dict[str, float] = field(default_factory=lambda: {
        "blackhat_gain": 1.00,
        "tophat_gain": 0.95,
        "persistent_abs": 0.82,
        "gradient_gain": 0.78,
        "persistent_dark": 0.72,
    })


@dataclass(frozen=True)
class MaskedDirectConfigV2213:
    margin_px: int = 34
    source_names: tuple[str, ...] = (
        "blackhat_gain",
        "tophat_gain",
        "persistent_abs",
        "gradient_gain",
    )
    source_weights: dict[str, float] = field(default_factory=lambda: {
        "blackhat_gain": 1.00,
        "tophat_gain": 0.95,
        "persistent_abs": 0.82,
        "gradient_gain": 0.78,
    })
    threshold_percentile: float = 99.0
    minimum_score: float = 0.15
    component_max_area: int = 260
    component_min_compactness: float = 0.10
    cluster_radius_px: float = 6.0
    proposal_limit: int = 240


def _as_map(name: str, maps: Mapping[str, np.ndarray]) -> np.ndarray | None:
    value = maps.get(name)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    return arr if arr.ndim == 2 else None


def _roi_bounds(cx: float, cy: float, radius: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, int(math.floor(cx - radius)))
    y0 = max(0, int(math.floor(cy - radius)))
    x1 = min(width, int(math.ceil(cx + radius + 1)))
    y1 = min(height, int(math.ceil(cy + radius + 1)))
    return x0, y0, x1, y1


def _component_candidates(
    values: np.ndarray,
    *,
    cx: float,
    cy: float,
    radius: int,
    percentile: float,
    minimum_local_contrast: float,
    local_max_kernel: int,
    max_components: int,
    max_area: int,
    min_compactness: float,
) -> list[dict[str, float]]:
    h, w = values.shape[:2]
    x0, y0, x1, y1 = _roi_bounds(cx, cy, radius, w, h)
    if x1 <= x0 or y1 <= y0:
        return []
    roi = np.asarray(values[y0:y1, x0:x1], dtype=np.float32)
    if roi.size == 0:
        return []

    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= float(radius * radius)
    sample = roi[circle]
    if not sample.size:
        return []

    baseline = float(np.median(sample))
    upper = float(np.percentile(sample, 99.5))
    if upper - baseline < float(minimum_local_contrast):
        return []
    threshold = max(
        baseline + float(minimum_local_contrast),
        float(np.percentile(sample, float(np.clip(percentile, 50.0, 99.9)))),
    )

    k = max(1, int(local_max_kernel)) | 1
    dilated = cv2.dilate(roi, np.ones((k, k), dtype=np.uint8))
    maxima = (roi >= threshold) & (roi >= dilated - 1e-7) & circle
    if not np.any(maxima):
        return []

    # Critical V2.21.3 change: equal-valued plateaus become ONE object.  V2.21.2
    # could choose arbitrary pixels from a saturated line/plateau.
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(maxima.astype(np.uint8), connectivity=8)
    rows: list[dict[str, float]] = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0 or area > max(1, int(max_area)):
            continue
        bw = max(1, int(stats[label, cv2.CC_STAT_WIDTH]))
        bh = max(1, int(stats[label, cv2.CC_STAT_HEIGHT]))
        aspect = float(min(bw, bh) / max(bw, bh))
        fill = float(area / max(1, bw * bh))
        compactness = float(math.sqrt(max(0.0, aspect * fill)))
        if compactness < float(min_compactness):
            continue

        mask = labels == label
        component_values = roi[mask]
        if not component_values.size:
            continue
        peak = float(np.max(component_values))
        contrast = float(max(0.0, peak - baseline))

        # Use the highest-value pixels to define a stable sub-pixel-ish centre,
        # rather than the centroid of an elongated plateau.
        coords_y, coords_x = np.nonzero(mask)
        weights = np.maximum(roi[coords_y, coords_x] - baseline, 1e-6)
        keep = roi[coords_y, coords_x] >= peak - max(0.01, 0.12 * max(contrast, 1e-6))
        if np.any(keep):
            coords_y = coords_y[keep]
            coords_x = coords_x[keep]
            weights = weights[keep]
        sw = float(np.sum(weights))
        if sw > 0:
            lx = float(np.sum(coords_x * weights) / sw)
            ly = float(np.sum(coords_y * weights) / sw)
        else:
            lx, ly = float(centroids[label][0]), float(centroids[label][1])
        x = float(x0 + lx)
        y = float(y0 + ly)
        shift = float(math.hypot(x - cx, y - cy))
        if shift > radius + 0.5:
            continue

        rows.append({
            "camera_x": x,
            "camera_y": y,
            "peak": peak,
            "contrast": contrast,
            "compactness": compactness,
            "area": float(area),
            "shift_px": shift,
            "quality": float(0.52 * contrast + 0.30 * peak + 0.18 * compactness),
        })

    rows.sort(key=lambda r: float(r["quality"]), reverse=True)
    return rows[: max(1, int(max_components))]


def _cluster_source_rows(rows: Sequence[dict[str, Any]], radius: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    r2 = float(max(0.5, radius) ** 2)
    ordered = sorted(rows, key=lambda r: float(r.get("source_quality", 0.0)), reverse=True)
    clusters: list[dict[str, Any]] = []
    for row in ordered:
        x = float(row["camera_x"]); y = float(row["camera_y"])
        match = None
        for idx, old in enumerate(clusters):
            if (x - float(old["camera_x"])) ** 2 + (y - float(old["camera_y"])) ** 2 <= r2:
                match = idx
                break
        if match is None:
            clusters.append({
                "camera_x": x,
                "camera_y": y,
                "members": [dict(row)],
                "sources": {str(row["source_name"])},
            })
        else:
            clusters[match]["members"].append(dict(row))
            clusters[match]["sources"].add(str(row["source_name"]))

    out: list[dict[str, Any]] = []
    for cluster in clusters:
        members = cluster["members"]
        weights = np.asarray([max(1e-6, float(m.get("source_quality", 0.0))) for m in members], dtype=np.float64)
        xs = np.asarray([float(m["camera_x"]) for m in members], dtype=np.float64)
        ys = np.asarray([float(m["camera_y"]) for m in members], dtype=np.float64)
        sw = float(np.sum(weights))
        x = float(np.sum(xs * weights) / sw)
        y = float(np.sum(ys * weights) / sw)
        source_support = len(cluster["sources"])
        best_quality = max(float(m.get("source_quality", 0.0)) for m in members)
        mean_compactness = float(np.mean([float(m.get("compactness", 0.0)) for m in members]))
        out.append({
            "camera_x": x,
            "camera_y": y,
            "source_support": int(source_support),
            "best_quality": best_quality,
            "mean_compactness": mean_compactness,
            "evidence_sources": sorted(cluster["sources"]),
        })
    return out


def _final_nms(rows: Sequence[dict[str, Any]], radius: float, limit: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)
    selected: list[dict[str, Any]] = []
    r2 = float(max(0.5, radius) ** 2)
    for row in ordered:
        x = float(row["camera_x"]); y = float(row["camera_y"])
        if any((x - float(old["camera_x"])) ** 2 + (y - float(old["camera_y"])) ** 2 <= r2 for old in selected):
            continue
        selected.append(dict(row))
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def propose_temporal_consensus_v2213(
    current_candidates: Sequence[dict[str, Any]],
    maps: Mapping[str, np.ndarray],
    *,
    config: TemporalConsensusConfigV2213 | None = None,
) -> list[dict[str, Any]]:
    """Return evidence-backed refined points near current candidates.

    The anchor is only a spatial prior.  Every returned coordinate must be a
    temporal-map component; no geometric padding/grid points are emitted.
    """
    cfg = config or TemporalConsensusConfigV2213()
    output: list[dict[str, Any]] = []
    for parent_index, parent in enumerate(current_candidates):
        cx = float(parent.get("camera_x", 0.0)); cy = float(parent.get("camera_y", 0.0))
        source_rows: list[dict[str, Any]] = []
        for source_name in cfg.source_names:
            values = _as_map(source_name, maps)
            if values is None:
                continue
            comps = _component_candidates(
                values,
                cx=cx,
                cy=cy,
                radius=max(1, int(cfg.search_radius_px)),
                percentile=float(cfg.threshold_percentile),
                minimum_local_contrast=float(cfg.minimum_local_contrast),
                local_max_kernel=int(cfg.local_max_kernel),
                max_components=int(cfg.components_per_source),
                max_area=int(cfg.component_max_area),
                min_compactness=float(cfg.component_min_compactness),
            )
            source_weight = max(0.0, float(cfg.source_weights.get(source_name, 1.0)))
            for comp in comps:
                row = dict(comp)
                row["source_name"] = source_name
                row["source_quality"] = float(source_weight * comp["quality"])
                source_rows.append(row)

        clusters = _cluster_source_rows(source_rows, cfg.cluster_radius_px)
        ranked: list[dict[str, Any]] = []
        for cluster in clusters:
            shift = float(math.hypot(float(cluster["camera_x"]) - cx, float(cluster["camera_y"]) - cy))
            distance_prior = max(0.0, 1.0 - shift / max(1.0, float(cfg.search_radius_px)))
            support = min(1.0, float(cluster["source_support"]) / 3.0)
            score01 = (
                0.46 * support
                + 0.34 * min(1.0, float(cluster["best_quality"]))
                + 0.12 * min(1.0, float(cluster["mean_compactness"]))
                + float(cfg.distance_prior_weight) * distance_prior
            )
            ranked.append({
                "camera_x": float(cluster["camera_x"]),
                "camera_y": float(cluster["camera_y"]),
                "score": float(100.0 * score01),
                "parent_index": int(parent_index),
                "parent_camera_x": cx,
                "parent_camera_y": cy,
                "local_shift_px": shift,
                "temporal_consensus_support": int(cluster["source_support"]),
                "evidence_source": "temporal_consensus_v2213",
                "evidence_sources": [f"temporal_consensus_v2213:{name}" for name in cluster["evidence_sources"]],
                "ai_temporal_consensus_v2213": 1.0,
            })
        ranked.sort(key=lambda r: float(r["score"]), reverse=True)
        output.extend(ranked[: max(1, int(cfg.top_per_anchor))])

    return _final_nms(output, cfg.final_nms_radius_px, cfg.proposal_limit)


def candidate_target_mask_v2213(
    current_candidates: Sequence[dict[str, Any]],
    shape: tuple[int, int],
    *,
    margin_px: int = 34,
) -> np.ndarray:
    """Infer a conservative target/search mask from the current candidate cloud.

    This is intentionally GT-free.  The first V2.21 debug images showed that
    global temporal maps were dominated by room/cabinet edges while the legacy
    detector already concentrates its candidate cloud on the projected target.
    """
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray([
        [float(c.get("camera_x", 0.0)), float(c.get("camera_y", 0.0))]
        for c in current_candidates
        if math.isfinite(float(c.get("camera_x", 0.0))) and math.isfinite(float(c.get("camera_y", 0.0)))
    ], dtype=np.float32)
    if len(pts) < 3:
        mask[:] = 1
        return mask.astype(bool)

    # Remove only extreme outliers before making the hull.  This keeps the mask
    # broad enough to rescue a missed hole while preventing room edges from
    # owning the global top-N list.
    lo = np.percentile(pts, 1.0, axis=0)
    hi = np.percentile(pts, 99.0, axis=0)
    keep = (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1])
    core = pts[keep]
    if len(core) < 3:
        core = pts
    hull = cv2.convexHull(np.round(core).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    margin = max(0, int(margin_px))
    if margin:
        k = margin * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


def _masked_component_points(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    percentile: float,
    minimum_score: float,
    max_area: int,
    min_compactness: float,
    limit: int,
) -> list[dict[str, float]]:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(arr)
    sample = arr[valid]
    if not sample.size:
        return []
    threshold = max(float(minimum_score), float(np.percentile(sample, float(np.clip(percentile, 50.0, 99.99)))))
    binary = valid & (arr >= threshold)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    rows: list[dict[str, float]] = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0 or area > max(1, int(max_area)):
            continue
        bw = max(1, int(stats[label, cv2.CC_STAT_WIDTH])); bh = max(1, int(stats[label, cv2.CC_STAT_HEIGHT]))
        aspect = float(min(bw, bh) / max(bw, bh)); fill = float(area / max(1, bw * bh))
        compactness = float(math.sqrt(max(0.0, aspect * fill)))
        if compactness < float(min_compactness):
            continue
        yy, xx = np.nonzero(labels == label)
        vals = arr[yy, xx]
        if not len(vals):
            continue
        peak = float(np.max(vals))
        weights = np.maximum(vals - threshold + 0.01, 1e-6)
        sw = float(np.sum(weights))
        x = float(np.sum(xx * weights) / sw); y = float(np.sum(yy * weights) / sw)
        rows.append({
            "camera_x": x,
            "camera_y": y,
            "peak": peak,
            "compactness": compactness,
            "quality": float(0.70 * peak + 0.30 * compactness),
        })
    rows.sort(key=lambda r: float(r["quality"]), reverse=True)
    return rows[: max(1, int(limit))]


def propose_masked_direct_v2213(
    current_candidates: Sequence[dict[str, Any]],
    maps: Mapping[str, np.ndarray],
    *,
    config: MaskedDirectConfigV2213 | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Global rescue proposals, but only inside a GT-free target mask."""
    cfg = config or MaskedDirectConfigV2213()
    first = next((np.asarray(v) for v in maps.values() if isinstance(v, np.ndarray) and v.ndim == 2), None)
    if first is None:
        return [], np.zeros((1, 1), dtype=bool)
    mask = candidate_target_mask_v2213(current_candidates, first.shape[:2], margin_px=cfg.margin_px)
    raw: list[dict[str, Any]] = []
    per_source = max(16, int(cfg.proposal_limit))
    for source_name in cfg.source_names:
        values = _as_map(source_name, maps)
        if values is None:
            continue
        points = _masked_component_points(
            values,
            mask,
            percentile=float(cfg.threshold_percentile),
            minimum_score=float(cfg.minimum_score),
            max_area=int(cfg.component_max_area),
            min_compactness=float(cfg.component_min_compactness),
            limit=per_source,
        )
        weight = max(0.0, float(cfg.source_weights.get(source_name, 1.0)))
        for point in points:
            raw.append({
                **point,
                "source_name": source_name,
                "source_quality": float(weight * point["quality"]),
            })
    clusters = _cluster_source_rows(raw, cfg.cluster_radius_px)
    rows: list[dict[str, Any]] = []
    for cluster in clusters:
        support = min(1.0, float(cluster["source_support"]) / 3.0)
        score01 = 0.58 * support + 0.34 * min(1.0, float(cluster["best_quality"])) + 0.08 * min(1.0, float(cluster["mean_compactness"]))
        rows.append({
            "camera_x": float(cluster["camera_x"]),
            "camera_y": float(cluster["camera_y"]),
            "score": float(100.0 * score01),
            "temporal_consensus_support": int(cluster["source_support"]),
            "evidence_source": "masked_direct_v2213",
            "evidence_sources": [f"masked_direct_v2213:{name}" for name in cluster["evidence_sources"]],
            "ai_masked_direct_v2213": 1.0,
        })
    return _final_nms(rows, cfg.cluster_radius_px, cfg.proposal_limit), mask


def consensus_config_dict_v2213(config: TemporalConsensusConfigV2213 | None = None) -> dict[str, Any]:
    return asdict(config or TemporalConsensusConfigV2213())


def masked_direct_config_dict_v2213(config: MaskedDirectConfigV2213 | None = None) -> dict[str, Any]:
    return asdict(config or MaskedDirectConfigV2213())
