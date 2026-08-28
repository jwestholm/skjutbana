from __future__ import annotations

"""V2.21.5 candidate-aligned dense proposal + listwise ranking helpers.

The V2.21.4 experiment proved that a broad GT-free temporal pool can contain
an honest physical hit at high recall, but its learned ranker discarded the
right proposals. V2.21.5 therefore keeps the *proposal* problem and *ranking*
problem strictly separated:

* ``propose_dense_pool_v2215`` never receives GT and emits only coordinates
  supported by real PRE->POST temporal evidence inside a conservative target
  mask inferred from the existing V1/V2 candidate cloud.
* ``extract_candidate_features_v2215`` is also GT-free. It produces the exact
  same feature representation at train and inference time.
* ``fit_listwise_ranker_v2215`` trains only on candidates that genuinely came
  out of the dense pool. There are no forced GT points and no GT jitter rows.

Ground truth is accepted only by the explicit training/evaluation helpers that
construct labels after the candidate pool already exists.

This module is offline/shadow-only and does not change live hit authority.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .temporal_consensus_v2213 import candidate_target_mask_v2213


MAP_NAMES: tuple[str, ...] = (
    "blackhat_gain",
    "tophat_gain",
    "persistent_abs",
    "gradient_gain",
    "persistent_dark",
    "persistent_bright",
    "fused",
    "compact_change",
)


@dataclass(frozen=True)
class DensePoolConfigV2215:
    # Based on the successful V2.21.4 broad-pool operating point, but this
    # implementation is self-contained and candidate-aligned end-to-end.
    source_percentiles: dict[str, float] = field(default_factory=lambda: {
        "blackhat_gain": 82.0,
        "tophat_gain": 82.0,
        "persistent_abs": 72.0,
        "gradient_gain": 70.0,
        "persistent_dark": 68.0,
        "persistent_bright": 68.0,
        "fused": 64.0,
        "compact_change": 55.0,
    })
    target_margin_px: int = 46
    local_max_kernel: int = 3
    per_source_limit: int = 2600
    elongated_sample_step_px: int = 9
    component_max_area_for_centroid: int = 96
    cross_source_cluster_radius_px: float = 3.0
    final_nms_radius_px: float = 2.5
    pool_limit: int = 14000
    # Preserve moderate-score spatial evidence rather than retaining only the
    # strongest projector edges when a source exceeds its cap.
    score_keep_fraction: float = 0.60


@dataclass(frozen=True)
class ListwiseConfigV2215:
    positive_radius_px: float = 20.0
    target_sigma_px: float = 8.0
    candidates_per_shot: int = 2400
    hard_candidates_per_shot: int = 1500
    random_candidates_per_shot: int = 800
    stage1_epochs: int = 180
    stage2_epochs: int = 120
    learning_rate: float = 0.025
    l2: float = 0.0015
    temperature: float = 0.75
    seed: int = 2215
    cv_folds: int = 3
    top_k_values: tuple[int, ...] = (64, 128, 256, 512, 1024)
    frozen_top_k: int = 512


@dataclass
class DensePoolResultV2215:
    candidates: list[dict[str, Any]]
    target_mask: np.ndarray
    metadata: dict[str, Any]


@dataclass
class CandidateFeatureBatchV2215:
    matrix: np.ndarray
    feature_names: tuple[str, ...]


@dataclass
class ListwiseShotV2215:
    key: str
    matrix: np.ndarray
    distances_px: np.ndarray
    dense_scores: np.ndarray


@dataclass
class ListwiseModelV2215:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    metadata: dict[str, Any]

    def score_matrix(self, matrix: np.ndarray) -> np.ndarray:
        x = np.asarray(matrix, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature matrix mismatch: got {x.shape}, expected (*,{len(self.feature_names)})"
            )
        z = (x - self.mean[None, :]) / self.scale[None, :]
        return np.asarray(z @ self.weights, dtype=np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_names=np.asarray(self.feature_names, dtype=object),
            mean=np.asarray(self.mean, dtype=np.float32),
            scale=np.asarray(self.scale, dtype=np.float32),
            weights=np.asarray(self.weights, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(self.metadata, ensure_ascii=False, sort_keys=True), dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "ListwiseModelV2215":
        with np.load(Path(path), allow_pickle=True) as data:
            names = tuple(str(x) for x in data["feature_names"].tolist())
            metadata_raw = data["metadata_json"].item()
            metadata = json.loads(str(metadata_raw))
            return cls(
                feature_names=names,
                mean=np.asarray(data["mean"], dtype=np.float32),
                scale=np.asarray(data["scale"], dtype=np.float32),
                weights=np.asarray(data["weights"], dtype=np.float32),
                metadata=metadata,
            )


def dense_pool_config_dict_v2215(config: DensePoolConfigV2215 | None = None) -> dict[str, Any]:
    return asdict(config or DensePoolConfigV2215())


def listwise_config_dict_v2215(config: ListwiseConfigV2215 | None = None) -> dict[str, Any]:
    return asdict(config or ListwiseConfigV2215())


def _as_map(name: str, maps: Mapping[str, np.ndarray]) -> np.ndarray | None:
    value = maps.get(name)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    return arr if arr.ndim == 2 else None


def _finite_xy(row: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        x = float(row.get("camera_x", 0.0))
        y = float(row.get("camera_y", 0.0))
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _balanced_limit(rows: Sequence[dict[str, Any]], limit: int, score_key: str = "source_value", keep_fraction: float = 0.60) -> list[dict[str, Any]]:
    """Keep strong rows *and* spatially spread moderate rows deterministically."""
    limit = max(1, int(limit))
    if len(rows) <= limit:
        return [dict(r) for r in rows]
    ordered = sorted(rows, key=lambda r: float(r.get(score_key, 0.0)), reverse=True)
    strong_n = max(1, min(limit, int(round(limit * float(np.clip(keep_fraction, 0.10, 0.95))))))
    strong = ordered[:strong_n]
    remaining = ordered[strong_n:]
    need = limit - strong_n
    if need <= 0 or not remaining:
        return strong[:limit]
    # Spatially deterministic ordering prevents the cap from becoming another
    # global-score shortcut dominated by a few bright projector bands.
    spatial = sorted(
        remaining,
        key=lambda r: (
            int(float(r.get("camera_y", 0.0)) // 16),
            int(float(r.get("camera_x", 0.0)) // 16),
            -float(r.get(score_key, 0.0)),
        ),
    )
    if len(spatial) <= need:
        return strong + spatial
    idx = np.linspace(0, len(spatial) - 1, need, dtype=np.int64)
    return strong + [spatial[int(i)] for i in idx]


def _source_points(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    source_name: str,
    percentile: float,
    config: DensePoolConfigV2215,
) -> list[dict[str, Any]]:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(arr)
    sample = arr[valid]
    if sample.size == 0:
        return []
    threshold = float(np.percentile(sample, float(np.clip(percentile, 40.0, 99.9))))
    binary = valid & (arr >= threshold)
    if not np.any(binary):
        return []

    rows: list[dict[str, Any]] = []
    k = max(1, int(config.local_max_kernel)) | 1
    dilated = cv2.dilate(arr, np.ones((k, k), dtype=np.uint8))
    maxima = binary & (arr >= dilated - 1e-7)

    # Plateau/local-maximum components.  They are genuine temporal evidence,
    # not geometric padding.
    nmax, labels_max, stats_max, centroids_max = cv2.connectedComponentsWithStats(
        maxima.astype(np.uint8), connectivity=8
    )
    for label in range(1, nmax):
        area = int(stats_max[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = float(centroids_max[label][0])
        y = float(centroids_max[label][1])
        xi = int(np.clip(round(x), 0, arr.shape[1] - 1))
        yi = int(np.clip(round(y), 0, arr.shape[0] - 1))
        rows.append({
            "camera_x": x,
            "camera_y": y,
            "source_name": source_name,
            "source_value": float(arr[yi, xi]),
            "source_threshold": threshold,
            "source_kind": "local_max_component",
        })

    # Small threshold components get a weighted centroid even when their centre
    # is not a strict local maximum.  This helps compact weak holes.
    nbin, labels_bin, stats_bin, centroids_bin = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    small_limit = max(1, int(config.component_max_area_for_centroid))
    for label in range(1, nbin):
        area = int(stats_bin[label, cv2.CC_STAT_AREA])
        if area <= 0 or area > small_limit:
            continue
        x = float(centroids_bin[label][0])
        y = float(centroids_bin[label][1])
        xi = int(np.clip(round(x), 0, arr.shape[1] - 1))
        yi = int(np.clip(round(y), 0, arr.shape[0] - 1))
        rows.append({
            "camera_x": x,
            "camera_y": y,
            "source_name": source_name,
            "source_value": float(arr[yi, xi]),
            "source_threshold": threshold,
            "source_kind": "compact_component_centroid",
        })

    # Long projector/ridge components are deliberately sampled sparsely.  Every
    # sample still sits on source evidence; the grid is only a compression of a
    # large connected temporal region, not a GT-centred rescue grid.
    step = max(3, int(config.elongated_sample_step_px))
    large_area = stats_bin[:, cv2.CC_STAT_AREA]
    area_map = large_area[labels_bin]
    offset = sum(ord(ch) for ch in source_name) % step
    grid = np.zeros(arr.shape, dtype=bool)
    # A sparse 2-D lattice avoids allocating two full-size coordinate arrays
    # on 4K camera frames. Offset is deterministic per evidence source.
    y0 = (-2 * offset) % step
    x0 = (-offset) % step
    grid[y0::step, x0::step] = True
    ridge = binary & (area_map > small_limit) & grid
    ys, xs = np.nonzero(ridge)
    if len(xs):
        vals = arr[ys, xs]
        order = np.argsort(vals)[::-1]
        ridge_cap = max(64, int(config.per_source_limit))
        for idx in order[: ridge_cap * 2]:
            rows.append({
                "camera_x": float(xs[idx]),
                "camera_y": float(ys[idx]),
                "source_name": source_name,
                "source_value": float(vals[idx]),
                "source_threshold": threshold,
                "source_kind": "elongated_component_sample",
            })

    # Remove exact-ish duplicates inside the source before its balanced cap.
    dedup: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(round(float(row["camera_x"]) * 2.0)), int(round(float(row["camera_y"]) * 2.0)))
        old = dedup.get(key)
        if old is None or float(row["source_value"]) > float(old.get("source_value", -1e9)):
            dedup[key] = row
    return _balanced_limit(list(dedup.values()), int(config.per_source_limit), keep_fraction=float(config.score_keep_fraction))


def _cluster_points(rows: Sequence[dict[str, Any]], radius: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    radius = max(0.5, float(radius))
    r2 = radius * radius
    cell = radius
    ordered = sorted(rows, key=lambda r: float(r.get("source_value", 0.0)), reverse=True)
    clusters: list[dict[str, Any]] = []
    grid: dict[tuple[int, int], list[int]] = {}
    for row in ordered:
        x = float(row["camera_x"]); y = float(row["camera_y"])
        gx = int(math.floor(x / cell)); gy = int(math.floor(y / cell))
        found: int | None = None
        for ny in range(gy - 1, gy + 2):
            for nx in range(gx - 1, gx + 2):
                for idx in grid.get((nx, ny), []):
                    c = clusters[idx]
                    if (x - float(c["camera_x"])) ** 2 + (y - float(c["camera_y"])) ** 2 <= r2:
                        found = idx
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            idx = len(clusters)
            clusters.append({
                "camera_x": x,
                "camera_y": y,
                "weight_sum": max(1e-6, float(row.get("source_value", 0.0))),
                "x_weighted": x * max(1e-6, float(row.get("source_value", 0.0))),
                "y_weighted": y * max(1e-6, float(row.get("source_value", 0.0))),
                "source_names": {str(row.get("source_name", "unknown"))},
                "source_values": {str(row.get("source_name", "unknown")): float(row.get("source_value", 0.0))},
                "source_kinds": {str(row.get("source_kind", "unknown"))},
            })
            grid.setdefault((gx, gy), []).append(idx)
        else:
            c = clusters[found]
            weight = max(1e-6, float(row.get("source_value", 0.0)))
            c["weight_sum"] += weight
            c["x_weighted"] += x * weight
            c["y_weighted"] += y * weight
            c["camera_x"] = float(c["x_weighted"] / c["weight_sum"])
            c["camera_y"] = float(c["y_weighted"] / c["weight_sum"])
            source = str(row.get("source_name", "unknown"))
            c["source_names"].add(source)
            c["source_values"][source] = max(float(c["source_values"].get(source, 0.0)), float(row.get("source_value", 0.0)))
            c["source_kinds"].add(str(row.get("source_kind", "unknown")))

    out: list[dict[str, Any]] = []
    for c in clusters:
        names = sorted(c["source_names"])
        vals = list(c["source_values"].values())
        support = len(names)
        max_value = max(vals) if vals else 0.0
        mean_value = float(np.mean(vals)) if vals else 0.0
        dense_score = float(0.55 * min(1.0, support / 3.0) + 0.30 * max_value + 0.15 * mean_value)
        out.append({
            "camera_x": float(c["camera_x"]),
            "camera_y": float(c["camera_y"]),
            "score": float(100.0 * dense_score),
            "dense_score": dense_score,
            "dense_source_support": int(support),
            "dense_source_names": names,
            "dense_source_values": {k: float(v) for k, v in c["source_values"].items()},
            "dense_source_kinds": sorted(c["source_kinds"]),
            "evidence_source": "physical_dense_v2215",
            "evidence_sources": [f"physical_dense_v2215:{name}" for name in names],
            "ai_physical_dense_v2215": 1.0,
        })
    return out


def _nms(rows: Sequence[dict[str, Any]], radius: float, limit: int) -> list[dict[str, Any]]:
    radius = max(0.5, float(radius)); r2 = radius * radius
    cell = radius
    ordered = sorted(
        rows,
        key=lambda r: (
            int(r.get("dense_source_support", 0)),
            float(r.get("dense_score", 0.0)),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    grid: dict[tuple[int, int], list[int]] = {}
    for row in ordered:
        x = float(row["camera_x"]); y = float(row["camera_y"])
        gx = int(math.floor(x / cell)); gy = int(math.floor(y / cell))
        reject = False
        for ny in range(gy - 1, gy + 2):
            for nx in range(gx - 1, gx + 2):
                for idx in grid.get((nx, ny), []):
                    old = selected[idx]
                    if (x - float(old["camera_x"])) ** 2 + (y - float(old["camera_y"])) ** 2 <= r2:
                        reject = True
                        break
                if reject:
                    break
            if reject:
                break
        if reject:
            continue
        idx = len(selected)
        selected.append(dict(row))
        grid.setdefault((gx, gy), []).append(idx)
        if len(selected) >= max(1, int(limit)):
            break
    return selected


def propose_dense_pool_v2215(
    current_candidates: Sequence[dict[str, Any]],
    maps: Mapping[str, np.ndarray],
    *,
    config: DensePoolConfigV2215 | None = None,
) -> DensePoolResultV2215:
    """Create a broad GT-free temporal proposal pool.

    No ground-truth coordinate is accepted.  The only spatial prior is a target
    mask inferred from the current detector's candidate cloud.
    """
    cfg = config or DensePoolConfigV2215()
    first = next((_as_map(name, maps) for name in MAP_NAMES if _as_map(name, maps) is not None), None)
    if first is None:
        return DensePoolResultV2215([], np.zeros((1, 1), dtype=bool), {"reason": "no_maps"})
    mask = candidate_target_mask_v2213(current_candidates, first.shape[:2], margin_px=int(cfg.target_margin_px))
    raw: list[dict[str, Any]] = []
    per_source_counts: dict[str, int] = {}
    thresholds: dict[str, float] = {}
    for source_name in MAP_NAMES:
        arr = _as_map(source_name, maps)
        if arr is None:
            continue
        pct = float(cfg.source_percentiles.get(source_name, 75.0))
        sample = arr[np.asarray(mask, dtype=bool) & np.isfinite(arr)]
        thresholds[source_name] = float(np.percentile(sample, pct)) if sample.size else 0.0
        items = _source_points(arr, mask, source_name=source_name, percentile=pct, config=cfg)
        per_source_counts[source_name] = len(items)
        raw.extend(items)
    clustered = _cluster_points(raw, float(cfg.cross_source_cluster_radius_px))
    final = _nms(clustered, float(cfg.final_nms_radius_px), int(cfg.pool_limit))
    return DensePoolResultV2215(
        candidates=final,
        target_mask=mask,
        metadata={
            "raw_points": int(len(raw)),
            "clustered_points": int(len(clustered)),
            "final_points": int(len(final)),
            "per_source_counts": per_source_counts,
            "thresholds": thresholds,
            "mask_fraction": float(np.mean(mask)) if mask.size else 0.0,
            "config": dense_pool_config_dict_v2215(cfg),
        },
    )


def _sample_map(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    xi = np.clip(np.rint(xs).astype(np.int64), 0, w - 1)
    yi = np.clip(np.rint(ys).astype(np.int64), 0, h - 1)
    return np.asarray(arr[yi, xi], dtype=np.float32)


def _relative_percentile(values: np.ndarray, sample: np.ndarray) -> np.ndarray:
    finite = np.asarray(sample[np.isfinite(sample)], dtype=np.float32)
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    ordered = np.sort(finite)
    ranks = np.searchsorted(ordered, np.asarray(values, dtype=np.float32), side="right")
    return np.asarray(ranks / max(1, len(ordered)), dtype=np.float32)


def _nearest_distance(xs: np.ndarray, ys: np.ndarray, points: Sequence[dict[str, Any]], chunk: int = 1024) -> np.ndarray:
    coords = [p for p in (_finite_xy(row) for row in points) if p is not None]
    if not coords:
        return np.full(len(xs), 9999.0, dtype=np.float32)
    pts = np.asarray(coords, dtype=np.float32)
    out = np.empty(len(xs), dtype=np.float32)
    for start in range(0, len(xs), max(1, int(chunk))):
        stop = min(len(xs), start + max(1, int(chunk)))
        dx = xs[start:stop, None] - pts[None, :, 0]
        dy = ys[start:stop, None] - pts[None, :, 1]
        out[start:stop] = np.sqrt(np.min(dx * dx + dy * dy, axis=1))
    return out


def extract_candidate_features_v2215(
    candidates: Sequence[dict[str, Any]],
    maps: Mapping[str, np.ndarray],
    target_mask: np.ndarray,
    *,
    current_candidates: Sequence[dict[str, Any]] = (),
    local_candidates: Sequence[dict[str, Any]] = (),
) -> CandidateFeatureBatchV2215:
    """Extract train/inference-identical, shot-relative candidate features."""
    n = len(candidates)
    xs = np.asarray([float(c.get("camera_x", 0.0)) for c in candidates], dtype=np.float32)
    ys = np.asarray([float(c.get("camera_y", 0.0)) for c in candidates], dtype=np.float32)
    cols: list[np.ndarray] = []
    names: list[str] = []
    pct_by_name: dict[str, np.ndarray] = {}

    mask = np.asarray(target_mask, dtype=bool)
    for source_name in MAP_NAMES:
        arr = _as_map(source_name, maps)
        if arr is None:
            value = np.zeros(n, dtype=np.float32)
            local_contrast = np.zeros(n, dtype=np.float32)
            peak_ratio = np.zeros(n, dtype=np.float32)
            percentile = np.zeros(n, dtype=np.float32)
        else:
            value = _sample_map(arr, xs, ys)
            mean9 = cv2.blur(arr, (9, 9))
            max7 = cv2.dilate(arr, np.ones((7, 7), dtype=np.uint8))
            local_contrast = value - _sample_map(mean9, xs, ys)
            denom = np.maximum(_sample_map(max7, xs, ys), 1e-5)
            peak_ratio = np.clip(value / denom, 0.0, 2.0).astype(np.float32)
            sample = arr[mask & np.isfinite(arr)] if mask.shape == arr.shape else arr[np.isfinite(arr)]
            percentile = _relative_percentile(value, sample)
        pct_by_name[source_name] = percentile
        for suffix, vector in (
            ("value", value),
            ("contrast9", local_contrast),
            ("peak_ratio7", peak_ratio),
            ("shot_percentile", percentile),
        ):
            names.append(f"{source_name}:{suffix}")
            cols.append(np.asarray(vector, dtype=np.float32))

    support = np.asarray([min(8, int(c.get("dense_source_support", 0))) / 8.0 for c in candidates], dtype=np.float32)
    dense_score = np.asarray([float(c.get("dense_score", 0.0)) for c in candidates], dtype=np.float32)
    names += ["dense_source_support", "dense_score"]
    cols += [support, dense_score]

    # Exact source membership from the GT-free dense proposal construction.
    source_sets = [set(str(x) for x in (c.get("dense_source_names") or [])) for c in candidates]
    for source_name in MAP_NAMES:
        names.append(f"member:{source_name}")
        cols.append(np.asarray([1.0 if source_name in s else 0.0 for s in source_sets], dtype=np.float32))

    pct_stack = np.stack([pct_by_name[name] for name in MAP_NAMES], axis=1) if n else np.zeros((0, len(MAP_NAMES)), dtype=np.float32)
    if n:
        pct_sorted = np.sort(pct_stack, axis=1)
        max_pct = pct_sorted[:, -1]
        top3_pct = np.mean(pct_sorted[:, -3:], axis=1)
        mean_pct = np.mean(pct_stack, axis=1)
    else:
        max_pct = top3_pct = mean_pct = np.zeros(0, dtype=np.float32)
    names += ["map_percentile:max", "map_percentile:top3_mean", "map_percentile:mean"]
    cols += [max_pct, top3_pct, mean_pct]

    d_current = _nearest_distance(xs, ys, current_candidates)
    d_local = _nearest_distance(xs, ys, local_candidates)
    for prefix, dist in (("current", d_current), ("v2212_local", d_local)):
        names += [
            f"distance:{prefix}:clip100",
            f"distance:{prefix}:exp24",
            f"distance:{prefix}:within20",
            f"distance:{prefix}:within42",
        ]
        cols += [
            np.clip(dist, 0.0, 100.0) / 100.0,
            np.exp(-np.clip(dist, 0.0, 500.0) / 24.0).astype(np.float32),
            (dist <= 20.0).astype(np.float32),
            (dist <= 42.0).astype(np.float32),
        ]

    # Explicit interactions let the small linear listwise model express the
    # kinds of agreement that V2.21.3 tried to hand-code, without changing the
    # candidate set or inventing GT-centred examples.
    interactions = (
        ("blackhat_gain", "tophat_gain"),
        ("blackhat_gain", "persistent_abs"),
        ("persistent_dark", "persistent_abs"),
        ("persistent_bright", "tophat_gain"),
        ("compact_change", "persistent_abs"),
        ("gradient_gain", "blackhat_gain"),
    )
    for left, right in interactions:
        names.append(f"interaction_pct:{left}*{right}")
        cols.append((pct_by_name[left] * pct_by_name[right]).astype(np.float32))
    names.append("interaction_pct:max*support")
    cols.append((max_pct * support).astype(np.float32))

    matrix = np.column_stack(cols).astype(np.float32) if cols else np.zeros((n, 0), dtype=np.float32)
    return CandidateFeatureBatchV2215(matrix=matrix, feature_names=tuple(names))


def candidate_distances_v2215(candidates: Sequence[dict[str, Any]], gt: tuple[float, float]) -> np.ndarray:
    gx, gy = float(gt[0]), float(gt[1])
    return np.asarray([
        math.hypot(float(c.get("camera_x", 0.0)) - gx, float(c.get("camera_y", 0.0)) - gy)
        for c in candidates
    ], dtype=np.float32)


def shot_key_v2215(session_id: str, round_id: int) -> str:
    return f"{session_id}:{int(round_id)}"


def _stable_fold(key: str, folds: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % max(2, int(folds))


def _standardisation(shots: Sequence[ListwiseShotV2215]) -> tuple[np.ndarray, np.ndarray]:
    if not shots:
        raise ValueError("No shots for standardisation")
    chunks: list[np.ndarray] = []
    for shot in shots:
        x = np.asarray(shot.matrix, dtype=np.float32)
        if len(x) > 3000:
            idx = np.linspace(0, len(x) - 1, 3000, dtype=np.int64)
            x = x[idx]
        chunks.append(x)
    stack = np.concatenate(chunks, axis=0)
    mean = np.mean(stack, axis=0).astype(np.float32)
    scale = np.std(stack, axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-4).astype(np.float32)
    return mean, scale


def _training_subset(
    shot: ListwiseShotV2215,
    config: ListwiseConfigV2215,
    rng: np.random.Generator,
    model_scores: np.ndarray | None = None,
) -> np.ndarray:
    distances = np.asarray(shot.distances_px, dtype=np.float32)
    positives = np.flatnonzero(distances <= float(config.positive_radius_px))
    if len(positives) == 0:
        return np.asarray([], dtype=np.int64)
    all_idx = np.arange(len(distances), dtype=np.int64)
    negatives = all_idx[distances > float(config.positive_radius_px)]
    if len(negatives) == 0:
        return positives

    if model_scores is None:
        hardness = np.asarray(shot.dense_scores, dtype=np.float32)
    else:
        hardness = np.asarray(model_scores, dtype=np.float32)
    hard_n = min(len(negatives), max(0, int(config.hard_candidates_per_shot)))
    if hard_n:
        neg_scores = hardness[negatives]
        if len(negatives) > hard_n:
            pick = np.argpartition(neg_scores, -hard_n)[-hard_n:]
            hard = negatives[pick]
        else:
            hard = negatives
    else:
        hard = np.asarray([], dtype=np.int64)

    remaining = np.setdiff1d(negatives, hard, assume_unique=False)
    random_n = min(len(remaining), max(0, int(config.random_candidates_per_shot)))
    random_pick = rng.choice(remaining, size=random_n, replace=False) if random_n else np.asarray([], dtype=np.int64)
    subset = np.unique(np.concatenate([positives, hard, random_pick])).astype(np.int64)
    cap = max(len(positives), int(config.candidates_per_shot))
    if len(subset) > cap:
        # Never drop a real candidate-aligned positive.
        negative_subset = subset[distances[subset] > float(config.positive_radius_px)]
        keep_neg = max(0, cap - len(positives))
        if len(negative_subset) > keep_neg:
            order = np.argsort(hardness[negative_subset])[::-1][:keep_neg]
            negative_subset = negative_subset[order]
        subset = np.unique(np.concatenate([positives, negative_subset]))
    return subset.astype(np.int64)


def _soft_targets(distances: np.ndarray, radius: float, sigma: float) -> np.ndarray:
    d = np.asarray(distances, dtype=np.float32)
    positive = d <= float(radius)
    q = np.zeros(len(d), dtype=np.float32)
    if not np.any(positive):
        return q
    sigma = max(1.0, float(sigma))
    q[positive] = np.exp(-0.5 * (d[positive] / sigma) ** 2).astype(np.float32) + 0.05
    total = float(np.sum(q))
    if total > 0:
        q /= total
    return q


def _softmax(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    s = s - np.max(s)
    e = np.exp(np.clip(s, -40.0, 40.0))
    return np.asarray(e / max(1e-12, float(np.sum(e))), dtype=np.float32)


def _fit_weights(
    shots: Sequence[ListwiseShotV2215],
    mean: np.ndarray,
    scale: np.ndarray,
    config: ListwiseConfigV2215,
    *,
    initial: np.ndarray | None = None,
    stage: int = 1,
) -> tuple[np.ndarray, list[float]]:
    if not shots:
        raise ValueError("No shots to fit")
    dim = shots[0].matrix.shape[1]
    weights = np.zeros(dim, dtype=np.float32) if initial is None else np.asarray(initial, dtype=np.float32).copy()
    m = np.zeros_like(weights); v = np.zeros_like(weights)
    beta1, beta2 = 0.9, 0.999
    rng = np.random.default_rng(int(config.seed) + 1000 * int(stage))
    epochs = int(config.stage1_epochs if stage == 1 else config.stage2_epochs)
    history: list[float] = []
    t = 0
    for epoch in range(max(1, epochs)):
        order = rng.permutation(len(shots))
        losses: list[float] = []
        for shot_index in order:
            shot = shots[int(shot_index)]
            x_all = (np.asarray(shot.matrix, dtype=np.float32) - mean[None, :]) / scale[None, :]
            model_scores_all = x_all @ weights if stage >= 2 else None
            subset = _training_subset(shot, config, rng, model_scores=model_scores_all)
            if len(subset) == 0:
                continue
            x = x_all[subset]
            d = np.asarray(shot.distances_px, dtype=np.float32)[subset]
            q = _soft_targets(d, float(config.positive_radius_px), float(config.target_sigma_px))
            if float(np.sum(q)) <= 0:
                continue
            temperature = max(0.15, float(config.temperature))
            scores = (x @ weights) / temperature
            p = _softmax(scores)
            loss = -float(np.sum(q * np.log(np.maximum(p, 1e-8)))) + 0.5 * float(config.l2) * float(np.sum(weights * weights))
            grad = (x.T @ (p - q)) / temperature + float(config.l2) * weights
            grad = np.asarray(grad, dtype=np.float32)
            # Adam keeps the tiny implementation numerically stable even when
            # percentile and distance features have very different curvature.
            t += 1
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            mhat = m / (1.0 - beta1 ** t)
            vhat = v / (1.0 - beta2 ** t)
            weights -= float(config.learning_rate) * mhat / (np.sqrt(vhat) + 1e-8)
            losses.append(loss)
        history.append(float(np.mean(losses)) if losses else 0.0)
    return weights.astype(np.float32), history


def _oracle_at_k(shots: Sequence[ListwiseShotV2215], model: ListwiseModelV2215, k: int, radius: float = 20.0) -> float:
    if not shots:
        return 0.0
    ok = 0
    for shot in shots:
        scores = model.score_matrix(shot.matrix)
        order = np.argsort(scores)[::-1][: max(1, int(k))]
        if np.any(np.asarray(shot.distances_px)[order] <= float(radius)):
            ok += 1
    return float(ok / len(shots))


def _rank20(shot: ListwiseShotV2215, model: ListwiseModelV2215, radius: float = 20.0) -> int:
    positives = np.flatnonzero(np.asarray(shot.distances_px) <= float(radius))
    if len(positives) == 0:
        return 9999
    scores = model.score_matrix(shot.matrix)
    order = np.argsort(scores)[::-1]
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(1, len(order) + 1)
    return int(np.min(inverse[positives]))


def fit_listwise_ranker_v2215(
    shots: Sequence[ListwiseShotV2215],
    feature_names: Sequence[str],
    *,
    config: ListwiseConfigV2215 | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[ListwiseModelV2215, dict[str, Any]]:
    """Fit a candidate-aligned, whole-shot listwise linear ranker.

    ``shots`` must already contain *real dense-pool candidates*.  This function
    never inserts or jitters GT coordinates.
    """
    cfg = config or ListwiseConfigV2215()
    usable = [s for s in shots if np.any(np.asarray(s.distances_px) <= float(cfg.positive_radius_px))]
    if not usable:
        raise RuntimeError("No development shots contain a real dense-pool candidate within the positive radius")
    mean, scale = _standardisation(usable)
    w1, loss1 = _fit_weights(usable, mean, scale, cfg, stage=1)
    w2, loss2 = _fit_weights(usable, mean, scale, cfg, initial=w1, stage=2)
    model_meta = dict(metadata or {})
    model_meta.update({
        "schema_version": "2.21.5",
        "training_scope": "physical_development_candidate_aligned_listwise",
        "candidate_aligned_only": True,
        "forced_positive_jitter_count": 0,
        "development_shots_seen": int(len(shots)),
        "development_shots_with_positive20": int(len(usable)),
        "training_config": listwise_config_dict_v2215(cfg),
        "stage1_loss_head_tail": [float(loss1[0]), float(loss1[-1])] if loss1 else [],
        "stage2_loss_head_tail": [float(loss2[0]), float(loss2[-1])] if loss2 else [],
    })
    model = ListwiseModelV2215(
        feature_names=tuple(feature_names),
        mean=mean,
        scale=scale,
        weights=w2,
        metadata=model_meta,
    )
    report = {
        "usable_shots": len(usable),
        "skipped_no_positive20": len(shots) - len(usable),
        "stage1_loss_head_tail": model_meta["stage1_loss_head_tail"],
        "stage2_loss_head_tail": model_meta["stage2_loss_head_tail"],
        "train_oracle20": {
            str(k): _oracle_at_k(usable, model, int(k), 20.0) for k in cfg.top_k_values
        },
        "train_rank20_median": float(np.median([_rank20(s, model) for s in usable])) if usable else 9999.0,
    }
    return model, report


def cross_validate_listwise_v2215(
    shots: Sequence[ListwiseShotV2215],
    feature_names: Sequence[str],
    *,
    config: ListwiseConfigV2215 | None = None,
) -> dict[str, Any]:
    """Shot-level DEVELOPMENT-only cross-fit; protected data is never passed in."""
    cfg = config or ListwiseConfigV2215()
    folds = max(2, min(int(cfg.cv_folds), len(shots)))
    rows: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [s for s in shots if _stable_fold(s.key, folds) != fold]
        valid = [s for s in shots if _stable_fold(s.key, folds) == fold]
        if not train or not valid:
            continue
        # CV gets fewer epochs to keep this diagnostic cheap while preserving
        # the exact candidate-aligned objective.
        cv_cfg = ListwiseConfigV2215(**{
            **asdict(cfg),
            "stage1_epochs": max(50, int(cfg.stage1_epochs) // 2),
            "stage2_epochs": max(35, int(cfg.stage2_epochs) // 2),
        })
        model, _ = fit_listwise_ranker_v2215(train, feature_names, config=cv_cfg, metadata={"cv_fold": fold})
        for shot in valid:
            scores = model.score_matrix(shot.matrix)
            order = np.argsort(scores)[::-1]
            d = np.asarray(shot.distances_px)
            row = {"fold": fold, "key": shot.key, "has_pool20": bool(np.any(d <= 20.0))}
            for k in cfg.top_k_values:
                row[f"top{k}_oracle20"] = bool(np.any(d[order[: int(k)]] <= 20.0))
            row["rank20"] = _rank20(shot, model)
            rows.append(row)
    summary: dict[str, Any] = {
        "folds": folds,
        "rows": rows,
        "shots": len(rows),
    }
    for k in cfg.top_k_values:
        summary[f"top{k}_oracle20"] = float(np.mean([float(r[f"top{k}_oracle20"]) for r in rows])) if rows else 0.0
    ranks = [int(r["rank20"]) for r in rows if int(r["rank20"]) < 9999]
    summary["rank20_median_when_present"] = float(np.median(ranks)) if ranks else 9999.0
    summary["pool20_oracle"] = float(np.mean([float(r["has_pool20"]) for r in rows])) if rows else 0.0
    return summary


def rank_candidates_v2215(
    candidates: Sequence[dict[str, Any]],
    features: CandidateFeatureBatchV2215,
    model: ListwiseModelV2215,
) -> list[dict[str, Any]]:
    if tuple(features.feature_names) != tuple(model.feature_names):
        raise RuntimeError("V2.21.5 feature schema mismatch between model and inference")
    scores = model.score_matrix(features.matrix)
    order = np.argsort(scores)[::-1]
    out: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, 1):
        row = dict(candidates[int(idx)])
        row["v2215_score"] = float(scores[int(idx)])
        row["v2215_rank"] = int(rank)
        row["evidence_source"] = "physical_dense_ranked_v2215"
        out.append(row)
    return out
