from __future__ import annotations

"""V2.21.4 learned physical-domain dense temporal proposals.

This module deliberately separates two questions:

1. Candidate generation: can broad PRE->POST temporal evidence put *some* point
   near the new hole without receiving GT?
2. Candidate ranking: can a lightweight model trained only on physical
   DEVELOPMENT full-frame shots rank those broad proposals high enough to be
   useful?

Ground truth is used only by the training-data builder.  The proposal and
inference functions do not accept GT.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .temporal_consensus_v2213 import candidate_target_mask_v2213


SCHEMA_VERSION = "2.21.4"
DEFAULT_CONFIG_PATH = Path("content/ai/physical_dense_v2214.json")
DEFAULT_MODEL_PATH = Path("content/ai/reports/v2214/physical_dense_ranker_v2214.npz")


SOURCE_NAMES = (
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
class DensePoolConfigV2214:
    # The V2.21.2 DEVELOPMENT diagnostics showed that GT is often around the
    # 60-90th map percentile, not the 99th.  These intentionally broad
    # thresholds create a high-recall pool which the learned ranker must prune.
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


@dataclass(frozen=True)
class DenseTrainingConfigV2214:
    positive_radius_px: float = 10.0
    forced_positive_jitter_px: tuple[int, ...] = (-4, -2, 0, 2, 4)
    negative_exclusion_radius_px: float = 42.0
    hard_negatives_per_shot: int = 800
    random_negatives_per_shot: int = 240
    pairs_per_positive: int = 96
    stage1_epochs: int = 180
    stage2_epochs: int = 120
    stage2_mined_negatives_per_shot: int = 420
    learning_rate: float = 0.035
    l2: float = 0.0025
    seed: int = 2214
    top_k_values: tuple[int, ...] = (64, 128, 256, 512)
    frozen_top_k: int = 512


@dataclass
class DenseRankerV2214:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    metadata: dict[str, Any]

    def score_features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        z = (x - self.mean[None, :]) / self.scale[None, :]
        return (z @ self.weights).astype(np.float32)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray([SCHEMA_VERSION]),
            feature_names=np.asarray(self.feature_names),
            mean=np.asarray(self.mean, dtype=np.float32),
            scale=np.asarray(self.scale, dtype=np.float32),
            weights=np.asarray(self.weights, dtype=np.float32),
            metadata_json=np.asarray([json.dumps(self.metadata, ensure_ascii=False, sort_keys=True)]),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "DenseRankerV2214":
        with np.load(Path(path), allow_pickle=False) as data:
            names = tuple(str(x) for x in np.asarray(data["feature_names"]).tolist())
            metadata_raw = str(np.asarray(data["metadata_json"]).reshape(-1)[0])
            return cls(
                feature_names=names,
                mean=np.asarray(data["mean"], dtype=np.float32),
                scale=np.asarray(data["scale"], dtype=np.float32),
                weights=np.asarray(data["weights"], dtype=np.float32),
                metadata=json.loads(metadata_raw),
            )


def load_dense_configs_v2214(path: Path | None = None) -> tuple[DensePoolConfigV2214, DenseTrainingConfigV2214]:
    path = Path(path or DEFAULT_CONFIG_PATH)
    if not path.exists():
        return DensePoolConfigV2214(), DenseTrainingConfigV2214()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return DensePoolConfigV2214(), DenseTrainingConfigV2214()
    if not isinstance(raw, dict):
        return DensePoolConfigV2214(), DenseTrainingConfigV2214()

    pool_raw = raw.get("pool", {}) if isinstance(raw.get("pool"), dict) else {}
    train_raw = raw.get("training", {}) if isinstance(raw.get("training"), dict) else {}

    pool_allowed = {k for k in DensePoolConfigV2214.__dataclass_fields__}
    train_allowed = {k for k in DenseTrainingConfigV2214.__dataclass_fields__}
    pool_kwargs = {k: v for k, v in pool_raw.items() if k in pool_allowed}
    train_kwargs = {k: v for k, v in train_raw.items() if k in train_allowed}
    if isinstance(pool_kwargs.get("source_percentiles"), dict):
        pool_kwargs["source_percentiles"] = {str(k): float(v) for k, v in pool_kwargs["source_percentiles"].items()}
    for key in ("forced_positive_jitter_px", "top_k_values"):
        if key in train_kwargs:
            train_kwargs[key] = tuple(int(v) for v in train_kwargs[key])
    return DensePoolConfigV2214(**pool_kwargs), DenseTrainingConfigV2214(**train_kwargs)


def _as_map(name: str, maps: Mapping[str, np.ndarray], fused: np.ndarray | None) -> np.ndarray | None:
    if name == "fused":
        value = fused
    else:
        value = maps.get(name)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    return arr if arr.ndim == 2 else None


def _source_pool_points(
    values: np.ndarray,
    target_mask: np.ndarray,
    *,
    percentile: float,
    local_max_kernel: int,
    per_source_limit: int,
    elongated_sample_step_px: int,
    component_max_area_for_centroid: int,
) -> list[dict[str, Any]]:
    """Broad spatially distributed local maxima for one evidence source.

    We intentionally keep at most one maximum per small grid cell.  This avoids
    the V2.21.3 failure mode where a saturated line collapsed to one centroid,
    while also avoiding an expensive connected-component pass over every 4K
    source map.  The learned ranker later decides which of these broad points
    matter.
    """
    arr = np.asarray(values, dtype=np.float32)
    valid = np.asarray(target_mask, dtype=bool) & np.isfinite(arr)
    sample = arr[valid]
    if not sample.size:
        return []
    threshold = float(np.percentile(sample, float(np.clip(percentile, 20.0, 99.9))))
    k = max(1, int(local_max_kernel)) | 1
    dilated = cv2.dilate(arr, np.ones((k, k), dtype=np.uint8))
    maxima = valid & (arr >= threshold) & (arr >= dilated - 1e-7)
    yy, xx = np.nonzero(maxima)
    if not len(xx):
        return []
    vals = arr[yy, xx]

    # One representative per spatial cell preserves coverage along long
    # projector/noise plateaus instead of keeping thousands of adjacent pixels.
    step = max(3, int(elongated_sample_step_px))
    ncols = int(math.ceil(arr.shape[1] / float(step)))
    cell_id = (yy // step).astype(np.int64) * ncols + (xx // step).astype(np.int64)
    order = np.lexsort((-vals, cell_id))
    ordered_cells = cell_id[order]
    first = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        first[1:] = ordered_cells[1:] != ordered_cells[:-1]
    keep = order[first]
    keep_vals = vals[keep]
    if len(keep) > max(1, int(per_source_limit)):
        top = np.argpartition(keep_vals, -int(per_source_limit))[-int(per_source_limit):]
        keep = keep[top]
        keep_vals = vals[keep]
    rank = np.argsort(keep_vals)[::-1]
    keep = keep[rank]
    return [
        {"camera_x": float(xx[i]), "camera_y": float(yy[i]), "source_peak": float(vals[i])}
        for i in keep
    ]

def _cluster_rows(rows: Sequence[dict[str, Any]], radius: float, limit: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: float(r.get("source_peak", 0.0)), reverse=True)
    r = max(0.5, float(radius)); r2 = r * r
    cell = max(1.0, r)
    buckets: dict[tuple[int, int], list[int]] = {}
    clusters: list[dict[str, Any]] = []
    for row in ordered:
        x = float(row["camera_x"]); y = float(row["camera_y"])
        gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
        found = None
        for by in range(gy - 1, gy + 2):
            for bx in range(gx - 1, gx + 2):
                for i in buckets.get((bx, by), []):
                    old = clusters[i]
                    if (x - float(old["camera_x"])) ** 2 + (y - float(old["camera_y"])) ** 2 <= r2:
                        found = i; break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            idx = len(clusters)
            clusters.append({
                "camera_x": x, "camera_y": y,
                "best_peak": float(row.get("source_peak", 0.0)),
                "sources": {str(row.get("source_name", "unknown"))},
                "members": 1,
            })
            buckets.setdefault((gx, gy), []).append(idx)
        else:
            old = clusters[found]
            old["sources"].add(str(row.get("source_name", "unknown")))
            old["members"] = int(old["members"]) + 1
            # Ordered strongest-first, so keep the original coordinate.

    out = [{
        "camera_x": float(row["camera_x"]),
        "camera_y": float(row["camera_y"]),
        "pool_support": int(len(row["sources"])),
        "pool_members": int(row["members"]),
        "pool_best_peak": float(row["best_peak"]),
        "pool_sources": sorted(row["sources"]),
        "evidence_source": "physical_dense_pool_v2214",
        "evidence_sources": [f"physical_dense_pool_v2214:{name}" for name in sorted(row["sources"])],
    } for row in clusters]
    out.sort(key=lambda row: (int(row["pool_support"]), float(row["pool_best_peak"])), reverse=True)
    return out[: max(1, int(limit))]

def build_dense_pool_v2214(
    current_candidates: Sequence[dict[str, Any]],
    maps: Mapping[str, np.ndarray],
    fused: np.ndarray,
    *,
    config: DensePoolConfigV2214 | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Create a broad full-target temporal proposal pool.  GT-free."""
    cfg = config or DensePoolConfigV2214()
    first = next((np.asarray(v) for v in maps.values() if isinstance(v, np.ndarray) and np.asarray(v).ndim == 2), None)
    if first is None:
        return [], np.zeros((1, 1), dtype=bool)
    target_mask = candidate_target_mask_v2213(
        current_candidates,
        first.shape[:2],
        margin_px=int(cfg.target_margin_px),
    )
    raw: list[dict[str, Any]] = []
    for source_name in SOURCE_NAMES:
        arr = _as_map(source_name, maps, fused)
        if arr is None:
            continue
        percentile = float(cfg.source_percentiles.get(source_name, 75.0))
        points = _source_pool_points(
            arr,
            target_mask,
            percentile=percentile,
            local_max_kernel=int(cfg.local_max_kernel),
            per_source_limit=int(cfg.per_source_limit),
            elongated_sample_step_px=int(cfg.elongated_sample_step_px),
            component_max_area_for_centroid=int(cfg.component_max_area_for_centroid),
        )
        for point in points:
            raw.append({**point, "source_name": source_name})
    return _cluster_rows(raw, cfg.cross_source_cluster_radius_px, cfg.pool_limit), target_mask


FEATURE_NAMES = (
    "blackhat_gain",
    "tophat_gain",
    "persistent_abs",
    "gradient_gain",
    "persistent_dark",
    "persistent_bright",
    "fused",
    "compact_change",
    "blackhat_peakness",
    "tophat_peakness",
    "abs_peakness",
    "gradient_peakness",
    "dark_peakness",
    "bright_peakness",
    "fused_peakness",
    "blackhat_x_abs",
    "tophat_x_abs",
    "gradient_x_abs",
    "dark_x_abs",
    "bright_x_tophat",
    "compact_x_abs",
)


def features_from_maps_at_points_v2214(
    maps: Mapping[str, np.ndarray],
    fused: np.ndarray,
    points: Sequence[dict[str, Any]] | np.ndarray,
) -> np.ndarray:
    """Extract V2.21.4 features without materialising 21 full 4K maps.

    Direct V2.21 already holds several float32 evidence maps.  Creating another
    full-frame array per learned feature would push memory close to a gigabyte
    on 3840x2160 input.  This function samples base channels first, then creates
    only one temporary blurred map at a time for peakness features.
    """
    if isinstance(points, np.ndarray):
        coords = np.asarray(points, dtype=np.float32)
    else:
        coords = np.asarray([[float(p["camera_x"]), float(p["camera_y"])] for p in points], dtype=np.float32)
    if not len(coords):
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    shape = np.asarray(fused).shape[:2]
    h, w = int(shape[0]), int(shape[1])
    xx = np.clip(np.rint(coords[:, 0]).astype(np.int64), 0, w - 1)
    yy = np.clip(np.rint(coords[:, 1]).astype(np.int64), 0, h - 1)

    base_values: dict[str, np.ndarray] = {}
    for name in SOURCE_NAMES:
        arr = _as_map(name, maps, fused)
        if arr is None:
            base_values[name] = np.zeros((len(coords),), dtype=np.float32)
        else:
            base_values[name] = np.asarray(arr, dtype=np.float32)[yy, xx].astype(np.float32)

    peak_values: dict[str, np.ndarray] = {}
    peak_sources = {
        "blackhat_gain": "blackhat_peakness",
        "tophat_gain": "tophat_peakness",
        "persistent_abs": "abs_peakness",
        "gradient_gain": "gradient_peakness",
        "persistent_dark": "dark_peakness",
        "persistent_bright": "bright_peakness",
        "fused": "fused_peakness",
    }
    for src, dst in peak_sources.items():
        arr = _as_map(src, maps, fused)
        if arr is None:
            peak_values[dst] = np.zeros((len(coords),), dtype=np.float32)
            continue
        arr32 = np.asarray(arr, dtype=np.float32)
        smooth = cv2.GaussianBlur(arr32, (0, 0), 3.2)
        peak_values[dst] = np.clip(arr32[yy, xx] - smooth[yy, xx], -1.0, 1.0).astype(np.float32)
        del smooth

    cols = [
        base_values["blackhat_gain"],
        base_values["tophat_gain"],
        base_values["persistent_abs"],
        base_values["gradient_gain"],
        base_values["persistent_dark"],
        base_values["persistent_bright"],
        base_values["fused"],
        base_values["compact_change"],
        peak_values["blackhat_peakness"],
        peak_values["tophat_peakness"],
        peak_values["abs_peakness"],
        peak_values["gradient_peakness"],
        peak_values["dark_peakness"],
        peak_values["bright_peakness"],
        peak_values["fused_peakness"],
        base_values["blackhat_gain"] * base_values["persistent_abs"],
        base_values["tophat_gain"] * base_values["persistent_abs"],
        base_values["gradient_gain"] * base_values["persistent_abs"],
        base_values["persistent_dark"] * base_values["persistent_abs"],
        base_values["persistent_bright"] * base_values["tophat_gain"],
        base_values["compact_change"] * base_values["persistent_abs"],
    ]
    return np.stack(cols, axis=1).astype(np.float32)


# Backwards-friendly small-image helper retained for diagnostics/selftests.
def build_feature_maps_v2214(maps: Mapping[str, np.ndarray], fused: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in SOURCE_NAMES:
        arr = _as_map(name, maps, fused)
        out[name] = np.zeros_like(fused, dtype=np.float32) if arr is None else np.asarray(arr, dtype=np.float32)
    return out


def features_at_points_v2214(
    feature_maps: Mapping[str, np.ndarray],
    points: Sequence[dict[str, Any]] | np.ndarray,
) -> np.ndarray:
    """Legacy helper for already-materialised FEATURE_NAMES maps.

    V2.21.4 runtime/training paths use features_from_maps_at_points_v2214 to
    remain storage-friendly at 4K.
    """
    if any(name not in feature_maps for name in FEATURE_NAMES):
        raise ValueError("features_at_points_v2214 requires all FEATURE_NAMES maps; use features_from_maps_at_points_v2214 for normal V2.21.4 use")
    if isinstance(points, np.ndarray):
        coords = np.asarray(points, dtype=np.float32)
    else:
        coords = np.asarray([[float(p["camera_x"]), float(p["camera_y"])] for p in points], dtype=np.float32)
    if not len(coords):
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    first = np.asarray(feature_maps[FEATURE_NAMES[0]])
    h, w = first.shape[:2]
    xx = np.clip(np.rint(coords[:, 0]).astype(np.int64), 0, w - 1)
    yy = np.clip(np.rint(coords[:, 1]).astype(np.int64), 0, h - 1)
    return np.stack([np.asarray(feature_maps[name], dtype=np.float32)[yy, xx] for name in FEATURE_NAMES], axis=1).astype(np.float32)

def _nms_scored(rows: Sequence[dict[str, Any]], radius: float, limit: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: float(r.get("learned_dense_score", -1e9)), reverse=True)
    selected: list[dict[str, Any]] = []
    r = max(0.5, float(radius)); r2 = r * r; cell = max(1.0, r)
    buckets: dict[tuple[int, int], list[int]] = {}
    for row in ordered:
        x = float(row["camera_x"]); y = float(row["camera_y"])
        gx, gy = int(math.floor(x / cell)), int(math.floor(y / cell))
        reject = False
        for by in range(gy - 1, gy + 2):
            for bx in range(gx - 1, gx + 2):
                for idx in buckets.get((bx, by), []):
                    old = selected[idx]
                    if (x - float(old["camera_x"])) ** 2 + (y - float(old["camera_y"])) ** 2 <= r2:
                        reject = True; break
                if reject:
                    break
            if reject:
                break
        if reject:
            continue
        idx = len(selected); selected.append(dict(row)); buckets.setdefault((gx, gy), []).append(idx)
        if len(selected) >= max(1, int(limit)):
            break
    return selected

def rank_dense_pool_v2214(
    pool: Sequence[dict[str, Any]],
    maps: Mapping[str, np.ndarray],
    fused: np.ndarray,
    model: DenseRankerV2214,
    *,
    limit: int = 512,
    nms_radius_px: float = 4.0,
) -> list[dict[str, Any]]:
    """Rank an already-generated pool.  No GT and no training occurs here."""
    if tuple(model.feature_names) != tuple(FEATURE_NAMES):
        raise ValueError("V2.21.4 model feature schema mismatch")
    if not pool:
        return []
    features = features_from_maps_at_points_v2214(maps, fused, pool)
    scores = model.score_features(features)
    rows: list[dict[str, Any]] = []
    for raw, score in zip(pool, scores):
        row = dict(raw)
        row["learned_dense_score"] = float(score)
        row["score"] = float(score)
        row["evidence_source"] = "physical_dense_ranker_v2214"
        sources = list(row.get("evidence_sources") or [])
        sources.append("physical_dense_ranker_v2214")
        row["evidence_sources"] = sources
        row["ai_physical_dense_v2214"] = 1.0
        rows.append(row)
    return _nms_scored(rows, nms_radius_px, limit)


def _distance_xy(x: float, y: float, gt: tuple[float, float]) -> float:
    return float(math.hypot(float(x) - float(gt[0]), float(y) - float(gt[1])))


def _jitter_positive_points(gt: tuple[float, float], jitters: Sequence[int], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    pts = []
    # Compact deterministic cross/diagonal cloud; all points remain very close
    # to the known new-hole centre and are training-only.
    for dx in jitters:
        for dy in jitters:
            if math.hypot(dx, dy) > 5.7:
                continue
            x = min(max(float(gt[0]) + float(dx), 0.0), float(w - 1))
            y = min(max(float(gt[1]) + float(dy), 0.0), float(h - 1))
            pts.append((x, y))
    return np.asarray(pts, dtype=np.float32)


@dataclass
class DenseShotTrainingDataV2214:
    shot_key: str
    pool_rows: list[dict[str, Any]]
    pool_features: np.ndarray
    pool_distances: np.ndarray
    positive_features: np.ndarray
    mask_fraction: float


def make_shot_training_data_v2214(
    shot_key: str,
    current_candidates: Sequence[dict[str, Any]],
    maps: Mapping[str, np.ndarray],
    fused: np.ndarray,
    gt: tuple[float, float],
    *,
    pool_config: DensePoolConfigV2214 | None = None,
    training_config: DenseTrainingConfigV2214 | None = None,
) -> DenseShotTrainingDataV2214:
    """Build DEVELOPMENT training samples.  This is the only GT-aware builder."""
    pcfg = pool_config or DensePoolConfigV2214()
    tcfg = training_config or DenseTrainingConfigV2214()
    pool, mask = build_dense_pool_v2214(current_candidates, maps, fused, config=pcfg)
    distances = np.asarray([
        _distance_xy(float(row["camera_x"]), float(row["camera_y"]), gt) for row in pool
    ], dtype=np.float32)
    shape = np.asarray(fused).shape[:2]
    positive_points = _jitter_positive_points(gt, tcfg.forced_positive_jitter_px, shape)
    # Also use any naturally occurring broad-pool points already close to GT.
    if len(pool):
        near = np.asarray([
            [float(pool[i]["camera_x"]), float(pool[i]["camera_y"])]
            for i in np.flatnonzero(distances <= float(tcfg.positive_radius_px))
        ], dtype=np.float32)
        if near.size:
            positive_points = np.concatenate([positive_points, near], axis=0)

    # Extract all point features in one pass so the seven temporary 4K Gaussian
    # blurs are built once per shot, not once for pool and once for positives.
    pool_coords = np.asarray([[float(r["camera_x"]), float(r["camera_y"])] for r in pool], dtype=np.float32)
    if len(pool_coords):
        combined = np.concatenate([pool_coords, positive_points], axis=0)
        combined_features = features_from_maps_at_points_v2214(maps, fused, combined)
        pool_features = combined_features[: len(pool_coords)]
        positive_features = combined_features[len(pool_coords):]
    else:
        pool_features = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        positive_features = features_from_maps_at_points_v2214(maps, fused, positive_points)
    return DenseShotTrainingDataV2214(
        shot_key=str(shot_key),
        pool_rows=list(pool),
        pool_features=pool_features,
        pool_distances=distances,
        positive_features=positive_features,
        mask_fraction=float(np.mean(mask)),
    )


def _standardise_training(shots: Sequence[DenseShotTrainingDataV2214], cfg: DenseTrainingConfigV2214) -> tuple[np.ndarray, np.ndarray]:
    chunks = []
    for shot in shots:
        chunks.append(shot.positive_features)
        neg = shot.pool_features[shot.pool_distances >= float(cfg.negative_exclusion_radius_px)]
        if len(neg):
            chunks.append(neg[: min(len(neg), 1200)])
    x = np.concatenate(chunks, axis=0).astype(np.float32)
    mean = np.mean(x, axis=0).astype(np.float32)
    scale = np.std(x, axis=0).astype(np.float32)
    scale = np.where(scale < 1e-4, 1.0, scale).astype(np.float32)
    return mean, scale


def _salience(features: np.ndarray) -> np.ndarray:
    # Feature indexes follow FEATURE_NAMES.  This is only for choosing hard
    # negatives before the learned model exists; it is not the final score.
    if not len(features):
        return np.empty((0,), dtype=np.float32)
    return np.maximum.reduce([
        features[:, 0], features[:, 1], features[:, 2], features[:, 3],
        0.8 * features[:, 4], 0.8 * features[:, 5], 0.8 * features[:, 6],
    ]).astype(np.float32)


def _make_pairs(
    shots: Sequence[DenseShotTrainingDataV2214],
    mean: np.ndarray,
    scale: np.ndarray,
    cfg: DenseTrainingConfigV2214,
    *,
    model_weights: np.ndarray | None = None,
    stage2: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(int(cfg.seed) + (1000 if stage2 else 0))
    pairs: list[np.ndarray] = []
    for shot in shots:
        pos = (shot.positive_features - mean[None, :]) / scale[None, :]
        neg_mask = shot.pool_distances >= float(cfg.negative_exclusion_radius_px)
        neg_raw = shot.pool_features[neg_mask]
        if not len(pos) or not len(neg_raw):
            continue
        neg = (neg_raw - mean[None, :]) / scale[None, :]
        if stage2 and model_weights is not None:
            score = neg @ model_weights
            order = np.argsort(score)[::-1]
            hard_n = min(len(order), int(cfg.stage2_mined_negatives_per_shot))
            chosen = order[:hard_n]
        else:
            sal = _salience(neg_raw)
            order = np.argsort(sal)[::-1]
            hard_n = min(len(order), int(cfg.hard_negatives_per_shot))
            hard = order[:hard_n]
            remaining = order[hard_n:]
            random_n = min(len(remaining), int(cfg.random_negatives_per_shot))
            rnd = rng.choice(remaining, size=random_n, replace=False) if random_n else np.empty((0,), dtype=np.int64)
            chosen = np.concatenate([hard, rnd])
        if not len(chosen):
            continue
        neg = neg[chosen]
        per_pos = max(1, int(cfg.pairs_per_positive))
        for p in pos:
            count = min(per_pos, len(neg))
            indexes = rng.choice(len(neg), size=count, replace=False if count <= len(neg) else True)
            pairs.append((p[None, :] - neg[indexes]).astype(np.float32))
    if not pairs:
        raise RuntimeError("V2.21.4 could not build any pairwise training examples")
    return np.concatenate(pairs, axis=0).astype(np.float32)


def _train_pairwise_linear(
    pairs: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    x = np.asarray(pairs, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    w = np.zeros((x.shape[1],), dtype=np.float32) if initial is None else np.asarray(initial, dtype=np.float32).copy()
    m = np.zeros_like(w); v = np.zeros_like(w)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    losses: list[float] = []
    step = 0
    batch_size = min(2048, max(128, len(x)))
    for epoch in range(max(1, int(epochs))):
        order = rng.permutation(len(x))
        epoch_loss = 0.0; seen = 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            b = x[idx]
            margin = np.clip(b @ w, -30.0, 30.0)
            # softplus(-margin)
            loss_vec = np.logaddexp(0.0, -margin)
            sig = 1.0 / (1.0 + np.exp(np.clip(margin, -30.0, 30.0)))
            grad = -np.mean(b * sig[:, None], axis=0) + float(l2) * w
            step += 1
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)
            mhat = m / (1.0 - beta1 ** step)
            vhat = v / (1.0 - beta2 ** step)
            w -= float(learning_rate) * mhat / (np.sqrt(vhat) + eps)
            epoch_loss += float(np.sum(loss_vec)); seen += len(b)
        losses.append(float(epoch_loss / max(1, seen)))
    return w.astype(np.float32), losses


def train_dense_ranker_v2214(
    shots: Sequence[DenseShotTrainingDataV2214],
    *,
    training_config: DenseTrainingConfigV2214 | None = None,
    pool_config: DensePoolConfigV2214 | None = None,
) -> tuple[DenseRankerV2214, dict[str, Any]]:
    cfg = training_config or DenseTrainingConfigV2214()
    pcfg = pool_config or DensePoolConfigV2214()
    if len(shots) < 2:
        raise RuntimeError("V2.21.4 needs at least two DEVELOPMENT full-frame shots")
    mean, scale = _standardise_training(shots, cfg)
    pairs1 = _make_pairs(shots, mean, scale, cfg, stage2=False)
    w1, losses1 = _train_pairwise_linear(
        pairs1,
        epochs=cfg.stage1_epochs,
        learning_rate=cfg.learning_rate,
        l2=cfg.l2,
        seed=cfg.seed,
    )
    pairs2 = _make_pairs(shots, mean, scale, cfg, model_weights=w1, stage2=True)
    w2, losses2 = _train_pairwise_linear(
        pairs2,
        epochs=cfg.stage2_epochs,
        learning_rate=cfg.learning_rate * 0.55,
        l2=cfg.l2,
        seed=cfg.seed + 1,
        initial=w1,
    )
    positives = int(sum(len(s.positive_features) for s in shots))
    negative_pool = int(sum(np.sum(s.pool_distances >= cfg.negative_exclusion_radius_px) for s in shots))
    pool20 = float(np.mean([
        bool(np.any(s.pool_distances <= 20.0)) for s in shots
    ]))
    pool42 = float(np.mean([
        bool(np.any(s.pool_distances <= 42.0)) for s in shots
    ]))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "training_scope": "physical_development_fullframe_only",
        "development_shots": len(shots),
        "shot_keys": [s.shot_key for s in shots],
        "positive_samples": positives,
        "negative_pool_samples": negative_pool,
        "pair_count_stage1": int(len(pairs1)),
        "pair_count_stage2": int(len(pairs2)),
        "stage1_final_loss": float(losses1[-1]),
        "stage2_final_loss": float(losses2[-1]),
        "development_dense_pool_oracle20": pool20,
        "development_dense_pool_oracle42": pool42,
        "mean_mask_fraction": float(np.mean([s.mask_fraction for s in shots])),
        "pool_config": asdict(pcfg),
        "training_config": asdict(cfg),
        "semantic_note": (
            "No confirmation/holdout shot is used for fitting, standardisation, hard-negative mining, "
            "feature selection or top-k selection. Splits may still be provisional when only one full-frame session exists."
        ),
    }
    model = DenseRankerV2214(
        feature_names=tuple(FEATURE_NAMES),
        mean=mean,
        scale=scale,
        weights=w2,
        metadata=metadata,
    )
    report = {
        **metadata,
        "weight_by_feature": {name: float(value) for name, value in zip(FEATURE_NAMES, w2)},
        "stage1_loss_head_tail": [float(losses1[0]), float(losses1[-1])],
        "stage2_loss_head_tail": [float(losses2[0]), float(losses2[-1])],
    }
    return model, report


def dense_pool_config_dict_v2214(config: DensePoolConfigV2214 | None = None) -> dict[str, Any]:
    return asdict(config or DensePoolConfigV2214())


def dense_training_config_dict_v2214(config: DenseTrainingConfigV2214 | None = None) -> dict[str, Any]:
    return asdict(config or DenseTrainingConfigV2214())
