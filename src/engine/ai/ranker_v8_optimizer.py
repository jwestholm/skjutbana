from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from src.engine.ai.ranker_v7 import FEATURE_KEYS
from src.engine.ai.ranker_v8 import percentile_feature_matrix


FEATURE_INDEX = {key: index for index, key in enumerate(FEATURE_KEYS)}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float(default)
    except Exception:
        return float(default)


def _pool(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and bool(candidate.get("membership", {}).get("hypothesis_pool"))
    ]


def _distance(candidate: dict[str, Any]) -> float:
    return _finite(candidate.get("distance_gt_px"), float("inf"))


def _baseline_rank(candidate: dict[str, Any]) -> int:
    ranks = candidate.get("ranks")
    if isinstance(ranks, dict):
        try:
            value = ranks.get("baseline")
            if value is not None:
                return int(value)
        except Exception:
            pass
    return 10**9


def _label_weight(distance: float) -> float:
    if distance <= 12.0:
        return 1.0
    if distance <= 20.0:
        return 0.72
    if distance <= 42.0:
        return 0.18
    return 0.0


@dataclass
class PreparedShot:
    row_id: str
    sequence: int
    features: np.ndarray
    percentiles: np.ndarray
    distances: np.ndarray
    baseline_ranks: np.ndarray
    baseline_percentiles: np.ndarray

    @property
    def size(self) -> int:
        return int(self.features.shape[0])


@dataclass
class FeatureProfile:
    direction: np.ndarray
    strength: np.ndarray
    comparisons: np.ndarray
    win_rate: np.ndarray
    correlation: np.ndarray


@dataclass(frozen=True)
class SearchConfig:
    feature_count: int
    min_strength: float
    corr_limit: float
    evidence_power: float
    weight_power: float
    baseline_blend: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_count": self.feature_count,
            "min_strength": self.min_strength,
            "corr_limit": self.corr_limit,
            "evidence_power": self.evidence_power,
            "weight_power": self.weight_power,
            "baseline_blend": self.baseline_blend,
        }


@dataclass
class FittedMonotonicModel:
    feature_keys: list[str]
    weights: np.ndarray
    evidence_power: float
    baseline_blend: float
    feature_profile: dict[str, dict[str, float]]

    def as_payload(self) -> dict[str, Any]:
        return {
            "feature_keys": list(self.feature_keys),
            "weights": {
                key: float(weight)
                for key, weight in zip(self.feature_keys, self.weights.tolist())
            },
            "evidence_power": float(self.evidence_power),
            "baseline_blend": float(self.baseline_blend),
            "feature_profile": self.feature_profile,
        }


def prepare_rows(rows: Sequence[dict[str, Any]]) -> list[PreparedShot]:
    prepared: list[PreparedShot] = []
    for index, row in enumerate(rows):
        pool = _pool(row)
        if not pool:
            continue
        feature_dicts = [
            candidate.get("features") if isinstance(candidate.get("features"), dict) else {}
            for candidate in pool
        ]
        features = np.asarray(
            [
                [_finite(feature_dict.get(key)) for key in FEATURE_KEYS]
                for feature_dict in feature_dicts
            ],
            dtype=np.float64,
        )
        percentiles = np.asarray(
            percentile_feature_matrix(feature_dicts, FEATURE_KEYS),
            dtype=np.float64,
        )
        distances = np.asarray([_distance(candidate) for candidate in pool], dtype=np.float64)
        baseline_ranks = np.asarray([_baseline_rank(candidate) for candidate in pool], dtype=np.int64)
        baseline_values = np.asarray(
            [_finite(feature_dict.get("baseline_score")) for feature_dict in feature_dicts],
            dtype=np.float64,
        )
        # stable average-rank percentile via ranker_v8 helper
        baseline_percentiles = np.asarray(
            percentile_feature_matrix(
                [{"baseline_score": float(value)} for value in baseline_values],
                ["baseline_score"],
            ),
            dtype=np.float64,
        )[:, 0]
        sequence = int(row.get("sequence", index + 1) or index + 1)
        row_id = f"{row.get('session_id', 'session')}:{sequence}"
        prepared.append(
            PreparedShot(
                row_id=row_id,
                sequence=sequence,
                features=features,
                percentiles=percentiles,
                distances=distances,
                baseline_ranks=baseline_ranks,
                baseline_percentiles=baseline_percentiles,
            )
        )
    return prepared


def stable_confirmation_split(
    shots: Sequence[PreparedShot],
    *,
    confirmation_fraction: float = 0.20,
) -> tuple[list[PreparedShot], list[PreparedShot]]:
    development: list[PreparedShot] = []
    confirmation: list[PreparedShot] = []
    threshold = max(1, min(9999, int(round(10000 * confirmation_fraction))))
    for shot in shots:
        digest = hashlib.sha1(shot.row_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10000
        if bucket < threshold:
            confirmation.append(shot)
        else:
            development.append(shot)
    # Very small datasets need both sides non-empty.
    if len(confirmation) < 8 and len(shots) >= 20:
        ordered = sorted(shots, key=lambda shot: shot.row_id)
        count = max(8, int(round(len(shots) * confirmation_fraction)))
        confirmation_ids = {shot.row_id for shot in ordered[:: max(1, len(ordered) // count)][:count]}
        development = [shot for shot in shots if shot.row_id not in confirmation_ids]
        confirmation = [shot for shot in shots if shot.row_id in confirmation_ids]
    return development, confirmation


def _positive_and_negatives(
    shot: PreparedShot,
    *,
    negative_radius_px: float = 55.0,
    hard_negatives: int = 12,
) -> tuple[int | None, list[int], float]:
    if shot.size == 0:
        return None, [], 0.0
    positive = int(np.argmin(shot.distances))
    distance = float(shot.distances[positive])
    weight = _label_weight(distance)
    if weight <= 0.0:
        return None, [], 0.0
    negatives = [
        index
        for index in range(shot.size)
        if index != positive and float(shot.distances[index]) >= float(negative_radius_px)
    ]
    negatives.sort(key=lambda index: int(shot.baseline_ranks[index]))
    return positive, negatives[: max(1, int(hard_negatives))], weight


def fit_feature_profile(
    shots: Sequence[PreparedShot],
    *,
    hard_negatives: int = 12,
    negative_radius_px: float = 55.0,
    correlation_sample_limit: int = 8000,
) -> FeatureProfile:
    feature_count = len(FEATURE_KEYS)
    high = np.zeros((feature_count,), dtype=np.float64)
    low = np.zeros((feature_count,), dtype=np.float64)
    comparisons = np.zeros((feature_count,), dtype=np.float64)

    for shot in shots:
        positive, negatives, label_weight = _positive_and_negatives(
            shot,
            negative_radius_px=negative_radius_px,
            hard_negatives=hard_negatives,
        )
        if positive is None or not negatives:
            continue
        pos = shot.features[positive]
        for neg_order, negative in enumerate(negatives):
            hardness = 1.0 / math.sqrt(1.0 + 0.12 * neg_order)
            weight = label_weight * hardness
            diff = pos - shot.features[negative]
            high += weight * (diff > 1e-9)
            low += weight * (diff < -1e-9)
            comparisons += weight * (np.abs(diff) > 1e-9)

    total = high + low
    direction = np.where(high >= low, 1.0, -1.0)
    win = np.divide(
        np.maximum(high, low),
        np.maximum(total, 1e-12),
        out=np.full_like(total, 0.5),
        where=total > 1e-12,
    )
    strength = np.clip(2.0 * (win - 0.5), 0.0, 1.0)

    samples: list[np.ndarray] = []
    remaining = max(0, int(correlation_sample_limit))
    for shot in shots:
        if remaining <= 0:
            break
        take = min(shot.size, remaining)
        if take > 0:
            # Deterministic spread through the candidate list.
            indices = np.linspace(0, shot.size - 1, num=take, dtype=int)
            samples.append(shot.features[indices])
            remaining -= take
    if samples:
        matrix = np.vstack(samples)
        std = np.std(matrix, axis=0)
        safe = std > 1e-9
        correlation = np.eye(feature_count, dtype=np.float64)
        if int(np.sum(safe)) >= 2:
            corr_sub = np.corrcoef(matrix[:, safe], rowvar=False)
            safe_indices = np.flatnonzero(safe)
            for i, fi in enumerate(safe_indices):
                for j, fj in enumerate(safe_indices):
                    value = float(corr_sub[i, j]) if np.ndim(corr_sub) == 2 else 0.0
                    correlation[fi, fj] = value if math.isfinite(value) else 0.0
    else:
        correlation = np.eye(feature_count, dtype=np.float64)

    return FeatureProfile(
        direction=direction,
        strength=strength,
        comparisons=comparisons,
        win_rate=win,
        correlation=correlation,
    )


def fit_model_from_profile(
    profile: FeatureProfile,
    config: SearchConfig,
) -> FittedMonotonicModel:
    eligible = [
        index
        for index in range(len(FEATURE_KEYS))
        if float(profile.strength[index]) >= float(config.min_strength)
        and float(profile.comparisons[index]) >= 8.0
    ]
    eligible.sort(
        key=lambda index: (
            float(profile.strength[index]),
            float(profile.comparisons[index]),
        ),
        reverse=True,
    )

    selected: list[int] = []
    for index in eligible:
        if any(
            abs(float(profile.correlation[index, previous])) >= float(config.corr_limit)
            for previous in selected
        ):
            continue
        selected.append(index)
        if len(selected) >= int(config.feature_count):
            break

    # If aggressive correlation filtering leaves too few features, backfill
    # with the strongest remaining evidence rather than returning no model.
    if len(selected) < min(2, int(config.feature_count)):
        for index in eligible:
            if index not in selected:
                selected.append(index)
            if len(selected) >= min(max(2, int(config.feature_count)), len(eligible)):
                break

    if not selected:
        # compactness is the safest deterministic fallback seen in V2.9 data.
        fallback = FEATURE_INDEX.get("compactness", 0)
        selected = [fallback]

    strengths = np.asarray([float(profile.strength[index]) for index in selected], dtype=np.float64)
    magnitude = np.maximum(strengths, 1e-4) ** float(config.weight_power)
    magnitude /= max(1e-9, float(np.sum(magnitude)))
    signed = magnitude * np.asarray([float(profile.direction[index]) for index in selected])

    feature_profile = {
        FEATURE_KEYS[index]: {
            "direction": float(profile.direction[index]),
            "strength": float(profile.strength[index]),
            "win_rate": float(profile.win_rate[index]),
            "comparisons": float(profile.comparisons[index]),
        }
        for index in selected
    }
    return FittedMonotonicModel(
        feature_keys=[FEATURE_KEYS[index] for index in selected],
        weights=signed,
        evidence_power=float(config.evidence_power),
        baseline_blend=float(config.baseline_blend),
        feature_profile=feature_profile,
    )


def score_shot(shot: PreparedShot, model: FittedMonotonicModel) -> np.ndarray:
    indices = [FEATURE_INDEX[key] for key in model.feature_keys]
    centered = 2.0 * (shot.percentiles[:, indices] - 0.5)
    shaped = np.sign(centered) * (np.abs(centered) ** float(model.evidence_power))
    raw = shaped @ model.weights
    if model.baseline_blend > 0.0:
        baseline = 2.0 * shot.baseline_percentiles - 1.0
        raw = (1.0 - model.baseline_blend) * raw + model.baseline_blend * baseline
    return np.asarray(raw, dtype=np.float64)


def _rank_for_radius(shot: PreparedShot, scores: np.ndarray, radius: float) -> int | None:
    order = np.lexsort((shot.baseline_ranks, -scores))
    for rank, index in enumerate(order, start=1):
        if float(shot.distances[index]) <= float(radius):
            return rank
    return None


def _baseline_rank_for_radius(shot: PreparedShot, radius: float) -> int | None:
    order = np.argsort(shot.baseline_ranks, kind="stable")
    for rank, index in enumerate(order, start=1):
        if float(shot.distances[index]) <= float(radius):
            return rank
    return None


def _metrics(ranks: Sequence[int]) -> dict[str, Any]:
    if not ranks:
        return {
            "covered": 0,
            "top1_pct": 0.0,
            "top3_pct": 0.0,
            "top5_pct": 0.0,
            "median_rank": None,
            "mrr": None,
        }
    array = np.asarray(list(ranks), dtype=np.float64)
    return {
        "covered": len(ranks),
        "top1_pct": round(100.0 * float(np.mean(array <= 1.0)), 3),
        "top3_pct": round(100.0 * float(np.mean(array <= 3.0)), 3),
        "top5_pct": round(100.0 * float(np.mean(array <= 5.0)), 3),
        "median_rank": float(np.median(array)),
        "mrr": round(float(np.mean(1.0 / array)), 6),
    }


def evaluate_model(
    shots: Sequence[PreparedShot],
    model: FittedMonotonicModel,
) -> dict[str, Any]:
    model_ranks: dict[int, list[int]] = {10: [], 20: [], 42: []}
    baseline_ranks: dict[int, list[int]] = {10: [], 20: [], 42: []}
    for shot in shots:
        scores = score_shot(shot, model)
        for radius in (10, 20, 42):
            baseline_rank = _baseline_rank_for_radius(shot, radius)
            if baseline_rank is None:
                continue
            model_rank = _rank_for_radius(shot, scores, radius)
            baseline_ranks[radius].append(int(baseline_rank))
            model_ranks[radius].append(int(model_rank if model_rank is not None else shot.size + 1))
    return {
        "model": {str(radius): _metrics(model_ranks[radius]) for radius in (10, 20, 42)},
        "baseline": {str(radius): _metrics(baseline_ranks[radius]) for radius in (10, 20, 42)},
    }


def objective(result: dict[str, Any]) -> float:
    model20 = result["model"]["20"]
    model42 = result["model"]["42"]
    base20 = result["baseline"]["20"]
    # Reward genuine top-of-list success heavily. A model that merely improves
    # median rank but never selects the hole cannot win V2.10 optimization.
    gain_top1_20 = float(model20["top1_pct"]) - float(base20["top1_pct"])
    return (
        4.2 * float(model20["top1_pct"])
        + 2.0 * float(model20["top3_pct"])
        + 0.75 * float(model20["top5_pct"])
        + 1.0 * float(model42["top1_pct"])
        + 0.45 * float(model42["top3_pct"])
        + 145.0 * float(model20["mrr"] or 0.0)
        + 50.0 * float(model42["mrr"] or 0.0)
        + 1.5 * gain_top1_20
        - 0.020 * float(model20["median_rank"] or 999.0)
        - 0.006 * float(model42["median_rank"] or 999.0)
    )


def default_search_configs() -> list[SearchConfig]:
    configs: list[SearchConfig] = []
    for values in itertools.product(
        (3, 5, 8, 12),          # feature_count
        (0.15, 0.25, 0.35),     # min_strength
        (0.86, 0.94),           # corr_limit
        (0.70, 1.00, 1.45),     # evidence_power
        (0.75, 1.50),           # weight_power
        (0.00, 0.08),           # baseline_blend
    ):
        configs.append(SearchConfig(*values))
    return configs


def cross_validate_config(
    shots: Sequence[PreparedShot],
    config: SearchConfig,
    *,
    folds: int,
) -> dict[str, Any]:
    aggregated_model: dict[int, list[int]] = {10: [], 20: [], 42: []}
    aggregated_baseline: dict[int, list[int]] = {10: [], 20: [], 42: []}
    selected_features: list[list[str]] = []

    for fold in range(folds):
        train = [shot for index, shot in enumerate(shots) if index % folds != fold]
        test = [shot for index, shot in enumerate(shots) if index % folds == fold]
        if not train or not test:
            continue
        profile = fit_feature_profile(train)
        model = fit_model_from_profile(profile, config)
        selected_features.append(list(model.feature_keys))
        for shot in test:
            scores = score_shot(shot, model)
            for radius in (10, 20, 42):
                baseline_rank = _baseline_rank_for_radius(shot, radius)
                if baseline_rank is None:
                    continue
                model_rank = _rank_for_radius(shot, scores, radius)
                aggregated_baseline[radius].append(int(baseline_rank))
                aggregated_model[radius].append(int(model_rank if model_rank is not None else shot.size + 1))

    result = {
        "model": {str(radius): _metrics(aggregated_model[radius]) for radius in (10, 20, 42)},
        "baseline": {str(radius): _metrics(aggregated_baseline[radius]) for radius in (10, 20, 42)},
        "selected_features_by_fold": selected_features,
    }
    result["objective"] = round(objective(result), 6)
    return result


def search_configs(
    development: Sequence[PreparedShot],
    configs: Sequence[SearchConfig] | None = None,
) -> list[dict[str, Any]]:
    if configs is None:
        configs = default_search_configs()
    folds = 5 if len(development) >= 70 else 4 if len(development) >= 40 else 3
    results: list[dict[str, Any]] = []
    for config in configs:
        cv = cross_validate_config(development, config, folds=folds)
        results.append({
            "config": config,
            "cv": cv,
            "objective": float(cv["objective"]),
        })
    results.sort(key=lambda item: float(item["objective"]), reverse=True)
    return results


def feature_profile_table(profile: FeatureProfile, limit: int = 20) -> list[dict[str, Any]]:
    rows = []
    for index, key in enumerate(FEATURE_KEYS):
        rows.append({
            "feature": key,
            "direction": "HIGH" if profile.direction[index] > 0 else "LOW",
            "strength": round(float(profile.strength[index]), 5),
            "win_rate": round(float(profile.win_rate[index]), 5),
            "comparisons": round(float(profile.comparisons[index]), 2),
        })
    rows.sort(key=lambda row: (float(row["strength"]), float(row["comparisons"])), reverse=True)
    return rows[: max(1, int(limit))]


def recommendation(
    *,
    shots: int,
    development_cv: dict[str, Any],
    confirmation: dict[str, Any],
) -> dict[str, Any]:
    cv20 = development_cv["model"]["20"]
    cvbase20 = development_cv["baseline"]["20"]
    conf20 = confirmation["model"]["20"]
    confbase20 = confirmation["baseline"]["20"]
    conf42 = confirmation["model"]["42"]
    confbase42 = confirmation["baseline"]["42"]

    cv_gain = float(cv20["top1_pct"]) - float(cvbase20["top1_pct"])
    conf_gain = float(conf20["top1_pct"]) - float(confbase20["top1_pct"])
    conf_top3_gain = float(conf20["top3_pct"]) - float(confbase20["top3_pct"])
    conf42_top3_gain = float(conf42["top3_pct"]) - float(confbase42["top3_pct"])

    enough_data = (
        shots >= 350
        and int(conf20.get("covered", 0)) >= 20
        and int(conf42.get("covered", 0)) >= 50
    )
    passes_quality = (
        cv_gain >= 2.0
        and conf_gain >= 0.0
        and conf_top3_gain >= 2.0
        and conf42_top3_gain >= -2.0
    )
    return {
        "enough_data": bool(enough_data),
        "passes_quality": bool(passes_quality),
        "recommended_for_future_authority": bool(enough_data and passes_quality),
        "development_top1_gain_20_pp": round(cv_gain, 3),
        "confirmation_top1_gain_20_pp": round(conf_gain, 3),
        "confirmation_top3_gain_20_pp": round(conf_top3_gain, 3),
        "confirmation_top3_gain_42_pp": round(conf42_top3_gain, 3),
        "note": "V2.10 is shadow-only regardless of this recommendation.",
    }


__all__ = [
    "FEATURE_KEYS",
    "FeatureProfile",
    "FittedMonotonicModel",
    "PreparedShot",
    "SearchConfig",
    "cross_validate_config",
    "default_search_configs",
    "evaluate_model",
    "feature_profile_table",
    "fit_feature_profile",
    "fit_model_from_profile",
    "objective",
    "prepare_rows",
    "recommendation",
    "score_shot",
    "search_configs",
    "stable_confirmation_split",
]
