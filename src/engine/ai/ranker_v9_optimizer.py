from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.engine.ai.ranker_v9 import (
    FEATURE_FAMILIES,
    PHYSICAL_FEATURE_KEYS,
    percentile_matrix,
)


FEATURE_INDEX = {key: index for index, key in enumerate(PHYSICAL_FEATURE_KEYS)}


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
            return int(value) if value is not None else 10**9
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
    camera_xy: np.ndarray

    @property
    def size(self) -> int:
        return int(self.features.shape[0])


@dataclass
class FeatureEvidence:
    direction: np.ndarray
    win_rate: np.ndarray
    strength: np.ndarray
    comparisons: np.ndarray
    auc_like: np.ndarray
    correlation: np.ndarray


@dataclass(frozen=True)
class RuleConfig:
    feature_keys: tuple[str, ...]
    weights: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_keys": list(self.feature_keys),
            "weights": [float(value) for value in self.weights],
        }


@dataclass
class FittedRule:
    feature_keys: list[str]
    signed_weights: np.ndarray
    feature_evidence: dict[str, dict[str, float]]

    def payload(self) -> dict[str, Any]:
        return {
            "feature_keys": list(self.feature_keys),
            "weights": {
                key: float(weight)
                for key, weight in zip(self.feature_keys, self.signed_weights.tolist())
            },
            "feature_evidence": self.feature_evidence,
        }


def prepare_rows(rows: Sequence[dict[str, Any]]) -> list[PreparedShot]:
    prepared: list[PreparedShot] = []

    for index, row in enumerate(rows):
        pool = _pool(row)
        if not pool:
            continue

        feature_dicts: list[dict[str, float]] = []
        for candidate in pool:
            source = candidate.get("features")
            source = source if isinstance(source, dict) else {}
            feature_dicts.append({
                key: _finite(source.get(key))
                for key in PHYSICAL_FEATURE_KEYS
            })

        features = np.asarray(
            [
                [feature_dict[key] for key in PHYSICAL_FEATURE_KEYS]
                for feature_dict in feature_dicts
            ],
            dtype=np.float64,
        )
        percentiles = np.asarray(
            percentile_matrix(feature_dicts, PHYSICAL_FEATURE_KEYS),
            dtype=np.float64,
        )
        distances = np.asarray(
            [_distance(candidate) for candidate in pool],
            dtype=np.float64,
        )
        baseline_ranks = np.asarray(
            [_baseline_rank(candidate) for candidate in pool],
            dtype=np.int64,
        )
        camera_xy = np.asarray(
            [
                [
                    _finite(candidate.get("camera_x")),
                    _finite(candidate.get("camera_y")),
                ]
                for candidate in pool
            ],
            dtype=np.float64,
        )

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
                camera_xy=camera_xy,
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
    threshold = max(1, min(9999, int(round(10000.0 * confirmation_fraction))))

    for shot in shots:
        digest = hashlib.sha1(shot.row_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10000
        if bucket < threshold:
            confirmation.append(shot)
        else:
            development.append(shot)

    if len(confirmation) < 8 and len(shots) >= 20:
        ordered = sorted(shots, key=lambda shot: shot.row_id)
        wanted = max(8, int(round(len(shots) * confirmation_fraction)))
        stride = max(1, len(ordered) // wanted)
        confirmation_ids = {
            shot.row_id
            for shot in ordered[::stride][:wanted]
        }
        development = [
            shot for shot in shots if shot.row_id not in confirmation_ids
        ]
        confirmation = [
            shot for shot in shots if shot.row_id in confirmation_ids
        ]

    return development, confirmation


def _positive_index(shot: PreparedShot) -> tuple[int | None, float]:
    if shot.size <= 0:
        return None, 0.0
    index = int(np.argmin(shot.distances))
    distance = float(shot.distances[index])
    return (index, _label_weight(distance))


def broad_negative_indices(
    shot: PreparedShot,
    *,
    positive_index: int,
    negative_radius_px: float = 55.0,
    max_negatives: int = 64,
) -> list[int]:
    """Deterministic broad negative population.

    V2.10 only compared GT with the candidates favoured by the baseline.
    V2.11 explicitly samples many false-candidate phenotypes:
    - baseline hard negatives,
    - high AND low extremes for every physical feature,
    - candidates spread through baseline rank,
    - spatially diverse candidates,
    - deterministic spread through all eligible negatives.
    """
    eligible = [
        index
        for index in range(shot.size)
        if index != positive_index
        and float(shot.distances[index]) >= float(negative_radius_px)
    ]
    if len(eligible) <= max_negatives:
        return eligible

    selected: list[int] = []
    seen: set[int] = set()

    def add(indices: Sequence[int]) -> None:
        for index in indices:
            if index not in seen and index in eligible:
                seen.add(index)
                selected.append(index)
                if len(selected) >= max_negatives:
                    return

    # 1) What the existing baseline considers dangerous.
    add(sorted(eligible, key=lambda i: int(shot.baseline_ranks[i]))[:12])

    # 2) Feature extremes. We include both directions because a false positive
    # phenotype can live at either end of a physical signal.
    for feature_index in range(len(PHYSICAL_FEATURE_KEYS)):
        if len(selected) >= max_negatives:
            break
        ordered = sorted(
            eligible,
            key=lambda i: float(shot.features[i, feature_index]),
        )
        add(ordered[:1])
        add(ordered[-1:])

    # 3) Baseline-rank diversity through the whole list.
    ordered_by_baseline = sorted(eligible, key=lambda i: int(shot.baseline_ranks[i]))
    if ordered_by_baseline:
        positions = np.linspace(
            0,
            len(ordered_by_baseline) - 1,
            num=min(12, len(ordered_by_baseline)),
            dtype=int,
        )
        add([ordered_by_baseline[int(position)] for position in positions])

    # 4) Spatial diversity: one farthest-from-current-selected point at a time.
    if len(selected) < max_negatives:
        remaining = [i for i in eligible if i not in seen]
        if remaining:
            seed = remaining[0]
            spatial = [seed]
            while remaining and len(spatial) < 12:
                last_points = shot.camera_xy[np.asarray(spatial, dtype=int)]
                best = max(
                    remaining,
                    key=lambda i: float(
                        np.min(
                            np.linalg.norm(
                                last_points - shot.camera_xy[i],
                                axis=1,
                            )
                        )
                    ),
                )
                spatial.append(best)
                remaining.remove(best)
            add(spatial)

    # 5) Final deterministic spread through remaining negatives.
    if len(selected) < max_negatives:
        remaining = [i for i in eligible if i not in seen]
        if remaining:
            positions = np.linspace(
                0,
                len(remaining) - 1,
                num=min(max_negatives - len(selected), len(remaining)),
                dtype=int,
            )
            add([remaining[int(position)] for position in positions])

    return selected[:max_negatives]


def _correlation_matrix(shots: Sequence[PreparedShot]) -> np.ndarray:
    samples: list[np.ndarray] = []
    remaining = 8000

    for shot in shots:
        if remaining <= 0:
            break
        take = min(shot.size, remaining)
        if take <= 0:
            continue
        indices = np.linspace(0, shot.size - 1, num=take, dtype=int)
        samples.append(shot.features[indices])
        remaining -= take

    count = len(PHYSICAL_FEATURE_KEYS)
    result = np.eye(count, dtype=np.float64)

    if not samples:
        return result

    matrix = np.vstack(samples)
    std = np.std(matrix, axis=0)
    safe = std > 1e-9

    if int(np.sum(safe)) < 2:
        return result

    corr = np.corrcoef(matrix[:, safe], rowvar=False)
    safe_indices = np.flatnonzero(safe)

    for i, feature_i in enumerate(safe_indices):
        for j, feature_j in enumerate(safe_indices):
            value = float(corr[i, j]) if np.ndim(corr) == 2 else 0.0
            result[feature_i, feature_j] = value if math.isfinite(value) else 0.0

    return result


def fit_feature_evidence(
    shots: Sequence[PreparedShot],
    *,
    negative_radius_px: float = 55.0,
    max_negatives: int = 64,
) -> FeatureEvidence:
    count = len(PHYSICAL_FEATURE_KEYS)
    high = np.zeros(count, dtype=np.float64)
    low = np.zeros(count, dtype=np.float64)
    comparisons = np.zeros(count, dtype=np.float64)

    for shot in shots:
        positive, label_weight = _positive_index(shot)
        if positive is None or label_weight <= 0.0:
            continue

        negatives = broad_negative_indices(
            shot,
            positive_index=positive,
            negative_radius_px=negative_radius_px,
            max_negatives=max_negatives,
        )
        if not negatives:
            continue

        pos = shot.features[positive]
        for negative in negatives:
            diff = pos - shot.features[negative]
            high += label_weight * (diff > 1e-9)
            low += label_weight * (diff < -1e-9)
            comparisons += label_weight * (np.abs(diff) > 1e-9)

    total = high + low
    direction = np.where(high >= low, 1.0, -1.0)
    win_rate = np.divide(
        np.maximum(high, low),
        np.maximum(total, 1e-12),
        out=np.full_like(total, 0.5),
        where=total > 1e-12,
    )
    strength = np.clip(2.0 * (win_rate - 0.5), 0.0, 1.0)

    # "AUC-like" is intentionally the same pairwise probability but kept as a
    # separate named metric in reports: P(correct candidate beats broad false
    # candidate when this feature is used in its learned monotonic direction).
    auc_like = win_rate.copy()

    return FeatureEvidence(
        direction=direction,
        win_rate=win_rate,
        strength=strength,
        comparisons=comparisons,
        auc_like=auc_like,
        correlation=_correlation_matrix(shots),
    )


def _score_feature(
    shot: PreparedShot,
    feature_key: str,
    direction: float,
) -> np.ndarray:
    index = FEATURE_INDEX[feature_key]
    centered = 2.0 * (shot.percentiles[:, index] - 0.5)
    return np.asarray(direction * centered, dtype=np.float64)


def score_rule(shot: PreparedShot, rule: FittedRule) -> np.ndarray:
    score = np.zeros(shot.size, dtype=np.float64)
    for key, signed_weight in zip(rule.feature_keys, rule.signed_weights):
        index = FEATURE_INDEX[key]
        centered = 2.0 * (shot.percentiles[:, index] - 0.5)
        score += float(signed_weight) * centered
    return score


def fit_rule(
    evidence: FeatureEvidence,
    config: RuleConfig,
) -> FittedRule:
    directions = []
    feature_evidence: dict[str, dict[str, float]] = {}

    for key in config.feature_keys:
        index = FEATURE_INDEX[key]
        directions.append(float(evidence.direction[index]))
        feature_evidence[key] = {
            "direction": float(evidence.direction[index]),
            "win_rate": float(evidence.win_rate[index]),
            "strength": float(evidence.strength[index]),
            "comparisons": float(evidence.comparisons[index]),
            "auc_like": float(evidence.auc_like[index]),
        }

    magnitudes = np.asarray(config.weights, dtype=np.float64)
    magnitudes = np.maximum(magnitudes, 0.0)
    total = float(np.sum(magnitudes))
    if total <= 1e-12:
        magnitudes = np.ones(len(config.feature_keys), dtype=np.float64)
        total = float(len(config.feature_keys))
    magnitudes /= total

    signed = magnitudes * np.asarray(directions, dtype=np.float64)

    return FittedRule(
        feature_keys=list(config.feature_keys),
        signed_weights=signed,
        feature_evidence=feature_evidence,
    )


def _rank_for_radius(
    shot: PreparedShot,
    scores: np.ndarray,
    radius: float,
) -> int | None:
    # Baseline rank only breaks exact score ties.
    order = np.lexsort((shot.baseline_ranks, -scores))
    for rank, index in enumerate(order, start=1):
        if float(shot.distances[index]) <= float(radius):
            return rank
    return None


def _baseline_rank_for_radius(
    shot: PreparedShot,
    radius: float,
) -> int | None:
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


def evaluate_rule(
    shots: Sequence[PreparedShot],
    rule: FittedRule,
) -> dict[str, Any]:
    model_ranks: dict[int, list[int]] = {10: [], 20: [], 42: []}
    baseline_ranks: dict[int, list[int]] = {10: [], 20: [], 42: []}

    for shot in shots:
        scores = score_rule(shot, rule)

        for radius in (10, 20, 42):
            baseline_rank = _baseline_rank_for_radius(shot, radius)
            if baseline_rank is None:
                continue

            rank = _rank_for_radius(shot, scores, radius)
            baseline_ranks[radius].append(int(baseline_rank))
            model_ranks[radius].append(
                int(rank if rank is not None else shot.size + 1)
            )

    return {
        "model": {
            str(radius): _metrics(model_ranks[radius])
            for radius in (10, 20, 42)
        },
        "baseline": {
            str(radius): _metrics(baseline_ranks[radius])
            for radius in (10, 20, 42)
        },
    }


def _objective(result: dict[str, Any]) -> float:
    model20 = result["model"]["20"]
    base20 = result["baseline"]["20"]
    model42 = result["model"]["42"]
    base42 = result["baseline"]["42"]

    # Hard philosophy for V2.11:
    # "never select a model that loses Top-1<=20 to baseline".
    top1_gain20 = float(model20["top1_pct"]) - float(base20["top1_pct"])
    if top1_gain20 < -1e-9:
        return -1_000_000.0 + top1_gain20

    top3_gain20 = float(model20["top3_pct"]) - float(base20["top3_pct"])
    top1_gain42 = float(model42["top1_pct"]) - float(base42["top1_pct"])
    top3_gain42 = float(model42["top3_pct"]) - float(base42["top3_pct"])

    return (
        12.0 * top1_gain20
        + 4.0 * top3_gain20
        + 1.5 * top1_gain42
        + 0.8 * top3_gain42
        + 180.0 * float(model20["mrr"] or 0.0)
        + 55.0 * float(model42["mrr"] or 0.0)
        - 0.01 * float(model20["median_rank"] or 999.0)
        - 0.003 * float(model42["median_rank"] or 999.0)
    )


def _cv_folds(shots: Sequence[PreparedShot]) -> int:
    return 5 if len(shots) >= 70 else 4 if len(shots) >= 40 else 3


def build_cv_context(
    shots: Sequence[PreparedShot],
) -> list[tuple[FeatureEvidence, list[PreparedShot]]]:
    """Fit broad-negative evidence once per fold and reuse for every rule."""
    folds = _cv_folds(shots)
    context: list[tuple[FeatureEvidence, list[PreparedShot]]] = []

    for fold in range(folds):
        train = [
            shot
            for index, shot in enumerate(shots)
            if index % folds != fold
        ]
        test = [
            shot
            for index, shot in enumerate(shots)
            if index % folds == fold
        ]
        if not train or not test:
            continue
        context.append((fit_feature_evidence(train), test))

    return context


def cross_validate_config(
    shots: Sequence[PreparedShot],
    config: RuleConfig,
    *,
    cv_context: Sequence[tuple[FeatureEvidence, list[PreparedShot]]] | None = None,
) -> dict[str, Any]:
    context = list(cv_context) if cv_context is not None else build_cv_context(shots)
    aggregated_model: dict[int, list[int]] = {10: [], 20: [], 42: []}
    aggregated_baseline: dict[int, list[int]] = {10: [], 20: [], 42: []}
    directions_by_fold: list[dict[str, str]] = []

    for evidence, test in context:
        rule = fit_rule(evidence, config)
        directions_by_fold.append({
            key: ("HIGH" if weight > 0 else "LOW")
            for key, weight in zip(rule.feature_keys, rule.signed_weights.tolist())
        })

        for shot in test:
            scores = score_rule(shot, rule)
            for radius in (10, 20, 42):
                baseline_rank = _baseline_rank_for_radius(shot, radius)
                if baseline_rank is None:
                    continue
                rank = _rank_for_radius(shot, scores, radius)
                aggregated_baseline[radius].append(int(baseline_rank))
                aggregated_model[radius].append(
                    int(rank if rank is not None else shot.size + 1)
                )

    result = {
        "model": {
            str(radius): _metrics(aggregated_model[radius])
            for radius in (10, 20, 42)
        },
        "baseline": {
            str(radius): _metrics(aggregated_baseline[radius])
            for radius in (10, 20, 42)
        },
        "directions_by_fold": directions_by_fold,
    }
    result["top1_gate_pass"] = bool(
        float(result["model"]["20"]["top1_pct"])
        >= float(result["baseline"]["20"]["top1_pct"])
    )
    result["objective"] = round(_objective(result), 6)
    return result


def single_feature_sweep(
    development: Sequence[PreparedShot],
    *,
    cv_context: Sequence[tuple[FeatureEvidence, list[PreparedShot]]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for key in PHYSICAL_FEATURE_KEYS:
        config = RuleConfig(feature_keys=(key,), weights=(1.0,))
        cv = cross_validate_config(
            development,
            config,
            cv_context=cv_context,
        )
        rows.append({
            "feature": key,
            "family": FEATURE_FAMILIES.get(key, key),
            "cv": cv,
            "objective": float(cv["objective"]),
            "top1_gate_pass": bool(cv["top1_gate_pass"]),
        })

    rows.sort(
        key=lambda row: (
            bool(row["top1_gate_pass"]),
            float(row["objective"]),
        ),
        reverse=True,
    )
    return rows


def _family_diverse(
    keys: Sequence[str],
) -> bool:
    families = [
        FEATURE_FAMILIES.get(key, key)
        for key in keys
    ]
    return len(set(families)) == len(families)


def _correlation_ok(
    keys: Sequence[str],
    evidence: FeatureEvidence,
    *,
    limit: float = 0.94,
) -> bool:
    indices = [FEATURE_INDEX[key] for key in keys]
    for i, left in enumerate(indices):
        for right in indices[i + 1:]:
            if abs(float(evidence.correlation[left, right])) >= float(limit):
                return False
    return True


def _weight_patterns(count: int) -> list[tuple[float, ...]]:
    if count == 1:
        return [(1.0,)]
    if count == 2:
        return [
            (1.0, 1.0),
            (2.0, 1.0),
            (1.0, 2.0),
            (3.0, 1.0),
            (1.0, 3.0),
        ]
    if count == 3:
        return [
            (1.0, 1.0, 1.0),
            (2.0, 1.0, 1.0),
            (1.0, 2.0, 1.0),
            (1.0, 1.0, 2.0),
            (3.0, 1.0, 1.0),
            (1.0, 3.0, 1.0),
            (1.0, 1.0, 3.0),
            (2.0, 2.0, 1.0),
            (2.0, 1.0, 2.0),
            (1.0, 2.0, 2.0),
        ]
    raise ValueError("V2.11 searches only one, two and three-feature rules.")


def generate_search_configs(
    development: Sequence[PreparedShot],
    *,
    top_features: int = 12,
    singles: Sequence[dict[str, Any]] | None = None,
    evidence: FeatureEvidence | None = None,
) -> tuple[list[RuleConfig], list[dict[str, Any]], FeatureEvidence]:
    singles = list(singles) if singles is not None else single_feature_sweep(development)
    evidence = evidence if evidence is not None else fit_feature_evidence(development)

    # Prefer features that pass the Top-1 gate alone, then strongest remaining
    # listwise performers. This list is intentionally small to reduce search
    # overfitting on a 100-shot dataset.
    feature_order: list[str] = []
    for row in singles:
        key = str(row["feature"])
        if key not in feature_order:
            feature_order.append(key)
        if len(feature_order) >= max(4, int(top_features)):
            break

    configs: list[RuleConfig] = []

    # Every single feature is always tested.
    for key in feature_order:
        configs.append(RuleConfig((key,), (1.0,)))

    for count in (2, 3):
        for keys in itertools.combinations(feature_order, count):
            if not _family_diverse(keys):
                continue
            if not _correlation_ok(keys, evidence):
                continue
            for weights in _weight_patterns(count):
                configs.append(
                    RuleConfig(
                        feature_keys=tuple(keys),
                        weights=tuple(float(value) for value in weights),
                    )
                )

    return configs, singles, evidence


def search_rules(
    development: Sequence[PreparedShot],
    *,
    top_features: int = 12,
) -> dict[str, Any]:
    cv_context = build_cv_context(development)
    evidence = fit_feature_evidence(development)
    singles = single_feature_sweep(
        development,
        cv_context=cv_context,
    )
    configs, singles, evidence = generate_search_configs(
        development,
        top_features=top_features,
        singles=singles,
        evidence=evidence,
    )

    evaluated: list[dict[str, Any]] = []
    for config in configs:
        cv = cross_validate_config(
            development,
            config,
            cv_context=cv_context,
        )
        evaluated.append({
            "config": config,
            "cv": cv,
            "objective": float(cv["objective"]),
            "top1_gate_pass": bool(cv["top1_gate_pass"]),
        })

    evaluated.sort(
        key=lambda item: (
            bool(item["top1_gate_pass"]),
            float(item["objective"]),
        ),
        reverse=True,
    )

    passing = [
        item
        for item in evaluated
        if bool(item["top1_gate_pass"])
    ]

    return {
        "configs_evaluated": len(evaluated),
        "single_features": singles,
        "evidence": evidence,
        "all": evaluated,
        "passing": passing,
        "best": passing[0] if passing else None,
    }


def feature_evidence_table(
    evidence: FeatureEvidence,
) -> list[dict[str, Any]]:
    rows = []

    for index, key in enumerate(PHYSICAL_FEATURE_KEYS):
        rows.append({
            "feature": key,
            "family": FEATURE_FAMILIES.get(key, key),
            "direction": "HIGH" if float(evidence.direction[index]) > 0 else "LOW",
            "win_rate": round(float(evidence.win_rate[index]), 5),
            "auc_like": round(float(evidence.auc_like[index]), 5),
            "strength": round(float(evidence.strength[index]), 5),
            "comparisons": round(float(evidence.comparisons[index]), 2),
        })

    rows.sort(
        key=lambda row: (
            float(row["strength"]),
            float(row["comparisons"]),
        ),
        reverse=True,
    )
    return rows


def recommendation(
    *,
    total_shots: int,
    development_cv: dict[str, Any] | None,
    confirmation: dict[str, Any] | None,
) -> dict[str, Any]:
    if development_cv is None or confirmation is None:
        return {
            "enough_data": False,
            "development_gate": False,
            "confirmation_gate": False,
            "shadow_ready": False,
            "reason": "no model passed the hard development Top-1<=20 gate",
        }

    dev20 = development_cv["model"]["20"]
    devbase20 = development_cv["baseline"]["20"]
    conf20 = confirmation["model"]["20"]
    confbase20 = confirmation["baseline"]["20"]
    conf42 = confirmation["model"]["42"]
    confbase42 = confirmation["baseline"]["42"]

    dev_top1_gain = float(dev20["top1_pct"]) - float(devbase20["top1_pct"])
    dev_top3_gain = float(dev20["top3_pct"]) - float(devbase20["top3_pct"])
    conf_top1_gain = float(conf20["top1_pct"]) - float(confbase20["top1_pct"])
    conf_top3_gain = float(conf20["top3_pct"]) - float(confbase20["top3_pct"])
    conf42_top3_gain = float(conf42["top3_pct"]) - float(confbase42["top3_pct"])

    enough_data = (
        total_shots >= 350
        and int(conf20.get("covered", 0)) >= 20
        and int(conf42.get("covered", 0)) >= 50
    )
    development_gate = dev_top1_gain >= 0.0
    # Confirmation is deliberately conservative. With only ~20 confirmation
    # shots it is mostly a warning signal, not proof of quality.
    confirmation_gate = (
        conf_top1_gain >= 0.0
        and conf_top3_gain >= -2.0
        and conf42_top3_gain >= -3.0
    )

    return {
        "enough_data": bool(enough_data),
        "development_gate": bool(development_gate),
        "confirmation_gate": bool(confirmation_gate),
        "shadow_ready": bool(development_gate and confirmation_gate),
        "recommended_for_future_authority": bool(
            enough_data and development_gate and confirmation_gate
            and dev_top1_gain >= 2.0
            and conf_top1_gain >= 0.0
        ),
        "development_top1_gain_20_pp": round(dev_top1_gain, 3),
        "development_top3_gain_20_pp": round(dev_top3_gain, 3),
        "confirmation_top1_gain_20_pp": round(conf_top1_gain, 3),
        "confirmation_top3_gain_20_pp": round(conf_top3_gain, 3),
        "confirmation_top3_gain_42_pp": round(conf42_top3_gain, 3),
        "note": "V2.11/V9 is shadow-only regardless of recommendation.",
    }


__all__ = [
    "FeatureEvidence",
    "FittedRule",
    "PHYSICAL_FEATURE_KEYS",
    "PreparedShot",
    "RuleConfig",
    "broad_negative_indices",
    "build_cv_context",
    "cross_validate_config",
    "evaluate_rule",
    "feature_evidence_table",
    "fit_feature_evidence",
    "fit_rule",
    "generate_search_configs",
    "prepare_rows",
    "recommendation",
    "search_rules",
    "single_feature_sweep",
    "stable_confirmation_split",
]
