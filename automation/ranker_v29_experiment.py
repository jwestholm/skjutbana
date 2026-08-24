from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.engine.ai.ranker_v7 import FEATURE_KEYS
from src.engine.ai.ranking_dataset_v29 import DATA_ROOT, load_session


MODEL_PATH = Path("content/ai/ranker_v7_offline.json")
EXPERIMENT_ROOT = DATA_ROOT / "experiments"


FEATURE_VARIANTS: dict[str, list[str]] = {
    "all_features": list(FEATURE_KEYS),
    "no_baseline_score": [
        key for key in FEATURE_KEYS
        if key not in {"baseline_score", "rel_baseline_score"}
    ],
    "primitive_evidence": [
        key for key in FEATURE_KEYS
        if key not in {
            "baseline_score",
            "support_score",
            "signal_score",
            "rel_baseline_score",
            "rel_support_score",
            "rel_signal_score",
            "reason_core",
            "reason_core_baseline",
            "reason_baseline_fill",
        }
    ],
    "support_focused": [
        key for key in FEATURE_KEYS
        if key in {
            "support_score",
            "member_count",
            "compactness",
            "source_diversity",
            "current_fraction",
            "carried_fraction",
            "hits_max",
            "hits_mean",
            "v1_fraction",
            "v2_fraction",
            "tile_fraction",
            "agreement_fraction",
            "not_existed_before",
            "age_good",
            "spread_good",
            "single_member",
            "core_member",
            "rel_support_score",
            "rel_member_count",
            "rel_compactness",
            "rel_source_diversity",
            "rel_current_fraction",
            "rel_hits_max",
            "rel_spread_good",
            "support_x_current",
            "support_x_diversity",
            "tile_x_current",
            "agreement_x_current",
            "fresh_single",
        }
    ],
    "signal_focused": [
        key for key in FEATURE_KEYS
        if key in {
            "signal_score",
            "patch_prior_max",
            "patch_prior_median",
            "zscore",
            "absdiff",
            "dog",
            "saliency",
            "persistence",
            "current_fraction",
            "not_existed_before",
            "spread_good",
            "rel_signal_score",
            "rel_patch_prior_max",
            "rel_zscore",
            "rel_absdiff",
            "rel_dog",
            "rel_saliency",
            "rel_current_fraction",
            "rel_spread_good",
            "signal_x_patch",
            "low_signal_current",
        }
    ],
}


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


def _positive_weight(distance: float) -> float:
    if distance <= 12.0:
        return 1.0
    if distance <= 20.0:
        return 0.68
    if distance <= 42.0:
        return 0.24
    return 0.0


def _vector(candidate: dict[str, Any], feature_keys: Sequence[str]) -> np.ndarray:
    features = candidate.get("features")
    if not isinstance(features, dict):
        features = {}
    return np.asarray(
        [_finite(features.get(key)) for key in feature_keys],
        dtype=np.float64,
    )


def _baseline_rank(candidate: dict[str, Any]) -> int:
    ranks = candidate.get("ranks")
    if isinstance(ranks, dict):
        value = ranks.get("baseline")
        try:
            if value is not None:
                return int(value)
        except Exception:
            pass
    return 10**9


def build_pair_matrix(
    rows: Sequence[dict[str, Any]],
    feature_keys: Sequence[str],
    *,
    hard_negatives: int,
    negative_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    differences: list[np.ndarray] = []
    sample_weights: list[float] = []
    stats = {
        "shots": len(rows),
        "positive_shots": 0,
        "strict": 0,
        "soft20": 0,
        "weak42": 0,
        "skipped_no_positive": 0,
        "skipped_no_negative": 0,
        "pairs": 0,
    }

    for row in rows:
        pool = _pool(row)
        if not pool:
            stats["skipped_no_positive"] += 1
            continue

        positive = min(pool, key=_distance)
        distance = _distance(positive)
        label_weight = _positive_weight(distance)
        if label_weight <= 0.0:
            stats["skipped_no_positive"] += 1
            continue

        if distance <= 12.0:
            stats["strict"] += 1
        elif distance <= 20.0:
            stats["soft20"] += 1
        else:
            stats["weak42"] += 1

        negatives = [
            candidate
            for candidate in pool
            if candidate is not positive and _distance(candidate) >= negative_radius_px
        ]
        negatives.sort(
            key=lambda candidate: (
                _baseline_rank(candidate),
                -_finite(candidate.get("features", {}).get("baseline_score")),
            )
        )
        negatives = negatives[: max(1, int(hard_negatives))]
        if not negatives:
            stats["skipped_no_negative"] += 1
            continue

        pos_vector = _vector(positive, feature_keys)
        for index, negative in enumerate(negatives):
            # Highest baseline-ranked wrong answers are the hardest and receive
            # slightly more weight.
            hardness = 1.0 / math.sqrt(1.0 + 0.10 * index)
            differences.append(pos_vector - _vector(negative, feature_keys))
            sample_weights.append(label_weight * hardness)

        stats["positive_shots"] += 1

    stats["pairs"] = len(differences)
    if not differences:
        return (
            np.zeros((0, len(feature_keys)), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            stats,
        )
    return (
        np.vstack(differences),
        np.asarray(sample_weights, dtype=np.float64),
        stats,
    )


def train_pairwise_linear(
    rows: Sequence[dict[str, Any]],
    feature_keys: Sequence[str],
    *,
    hard_negatives: int = 30,
    negative_radius_px: float = 55.0,
    epochs: int = 90,
    learning_rate: float = 0.42,
    l2: float = 0.012,
) -> tuple[np.ndarray, dict[str, Any]]:
    x, sample_weights, pair_stats = build_pair_matrix(
        rows,
        feature_keys,
        hard_negatives=hard_negatives,
        negative_radius_px=negative_radius_px,
    )
    weights = np.zeros((len(feature_keys),), dtype=np.float64)
    if x.shape[0] == 0:
        return weights, {**pair_stats, "loss": None}

    weight_total = max(1e-9, float(np.sum(sample_weights)))
    best_weights = weights.copy()
    best_loss = float("inf")

    for epoch in range(max(1, int(epochs))):
        margin = np.clip(x @ weights, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-margin))
        residual = (1.0 - probability) * sample_weights
        gradient = (x.T @ residual) / weight_total
        gradient -= float(l2) * weights

        # Gentle learning-rate decay is more stable across 100 vs 1000 shots.
        lr = float(learning_rate) / math.sqrt(1.0 + epoch / 30.0)
        weights += lr * gradient
        weights = np.clip(weights, -8.0, 8.0)

        loss = float(
            np.sum(
                sample_weights * np.log1p(np.exp(-margin))
            )
            / weight_total
            + 0.5 * float(l2) * float(weights @ weights)
        )
        if loss < best_loss:
            best_loss = loss
            best_weights = weights.copy()

    return best_weights, {
        **pair_stats,
        "loss": round(best_loss, 7),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "l2": float(l2),
    }


def _rank_for_radius(
    candidates: Sequence[dict[str, Any]],
    scores: Sequence[float],
    radius: float,
) -> int | None:
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            float(scores[index]),
            _finite(candidates[index].get("features", {}).get("baseline_score")),
        ),
        reverse=True,
    )
    for rank, index in enumerate(order, start=1):
        if _distance(candidates[index]) <= radius:
            return rank
    return None


def _baseline_rank_for_radius(
    candidates: Sequence[dict[str, Any]],
    radius: float,
) -> int | None:
    ordered = sorted(
        candidates,
        key=lambda candidate: _baseline_rank(candidate),
    )
    for rank, candidate in enumerate(ordered, start=1):
        if _distance(candidate) <= radius:
            return rank
    return None


def evaluate_model(
    rows: Sequence[dict[str, Any]],
    feature_keys: Sequence[str],
    weights: np.ndarray,
) -> dict[str, Any]:
    model_ranks: dict[int, list[int]] = {10: [], 20: [], 42: []}
    baseline_ranks: dict[int, list[int]] = {10: [], 20: [], 42: []}

    for row in rows:
        pool = _pool(row)
        if not pool:
            continue
        matrix = np.vstack([_vector(candidate, feature_keys) for candidate in pool])
        scores = matrix @ weights
        for radius in (10, 20, 42):
            model_rank = _rank_for_radius(pool, scores, radius)
            baseline_rank = _baseline_rank_for_radius(pool, radius)
            if baseline_rank is not None:
                baseline_ranks[radius].append(baseline_rank)
                # Model is compared only on shots where the answer exists in
                # exactly the same pool.
                if model_rank is not None:
                    model_ranks[radius].append(model_rank)
                else:
                    model_ranks[radius].append(len(pool) + 1)

    def metrics(ranks: list[int]) -> dict[str, Any]:
        if not ranks:
            return {
                "covered": 0,
                "top1_pct": 0.0,
                "top3_pct": 0.0,
                "top5_pct": 0.0,
                "median_rank": None,
                "mrr": None,
            }
        return {
            "covered": len(ranks),
            "top1_pct": round(100.0 * sum(rank == 1 for rank in ranks) / len(ranks), 3),
            "top3_pct": round(100.0 * sum(rank <= 3 for rank in ranks) / len(ranks), 3),
            "top5_pct": round(100.0 * sum(rank <= 5 for rank in ranks) / len(ranks), 3),
            "median_rank": float(np.median(np.asarray(ranks, dtype=np.float64))),
            "mrr": round(float(np.mean([1.0 / rank for rank in ranks])), 6),
        }

    return {
        "model": {str(radius): metrics(model_ranks[radius]) for radius in (10, 20, 42)},
        "baseline": {str(radius): metrics(baseline_ranks[radius]) for radius in (10, 20, 42)},
    }


def _fold_count(shots: int) -> int:
    if shots >= 80:
        return 5
    if shots >= 40:
        return 4
    if shots >= 20:
        return 3
    return 2


def cross_validate(
    rows: Sequence[dict[str, Any]],
    feature_keys: Sequence[str],
    *,
    epochs: int,
) -> dict[str, Any]:
    folds = _fold_count(len(rows))
    aggregated_model: dict[int, list[int]] = {10: [], 20: [], 42: []}
    aggregated_baseline: dict[int, list[int]] = {10: [], 20: [], 42: []}
    train_stats: list[dict[str, Any]] = []
    fold_weights: list[np.ndarray] = []

    for fold in range(folds):
        train_rows = [
            row
            for index, row in enumerate(rows)
            if index % folds != fold
        ]
        test_rows = [
            row
            for index, row in enumerate(rows)
            if index % folds == fold
        ]
        weights, stats = train_pairwise_linear(
            train_rows,
            feature_keys,
            epochs=epochs,
        )
        train_stats.append(stats)
        fold_weights.append(weights)

        for row in test_rows:
            pool = _pool(row)
            if not pool:
                continue
            matrix = np.vstack([_vector(candidate, feature_keys) for candidate in pool])
            scores = matrix @ weights
            for radius in (10, 20, 42):
                baseline_rank = _baseline_rank_for_radius(pool, radius)
                if baseline_rank is None:
                    continue
                model_rank = _rank_for_radius(pool, scores, radius)
                aggregated_baseline[radius].append(baseline_rank)
                aggregated_model[radius].append(
                    model_rank if model_rank is not None else len(pool) + 1
                )

    def metrics(ranks: list[int]) -> dict[str, Any]:
        if not ranks:
            return {
                "covered": 0,
                "top1_pct": 0.0,
                "top3_pct": 0.0,
                "top5_pct": 0.0,
                "median_rank": None,
                "mrr": None,
            }
        return {
            "covered": len(ranks),
            "top1_pct": round(100.0 * sum(rank == 1 for rank in ranks) / len(ranks), 3),
            "top3_pct": round(100.0 * sum(rank <= 3 for rank in ranks) / len(ranks), 3),
            "top5_pct": round(100.0 * sum(rank <= 5 for rank in ranks) / len(ranks), 3),
            "median_rank": float(np.median(np.asarray(ranks, dtype=np.float64))),
            "mrr": round(float(np.mean([1.0 / rank for rank in ranks])), 6),
        }

    model_metrics = {
        str(radius): metrics(aggregated_model[radius])
        for radius in (10, 20, 42)
    }
    baseline_metrics = {
        str(radius): metrics(aggregated_baseline[radius])
        for radius in (10, 20, 42)
    }

    return {
        "folds": folds,
        "model": model_metrics,
        "baseline": baseline_metrics,
        "training": train_stats,
        "mean_fold_weights": (
            np.mean(np.vstack(fold_weights), axis=0).tolist()
            if fold_weights
            else [0.0] * len(feature_keys)
        ),
    }


def _objective(cv: dict[str, Any]) -> float:
    model20 = cv["model"]["20"]
    model42 = cv["model"]["42"]
    return (
        3.0 * float(model20["top1_pct"])
        + 1.4 * float(model20["top3_pct"])
        + 0.55 * float(model20["top5_pct"])
        + 0.8 * float(model42["top1_pct"])
        + 0.35 * float(model42["top3_pct"])
        + 120.0 * float(model20["mrr"] or 0.0)
        + 45.0 * float(model42["mrr"] or 0.0)
        - 0.025 * float(model20["median_rank"] or 999.0)
        - 0.010 * float(model42["median_rank"] or 999.0)
    )


def _recommendation(cv: dict[str, Any], shots: int) -> dict[str, Any]:
    model20 = cv["model"]["20"]
    base20 = cv["baseline"]["20"]
    model42 = cv["model"]["42"]
    base42 = cv["baseline"]["42"]

    enough_data = (
        shots >= 300
        and int(model20.get("covered", 0)) >= 70
        and int(model42.get("covered", 0)) >= 150
    )
    top1_gain20 = float(model20["top1_pct"]) - float(base20["top1_pct"])
    top3_gain20 = float(model20["top3_pct"]) - float(base20["top3_pct"])
    top3_gain42 = float(model42["top3_pct"]) - float(base42["top3_pct"])
    median20 = model20.get("median_rank")
    base_median20 = base20.get("median_rank")
    median_ratio = (
        float(median20) / max(1.0, float(base_median20))
        if median20 is not None and base_median20 is not None
        else None
    )

    passes_quality = (
        (top1_gain20 >= 3.0 or top3_gain20 >= 5.0)
        and top3_gain42 >= -1.0
        and median_ratio is not None
        and median_ratio <= 0.88
    )
    return {
        "enough_data": bool(enough_data),
        "passes_quality": bool(passes_quality),
        "recommended_for_future_authority": bool(enough_data and passes_quality),
        "top1_gain_20_pp": round(top1_gain20, 3),
        "top3_gain_20_pp": round(top3_gain20, 3),
        "top3_gain_42_pp": round(top3_gain42, 3),
        "median_rank_ratio_20": round(median_ratio, 4) if median_ratio is not None else None,
        "note": (
            "V2.9 remains shadow-only regardless of this flag. "
            "The flag is only a future-development recommendation."
        ),
    }


def print_variant(name: str, result: dict[str, Any]) -> None:
    print(f"\n{name}")
    for radius in (20, 42):
        model = result["model"][str(radius)]
        base = result["baseline"][str(radius)]
        print(
            f"  <= {radius}px "
            f"MODEL top1={model['top1_pct']:6.2f}% top3={model['top3_pct']:6.2f}% "
            f"med={str(model['median_rank']):>6s} MRR={model['mrr']} | "
            f"BASE top1={base['top1_pct']:6.2f}% top3={base['top3_pct']:6.2f}% "
            f"med={str(base['median_rank']):>6s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline V2.9 ranker laboratory using captured hypothesis datasets"
    )
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--no-save-model", action="store_true")
    args = parser.parse_args()

    rows, session = load_session(args.session)
    if not rows:
        print("No V2.9 ranking dataset found.")
        raise SystemExit(1)

    print("=" * 78)
    print("V2.9 OFFLINE RANKER LAB")
    print("=" * 78)
    print(f"Session: {session}")
    print(f"Shots: {len(rows)}")
    print(
        "No camera input is used here. Every model is evaluated on held-out "
        "shots from the saved dataset."
    )

    variants: dict[str, Any] = {}
    for name, keys in FEATURE_VARIANTS.items():
        cv = cross_validate(rows, keys, epochs=args.epochs)
        cv["feature_keys"] = list(keys)
        cv["objective"] = round(_objective(cv), 6)
        variants[name] = cv
        print_variant(name, cv)

    best_name = max(
        variants,
        key=lambda name: float(variants[name].get("objective", -1e18)),
    )
    best_cv = variants[best_name]
    best_keys = list(best_cv["feature_keys"])

    final_weights, final_training = train_pairwise_linear(
        rows,
        best_keys,
        epochs=max(args.epochs, 110),
    )
    recommendation = _recommendation(best_cv, len(rows))

    strongest = sorted(
        zip(best_keys, final_weights.tolist()),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:15]

    report = {
        "schema_version": "2.9",
        "created_at": time.time(),
        "session": session,
        "shots": len(rows),
        "variants": variants,
        "best_variant": best_name,
        "best_cv": best_cv,
        "final_training": final_training,
        "recommendation": recommendation,
        "strongest_weights": [
            [key, round(float(value), 7)]
            for key, value in strongest
        ],
    }

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = (
        EXPERIMENT_ROOT
        / f"{time.strftime('%Y%m%d_%H%M%S')}_{session or 'session'}.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.no_save_model:
        payload = {
            "schema_version": "2.9",
            "model_version": 7,
            "model_type": "pairwise_linear_cross_validated",
            "shadow_only": True,
            "trained_session": session,
            "trained_shots": len(rows),
            "created_at": time.time(),
            "best_variant": best_name,
            "feature_keys": best_keys,
            "weights": {
                key: float(value)
                for key, value in zip(best_keys, final_weights.tolist())
            },
            "cv": best_cv,
            "training": final_training,
            "recommendation": recommendation,
        }
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print()
    print("=" * 78)
    print(f"BEST VARIANT: {best_name}")
    print_variant("held-out cross-validation", best_cv)
    print()
    print("STRONGEST FINAL WEIGHTS:")
    for key, value in strongest:
        direction = "+" if value >= 0 else "-"
        print(f"  {key:26s} {direction}{abs(float(value)):.5f}")
    print()
    print("RECOMMENDATION:")
    for key, value in recommendation.items():
        print(f"  {key}: {value}")
    print()
    print(f"Experiment report: {report_path}")
    if not args.no_save_model:
        print(f"Shadow model:      {MODEL_PATH}")
    print(
        "V2.9 NEVER gives V7 authority. The saved model is only available for "
        "shadow comparison on a later run."
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
