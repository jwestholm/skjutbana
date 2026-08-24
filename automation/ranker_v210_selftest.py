from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from src.engine.ai.ranker_v7 import FEATURE_KEYS
from src.engine.ai.ranker_v8 import RankerV8ShadowModel
from src.engine.ai.ranker_v8_optimizer import (
    SearchConfig,
    evaluate_model,
    fit_feature_profile,
    fit_model_from_profile,
    prepare_rows,
    search_configs,
    stable_confirmation_split,
)


def _blank_features() -> dict[str, float]:
    return {key: 0.5 for key in FEATURE_KEYS}


def _synthetic_rows(shots: int = 120, candidates_per_shot: int = 55) -> list[dict]:
    rng = random.Random(210)
    rows = []
    for shot in range(1, shots + 1):
        positive_index = rng.randrange(candidates_per_shot)
        candidates = []
        baseline_order = []
        for index in range(candidates_per_shot):
            features = _blank_features()
            is_positive = index == positive_index
            if is_positive:
                # Mimic the real V2.9 finding: the actual hole is relatively
                # weak/unsupported but compact.
                features.update(
                    {
                        "support_score": rng.uniform(0.12, 0.36),
                        "rel_support_score": rng.uniform(0.05, 0.30),
                        "support_x_diversity": rng.uniform(0.04, 0.25),
                        "support_x_current": rng.uniform(0.08, 0.32),
                        "source_diversity": rng.uniform(0.20, 0.50),
                        "zscore": rng.uniform(0.18, 0.42),
                        "rel_zscore": rng.uniform(0.10, 0.38),
                        "signal_score": rng.uniform(0.18, 0.42),
                        "member_count": rng.uniform(0.10, 0.35),
                        "compactness": rng.uniform(0.82, 1.00),
                        "baseline_score": rng.uniform(0.30, 0.55),
                    }
                )
                distance = rng.uniform(4.0, 18.0)
            else:
                features.update(
                    {
                        "support_score": rng.uniform(0.62, 0.98),
                        "rel_support_score": rng.uniform(0.68, 1.00),
                        "support_x_diversity": rng.uniform(0.55, 0.98),
                        "support_x_current": rng.uniform(0.58, 0.98),
                        "source_diversity": rng.uniform(0.60, 1.00),
                        "zscore": rng.uniform(0.58, 1.00),
                        "rel_zscore": rng.uniform(0.60, 1.00),
                        "signal_score": rng.uniform(0.55, 0.98),
                        "member_count": rng.uniform(0.55, 0.98),
                        "compactness": rng.uniform(0.20, 0.72),
                        "baseline_score": rng.uniform(0.60, 1.00),
                    }
                )
                distance = rng.uniform(65.0, 500.0)
            candidate = {
                "distance_gt_px": distance,
                "membership": {"hypothesis_pool": True, "core": False},
                "features": features,
                "ranks": {},
            }
            candidates.append(candidate)
            baseline_order.append((features["baseline_score"], index))

        for rank, (_score, index) in enumerate(
            sorted(baseline_order, reverse=True), start=1
        ):
            candidates[index]["ranks"]["baseline"] = rank

        rows.append(
            {
                "session_id": "selftest",
                "sequence": shot,
                "candidates": candidates,
            }
        )
    return rows


def main() -> None:
    print("=" * 76)
    print("V2.10 OFFLINE RANK OPTIMIZER SELFTEST")
    print("=" * 76)

    shots = prepare_rows(_synthetic_rows())
    development, confirmation = stable_confirmation_split(shots)
    assert len(development) > 70
    assert len(confirmation) >= 8
    print("PASS: deterministic development/confirmation split")

    profile = fit_feature_profile(development)
    configs = [
        SearchConfig(3, 0.15, 0.86, 0.70, 0.75, 0.00),
        SearchConfig(5, 0.15, 0.94, 1.00, 0.75, 0.00),
        SearchConfig(8, 0.25, 0.94, 1.00, 1.50, 0.00),
        SearchConfig(5, 0.25, 0.86, 1.45, 1.50, 0.08),
    ]
    search = search_configs(development, configs=configs)
    assert search
    best = search[0]
    model = fit_model_from_profile(profile, best["config"])
    result = evaluate_model(confirmation, model)
    baseline_top1 = result["baseline"]["20"]["top1_pct"]
    model_top1 = result["model"]["20"]["top1_pct"]
    assert model_top1 >= baseline_top1 + 50.0, (baseline_top1, model_top1)
    print(
        "PASS: monotonic search beats deliberately wrong baseline on held-out shots "
        f"({baseline_top1:.1f}% -> {model_top1:.1f}% Top-1)"
    )

    # Model JSON load sanity. The runtime shadow model uses the same signed
    # monotonic weight representation as the offline optimizer.
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "ranker_v8.json"
        payload = {
            "schema_version": "2.10",
            "model_version": 8,
            "model_type": "monotonic_percentile_rank_ensemble",
            "shadow_only": True,
            "feature_keys": list(model.feature_keys),
            "weights": {
                key: float(weight)
                for key, weight in zip(model.feature_keys, model.weights.tolist())
            },
            "evidence_power": model.evidence_power,
            "baseline_blend": model.baseline_blend,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        runtime_model = RankerV8ShadowModel(path)
        assert runtime_model.loaded
    print("PASS: V8 shadow model persistence")
    print("=" * 76)
    print("All V2.10 selftests passed.")


if __name__ == "__main__":
    main()
