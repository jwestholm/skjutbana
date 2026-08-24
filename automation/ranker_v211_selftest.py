from __future__ import annotations

import math
import tempfile
from pathlib import Path

from src.engine.ai.ranker_v9 import (
    FEATURE_FAMILIES,
    PHYSICAL_FEATURE_KEYS,
)
from src.engine.ai.ranker_v9_optimizer import (
    broad_negative_indices,
    evaluate_rule,
    fit_feature_evidence,
    fit_rule,
    prepare_rows,
    search_rules,
    stable_confirmation_split,
)


FORBIDDEN_EXACT = {
    "baseline_score",
    "rel_baseline_score",
    "core_member",
}


def _synthetic_rows(
    *,
    shots: int = 80,
    candidates_per_shot: int = 36,
) -> list[dict]:
    rows = []

    for shot_index in range(shots):
        candidates = []

        # Correct hole-like hypothesis: compact, weak, fresh, low support.
        gt_features = {key: 0.0 for key in PHYSICAL_FEATURE_KEYS}
        gt_features.update(
            {
                "support_score": 0.16 + 0.01 * (shot_index % 3),
                "signal_score": 0.18,
                "member_count": 0.10,
                "compactness": 0.92,
                "source_diversity": 0.22,
                "current_fraction": 0.72,
                "carried_fraction": 0.05,
                "hits_max": 0.12,
                "hits_mean": 0.10,
                "tile_fraction": 0.56,
                "patch_prior_max": 0.42,
                "zscore": 0.20,
                "absdiff": 0.18,
                "dog": 0.28,
                "saliency": 0.20,
                "persistence": 0.12,
                "not_existed_before": 0.92,
                "age_good": 0.90,
                "spread_good": 0.95,
                "single_member": 1.0,
                "low_signal_current": 0.60,
            }
        )
        candidates.append(
            {
                "camera_x": 100.0,
                "camera_y": 100.0,
                "distance_gt_px": 4.0,
                "membership": {"hypothesis_pool": True},
                "ranks": {"baseline": candidates_per_shot},
                "features": gt_features,
            }
        )

        # False projector artifacts. Baseline intentionally likes these.
        for index in range(1, candidates_per_shot):
            phase = (shot_index * 17 + index * 13) % 101
            jitter = phase / 1000.0

            features = {key: 0.0 for key in PHYSICAL_FEATURE_KEYS}
            features.update(
                {
                    "support_score": 0.72 + 0.20 * ((index % 5) / 4.0),
                    "signal_score": 0.68 + 0.22 * ((index % 7) / 6.0),
                    "member_count": 0.58 + 0.30 * ((index % 4) / 3.0),
                    "compactness": 0.18 + 0.42 * ((index % 8) / 7.0),
                    "source_diversity": 0.62 + 0.30 * ((index % 3) / 2.0),
                    "current_fraction": 0.58 + jitter,
                    "carried_fraction": 0.35,
                    "hits_max": 0.62,
                    "hits_mean": 0.60,
                    "tile_fraction": 0.30,
                    "patch_prior_max": 0.58,
                    "zscore": 0.72,
                    "absdiff": 0.75,
                    "dog": 0.54,
                    "saliency": 0.76,
                    "persistence": 0.70,
                    "not_existed_before": 0.28,
                    "age_good": 0.42,
                    "spread_good": 0.48,
                    "single_member": 0.0,
                    "low_signal_current": 0.14,
                }
            )

            candidates.append(
                {
                    "camera_x": 200.0 + 11.0 * index,
                    "camera_y": 250.0 + 7.0 * index,
                    "distance_gt_px": 80.0 + index,
                    "membership": {"hypothesis_pool": True},
                    "ranks": {"baseline": index},
                    "features": features,
                }
            )

        rows.append(
            {
                "schema_version": "selftest",
                "session_id": "v211_selftest",
                "sequence": shot_index + 1,
                "candidates": candidates,
            }
        )

    return rows


def test_policy_features_forbidden() -> None:
    assert not any(key in PHYSICAL_FEATURE_KEYS for key in FORBIDDEN_EXACT)
    assert not any(key.startswith("reason_") for key in PHYSICAL_FEATURE_KEYS)
    assert not any(key.startswith("rel_") for key in PHYSICAL_FEATURE_KEYS)
    print("PASS: policy/ranking-leakage features are forbidden")


def test_broad_negative_sampling() -> None:
    shots = prepare_rows(_synthetic_rows(shots=1, candidates_per_shot=90))
    shot = shots[0]
    negatives = broad_negative_indices(
        shot,
        positive_index=0,
        max_negatives=40,
    )
    assert len(negatives) >= 20
    # Baseline hard negative must be represented.
    assert 1 in negatives
    # Sampling must not collapse to merely the first baseline ranks.
    assert max(negatives) > 40
    print("PASS: broad negative sampler covers multiple false-candidate phenotypes")


def test_split_is_shot_level() -> None:
    shots = prepare_rows(_synthetic_rows(shots=50))
    development, confirmation = stable_confirmation_split(shots)
    dev_ids = {shot.row_id for shot in development}
    conf_ids = {shot.row_id for shot in confirmation}
    assert dev_ids
    assert conf_ids
    assert dev_ids.isdisjoint(conf_ids)
    assert len(dev_ids | conf_ids) == len(shots)
    print("PASS: development/confirmation split has no shot leakage")


def test_search_beats_bad_baseline() -> None:
    shots = prepare_rows(_synthetic_rows(shots=80))
    development, confirmation = stable_confirmation_split(shots)

    result = search_rules(development, top_features=7)
    best = result.get("best")
    assert best is not None, "expected at least one rule to pass Top-1 gate"

    cv = best["cv"]
    assert cv["model"]["20"]["top1_pct"] >= cv["baseline"]["20"]["top1_pct"]
    assert cv["model"]["20"]["top1_pct"] >= 80.0

    evidence = fit_feature_evidence(development)
    rule = fit_rule(evidence, best["config"])
    held_out = evaluate_rule(confirmation, rule)

    assert held_out["model"]["20"]["top1_pct"] >= 80.0
    assert held_out["model"]["20"]["top1_pct"] > held_out["baseline"]["20"]["top1_pct"]
    assert all(
        key in PHYSICAL_FEATURE_KEYS
        for key in rule.feature_keys
    )
    print("PASS: physical/listwise rule beats artifact-loving baseline on held-out shots")


def main() -> None:
    print("=" * 76)
    print("V2.11 SELFTEST")
    print("=" * 76)
    test_policy_features_forbidden()
    test_broad_negative_sampling()
    test_split_is_shot_level()
    test_search_beats_bad_baseline()
    print("=" * 76)
    print("All V2.11 selftests passed.")


if __name__ == "__main__":
    main()
