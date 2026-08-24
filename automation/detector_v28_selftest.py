from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from src.engine.ai.hypothesis_v28 import HypothesisBuilderV28
from src.engine.ai.ranker_v6 import RankerV6


def _candidate(x: float, y: float, *, score: float = 8.0, **extra):
    value = {
        "camera_x": float(x),
        "camera_y": float(y),
        "score": float(score),
        "detector_v1": 0.0,
        "detector_v2": 1.0,
        "v24_tile_probe": 0.0,
        "detector_agreement": 0.0,
        "v26_vault_hits": 1.0,
        "v26_vault_carried": 0.0,
        "v26_vault_age_s": 0.0,
        "persistence": 0.5,
        "existed_before": 0.0,
        "v2_absdiff": 3.0,
        "v2_zscore": 1.5,
        "v2_dog": 3.0,
        "v2_saliency": 10.0,
        "v24_patch_prior": 0.35,
    }
    value.update(extra)
    return value


def _distance(candidate, xy):
    return math.hypot(candidate["camera_x"] - xy[0], candidate["camera_y"] - xy[1])


def test_cluster_center() -> None:
    builder = HypothesisBuilderV28()
    gt = (500.0, 400.0)
    candidates = [
        _candidate(496, 401, detector_v1=1.0, detector_agreement=1.0, v26_vault_hits=3),
        _candidate(503, 398, v24_tile_probe=1.0, v26_vault_hits=2),
        _candidate(501, 404, detector_v1=1.0, v26_vault_hits=2),
        _candidate(498, 397, v24_tile_probe=1.0),
    ]
    hypotheses, pool, core, stats = builder.build(candidates)
    assert len(hypotheses) == 1
    assert len(pool) == 1 and len(core) == 1
    assert _distance(hypotheses[0], gt) < 3.0
    assert int(hypotheses[0]["v27_member_count"]) == 4
    assert stats["schema_version"] == "2.8"


def test_recall_pool_keeps_typical_189_clusters() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "hyp.json"
        cfg.write_text(json.dumps({
            "enabled": True,
            "merge_radius_px": 4.0,
            "max_cluster_diameter_px": 6.0,
            "core_hypotheses": 120,
            "max_hypotheses": 220,
            "macro_cell_px": 180.0,
            "macro_bucket_depth": 4,
        }))
        builder = HypothesisBuilderV28(cfg)
        candidates = []
        for i in range(189):
            x = 50.0 + (i % 21) * 75.0
            y = 50.0 + (i // 21) * 75.0
            candidates.append(_candidate(x, y, score=2.0 + (i % 11)))
        all_h, pool, core, stats = builder.build(candidates)
        assert len(all_h) == 189
        assert len(pool) == 189, (len(pool), stats)
        assert len(core) == 120
        assert stats["pool_dropped"] == 0
        assert stats["pool_mode"] == "keep_all"


def test_overflow_pool_preserves_spatial_and_evidence_diversity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "hyp.json"
        cfg.write_text(json.dumps({
            "enabled": True,
            "merge_radius_px": 4.0,
            "max_cluster_diameter_px": 6.0,
            "core_hypotheses": 120,
            "max_hypotheses": 220,
            "macro_cell_px": 160.0,
            "macro_bucket_depth": 3,
            "overflow_baseline_reserve": 50,
            "overflow_support_reserve": 30,
            "overflow_signal_reserve": 30,
            "overflow_diversity_reserve": 20,
            "overflow_vault_reserve": 20,
        }))
        builder = HypothesisBuilderV28(cfg)
        candidates = []
        # Flood centre-left with strong baseline noise.
        for i in range(340):
            candidates.append(_candidate(
                100 + (i % 34) * 18,
                100 + (i // 34) * 18,
                score=20.0,
            ))
        # Weak but multi-source remote hypotheses near all four edges.
        remotes = [(80, 80), (3650, 90), (90, 2050), (3650, 2050)]
        for x, y in remotes:
            candidates.append(_candidate(
                x, y, score=1.5,
                detector_v1=1.0,
                detector_v2=1.0,
                v24_tile_probe=1.0,
                detector_agreement=1.0,
                v26_vault_hits=4.0,
            ))
        _all, pool, _core, stats = builder.build(candidates)
        assert len(pool) <= 220
        for point in remotes:
            assert any(_distance(item, point) <= 10.0 for item in pool), point
        assert stats["pool_mode"] == "recall_overflow"


def _hypothesis(x, y, *, positive: bool):
    if positive:
        return {
            "camera_x": x, "camera_y": y,
            "v27_baseline_score": 0.40,
            "v27_support_score": 0.88,
            "v27_signal_score": 0.62,
            "v27_member_count": 5.0,
            "v27_compactness": 0.90,
            "v27_source_diversity": 3.0,
            "v27_current_fraction": 0.85,
            "v27_carried_fraction": 0.15,
            "v27_hits_max": 4.0,
            "v27_hits_mean": 2.7,
            "v27_v1_fraction": 0.45,
            "v27_v2_fraction": 0.65,
            "v27_tile_fraction": 0.70,
            "v27_agreement_fraction": 0.30,
            "v27_patch_prior_max": 0.65,
            "v27_patch_prior_median": 0.48,
            "v27_zscore_norm": 0.55,
            "v27_absdiff_norm": 0.55,
            "v27_dog_norm": 0.45,
            "v27_saliency_norm": 0.45,
            "v27_persistence_median": 0.65,
            "v27_existed_before_median": 0.05,
            "v27_age_median_s": 0.15,
        }
    return {
        "camera_x": x, "camera_y": y,
        "v27_baseline_score": 0.86,
        "v27_support_score": 0.25,
        "v27_signal_score": 0.72,
        "v27_member_count": 1.0,
        "v27_compactness": 0.45,
        "v27_source_diversity": 1.0,
        "v27_current_fraction": 0.45,
        "v27_carried_fraction": 0.55,
        "v27_hits_max": 1.0,
        "v27_hits_mean": 1.0,
        "v27_v1_fraction": 0.0,
        "v27_v2_fraction": 1.0,
        "v27_tile_fraction": 0.0,
        "v27_agreement_fraction": 0.0,
        "v27_patch_prior_max": 0.30,
        "v27_patch_prior_median": 0.25,
        "v27_zscore_norm": 0.72,
        "v27_absdiff_norm": 0.75,
        "v27_dog_norm": 0.72,
        "v27_saliency_norm": 0.72,
        "v27_persistence_median": 0.35,
        "v27_existed_before_median": 0.30,
        "v27_age_median_s": 0.8,
    }


def test_ranker_still_learns() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = root / "config.json"
        config.write_text(json.dumps({
            "enabled": True,
            "strict_positive_radius_px": 12.0,
            "soft_positive_radius_px": 20.0,
            "negative_safe_radius_px": 45.0,
            "hard_negatives": 12,
            "learning_rate": 0.09,
            "l2": 0.0005,
            "min_validation_shots": 30,
            "validation_window": 80,
            "min_conditional_top1_pct": 20.0,
            "min_advantage_pp": 10.0,
            "min_top3_advantage_pp": 10.0,
            "max_median_rank_ratio": 0.7,
            "min_top_score": 0.51,
            "min_score_margin": 0.005,
        }))
        model = RankerV6(root / "model.json", config)
        for shot in range(70):
            gt = (400.0 + (shot % 3), 300.0 + (shot % 5))
            pool = [_hypothesis(gt[0] + 2, gt[1] - 1, positive=True)]
            for index in range(14):
                pool.append(_hypothesis(80 + index * 45, 70 + (index % 4) * 70, positive=False))
            assert model.learn_from_ground_truth(gt, pool)["trained"]
        test_gt = (403.0, 304.0)
        test_pool = [_hypothesis(404, 304, positive=True)] + [
            _hypothesis(60 + i * 50, 80 + (i % 5) * 60, positive=False)
            for i in range(16)
        ]
        ranked = model.rank(test_pool)
        assert _distance(ranked[0], test_gt) <= 12.0


def main() -> None:
    tests = [
        ("robust micro-cluster centre", test_cluster_center),
        ("189-cluster recall pool keeps all", test_recall_pool_keeps_typical_189_clusters),
        ("overflow pool preserves diversity", test_overflow_pool_preserves_spatial_and_evidence_diversity),
        ("V6 learning still works", test_ranker_still_learns),
    ]
    print("=" * 76)
    print("DETECTOR / HYPOTHESIS V2.8 SELFTEST")
    print("=" * 76)
    for name, test in tests:
        test()
        print(f"PASS: {name}")
    print("=" * 76)
    print("All V2.8 selftests passed.")


if __name__ == "__main__":
    main()
