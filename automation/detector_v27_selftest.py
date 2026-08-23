from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
import sys

# Allow this script to be run directly from the project root:
#   python3 automation/detector_v27_selftest.py
# When Python executes a file inside automation/, sys.path[0] points at
# automation/ rather than the repository root, so add the parent explicitly
# before importing src.* modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.ai.hypothesis_v27 import HypothesisBuilderV27
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
    builder = HypothesisBuilderV27()
    gt = (500.0, 400.0)
    candidates = [
        _candidate(496, 401, detector_v1=1.0, detector_agreement=1.0, v26_vault_hits=3),
        _candidate(503, 398, v24_tile_probe=1.0, v26_vault_hits=2),
        _candidate(501, 404, detector_v1=1.0, v26_vault_hits=2),
        _candidate(498, 397, v24_tile_probe=1.0),
    ]
    hypotheses, _pool, _stats = builder.build(candidates)
    assert len(hypotheses) == 1, len(hypotheses)
    assert _distance(hypotheses[0], gt) < 3.0, hypotheses[0]
    assert int(hypotheses[0]["v27_member_count"]) == 4


def test_spatial_pool_preserves_remote() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "hyp.json"
        cfg.write_text(json.dumps({
            "enabled": True,
            "merge_radius_px": 7.0,
            "max_cluster_diameter_px": 12.0,
            "max_hypotheses": 24,
            "macro_cell_px": 100.0,
            "macro_bucket_depth": 2,
        }))
        builder = HypothesisBuilderV27(cfg)
        candidates = []
        # Deliberately flood one macro area with strong-looking noise.
        for i in range(120):
            candidates.append(_candidate(20 + (i % 10) * 8, 20 + (i // 10) * 8, score=20.0))
        remote = _candidate(780, 620, score=2.0, detector_v1=1.0, v24_tile_probe=1.0)
        candidates.append(remote)
        _all, pool, stats = builder.build(candidates)
        assert len(pool) <= 24
        assert any(_distance(item, (780, 620)) <= 8.0 for item in pool), "remote hypothesis starved"
        assert stats["input_count"] == 121


def test_4k_pool_has_no_coordinate_edge_starvation() -> None:
    builder = HypothesisBuilderV27()
    candidates = []
    # More nominal 240px regions than the 120-hypothesis output limit on a 4K
    # coordinate plane. The adaptive macro grid must not simply truncate by x/y.
    for row in range(9):
        for col in range(16):
            candidates.append(
                _candidate(
                    80 + col * 240,
                    80 + row * 240,
                    score=8.0 + ((row + col) % 3),
                    detector_v1=float((row + col) % 2 == 0),
                )
            )
    # Explicit weak corner hypothesis: coordinate ordering must not be the
    # reason it disappears.
    corner = _candidate(3780, 2080, score=2.0, detector_v1=1.0, v24_tile_probe=1.0)
    candidates.append(corner)
    _all, pool, _stats = builder.build(candidates)
    assert len(pool) <= 120
    assert any(_distance(item, (3780, 2080)) <= 20.0 for item in pool), "4K edge starved by pooling"


def _hypothesis(x, y, *, positive: bool, misleading_baseline: bool = False):
    # Positive/negative are made distinct mainly on aggregate hypothesis support,
    # not absolute XY. Baseline can intentionally prefer negatives so learning
    # has something real to improve.
    if positive:
        support = 0.40 if misleading_baseline else 0.80
        return {
            "camera_x": x, "camera_y": y,
            "v27_baseline_score": support,
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
        "v27_baseline_score": 0.86 if misleading_baseline else 0.45,
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


def test_ranker_learns_hypothesis_support() -> None:
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
        baseline_top1 = 0
        v6_top1_pre = 0
        validation_shots = 90
        for shot in range(validation_shots):
            gt = (400.0 + (shot % 3), 300.0 + (shot % 5))
            pool = [_hypothesis(gt[0] + 2, gt[1] - 1, positive=True, misleading_baseline=True)]
            for index in range(14):
                pool.append(_hypothesis(80 + index * 45, 70 + (index % 4) * 70, positive=False, misleading_baseline=True))
            baseline = sorted(pool, key=lambda c: c["v27_baseline_score"], reverse=True)
            v6_before = model.rank(pool)
            if _distance(baseline[0], gt) <= 20:
                baseline_top1 += 1
            if _distance(v6_before[0], gt) <= 20:
                v6_top1_pre += 1
            model.record_validation(gt, baseline, v6_before)
            trained = model.learn_from_ground_truth(gt, pool)
            assert trained["trained"]

        test_gt = (403.0, 304.0)
        test_pool = [_hypothesis(404, 304, positive=True, misleading_baseline=True)] + [
            _hypothesis(60 + i * 50, 80 + (i % 5) * 60, positive=False, misleading_baseline=True)
            for i in range(16)
        ]
        ranked = model.rank(test_pool)
        assert _distance(ranked[0], test_gt) <= 12.0, "V6 failed held-out Top-1 synthetic test"
        assert model.stats["pair_updates"] > 500
        # We do not demand that the production gate opens in this miniature
        # test; we do demand a large learned separation from misleading baseline.
        assert model.score(test_pool[0]) > model.score(test_pool[-1]) + 0.10


def main() -> None:
    tests = [
        ("robust hypothesis centre", test_cluster_center),
        ("spatial pool coverage", test_spatial_pool_preserves_remote),
        ("4K spatial pool coverage", test_4k_pool_has_no_coordinate_edge_starvation),
        ("V6 pairwise learning", test_ranker_learns_hypothesis_support),
    ]
    print("=" * 72)
    print("DETECTOR / RANKER V2.7 SELFTEST")
    print("=" * 72)
    for name, test in tests:
        test()
        print(f"PASS: {name}")
    print("=" * 72)
    print("All V2.7 selftests passed.")


if __name__ == "__main__":
    main()
