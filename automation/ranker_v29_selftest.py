from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from src.engine.ai.ranker_v7 import FEATURE_KEYS, vectors_for_pool
import src.engine.ai.ranking_dataset_v29 as dataset
from automation.ranker_v29_experiment import (
    evaluate_model,
    train_pairwise_linear,
)


def _candidate(x: float, y: float, compactness: float, baseline: float, *, good: bool) -> dict:
    return {
        "camera_x": x,
        "camera_y": y,
        "v27_baseline_score": baseline,
        "v27_support_score": 0.25 if good else 0.75,
        "v27_signal_score": 0.22 if good else 0.82,
        "v27_member_count": 1.0 if good else 6.0,
        "v27_compactness": compactness,
        "v27_source_diversity": 1.0 if good else 3.0,
        "v27_current_fraction": 1.0,
        "v27_carried_fraction": 0.0,
        "v27_hits_max": 1.0 if good else 4.0,
        "v27_hits_mean": 1.0 if good else 3.0,
        "v27_v1_fraction": 1.0 if good else 0.5,
        "v27_v2_fraction": 0.0 if good else 1.0,
        "v27_tile_fraction": 1.0 if good else 0.0,
        "v27_agreement_fraction": 0.0 if good else 1.0,
        "v27_patch_prior_max": 0.30 if good else 0.80,
        "v27_patch_prior_median": 0.25 if good else 0.75,
        "v27_zscore_norm": 0.25 if good else 0.85,
        "v27_absdiff_norm": 0.20 if good else 0.90,
        "v27_dog_norm": 0.25 if good else 0.80,
        "v27_saliency_norm": 0.25 if good else 0.85,
        "v27_persistence_median": 0.25 if good else 0.85,
        "v27_existed_before_median": 0.0 if good else 0.6,
        "v27_age_median_s": 0.05 if good else 0.8,
        "v27_spread_px": 2.0 if good else 15.0,
        "v28_core_pool": 0.0 if good else 1.0,
        "v28_pool_reasons": ["keep_all"],
    }


def _synthetic_rows(count: int = 60) -> list[dict]:
    rows = []
    for shot in range(count):
        # Baseline deliberately prefers artifacts. The true hypothesis has a
        # consistent low-signal/fresh/single-member pattern that V7 should learn.
        raw_candidates = []
        positive = _candidate(100.0, 100.0, 0.92, 0.15, good=True)
        raw_candidates.append(positive)
        for index in range(12):
            artifact = _candidate(
                300.0 + index * 20.0,
                280.0 + index * 15.0,
                0.20,
                0.95 - index * 0.01,
                good=False,
            )
            raw_candidates.append(artifact)

        vectors = vectors_for_pool(raw_candidates)
        candidates = []
        baseline_order = sorted(
            range(len(raw_candidates)),
            key=lambda i: raw_candidates[i]["v27_baseline_score"],
            reverse=True,
        )
        baseline_rank = {index: rank + 1 for rank, index in enumerate(baseline_order)}

        for index, (raw, vector) in enumerate(zip(raw_candidates, vectors)):
            distance = math.hypot(raw["camera_x"] - 100.0, raw["camera_y"] - 100.0)
            candidates.append(
                {
                    "id": f"{raw['camera_x']:.4f},{raw['camera_y']:.4f}",
                    "camera_x": raw["camera_x"],
                    "camera_y": raw["camera_y"],
                    "distance_gt_px": distance,
                    "membership": {"hypothesis_pool": True, "core": bool(index)},
                    "ranks": {"baseline": baseline_rank[index]},
                    "features": vector,
                    "raw": {},
                }
            )

        rows.append(
            {
                "sequence": shot + 1,
                "candidates": candidates,
            }
        )
    return rows


def test_relative_features() -> None:
    candidates = [
        _candidate(0, 0, 0.2, 0.1, good=True),
        _candidate(1, 1, 0.5, 0.5, good=True),
        _candidate(2, 2, 0.8, 0.9, good=True),
    ]
    vectors = vectors_for_pool(candidates)
    assert vectors[0]["rel_baseline_score"] < vectors[1]["rel_baseline_score"] < vectors[2]["rel_baseline_score"]
    print("PASS: within-shot relative features")


def test_atomic_writer() -> None:
    old_root = dataset.DATA_ROOT
    old_session_root = dataset.SESSION_ROOT
    old_jsonl = dataset.JSONL_PATH
    old_status = dataset.STATUS_PATH
    try:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            dataset.DATA_ROOT = base / "ranking_v29"
            dataset.SESSION_ROOT = dataset.DATA_ROOT / "sessions"
            dataset.JSONL_PATH = dataset.DATA_ROOT / "ranking_dataset.jsonl"
            dataset.STATUS_PATH = dataset.DATA_ROOT / "status.json"

            writer = dataset.RankingDatasetWriter("selftest")
            candidates = [
                _candidate(100, 100, 0.9, 0.2, good=True),
                _candidate(300, 300, 0.2, 0.9, good=False),
            ]
            writer.write_shot(
                gt_xy=(100.0, 100.0),
                all_hypotheses=candidates,
                hypothesis_pool=candidates,
                core_pool=candidates[:1],
                baseline_pool=list(reversed(candidates)),
                recall_baseline_pool=list(reversed(candidates)),
                v6_pool=list(reversed(candidates)),
                actual_pool=list(reversed(candidates)),
                filtered_input=candidates,
            )
            rows = list(writer.session_dir.glob("shot_*.json"))
            assert len(rows) == 1
            payload = json.loads(rows[0].read_text(encoding="utf-8"))
            assert len(payload["candidates"]) == 2
            assert payload["oracle"]["pool_within_10"] is True
    finally:
        dataset.DATA_ROOT = old_root
        dataset.SESSION_ROOT = old_session_root
        dataset.JSONL_PATH = old_jsonl
        dataset.STATUS_PATH = old_status
    print("PASS: atomic ranking dataset writer")


def test_pairwise_learning() -> None:
    rows = _synthetic_rows()
    feature_keys = [
        "signal_score",
        "member_count",
        "current_fraction",
        "zscore",
        "absdiff",
        "persistence",
        "not_existed_before",
        "single_member",
        "low_signal_current",
    ]
    weights, stats = train_pairwise_linear(
        rows[:48],
        feature_keys,
        epochs=80,
        hard_negatives=10,
    )
    result = evaluate_model(rows[48:], feature_keys, weights)
    baseline = result["baseline"]["20"]
    model = result["model"]["20"]
    assert stats["positive_shots"] > 0
    assert model["top1_pct"] >= 90.0
    assert model["top1_pct"] > baseline["top1_pct"]
    print(
        "PASS: offline pairwise learner "
        f"(baseline top1={baseline['top1_pct']}%, model top1={model['top1_pct']}%)"
    )


def main() -> None:
    print("=" * 78)
    print("V2.9 OFFLINE RANKING SELFTEST")
    print("=" * 78)
    test_relative_features()
    test_atomic_writer()
    test_pairwise_learning()
    print("=" * 78)
    print("All V2.9 selftests passed.")


if __name__ == "__main__":
    main()
