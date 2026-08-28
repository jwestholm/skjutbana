from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.offline.physical_dense_v2215 import (
    DensePoolConfigV2215,
    ListwiseConfigV2215,
    ListwiseModelV2215,
    ListwiseShotV2215,
    candidate_distances_v2215,
    cross_validate_listwise_v2215,
    fit_listwise_ranker_v2215,
    propose_dense_pool_v2215,
    rank_candidates_v2215,
    CandidateFeatureBatchV2215,
)


def _blob(shape: tuple[int, int], x: int, y: int, sigma: float = 2.0, strength: float = 1.0) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return (strength * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma))).astype(np.float32)


def _dense_pool_test() -> None:
    shape = (220, 260)
    gx, gy = 137, 119
    rng = np.random.default_rng(2215)
    current = [
        {"camera_x": 55.0, "camera_y": 55.0},
        {"camera_x": 205.0, "camera_y": 55.0},
        {"camera_x": 205.0, "camera_y": 175.0},
        {"camera_x": 55.0, "camera_y": 175.0},
        {"camera_x": 130.0, "camera_y": 80.0},
        {"camera_x": 130.0, "camera_y": 155.0},
    ]
    maps: dict[str, np.ndarray] = {}
    for i, name in enumerate((
        "blackhat_gain", "tophat_gain", "persistent_abs", "gradient_gain",
        "persistent_dark", "persistent_bright", "fused", "compact_change",
    )):
        arr = np.clip(rng.normal(0.08 + 0.005 * i, 0.025, size=shape), 0.0, 1.0).astype(np.float32)
        arr += _blob(shape, gx + (i % 2), gy + ((i // 2) % 2), sigma=1.7 + 0.1 * i, strength=0.75)
        # Strong, elongated nuisance that should not be the only retained evidence.
        cv2.line(arr, (40, 35 + i * 7), (225, 35 + i * 7), 0.95, 2)
        maps[name] = np.clip(arr, 0.0, 1.0)
    result = propose_dense_pool_v2215(current, maps, config=DensePoolConfigV2215(pool_limit=5000, per_source_limit=1000))
    d = candidate_distances_v2215(result.candidates, (gx, gy))
    assert len(d) > 0, "dense pool is empty"
    assert float(np.min(d)) <= 5.0, f"dense pool missed compact evidence: nearest={float(np.min(d)):.2f}px"
    assert "gt" not in inspect.signature(propose_dense_pool_v2215).parameters
    print("[PASS] GT-free broad dense pool retains compact temporal evidence amid strong ridges")


def _listwise_test() -> None:
    rng = np.random.default_rng(2215)
    names = ("hole_evidence", "nuisance_strength", "agreement", "anchor_distance")
    shots: list[ListwiseShotV2215] = []
    for shot_id in range(12):
        n = 240
        distances = rng.uniform(45.0, 300.0, size=n).astype(np.float32)
        pos = int((shot_id * 37 + 11) % n)
        distances[pos] = rng.uniform(1.0, 8.0)
        # A few candidate-aligned near positives, all actual rows in the pool.
        for off in (1, 2):
            idx = (pos + off) % n
            distances[idx] = rng.uniform(9.0, 18.0)
        x = rng.normal(0.0, 1.0, size=(n, len(names))).astype(np.float32)
        positive = distances <= 20.0
        x[positive, 0] += 4.2
        x[positive, 2] += 2.5
        x[positive, 3] -= 2.0
        dense_score = (0.2 * x[:, 0] + rng.normal(0.0, 0.5, size=n)).astype(np.float32)
        shots.append(ListwiseShotV2215(
            key=f"selftest:{shot_id}", matrix=x, distances_px=distances, dense_scores=dense_score
        ))

    cfg = ListwiseConfigV2215(
        stage1_epochs=35,
        stage2_epochs=20,
        candidates_per_shot=240,
        hard_candidates_per_shot=120,
        random_candidates_per_shot=80,
        cv_folds=3,
        top_k_values=(8, 16, 32, 64),
        frozen_top_k=32,
        learning_rate=0.02,
    )
    cv = cross_validate_listwise_v2215(shots, names, config=cfg)
    assert float(cv.get("top32_oracle20", 0.0)) >= 0.80, cv
    model, report = fit_listwise_ranker_v2215(shots, names, config=cfg, metadata={"shot_keys": [s.key for s in shots]})
    assert model.metadata.get("candidate_aligned_only") is True
    assert int(model.metadata.get("forced_positive_jitter_count", -1)) == 0
    assert float(report["train_oracle20"].get("32", 0.0)) >= 0.90, report

    fake_rows = [{"camera_x": float(i), "camera_y": 0.0} for i in range(len(shots[0].matrix))]
    ranked = rank_candidates_v2215(fake_rows, CandidateFeatureBatchV2215(shots[0].matrix, names), model)
    assert len(ranked) == len(fake_rows)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.npz"
        model.save(path)
        loaded = ListwiseModelV2215.load(path)
        a = model.score_matrix(shots[0].matrix)
        b = loaded.score_matrix(shots[0].matrix)
        assert np.allclose(a, b), "save/load changed scores"
    print("[PASS] actual-candidate listwise objective learns and cross-fits without GT/jitter proposals")
    print("[PASS] V2.21.5 model save/load and frozen inference are deterministic")


def main() -> int:
    print("V2.21.5 SELFTEST")
    print("================")
    _dense_pool_test()
    _listwise_test()
    print("[PASS] inference proposal API has no GT parameter")
    print()
    print("All V2.21.5 selftests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
