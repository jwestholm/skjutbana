from __future__ import annotations

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.offline.physical_dense_v2214 import (
    DensePoolConfigV2214,
    DenseRankerV2214,
    DenseTrainingConfigV2214,
    FEATURE_NAMES,
    build_dense_pool_v2214,
    make_shot_training_data_v2214,
    rank_dense_pool_v2214,
    train_dense_ranker_v2214,
)


def _current_cloud() -> list[dict]:
    rows = []
    for y in range(52, 205, 18):
        for x in range(42, 218, 18):
            rows.append({"camera_x": float(x), "camera_y": float(y), "score": 1.0})
    return rows


def _world(seed: int, gt: tuple[float, float]):
    rng = np.random.default_rng(seed)
    h = w = 256
    names = (
        "blackhat_gain", "tophat_gain", "persistent_abs", "gradient_gain",
        "persistent_dark", "persistent_bright", "compact_change",
    )
    maps = {}
    yy, xx = np.mgrid[0:h, 0:w]
    for i, name in enumerate(names):
        arr = rng.random((h, w), dtype=np.float32) * (0.12 + 0.015 * i)
        # Strong nuisance structures that should not automatically win.
        arr[84:87, 30:226] += 0.60
        arr[154:158, 35:220] += 0.52
        # Hole signature: strongest in blackhat/tophat/abs, still visible in others.
        sigma = 2.0 + 0.15 * i
        blob = np.exp(-((xx - gt[0]) ** 2 + (yy - gt[1]) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        gain = 0.92 if name in ("blackhat_gain", "tophat_gain", "persistent_abs") else 0.64
        arr += gain * blob
        maps[name] = np.clip(arr, 0.0, 1.0).astype(np.float32)
    fused = np.mean(np.stack([maps[n] for n in names], axis=0), axis=0).astype(np.float32)
    return maps, fused


def _nearest(rows, gt):
    return min((math.hypot(float(r["camera_x"]) - gt[0], float(r["camera_y"]) - gt[1]) for r in rows), default=9999.0)


def main() -> int:
    print("V2.21.4 SELFTEST")
    print("================")
    pool_cfg = DensePoolConfigV2214(
        per_source_limit=800,
        pool_limit=5000,
        target_margin_px=30,
    )
    train_cfg = DenseTrainingConfigV2214(
        stage1_epochs=45,
        stage2_epochs=30,
        hard_negatives_per_shot=160,
        random_negatives_per_shot=80,
        stage2_mined_negatives_per_shot=120,
        pairs_per_positive=32,
        top_k_values=(32, 64),
        frozen_top_k=64,
    )
    current = _current_cloud()

    train_shots = []
    for i, gt in enumerate(((103.0, 117.0), (131.0, 132.0), (174.0, 109.0), (92.0, 174.0)), 1):
        maps, fused = _world(100 + i, gt)
        pool, _mask = build_dense_pool_v2214(current, maps, fused, config=pool_cfg)
        assert _nearest(pool, gt) <= 10.0, "broad dense pool failed to include synthetic new-hole evidence"
        train_shots.append(make_shot_training_data_v2214(
            f"dev:{i}", current, maps, fused, gt,
            pool_config=pool_cfg, training_config=train_cfg,
        ))
    print("[PASS] broad GT-free dense pool contains compact temporal evidence")

    model, report = train_dense_ranker_v2214(train_shots, training_config=train_cfg, pool_config=pool_cfg)
    assert len(model.weights) == len(FEATURE_NAMES)
    assert report["development_shots"] == 4
    print("[PASS] pairwise physical-domain ranker trains on DEVELOPMENT-style samples")

    heldout_gt = (158.0, 181.0)
    maps, fused = _world(999, heldout_gt)
    pool, _mask = build_dense_pool_v2214(current, maps, fused, config=pool_cfg)
    ranked = rank_dense_pool_v2214(pool, maps, fused, model, limit=64)
    assert _nearest(ranked, heldout_gt) <= 20.0, "learned dense top-K did not retain held-out hole"
    print("[PASS] frozen learned ranker retains held-out hole in top-K")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.npz"
        model.save(path)
        loaded = DenseRankerV2214.load(path)
        a = model.score_features(train_shots[0].positive_features[:3])
        b = loaded.score_features(train_shots[0].positive_features[:3])
        assert np.allclose(a, b)
    print("[PASS] model save/load is deterministic")

    # Public proposal/ranking functions intentionally have no GT argument.
    import inspect
    assert "gt" not in inspect.signature(build_dense_pool_v2214).parameters
    assert "gt" not in inspect.signature(rank_dense_pool_v2214).parameters
    print("[PASS] inference path has no GT input")

    print()
    print("All V2.21.4 selftests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
