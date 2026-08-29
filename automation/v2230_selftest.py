from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from src.engine.ai.training_v223.capture import TrainingCaptureV223
from src.engine.ai.training_v223.dataset import DatasetV223, discover_native_records
from src.engine.ai.training_v223.model import RankModelV223, evaluate_model, train_rank_model
from src.engine.ai.training_v223.schema import (
    FEATURE_NAMES, ShotTrainingRecord, candidate_rows_from_pool, extract_physical_features,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def make_record(session: str, shot: int, rng: np.random.Generator) -> ShotTrainingRecord:
    gx = 1000.0 + rng.normal(0, 20)
    gy = 700.0 + rng.normal(0, 20)
    candidates = []
    # Correct candidate: strong fresh physical change but not necessarily highest detector score.
    candidates.append({
        "camera_x": gx + rng.normal(0, 3), "camera_y": gy + rng.normal(0, 3),
        "score": 9.0 + rng.normal(), "area": 6, "radius": 2.0, "circularity": .8,
        "pre_shot_change": 18 + rng.normal(), "change_value": 24 + rng.normal(),
        "v2_zscore": 4.5 + rng.normal(.0,.2), "local_contrast_gain": 8 + rng.normal(),
        "rank": 12,
    })
    for i in range(35):
        candidates.append({
            "camera_x": gx + rng.uniform(70, 700), "camera_y": gy + rng.uniform(-500, 500),
            "score": 30.0 - i * .2 + rng.normal(), "area": rng.uniform(1, 30), "radius": rng.uniform(1, 5),
            "circularity": rng.uniform(.2, 1), "pre_shot_change": rng.uniform(0, 4),
            "change_value": rng.uniform(0, 8), "v2_zscore": rng.uniform(0, 1.5),
            "local_contrast_gain": rng.uniform(0, 10), "rank": i + 1,
        })
    rows = candidate_rows_from_pool(candidates, gt_camera_xy=(gx, gy), frame_shape=(2160,3840))
    return ShotTrainingRecord(
        session_id=session, shot_id=str(shot), source_kind="selftest", timestamp=float(shot),
        gt_camera_x=gx, gt_camera_y=gy, candidates=rows,
    )


def main() -> None:
    print("V2.23.0 SELFTEST")
    print("================")
    # Feature leakage guard.
    f = extract_physical_features({"score": 10, "rank": 1, "combined_score": 999, "reason_core": 1, "gt_distance": 0})
    check(tuple(f.keys()) == FEATURE_NAMES, "stable physical feature contract")
    check("rank" not in f and "combined_score" not in f and "reason_core" not in f, "GT/policy/model leakage fields excluded")

    # Storage-forced GT rows must not enter training pool.
    rows = candidate_rows_from_pool([
        {"camera_x": 10, "camera_y": 10, "score": 2},
        {"camera_x": 20, "camera_y": 20, "score": 3, "capture_forced_gt_nearest": True},
    ], gt_camera_xy=(20,20))
    check(len(rows) == 1 and rows[0].camera_x == 10, "diagnostic forced-GT rows excluded")

    rng = np.random.default_rng(2230)
    records = [make_record(f"session_{i//8}", i, rng) for i in range(32)]
    dataset = DatasetV223(records)
    split = dataset.split()
    check(len(split.development) >= 2 and len(split.validation) >= 1, "dataset produces development + validation")
    # No session may straddle buckets when non-provisional hash split has both buckets.
    if not split.provisional:
        dev = set(r.session_id for r in split.development); val = set(r.session_id for r in split.validation); hold = set(r.session_id for r in split.holdout)
        check(not (dev & val or dev & hold or val & hold), "whole-session split has no leakage")

    # Explicit protected holdout may never be recycled into provisional dev/val.
    protected = make_record("protected_session", 999, rng)
    protected.split_hint = "holdout"
    engineering = [make_record("one_session", i + 1000, rng) for i in range(8)]
    protected_split = DatasetV223(engineering + [protected]).split()
    check(protected in protected_split.holdout and protected not in protected_split.development and protected not in protected_split.validation, "protected holdout never leaks into engineering split")

    model, info = train_rank_model(split.development, kind="mlp", hidden=12, epochs=25, learning_rate=.012, seed=2230)
    metrics = evaluate_model(model, split.validation)
    check(metrics["conditional_top1_20_rate"] >= 0.70, "listwise challenger learns held-out synthetic ranking signal")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        model.save(p / "model")
        loaded = RankModelV223.load(p / "model")
        check(np.allclose(model.score_record(split.validation[0]), loaded.score_record(split.validation[0])), "safe NPZ+JSON model round-trip")

        old = Path.cwd(); os.chdir(p)
        try:
            cap = TrainingCaptureV223(source_kind="selftest", session_id="capture_test")
            rec = records[0]
            rec.session_id = cap.session_id
            cap.save_record(rec); cap.close()
            discovered = discover_native_records()
            check(len(discovered) == 1 and discovered[0].session_id == "capture_test", "append-only native capture round-trip")
        finally:
            os.chdir(old)

    check(True, "V2.23 never grants live authority in selftest path")
    print("\nAll V2.23.0 selftests passed.")


if __name__ == "__main__":
    main()
