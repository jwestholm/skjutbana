from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.ai.hole_patch_ai_v214 import HolePatchAIV214, HolePatchAIConfigV214
from src.engine.ai.hole_patch_ensemble_v215 import (
    HolePatchEnsembleConfigV215,
    HolePatchEnsembleV215,
    extract_candidate_patch,
)
from src.engine.offline.hole_ensemble_v215 import (
    EnsembleSearchConfigV215,
    choose_blend_weight_v215,
)
from src.engine.offline.hole_training_v213 import EvaluationRows


def _rows(probabilities: list[float]) -> EvaluationRows:
    labels = np.asarray([1, 1, 0, 0], dtype=np.float32)
    return EvaluationRows(
        probabilities=np.asarray(probabilities, dtype=np.float32),
        labels=labels,
        predicted_offsets_px=np.zeros((4, 2), dtype=np.float32),
        target_offsets_px=np.zeros((4, 2), dtype=np.float32),
        baseline_scores=np.zeros((4,), dtype=np.float32),
        backgrounds=["x"] * 4,
        candidate_distance_px=np.asarray([2, 2, 28, 28], dtype=np.float32),
        asset_stems=["a", "b", "c", "d"],
    )


def main() -> int:
    print("V2.15 SELFTEST")
    print("==============")

    # 1. A model exactly at its own learned threshold must contribute zero
    # margin. Equal threshold probabilities therefore fuse to 0.5 independent
    # of calibration differences.
    fused = HolePatchEnsembleV215.fuse_probabilities(
        np.asarray([0.37], dtype=np.float32),
        np.asarray([0.63], dtype=np.float32),
        standard_threshold=0.37,
        mild_threshold=0.63,
        standard_weight=0.71,
    )
    assert abs(float(fused[0]) - 0.5) < 1e-5
    print("[PASS] threshold-centred fusion avoids raw-probability calibration bias")

    # 2. Construct a deliberately complementary paired problem. Each pure
    # model makes a different positive/negative mistake; the paired blend can
    # solve what neither pure endpoint solves alone.
    standard = _rows([0.90, 0.20, 0.80, 0.10])
    mild = _rows([0.20, 0.90, 0.10, 0.80])
    search = choose_blend_weight_v215(
        standard, mild, standard, mild,
        standard_threshold=0.5,
        mild_threshold=0.5,
        search=EnsembleSearchConfigV215(weight_step=0.05),
    )
    winner = search["winner"]
    assert bool(winner["blend_is_nontrivial"])
    assert float(winner["selection_score"]) > float(winner["best_pure_selection_score"])
    print("[PASS] paired selection can retain a genuinely complementary blend")

    # 3. Edge candidates must not disappear. V2.15 reflect-pads them.
    image = np.full((40, 40), 180, dtype=np.uint8)
    cv2.circle(image, (2, 2), 2, 20, -1)
    patch = extract_candidate_patch(image, (2, 2), 32)
    assert patch.shape == (32, 32)
    print("[PASS] candidate patch extraction handles image borders deterministically")

    # 4. Save/load two tiny V2.14 models and verify shadow annotation preserves
    # candidate order and original camera coordinates.
    with tempfile.TemporaryDirectory(prefix="v215_selftest_") as tmp:
        root = Path(tmp)
        cfg = HolePatchAIConfigV214(crop_size=32, input_size=8, hidden_size=6)
        std_model = HolePatchAIV214(cfg, seed=11)
        mild_model = HolePatchAIV214(cfg, seed=22)
        std_path = root / "standard.npz"
        mild_path = root / "mild.npz"
        std_model.save(std_path, metadata={"ai_threshold": 0.45})
        mild_model.save(mild_path, metadata={"ai_threshold": 0.55})
        ensemble_cfg = HolePatchEnsembleConfigV215(
            standard_model_path=str(std_path),
            mild_model_path=str(mild_path),
            standard_weight=0.6,
            fused_threshold=0.5,
            standard_threshold=0.45,
            mild_threshold=0.55,
        )
        config_path = root / "ensemble.json"
        config_path.write_text(json.dumps({"ensemble_config": ensemble_cfg.__dict__}, indent=2), encoding="utf-8")
        ensemble = HolePatchEnsembleV215.load(config_path)
        gray = np.full((80, 80), 160, dtype=np.uint8)
        cv2.circle(gray, (20, 20), 2, 25, -1)
        source = [
            {"id": "first", "camera_x": 20.0, "camera_y": 20.0, "score": 0.1},
            {"id": "second", "camera_x": 60.0, "camera_y": 60.0, "score": 0.9},
        ]
        annotated = ensemble.annotate_candidates(gray, source)
        assert [row["id"] for row in annotated] == ["first", "second"]
        assert [row["camera_x"] for row in annotated] == [20.0, 60.0]
        assert all(row.get("hole_v215_shadow_only") is True for row in annotated)
        assert all("hole_v215_fused_probability" in row for row in annotated)
        assert source[0].get("hole_v215_fused_probability") is None
    print("[PASS] shadow annotator preserves authority/order and does not mutate source candidates")

    print("\nAll V2.15 selftests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
