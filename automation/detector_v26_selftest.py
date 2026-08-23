from __future__ import annotations

import json
import math
import sys
import tempfile
import time
import types
from pathlib import Path

# This ZIP is an overlay and intentionally does not contain the whole camera
# package.  During the dependency-light self-test, install a namespace parent
# so importing detector_v26_extension does not execute camera/__init__.py and
# require the user's camera_manager/hit_scanner files. In the real project the
# normal package imports are used unchanged.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_camera_parent = types.ModuleType("src.engine.camera")
_camera_parent.__path__ = [str(_REPO_ROOT / "src" / "engine" / "camera")]
sys.modules.setdefault("src.engine.camera", _camera_parent)

from src.engine.ai.ranker_v5 import RankerV5
from src.engine.camera.detector_v26_extension import DEFAULT_CONFIG, ShotCandidateVault


def candidate(x: float, y: float, *, score: float = 5.0, tile: bool = False, good: bool = False) -> dict:
    item = {
        "camera_x": float(x),
        "camera_y": float(y),
        "score": float(score),
        "detector_v1": 0.0,
        "detector_v2": 1.0,
        "detector_agreement": 0.0,
        "v24_tile_probe": 1.0 if tile else 0.0,
        "v24_patch_core_abs": 6.0 if good else 1.0,
        "v24_patch_core_z": 3.0 if good else 0.4,
        "v24_patch_compactness": 0.85 if good else 0.15,
        "v24_patch_centeredness": 0.90 if good else 0.20,
        "v24_patch_isotropy": 0.85 if good else 0.10,
        "v24_patch_local_snr": 0.90 if good else 0.15,
    }
    # Coarse patch pattern: compact centre for positives, edge-like stripe for negatives.
    for r in range(5):
        for c in range(5):
            d = math.hypot(r - 2, c - 2)
            if good:
                abs_v = max(0.0, 1.0 - 0.32 * d)
                signed_v = max(-1.0, min(1.0, 0.9 - 0.42 * d))
                z_v = max(0.0, 1.0 - 0.28 * d)
            else:
                abs_v = 0.85 if c == 2 else 0.12
                signed_v = 0.70 if c == 2 else -0.08
                z_v = 0.75 if c == 2 else 0.10
            item[f"v24_patch_abs_g{r}{c}"] = abs_v
            item[f"v24_patch_signed_g{r}{c}"] = signed_v
            item[f"v24_patch_z_g{r}{c}"] = z_v
    return item


def test_shot_vault() -> None:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "shot_vault_cell_px": 10.0,
        "shot_vault_carried_slots": 40,
        "shot_vault_output_limit": 80,
        "shot_vault_max_cells": 120,
        "shot_vault_max_age_s": 3.0,
    })
    vault = ShotCandidateVault()
    shot_id = 7
    t0 = time.time()

    # True hypothesis is visible in frame 1 only.
    first = [candidate(405, 305, score=6.0, tile=True, good=True)]
    first += [candidate(40 + i * 8, 60 + (i % 5) * 7, score=7.0) for i in range(24)]
    vault.observe(shot_id, first, t0, cfg)
    out1, _ = vault.build_output(shot_id, first, t0, cfg)
    assert any(math.hypot(c["camera_x"] - 405, c["camera_y"] - 305) <= 5 for c in out1)

    # Frame 2 contains only unrelated candidates. The true point must survive
    # as carried history without requiring a second observation.
    second = [candidate(50 + i * 6, 80 + (i % 7) * 9, score=8.0) for i in range(28)]
    vault.observe(shot_id, second, t0 + 0.08, cfg)
    out2, stats = vault.build_output(shot_id, second, t0 + 0.08, cfg)
    carried_true = [
        c for c in out2
        if c.get("v26_vault_carried", 0.0) > 0.5
        and math.hypot(c["camera_x"] - 405, c["camera_y"] - 305) <= 8
    ]
    assert carried_true, "single-frame GT hypothesis was not preserved"
    assert stats["carried"] > 0

    # Spatial diversity test: fill one macro region heavily and keep one remote
    # point. The round-robin carry selection must not starve the remote region.
    third = [candidate(20 + (i % 18) * 5, 20 + (i // 18) * 5, score=9.0) for i in range(90)]
    third.append(candidate(780, 520, score=3.0, tile=True, good=True))
    vault.observe(8, third, t0, cfg)
    vault.observe(8, [candidate(30, 30, score=10.0)], t0 + 0.05, cfg)
    out3, _ = vault.build_output(8, [candidate(30, 30, score=10.0)], t0 + 0.05, cfg)
    assert any(math.hypot(c["camera_x"] - 780, c["camera_y"] - 520) <= 10 for c in out3), (
        "spatially remote vault hypothesis was starved by dense region"
    )


def test_ranker_v5() -> None:
    with tempfile.TemporaryDirectory(prefix="ranker_v5_selftest_") as tmp:
        root = Path(tmp)
        cfg_path = root / "ranker_v5_config.json"
        cfg = {
            "enabled": True,
            "positive_radius_px": 12.0,
            "negative_safe_radius_px": 55.0,
            "hard_negatives": 16,
            "learning_rate": 0.08,
            "l2": 0.0005,
            "epochs_per_shot": 1,
            "save_every_shots": 10000,
            "training_log_enabled": False,
            "auto_override_enabled": True,
            "validation_window": 120,
            "min_validation_shots": 80,
            "min_conditional_top1_pct": 12.0,
            "min_advantage_pp": 6.0,
            "min_score_margin": 0.01,
            "min_top_score": 0.50,
        }
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        model = RankerV5(
            model_path=root / "ranker_v5.json",
            config_path=cfg_path,
            log_path=root / "pairs.jsonl",
        )

        # Train only on ACTUAL generated positives close to GT.
        for shot in range(120):
            gt = (300.0, 250.0)
            positive = candidate(302 + (shot % 2), 249, score=4.0, tile=True, good=True)
            negatives = [
                candidate(40 + i * 18, 40 + (i % 8) * 35, score=12.0 - 0.1 * i, good=False)
                for i in range(24)
            ]
            result = model.learn_from_ground_truth(gt, [positive] + negatives)
            assert result.get("trained") is True

        held_out = [
            candidate(302, 250, score=3.0, tile=True, good=True),
            *[candidate(50 + i * 16, 70 + (i % 9) * 31, score=14.0, good=False) for i in range(30)],
        ]
        ranked = model.rank(held_out)
        assert math.hypot(ranked[0]["camera_x"] - 300, ranked[0]["camera_y"] - 250) <= 12

        # Gate is PRE-TRAIN validation driven. Demonstrate that it remains
        # closed before enough evidence and opens only after sustained advantage.
        model.stats["validation"] = []
        base_pool = [held_out[5], held_out[0], *held_out[1:5]]
        v5_pool = model.rank(held_out)
        for _ in range(79):
            model.record_validation((300, 250), base_pool, v5_pool, match_radius_px=12.0)
        assert model.gate_status()["open"] is False
        model.record_validation((300, 250), base_pool, v5_pool, match_radius_px=12.0)
        gate = model.gate_status()
        assert gate["open"] is True, gate


def main() -> None:
    test_shot_vault()
    test_ranker_v5()
    print("Detector V2.6 self-test: PASS")
    print(" - shot vault preserves single-frame hypotheses: PASS")
    print(" - shot vault spatial diversity: PASS")
    print(" - Ranker V5 strict actual-candidate learning: PASS")
    print(" - Ranker V5 evidence gate: PASS")


if __name__ == "__main__":
    main()
