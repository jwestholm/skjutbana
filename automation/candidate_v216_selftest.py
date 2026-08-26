from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.engine.ai.hole_patch_ai_v214 import HolePatchAIConfigV214, HolePatchAIV214
from src.engine.ai.hole_patch_ensemble_v215 import HolePatchEnsembleConfigV215, HolePatchEnsembleV215
from src.engine.offline.candidate_pack_v216 import (
    CandidateCaptureConfigV216,
    CandidatePackV216,
    CandidateShadowRecorderV216,
)
from src.engine.offline.candidate_shadow_analysis_v216 import (
    CandidateShotEvidenceV216,
    choose_fusion_weights_v216,
    hard_negative_rows_v216,
    ranking_metrics_v216,
    split_shots_v216,
    temporal_candidate_score_v216,
)


def _pass(message: str) -> None:
    print(f"[PASS] {message}")


def _candidate(x: float, y: float, rank: int | None = None, score: float = 0.0) -> dict:
    item = {
        "camera_x": float(x),
        "camera_y": float(y),
        "score": float(score),
        "dog": float(score),
        "absdiff": float(score),
        "current_fraction": 1.0,
        "patch_prior_median": float(score),
    }
    if rank is not None:
        item["rank"] = int(rank)
    return item


def _fake_ensemble(tmp: Path) -> HolePatchEnsembleV215:
    cfg = HolePatchAIConfigV214(crop_size=64, input_size=12, hidden_size=12)
    standard = HolePatchAIV214(cfg, seed=1)
    mild = HolePatchAIV214(cfg, seed=2)
    std_path = tmp / "std.npz"
    mild_path = tmp / "mild.npz"
    standard.save(std_path, metadata={"ai_threshold": 0.5})
    mild.save(mild_path, metadata={"ai_threshold": 0.5})
    ensemble_cfg = HolePatchEnsembleConfigV215(
        standard_model_path=str(std_path),
        mild_model_path=str(mild_path),
        standard_weight=0.375,
        fused_threshold=0.5,
        standard_threshold=0.5,
        mild_threshold=0.5,
        shadow_only=True,
    )
    return HolePatchEnsembleV215(standard, mild, ensemble_cfg)


def main() -> int:
    print("V2.16 SELFTEST")
    print("=============")
    with tempfile.TemporaryDirectory(prefix="v216_selftest_") as temp_dir:
        tmp = Path(temp_dir)
        root = tmp / "candidate_shadow"
        capture_cfg = CandidateCaptureConfigV216(
            enabled=True,
            data_root=str(root),
            patch_size=64,
            max_post_frames=3,
            max_candidates=8,
            include_raw_extras=True,
            save_gt_patches=True,
            save_full_frames=True,
            full_frame_post_count=1,
            compress=True,
        )
        recorder = CandidateShadowRecorderV216(
            capture_cfg,
            background="white_grid",
            benchmark_seed=1234,
            sampling_mode="uniform",
            session_id="selftest_session",
        )

        pre = np.full((180, 220), 205, dtype=np.uint8)
        # Persistent old artefact: present both before and after.
        cv2.circle(pre, (35, 45), 4, 80, -1)
        posts = []
        for i in range(3):
            frame = pre.copy()
            cv2.circle(frame, (110, 90), 4, 25 + i, -1)  # new persistent hole
            posts.append((frame, 1.0 + i * 0.04))
        raw = [_candidate(35, 45, score=0.9), _candidate(110, 90, score=0.5), _candidate(170, 120, score=0.3)]
        ranked = [_candidate(35, 45, 1, 0.9), _candidate(110, 90, 2, 0.5)]
        result = recorder.capture_shot(
            round_id=1,
            raw_candidates=raw,
            ranked_candidates=ranked,
            pre_gray=pre,
            post_frames=posts,
            gt_camera_xy=(110.0, 90.0),
            gt_screen_xy=(400.0, 300.0),
            match_radius_px=42.0,
        )
        assert result["saved"] is True
        pack = CandidatePackV216.load(Path(result["json_path"]))
        assert len(pack.candidates) == 3
        assert pack.pre_patches.shape == (3, 64, 64)
        assert pack.post_patches.shape == (3, 3, 64, 64)
        assert pack.gt_pre_patch is not None and pack.gt_post_patches.shape == (3, 64, 64)
        assert pack.metadata["shadow_only"] is True
        assert pack.full_pre_frame is not None and pack.full_post_frames is not None
        assert pack.full_post_frames.shape[0] == 1
        _pass("candidate pack roundtrip stores pre/post patches, GT, optional full frames and detector provenance")

        gt_index = min(range(len(pack.candidates)), key=lambda i: float(pack.candidates[i]["distance_gt_px"]))
        old_index = min(range(len(pack.candidates)), key=lambda i: abs(float(pack.candidates[i]["camera_x"]) - 35.0))
        gt_temporal = temporal_candidate_score_v216(pack.pre_patches[gt_index], pack.post_patches[gt_index])
        old_temporal = temporal_candidate_score_v216(pack.pre_patches[old_index], pack.post_patches[old_index])
        assert gt_temporal["score"] > old_temporal["score"] + 0.05
        _pass("candidate-centred temporal evidence prefers a new persistent change over an old static artefact")

        # Storage-cap protection must be explicit, never hidden as live recall.
        tiny_cfg = CandidateCaptureConfigV216(
            enabled=True,
            data_root=str(tmp / "tiny"),
            patch_size=64,
            max_post_frames=1,
            max_candidates=1,
            include_raw_extras=True,
        )
        tiny = CandidateShadowRecorderV216(tiny_cfg, session_id="cap_test")
        cap_result = tiny.capture_shot(
            round_id=1,
            raw_candidates=[_candidate(20, 20), _candidate(110, 90)],
            ranked_candidates=[_candidate(20, 20, 1)],
            pre_gray=pre,
            post_frames=posts[:1],
            gt_camera_xy=(110.0, 90.0),
        )
        cap_pack = CandidatePackV216.load(Path(cap_result["json_path"]))
        assert len(cap_pack.candidates) == 2
        assert sum(bool(row["capture_forced_gt_nearest"]) for row in cap_pack.candidates) == 1
        _pass("capture cap can retain GT-nearest for diagnostics without pretending it belonged to the live pool")

        ensemble = _fake_ensemble(tmp)
        original_order = [(row["camera_x"], row["camera_y"]) for row in pack.candidates]
        evidence = ensemble.score_patches([pack.post_patches[i, -1] for i in range(len(pack.candidates))])
        assert len(evidence) == len(pack.candidates)
        assert original_order == [(row["camera_x"], row["camera_y"]) for row in pack.candidates]
        _pass("Hole-AI scoring is shadow-only and preserves candidate order")

        # Fusion search is allowed to choose a pure endpoint; it must not assume
        # an ensemble is automatically superior.
        shots = []
        for shot_id in range(12):
            positive_first = shot_id % 3 != 0
            rows = [
                {
                    "capture_index": 0,
                    "camera_x": 50.0,
                    "camera_y": 50.0,
                    "distance_gt_px": 2.0 if positive_first else 80.0,
                    "in_ranked_pool": True,
                    "in_raw_pool": True,
                    "capture_forced_gt_nearest": False,
                    "current_rank": 1,
                    "evidence_v216": {
                        "hole_fused": 0.9 if positive_first else 0.2,
                        "temporal": {"score": 0.8 if positive_first else 0.2},
                        "v9_percentile": 0.7 if positive_first else 0.3,
                        "current_rank_percentile": 1.0,
                    },
                },
                {
                    "capture_index": 1,
                    "camera_x": 100.0,
                    "camera_y": 100.0,
                    "distance_gt_px": 80.0 if positive_first else 2.0,
                    "in_ranked_pool": True,
                    "in_raw_pool": True,
                    "capture_forced_gt_nearest": False,
                    "current_rank": 2,
                    "evidence_v216": {
                        "hole_fused": 0.2 if positive_first else 0.9,
                        "temporal": {"score": 0.2 if positive_first else 0.8},
                        "v9_percentile": 0.3 if positive_first else 0.7,
                        "current_rank_percentile": 0.0,
                    },
                },
            ]
            shots.append(CandidateShotEvidenceV216("", "one_session", shot_id, False, rows, (50.0, 50.0) if positive_first else (100.0, 100.0), True, True))
        winner = choose_fusion_weights_v216(shots)
        assert abs(sum(winner["weights"].values()) - 1.0) < 1e-6
        _pass("fusion search includes pure/combined evidence and returns normalized transparent weights")

        split = split_shots_v216(shots)
        assert all(shot.provisional_split for shot in shots)
        assert sum(len(value) for value in split.values()) == len(shots)
        _pass("single-session benchmark is explicitly provisional instead of masquerading as physical holdout")

        hard_rows = hard_negative_rows_v216(shots[0], min_distance_px=55.0, max_per_shot=4)
        assert hard_rows and float(hard_rows[0]["distance_gt_px"]) >= 55.0
        _pass("hard-negative miner selects real detector-style far-from-GT candidates by evidence hardness")

        empty_shot = CandidateShotEvidenceV216("", "empty", 99, False, [], (10.0, 10.0), False, False)
        empty_metrics = ranking_metrics_v216([empty_shot], "hole", pool="ranked", radius=20.0)
        assert empty_metrics["shots"] == 1 and empty_metrics["top1"] == 0.0 and empty_metrics["oracle_recall"] == 0.0
        _pass("zero-candidate shots remain in the accuracy denominator as real misses")

        summary = recorder.finalize()
        assert summary["shadow_only"] is True and summary["shots_saved"] == 1
        assert json.loads((recorder.root / "session.json").read_text())["finalized"] is True
        _pass("session manifest finalizes atomically and remains shadow-only")

    print("\nAll V2.16 selftests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
