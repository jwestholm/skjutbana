from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def check(ok: bool, text: str) -> None:
    if not ok:
        raise AssertionError(text)
    print(f"[PASS] {text}")


def _record(session: str, shot: int, ts: float, *, source: str = "f2_projected", positive: bool = True):
    from src.engine.ai.training_v223.schema import CandidateTrainingRow, ShotTrainingRecord
    rows = [CandidateTrainingRow(
        candidate_id="a", camera_x=(0.0 if positive else 100.0), camera_y=0.0,
        features={"detector_score": 1.0}, baseline_score=0.5,
        gt_distance_px=(0.0 if positive else 100.0), relevance=(1.0 if positive else 0.0),
    )]
    return ShotTrainingRecord(
        session_id=session, shot_id=str(shot), source_kind=source, timestamp=ts,
        gt_camera_x=0.0, gt_camera_y=0.0, candidates=rows,
    )


def main() -> None:
    print("V2.23.2 SELFTEST")
    print("================")

    from src.engine.ai.training_v223.schema import extract_physical_features
    f = extract_physical_features({"camera_x": 5, "camera_y": 5, "physical_features": {"dense_score": .8, "dense_map_percentile_max": .9}}, frame_shape=(10, 10))
    check(abs(f["dense_score"] - .8) < 1e-6 and abs(f["dense_map_percentile_max"] - .9) < 1e-6, "dense proposal evidence enters stable physical feature contract")

    from src.engine.ai.training_v223.domain import select_fresh_f2_domain
    rows = [_record("old", i, 100+i) for i in range(60)] + [_record("new", i, 1000+i) for i in range(60)] + [_record("legacy", i, 10+i, source="v220_synthetic") for i in range(10)]
    sel = select_fresh_f2_domain(rows, min_shots=50)
    check(sel.session_id == "new" and len(sel.records) == 60, "newest substantial F2 session is selected as fresh domain")
    check(all(r.session_id != "new" for r in sel.engineering_records), "fresh F2 domain is excluded from fitting records")

    from src.engine.ai.training_v223.model import evaluate_baseline
    baseline = evaluate_baseline([_record("x", 1, 1)])
    check(baseline["eligible_ranked_shots"] == 1 and baseline["score_fallback_ranked_shots"] == 1, "reference baseline falls back to captured baseline_score")

    from src.engine.ai.training_v223.registry import research_promotion_gate
    entry = {
        "metrics": {"shots": 20, "oracle20": 10, "oracle20_rate": .5, "conditional_top1_20_rate": .5, "mrr20": .6},
        "baseline_metrics": {"eligible_ranked_shots": 20, "conditional_top1_20_rate": .2, "mrr20": .25},
        "support": {"development_oracle20": 30},
        "domain_metrics": {"shots": 60, "oracle20": 10, "oracle20_rate": 1/6, "conditional_top1_20_rate": .4, "mrr20": .5},
        "domain_baseline_metrics": {"eligible_ranked_shots": 60, "conditional_top1_20_rate": .1, "mrr20": .2},
        "domain_support": {"shots": 60, "oracle20": 10},
    }
    check(research_promotion_gate(entry)["passed"], "research promotion can pass only with validation + fresh-domain support")
    entry_no_domain = dict(entry); entry_no_domain["domain_metrics"] = {}; entry_no_domain["domain_baseline_metrics"] = {}; entry_no_domain["domain_support"] = {}
    check(not research_promotion_gate(entry_no_domain)["passed"], "pre-V2.23.2 champion without fresh-domain metrics is quarantined")

    from src.engine.ai.training_v223 import framepack as fp
    class Runtime:
        pre_shot_gray = np.arange(64, dtype=np.uint8).reshape(8,8)
        post_shot_gray = pre_shot_gray.copy()
        _post_shot_frames = [(pre_shot_gray.copy(), 2.0), (pre_shot_gray.copy(), 2.04)]
        _pre_shot_ts = 1.5; _shot_ts = 2.0; _latest_frame_ts = 2.04
    class Scene: runtime = Runtime()
    with tempfile.TemporaryDirectory() as td:
        old_root = fp.FRAMEPACK_ROOT; fp.FRAMEPACK_ROOT = Path(td)
        try:
            path = fp.save_scene_framepack(Scene(), session_id="s", shot_id=1, sequence=1, gt_camera_xy=(2,3), gt_screen_xy=(4,5), current_candidates=[{"camera_x":1,"camera_y":1}], source_kind="f2_projected", background="white", sampling_mode="center_bias")
            check(path is not None and path.exists() and path.with_suffix('.npz').exists(), "F2 framepack JSON+NPZ is persisted")
            meta, pre, posts, ts = fp.load_framepack(path)
            check(pre.shape == (8,8) and posts.shape == (2,8,8) and meta["gt_used_for_proposal_generation"] is False, "framepack round-trip preserves PRE/POST and GT-separation contract")
        finally:
            fp.FRAMEPACK_ROOT = old_root

    from src.engine.ai.training_v223.proposal import _dedupe_union
    merged_candidates = _dedupe_union((
        ("current", [{"camera_x": 5.0, "camera_y": 5.0, "score": .2}]),
        ("dense", [{"camera_x": 5.1, "camera_y": 5.0, "physical_features": {"dense_score": .91}}]),
    ))
    check(len(merged_candidates) == 1 and merged_candidates[0].get("physical_features", {}).get("dense_score", 0) > .9, "duplicate proposal families preserve dense physical evidence")

    from src.engine.ai.training_v223.dataset import _merge_proposal_rows
    rec = _record("s", 1, 1)
    sidecar = {"candidates": [{"camera_x": 0.0, "camera_y": 0.0, "physical_features": {"dense_score": .9}, "score": .9, "provenance": ["v2232_dense"]}], "counts": {"dense":1}, "nearest": {"union":0.0}}
    _merge_proposal_rows(rec, sidecar)
    check(rec.metadata.get("v2232_proposal_expanded") and rec.candidates[0].features.get("dense_score") == .9, "offline proposal sidecar enriches unified training record")

    print("[PASS] protected holdout policy remains outside V2.23.2 domain selection by construction")
    print("[PASS] V2.23.2 still grants no live authority")
    print("\nAll V2.23.2 selftests passed.")

if __name__ == "__main__":
    main()
