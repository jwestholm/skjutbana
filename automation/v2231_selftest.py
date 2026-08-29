from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from src.engine.ai.training_v223 import dataset as ds
from src.engine.ai.training_v223.integration import _candidate_union
from src.engine.ai.training_v223.registry import research_promotion_gate
from src.engine.ai.training_v223.schema import ShotTrainingRecord, candidate_rows_from_pool


def check(ok: bool, text: str) -> None:
    if not ok:
        raise AssertionError(text)
    print(f"[PASS] {text}")


def main() -> None:
    print("V2.23.1 SELFTEST")
    print("================")

    gt = (100.0, 100.0)
    candidates = [
        {"camera_x": 100.0, "camera_y": 100.0, "score": 1.0, "area": 5},
        {"camera_x": 130.0, "camera_y": 100.0, "score": 2.0, "area": 6},
        {"camera_x": 180.0, "camera_y": 100.0, "score": 3.0, "area": 7},
    ]
    rows = candidate_rows_from_pool(candidates, gt_camera_xy=gt, frame_shape=(200, 200))
    rec = ShotTrainingRecord(
        session_id="s", shot_id="1", source_kind="test", timestamp=1.0,
        gt_camera_x=gt[0], gt_camera_y=gt[1], candidates=rows,
    )
    check(rec.oracle20, "actual candidate within 20px creates oracle20")
    check(ds.DatasetV223([rec]).summary()["oracle42"] == 1, "dataset reports independent proposal oracle radii")

    # V2.8 high-recall pools must be captured even if the displayed/final list is narrow.
    runtime = SimpleNamespace(
        _v28_all_hypotheses=[
            {"camera_x": 10.0, "camera_y": 10.0, "score": 1.0},
            {"camera_x": 20.0, "camera_y": 20.0, "score": 2.0},
            {"camera_x": 30.0, "camera_y": 30.0, "score": 3.0},
        ],
        _v28_hypothesis_pool=[
            {"camera_x": 20.0, "camera_y": 20.0, "score": 2.0},
            {"camera_x": 30.0, "camera_y": 30.0, "score": 3.0},
        ],
        latest_candidates=[{"camera_x": 10.0, "camera_y": 10.0, "score": 1.0}],
    )
    scene = SimpleNamespace(runtime=runtime, ranked_candidates=[])
    wide = _candidate_union(scene, include_v28_recall=True)
    narrow = _candidate_union(scene, include_v28_recall=False)
    check(len(wide) >= 3 and len(narrow) == 1, "native capture adds V2.8 all-hypotheses/recall pool without GT")
    check(any("_v28_all_hypotheses" in c.get("provenance", []) for c in wide), "high-recall provenance is preserved")

    # V2.23.0's zero-positive plumbing champion must never pass V2.23.1.
    bad = {
        "metrics": {"shots": 20, "oracle20": 0, "oracle20_rate": 0.0, "conditional_top1_20_rate": 0.0, "mrr20": 0.0},
        "support": {},
        "baseline_metrics": {},
    }
    check(not research_promotion_gate(bad)["passed"], "zero-positive validation champion is rejected/quarantined")

    good = {
        "metrics": {"shots": 20, "oracle20": 12, "oracle20_rate": 0.6, "conditional_top1_20_rate": 0.45, "mrr20": 0.55},
        "support": {"development_oracle20": 30},
        "baseline_metrics": {"conditional_top1_20_rate": 0.30, "mrr20": 0.40},
    }
    check(research_promotion_gate(good)["passed"], "support-gated challenger that beats baseline can become research champion")

    # Exercise the exact V2.16 adapter shape without needing a real 500MB archive.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "sessions" / "session_a" / "shot_000001.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        fake_pack = SimpleNamespace(
            metadata={"session_id": "session_a", "round_id": 1, "schema_version": "2.16"},
            gt_xy=(50.0, 60.0),
            candidates=[{"camera_x": 50.0, "camera_y": 60.0, "score": 4.0, "area": 6}],
            full_pre_frame=None, full_recent_pre_frame=None, full_post_frames=None,
            recent_pre_patches=None,
        )
        original = ds._official_pack_loader
        original_cache = ds.LEGACY_CACHE_ROOT
        ds.LEGACY_CACHE_ROOT = Path(td) / "cache"
        try:
            ds._official_pack_loader = lambda _path: (fake_pack, None)
            loaded, reason = ds.load_legacy_candidate_record(path, Path(td) / "candidate_shadow_v216" / "sessions")
            check(loaded is not None and loaded.oracle20 and reason == "loaded", "V2.16 CandidatePack adapter converts GT + actual candidates")
            # Second read should use cheap converted cache rather than legacy NPZ again.
            ds._official_pack_loader = lambda _path: (_ for _ in ()).throw(RuntimeError("should not be called"))
            cached, reason2 = ds.load_legacy_candidate_record(path, Path(td) / "candidate_shadow_v216" / "sessions")
            check(cached is not None and reason2 == "cache", "legacy conversion cache avoids rereading large JSON+NPZ packs")
        finally:
            ds._official_pack_loader = original
            ds.LEGACY_CACHE_ROOT = original_cache

    print("\nAll V2.23.1 selftests passed.")


if __name__ == "__main__":
    main()
