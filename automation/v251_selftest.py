from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.engine.input.object_hit_v2223 import CameraHitRegionV240, object_hit_registry_v2223
from src.engine.shot_region_proposal_v251 import (
    RegionGroupV251,
    _balance_confirmed,
    _balance_merged_candidates,
    _bbox_region_mask,
    _camera_regions_to_work_groups,
    _dedupe_candidates,
    _merge_groups,
    _region_physical_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def test_region_grouping() -> None:
    glass = RegionGroupV251("glass", ("glass",), ("target",), 10, 10, 110, 110)
    rear = RegionGroupV251("rear", ("rear",), ("target",), 10, 10, 110, 110)
    other = RegionGroupV251("no", ("no",), ("no_shoot",), 200, 10, 300, 110)
    groups = _merge_groups((glass, rear, other))
    check("same physical glass/rear area groups once", len(groups) == 2)
    first = next(g for g in groups if "glass" in g.object_ids)
    check("group retains both downstream object ids", set(first.object_ids) == {"glass", "rear"})
    check("separate physical area remains separate", any(g.object_ids == ("no",) for g in groups))


def test_full_camera_to_work_group() -> None:
    original = object_hit_registry_v2223.snapshot_for_shot
    try:
        snap = SimpleNamespace(camera_regions=(
            CameraHitRegionV240("crate", 1700, 1100, 200, 100, role="target"),
            CameraHitRegionV240("hard", 2600, 1200, 180, 100, role="no_shoot"),
        ))
        object_hit_registry_v2223.snapshot_for_shot = lambda sid: snap if int(sid) == 9 else None
        scanner = SimpleNamespace(_v244_roi_diag={
            "crop": (1500.0, 900.0, 1500.0, 750.0),
            "scale": (1.0, 1.0),
            "work_shape": (750, 1500),
        })
        groups = _camera_regions_to_work_groups(scanner, 9)
        crate = next(g for g in groups if "crate" in g.object_ids)
        check("camera region maps into V2.22.1 work plane", abs(crate.x0 - 164.0) < 1e-6 and abs(crate.y0 - 164.0) < 1e-6)
        check("no-shoot role has no mapping special case", any("no_shoot" in g.roles for g in groups))
    finally:
        object_hit_registry_v2223.snapshot_for_shot = original


def test_bbox_local_mask() -> None:
    valid = np.ones((100, 120), dtype=bool)
    group = RegionGroupV251("a", ("a",), ("target",), 70, 80, 100, 120)
    mask = _bbox_region_mask(valid.shape, valid, (50, 50, 170, 150), group)
    check("work-space region subtracts CandidateGenerator bbox origin", int(mask.sum()) == 30 * 40)
    check("bbox-local mask starts at expected pixel", bool(mask[30, 20]))
    check("bbox-local mask does not move outside region", not bool(mask[10, 10]))


def _candidate(group: str, x: float, evidence: float, *, registered: bool = True) -> dict:
    return {
        "camera_x": float(x), "camera_y": 10.0, "score": 10.0 + evidence,
        "v251_region_group": group,
        "v251_region_objects": group,
        "v251_region_roles": "target",
        "v251_region_evidence": float(evidence),
        "v251_registered_region_proposal": 1.0 if registered else 0.0,
    }


def test_merged_balance() -> None:
    groups = (
        RegionGroupV251("a", ("a",), ("target",), 0, 0, 100, 100),
        RegionGroupV251("b", ("b",), ("no_shoot",), 100, 0, 200, 100),
        RegionGroupV251("c", ("c",), ("target",), 200, 0, 300, 100),
    )
    noisy = [_candidate("a", 5 + i * 5, 30 - i) for i in range(20)]
    others = [_candidate("b", 120 + i * 5, 8 - i) for i in range(3)] + [_candidate("c", 220, 4)]
    balanced = _balance_merged_candidates(noisy + others, groups, shot_id=12)
    names = {str(c.get("v251_region_group")) for c in balanced}
    check("noisy region cannot crowd out other physical regions", names == {"a", "b", "c"})
    check("per-region proposal quota is bounded", sum(1 for c in balanced if c.get("v251_region_group") == "a") <= 8)
    check("shot id annotation survives pool balancing", all(int(c.get("v251_shot_id", 0)) == 12 for c in balanced))


def test_confirmation_balance_and_no_xy_snap() -> None:
    candidates = []
    for group, role, base in (("a", "target", 10), ("b", "no_shoot", 110), ("c", "target", 210)):
        for i in range(7):
            candidates.append({
                "camera_x": float(base + i), "camera_y": 20.0,
                "score": float(20 - i),
                "v251_region_group": group,
                "v251_region_roles": role,
                "v251_region_evidence": float(8 - i * 0.3),
                "v2225_confirm_center_abs": 3.0,
                "v2225_confirm_compact": 1.2,
                "v2225_confirm_peak_abs": 6.0,
                "v2225_confirm_darkening": 2.0,
            })
    before_xy = {(c["camera_x"], c["camera_y"]) for c in candidates}
    balanced = _balance_confirmed(candidates)
    names = {c["v251_region_group"] for c in balanced}
    check("confirmation preserves at least one survivor per physical region", names == {"a", "b", "c"})
    check("confirmation pool is globally bounded", len(balanced) <= 8)
    check("no-shoot receives same physical confirmation opportunity", any(c["v251_region_group"] == "b" for c in balanced))
    check("confirmation never invents/moves candidate XY", all((c["camera_x"], c["camera_y"]) in before_xy for c in balanced))


def test_overlap_dedupe_keeps_membership() -> None:
    a = _candidate("glass", 50, 9)
    a["v251_region_objects"] = "glass"
    b = _candidate("rear", 51, 8)
    b["v251_region_objects"] = "rear"
    result = _dedupe_candidates([a, b], radius=3.5)
    check("same physical candidate is deduplicated", len(result) == 1)
    check("dedupe retains overlapping object identities", set(result[0]["v251_region_objects"].split("|")) == {"glass", "rear"})



def test_region_physical_extractor() -> None:
    class FakeGenerator:
        def _refine_peak(self, *, px, py, **kwargs):
            return px, py, 0.0
        def _candidate_features(self, *, px, py, saliency, absdiff, darkening, dog, zscore):
            return {
                "area": 3.0, "radius": 1.2, "circularity": 0.8,
                "score": float(saliency[py, px]),
                "center_change": float(absdiff[py, px]),
                "local_contrast": float(dog[py, px]),
                "dog_value": float(dog[py, px]),
                "zscore": float(zscore[py, px]),
                "absdiff": float(absdiff[py, px]),
                "darkening": float(darkening[py, px]),
            }
        def _apply_known_hole_penalty(self, scanner, candidate):
            return None

    shape = (80, 100)
    saliency = np.zeros(shape, dtype=np.float32)
    absdiff = np.zeros(shape, dtype=np.float32)
    darkening = np.zeros(shape, dtype=np.float32)
    dog = np.zeros(shape, dtype=np.float32)
    zscore = np.zeros(shape, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    valid[10:60, 20:70] = True
    # One compact fresh physical change inside this region.
    absdiff[34, 45] = 8.0
    darkening[34, 45] = 7.0
    zscore[34, 45] = 5.0
    dog[34, 45] = 3.0
    saliency[34, 45] = 48.0
    group = RegionGroupV251("crate", ("crate",), ("target",), 120, 210, 170, 260)
    candidates, diag = _region_physical_candidates(
        FakeGenerator(), SimpleNamespace(), group,
        saliency=saliency, absdiff=absdiff, darkening=darkening, dog=dog,
        zscore=zscore, region_valid=valid, bbox=(100, 200, 200, 280),
        frame_ts=10.0, region_threshold=10.0,
        cfg={"min_temporal_change": 1.8, "min_zscore": 1.5, "strong_temporal_change": 4.0},
        shot_id=55,
    )
    check("per-region extractor finds compact registered PRE->POST peak", bool(candidates))
    best = candidates[0]
    check("per-region extractor preserves observed physical coordinate", best["camera_x"] == 145.0 and best["camera_y"] == 234.0)
    check("per-region candidate carries region + shot identity", best["v251_region_group"] == "crate" and best["v251_shot_id"] == 55)
    check("per-region evidence is positive", best["v251_region_evidence"] > 0 and diag["best"] > 0)


def test_source_invariants() -> None:
    source = (ROOT / "src/engine/shot_region_proposal_v251.py").read_text(encoding="utf-8")
    check("FULL rescue explicitly bypasses region proposal", "rescue_router_v2225.requested(sid)" in source)
    check("registered PRE->POST temporal map is present", "absdiff.astype(np.float32) * (1.0 + 0.55 * np.clip(zscore" in source)
    check("track selector uses physical confirm evidence", "v251_confirm_score" in source and "_best_track_for_event" in source)
    check("runtime does not contain target-role ranking weight", "role_weight" not in source and "target_bonus" not in source)


def test_menu_patch() -> None:
    from automation.v251_apply_menu import patch_menu_text
    entry = json.loads((ROOT / "menu_games_entry_v251.json").read_text(encoding="utf-8"))
    sample = json.dumps({"children": [{"id": "games", "children": [json.loads((ROOT / "menu_games_entry_v250.json").read_text(encoding="utf-8"))]}]}, ensure_ascii=False, indent=2) + "\n"
    patched, changed, found = patch_menu_text(sample, entry)
    check("existing GameObject test entry found", found)
    check("menu title updated to V2.25.1", changed and "Game Objects Test (V2.25.1)" in patched)
    patched2, changed2, found2 = patch_menu_text(patched, entry)
    check("V2.25.1 menu update is idempotent", found2 and not changed2 and patched2 == patched)


def test_install_order() -> None:
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    i250 = source.rfind("install_v250_runtime(App)")
    i251 = source.rfind("install_v251_runtime(App)")
    check("V2.25.1 installs after GameObject/shot-id foundation", i250 >= 0 and i251 > i250)


def main() -> None:
    print("V2.25.1 SELFTEST")
    print("===============")
    test_region_grouping()
    test_full_camera_to_work_group()
    test_bbox_local_mask()
    test_merged_balance()
    test_confirmation_balance_and_no_xy_snap()
    test_overlap_dedupe_keeps_membership()
    test_region_physical_extractor()
    test_source_invariants()
    test_menu_patch()
    test_install_order()
    print("\nAll V2.25.1 selftests passed.")


if __name__ == "__main__":
    main()
