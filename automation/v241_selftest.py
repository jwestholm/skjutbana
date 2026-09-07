from __future__ import annotations

import numpy as np

from src.engine.input.object_hit_v2223 import CameraHitRegionV240
from src.engine.shot_object_local_v241 import (
    LocalSearchWindowV241,
    build_local_valid_mask_v241,
    merge_camera_regions_v241,
)


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")



def _runtime_wrapper_smoke_test() -> None:
    import sys
    import threading
    import types
    import time

    import src.engine.shot_object_local_v241 as mod
    from src.engine.input.object_hit_v2223 import ObjectShotSnapshotV2223, object_hit_registry_v2223
    from src.engine.shot_fast_v2225 import rescue_router_v2225

    camera_pkg = types.ModuleType("src.engine.camera")
    camera_pkg.__path__ = []
    candidate_mod = types.ModuleType("src.engine.camera.candidate_generator_v2")

    class FakeGenerator:
        def _extract_candidates(self, *args, **kwargs):
            sid = 77
            rescue = rescue_router_v2225.consume(sid)
            return {
                "valid_pixels": int(np.count_nonzero(kwargs["valid"])),
                "rescue": bool(rescue),
            }

    candidate_mod.CandidateGeneratorV2 = FakeGenerator
    old_camera = sys.modules.get("src.engine.camera")
    old_candidate = sys.modules.get("src.engine.camera.candidate_generator_v2")
    sys.modules["src.engine.camera"] = camera_pkg
    sys.modules["src.engine.camera.candidate_generator_v2"] = candidate_mod

    class Event:
        state = "pending"
        shot_id = 77
        peak_ts = 1.0

    class Scanner:
        audio_events = [Event()]
        last_window_debug = {}

    snap = ObjectShotSnapshotV2223(
        shot_id=77, peak_ts=1.0, created_ts=time.time(), regions=(),
        camera_regions=(CameraHitRegionV240("obj", 120, 70, 20, 20),),
        transform_available=True, transform_method="selftest",
    )
    registry = object_hit_registry_v2223
    with registry._lock:
        old_snapshot = registry._snapshots.get(77)
        registry._snapshots[77] = snap

    rescue_router_v2225.clear(77)
    old_name = threading.current_thread().name
    try:
        mod._install_candidate_region_patch()
        threading.current_thread().name = "shot-cv-v2224-selftest"
        gen = FakeGenerator()
        valid = np.ones((100, 100), dtype=bool)
        local = gen._extract_candidates(scanner=Scanner(), valid=valid, bbox=(100, 50, 100, 100))
        check("runtime wrapper actually restricts first live valid mask", 0 < local["valid_pixels"] < valid.size)
        check("first local pass does not consume a rescue", not local["rescue"])

        check("runtime rescue request is accepted", rescue_router_v2225.request(77))
        rescued = gen._extract_candidates(scanner=Scanner(), valid=valid, bbox=(100, 50, 100, 100))
        check("FULL rescue receives original global valid mask", rescued["valid_pixels"] == valid.size)
        check("FULL rescue request reaches previous extractor", rescued["rescue"])
    finally:
        threading.current_thread().name = old_name
        rescue_router_v2225.clear(77)
        with registry._lock:
            if old_snapshot is None:
                registry._snapshots.pop(77, None)
            else:
                registry._snapshots[77] = old_snapshot
        if old_camera is None:
            sys.modules.pop("src.engine.camera", None)
        else:
            sys.modules["src.engine.camera"] = old_camera
        if old_candidate is None:
            sys.modules.pop("src.engine.camera.candidate_generator_v2", None)
        else:
            sys.modules["src.engine.camera.candidate_generator_v2"] = old_candidate

def main() -> None:
    print("V2.24.1 SELFTEST")
    print("===============")

    regions = [
        CameraHitRegionV240("target_a", 110, 110, 20, 20, role="target"),
        CameraHitRegionV240("no_shoot_b", 135, 110, 20, 20, role="no_shoot"),
        CameraHitRegionV240("target_c", 300, 180, 20, 20, role="target"),
    ]
    merged = merge_camera_regions_v241(regions, margin_px=5)
    check("overlapping expanded AABBs merge", len(merged) == 2)
    first = next(w for w in merged if "target_a" in w.object_ids)
    check("merged window preserves object provenance", set(first.object_ids) == {"target_a", "no_shoot_b"})
    check("no-shoot is search context, not excluded", "no_shoot" in first.roles)

    valid = np.ones((100, 200), dtype=bool)
    windows = (
        LocalSearchWindowV241(110, 60, 150, 90, ("one",), ("target",)),
        LocalSearchWindowV241(180, 120, 240, 180, ("two",), ("target",)),
    )
    local, diag = build_local_valid_mask_v241(valid, (100, 50, 200, 100), windows, source_region_count=2)
    check("camera AABB is translated into ROI-local mask", int(local[10:40, 10:50].sum()) == 1200)
    check("out-of-ROI window is clipped safely", diag.local_valid_pixels > 1200 and diag.local_valid_pixels < valid.size)
    check("diagnostics preserve source/merged counts", diag.region_count == 2 and diag.merged_count == 2)
    check("local valid fraction is bounded", 0.0 < diag.valid_fraction < 1.0)

    outside, outside_diag = build_local_valid_mask_v241(
        valid, (100, 50, 200, 100),
        (LocalSearchWindowV241(-500, -500, -400, -400),),
    )
    check("outside camera context produces empty local mask", not np.any(outside) and outside_diag.local_valid_pixels == 0)

    source = __import__("src.engine.shot_object_local_v241", fromlist=["x"]).__file__
    text = open(source, encoding="utf-8").read()
    rescue_pos = text.index("rescue_router_v2225.requested(sid)")
    mask_pos = text.index("build_local_valid_mask_v241(", rescue_pos)
    check("FULL rescue bypass is tested before local mask construction", rescue_pos < mask_pos)
    runtime_body = text.lower().split("def _install_candidate_region_patch", 1)[1]
    check("candidate coordinates are never snapped", "camera_x =" not in runtime_body and "camera_y =" not in runtime_body)

    _runtime_wrapper_smoke_test()

    print("\nAll V2.24.1 selftests passed.")


if __name__ == "__main__":
    main()
