from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def main() -> None:
    print("V2.24.1 INSTALL VERIFICATION")
    print("============================")
    required = [
        ROOT / "main.py",
        ROOT / "src/engine/shot_object_local_v241.py",
        ROOT / "src/engine/shot_fast_v2225.py",
        ROOT / "src/engine/input/object_hit_v2223.py",
        ROOT / "src/engine/input/hit_regions.py",
        ROOT / "src/engine/scenes/game.py",
        ROOT / "src/engine/scenes/overlay_scene.py",
        ROOT / "GAME_DEVELOPMENT.md",
        ROOT / "V241_OBJECT_LOCAL_PHYSICAL_SEARCH.md",
        ROOT / "V241_TEST_PLAN.md",
        ROOT / "automation/v241_apply_docs.py",
    ]
    check("required V2.24.1 files exist", all(p.exists() for p in required))

    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    local_text = (ROOT / "src/engine/shot_object_local_v241.py").read_text(encoding="utf-8")
    game_doc = (ROOT / "GAME_DEVELOPMENT.md").read_text(encoding="utf-8")

    order = [
        main_text.index("install_v2223_runtime(App)"),
        main_text.index("install_v2224_runtime(App)"),
        main_text.index("install_v2225_runtime(App)"),
        main_text.index("install_v2226_runtime(App)"),
        main_text.index("install_v241_runtime(App)"),
    ]
    check("runtime installers preserve required order", order == sorted(order))
    check("V2.24.1 consumes frozen camera regions", "snapshot_for_shot" in local_text and "camera_regions" in local_text)
    check("regions are expanded and merged", "merge_camera_regions_v241" in local_text and "margin_px" in local_text)
    check("local search intersects existing valid mask", "restricted = base & mask" in local_text)
    check("V2.22.5 FULL rescue remains global", "rescue_router_v2225.requested(sid)" in local_text and "reason=v2225_full_rescue" in local_text)
    check("missing/invalid context fails open", "return previous_extract(self, *args, **kwargs)" in local_text)
    check("game docs state no object authority", "search here" in game_doc.lower() and "physical pre->post evidence remains mandatory" in game_doc.lower())
    check("V2.24 stable HitRegion API is carried forward", "HitRegion" in (ROOT / "src/engine/input/hit_regions.py").read_text(encoding="utf-8"))

    print("\nV2.24.1 installation verification passed.")


if __name__ == "__main__":
    main()
