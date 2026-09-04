from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def main() -> None:
    print("V2.24.0 INSTALL VERIFICATION")
    print("============================")

    required = [
        ROOT / "src/engine/input/object_hit_v2223.py",
        ROOT / "src/engine/input/hit_regions.py",
        ROOT / "src/engine/scenes/game.py",
        ROOT / "src/engine/scenes/overlay_scene.py",
        ROOT / "GAME_DEVELOPMENT.md",
    ]
    check("required V2.24.0 files exist", all(p.exists() for p in required))

    object_text = required[0].read_text(encoding="utf-8")
    facade_text = required[1].read_text(encoding="utf-8")
    game_text = required[2].read_text(encoding="utf-8")
    overlay_text = required[3].read_text(encoding="utf-8")

    check("stable HitRegion facade exists", "class GameHitRegionV240" in object_text and "HitRegion" in facade_text)
    check("game regions are viewport-local AABBs", "width: float" in object_text and "height: float" in object_text)
    check("four-corner game->camera transform exists", "transform_game_rect_to_camera_aabb" in object_text)
    check("GameScene proxies optional game provider", "def get_hit_regions" in game_text)
    check("OverlayScene proxies GameScene provider", "def get_hit_regions" in overlay_text)
    check("legacy V2.22.3 registry API is preserved", "object_hit_registry_v2223" in object_text and "HitRegionV2223" in object_text)
    check("V2.24 does not patch live hit authority", "push_camera_hit(" not in object_text)

    main_py = ROOT / "main.py"
    if main_py.exists():
        text = main_py.read_text(encoding="utf-8", errors="replace")
        check(
            "existing V2.22.3 shot-critical installer remains present",
            "install_v2223_runtime" in text or "shot_critical_v2223" in text,
        )

    print("\nV2.24.0 installation verification passed.")


if __name__ == "__main__":
    main()
