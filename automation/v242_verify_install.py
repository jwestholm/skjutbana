from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def _contains_entry(node) -> bool:
    if isinstance(node, dict):
        if node.get("id") == "hit_context_test_v242":
            return True
        return any(_contains_entry(v) for v in node.values())
    if isinstance(node, list):
        return any(_contains_entry(v) for v in node)
    return False


def main() -> None:
    print("V2.24.2 INSTALL VERIFICATION")
    print("============================")
    required = [
        ROOT / "content/games/hit_context_test_v242.py",
        ROOT / "menu_games_entry_v242.json",
        ROOT / "automation/v242_apply_menu.py",
        ROOT / "automation/v242_apply_docs.py",
        ROOT / "automation/v242_prepare.py",
        ROOT / "automation/v242_selftest.py",
        ROOT / "V242_GAME_CONTEXT_TESTSCENE.md",
        ROOT / "V242_TEST_PLAN.md",
        ROOT / "GAME_DEVELOPMENT.md",
        ROOT / "ROADMAP.md",
        ROOT / "src/engine/shot_object_local_v241.py",
    ]
    check("required V2.24.2 files exist", all(p.exists() for p in required))

    scene = (ROOT / "content/games/hit_context_test_v242.py").read_text(encoding="utf-8")
    check("scene uses stable HitRegion API", "from src.engine.input.hit_regions import HitRegion" in scene)
    check("scene has shot-time snapshot diagnostics", "latest_hit_context_snapshot" in scene)
    check("scene has moving target", "moving_target" in scene and "moving.vx" in scene)
    check("scene has overlap and edge cases", "overlap_target" in scene and "edge_target" in scene)
    check("scene has outside-region challenge", "outside_challenge" in scene)
    check("scene has EMPTY/global mode", "EMPTY REGIONS" in scene or "empty_regions" in scene)

    menu = ROOT / "content/menu.json"
    check("content/menu.json exists", menu.exists())
    menu_data = json.loads(menu.read_text(encoding="utf-8"))
    check("menu contains V2.24.2 testscene after prepare", _contains_entry(menu_data))

    print("\nV2.24.2 installation verification passed.")


if __name__ == "__main__":
    main()
