from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def _find_entry(node):
    if isinstance(node, dict):
        if node.get("id") == "game_objects_test_v250":
            return node
        for value in node.values():
            result = _find_entry(value)
            if result is not None:
                return result
    elif isinstance(node, list):
        for value in node:
            result = _find_entry(value)
            if result is not None:
                return result
    return None


def main() -> None:
    print("V2.25.0 INSTALL VERIFICATION")
    print("============================")
    required = [
        "src/engine/game_objects/__init__.py",
        "src/engine/game_objects/geometry.py",
        "src/engine/game_objects/projectiles.py",
        "src/engine/game_objects/damage.py",
        "src/engine/game_objects/events.py",
        "src/engine/game_objects/model.py",
        "src/engine/game_objects/manager.py",
        "src/engine/game_objects/presets.py",
        "src/engine/shot_context_v250.py",
        "content/games/game_objects_test_v250.py",
        "menu_games_entry_v250.json",
        "automation/v250_prepare.py",
        "automation/v250_selftest.py",
        "automation/v250_verify_install.py",
        "GAME_OBJECT_SYSTEM.md",
        "AI_GAME_OBJECTS.md",
        "V250_GAME_OBJECT_SYSTEM_PLAN.md",
        "V250_TEST_PLAN.md",
        "main.py",
    ]
    check("required V2.25 files exist", all((ROOT / p).exists() for p in required))

    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    check("shot-id runtime wired in main", "install_v250_runtime(App)" in main_source)
    check("V2.25 installs after V2.24.4", main_source.rfind("install_v250_runtime(App)") > main_source.rfind("install_v244_runtime(App)"))

    model = (ROOT / "src/engine/game_objects/model.py").read_text(encoding="utf-8")
    manager = (ROOT / "src/engine/game_objects/manager.py").read_text(encoding="utf-8")
    bridge = (ROOT / "src/engine/shot_context_v250.py").read_text(encoding="utf-8")
    check("object schema includes entity/part identity", "entity_id" in model and "part_id" in model)
    check("exact frozen snapshot collision implemented", "_frozen_hits" in manager and "shape_contains" in manager)
    check("generation stale-shot guard implemented", "shot.stale_object" in manager)
    check("penetration chain implemented", "shot.penetrated" in manager and "shot.blocked" in manager)
    check("effect request boundary implemented", "effect.requested" in manager)
    check("HitEvent is annotated before subscribers", "_build_event_from_camera" in bridge and "annotate_hit_event_v250" in bridge)

    menu = ROOT / "content/menu.json"
    check("content/menu.json exists", menu.exists())
    entry = _find_entry(json.loads(menu.read_text(encoding="utf-8")))
    check("Game Objects Test menu entry exists", entry is not None)

    print("\nV2.25.0 installation verification passed.")


if __name__ == "__main__":
    main()
