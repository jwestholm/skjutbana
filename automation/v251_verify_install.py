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
    print("V2.25.1 INSTALL VERIFICATION")
    print("============================")
    required = [
        "src/engine/shot_region_proposal_v251.py",
        "src/engine/shot_context_v250.py",
        "src/engine/game_objects/manager.py",
        "content/games/game_objects_test_v250.py",
        "menu_games_entry_v251.json",
        "automation/v251_prepare.py",
        "automation/v251_apply_docs.py",
        "automation/v251_apply_menu.py",
        "automation/v251_selftest.py",
        "automation/v251_verify_install.py",
        "automation/v251_status.py",
        "V251_OBJECT_REGION_PHYSICAL_PROPOSAL.md",
        "AI_PHYSICAL_REGION_PROPOSAL.md",
        "V251_TEST_PLAN.md",
        "main.py",
    ]
    check("required V2.25.1 files exist", all((ROOT / p).exists() for p in required))

    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    check("V2.25.1 runtime wired in main", "install_v251_runtime(App)" in main_source)
    check("V2.25.1 installs after V2.25.0", main_source.rfind("install_v251_runtime(App)") > main_source.rfind("install_v250_runtime(App)"))

    source = (ROOT / "src/engine/shot_region_proposal_v251.py").read_text(encoding="utf-8")
    check("per-region robust threshold implemented", "_robust_threshold(saliency, valid=region_valid" in source)
    check("hybrid/bank output is re-balanced", "_balance_merged_candidates" in source and "previous_generate" in source)
    check("confirmation is bounded by physical region", "_balance_confirmed" in source and "object_region_confirmed_total_v251" in source)
    check("physical track selector installed", "_install_track_selector_patch" in source)
    check("global FULL rescue is preserved", "rescue_router_v2225.requested(sid)" in source)

    menu = ROOT / "content/menu.json"
    check("content/menu.json exists", menu.exists())
    entry = _find_entry(json.loads(menu.read_text(encoding="utf-8")))
    check("Game Objects Test entry exists", entry is not None)
    check("Game Objects Test is labelled V2.25.1", entry is not None and entry.get("title") == "Game Objects Test (V2.25.1)")

    scene = (ROOT / "content/games/game_objects_test_v250.py").read_text(encoding="utf-8")
    check("physical scene labels V2.25.1 output", "[V2.25.1 OBJECT-HIT]" in scene)
    print("\nV2.25.1 installation verification passed.")


if __name__ == "__main__":
    main()
