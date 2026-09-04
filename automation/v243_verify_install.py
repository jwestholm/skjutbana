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
        if node.get("id") == "hit_context_test_v242":
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
    print("V2.24.3 INSTALL VERIFICATION")
    print("============================")
    required = [
        ROOT / "src/engine/shot_object_local_v243.py",
        ROOT / "content/games/hit_context_test_v242.py",
        ROOT / "automation/v243_apply_docs.py",
        ROOT / "automation/v243_apply_menu.py",
        ROOT / "automation/v243_prepare.py",
        ROOT / "automation/v243_selftest.py",
        ROOT / "V243_LOCAL_ROI_FIX.md",
        ROOT / "V243_TEST_PLAN.md",
        ROOT / "DELTA_README_V243.md",
        ROOT / "GAME_DEVELOPMENT.md",
        ROOT / "ROADMAP.md",
        ROOT / "main.py",
    ]
    check("required V2.24.3 files exist", all(p.exists() for p in required))

    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    check("runtime installer wired in main.py", "install_v243_runtime(App)" in main_source)

    runtime = (ROOT / "src/engine/shot_object_local_v243.py").read_text(encoding="utf-8")
    check("runtime patches HitScanner ROI", "_v243_object_roi_patch" in runtime)
    check("runtime preserves global rescue", "GLOBAL-RESCUE-ROI" in runtime)
    check("runtime has zero-overlap recovery", "ROI-RECOVERY" in runtime)

    menu = ROOT / "content/menu.json"
    check("content/menu.json exists", menu.exists())
    entry = _find_entry(json.loads(menu.read_text(encoding="utf-8")))
    check("Hit Context Test menu entry exists", entry is not None)
    check("menu labels testscene V2.24.3 after prepare", "V2.24.3" in str(entry.get("title", "")))

    print("\nV2.24.3 installation verification passed.")


if __name__ == "__main__":
    main()
