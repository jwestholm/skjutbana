from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def main() -> None:
    print("V2.24.2 SELFTEST")
    print("===============")

    scene_path = ROOT / "content/games/hit_context_test_v242.py"
    source = scene_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    check("testscene parses as Python", tree is not None)
    check("testscene implements create_game", "create_game" in names)
    check("testscene exposes get_hit_regions", "get_hit_regions" in names)
    check("testscene consumes normal HitEvent XY", "hit.game_x" in source and "hit.game_y" in source)
    check("moving-object validation reads frozen snapshot", "latest_hit_context_snapshot" in source and "game_regions" in source)
    check("EMPTY mode returns no HitRegions", "if self.empty_regions" in source and "return ()" in source)
    check("roles include target and no_shoot", '"target"' in source and '"no_shoot"' in source)
    check("scene never changes detector candidate coordinates", "camera_x =" not in source and "camera_y =" not in source)

    from automation.v242_apply_menu import patch_menu_data, patch_menu_text
    entry = json.loads((ROOT / "menu_games_entry_v242.json").read_text(encoding="utf-8"))
    fixture = {"children": [{"id": "games", "children": [{"id": "old_game"}]}]}
    patched, changed, found = patch_menu_data(fixture, entry)
    check("menu patch finds games folder", found)
    check("menu patch inserts testscene", changed and patched["children"][0]["children"][-1]["id"] == "hit_context_test_v242")
    patched2, changed2, found2 = patch_menu_data(patched, entry)
    check("menu patch is idempotent", found2 and not changed2 and patched2 == patched)

    fixture_text = '{\n  "id": "root",\n  "children": [\n    {"id": "games", "children": [\n      {"id": "old_game", "led": {"color": [1, 2, 3]}}\n    ]}\n  ]\n}\n'
    patched_text, text_changed, text_found = patch_menu_text(fixture_text, entry)
    check("text menu patch preserves existing prefix", text_found and text_changed and patched_text.startswith('{\n  "id": "root"'))
    check("text menu patch keeps existing game text", '{"id": "old_game", "led": {"color": [1, 2, 3]}}' in patched_text)
    patched_text2, text_changed2, _ = patch_menu_text(patched_text, entry)
    check("text menu patch is idempotent", not text_changed2 and patched_text2 == patched_text)

    # Carry forward the important V2.24.1 no-authority invariant by static check.
    local_source = (ROOT / "src/engine/shot_object_local_v241.py").read_text(encoding="utf-8")
    check("V2.24.1 global rescue bypass remains present", "reason=v2225_full_rescue" in local_source)
    check("V2.24.1 still consumes frozen camera regions", "camera_regions" in local_source)

    print("\nAll V2.24.2 selftests passed.")


if __name__ == "__main__":
    main()
