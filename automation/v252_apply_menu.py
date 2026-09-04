from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MENU_PATH = ROOT / "content/menu.json"
ENTRY_PATH = ROOT / "menu_games_entry_v252.json"
ENTRY_ID = "game_objects_test_v250"


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def patch_menu_data(data: Any, entry: dict[str, Any]) -> tuple[Any, bool]:
    changed = False
    for node in _walk(data):
        if str(node.get("id", "")) != ENTRY_ID:
            continue
        for key, value in entry.items():
            if node.get(key) != value:
                node[key] = value
                changed = True
        return data, changed
    return data, False


def patch_menu_text(text: str, entry: dict[str, Any]) -> tuple[str, bool, bool]:
    """Update only the existing V2.25 test entry.

    JSON reformatting is deliberately avoided in the normal repository path by
    replacing scalar title/description fields inside the located object text.
    The structural helper remains available for selftests.
    """
    data = json.loads(text)
    found = any(str(node.get("id", "")) == ENTRY_ID for node in _walk(data))
    if not found:
        return text, False, False

    # Find the id and a conservative object range around it. The V2.25.0 menu
    # entry is a flat object (led is its only nested mapping), so brace matching
    # from the containing object is safe and preserves all unrelated formatting.
    marker = f'"id": "{ENTRY_ID}"'
    idx = text.find(marker)
    if idx < 0:
        marker = f'"id":"{ENTRY_ID}"'
        idx = text.find(marker)
    if idx < 0:
        # Fallback to structural JSON only if unusual formatting hides the token.
        patched_data, changed = patch_menu_data(data, entry)
        return (json.dumps(patched_data, ensure_ascii=False, indent=2) + "\n", changed, True)

    start = text.rfind("{", 0, idx)
    if start < 0:
        return text, False, False
    depth = 0
    in_string = False
    escaped = False
    end = None
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = pos + 1
                break
    if end is None:
        return text, False, False

    original_obj = text[start:end]
    obj = json.loads(original_obj)
    changed = False
    for key in ("title", "description", "script"):
        if key in entry and obj.get(key) != entry[key]:
            old_json = json.dumps(obj.get(key), ensure_ascii=False)
            new_json = json.dumps(entry[key], ensure_ascii=False)
            needle = f'"{key}": {old_json}'
            if needle not in original_obj:
                needle = f'"{key}":{old_json}'
            if needle in original_obj:
                replacement = needle.split(old_json)[0] + new_json
                original_obj = original_obj.replace(needle, replacement, 1)
                obj[key] = entry[key]
                changed = True
    patched = text[:start] + original_obj + text[end:]
    json.loads(patched)
    return patched, changed, True


def main() -> None:
    if not MENU_PATH.exists():
        raise FileNotFoundError(MENU_PATH)
    # Ensure the V2.25 entry exists first.
    try:
        from automation.v250_apply_menu import main as apply_v250_menu
        apply_v250_menu()
    except Exception as exc:
        print(f"[WARN] V2.25.0 menu preparation could not run: {exc}")
    entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    text = MENU_PATH.read_text(encoding="utf-8")
    patched, changed, found = patch_menu_text(text, entry)
    if not found:
        raise RuntimeError(f"Could not find {ENTRY_ID} after V2.25.0 menu preparation")
    if changed:
        MENU_PATH.write_text(patched, encoding="utf-8")
        print("[PATCH] content/menu.json -> Game Objects Test (V2.25.2)")
    else:
        print("[OK] content/menu.json already labels Game Objects Test (V2.25.2)")


if __name__ == "__main__":
    main()
