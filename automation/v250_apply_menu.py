from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MENU_PATH = ROOT / "content/menu.json"
ENTRY_PATH = ROOT / "menu_games_entry_v250.json"
ENTRY_ID = "game_objects_test_v250"


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def patch_menu_data(data: Any, entry: dict[str, Any]) -> tuple[Any, bool, bool]:
    result = deepcopy(data)
    for node in _walk(result):
        if str(node.get("id", "")) != "games":
            continue
        children = node.get("children")
        if not isinstance(children, list):
            continue
        for existing in children:
            if isinstance(existing, dict) and str(existing.get("id", "")) == ENTRY_ID:
                return result, False, True
        children.append(deepcopy(entry))
        return result, True, True
    return result, False, False


def _matching_array_end(text: str, open_index: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for i in range(open_index, len(text)):
        ch = text[i]
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
        elif ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unterminated games children array")


def patch_menu_text(text: str, entry: dict[str, Any]) -> tuple[str, bool, bool]:
    data = json.loads(text)
    _, structural_changed, found = patch_menu_data(data, entry)
    if not found:
        return text, False, False
    if not structural_changed:
        return text, False, True

    games_match = re.search(r'"id"\s*:\s*"games"', text)
    if games_match is None:
        return text, False, False
    children_match = re.search(r'"children"\s*:\s*\[', text[games_match.end():])
    if children_match is None:
        return text, False, False
    open_index = games_match.end() + children_match.end() - 1
    close_index = _matching_array_end(text, open_index)

    line_start = text.rfind("\n", 0, close_index) + 1
    close_indent = text[line_start:close_index]
    if close_indent.strip():
        close_indent = re.match(r"[ \t]*", close_indent).group(0)

    body = text[open_index + 1:close_index]
    item_indent = None
    for line in body.splitlines():
        if line.strip():
            item_indent = re.match(r"[ \t]*", line).group(0)
            break
    if item_indent is None:
        item_indent = close_indent + "  "

    raw_entry = json.dumps(entry, ensure_ascii=False, indent=2)
    indented_entry = "\n".join(item_indent + line for line in raw_entry.splitlines())
    core = body.rstrip()
    if core.strip():
        new_body = core + ",\n" + indented_entry + "\n" + close_indent
    else:
        new_body = "\n" + indented_entry + "\n" + close_indent
    patched = text[:open_index + 1] + new_body + text[close_index:]
    json.loads(patched)
    return patched, True, True


def main() -> None:
    if not MENU_PATH.exists():
        raise FileNotFoundError(MENU_PATH)
    if not ENTRY_PATH.exists():
        raise FileNotFoundError(ENTRY_PATH)

    # Ensure the V2.24 diagnostic entry remains installed/labeled first.
    try:
        from automation.v244_apply_menu import main as apply_v244_menu
        apply_v244_menu()
    except Exception as exc:
        print(f"[WARN] V2.24.4 menu preparation could not run: {exc}")

    text = MENU_PATH.read_text(encoding="utf-8")
    entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    patched, changed, found = patch_menu_text(text, entry)
    if not found:
        raise RuntimeError('Could not find menu folder with id="games" and a children list')
    if not changed:
        print("[OK] content/menu.json already contains Game Objects Test (V2.25.0)")
        return
    MENU_PATH.write_text(patched, encoding="utf-8")
    print("[PATCH] content/menu.json -> added Game Objects Test (V2.25.0) without reformatting existing menu")


if __name__ == "__main__":
    main()
