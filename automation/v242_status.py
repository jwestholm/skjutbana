from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def _contains(node) -> bool:
    if isinstance(node, dict):
        if node.get("id") == "hit_context_test_v242":
            return True
        return any(_contains(v) for v in node.values())
    if isinstance(node, list):
        return any(_contains(v) for v in node)
    return False


def main() -> None:
    print("V2.24.2 STATUS")
    print("===============")
    scene = ROOT / "content/games/hit_context_test_v242.py"
    menu = ROOT / "content/menu.json"
    print(f"Scene file: {'OK' if scene.exists() else 'MISSING'}")
    installed = False
    if menu.exists():
        try:
            installed = _contains(json.loads(menu.read_text(encoding='utf-8')))
        except Exception:
            pass
    print(f"Menu entry: {'OK' if installed else 'NOT PREPARED'}")
    print("Live hit authority: unchanged")
    print("Purpose: physical acceptance harness for V2.24.0/1")
    print("Next: V2.25.0 only after short physical V2.24.2 matrix is clean")


if __name__ == "__main__":
    main()
