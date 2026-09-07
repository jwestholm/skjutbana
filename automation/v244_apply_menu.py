from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MENU_PATH = ROOT / "content/menu.json"
OLD_TITLE = '"title": "Hit Context Test (V2.24.3)"'
NEW_TITLE = '"title": "Hit Context Test (V2.24.4)"'
OLD_DESC = '"description": "Diagnostik för HitRegions V2.24.3: lokal ROI, rörligt mål, overlap, kant, outside-region och global rescue."'
NEW_DESC = '"description": "Diagnostik för HitRegions V2.24.4: full-camera till detector working-space ROI, rörligt mål, overlap, kant och global rescue."'


def main() -> None:
    if not MENU_PATH.exists():
        raise FileNotFoundError(MENU_PATH)
    text = MENU_PATH.read_text(encoding="utf-8")
    if NEW_TITLE in text:
        print("[OK] content/menu.json already labels Hit Context Test as V2.24.4")
        return

    from automation.v243_apply_menu import main as apply_v243_menu
    apply_v243_menu()
    text = MENU_PATH.read_text(encoding="utf-8")
    if OLD_TITLE not in text:
        raise RuntimeError("V2.24.3 Hit Context Test menu entry was not found")
    patched = text.replace(OLD_TITLE, NEW_TITLE, 1)
    if OLD_DESC in patched:
        patched = patched.replace(OLD_DESC, NEW_DESC, 1)
    MENU_PATH.write_text(patched, encoding="utf-8")
    print("[PATCH] content/menu.json -> Hit Context Test title updated to V2.24.4")


if __name__ == "__main__":
    main()
