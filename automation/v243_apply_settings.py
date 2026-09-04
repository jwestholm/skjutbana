from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "src/engine/settings.py"

OLD = '''def load_content_rect() -> pygame.Rect:\n    rect = _rect_from_value(_load_settings_dict().get("content_rect"))\n    if rect is not None:\n        return _sanitize_content_rect(rect)\n    return load_viewport_rect().copy()\n'''

NEW = '''def load_content_rect() -> pygame.Rect:\n    rect = _rect_from_value(_load_settings_dict().get("content_rect"))\n    if rect is not None:\n        return _sanitize_content_rect(rect)\n    # content_rect is viewport-local throughout the hit/input pipeline.\n    # Defaulting to viewport.copy() incorrectly carries absolute viewport x/y\n    # and HitScanner then adds viewport.x/y a second time.\n    viewport = load_viewport_rect()\n    return pygame.Rect(0, 0, viewport.w, viewport.h)\n'''


def patch_text(text: str) -> tuple[str, bool]:
    if NEW in text:
        return text, False
    if OLD not in text:
        raise RuntimeError(
            "Could not find the expected load_content_rect() default block. "
            "Refusing to guess-edit src/engine/settings.py."
        )
    return text.replace(OLD, NEW, 1), True


def main() -> None:
    if not SETTINGS.exists():
        raise FileNotFoundError(SETTINGS)
    text = SETTINGS.read_text(encoding="utf-8")
    patched, changed = patch_text(text)
    if not changed:
        print("[OK] src/engine/settings.py already uses viewport-local default content_rect")
        return
    SETTINGS.write_text(patched, encoding="utf-8")
    print("[PATCH] src/engine/settings.py -> default content_rect is now viewport-local (0,0,w,h)")


if __name__ == "__main__":
    main()
