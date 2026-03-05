from __future__ import annotations
import json
from pathlib import Path
import pygame

import config


def _clamp_viewport(rect: pygame.Rect) -> pygame.Rect:
    # Minstorlek så man inte råkar göra den 0
    min_w, min_h = 200, 200
    rect.w = max(min_w, rect.w)
    rect.h = max(min_h, rect.h)

    # håll inom skärmen
    rect.x = max(0, min(rect.x, config.SCREEN_WIDTH - rect.w))
    rect.y = max(0, min(rect.y, config.SCREEN_HEIGHT - rect.h))
    return rect


def load_viewport_rect() -> pygame.Rect:
    path = Path(config.SETTINGS_PATH)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            vp = data.get("viewport", None)
            if isinstance(vp, list) and len(vp) == 4:
                rect = pygame.Rect(int(vp[0]), int(vp[1]), int(vp[2]), int(vp[3]))
                return _clamp_viewport(rect)
        except Exception:
            pass

    x, y, w, h = config.DEFAULT_VIEWPORT
    return _clamp_viewport(pygame.Rect(x, y, w, h))


def save_viewport_rect(rect: pygame.Rect) -> None:
    rect = _clamp_viewport(rect.copy())
    path = Path(config.SETTINGS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"viewport": [rect.x, rect.y, rect.w, rect.h]}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")