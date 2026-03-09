from __future__ import annotations

import json
from pathlib import Path

import pygame

import config


def _settings_path() -> Path:
    settings_path = getattr(config, "SETTINGS_PATH", "content/settings.json")
    path = Path(settings_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_settings_data() -> dict:
    path = _settings_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save_settings_data(data: dict) -> None:
    path = _settings_path()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _clamp_viewport(rect: pygame.Rect) -> pygame.Rect:
    min_w, min_h = 200, 200
    rect.w = max(min_w, rect.w)
    rect.h = max(min_h, rect.h)

    rect.x = max(0, min(rect.x, config.SCREEN_WIDTH - rect.w))
    rect.y = max(0, min(rect.y, config.SCREEN_HEIGHT - rect.h))
    return rect


def _clamp_scanport_norm(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    min_w = 100 / max(1, config.SCREEN_WIDTH)
    min_h = 100 / max(1, config.SCREEN_HEIGHT)

    w = max(min_w, min(1.0, float(w)))
    h = max(min_h, min(1.0, float(h)))

    x = max(0.0, min(float(x), 1.0 - w))
    y = max(0.0, min(float(y), 1.0 - h))

    return (x, y, w, h)


def load_viewport_rect() -> pygame.Rect:
    data = _load_settings_data()
    vp = data.get("viewport", None)

    if isinstance(vp, list) and len(vp) == 4:
        try:
            rect = pygame.Rect(int(vp[0]), int(vp[1]), int(vp[2]), int(vp[3]))
            return _clamp_viewport(rect)
        except Exception:
            pass

    x, y, w, h = config.DEFAULT_VIEWPORT
    return _clamp_viewport(pygame.Rect(x, y, w, h))


def save_viewport_rect(rect: pygame.Rect) -> None:
    rect = _clamp_viewport(rect.copy())
    data = _load_settings_data()
    data["viewport"] = [rect.x, rect.y, rect.w, rect.h]
    _save_settings_data(data)


def load_scanport_norm() -> tuple[float, float, float, float]:
    data = _load_settings_data()
    sp = data.get("scanport", None)

    if isinstance(sp, list) and len(sp) == 4:
        try:
            return _clamp_scanport_norm(
                float(sp[0]),
                float(sp[1]),
                float(sp[2]),
                float(sp[3]),
            )
        except Exception:
            pass

    return _clamp_scanport_norm(0.10, 0.10, 0.80, 0.80)


def save_scanport_norm(x: float, y: float, w: float, h: float) -> None:
    x, y, w, h = _clamp_scanport_norm(x, y, w, h)
    data = _load_settings_data()
    data["scanport"] = [x, y, w, h]
    _save_settings_data(data)


def scanport_norm_to_screen_rect(
    scanport: tuple[float, float, float, float],
) -> pygame.Rect:
    x, y, w, h = scanport
    rect = pygame.Rect(
        int(round(x * config.SCREEN_WIDTH)),
        int(round(y * config.SCREEN_HEIGHT)),
        int(round(w * config.SCREEN_WIDTH)),
        int(round(h * config.SCREEN_HEIGHT)),
    )
    return _clamp_viewport(rect)


def screen_rect_to_scanport_norm(rect: pygame.Rect) -> tuple[float, float, float, float]:
    x = rect.x / max(1, config.SCREEN_WIDTH)
    y = rect.y / max(1, config.SCREEN_HEIGHT)
    w = rect.w / max(1, config.SCREEN_WIDTH)
    h = rect.h / max(1, config.SCREEN_HEIGHT)
    return _clamp_scanport_norm(x, y, w, h)


def scanport_norm_to_frame_rect(
    scanport: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> pygame.Rect:
    x, y, w, h = scanport
    rect = pygame.Rect(
        int(round(x * frame_width)),
        int(round(y * frame_height)),
        int(round(w * frame_width)),
        int(round(h * frame_height)),
    )

    min_w, min_h = 20, 20
    rect.w = max(min_w, min(rect.w, frame_width))
    rect.h = max(min_h, min(rect.h, frame_height))
    rect.x = max(0, min(rect.x, frame_width - rect.w))
    rect.y = max(0, min(rect.y, frame_height - rect.h))
    return rect