from __future__ import annotations

import json
from pathlib import Path

import pygame

import config


def _settings_path() -> Path:
    return Path(getattr(config, "SETTINGS_PATH", "content/settings.json"))


def _load_settings_dict() -> dict:
    path = _settings_path()
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return {}


def _save_settings_dict(data: dict) -> None:
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _clamp_viewport(rect: pygame.Rect) -> pygame.Rect:
    min_w, min_h = 200, 200

    rect.w = max(min_w, rect.w)
    rect.h = max(min_h, rect.h)

    rect.x = max(0, min(rect.x, config.SCREEN_WIDTH - rect.w))
    rect.y = max(0, min(rect.y, config.SCREEN_HEIGHT - rect.h))
    return rect


def _sanitize_scanport(rect: pygame.Rect) -> pygame.Rect:
    min_w, min_h = 50, 50

    rect.x = max(0, rect.x)
    rect.y = max(0, rect.y)
    rect.w = max(min_w, rect.w)
    rect.h = max(min_h, rect.h)
    return rect


def _rect_from_value(value) -> pygame.Rect | None:
    if isinstance(value, list) and len(value) == 4:
        try:
            return pygame.Rect(
                int(value[0]),
                int(value[1]),
                int(value[2]),
                int(value[3]),
            )
        except Exception:
            return None
    return None


def _rect_to_list(rect: pygame.Rect) -> list[int]:
    return [int(rect.x), int(rect.y), int(rect.w), int(rect.h)]


def load_viewport_rect() -> pygame.Rect:
    data = _load_settings_dict()
    rect = _rect_from_value(data.get("viewport"))
    if rect is not None:
        return _clamp_viewport(rect)

    x, y, w, h = config.DEFAULT_VIEWPORT
    return _clamp_viewport(pygame.Rect(x, y, w, h))


def save_viewport_rect(rect: pygame.Rect) -> None:
    data = _load_settings_dict()
    data["viewport"] = _rect_to_list(_clamp_viewport(rect.copy()))
    _save_settings_dict(data)


def load_scanport_rect() -> pygame.Rect | None:
    data = _load_settings_dict()
    rect = _rect_from_value(data.get("scanport"))
    if rect is None:
        return None
    return _sanitize_scanport(rect)


def save_scanport_rect(rect: pygame.Rect) -> None:
    data = _load_settings_dict()
    data["scanport"] = _rect_to_list(_sanitize_scanport(rect.copy()))
    _save_settings_dict(data)


def load_camera_calibration() -> dict | None:
    data = _load_settings_dict()
    calibration = data.get("camera_calibration")
    if isinstance(calibration, dict):
        return calibration
    return None


def save_camera_calibration(calibration: dict) -> None:
    data = _load_settings_dict()
    data["camera_calibration"] = calibration
    _save_settings_dict(data)