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


def load_viewport_rect() -> pygame.Rect:
    data = _load_settings_dict()
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
    data = _load_settings_dict()
    data["viewport"] = [rect.x, rect.y, rect.w, rect.h]
    _save_settings_dict(data)


# --------------------------------------------------------------------
# Audio peak settings
# --------------------------------------------------------------------

def load_audio_peak_threshold() -> float:
    data = _load_settings_dict()

    try:
        value = float(data.get("audio_peak_threshold", 0.12))
    except Exception:
        value = 0.12

    return max(0.005, min(0.95, value))


def save_audio_peak_threshold(value: float) -> None:
    data = _load_settings_dict()
    clamped = max(0.005, min(0.95, float(value)))
    data["audio_peak_threshold"] = clamped
    _save_settings_dict(data)


# --------------------------------------------------------------------
# Range / distance projection settings
# --------------------------------------------------------------------

def _default_range_projection_settings() -> dict:
    return {
        "wall_distance_m": 6.0,
        "viewport_physical_width_cm": 100.0,
        "viewport_physical_height_cm": 50.0,
    }


def load_range_projection_settings() -> dict:
    data = _load_settings_dict()
    raw = data.get("range_projection", {})
    defaults = _default_range_projection_settings()

    if not isinstance(raw, dict):
        return defaults.copy()

    merged = defaults.copy()
    merged.update(raw)
    return merged


def save_range_projection_settings(settings: dict) -> None:
    data = _load_settings_dict()
    current = load_range_projection_settings()
    current.update(settings)
    data["range_projection"] = current
    _save_settings_dict(data)


def load_wall_distance_m() -> float:
    settings = load_range_projection_settings()

    try:
        value = float(settings.get("wall_distance_m", 6.0))
    except Exception:
        value = 6.0

    return max(0.1, value)


def save_wall_distance_m(value: float) -> None:
    save_range_projection_settings({"wall_distance_m": max(0.1, float(value))})


def load_viewport_physical_width_cm() -> float:
    settings = load_range_projection_settings()

    try:
        value = float(settings.get("viewport_physical_width_cm", 100.0))
    except Exception:
        value = 100.0

    return max(1.0, value)


def save_viewport_physical_width_cm(value: float) -> None:
    save_range_projection_settings(
        {"viewport_physical_width_cm": max(1.0, float(value))}
    )


def load_viewport_physical_height_cm() -> float:
    settings = load_range_projection_settings()

    try:
        value = float(settings.get("viewport_physical_height_cm", 50.0))
    except Exception:
        value = 50.0

    return max(1.0, value)


def save_viewport_physical_height_cm(value: float) -> None:
    save_range_projection_settings(
        {"viewport_physical_height_cm": max(1.0, float(value))}
    )