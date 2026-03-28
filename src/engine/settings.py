from __future__ import annotations

import json
from pathlib import Path

import pygame

import config


# --------------------------------------------------------------------
# Core IO
# --------------------------------------------------------------------

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


# --------------------------------------------------------------------
# Viewport
# --------------------------------------------------------------------

def _clamp_rect(rect: pygame.Rect) -> pygame.Rect:
    min_w, min_h = 50, 50

    rect.w = max(min_w, rect.w)
    rect.h = max(min_h, rect.h)
    rect.x = max(0, min(rect.x, config.SCREEN_WIDTH - rect.w))
    rect.y = max(0, min(rect.y, config.SCREEN_HEIGHT - rect.h))
    return rect


def load_viewport_rect() -> pygame.Rect:
    data = _load_settings_dict()
    vp = data.get("viewport")

    if isinstance(vp, list) and len(vp) == 4:
        try:
            return _clamp_rect(pygame.Rect(*map(int, vp)))
        except Exception:
            pass

    x, y, w, h = config.DEFAULT_VIEWPORT
    return _clamp_rect(pygame.Rect(x, y, w, h))


def save_viewport_rect(rect: pygame.Rect) -> None:
    rect = _clamp_rect(rect.copy())
    data = _load_settings_dict()
    data["viewport"] = [rect.x, rect.y, rect.w, rect.h]
    _save_settings_dict(data)


# --------------------------------------------------------------------
# Content Rect (NY – fixar ditt fel)
# --------------------------------------------------------------------

def load_content_rect() -> pygame.Rect:
    data = _load_settings_dict()
    cr = data.get("content_rect")

    if isinstance(cr, list) and len(cr) == 4:
        try:
            return _clamp_rect(pygame.Rect(*map(int, cr)))
        except Exception:
            pass

    # fallback = hela skärmen
    return pygame.Rect(0, 0, config.SCREEN_WIDTH, config.SCREEN_HEIGHT)


def save_content_rect(rect: pygame.Rect) -> None:
    rect = _clamp_rect(rect.copy())
    data = _load_settings_dict()
    data["content_rect"] = [rect.x, rect.y, rect.w, rect.h]
    _save_settings_dict(data)


# --------------------------------------------------------------------
# Audio Peak
# --------------------------------------------------------------------

def load_audio_peak_threshold() -> float:
    data = _load_settings_dict()
    return float(data.get("audio_peak_threshold", 0.12))


def save_audio_peak_threshold(value: float) -> None:
    data = _load_settings_dict()
    data["audio_peak_threshold"] = float(value)
    _save_settings_dict(data)


# --------------------------------------------------------------------
# Camera Calibration
# --------------------------------------------------------------------

def load_camera_calibration() -> dict:
    data = _load_settings_dict()
    return data.get("camera_calibration", {})


def save_camera_calibration(calibration: dict) -> None:
    data = _load_settings_dict()
    data["camera_calibration"] = calibration
    _save_settings_dict(data)


# --------------------------------------------------------------------
# Range Projection
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
    return float(load_range_projection_settings().get("wall_distance_m", 6.0))


def save_wall_distance_m(value: float) -> None:
    save_range_projection_settings({"wall_distance_m": float(value)})


def load_viewport_physical_width_cm() -> float:
    return float(load_range_projection_settings().get("viewport_physical_width_cm", 100.0))


def save_viewport_physical_width_cm(value: float) -> None:
    save_range_projection_settings({"viewport_physical_width_cm": float(value)})


def load_viewport_physical_height_cm() -> float:
    return float(load_range_projection_settings().get("viewport_physical_height_cm", 50.0))


def save_viewport_physical_height_cm(value: float) -> None:
    save_range_projection_settings({"viewport_physical_height_cm": float(value)})