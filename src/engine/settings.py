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
# Helpers
# --------------------------------------------------------------------


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


def _sanitize_content_rect(rect: pygame.Rect) -> pygame.Rect:
    min_w, min_h = 1, 1
    rect.w = max(min_w, rect.w)
    rect.h = max(min_h, rect.h)
    rect.x = max(0, rect.x)
    rect.y = max(0, rect.y)
    return rect


def _parse_bool(value, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    return fallback


def _parse_str(value, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value)


def _parse_float(value, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _parse_int(value, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


# --------------------------------------------------------------------
# Viewport
# --------------------------------------------------------------------


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


# --------------------------------------------------------------------
# Scanport
# --------------------------------------------------------------------


def load_scanport_rect() -> pygame.Rect | None:
    data = _load_settings_dict()
    rect = _rect_from_value(data.get("scanport"))
    if rect is None:
        rect = _rect_from_value(data.get("scanport_rect"))
    if rect is None:
        return None
    return _sanitize_scanport(rect)


def save_scanport_rect(rect: pygame.Rect) -> None:
    data = _load_settings_dict()
    sanitized = _rect_to_list(_sanitize_scanport(rect.copy()))
    data["scanport"] = sanitized
    data["scanport_rect"] = sanitized
    _save_settings_dict(data)


# --------------------------------------------------------------------
# Content rect
# --------------------------------------------------------------------


def load_content_rect() -> pygame.Rect:
    rect = _rect_from_value(_load_settings_dict().get("content_rect"))
    if rect is not None:
        return _sanitize_content_rect(rect)
    return load_viewport_rect().copy()


def save_content_rect(rect: pygame.Rect) -> None:
    data = _load_settings_dict()
    data["content_rect"] = _rect_to_list(_sanitize_content_rect(rect.copy()))
    _save_settings_dict(data)


def clear_content_rect() -> None:
    data = _load_settings_dict()
    if "content_rect" in data:
        del data["content_rect"]
    _save_settings_dict(data)


# --------------------------------------------------------------------
# Camera calibration
# --------------------------------------------------------------------


def load_camera_calibration() -> dict | None:
    calibration = _load_settings_dict().get("camera_calibration")
    if isinstance(calibration, dict):
        return calibration
    return None


def save_camera_calibration(calibration: dict) -> None:
    data = _load_settings_dict()
    data["camera_calibration"] = calibration
    _save_settings_dict(data)


# --------------------------------------------------------------------
# Camera transform settings
# --------------------------------------------------------------------


def _default_camera_transform_settings() -> dict:
    return {
        "rotation": 0,
        "mirror_horizontal": False,
        "mirror_vertical": False,
    }


def _sanitize_camera_rotation(value) -> int:
    rotation = _parse_int(value, 0)
    if rotation not in (0, 90, 180, 270):
        rotation = 0
    return rotation


def load_camera_transform_settings() -> dict:
    data = _load_settings_dict()
    raw = data.get("camera", {})
    defaults = _default_camera_transform_settings()
    if not isinstance(raw, dict):
        return defaults.copy()

    merged = defaults.copy()
    merged.update(raw)
    merged["rotation"] = _sanitize_camera_rotation(merged.get("rotation", 0))
    merged["mirror_horizontal"] = _parse_bool(merged.get("mirror_horizontal"), False)
    merged["mirror_vertical"] = _parse_bool(merged.get("mirror_vertical"), False)
    return merged


def save_camera_transform_settings(settings: dict) -> None:
    data = _load_settings_dict()
    current = load_camera_transform_settings()
    current.update(settings)
    current["rotation"] = _sanitize_camera_rotation(current.get("rotation", 0))
    current["mirror_horizontal"] = bool(current.get("mirror_horizontal", False))
    current["mirror_vertical"] = bool(current.get("mirror_vertical", False))
    data["camera"] = current
    _save_settings_dict(data)


def load_camera_rotation() -> int:
    return int(load_camera_transform_settings().get("rotation", 0))


def save_camera_rotation(rotation: int) -> None:
    save_camera_transform_settings({"rotation": rotation})


def load_camera_mirror_horizontal() -> bool:
    return bool(load_camera_transform_settings().get("mirror_horizontal", False))


def save_camera_mirror_horizontal(enabled: bool) -> None:
    save_camera_transform_settings({"mirror_horizontal": bool(enabled)})


def load_camera_mirror_vertical() -> bool:
    return bool(load_camera_transform_settings().get("mirror_vertical", False))


def save_camera_mirror_vertical(enabled: bool) -> None:
    save_camera_transform_settings({"mirror_vertical": bool(enabled)})


# --------------------------------------------------------------------
# Visual hits settings
# --------------------------------------------------------------------


def _default_visual_hits_dict() -> dict:
    return {
        "enabled": True,
        "mode": "fade",
        "lifetime_ms": 900,
        "radius": 18,
        "show_all_planes": False,
    }


def load_visual_hits_settings() -> dict:
    data = _load_settings_dict()
    value = data.get("visual_hits")
    defaults = _default_visual_hits_dict()
    if not isinstance(value, dict):
        return defaults.copy()
    merged = defaults.copy()
    merged.update(value)
    return merged


def save_visual_hits_settings(settings: dict) -> None:
    data = _load_settings_dict()
    current = load_visual_hits_settings()
    current.update(settings)
    data["visual_hits"] = current
    _save_settings_dict(data)


def load_visual_hits_enabled() -> bool:
    return bool(load_visual_hits_settings().get("enabled", True))


def save_visual_hits_enabled(enabled: bool) -> None:
    save_visual_hits_settings({"enabled": bool(enabled)})


def load_visual_hits_mode() -> str:
    mode = str(load_visual_hits_settings().get("mode", "fade")).strip().lower()
    if mode not in ("fade", "persistent"):
        mode = "fade"
    return mode


def save_visual_hits_mode(mode: str) -> None:
    mode = str(mode).strip().lower()
    if mode not in ("fade", "persistent"):
        mode = "fade"
    save_visual_hits_settings({"mode": mode})


def load_visual_hits_lifetime_ms() -> int:
    try:
        value = int(load_visual_hits_settings().get("lifetime_ms", 900))
    except Exception:
        value = 900
    return max(0, value)


def save_visual_hits_lifetime_ms(lifetime_ms: int) -> None:
    save_visual_hits_settings({"lifetime_ms": max(0, int(lifetime_ms))})


def load_visual_hits_radius() -> int:
    try:
        value = int(load_visual_hits_settings().get("radius", 18))
    except Exception:
        value = 18
    return max(1, value)


def save_visual_hits_radius(radius: int) -> None:
    save_visual_hits_settings({"radius": max(1, int(radius))})


def load_visual_hits_show_all_planes() -> bool:
    return bool(load_visual_hits_settings().get("show_all_planes", False))


def save_visual_hits_show_all_planes(enabled: bool) -> None:
    save_visual_hits_settings({"show_all_planes": bool(enabled)})


# --------------------------------------------------------------------
# Scanner debug overlay settings
# --------------------------------------------------------------------


def _default_scanner_debug_dict() -> dict:
    return {"enabled": False}


def load_scanner_debug_overlay_settings() -> dict:
    data = _load_settings_dict()
    value = data.get("scanner_debug_overlay")
    defaults = _default_scanner_debug_dict()
    if not isinstance(value, dict):
        return defaults.copy()
    merged = defaults.copy()
    merged.update(value)
    return merged


def save_scanner_debug_overlay_settings(settings: dict) -> None:
    data = _load_settings_dict()
    current = load_scanner_debug_overlay_settings()
    current.update(settings)
    data["scanner_debug_overlay"] = current
    _save_settings_dict(data)


def load_scanner_debug_overlay_enabled() -> bool:
    return bool(load_scanner_debug_overlay_settings().get("enabled", False))


def save_scanner_debug_overlay_enabled(enabled: bool) -> None:
    save_scanner_debug_overlay_settings({"enabled": bool(enabled)})


# --------------------------------------------------------------------
# Audio peak settings
# --------------------------------------------------------------------


def _default_audio_peak_dict() -> dict:
    return {
        "threshold": 0.10,
        "show_status_overlay": True,
    }


def load_audio_peak_settings() -> dict:
    data = _load_settings_dict()
    value = data.get("audio_peak")
    defaults = _default_audio_peak_dict()

    if not isinstance(value, dict):
        merged = defaults.copy()
        if "audio_peak_threshold" in data:
            try:
                merged["threshold"] = float(data["audio_peak_threshold"])
            except Exception:
                pass
        return merged

    merged = defaults.copy()
    merged.update(value)
    return merged


def save_audio_peak_settings(settings: dict) -> None:
    data = _load_settings_dict()
    current = load_audio_peak_settings()
    current.update(settings)
    data["audio_peak"] = current
    if "threshold" in current:
        data["audio_peak_threshold"] = float(current["threshold"])
    _save_settings_dict(data)


def load_audio_peak_threshold() -> float:
    try:
        value = float(load_audio_peak_settings().get("threshold", 0.10))
    except Exception:
        value = 0.10
    return max(0.005, min(0.95, value))


def save_audio_peak_threshold(threshold: float) -> None:
    save_audio_peak_settings({"threshold": max(0.005, min(0.95, float(threshold)))})


def load_audio_status_overlay_enabled() -> bool:
    return bool(load_audio_peak_settings().get("show_status_overlay", True))


def save_audio_status_overlay_enabled(enabled: bool) -> None:
    save_audio_peak_settings({"show_status_overlay": bool(enabled)})


# --------------------------------------------------------------------
# Range projection settings
# --------------------------------------------------------------------


def _default_range_projection_settings() -> dict:
    return {
        "wall_distance_m": 6.0,
        "viewport_physical_width_cm": 100.0,
        "viewport_physical_height_cm": 50.0,
        "viewport_bottom_world_cm": 105.0,
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
    try:
        value = float(load_range_projection_settings().get("wall_distance_m", 6.0))
    except Exception:
        value = 6.0
    return max(0.1, value)


def save_wall_distance_m(value: float) -> None:
    save_range_projection_settings({"wall_distance_m": max(0.1, float(value))})


def load_viewport_physical_width_cm() -> float:
    try:
        value = float(load_range_projection_settings().get("viewport_physical_width_cm", 100.0))
    except Exception:
        value = 100.0
    return max(1.0, value)


def save_viewport_physical_width_cm(value: float) -> None:
    save_range_projection_settings({"viewport_physical_width_cm": max(1.0, float(value))})


def load_viewport_physical_height_cm() -> float:
    try:
        value = float(load_range_projection_settings().get("viewport_physical_height_cm", 50.0))
    except Exception:
        value = 50.0
    return max(1.0, value)


def save_viewport_physical_height_cm(value: float) -> None:
    save_range_projection_settings({"viewport_physical_height_cm": max(1.0, float(value))})


def load_viewport_bottom_world_cm() -> float:
    try:
        value = float(load_range_projection_settings().get("viewport_bottom_world_cm", 105.0))
    except Exception:
        value = 105.0
    return max(-500.0, min(500.0, value))


def save_viewport_bottom_world_cm(value: float) -> None:
    save_range_projection_settings(
        {"viewport_bottom_world_cm": max(-500.0, min(500.0, float(value)))}
    )


# --------------------------------------------------------------------
# LED settings
# --------------------------------------------------------------------


def _default_led_settings() -> dict:
    return {
        "enabled": False,
        "device_id": "",
        "ip_address": "",
        "local_key": "",
        "version": 3.3,
        "default_mode": "colour",
        "default_brightness": 1000,
        "default_temperature": 500,
        "default_colour": [255, 255, 255],
    }


def load_led_settings() -> dict:
    data = _load_settings_dict()
    raw = data.get("led")
    defaults = _default_led_settings()
    if not isinstance(raw, dict):
        return defaults.copy()

    merged = defaults.copy()
    merged.update(raw)
    merged["enabled"] = _parse_bool(merged.get("enabled"), False)
    merged["device_id"] = _parse_str(merged.get("device_id"), "")
    merged["ip_address"] = _parse_str(merged.get("ip_address"), "")
    merged["local_key"] = _parse_str(merged.get("local_key"), "")

    # För SH-LS3M ska vi låsa till 3.3
    merged["version"] = 3.3

    try:
        brightness = int(merged.get("default_brightness", 1000))
    except Exception:
        brightness = 1000
    merged["default_brightness"] = max(10, min(1000, brightness))

    try:
        temperature = int(merged.get("default_temperature", 500))
    except Exception:
        temperature = 500
    merged["default_temperature"] = max(0, min(1000, temperature))

    color = merged.get("default_colour", [255, 255, 255])
    if not (isinstance(color, list) and len(color) == 3):
        color = [255, 255, 255]
    merged["default_colour"] = [
        max(0, min(255, int(color[0]))),
        max(0, min(255, int(color[1]))),
        max(0, min(255, int(color[2]))),
    ]

    mode = str(merged.get("default_mode", "colour")).strip().lower()
    if mode not in ("off", "white", "colour"):
        mode = "colour"
    merged["default_mode"] = mode
    return merged


def save_led_settings(settings: dict) -> None:
    data = _load_settings_dict()
    current = load_led_settings()
    current.update(settings)
    current["version"] = 3.3
    data["led"] = current
    _save_settings_dict(data)


def load_led_enabled() -> bool:
    return bool(load_led_settings().get("enabled", False))


def save_led_enabled(enabled: bool) -> None:
    save_led_settings({"enabled": bool(enabled)})


def load_led_device_id() -> str:
    return str(load_led_settings().get("device_id", ""))


def save_led_device_id(value: str) -> None:
    save_led_settings({"device_id": str(value)})


def load_led_ip_address() -> str:
    return str(load_led_settings().get("ip_address", ""))


def save_led_ip_address(value: str) -> None:
    save_led_settings({"ip_address": str(value)})


def load_led_local_key() -> str:
    return str(load_led_settings().get("local_key", ""))


def save_led_local_key(value: str) -> None:
    save_led_settings({"local_key": str(value)})


def load_led_version() -> float:
    return 3.3


def save_led_version(value: float) -> None:
    del value
    save_led_settings({"version": 3.3})
