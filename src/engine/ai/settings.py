"""
AI settings — convenience helpers for reading/writing content/ai/settings.json.

The canonical settings live in AIRuntime.settings (runtime.py).
These helpers exist for backward compatibility and for code that
needs to read settings without instantiating the full runtime.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AI_DIR = Path("content/ai")
AI_SETTINGS_PATH = AI_DIR / "settings.json"
AI_MODEL_PATH = AI_DIR / "model.json"  # Legacy path, kept for compat
AI_SESSION_DIR = AI_DIR / "sessions"


def _default_settings() -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "train_only",
        "top_k": 10,
        "memory_limit_positive": 400,
        "memory_limit_negative": 1200,
        "click_match_radius_px": 42.0,
        "min_confidence": 0.58,
        "override_confidence": 0.92,
        "max_negatives_per_click": 3,
        "trust_percent": 0,
        "show_overlay": True,
        "auto_learn": True,
    }


def _ensure_dir() -> None:
    AI_DIR.mkdir(parents=True, exist_ok=True)


def load_ai_settings() -> dict[str, Any]:
    _ensure_dir()
    defaults = _default_settings()
    if not AI_SETTINGS_PATH.exists():
        return defaults.copy()
    try:
        raw = json.loads(AI_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return defaults.copy()
    if not isinstance(raw, dict):
        return defaults.copy()
    merged = defaults.copy()
    merged.update(raw)
    return merged


def save_ai_settings(settings: dict[str, Any]) -> dict[str, Any]:
    _ensure_dir()
    merged = load_ai_settings()
    merged.update(settings)
    AI_SETTINGS_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def load_ai_enabled() -> bool:
    return bool(load_ai_settings().get("enabled", True))


def load_ai_mode() -> str:
    return str(load_ai_settings().get("mode", "train_only"))
