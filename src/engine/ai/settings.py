from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AI_DIR = Path("content/ai")
AI_SETTINGS_PATH = AI_DIR / "settings.json"
AI_MODEL_PATH = AI_DIR / "model.json"
AI_SESSION_DIR = AI_DIR / "sessions"


def _default_settings() -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "train_only",  # off | train_only | advisory | blended | ai_priority | ai_only
        "blend_percent": 0.0,
        "min_confidence": 0.58,
        "override_confidence": 0.92,
        "max_positive_memories": 256,
        "max_negative_memories": 384,
        "auto_learn": True,
        "show_overlay": True,
        "top_k": 5,
        "click_match_radius_px": 36.0,
    }


def _ensure_dir() -> None:
    AI_DIR.mkdir(parents=True, exist_ok=True)
    AI_SESSION_DIR.mkdir(parents=True, exist_ok=True)


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
    merged["enabled"] = bool(merged.get("enabled", True))
    mode = str(merged.get("mode", "train_only")).strip().lower()
    if mode not in {"off", "train_only", "advisory", "blended", "ai_priority", "ai_only"}:
        mode = "train_only"
    merged["mode"] = mode
    try:
        merged["blend_percent"] = max(0.0, min(100.0, float(merged.get("blend_percent", 0.0))))
    except Exception:
        merged["blend_percent"] = 0.0
    try:
        merged["min_confidence"] = max(0.0, min(1.0, float(merged.get("min_confidence", 0.58))))
    except Exception:
        merged["min_confidence"] = 0.58
    try:
        merged["override_confidence"] = max(0.0, min(1.0, float(merged.get("override_confidence", 0.92))))
    except Exception:
        merged["override_confidence"] = 0.92
    merged["max_positive_memories"] = max(16, int(merged.get("max_positive_memories", 256)))
    merged["max_negative_memories"] = max(16, int(merged.get("max_negative_memories", 384)))
    merged["auto_learn"] = bool(merged.get("auto_learn", True))
    merged["show_overlay"] = bool(merged.get("show_overlay", True))
    merged["top_k"] = max(1, min(10, int(merged.get("top_k", 5))))
    try:
        merged["click_match_radius_px"] = max(4.0, min(200.0, float(merged.get("click_match_radius_px", 36.0))))
    except Exception:
        merged["click_match_radius_px"] = 36.0
    return merged


def save_ai_settings(settings: dict[str, Any]) -> dict[str, Any]:
    _ensure_dir()
    merged = load_ai_settings()
    merged.update(settings)
    merged = load_ai_settings() | merged
    AI_SETTINGS_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


# Small convenience helpers -------------------------------------------------

def load_ai_enabled() -> bool:
    return bool(load_ai_settings().get("enabled", True))


def save_ai_enabled(value: bool) -> None:
    save_ai_settings({"enabled": bool(value)})


def load_ai_mode() -> str:
    return str(load_ai_settings().get("mode", "train_only"))


def save_ai_mode(value: str) -> None:
    save_ai_settings({"mode": str(value)})


def load_ai_blend_percent() -> float:
    return float(load_ai_settings().get("blend_percent", 0.0))


def save_ai_blend_percent(value: float) -> None:
    save_ai_settings({"blend_percent": float(value)})


def load_ai_min_confidence() -> float:
    return float(load_ai_settings().get("min_confidence", 0.58))


def save_ai_min_confidence(value: float) -> None:
    save_ai_settings({"min_confidence": float(value)})


def load_ai_override_confidence() -> float:
    return float(load_ai_settings().get("override_confidence", 0.92))


def save_ai_override_confidence(value: float) -> None:
    save_ai_settings({"override_confidence": float(value)})
