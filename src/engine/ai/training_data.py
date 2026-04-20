from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .settings import AI_SESSION_DIR


def save_training_example(payload: dict[str, Any]) -> Path:
    AI_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = AI_SESSION_DIR / f"session_{stamp}_{int(time.time() * 1000) % 100000}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
