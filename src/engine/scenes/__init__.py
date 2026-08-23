"""Scene package startup hooks for V2.7.1."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

_STATUS_PATH = Path("content/ai/v27_runtime_status.json")


def _write_status(payload: dict) -> None:
    try:
        _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATUS_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


try:
    from src.engine.ai.ranker_v6_extension import install_ranker_v6_extension

    install_ranker_v6_extension()
    _write_status(
        {
            "schema_version": "2.7.1",
            "installed": True,
            "pid": os.getpid(),
            "installed_at": time.time(),
            "startup_path": "src.engine.scenes.__init__",
        }
    )
except Exception as exc:
    _write_status(
        {
            "schema_version": "2.7.1",
            "installed": False,
            "pid": os.getpid(),
            "installed_at": time.time(),
            "startup_path": "src.engine.scenes.__init__",
            "error": repr(exc),
        }
    )
    print(f"[RANKER-V6] V2.7.1 STARTUP ERROR: {exc!r}")
