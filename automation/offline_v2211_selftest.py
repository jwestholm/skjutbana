from __future__ import annotations

import json
import tempfile
from pathlib import Path

from automation.v221_collect_fullframe import (
    _atomic_json,
    _background_name,
    _disable_control,
    _validate_capture_config,
    parse_background,
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    print("V2.21.1 SELFTEST")
    print("================")

    cfg = _validate_capture_config()
    _assert(cfg["save_full_frames"], "full-frame capture must be enabled")
    _assert(cfg["save_full_recent_pre"], "true recent PRE full-frame capture must be enabled")
    _assert(cfg["full_frame_post_count"] >= 1, "at least one full POST is required")
    print("[PASS] full-frame recorder config is capture-ready")

    _assert(parse_background("1") == 1, "numeric background parsing failed")
    _assert(parse_background("white") == "white", "named background parsing failed")
    _assert(_background_name(2) == "white_grid", "background naming failed")
    print("[PASS] collector background parsing")

    with tempfile.TemporaryDirectory(prefix="v2211_") as td:
        path = Path(td) / "control.json"
        _atomic_json(path, {"enabled": True, "token": "abc", "shots": 30})
        _disable_control(path, token="abc", reason="selftest")
        payload = json.loads(path.read_text(encoding="utf-8"))
        _assert(payload.get("enabled") is False, "control did not disable")
        _assert(payload.get("disabled_reason") == "selftest", "disable reason missing")
    print("[PASS] one-shot control is atomically disabled")

    scene_path = Path("src/engine/scenes/automation_ai_training.py")
    source = scene_path.read_text(encoding="utf-8")
    required = (
        "V221_CAPTURE_CONTROL_PATH",
        "_apply_v221_capture_control",
        "_restore_v221_capture_mode",
        "_resolve_v217_recent_pre_frame",
        "recent_pre_gray=snapshot.get(\"recent_pre_gray\")",
        "capture_control_v221",
    )
    missing = [needle for needle in required if needle not in source]
    _assert(not missing, f"automation scene is missing V2.21.1 wiring: {missing}")
    print("[PASS] automation scene contains recent-PRE + short capture wiring")

    print("\nAll V2.21.1 selftests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
