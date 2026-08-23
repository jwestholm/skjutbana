from __future__ import annotations

import json
import time
from pathlib import Path

STATUS_PATH = Path("content/ai/v27_runtime_status.json")
STANDALONE_PATH = Path("content/ai/detector_v27/shot_diagnostics.jsonl")


def _read_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        value = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        return {"error": f"could not read marker: {exc}"}


def _count_rows() -> int:
    if not STANDALONE_PATH.exists():
        return 0
    total = 0
    with STANDALONE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and isinstance(row.get("v27_hypotheses"), dict):
                total += 1
    return total


def main() -> None:
    print("=" * 72)
    print("V2.7.1 RUNTIME VERIFY")
    print("=" * 72)

    # Importing scenes is the exact additional startup path used by V2.7.1.
    import src.engine.scenes  # noqa: F401
    from src.engine.ai.runtime import AIRuntime

    class_installed = bool(
        getattr(AIRuntime, "_ranker_v6_extension_installed", False)
    )
    print(f"V6 installed in this Python process: {class_installed}")

    status = _read_status()
    print(f"Runtime marker: {STATUS_PATH}")
    print(f"Marker exists: {STATUS_PATH.exists()}")
    if status:
        print(f"Marker installed: {status.get('installed')}")
        print(f"Marker PID: {status.get('pid')}")
        print(f"Marker session: {status.get('runtime_session_id')}")
        print(f"Marker startup: {status.get('startup_path', status.get('extension'))}")
        if status.get("error"):
            print(f"Marker ERROR: {status.get('error')}")
        try:
            print(
                "Marker age: "
                f"{max(0.0, time.time() - float(status.get('installed_at'))):.1f}s"
            )
        except Exception:
            pass

    print(f"Standalone V2.7 rows: {_count_rows()}")
    print(f"Standalone path: {STANDALONE_PATH}")
    print("=" * 72)

    if not class_installed:
        raise SystemExit("FAIL: V2.7.1 could not install against this project")
    print("PASS: V2.7.1 installation path works in this project.")
    print(
        "Restart the GAME process too and confirm the terminal prints "
        "'[RANKER-V6] V2.7.1 ... installed'."
    )


if __name__ == "__main__":
    main()
