from __future__ import annotations

import json
import os
import time
from pathlib import Path


STATUS_PATH = Path("content/ai/ranking_v210/status.json")
MODEL_PATH = Path("content/ai/ranker_v8_offline.json")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def main() -> None:
    print("=" * 76)
    print("V2.10 GAME-PROCESS / V8 SHADOW VERIFY (READ ONLY)")
    print("=" * 76)
    if not STATUS_PATH.exists():
        print(f"No V2.10 status marker: {STATUS_PATH}")
        print("Install V2.10 and restart python3 main.py.")
        raise SystemExit(1)
    try:
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not read status: {exc}")

    pid = int(status.get("pid", 0) or 0)
    print(f"Installed              : {status.get('installed')}")
    print(f"Game/runtime PID       : {pid}")
    print(f"PID currently alive    : {_pid_alive(pid) if pid else False}")
    print(f"Runtime session        : {status.get('runtime_session_id')}")
    print(f"Install source         : {status.get('install_source')}")
    print(f"rank_with_funnel calls : {status.get('rank_with_funnel_calls', 0)}")
    print(f"Labelled GT calls      : {status.get('labelled_calls', 0)}")
    print(f"Shadow rows            : {status.get('shadow_rows', 0)}")
    print(f"V8 model file exists   : {MODEL_PATH.exists()}")
    print(f"V8 model loaded        : {status.get('model_loaded')}")
    print(f"Last error             : {status.get('last_error')}")
    try:
        age = max(0.0, time.time() - float(status.get("updated_at")))
        print(f"Status age             : {age:.1f}s")
    except Exception:
        pass
    print("=" * 76)

    if not bool(status.get("installed")) or not _pid_alive(pid):
        raise SystemExit("FAIL: V2.10 is not installed in a live game process.")
    print("PASS: V2.10 shadow integration is installed in the live game process.")
    if MODEL_PATH.exists() and not bool(status.get("model_loaded")):
        print("NOTE: run/restart one ranking call or restart the game so the model reloads.")


if __name__ == "__main__":
    main()
