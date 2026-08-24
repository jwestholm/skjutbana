from __future__ import annotations

import json
import os
from pathlib import Path


STATUS_PATH = Path("content/ai/ranking_v211/status.json")
MODEL_PATH = Path("content/ai/ranker_v9_offline.json")
CANDIDATE_PATH = Path("content/ai/ranker_v9_candidate.json")


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def main() -> None:
    print("=" * 82)
    print("V2.11 V9 SHADOW VERIFY (READ ONLY)")
    print("=" * 82)

    if not STATUS_PATH.exists():
        print("No live V2.11 status marker found.")
        print("This is normal if the game is not running.")
        print(f"Offline candidate exists: {CANDIDATE_PATH.exists()}")
        print(f"Shadow-ready model exists: {MODEL_PATH.exists()}")
        return

    try:
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not read {STATUS_PATH}: {exc}")

    pid = status.get("pid")
    print(f"Installed              : {status.get('installed')}")
    print(f"Game/runtime PID       : {pid}")
    print(f"PID currently alive    : {_alive(pid)}")
    print(f"Runtime session        : {status.get('runtime_session_id')}")
    print(f"Install source         : {status.get('install_source')}")
    print(f"rank_with_funnel calls : {status.get('rank_with_funnel_calls')}")
    print(f"Labelled GT calls      : {status.get('labelled_calls')}")
    print(f"Shadow rows            : {status.get('shadow_rows')}")
    print(f"V9 model loaded        : {status.get('model_loaded')}")
    print(f"V9 shadow_ready        : {status.get('shadow_ready')}")
    print(f"Last error             : {status.get('last_error')}")
    print(f"Candidate model        : {CANDIDATE_PATH.exists()}")
    print(f"Shadow-ready model     : {MODEL_PATH.exists()}")
    print("=" * 82)

    if not bool(status.get("installed")) or not _alive(pid):
        raise SystemExit("FAIL: V2.11 is not installed in a live game process.")

    print("PASS: V2.11 is installed in the live game process.")
    if not MODEL_PATH.exists():
        print("No V9 shadow-ready model exists. Do NOT run a camera holdout for V9 yet.")
    elif not bool(status.get("model_loaded")):
        print("Restart the game after optimizer created the V9 model so it can be loaded.")


if __name__ == "__main__":
    main()
