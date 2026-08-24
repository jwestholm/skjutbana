from __future__ import annotations

import json
import os
import time
from pathlib import Path

STATUS_PATH = Path("content/ai/ranking_v29/status.json")


def _read_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        return {"error": f"could not read status: {exc}"}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    return Path(f"/proc/{pid}").exists()


def _count_dataset_rows(path_value: object) -> int:
    if not path_value:
        return 0
    folder = Path(str(path_value))
    if not folder.is_dir():
        return 0
    return len(list(folder.glob("shot_*.json")))


def main() -> None:
    print("=" * 78)
    print("V2.9 GAME-PROCESS / DATASET VERIFY (READ ONLY)")
    print("=" * 78)

    status = _read_status()
    if not status:
        print(f"Status file missing: {STATUS_PATH}")
        print("FAIL: V2.9 has not been observed in a running game process.")
        raise SystemExit(1)

    if status.get("error"):
        print(f"Status read error: {status.get('error')}")
        raise SystemExit(1)

    pid = int(status.get("pid", 0) or 0)
    installed = bool(status.get("installed"))
    alive = _pid_alive(pid)
    dataset_path = status.get("dataset_path")
    rows_on_disk = _count_dataset_rows(dataset_path)

    print(f"Installed                 : {installed}")
    print(f"Game/runtime PID          : {pid}")
    print(f"PID currently alive       : {alive}")
    print(f"Runtime session           : {status.get('runtime_session_id')}")
    print(f"Install source            : {status.get('install_source')}")
    print(f"Install sources           : {status.get('install_sources', [])}")
    print(f"rank_with_funnel calls    : {status.get('rank_with_funnel_calls', 0)}")
    print(f"Labelled GT calls         : {status.get('labelled_calls', 0)}")
    print(f"Dataset rows status       : {status.get('dataset_rows', 0)}")
    print(f"Dataset rows on disk      : {rows_on_disk}")
    print(f"Dataset session           : {status.get('dataset_session')}")
    print(f"Dataset path              : {dataset_path}")
    print(f"V2.8 integration present  : {status.get('v28_integration_present')}")
    print(f"V7 shadow model loaded    : {status.get('v7_shadow_loaded', False)}")
    print(f"Last error                : {status.get('last_error')}")
    try:
        updated = float(status.get("updated_at"))
        print(f"Status age                : {max(0.0, time.time() - updated):.1f}s")
    except Exception:
        pass
    print("=" * 78)

    if not installed:
        raise SystemExit("FAIL: V2.9 status says extension is not installed.")
    if status.get("v28_integration_present") is False:
        raise SystemExit(
            "FAIL: V2.9 is installed but the V2.8 ranker/hypothesis integration "
            "was not present when it started."
        )
    if not alive:
        raise SystemExit("FAIL: status belongs to a game process that is no longer alive.")
    if status.get("last_error"):
        print("WARNING: V2.9 reported an error; inspect Last error above.")

    calls = int(status.get("labelled_calls", 0) or 0)
    rows = int(status.get("dataset_rows", 0) or 0)

    if calls == 0:
        print("PASS: V2.9 is installed in the live game process.")
        print("No labelled F2 shots have run yet. This is correct before the first test.")
        return

    if rows != calls or rows_on_disk != rows:
        raise SystemExit(
            "FAIL: labelled call count and atomic dataset rows do not match "
            f"(calls={calls}, status_rows={rows}, disk_rows={rows_on_disk})."
        )

    print(
        "PASS: V2.9 is installed and every labelled F2 call has one atomic "
        "ranking-dataset row."
    )


if __name__ == "__main__":
    main()
