from __future__ import annotations

import json
import time
from pathlib import Path

STATUS_PATH = Path("content/ai/v28_runtime_status.json")
SESSION_ROOT = Path("content/ai/detector_v28/sessions")
JSONL_PATH = Path("content/ai/detector_v28/shot_diagnostics.jsonl")


def _load_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        value = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        return {"error": repr(exc)}


def _pid_alive(pid: int | None) -> bool:
    return bool(pid and Path(f"/proc/{int(pid)}").exists())


def _atomic_rows(session: str | None) -> int:
    if not session:
        return 0
    folder = SESSION_ROOT / str(session)
    return len(list(folder.glob("shot_*.json"))) if folder.is_dir() else 0


def _jsonl_rows(session: str | None) -> int:
    if not JSONL_PATH.exists():
        return 0
    count = 0
    with JSONL_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and row.get("runtime_session_id") == session and isinstance(row.get("v28_hypotheses"), dict):
                count += 1
    return count


def main() -> None:
    print("=" * 78)
    print("V2.8 GAME-PROCESS VERIFY (READ ONLY)")
    print("=" * 78)
    status = _load_status()
    if not status:
        raise SystemExit("FAIL: no V2.8 runtime marker. Restart game with python3 main.py")

    pid = status.get("pid")
    alive = _pid_alive(int(pid) if pid is not None else None)
    session = status.get("runtime_session_id")
    print(f"Installed              : {status.get('installed')}")
    print(f"Game/runtime PID       : {pid}")
    print(f"PID currently alive    : {alive}")
    print(f"Runtime session        : {session}")
    print(f"Install source         : {status.get('install_source')}")
    print(f"Primary install source : {status.get('primary_install_source')}")
    print(f"Install sources        : {status.get('install_sources', [])}")
    print(f"rank_with_funnel calls : {status.get('rank_with_funnel_calls', 0)}")
    print(f"rank_candidates calls  : {status.get('rank_candidates_calls', 0)}")
    print(f"GT calls               : {status.get('gt_calls', 0)}")
    print(f"Atomic diagnostic rows : {_atomic_rows(session)}")
    print(f"JSONL rows             : {_jsonl_rows(session)}")
    print(f"Status diag counter    : {status.get('diagnostic_rows', 0)}")
    print(f"Status JSONL counter   : {status.get('jsonl_rows', 0)}")
    updated = status.get("updated_at")
    if updated is not None:
        try:
            print(f"Marker age             : {max(0.0, time.time()-float(updated)):.1f}s")
        except Exception:
            pass
    if status.get("error"):
        print(f"ERROR                  : {status.get('error')}")
    print("=" * 78)

    if not bool(status.get("installed")):
        raise SystemExit("FAIL: V2.8 marker says installation failed")
    if not alive:
        raise SystemExit("FAIL: V2.8 marker belongs to dead process")
    sources = [str(v) for v in status.get("install_sources", []) if v is not None]
    if status.get("primary_install_source") != "main.py" and "main.py" not in sources:
        raise SystemExit("FAIL: no evidence V2.8 was installed by live main.py")
    print("PASS: V2.8 is installed in the live main.py process.")
    if int(status.get("rank_with_funnel_calls", 0) or 0):
        print("F2/ranking hook has been exercised.")


if __name__ == "__main__":
    main()
