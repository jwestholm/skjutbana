from __future__ import annotations

import json
import os
import time
from pathlib import Path

STATUS_PATH = Path('content/ai/v27_runtime_status.json')
DIAG_PATH = Path('content/ai/detector_v27/shot_diagnostics.jsonl')


def _load_status() -> dict:
    if not STATUS_PATH.exists():
        return {}
    try:
        data = json.loads(STATUS_PATH.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        return {'error': f'cannot read marker: {exc!r}'}


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    return Path(f'/proc/{int(pid)}').exists()


def _diag_rows() -> int:
    if not DIAG_PATH.exists():
        return 0
    count = 0
    with DIAG_PATH.open('r', encoding='utf-8') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and isinstance(row.get('v27_hypotheses'), dict):
                count += 1
    return count


def main() -> None:
    print('=' * 76)
    print('V2.7.3 GAME-PROCESS VERIFY (READ ONLY)')
    print('=' * 76)
    status = _load_status()
    if not status:
        print('FAIL: no runtime marker exists.')
        print(f'Expected: {STATUS_PATH}')
        print('Restart the game with: python3 main.py')
        raise SystemExit(1)

    pid = status.get('pid')
    alive = _pid_alive(int(pid) if pid is not None else None)
    print(f"Installed              : {status.get('installed')}")
    print(f"Game/runtime PID       : {pid}")
    print(f"PID currently alive    : {alive}")
    print(f"Runtime session        : {status.get('runtime_session_id')}")
    print(f"Install source         : {status.get('install_source')}")
    print(f"Primary install source : {status.get('primary_install_source')}")
    print(f"Install sources        : {status.get('install_sources', [])}")
    print(f"rank_with_funnel calls : {status.get('rank_with_funnel_calls', 0)}")
    print(f"rank_candidates calls  : {status.get('rank_candidates_calls', 0)}")
    print(f"GT calls               : {status.get('gt_calls', 0)}")
    print(f"Diagnostic rows        : {status.get('diagnostic_rows', 0)}")
    print(f"Rows on disk           : {_diag_rows()}")
    updated = status.get('updated_at')
    if updated is not None:
        try:
            print(f"Marker age             : {max(0.0, time.time()-float(updated)):.1f}s")
        except Exception:
            pass
    if status.get('error'):
        print(f"ERROR                   : {status.get('error')}")
    print('=' * 76)

    if not bool(status.get('installed')):
        raise SystemExit('FAIL: V2.7.3 marker says installation failed.')
    if not alive:
        raise SystemExit('FAIL: marker belongs to a process that is no longer alive. Restart the game.')
    sources = status.get('install_sources', [])
    if not isinstance(sources, list):
        sources = []
    primary = str(status.get('primary_install_source') or '')
    current = str(status.get('install_source') or '')

    installed_from_main = (
        primary == 'main.py'
        or current == 'main.py'
        or 'main.py' in [str(item) for item in sources]
    )

    if not installed_from_main:
        raise SystemExit(
            'FAIL: no evidence that the live game process installed V2.7 from main.py.'
        )

    print('PASS: V2.7.3 is installed in the live main.py game process.')
    if int(status.get('rank_with_funnel_calls', 0) or 0) == 0:
        print('No F2 ranking call has happened yet. That is OK before the first test.')
    else:
        print('F2/ranking hook has been exercised in this same live process.')


if __name__ == '__main__':
    main()
