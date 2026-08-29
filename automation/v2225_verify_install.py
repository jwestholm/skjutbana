from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def ok(label: str, cond: bool) -> bool:
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    return bool(cond)


def main() -> None:
    print('V2.22.5 INSTALL VERIFY')
    print('======================')
    checks = []
    checks.append(ok('src/engine/shot_fast_v2225.py exists', (ROOT/'src/engine/shot_fast_v2225.py').exists()))
    checks.append(ok('automation/v2225_selftest.py exists', (ROOT/'automation/v2225_selftest.py').exists()))
    text = (ROOT/'main.py').read_text(encoding='utf-8') if (ROOT/'main.py').exists() else ''
    checks.append(ok('main.py imports install_v2225_runtime', 'from src.engine.shot_fast_v2225 import install_v2225_runtime' in text))
    checks.append(ok('main.py calls install_v2225_runtime(App)', 'install_v2225_runtime(App)' in text))
    try:
        branch = subprocess.check_output(['git','branch','--show-current'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        head = subprocess.check_output(['git','rev-parse','--short','HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        print(f'branch={branch or "?"} HEAD={head or "?"}')
    except Exception:
        print('branch/HEAD unavailable (not fatal)')
    if not all(checks):
        raise SystemExit(1)
    print('V2.22.5 install layout looks correct.')


if __name__ == '__main__':
    main()
