"""Verify that the V2.22.2 delta is installed over V2.22.1."""
from __future__ import annotations

from pathlib import Path
import subprocess


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def main() -> None:
    required = [
        Path("src/engine/camera/analysis_geometry_v2221.py"),
        Path("src/engine/camera/hit_scanner_v2221.py"),
        Path("src/engine/camera/analysis_filters_v2222.py"),
        Path("src/engine/camera/hit_scanner_v2222.py"),
        Path("src/engine/ai/runtime_v222.py"),
        Path("src/engine/ai/bootstrap.py"),
        Path("automation/v2222_selftest.py"),
        Path("V2222_TEST_PLAN.md"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(f"[FAIL] missing required files: {missing}")
        raise SystemExit(1)
    print("[PASS] required V2.22.2 + V2.22.1 files exist")

    bootstrap = Path("src/engine/ai/bootstrap.py").read_text(encoding="utf-8")
    checks = {
        "V2.22 runtime installer": "_install_v222_runtime()" in bootstrap,
        "V2.22.1 perspective ROI installer": "_install_v2221_hit_scanner()" in bootstrap,
        "V2.22.2 fast-path installer": "_install_v2222_hit_scanner()" in bootstrap,
        "AI results mapping preserved": 'item.type == "ai_results"' in bootstrap,
        "legacy AIRuntime/HitScanner wrapper preserved": "runtime.observe_scanner(self" in bootstrap,
    }
    failed = False
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        failed = failed or not ok

    print(f"git branch : {_git('branch', '--show-current')}")
    print(f"git HEAD   : {_git('rev-parse', '--short', 'HEAD')}")
    print(f"git status : {_git('status', '--short') or '<clean>'}")
    if failed:
        raise SystemExit(1)
    print("\nV2.22.2 installation verification passed.")


if __name__ == "__main__":
    main()
