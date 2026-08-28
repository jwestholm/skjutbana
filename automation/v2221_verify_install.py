"""Verify that V2.22.1 delta is installed over the repository checkout."""
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
        Path("src/engine/ai/runtime_v222.py"),
        Path("src/engine/ai/bootstrap.py"),
        Path("automation/v2221_selftest.py"),
        Path("V2221_TEST_PLAN.md"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(f"[FAIL] missing required files: {missing}")
        raise SystemExit(1)
    print("[PASS] required V2.22.1 delta files exist")

    bootstrap = Path("src/engine/ai/bootstrap.py").read_text(encoding="utf-8")
    checks = {
        "V2.22 runtime installer": "_install_v222_runtime()" in bootstrap,
        "V2.22.1 ROI installer": "_install_v2221_hit_scanner()" in bootstrap,
        "AI settings mapping": 'item.type == "ai_settings"' in bootstrap,
        "AI training mapping": 'item.type == "ai_training"' in bootstrap,
        "AI results mapping": 'item.type == "ai_results"' in bootstrap,
        "legacy AI HitScanner wrapper": "runtime.observe_scanner(self" in bootstrap,
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
    print("\nV2.22.1 installation verification passed.")


if __name__ == "__main__":
    main()
