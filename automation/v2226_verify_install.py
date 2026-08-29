from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def ok(label: str, cond: bool) -> bool:
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    return bool(cond)


def main() -> None:
    print("V2.22.6 INSTALL VERIFY")
    print("======================")
    checks = []
    checks.append(ok("src/engine/shot_track_v2226.py exists", (ROOT / "src/engine/shot_track_v2226.py").exists()))
    checks.append(ok("automation/v2226_selftest.py exists", (ROOT / "automation/v2226_selftest.py").exists()))
    main_text = (ROOT / "main.py").read_text(encoding="utf-8") if (ROOT / "main.py").exists() else ""
    checks.append(ok("main.py imports V2.22.6 installer", "from src.engine.shot_track_v2226 import install_v2226_runtime" in main_text))
    checks.append(ok("main.py calls V2.22.6 installer", "install_v2226_runtime(App)" in main_text))
    checks.append(ok("V2.22.6 installs after V2.22.5", main_text.find("install_v2225_runtime(App)") >= 0 and main_text.find("install_v2225_runtime(App)") < main_text.find("install_v2226_runtime(App)")))
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        print(f"branch={branch or '?'} HEAD={head or '?'}")
    except Exception:
        print("branch/HEAD unavailable (not fatal)")
    if not all(checks):
        raise SystemExit(1)
    print("V2.22.6 install layout looks correct.")


if __name__ == "__main__":
    main()
