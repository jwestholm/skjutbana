from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "main.py",
        root / "src/engine/shot_critical_v2223.py",
        root / "src/engine/input/object_hit_v2223.py",
        root / "automation/v2223_selftest.py",
        root / "V2223_PLAN.md",
        root / "V2223_SHOT_CRITICAL_AND_OBJECT_HITS.md",
        root / "V2223_TEST_PLAN.md",
        root / "V2223_DOC_PATCH.md",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing V2.22.3 files: " + ", ".join(missing))
    print("[PASS] required V2.22.3 delta files exist")

    text = (root / "main.py").read_text(encoding="utf-8")
    if "install_v2223_runtime(App)" not in text:
        raise SystemExit("main.py does not install V2.22.3")
    print("[PASS] main.py owns V2.22.3 top-level runtime installation")

    from src.engine.camera.camera_manager import CameraManager
    from src.engine.camera.hit_scanner import HitScanner
    from src.engine.app import App
    from src.engine.shot_critical_v2223 import install_v2223_runtime

    install_v2223_runtime(App)
    assert getattr(CameraManager, "_v2223_fast_update_patch", False)
    assert getattr(HitScanner, "_v2223_recent_pre_patch", False)
    assert getattr(App, "_v2223_shot_critical_patch", False)
    print("[PASS] camera, recent-PRE and App shot-critical patches install")

    for rel in [
        "src/engine/camera/hit_scanner_v2221.py",
        "src/engine/camera/hit_scanner_v2222.py",
        "src/engine/ai/runtime_v222.py",
    ]:
        status = "PASS" if (root / rel).exists() else "WARN"
        print(f"[{status}] prior runtime component: {rel}")

    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
        print(f"[INFO] git branch: {branch}")
        print(f"[INFO] git HEAD  : {head}")
    except Exception:
        pass

    print("\nV2.22.3 installation verification passed.")
    print("NEXT: run automation.v2223_selftest, then start main.py and inspect startup/latency logs before shooting.")


if __name__ == "__main__":
    main()
