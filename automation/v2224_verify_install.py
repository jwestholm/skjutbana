from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def check(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    print("V2.22.4 INSTALL VERIFY")
    print("======================")
    print(f"repo:   {ROOT}")
    print(f"branch: {git_value('branch', '--show-current')}")
    print(f"HEAD:   {git_value('rev-parse', '--short', 'HEAD')}")

    required = [
        "main.py",
        "src/engine/shot_async_v2224.py",
        "src/engine/shot_critical_v2223.py",
        "automation/v2224_selftest.py",
        "automation/v2224_apply_docs.py",
        "V2224_PLAN.md",
        "V2224_ASYNC_ARCHITECTURE.md",
        "V2224_TEST_PLAN.md",
    ]
    for rel in required:
        check((ROOT / rel).exists(), rel)

    main_source = (ROOT / "main.py").read_text(encoding="utf-8")
    async_source = (ROOT / "src/engine/shot_async_v2224.py").read_text(encoding="utf-8")
    check("install_v2223_runtime(App)" in main_source, "V2.22.3 installed by main.py")
    check("install_v2224_runtime(App)" in main_source, "V2.22.4 installed by main.py")
    check("_v2224_async_detector_patch" in async_source, "async detector patch present")
    check("_v2224_async_ai_shadow_patch" in async_source, "async AI shadow patch present")
    check("v2224_async_shadow_passthrough" in async_source, "non-authoritative AI critical-path passthrough present")

    # Import the real project classes and install patches, but do not start
    # camera/audio/Pygame. This catches class/module and startup-order errors.
    from src.engine.app import App
    from src.engine.shot_critical_v2223 import install_v2223_runtime
    from src.engine.shot_async_v2224 import install_v2224_runtime, SCHEMA_VERSION

    install_v2223_runtime(App)
    install_v2224_runtime(App)

    from src.engine.camera.hit_scanner import HitScanner
    from src.engine.ai.runtime import AIRuntime
    check(SCHEMA_VERSION == "2.22.4", "runtime schema 2.22.4")
    check(bool(getattr(App, "_v2224_async_shot_patch", False)), "App.run has V2.22.4 async policy")
    check(bool(getattr(HitScanner, "_v2224_async_detector_patch", False)), "real HitScanner has async detector patch")
    check(bool(getattr(AIRuntime, "_v2224_async_ai_shadow_patch", False)), "real AIRuntime has async shadow patch")

    print("\nV2.22.4 install verification passed.")


if __name__ == "__main__":
    main()
