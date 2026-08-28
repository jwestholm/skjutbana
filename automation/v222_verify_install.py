"""Verify that the V2.22 delta is installed in a full skjutbana checkout.

Run from repository root:
    python3 -m automation.v222_verify_install
"""
from __future__ import annotations

from pathlib import Path
import subprocess


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main() -> None:
    required = [
        Path("src/engine/ai/shot_resolver_v222.py"),
        Path("src/engine/ai/runtime_v222.py"),
        Path("src/engine/ai/bootstrap.py"),
        Path("automation/offline_v222_selftest.py"),
        Path("automation/runtime_v222_selftest.py"),
        Path("V222_SHOT_RESOLVER.md"),
        Path("V222_TEST_PLAN.md"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing V2.22 files: " + ", ".join(missing))

    bootstrap_text = Path("src/engine/ai/bootstrap.py").read_text(encoding="utf-8")
    anchors = [
        "_install_v222_runtime()",
        "def _patch_menu_loader()",
        "def _patch_scene_factory()",
        "def _patch_hit_scanner()",
        "runtime.observe_scanner(self",
        "runtime.choose_for_emission(",
        "runtime.mark_shot_finished(",
    ]
    missing_anchors = [anchor for anchor in anchors if anchor not in bootstrap_text]
    if missing_anchors:
        raise SystemExit("bootstrap.py lost expected integration anchors: " + ", ".join(missing_anchors))

    physical_expected = [
        Path("automation/offline_v2215_selftest.py"),
        Path("automation/physical_dense_v2215_train.py"),
        Path("automation/physical_dense_v2215_benchmark.py"),
    ]
    physical_missing = [str(path) for path in physical_expected if not path.exists()]

    from src.engine.ai.runtime_v222 import install_v222_runtime_patch
    import src.engine.ai.runtime as runtime_module

    install_v222_runtime_patch()
    if not getattr(runtime_module.AIRuntime, "_v222_runtime_patch", False):
        raise SystemExit("AIRuntime V2.22 patch marker missing")

    runtime = runtime_module.get_ai_runtime()
    status = runtime.resolver_status()
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "--short", "HEAD")

    print("V2.22 INSTALL VERIFY")
    print("====================")
    print("[PASS] required delta files exist")
    print("[PASS] bootstrap retains legacy HitScanner/AI anchors plus V2.22 installer")
    print("[PASS] AIRuntime has V2.22 resolver patch")
    if branch:
        print(f"[INFO] git branch             : {branch}")
        if branch.lower() != "dev":
            print("[WARN] expected to test/commit V2.22 on branch dev")
    if head:
        print(f"[INFO] git HEAD               : {head}")
    if physical_missing:
        print(f"[WARN] V2.21.5 files missing  : {', '.join(physical_missing)}")
    else:
        print("[PASS] V2.21.5 train/benchmark/selftest files still present")
    print(f"[INFO] AI mode                : {runtime.settings.get('mode')}")
    print(f"[INFO] resolver enabled       : {runtime.settings.get('resolver_v222_enabled', True)}")
    print(f"[INFO] resolver logging       : {runtime.settings.get('resolver_v222_log', False)}")
    print(f"[INFO] trust percent          : {runtime.settings.get('trust_percent', 0)}")
    print(f"[INFO] min confidence         : {runtime.settings.get('min_confidence', 0.58)}")
    print(f"[INFO] override confidence    : {runtime.settings.get('override_confidence', 0.92)}")
    print(f"[INFO] resolver status schema : {status.get('schema_version')}")
    print()
    print("Install verification passed.")
    print("NEXT: use AI mode 'advisory' for live shadow verification before ai_priority.")


if __name__ == "__main__":
    main()
