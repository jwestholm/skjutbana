from __future__ import annotations

import shutil
from pathlib import Path


CAMERA_MARKER = "# --- V2.7.1 RANKER-V6 AUTOLOAD ---"
BOOTSTRAP_MARKER = "# --- V2.7.1 RANKER-V6 BOOTSTRAP FALLBACK ---"

CAMERA_BLOCK = """
# --- V2.7.1 RANKER-V6 AUTOLOAD ---
# V2.7 originally relied on the AI bootstrap path. Some project revisions do
# not execute that hook early enough (or at all) before automated F2 training.
# The camera package is guaranteed to load because HitScanner/Detector V2.x is
# already active. Install V2.7 ranking here as a second, independent hook.
try:
    from src.engine.ai.ranker_v6_extension import install_ranker_v6_extension
    install_ranker_v6_extension()
except Exception as exc:
    print(f"[RANKER-V6] V2.7.1 camera autoload failed: {exc}")
# --- END V2.7.1 RANKER-V6 AUTOLOAD ---
"""

BOOTSTRAP_FUNCTION = """
# --- V2.7.1 RANKER-V6 BOOTSTRAP FALLBACK ---
def _patch_ranker_v6_v271() -> None:
    try:
        from src.engine.ai.ranker_v6_extension import install_ranker_v6_extension
        install_ranker_v6_extension()
    except Exception as exc:
        print(f"[RANKER-V6] V2.7.1 bootstrap fallback failed: {exc}")
# --- END V2.7.1 RANKER-V6 BOOTSTRAP FALLBACK ---
"""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _backup_once(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + ".bak_v27_1")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _patch_camera_init(root: Path) -> tuple[bool, str]:
    path = root / "src/engine/camera/__init__.py"
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    if CAMERA_MARKER in text:
        return False, str(path)

    _backup_once(path)
    if not text.endswith("\n"):
        text += "\n"
    text += "\n" + CAMERA_BLOCK.strip() + "\n"
    path.write_text(text, encoding="utf-8")
    return True, str(path)


def _patch_bootstrap(root: Path) -> tuple[bool, str]:
    path = root / "src/engine/ai/bootstrap.py"
    if not path.exists():
        return False, str(path)

    text = path.read_text(encoding="utf-8")
    changed = False
    _backup_once(path)

    has_original = "def _patch_ranker_v6(" in text and "_patch_ranker_v6()" in text

    if not has_original and BOOTSTRAP_MARKER not in text:
        text += "\n" + BOOTSTRAP_FUNCTION.strip() + "\n"
        changed = True

        anchor = "    _patch_hit_scanner()\n"
        if anchor in text and "    _patch_ranker_v6_v271()\n" not in text:
            text = text.replace(
                anchor,
                "    _patch_ranker_v6_v271()\n" + anchor,
                1,
            )

    if changed:
        path.write_text(text, encoding="utf-8")
    return changed, str(path)


def _verify_files(root: Path) -> None:
    required = [
        root / "src/engine/ai/hypothesis_v27.py",
        root / "src/engine/ai/ranker_v6.py",
        root / "src/engine/ai/ranker_v6_extension.py",
        root / "content/ai/hypothesis_v27_config.json",
        root / "content/ai/ranker_v6_config.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing V2.7 files:\n  " + "\n  ".join(missing))

    for path in required[:3]:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def main() -> None:
    root = _project_root()
    print("=" * 72)
    print("SKJUTBANA V2.7.1 HOTFIX INSTALLER")
    print("=" * 72)
    print(f"Project root: {root}")

    _verify_files(root)

    camera_changed, camera_path = _patch_camera_init(root)
    bootstrap_changed, bootstrap_path = _patch_bootstrap(root)

    marker = root / "content/ai/v27_1_install_marker.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        "V2.7.1 autoload hotfix installed.\n"
        f"camera_init={camera_path}\n"
        f"camera_changed={camera_changed}\n"
        f"bootstrap={bootstrap_path}\n"
        f"bootstrap_changed={bootstrap_changed}\n",
        encoding="utf-8",
    )

    print(f"Camera autoload: {'PATCHED' if camera_changed else 'already present'}")
    print(
        "Bootstrap fallback: "
        f"{'PATCHED' if bootstrap_changed else 'already sufficient / unchanged'}"
    )
    print()
    print("IMPORTANT:")
    print("1. Close the game COMPLETELY.")
    print("2. Start the game again.")
    print("3. At startup you MUST see:")
    print("   [RANKER-V6] V2.7 hypothesis clustering + validated ranker installed")
    print()
    print("Do NOT run another 100-shot test until that startup line is visible.")
    print("=" * 72)


if __name__ == "__main__":
    main()
