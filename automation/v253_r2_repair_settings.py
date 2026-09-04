from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "src/engine/settings.py"
BACKUP = ROOT / "src/engine/settings.py.v253r2-broken.bak"

REQUIRED_MARKERS = (
    "def load_viewport_rect",
    "def load_scanport_rect",
    "def load_content_rect",
    "def load_camera_calibration",
    "def load_audio_peak_settings",
    "def load_audio_peak_threshold",
    "def load_led_settings",
)

PATCHED_LINES = (
    "viewport = load_viewport_rect()",
    "return pygame.Rect(0, 0, viewport.w, viewport.h)",
)


def _is_complete(text: str) -> bool:
    # The bad V2.25.3 package accidentally shipped a ~20-line unit-test stub.
    # A real settings.py is much larger and exposes all long-lived settings APIs.
    return len(text) > 5000 and all(marker in text for marker in REQUIRED_MARKERS)


def _run_git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def _latest_complete_committed_settings() -> tuple[str, str]:
    commits: list[str] = []
    try:
        head = _run_git("rev-parse", "HEAD").strip()
        if head:
            commits.append(head)
        history = _run_git(
            "rev-list", "--max-count=80", "HEAD", "--", "src/engine/settings.py"
        )
        for commit in history.splitlines():
            commit = commit.strip()
            if commit and commit not in commits:
                commits.append(commit)
    except Exception as exc:
        raise RuntimeError(
            "Could not inspect Git history for src/engine/settings.py. "
            "V2.25.3-r2 refuses to invent a replacement settings module."
        ) from exc

    for commit in commits:
        proc = subprocess.run(
            ["git", "show", f"{commit}:src/engine/settings.py"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if proc.returncode == 0 and _is_complete(proc.stdout):
            return proc.stdout, commit

    raise RuntimeError(
        "No complete committed src/engine/settings.py was found in the recent Git history. "
        "The current file was not changed."
    )


def _patch_content_rect(text: str) -> tuple[str, bool]:
    if all(line in text for line in PATCHED_LINES):
        return text, False

    lines = text.splitlines(keepends=True)
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("def load_content_rect("):
            start = i
            break
    if start is None:
        raise RuntimeError("Complete settings.py has no load_content_rect() function")

    for i in range(start + 1, len(lines)):
        if lines[i].startswith("def "):
            end = i
            break

    target = None
    for i in range(start, end):
        if "return load_viewport_rect().copy()" in lines[i]:
            target = i
            break

    if target is None:
        # A future/locally modified implementation may already be correct but use
        # different wording. Never guess-edit a valid user implementation.
        block = "".join(lines[start:end])
        if "pygame.Rect(0, 0," in block and "load_viewport_rect" in block:
            return text, False
        raise RuntimeError(
            "load_content_rect() exists but its fallback is unfamiliar. "
            "Refusing to guess-edit the user's settings implementation."
        )

    indent = lines[target][: len(lines[target]) - len(lines[target].lstrip())]
    replacement = [
        f"{indent}# content_rect is viewport-local throughout the hit/input pipeline.\n",
        f"{indent}# Carrying viewport x/y here would apply that offset twice later.\n",
        f"{indent}viewport = load_viewport_rect()\n",
        f"{indent}return pygame.Rect(0, 0, viewport.w, viewport.h)\n",
    ]
    lines[target:target + 1] = replacement
    return "".join(lines), True


def repair_settings() -> None:
    current = SETTINGS.read_text(encoding="utf-8") if SETTINGS.exists() else ""

    if _is_complete(current):
        source = current
        source_desc = "current working-tree settings.py"
        restored = False
    else:
        if SETTINGS.exists() and not BACKUP.exists():
            shutil.copy2(SETTINGS, BACKUP)
            print(f"[BACKUP] broken settings.py -> {BACKUP.relative_to(ROOT)}")
        source, commit = _latest_complete_committed_settings()
        source_desc = f"Git {commit[:12]}"
        restored = True

    patched, changed = _patch_content_rect(source)
    if not _is_complete(patched):
        raise RuntimeError("Repaired settings.py failed completeness validation")

    compile(patched, str(SETTINGS), "exec")
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(patched, encoding="utf-8")

    if restored:
        print(f"[REPAIR] restored complete src/engine/settings.py from {source_desc}")
    else:
        print(f"[OK] using {source_desc}")
    if changed:
        print("[PATCH] content_rect default -> viewport-local (0,0,w,h)")
    else:
        print("[OK] content_rect viewport-local fallback already present")

    # Import-surface validation without importing pygame-dependent runtime services.
    final = SETTINGS.read_text(encoding="utf-8")
    missing = [m for m in REQUIRED_MARKERS if m not in final]
    if missing:
        raise RuntimeError(f"settings.py still misses APIs: {missing}")
    print("[PASS] settings API surface restored (audio/camera/viewport/LED)")


def main() -> None:
    print("V2.25.3-r2 SETTINGS REPAIR\n============================")
    repair_settings()


if __name__ == "__main__":
    main()
