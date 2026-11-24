"""Disk guard for experiments, benchmarks, nightly training and live tests (2026-11-20).

Principles (user requirements):
  * Each experiment declares its own TEMPORARY files (globs inside its own output dir).  They are removed
    only after the experiment finished successfully and its results are written.
  * Originals (recordings, physical shot data), training data, models, archives and reference data are
    NEVER deleted automatically - classify() refuses, and reclaim only ever touches declared temp files.
  * Before a larger run: ensure_space(estimate).  Target: keep >= MIN_FREE_GB (20) free AFTER the
    estimated output.  If short, reclaim declared temp files of finished/failed/abandoned runs (oldest
    first, never files open by any process, never runs whose process is alive).  If still short: refuse
    to start (DiskBudgetError) and log it.
  * Crash/abort safe: a run directory carries `.disk_run.json` (status running/completed/failed + pid).
    A run that died while 'running' is 'abandoned'; its declared temp files become reclaimable, nothing else.
  * Only paths under /data/skjutbana (or an explicit test root) are ever touched.
  * Everything removed is logged to <root>/disk_guard_log.jsonl (what, why, bytes).

Usage in code:
    from automation import disk_guard
    with disk_guard.experiment(out_dir, estimate_gb=2.0, temp_globs=["*_gstack.npy", "frames/*.npy"]):
        ... write results ...
CLI:
    python3 -m automation.disk_guard status [--root R]
    python3 -m automation.disk_guard check --need-gb 3
    python3 -m automation.disk_guard reclaim [--dry-run] [--target-gb 20]
    python3 -m automation.disk_guard classify PATH
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(os.environ.get("SKJUTBANA_DISK_ROOT", "/data/skjutbana"))
MIN_FREE_GB = float(os.environ.get("SKJUTBANA_DISK_MIN_FREE_GB", "20"))
MARKER = ".disk_run.json"
MARKER_PARTIAL = MARKER + ".partial"
MARKER_CLAIM = MARKER + ".claim"
MARKER_FILES = frozenset((MARKER, MARKER_PARTIAL, MARKER_CLAIM))

# Never deleted automatically.  First match wins (relative to ROOT).
PROTECTED_RULES: list[tuple[str, str]] = [
    ("source_sessions/*", "original: recorded session (camera/audio/ground truth)"),
    ("physical_traces/*", "original: physical shot traces"),
    ("hole_library/*", "reference: hole stamp library"),
    ("board_snapshots/*", "reference: board snapshots"),
    ("skott*/*", "original: physical shooting data"),
    ("ir_test_*/*", "original: unique IR/exposure captures"),
    ("physical_targets/*", "original: live engine logs / physical target records"),
    ("nightly/dataset/*", "training data"),
    ("nightly_evidence/*", "training data"),
    ("physical_ai_lab/datasets/*", "training data"),
    ("backups/*", "backup"),
    ("demo_backups/*", "backup"),
    ("*.tar.zst", "archive"),
    ("*.tar.zst.sha256.json", "archive manifest"),
    ("*/model/*", "model"),
    ("*.npz", "model/array artefact"),
]
# Generic temporary patterns (still only removed inside a run dir whose marker allows it).
GENERIC_TEMP = ["*.partial", "*.tmp", "__pycache__/*"]


# Experiment roots scanned for run markers (never the 1.9M-file archives).  Markers up to 4 levels deep.
EXPERIMENT_DIRS = ["evaluation_runs", "nightly"]


def _markers(root: Path):
    for d in EXPERIMENT_DIRS:
        base = root / d
        for depth in range(0, 5):
            yield from base.glob("/".join(["*"] * depth + [MARKER]))


class DiskBudgetError(RuntimeError):
    pass


def _log(record: dict, root: Path | None = None) -> None:
    root = root or ROOT
    record = dict(record, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    try:
        with open(root / "disk_guard_log.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass
    print(f"[DISK-GUARD] {json.dumps(record)}", flush=True)


def free_gb(path: Path | None = None) -> float:
    return shutil.disk_usage(path or ROOT).free / 1e9


def _rel(path: Path, root: Path) -> str | None:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def classify(path: Path, root: Path | None = None) -> tuple[str, str]:
    """('protected'|'outside'|'unclassified', reason).  'unclassified' is still NOT deletable unless it is a
    declared temp file of a finished run."""
    root = root or ROOT
    rel = _rel(path, root)
    if rel is None:
        return "outside", f"not under {root}"
    for pattern, why in PROTECTED_RULES:
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel + "/", pattern.rstrip("*")):
            return "protected", why
    return "unclassified", "experiment output"


def _open_files() -> set[str]:
    """Absolute paths currently open by any process we can inspect (/proc/*/fd)."""
    out: set[str] = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    out.add(os.readlink(f"{fd_dir}/{fd}"))
                except OSError:
                    pass
        except OSError:
            pass
    return out


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_marker(run_dir: Path) -> dict | None:
    try:
        return json.loads((run_dir / MARKER).read_text())
    except (OSError, ValueError):
        return None


def _write_marker(run_dir: Path, data: dict) -> None:
    tmp = run_dir / MARKER_PARTIAL
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, run_dir / MARKER)


def _temp_files(run_dir: Path, globs: list[str]) -> list[Path]:
    files = []
    for dp, _dn, fn in os.walk(run_dir):
        for f in fn:
            p = Path(dp) / f
            rel = str(p.relative_to(run_dir))
            if f == MARKER:
                continue
            if any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(f, g) for g in list(globs) + GENERIC_TEMP):
                files.append(p)
    return files


def _remove_temp(run_dir: Path, globs: list[str], why: str, root: Path, dry_run: bool = False,
                 open_files: set[str] | None = None) -> int:
    """Remove declared temp files of ONE run dir.  Returns bytes freed.  Protected/outside/open files are
    skipped and logged."""
    open_files = _open_files() if open_files is None else open_files
    freed = 0
    removed = 0
    skipped = []
    for p in _temp_files(run_dir, globs):
        cls, reason = classify(p, root)
        if cls != "unclassified":
            skipped.append((str(p), reason)); continue
        if str(p.resolve()) in open_files:
            skipped.append((str(p), "open by a process")); continue
        try:
            size = p.stat().st_size
            if not dry_run:
                p.unlink()
            freed += size; removed += 1
        except OSError as exc:
            skipped.append((str(p), str(exc)))
    if removed or skipped:
        _log(dict(action="remove_temp" if not dry_run else "dry_run_remove_temp", run=str(run_dir), why=why,
                  files=removed, freed_mb=round(freed / 1e6, 1), skipped=skipped[:10]), root)
    return freed


def assert_output_directory_available(out_dir: Path | str) -> None:
    """Allow only this experiment's disk-guard markers in a claimed output directory."""
    out = Path(out_dir)
    if out.exists() and any(p.name not in MARKER_FILES for p in out.iterdir()):
        raise FileExistsError(f"output exists and is not empty: {out}")


def ensure_space(estimate_gb: float, *, label: str = "", root: Path | None = None,
                 min_free_gb: float | None = None) -> float:
    """Make sure free >= estimate + min_free.  Reclaims declared temp files of finished runs if needed.
    Raises DiskBudgetError (and logs) when the space cannot be guaranteed."""
    root = root or ROOT
    need = float(estimate_gb) + (MIN_FREE_GB if min_free_gb is None else float(min_free_gb))
    free = free_gb(root)
    if free >= need:
        return free
    freed = reclaim(target_free_gb=need, root=root)
    free = free_gb(root)
    if free < need:
        _log(dict(action="refuse_start", label=label, free_gb=round(free, 2), need_gb=round(need, 2),
                  reclaimed_mb=round(freed / 1e6, 1),
                  message="not enough space after reclaiming temp files; originals are never deleted - "
                          "free space manually or archive old data"), root)
        raise DiskBudgetError(f"{label}: free {free:.1f} GB < needed {need:.1f} GB (estimate {estimate_gb} GB + "
                              f"reserve); reclaimed {freed / 1e9:.2f} GB of temp files")
    return free


def reclaim(target_free_gb: float | None = None, *, root: Path | None = None, dry_run: bool = False) -> int:
    """Remove declared temp files of completed / failed / abandoned runs, oldest first, until the target is
    reached (or all reclaimable temp is gone).  Running runs (process alive) are never touched."""
    root = root or ROOT
    target = MIN_FREE_GB if target_free_gb is None else float(target_free_gb)
    runs = []
    for marker in _markers(root):
        run_dir = marker.parent
        m = _read_marker(run_dir) or {}
        status = m.get("status")
        if _pid_alive(m.get("pid")):
            continue                       # a live owner protects the run whatever its marker says
        if status == "running":
            status = "abandoned"          # owner verified dead
        elif status not in ("completed", "failed"):
            continue                       # unknown / broken marker: never reclaim
        runs.append((m.get("finished") or m.get("started") or 0, run_dir, m, status))
    runs.sort(key=lambda r: r[0])
    open_files = _open_files()
    freed = 0
    for _t, run_dir, m, status in runs:
        if not dry_run and free_gb(root) >= target:
            break
        freed += _remove_temp(run_dir, list(m.get("temp_globs") or []), f"reclaim ({status})", root,
                              dry_run=dry_run, open_files=open_files)
    return freed


@contextlib.contextmanager
def experiment(out_dir: Path | str, *, estimate_gb: float = 1.0, temp_globs: list[str] | None = None,
               label: str = "", keep_temp_on_failure: bool = True, root: Path | None = None,
               min_free_gb: float | None = None):
    """Lifecycle for one experiment output directory (see module doc).  The directory must be under root."""
    root = root or ROOT
    out = Path(out_dir)
    if _rel(out, root) is None:
        raise ValueError(f"{out} is not under {root}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty (never overwrite experiment results)")
    ensure_space(estimate_gb, label=label or out.name, root=root, min_free_gb=min_free_gb)
    out.mkdir(parents=True, exist_ok=True)
    try:   # atomic claim: two processes cannot both own (and later write markers into) the same run dir
        with open(out / MARKER_CLAIM, "x") as fh:
            fh.write(str(os.getpid()))
    except FileExistsError:
        raise FileExistsError(f"{out} is already claimed by another experiment") from None
    marker = dict(status="running", pid=os.getpid(), started=time.time(), label=label or out.name,
                  temp_globs=list(temp_globs or []), estimate_gb=estimate_gb)
    _write_marker(out, marker)
    try:
        yield out
    except BaseException as exc:
        marker.update(status="failed", finished=time.time(), error=f"{type(exc).__name__}: {exc}"[:300])
        _write_marker(out, marker)
        if not keep_temp_on_failure:
            _remove_temp(out, marker["temp_globs"], "experiment failed (keep_temp_on_failure=False)", root)
        raise
    marker.update(status="completed", finished=time.time())
    _write_marker(out, marker)      # results are on disk before anything is removed
    _remove_temp(out, marker["temp_globs"], "experiment completed", root)


def maybe_experiment(out_dir: Path | str, *, estimate_gb: float = 1.0, temp_globs: list[str] | None = None,
                     label: str = ""):
    """experiment() when out_dir is under ROOT; otherwise (scratch/test dirs) only the space check on ROOT
    when it exists - nothing is ever removed outside ROOT."""
    if ROOT.exists() and _rel(Path(out_dir), ROOT) is not None:
        return experiment(out_dir, estimate_gb=estimate_gb, temp_globs=temp_globs, label=label)
    if ROOT.exists():
        ensure_space(estimate_gb, label=label or str(out_dir))
    return contextlib.nullcontext(Path(out_dir))


def status(root: Path | None = None) -> dict:
    root = root or ROOT
    runs = {"running": 0, "completed": 0, "failed": 0, "abandoned": 0}
    reclaimable = 0
    for marker in _markers(root):
        m = _read_marker(marker.parent) or {}
        st = m.get("status", "?")
        if st == "running" and not _pid_alive(m.get("pid")):
            st = "abandoned"
        runs[st] = runs.get(st, 0) + 1
        if st != "running":
            reclaimable += sum(p.stat().st_size for p in _temp_files(marker.parent, m.get("temp_globs") or []))
    return dict(root=str(root), free_gb=round(free_gb(root), 2), min_free_gb=MIN_FREE_GB, runs=runs,
                reclaimable_temp_gb=round(reclaimable / 1e9, 3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m automation.disk_guard")
    ap.add_argument("--root", default=str(ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    c = sub.add_parser("check"); c.add_argument("--need-gb", type=float, default=0.0)
    r = sub.add_parser("reclaim"); r.add_argument("--dry-run", action="store_true"); r.add_argument("--target-gb", type=float)
    k = sub.add_parser("classify"); k.add_argument("path")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.cmd == "status":
        print(json.dumps(status(root), indent=1)); return 0
    if a.cmd == "check":
        try:
            ensure_space(a.need_gb, label="cli check", root=root)
            print(f"OK free {free_gb(root):.1f} GB"); return 0
        except DiskBudgetError as exc:
            print(f"REFUSE {exc}"); return 3
    if a.cmd == "reclaim":
        freed = reclaim(a.target_gb if a.target_gb is not None else float("inf") if not a.dry_run else None,
                        root=root, dry_run=a.dry_run)
        print(f"{'would free' if a.dry_run else 'freed'} {freed / 1e9:.3f} GB; free now {free_gb(root):.1f} GB"); return 0
    if a.cmd == "classify":
        print(classify(Path(a.path), root)); return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
