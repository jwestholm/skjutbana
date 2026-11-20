"""Selftest for automation/disk_guard.py on a temporary root (never touches /data).
Run: PYTHONPATH=. python3 automation/disk_guard_selftest.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from automation import disk_guard as dg


def _mk(p: Path, n: int = 1000) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * n)
    return p


def test_completed_run_removes_only_declared_temp() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        out = root / "evaluation_runs" / "exp1"
        with dg.experiment(out, estimate_gb=0, temp_globs=["*_gstack.npy", "frames/*.npy"], root=root, min_free_gb=0):
            _mk(out / "a_gstack.npy"); _mk(out / "frames" / "f1.npy"); _mk(out / "summary.json"); _mk(out / "x.png")
            _mk(out / "model.npz")              # protected even if it matched
        assert not (out / "a_gstack.npy").exists() and not (out / "frames" / "f1.npy").exists()
        assert (out / "summary.json").exists() and (out / "x.png").exists() and (out / "model.npz").exists()
        m = json.loads((out / dg.MARKER).read_text())
        assert m["status"] == "completed"
        log = (root / "disk_guard_log.jsonl").read_text()
        assert "experiment completed" in log


def test_failed_run_keeps_temp_then_reclaimable() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        out = root / "evaluation_runs" / "exp_fail"
        try:
            with dg.experiment(out, estimate_gb=0, temp_globs=["*.raw"], root=root, min_free_gb=0):
                _mk(out / "dump.raw"); _mk(out / "error.log")
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert (out / "dump.raw").exists(), "failed run must keep temp for debugging"
        assert json.loads((out / dg.MARKER).read_text())["status"] == "failed"
        freed = dg.reclaim(target_free_gb=1e9, root=root)
        assert freed == 1000 and not (out / "dump.raw").exists() and (out / "error.log").exists()


def test_running_run_is_never_reclaimed_but_abandoned_is() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        live = root / "evaluation_runs" / "live"
        live.mkdir(parents=True)
        dg._write_marker(live, dict(status="running", pid=os.getpid(), temp_globs=["*.raw"]))
        _mk(live / "a.raw")
        dead = root / "evaluation_runs" / "dead"
        dead.mkdir(parents=True)
        p = subprocess.Popen([sys.executable, "-c", "pass"]); p.wait()
        dg._write_marker(dead, dict(status="running", pid=p.pid, temp_globs=["*.raw"]))
        _mk(dead / "b.raw")
        dg.reclaim(target_free_gb=1e9, root=root)
        assert (live / "a.raw").exists(), "running experiment must not be touched"
        assert not (dead / "b.raw").exists(), "abandoned (crashed) run's temp should be reclaimable"


def test_open_file_is_skipped() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        out = root / "evaluation_runs" / "exp_open"
        out.mkdir(parents=True)
        dg._write_marker(out, dict(status="completed", pid=None, temp_globs=["*.raw"]))
        f = _mk(out / "busy.raw")
        with open(f, "rb"):
            dg.reclaim(target_free_gb=1e9, root=root)
            assert f.exists(), "open file must not be removed"


def test_protected_paths_never_removed() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        # a marker inside an experiment dir that (wrongly) declares originals as temp
        out = root / "evaluation_runs" / "evil"
        out.mkdir(parents=True)
        arc = _mk(out / "old.tar.zst")
        dg._write_marker(out, dict(status="completed", temp_globs=["*"]))
        dg.reclaim(target_free_gb=1e9, root=root)
        assert arc.exists(), "archives are protected"
        assert dg.classify(root / "source_sessions" / "s1" / "camera.bin", root)[0] == "protected"
        assert dg.classify(root / "physical_traces" / "x" / "f.npy", root)[0] == "protected"
        assert dg.classify(Path("/home/somewhere/movie.mkv"), root)[0] == "outside"


def test_refuse_when_space_cannot_be_guaranteed() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        try:
            dg.ensure_space(1e6, label="huge", root=root, min_free_gb=0)   # 1 PB
        except dg.DiskBudgetError:
            pass
        else:
            raise AssertionError("expected DiskBudgetError")
        assert "refuse_start" in (root / "disk_guard_log.jsonl").read_text()


def test_outside_root_rejected() -> None:
    with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as other:
        try:
            with dg.experiment(Path(other) / "x", estimate_gb=0, root=Path(t), min_free_gb=0):
                pass
        except ValueError:
            return
        raise AssertionError("experiment outside the root must be rejected")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
