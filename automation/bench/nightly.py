"""Nightly bench: code regression check + training data growth + shadow scorer training.

  python3 -m automation.bench nightly [--root /data/skjutbana/nightly] [--dry-run]

Safe by construction: skips when the game (main.py) runs, when memory/disk is
low, or when another nightly holds the lock; at most two worker processes; the
holdout set is generated ONCE and never trained on; models are written with
status OFFLINE_REPLAY_ONLY_UNVALIDATED and are never installed into the runtime.
Recorded sessions with ground_truth.jsonl under /data/skjutbana/source_sessions
are added to training automatically (except those listed as holdout).
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PILOT_ENV = {"SKJUTBANA_MIN_POST_AGE_S": "0.11", "SKJUTBANA_CAUSAL_OWNERSHIP": "1"}
STRESS = {"hole_diameter_mm": [1.5, 2.5], "hole_depth_gray": [15.0, 35.0]}
# Real-hole stamps (airgun holes from labelled shots, shrunk 50-80 % to CO2 AR15/pistol size).
STAMPS = {"hole_model": "stamps"}
DEFAULT_CONFIG = {
    "backgrounds": [
        "content/ai/physical_traces/session_20260926_162528_D02_4e1d6abd/shots/shot_00000001/frames/pre_history_000001.npy",
        "/data/skjutbana/source_sessions/20261117T220813Z_2842f9a5",
        "/data/skjutbana/source_sessions/20261117T222725Z_a5735418",
        "/data/skjutbana/source_sessions/20261117T220854Z_e1f8a9b4",
    ],
    # Holdout sessions must never also serve as synthetic backgrounds (scene leakage).
    "holdout_sessions": ["/data/skjutbana/source_sessions/20261117T222812Z_98e68fe9"],
    "regression_seeds": [2000, 3],
    "train_seeds_per_night": 4,
    "holdout_seeds": [900000, 6],
    "scenarios_train": "single,double,string3,echo,false_audio",
    "max_parallel": 2,
    "min_free_gb": 10.0,
    "min_available_mem_gb": 12.0,
    "session_root": "/data/skjutbana/source_sessions",
}


def _log(root: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(root / "nightly.log", "a") as handle:
        handle.write(line + "\n")


def _mem_available_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 0.0


def _game_running() -> bool:
    out = subprocess.run(["pgrep", "-af", "main.py"], capture_output=True, text=True).stdout
    return any("skjutbana" in line or "main.py" in line.split()[-1:] for line in out.splitlines()
               if "pgrep" not in line and "automation.bench" not in line)


def _run(cmd: list[str], env_extra: dict[str, str], log: Path) -> int:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SKJUTBANA_")}
    env.update(env_extra)
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    with open(log, "w") as handle:
        return subprocess.run(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
                              timeout=4 * 3600).returncode


def _synth(out: Path, bg: str, seeds: list[int], scenarios: str | None, env: dict, physics: dict | None,
           export: bool) -> list[str]:
    cmd = [sys.executable, "-m", "automation.bench", "synth", "--background", bg, "--output", str(out),
           "--seed-start", str(seeds[0]), "--seeds", str(seeds[1]), "--variant", out.name]
    if scenarios:
        cmd += ["--scenarios", scenarios]
    if physics:
        cmd += ["--physics", json.dumps(physics)]
    if export:
        cmd += ["--export-candidates"]
    return cmd


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m automation.bench nightly")
    p.add_argument("--root", default="/data/skjutbana/nightly")
    p.add_argument("--dry-run", action="store_true", help="only check preconditions and print the plan")
    a = p.parse_args(argv)
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n")
    cfg = {**DEFAULT_CONFIG, **json.loads(config_path.read_text())}

    lock = open(root / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _log(root, "SKIP another nightly is running")
        return 0
    if _game_running():
        _log(root, "SKIP game/main.py is running")
        return 0
    free = shutil.disk_usage(root).free / 1e9
    mem = _mem_available_gb()
    if free < cfg["min_free_gb"] or mem < cfg["min_available_mem_gb"]:
        _log(root, f"SKIP resources free_disk={free:.1f}GB mem_available={mem:.1f}GB")
        return 0

    # Optional per-root override of the pilot env (e.g. evidence-feature nights, 2026-11-19):
    # config "pilot_env_override": {"SKJUTBANA_MIN_POST_AGE_S": "0.15", "SKJUTBANA_H11_FULL_ROI_MAP": "1"}.
    # The env is applied to every pilot job (regression pilot arm, training, holdout, sessions).
    pilot_env = {**PILOT_ENV, **{str(k): str(v) for k, v in (cfg.get("pilot_env_override") or {}).items()}}
    day = (dt.date.today() - dt.date(2026, 11, 17)).days
    night = root / dt.date.today().isoformat()
    if night.exists():
        night = root / f"{dt.date.today().isoformat()}_{time.strftime('%H%M%S')}"
    if set(cfg["backgrounds"]) & set(cfg["holdout_sessions"]):
        _log(root, "ABORT a holdout session is also configured as a training background (leakage)")
        return 2
    bg = ",".join(b for b in cfg["backgrounds"] if Path(b if b.startswith("/") else REPO / b).exists())
    jobs: list[tuple[str, list[str], dict]] = []

    # 1) Code regression check (defaults and pilot flags), fixed seeds.
    for arm, env in (("defaults", {}), ("pilot", pilot_env)):
        out = night / "regression" / arm
        jobs.append((f"regression_{arm}", _synth(out, bg, cfg["regression_seeds"], None, env, None, False), env))
    # 2) New training data (new seeds every night), standard + stress physics.
    seeds = [100000 + day * 100, cfg["train_seeds_per_night"]]
    for variant, physics in (("standard", None), ("stress", STRESS), ("stamps", STAMPS)):
        out = root / "dataset" / "train" / f"{night.name}_{variant}"
        jobs.append((f"train_{variant}", _synth(out, bg, seeds, cfg["scenarios_train"], pilot_env, physics, True), pilot_env))
    # 3) Frozen holdout (generated once).
    for variant, physics in (("standard", None), ("stress", STRESS), ("stamps", STAMPS)):
        out = root / "dataset" / "holdout" / f"synth_{variant}"
        if not (out / "candidates.jsonl").exists():
            jobs.append((f"holdout_{variant}", _synth(out, bg, cfg["holdout_seeds"], cfg["scenarios_train"],
                                                      pilot_env, physics, True), pilot_env))
    # 4) Recorded sessions with ground truth (each exported once).
    holdout_sessions = set(cfg["holdout_sessions"])
    excluded = set(cfg.get("exclude_sessions") or [])   # e.g. GT from a drifted calibration
    for session in sorted(Path(cfg["session_root"]).glob("*/ground_truth.jsonl")):
        sdir = session.parent
        if str(sdir) in excluded or sdir.name in excluded:
            continue
        meta_path = sdir / "optical_holes.json"
        if meta_path.exists():
            try:
                if json.loads(meta_path.read_text()).get("flash"):
                    # precision-flash sessions: the flash frames would reach the ordinary
                    # detector in replay (never happens live: the flash controller hides
                    # them), so they are not training data for it
                    continue
            except (OSError, ValueError):
                pass
        split = "holdout" if str(sdir) in holdout_sessions else "train"
        out = root / "dataset" / split / f"session_{sdir.name}"
        if (out / "candidates.jsonl").exists():
            continue
        offset = "0.025" if (sdir / "optical_holes.json").exists() else "0.0"
        jobs.append((f"session_{sdir.name}", [sys.executable, "-m", "automation.bench", "replay", str(sdir),
                                             "--output", str(out), "--export-candidates", "--pulse-offset-s", offset],
                     pilot_env))

    # Disk guard (2026-11-20): estimate the night's output and keep >= 20 GB free after it; reclaims only
    # declared temp files of finished experiments, never recordings/training data/models.
    from automation import disk_guard
    estimate_gb = 0.25 * len(jobs) + 0.5
    if not a.dry_run:
        try:
            free = disk_guard.ensure_space(estimate_gb, label=f"nightly {night.name}", root=root.parent
                                           if root.parent.exists() else disk_guard.ROOT)
        except disk_guard.DiskBudgetError as exc:
            _log(root, f"SKIP disk guard: {exc}")
            return 0
    _log(root, f"START night={night.name} jobs={len(jobs)} free_disk={free:.1f}GB mem={mem:.1f}GB "
               f"estimate={estimate_gb:.1f}GB")
    if a.dry_run:
        for name, cmd, env in jobs:
            print(name, env, " ".join(cmd))
        return 0
    night.mkdir(parents=True, exist_ok=True)
    (night / "logs").mkdir(exist_ok=True)

    def execute(job):
        name, cmd, env = job
        if _mem_available_gb() < cfg["min_available_mem_gb"] / 2:
            _log(root, f"SKIP job {name}: memory low")
            return name, -1
        Path(cmd[cmd.index("--output") + 1]).parent.mkdir(parents=True, exist_ok=True)
        code = _run(cmd, env, night / "logs" / f"{name}.log")
        _log(root, f"job {name} exit={code}")
        return name, code

    with ThreadPoolExecutor(max_workers=int(cfg["max_parallel"])) as pool:
        results = dict(pool.map(execute, jobs))

    summary: dict = dict(night=night.name, jobs=results)
    # Regression comparison with the previous night (same arm), when comparable.
    previous = sorted(d for d in root.iterdir() if d.is_dir() and d.name[:4].isdigit() and d.name < night.name
                      and (d / "regression").exists())
    if previous:
        from automation.bench.__main__ import compare
        summary["regression_vs_previous"] = {}
        for arm in ("defaults", "pilot"):
            a_dir, b_dir = previous[-1] / "regression" / arm, night / "regression" / arm
            try:
                o = compare(a_dir, b_dir, quiet=True)["overall"]
                summary["regression_vs_previous"][arm] = dict(verdict=o["verdict"], before=o["baseline_correct"],
                                                              after=o["candidate_correct"], cases=o["cases"])
            except BaseException as exc:   # different settings/seeds = not comparable
                summary["regression_vs_previous"][arm] = dict(verdict="NOT_COMPARABLE", reason=str(exc)[:200])
    # Train on all accumulated training data, evaluate on the frozen holdout.
    train_dirs = sorted(str(d) for d in (root / "dataset" / "train").glob("*") if (d / "candidates.jsonl").exists())
    holdout_dirs = sorted(str(d) for d in (root / "dataset" / "holdout").glob("*") if (d / "candidates.jsonl").exists())
    if train_dirs and holdout_dirs:
        from automation.bench.train import train
        result = train(train_dirs, holdout_dirs, night / "model")
        summary["model"] = dict(sha256=result["model_sha256"], **result["metrics"])
        with open(root / "history.jsonl", "a") as handle:
            handle.write(json.dumps(dict(night=night.name, model=summary["model"])) + "\n")
    (night / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"# Nightly {night.name}", "", "Status: OFFLINE_REPLAY_ONLY_UNVALIDATED models, live_authority=false.", ""]
    for arm, v in (summary.get("regression_vs_previous") or {}).items():
        lines.append(f"- regression {arm}: {v.get('verdict')} {v.get('before', '')}→{v.get('after', '')}/{v.get('cases', '')}")
    if "model" in summary:
        m = summary["model"]["holdout_top1"]
        lines.append(f"- scorer holdout top-1: model {m['model']['hits']}/{m['model']['shots']} vs live score "
                     f"{m['live_score']['hits']}/{m['live_score']['shots']}")
        for name, v in (summary["model"].get("holdout_top1_by_source") or {}).items():
            lines.append(f"  - {name}: model {v['model']} vs live score {v['live_score']}")
    failed = [k for k, v in results.items() if v != 0]
    lines.append(f"- jobs: {len(results)} run, failed: {failed or 'none'}")
    (night / "REPORT.md").write_text("\n".join(lines) + "\n")
    shutil.copy2(night / "REPORT.md", root / "LATEST.md")
    _log(root, f"DONE night={night.name} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
