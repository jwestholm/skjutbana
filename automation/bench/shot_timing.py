"""Summarise a live per-shot timing log (src/engine/shot_timing_log.py).

  python3 -m automation.bench shot-timing LOGFILE [--state matched]

Prints median / p90 of: hit_ms (peak -> shot finished), jobs per shot, first job frame age,
submit age, queue, worker, applied age, and the summed worker stages over all jobs.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def _q(values):
    v = [float(x) for x in values if x is not None]
    if not v:
        return "-"
    return f"median {np.median(v):7.1f}  p90 {np.percentile(v, 90):7.1f}  n {len(v)}"


def summarise(rows: list[dict]) -> dict:
    out = {}
    out["hit_ms"] = [r["hit_ms"] for r in rows]
    out["jobs_per_shot"] = [r["n_jobs"] for r in rows]
    first = [r["jobs"][0] for r in rows if r["jobs"]]
    for key in ("frame_age_ms", "submit_age_ms", "queue_ms", "worker_ms", "applied_age_ms"):
        out["first_job_" + key] = [j[key] for j in first]
    out["all_jobs_worker_ms"] = [j["worker_ms"] for r in rows for j in r["jobs"]]
    out["all_jobs_queue_ms"] = [j["queue_ms"] for r in rows for j in r["jobs"]]
    out["last_job_applied_to_finish_ms"] = [r["hit_ms"] - r["jobs"][-1]["applied_age_ms"] for r in rows if r["jobs"]]
    stages = collections.defaultdict(list)
    for r in rows:
        for j in r["jobs"]:
            for k, v in (j.get("stages_ms") or {}).items():
                stages[k].append(v)
    out["stages"] = dict(stages)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m automation.bench shot-timing")
    ap.add_argument("log")
    ap.add_argument("--state", default=None, help="only shots with this final state (matched/missed/...)")
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in Path(args.log).read_text().splitlines() if l.strip()]
    if args.state:
        rows = [r for r in rows if r.get("state") == args.state]
    states = collections.Counter(r.get("state") for r in rows)
    print(f"shots {len(rows)}  states {dict(states)}")
    s = summarise(rows)
    for key, values in s.items():
        if key == "stages":
            continue
        print(f"{key:32s} {_q(values)}")
    print("worker stages (per job):")
    for key, values in sorted(s["stages"].items(), key=lambda kv: -np.median(kv[1])):
        print(f"  {key:30s} {_q(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
