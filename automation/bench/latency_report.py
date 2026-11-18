"""Read-only summary of offline executor jobs in a version 1 host-clock profile."""
from __future__ import annotations

import json
from pathlib import Path


_COLUMNS = ("submit_to_start_ms", "start_to_finish_ms", "submit_to_finish_ms")


def _ms(ns: int | None) -> str:
    if ns is None:
        return "UNKNOWN"
    return f"{ns // 1_000_000}.{ns % 1_000_000:06d}"


def format_report(directory: Path) -> str:
    path = directory / "wall_clock_profile.json"
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"{path} is missing. Use replay --wall-clock-profile to collect timings.") from exc
    if not isinstance(profile, dict) or type(profile.get("version")) is not int or profile["version"] != 1:
        raise ValueError("Expected a version 1 wall-clock profile")
    if profile.get("clock") != "host_monotonic_ns" or not isinstance(profile.get("jobs"), list):
        raise ValueError("Expected host_monotonic_ns timestamps and a jobs array")
    jobs = profile["jobs"]
    lines = ["\t".join(("job", "thread", *_COLUMNS))]
    counts = [0, 0, 0]
    sums = [0, 0, 0]
    for number, job in enumerate(jobs, start=1):
        if not isinstance(job, dict) or not {"submit_ns", "start_ns", "finish_ns", "thread"} <= job.keys():
            raise ValueError(f"Job {number} is missing profile fields")
        submit, start, finish = job["submit_ns"], job["start_ns"], job["finish_ns"]
        if type(submit) is not int or any(t is not None and type(t) is not int for t in (start, finish)):
            raise ValueError(f"Job {number} has invalid timestamps")
        if (start is not None and start < submit) or (finish is not None and (start is None or finish < start)):
            raise ValueError(f"Job {number} has inconsistent timestamps")
        thread = job["thread"]
        if thread is not None and not isinstance(thread, str):
            raise ValueError(f"Job {number} has an invalid thread name")
        durations = (None if start is None else start - submit,
                     None if finish is None else finish - start,
                     None if finish is None else finish - submit)
        lines.append("\t".join((str(number), thread or "UNKNOWN", *map(_ms, durations))))
        for i, duration in enumerate(durations):
            if duration is not None:
                sums[i] += duration
                counts[i] += 1

    lines.append(f"\nSubmitted jobs: {len(jobs)}")
    lines.append("Sums over observed job intervals (observed/submitted):")
    for name, total, count in zip(_COLUMNS, sums, counts):
        lines.append(f"  {name}  {_ms(total if count else None)}  ({count}/{len(jobs)})")
    lines.extend([
        "Submit to start includes replay-loop work before execution.",
        "Start to finish measures elapsed host time including waits. CPU time: UNKNOWN.",
        "Per-stage, emission and shot attribution: UNKNOWN in this profile.",
    ])
    return "\n".join(lines)
