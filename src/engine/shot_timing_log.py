"""Per-shot live timing log (diagnostic only; never changes behaviour).

Enable with SKJUTBANA_SHOT_TIMING_LOG=1 (path: SKJUTBANA_SHOT_TIMING_LOG_PATH, default
/data/skjutbana/shot_timing/<UTC>.jsonl).  One JSON line per finished audio shot:

    shot_id, state, peak_wall_ts, hit_ms (audio peak -> shot finished, wall clock)
    jobs: every detector worker result that was applied for this shot, in order:
        frame_age_ms   camera frame receive ts - audio peak
        submit_age_ms  job submitted (monotonic -> wall) - audio peak
        queue_ms       submitted -> started
        worker_ms      started -> finished
        applied_age_ms result applied on the main thread - audio peak
        stages_ms      worker stage profile (DetectorJobResultV2224.stages_ms)
        candidates     number of candidates returned
    timing: numeric fields of event.audio_trigger["timing"] (dispatch, decision, emission ...)

Analyse with `python3 -m automation.bench shot-timing LOGFILE`.
Added 2026-11-18/19 to find where ~1.2 s per shot goes (AUDIT_FINDINGS §18).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_JOBS: dict[int, list[dict[str, Any]]] = {}
_PATH: Path | None = None


def enabled() -> bool:
    return os.environ.get("SKJUTBANA_SHOT_TIMING_LOG") == "1"


def _path() -> Path:
    global _PATH
    if _PATH is None:
        custom = os.environ.get("SKJUTBANA_SHOT_TIMING_LOG_PATH")
        if custom:
            _PATH = Path(custom)
        else:
            _PATH = Path("/data/skjutbana/shot_timing") / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".jsonl")
        _PATH.parent.mkdir(parents=True, exist_ok=True)
    return _PATH


def _mono_to_wall(mono: float) -> float:
    return time.time() - (time.monotonic() - float(mono))


def record_job(result: Any) -> None:
    """Called when a detector job result is applied (main thread)."""
    if not enabled():
        return
    try:
        sid = int(getattr(result, "shot_id", 0) or 0)
        row = dict(
            frame_ts=float(getattr(result, "frame_ts", 0.0)),
            peak_ts=float(getattr(result, "peak_ts", 0.0)),
            submitted_wall=_mono_to_wall(getattr(result, "submitted_mono", time.monotonic())),
            queue_ms=float(getattr(result, "queue_ms", 0.0)),
            worker_ms=float(getattr(result, "worker_ms", 0.0)),
            applied_wall=time.time(),
            stages_ms={k: round(float(v), 2) for k, v in dict(getattr(result, "stages_ms", {}) or {}).items()},
            candidates=len(getattr(result, "candidates", []) or []),
            error=str(getattr(result, "error", "") or ""),
        )
        with _LOCK:
            _JOBS.setdefault(sid, []).append(row)
    except Exception:
        pass


def record_finish(shot_id: int, state: str, peak_wall_ts: float, hit_ms: float, event: Any = None) -> None:
    """Called once when a shot leaves the pending state (main thread)."""
    if not enabled():
        return
    try:
        sid = int(shot_id)
        peak = float(peak_wall_ts)
        with _LOCK:
            jobs = _JOBS.pop(sid, [])
        out_jobs = []
        for j in jobs:
            out_jobs.append(dict(
                frame_age_ms=round((j["frame_ts"] - peak) * 1000.0, 1),
                submit_age_ms=round((j["submitted_wall"] - peak) * 1000.0, 1),
                queue_ms=round(j["queue_ms"], 1), worker_ms=round(j["worker_ms"], 1),
                applied_age_ms=round((j["applied_wall"] - peak) * 1000.0, 1),
                stages_ms=j["stages_ms"], candidates=j["candidates"], error=j["error"]))
        timing = {}
        trigger = getattr(event, "audio_trigger", None) if event is not None else None
        if isinstance(trigger, dict) and isinstance(trigger.get("timing"), dict):
            for k, v in trigger["timing"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    timing[k] = float(v)
        try:
            import cv2
            cv_threads = int(cv2.getNumThreads())
        except Exception:
            cv_threads = None
        row = dict(shot_id=sid, state=str(state), peak_wall_ts=peak, hit_ms=round(float(hit_ms), 1),
                   cv2_threads=cv_threads, cpu_count=os.cpu_count(),
                   n_jobs=len(out_jobs), jobs=out_jobs, timing=timing,
                   mono_minus_wall=time.monotonic() - time.time())
        with open(_path(), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    except Exception:
        pass
