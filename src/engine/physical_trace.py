"""Asynchronous, observational physical-shot trace capture.

The recorder is intentionally downstream of detector decisions.  It receives
snapshots from the existing scanner/bootstrap hooks and never changes a result.
Tracing is disabled unless ``physical_trace_capture_enabled`` is true.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

TRACE_SCHEMA = "physical-shot-trace-1"
DEFAULT_ROOT = Path("content/ai/physical_traces")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"array": True, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _copy_candidates(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, (list, tuple)):
        return []
    return [_safe(dict(v)) for v in values if isinstance(v, Mapping)]


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


class PhysicalTraceRecorder:
    """Bounded asynchronous writer used by live bootstrap observation hooks."""

    def __init__(self, root: Path | str = DEFAULT_ROOT, *, repo_root: Path | None = None, enabled: bool = False) -> None:
        self.root = Path(root)
        self.repo_root = Path(repo_root or Path.cwd())
        self.enabled = bool(enabled)
        self._queue: queue.Queue[tuple[str, Any] | None] = queue.Queue(maxsize=32)
        self._active: dict[int, dict[str, Any]] = {}
        self._seen_events: set[int] = set()
        self._error_count = 0
        self._base_provenance: dict[str, Any] = {"status": "priming"}
        self._worker = threading.Thread(target=self._run, name="physical-trace-writer", daemon=True)
        self._worker.start()
        threading.Thread(target=self._prime_provenance, name="physical-trace-provenance", daemon=True).start()

    def _prime_provenance(self) -> None:
        try:
            self._base_provenance = {
                "git_commit": _git(self.repo_root, "rev-parse", "HEAD"),
                "git_branch": _git(self.repo_root, "branch", "--show-current"),
                "working_tree_dirty": bool(_git(self.repo_root, "status", "--porcelain", "--untracked-files=all")),
                "python_version": platform.python_version(),
                "model_identifiers": {
                    relative: _sha256(self.repo_root / relative)
                    for relative in ("content/ai/ranker_v3.json", "content/ai/memory.json", "content/ai/settings.json", "content/settings.json")
                    if (self.repo_root / relative).exists()
                },
            }
        except Exception as exc:
            self._base_provenance = {"status": "unavailable", "error": type(exc).__name__}

    def configure(self, settings: Mapping[str, Any] | None) -> None:
        values = settings if isinstance(settings, Mapping) else {}
        self.enabled = bool(values.get("physical_trace_capture_enabled", False))
        if values.get("physical_trace_root"):
            self.root = Path(str(values["physical_trace_root"]))

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            kind, payload = job
            try:
                if kind == "json":
                    path, value = payload
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(path.suffix + ".tmp")
                    tmp.write_text(_json(value), encoding="utf-8")
                    os.replace(tmp, path)
                elif kind == "array":
                    path, value = payload
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, np.asarray(value), allow_pickle=False)
            except Exception:
                self._error_count += 1
            finally:
                self._queue.task_done()

    def _enqueue(self, kind: str, payload: Any) -> bool:
        if not self.enabled:
            return False
        try:
            # Freeze mutable metadata before handing it to the asynchronous writer.
            if kind == "json":
                path, value = payload
                payload = (path, _safe(value))
            self._queue.put_nowait((kind, payload))
            return True
        except queue.Full:
            self._error_count += 1
            return False

    def _provenance(self, settings: Mapping[str, Any] | None) -> dict[str, Any]:
        settings_copy = _safe(dict(settings or {}))
        base = dict(self._base_provenance)
        # Keep the schema explicit even while the background provenance probe is
        # still running or unavailable.  Runtime capture must never wait for it.
        base.setdefault("git_commit", None)
        base.setdefault("git_branch", None)
        base.setdefault("working_tree_dirty", None)
        base.setdefault("python_version", platform.python_version())
        base.setdefault("model_identifiers", {})
        artifacts = dict(base.get("model_identifiers", {}))
        models = [{"path": path, "sha256": value} for path, value in artifacts.items() if "model" in path or "ranker" in path]
        return {
            **base,
            "effective_detector_settings": settings_copy,
            "settings_sha256": hashlib.sha256(_json(settings_copy).encode()).hexdigest(),
            "capture_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_identifiers": artifacts,
            "models": models,
            "calibration": {"status": "snapshot_required_from_runtime"},
        }

    def _shot_dir(self, shot_id: int) -> Path:
        return self.root / "shots" / f"shot_{int(shot_id):08d}"

    def start_from_scanner(self, scanner: Any, event: Any, settings: Mapping[str, Any] | None = None) -> None:
        """Start a trace and enqueue immutable PRE frames for one audio shot."""
        if not self.enabled:
            return
        try:
            shot_id = int(getattr(event, "shot_id"))
            if shot_id in self._active:
                return
            directory = self._shot_dir(shot_id)
            trace = {
                "schema_version": TRACE_SCHEMA, "shot_id": shot_id,
                "session_id": str(settings.get("physical_trace_session_id", "runtime") if isinstance(settings, Mapping) else "runtime"),
                "peak_ts": float(getattr(event, "peak_ts", 0.0)), "created_at": time.time(),
                "coordinate_space": "camera", "provenance": self._provenance(settings),
                "context": {"frame_shape": None, "camera_id": _safe(getattr(scanner, "camera_id", "UNAVAILABLE")), "calibration_id": _safe(getattr(scanner, "calibration_id", "UNAVAILABLE")), "roi_geometry": _safe(getattr(scanner, "roi_geometry", "UNAVAILABLE")), "crop_geometry": _safe(getattr(scanner, "crop_geometry", "UNAVAILABLE")), "calibration": _safe(getattr(scanner, "calibration", "UNAVAILABLE")), "game_hit_regions": _safe(getattr(scanner, "game_hit_regions", "UNAVAILABLE")), "scanner_config": {"candidate_limit": getattr(scanner, "candidate_limit", None), "event_timeout_s": getattr(scanner, "event_timeout_s", None), "track_confirm_frames": getattr(scanner, "track_confirm_frames", None), "track_confirm_span_s": getattr(scanner, "track_confirm_span_s", None), "diff_mode": getattr(scanner, "diff_mode", None)}},
                "frames": [], "stages": [], "outcome": {"status": "pending", "emitted": False, "ground_truth": None},
                "writer_errors": 0,
                "capture_control": {"enabled": True, "writer": "asynchronous_bounded_queue"},
            }
            history = list(getattr(scanner, "frame_history", []) or [])
            pre_snapshot = getattr(scanner, "pre_shot_snapshot", None)
            pre_ts = float(getattr(scanner, "pre_shot_snapshot_ts", 0.0) or 0.0)
            if isinstance(pre_snapshot, np.ndarray):
                self._save_frame(trace, directory, "pre_snapshot", pre_ts, pre_snapshot)
                trace["context"]["frame_shape"] = list(pre_snapshot.shape)
            for frame in history:
                ts = float(getattr(frame, "timestamp", 0.0))
                gray = getattr(frame, "gray", None)
                if ts <= float(getattr(event, "peak_ts", 0.0)) and isinstance(gray, np.ndarray):
                    self._save_frame(trace, directory, "pre_history", ts, gray)
            self._active[shot_id] = {"trace": trace, "directory": directory, "frame_ts": set(), "settings": dict(settings or {}), "map_seq": 0}
            self._seen_events.add(shot_id)
            self._enqueue("json", (directory / "trace.json", trace))
        except Exception:
            self._error_count += 1

    def _save_frame(self, trace: dict[str, Any], directory: Path, kind: str, timestamp: float, frame: np.ndarray) -> None:
        name = f"{kind}_{len(trace['frames']):06d}.npy"
        trace["frames"].append({"kind": kind, "timestamp": float(timestamp), "path": f"frames/{name}", "shape": list(frame.shape), "dtype": str(frame.dtype)})
        self._enqueue("array", (directory / "frames" / name, np.asarray(frame).copy()))

    def observe_scanner(self, scanner: Any, event: Any | None = None, settings: Mapping[str, Any] | None = None) -> None:
        """Capture new frames, maps, candidates, tracks, thresholds and event state."""
        if not self.enabled:
            return
        try:
            events = list(getattr(scanner, "audio_events", []) or [])
            if event is not None and int(getattr(event, "shot_id", 0)) not in self._seen_events:
                self.start_from_scanner(scanner, event, getattr(self._active.get(int(getattr(event, "shot_id", 0)), {}), "settings", {}))
            for item in events:
                sid = int(getattr(item, "shot_id", 0) or 0)
                if sid <= 0:
                    continue
                if sid not in self._seen_events:
                    self.start_from_scanner(scanner, item, settings)
                active = self._active.get(sid)
                if active is None:
                    continue
                trace, directory = active["trace"], active["directory"]
                for frame in list(getattr(scanner, "frame_history", []) or []):
                    ts = float(getattr(frame, "timestamp", 0.0))
                    if ts in active["frame_ts"] or ts <= float(trace["peak_ts"]):
                        continue
                    gray = getattr(frame, "gray", None)
                    if isinstance(gray, np.ndarray):
                        active["frame_ts"].add(ts)
                        self._save_frame(trace, directory, "post", ts, gray)
                maps = {}
                for name, value in dict(getattr(scanner, "debug_frames", {}) or {}).items():
                    if isinstance(value, np.ndarray) and value.ndim == 2 and name in {"pre_shot_delta", "combined_map", "change_map", "candidate_mask", "roi_polygon", "blackhat_map", "whitehat_map"}:
                        map_name = f"maps/{active['map_seq']:06d}_{name}.npy"
                        active["map_seq"] += 1
                        maps[name] = {"path": map_name, "shape": list(value.shape), "dtype": str(value.dtype)}
                        self._enqueue("array", (directory / map_name, value.copy()))
                trace["stages"].append({"timestamp": time.time(), "stage": "scanner_observation", "candidates": _copy_candidates(getattr(scanner, "last_candidates", [])), "tracks": _safe(getattr(scanner, "last_stable_tracks", [])), "pipeline": _safe(getattr(scanner, "last_trace_pipeline", "UNAVAILABLE")), "thresholds": {"combined": getattr(scanner, "last_threshold_value", None), "change": getattr(scanner, "last_change_threshold_value", None), "vote": getattr(scanner, "last_vote_threshold_value", None)}, "window_debug": _safe(getattr(scanner, "last_window_debug", {})), "evidence_maps": maps, "event": _safe(item)})
                if str(getattr(item, "state", "pending")) != "pending":
                    self.finish(sid, scanner, item)
        except Exception:
            self._error_count += 1

    def finish(self, shot_id: int, scanner: Any, event: Any) -> None:
        active = self._active.pop(int(shot_id), None)
        if active is None:
            return
        trace, directory = active["trace"], active["directory"]
        trace["outcome"] = {"status": str(getattr(event, "state", "finished")), "emitted": bool(getattr(event, "emitted", False)), "matched_track_id": getattr(event, "matched_track_id", None), "matched_hole_id": getattr(event, "matched_hole_id", None), "confidence": getattr(event, "confidence", None), "note": str(getattr(event, "note", "")), "final_camera_xy": None if getattr(scanner, "last_best_candidate", None) is None else _safe(getattr(scanner, "last_best_candidate")), "rescue_used": bool(getattr(scanner, "last_window_debug", {}).get("rescue_used", False)), "latency_ms": max(0.0, (time.time() - float(trace["peak_ts"])) * 1000.0), "timeout": str(getattr(event, "state", "")) == "missed", "ground_truth": None}
        trace["writer_errors"] = self._error_count
        self._enqueue("json", (directory / "trace.json", trace))

    def attach_ground_truth(self, shot_id: int, camera_xy: tuple[float, float], *, label_source: str = "manual_verified") -> Path:
        path = self._shot_dir(shot_id) / "ground_truth.json"
        payload = {"schema_version": TRACE_SCHEMA, "shot_id": int(shot_id), "space": "camera", "camera_x": float(camera_xy[0]), "camera_y": float(camera_xy[1]), "label_source": str(label_source), "attached_at": time.time()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json(payload), encoding="utf-8")
        return path

    def flush(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while self._queue.unfinished_tasks and time.time() < deadline:
            time.sleep(0.005)
        return self._queue.unfinished_tasks == 0


_RECORDER: PhysicalTraceRecorder | None = None


def get_physical_trace_recorder() -> PhysicalTraceRecorder:
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = PhysicalTraceRecorder()
    return _RECORDER
