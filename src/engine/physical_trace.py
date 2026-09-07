"""Asynchronous, fail-open physical-shot trace capture.

Detector hooks only copy/enqueue observations.  Artifact references are added
to a trace after the writer has successfully persisted the artifact; shutdown
finalization writes the JSON snapshot after the queue drains.
"""
from __future__ import annotations

import atexit
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
    if hasattr(value, "__dict__"):
        return _safe(vars(value))
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

    def __init__(self, root: Path | str = DEFAULT_ROOT, *, repo_root: Path | None = None, enabled: bool = False,
                 queue_maxsize: int = 128, max_queued_bytes: int = 1024 * 1024 * 1024,
                 writer_delay_s: float = 0.0, writer_workers: int = 8) -> None:
        self.root = Path(root)
        self.repo_root = Path(repo_root or Path.cwd())
        self.enabled = bool(enabled)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max(1, int(queue_maxsize)))
        self._active: dict[int, dict[str, Any]] = {}
        self._seen_events: set[int] = set()
        self._lock = threading.RLock()
        self._error_count = 0
        self._queue_drops = 0
        self._persisted_items = 0
        self._high_water = 0
        self._queued_bytes = 0
        self._high_water_bytes = 0
        self._writer_delay_s = max(0.0, float(writer_delay_s))
        self._max_queued_bytes = max(1, int(max_queued_bytes))
        self._capture_samples: list[dict[str, float | int | str]] = []
        self._base_provenance: dict[str, Any] = {"status": "priming"}
        self._workers = [threading.Thread(target=self._run, name=f"physical-trace-writer-{i}", daemon=True) for i in range(max(1, int(writer_workers)))]
        for worker in self._workers:
            worker.start()
        threading.Thread(target=self._prime_provenance, name="physical-trace-provenance", daemon=True).start()
        atexit.register(self.flush)

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

    def _record_error(self, shot_id: int | None, *, kind: str, path: Path | None = None,
                      exc: BaseException | None = None, queue_drop: bool = False) -> None:
        with self._lock:
            self._error_count += 1
            if queue_drop:
                self._queue_drops += 1
            if shot_id is not None and shot_id in self._active:
                active = self._active[shot_id]
                active["writer_errors"] += 1
                if queue_drop:
                    active["queue_drops"] += 1
                active["errors"].append({"kind": kind, "path": None if path is None else str(path),
                                         "exception_type": None if exc is None else type(exc).__name__,
                                         "message": None if exc is None else str(exc)})

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            try:
                if self._writer_delay_s:
                    time.sleep(self._writer_delay_s)
                path = Path(job["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                if job["kind"] == "array":
                    temporary = path.with_name(path.name + ".tmp")
                    with temporary.open("wb") as stream:
                        np.save(stream, np.asarray(job["value"]), allow_pickle=False)
                    os.replace(temporary, path)
                elif job["kind"] == "json":
                    temporary = path.with_name(path.name + ".tmp")
                    temporary.write_text(_json(job["value"]), encoding="utf-8")
                    os.replace(temporary, path)
                else:
                    raise ValueError(f"unknown trace writer job kind: {job['kind']}")
                self._artifact_persisted(job)
            except Exception as exc:
                self._record_error(job.get("shot_id"), kind="writer_exception", path=Path(job["path"]), exc=exc)
            finally:
                with self._lock:
                    self._persisted_items += 1 if job.get("persisted") else 0
                    self._queued_bytes = max(0, self._queued_bytes - int(job.get("nbytes", 0)))
                self._queue.task_done()

    def _artifact_persisted(self, job: dict[str, Any]) -> None:
        with self._lock:
            job["persisted"] = True
            sid = job.get("shot_id")
            active = self._active.get(sid)
            if active is None:
                return
            meta = dict(job["meta"])
            path = str(Path(job["path"]).relative_to(active["directory"]))
            meta["path"] = path
            if meta["artifact"] == "frame":
                active["trace"]["frames"].append(meta)
                category = "pre" if str(meta["kind"]).startswith("pre") else "post"
                active["persisted_counts"][category] += 1
            elif meta["artifact"] == "evidence":
                stage = active["stages_by_id"][meta["stage_id"]]
                stage["evidence_maps"][meta["name"]] = {k: meta[k] for k in ("path", "shape", "dtype")}
                active["persisted_counts"]["evidence"] += 1

    def _enqueue(self, job: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            with self._lock:
                if self._queued_bytes + int(job.get("nbytes", 0)) > self._max_queued_bytes:
                    self._record_error(job.get("shot_id"), kind="queue_byte_limit", path=Path(job["path"]), queue_drop=True)
                    return False
                self._queue.put_nowait(job)
                self._queued_bytes += int(job.get("nbytes", 0))
                self._high_water = max(self._high_water, self._queue.qsize())
                self._high_water_bytes = max(self._high_water_bytes, self._queued_bytes)
            return True
        except queue.Full:
            self._record_error(job.get("shot_id"), kind="queue_full", path=Path(job["path"]), queue_drop=True)
            return False

    def _provenance(self, settings: Mapping[str, Any] | None) -> dict[str, Any]:
        settings_copy = _safe(dict(settings or {}))
        base = dict(self._base_provenance)
        base.setdefault("git_commit", None); base.setdefault("git_branch", None)
        base.setdefault("working_tree_dirty", None); base.setdefault("python_version", platform.python_version())
        base.setdefault("model_identifiers", {})
        artifacts = dict(base.get("model_identifiers", {}))
        return {**base, "effective_detector_settings": settings_copy,
                "settings_sha256": hashlib.sha256(_json(settings_copy).encode()).hexdigest(),
                "capture_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_identifiers": artifacts,
                "models": [{"path": p, "sha256": v} for p, v in artifacts.items() if "model" in p or "ranker" in p],
                "calibration": {"status": "snapshot_required_from_runtime"}}

    def _shot_dir(self, shot_id: int) -> Path:
        return self.root / "shots" / f"shot_{int(shot_id):08d}"

    def start_from_scanner(self, scanner: Any, event: Any, settings: Mapping[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        try:
            shot_id = int(getattr(event, "shot_id"))
            with self._lock:
                if shot_id in self._active:
                    return
                directory = self._shot_dir(shot_id)
                trace = {"schema_version": TRACE_SCHEMA, "shot_id": shot_id,
                    "session_id": str(settings.get("physical_trace_session_id", "runtime") if isinstance(settings, Mapping) else "runtime"),
                    "peak_ts": float(getattr(event, "peak_ts", 0.0)), "created_at": time.time(),
                    "coordinate_space": "camera", "provenance": self._provenance(settings),
                    "context": {"frame_shape": None, "camera_id": _safe(getattr(scanner, "camera_id", "UNAVAILABLE")),
                        "calibration_id": _safe(getattr(scanner, "calibration_id", "UNAVAILABLE")),
                        "roi_geometry": _safe(getattr(scanner, "roi_geometry", "UNAVAILABLE")),
                        "crop_geometry": _safe(getattr(scanner, "crop_geometry", "UNAVAILABLE")),
                        "calibration": _safe(getattr(scanner, "calibration", "UNAVAILABLE")),
                        "game_hit_regions": _safe(getattr(scanner, "game_hit_regions", "UNAVAILABLE")),
                        "scanner_config": {"candidate_limit": getattr(scanner, "candidate_limit", None), "event_timeout_s": getattr(scanner, "event_timeout_s", None), "track_confirm_frames": getattr(scanner, "track_confirm_frames", None), "track_confirm_span_s": getattr(scanner, "track_confirm_span_s", None), "diff_mode": getattr(scanner, "diff_mode", None)}},
                    "frames": [], "stages": [], "outcome": {"status": "pending", "emitted": False, "ground_truth": None},
                    "writer_errors": 0, "capture_control": {"enabled": True, "writer": "asynchronous_bounded_queue"}}
                active = {"trace": trace, "directory": directory, "frame_ts": set(), "settings": dict(settings or {}),
                          "map_seq": 0, "map_keys": set(), "stages_by_id": {}, "finished": False,
                          "artifact_order": 0,
                          "writer_errors": 0, "queue_drops": 0, "errors": [], "expected_counts": {"pre": 0, "post": 0, "evidence": 0},
                          "persisted_counts": {"pre": 0, "post": 0, "evidence": 0}, "post_limit_drops": 0}
                self._active[shot_id] = active
                self._seen_events.add(shot_id)
            history = list(getattr(scanner, "frame_history", []) or [])
            pre_snapshot = getattr(scanner, "pre_shot_snapshot", None)
            pre_ts = float(getattr(scanner, "pre_shot_snapshot_ts", 0.0) or 0.0)
            if isinstance(pre_snapshot, np.ndarray):
                self._save_frame(shot_id, "pre_snapshot", pre_ts, pre_snapshot)
                with self._lock:
                    trace["context"]["frame_shape"] = list(pre_snapshot.shape)
            eligible = [(float(getattr(frame, "timestamp", 0.0)), getattr(frame, "gray", None)) for frame in history]
            eligible = [(ts, gray) for ts, gray in eligible if ts <= float(getattr(event, "peak_ts", 0.0)) and isinstance(gray, np.ndarray)]
            for ts, gray in eligible[-24:]:
                self._save_frame(shot_id, "pre_history", ts, gray)
        except Exception as exc:
            self._record_error(None, kind="capture_exception", exc=exc)

    def _save_frame(self, shot_id: int, kind: str, timestamp: float, frame: np.ndarray) -> None:
        started = time.perf_counter()
        with self._lock:
            active = self._active.get(shot_id)
            if active is None:
                return
            category = "pre" if kind.startswith("pre") else "post"
            seq = sum(1 for f in active["trace"]["frames"] if f.get("kind") == kind) + active["expected_counts"][category]
            # Sequence is local and deterministic even though references are added asynchronously.
            name = f"{kind}_{seq:06d}.npy"
            active["expected_counts"][category] += 1
            order = active["artifact_order"]; active["artifact_order"] += 1
            meta = {"artifact": "frame", "kind": kind, "timestamp": float(timestamp), "shape": list(frame.shape), "dtype": str(frame.dtype), "seq": seq, "order": order}
            directory = active["directory"]
            frozen = np.asarray(frame).copy()
        copy_finished = time.perf_counter()
        queued = self._enqueue({"kind": "array", "path": directory / "frames" / name, "value": frozen, "nbytes": int(frozen.nbytes), "shot_id": shot_id, "meta": meta})
        with self._lock:
            self._capture_samples.append({"kind": kind, "bytes": int(frozen.nbytes), "copy_ms": (copy_finished - started) * 1000.0,
                                          "enqueue_ms": (time.perf_counter() - copy_finished) * 1000.0, "total_ms": (time.perf_counter() - started) * 1000.0,
                                          "queued": int(bool(queued))})
            if len(self._capture_samples) > 10000:
                del self._capture_samples[:1000]

    def observe_scanner(self, scanner: Any, event: Any | None = None, settings: Mapping[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        try:
            events = list(getattr(scanner, "audio_events", []) or [])
            if event is not None and int(getattr(event, "shot_id", 0)) not in self._seen_events:
                self.start_from_scanner(scanner, event, settings)
            for item in events:
                sid = int(getattr(item, "shot_id", 0) or 0)
                if sid <= 0:
                    continue
                if sid not in self._seen_events:
                    self.start_from_scanner(scanner, item, settings)
                with self._lock:
                    active = self._active.get(sid)
                    if active is None:
                        continue
                    trace, directory = active["trace"], active["directory"]
                    new_post = False
                    for frame in list(getattr(scanner, "frame_history", []) or []):
                        ts = float(getattr(frame, "timestamp", 0.0))
                        if ts in active["frame_ts"] or ts <= float(trace["peak_ts"]):
                            continue
                        gray = getattr(frame, "gray", None)
                        if isinstance(gray, np.ndarray):
                            active["frame_ts"].add(ts)
                            if active["expected_counts"]["post"] < 64:
                                new_post = True
                                self._save_frame(sid, "post", ts, gray)
                            else:
                                active["post_limit_drops"] += 1
                    maps = {}
                    stage_id = len(trace["stages"])
                    stage = {"timestamp": time.time(), "stage": "scanner_observation", "candidates": _copy_candidates(getattr(scanner, "last_candidates", [])),
                             "tracks": _safe(getattr(scanner, "last_stable_tracks", [])), "pipeline": _safe(getattr(scanner, "last_trace_pipeline", "UNAVAILABLE")),
                             "thresholds": {"combined": getattr(scanner, "last_threshold_value", None), "change": getattr(scanner, "last_change_threshold_value", None), "vote": getattr(scanner, "last_vote_threshold_value", None)},
                             "window_debug": _safe(getattr(scanner, "last_window_debug", {})), "evidence_maps": {}, "event": _safe(item)}
                    stable_tracks = getattr(scanner, "last_stable_tracks", [])
                    if isinstance(stable_tracks, (list, tuple)):
                        confirmed = []
                        for track in stable_tracks:
                            value = _safe(track)
                            if isinstance(value, Mapping) and str(value.get("state", "")).lower() == "confirmed":
                                if "camera_x" in value and "camera_y" in value:
                                    confirmed.append({"camera_x": value["camera_x"], "camera_y": value["camera_y"], "track_id": value.get("track_id")})
                        stage["confirmed_candidates"] = confirmed if confirmed else None
                    trace["stages"].append(stage); active["stages_by_id"][stage_id] = stage
                    if new_post and active["expected_counts"]["evidence"] < 64:
                        for name, value in dict(getattr(scanner, "debug_frames", {}) or {}).items():
                            if not isinstance(value, np.ndarray) or value.ndim != 2 or name not in {"pre_shot_delta", "combined_map", "change_map", "candidate_mask", "roi_polygon", "blackhat_map", "whitehat_map"}:
                                continue
                            key = (max(active["frame_ts"], default=-1.0), name)
                            if key in active["map_keys"]:
                                continue
                            active["map_keys"].add(key)
                            map_name = f"maps/{active['map_seq']:06d}_{name}.npy"; active["map_seq"] += 1
                            active["expected_counts"]["evidence"] += 1
                            meta = {"artifact": "evidence", "stage_id": stage_id, "name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                            maps[name] = {"pending": True}
                            map_started = time.perf_counter()
                            frozen = value.copy()
                            map_copy_finished = time.perf_counter()
                            queued = self._enqueue({"kind": "array", "path": directory / map_name, "value": frozen, "nbytes": int(frozen.nbytes), "shot_id": sid, "meta": meta})
                            with self._lock:
                                self._capture_samples.append({"kind": "evidence", "bytes": int(frozen.nbytes),
                                                              "copy_ms": (map_copy_finished - map_started) * 1000.0,
                                                              "enqueue_ms": (time.perf_counter() - map_copy_finished) * 1000.0,
                                                              "total_ms": (time.perf_counter() - map_started) * 1000.0,
                                                              "queued": int(bool(queued))})
                    if str(getattr(item, "state", "pending")) != "pending":
                        self.finish(sid, scanner, item)
        except Exception as exc:
            self._record_error(None, kind="capture_exception", exc=exc)

    def finish(self, shot_id: int, scanner: Any, event: Any) -> None:
        with self._lock:
            active = self._active.get(int(shot_id))
            if active is None:
                return
            trace = active["trace"]
            debug = _safe(getattr(scanner, "last_event_debug", {}))
            debug_shot = debug.get("shot_id") if isinstance(debug, Mapping) else None
            detector_e2e = (debug.get("detector_e2e_ms")
                            if isinstance(debug, Mapping) and (debug_shot is None or int(debug_shot) == int(shot_id))
                            else None)
            trace_completion = max(0.0, (time.time() - float(trace["peak_ts"])) * 1000.0)
            trace["outcome"] = {"status": str(getattr(event, "state", "finished")), "emitted": bool(getattr(event, "emitted", False)),
                "matched_track_id": getattr(event, "matched_track_id", None), "matched_hole_id": getattr(event, "matched_hole_id", None),
                "confidence": getattr(event, "confidence", None), "note": str(getattr(event, "note", "")),
                "final_camera_xy": None if getattr(scanner, "last_best_candidate", None) is None else _safe(getattr(scanner, "last_best_candidate")),
                "rescue_used": bool(getattr(scanner, "last_window_debug", {}).get("rescue_used", False)),
                "detector_e2e_latency_ms": detector_e2e, "trace_completion_latency_ms": trace_completion,
                "runtime_event_debug": debug, "timeout": str(getattr(event, "state", "")) == "missed", "ground_truth": None}
            active["finished"] = True

    def _finalize_trace(self, shot_id: int, active: dict[str, Any], *, timed_out: bool) -> bool:
        with self._lock:
            expected, persisted = dict(active["expected_counts"]), dict(active["persisted_counts"])
            complete = (not timed_out and not active["errors"] and active["post_limit_drops"] == 0 and expected == persisted and active["finished"])
            trace = active["trace"]
            trace["frames"].sort(key=lambda f: f.get("order", 0))
            trace["writer_errors"] = active["writer_errors"]
            trace["completeness"] = {"pre_frames_expected": expected["pre"], "pre_frames_persisted": persisted["pre"], "post_frames_expected": expected["post"], "post_frames_persisted": persisted["post"], "evidence_artifacts_expected": expected["evidence"], "evidence_artifacts_persisted": persisted["evidence"], "writer_errors": active["writer_errors"], "queue_drops": active["queue_drops"], "post_frame_limit_drops": active["post_limit_drops"], "trace_complete": complete, "completion_reason": "complete" if complete else ("flush_timeout" if timed_out else "incomplete_persistence"), "errors": list(active["errors"])}
            path = active["directory"] / "trace.json"; path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(path.name + ".tmp")
            try:
                temporary.write_text(_json(trace), encoding="utf-8"); os.replace(temporary, path); return True
            except Exception as exc:
                active["errors"].append({"kind": "trace_finalize_exception", "path": str(path), "exception_type": type(exc).__name__, "message": str(exc)})
                return False

    def flush(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while self._queue.unfinished_tasks and time.time() < deadline:
            time.sleep(0.005)
        drained = self._queue.unfinished_tasks == 0
        with self._lock:
            items = list(self._active.items())
        for sid, active in items:
            self._finalize_trace(sid, active, timed_out=not drained)
        return drained

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return {"queued_items": self._queue.qsize(), "persisted_items": self._persisted_items,
                    "writer_errors": self._error_count, "queue_drops": self._queue_drops,
                    "queue_high_water": self._high_water, "queue_high_water_bytes": self._high_water_bytes,
                    "queue_maxsize": self._queue.maxsize, "max_queued_bytes": self._max_queued_bytes,
                    "flush_pending": bool(self._queue.unfinished_tasks)}

    def capture_samples(self) -> list[dict[str, float | int | str]]:
        with self._lock:
            return list(self._capture_samples)

    def shutdown(self, timeout: float = 5.0) -> dict[str, int | bool]:
        """Bounded final flush for application shutdown; never waits forever."""
        drained = self.flush(timeout)
        report = self.stats()
        report["drained"] = drained
        return report

    def attach_ground_truth(self, shot_id: int, camera_xy: tuple[float, float], *, label_source: str = "manual_verified") -> Path:
        path = self._shot_dir(shot_id) / "ground_truth.json"
        payload = {"schema_version": TRACE_SCHEMA, "shot_id": int(shot_id), "space": "camera", "camera_x": float(camera_xy[0]), "camera_y": float(camera_xy[1]), "label_source": str(label_source), "attached_at": time.time()}
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(_json(payload), encoding="utf-8"); return path


_RECORDER: PhysicalTraceRecorder | None = None


def get_physical_trace_recorder() -> PhysicalTraceRecorder:
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = PhysicalTraceRecorder()
    return _RECORDER
