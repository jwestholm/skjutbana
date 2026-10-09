"""Asynchronous, fail-open physical-shot trace capture.

Detector hooks only copy/enqueue observations. Artifact references are added
after persistence. Completed bounded captures checkpoint on the existing writer;
shutdown also flushes unfinished captures, with explicit incomplete coverage.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import math
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


def validate_capture_history(value):
    """Explicit capture-only bounds; None preserves the legacy 24/64 policy."""
    if value is None:
        return None
    limits = {"pre_seconds": 10., "post_seconds": 10., "pre_max_frames": 360,
              "post_max_frames": 600, "max_events": 32}
    if not isinstance(value, dict) or set(value) != set(limits):
        raise ValueError("Invalid capture history configuration")
    for key, maximum in limits.items():
        n = value[key]
        if type(n) not in (int, float) or not math.isfinite(n) or not 0 < n <= maximum:
            raise ValueError(f"Capture history {key} must be positive and <= {maximum}")
        if key not in ("pre_seconds", "post_seconds") and type(n) is not int:
            raise ValueError(f"Capture history {key} must be an integer")
    return dict(value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=_safe) + "\n"


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


def _diagnostic_source(candidate: Mapping[str, Any]) -> str:
    if candidate.get("v2225_fast_extract"): return "FAST_V2225"
    if candidate.get("v2_rescue_temporal") or candidate.get("v2_rescue_blob"): return "V2_RESCUE"
    if candidate.get("v26_vault_carried") or candidate.get("v26_vault_hits") or candidate.get("v26_vault_seen_frames"): return "V26_VAULT"
    if candidate.get("candidate_bank_confirmed") or candidate.get("v2_bank_confirmed"): return "V2_BANK"
    if candidate.get("detector_v1") and not candidate.get("detector_v2"): return "V1"
    if candidate.get("v2_primary_peak"): return "V2_PRIMARY"
    return "V2_OTHER"


def _annotate_score_diagnostics(trace: dict[str, Any]) -> None:
    """Annotate copied trace candidates only; never changes runtime objects."""
    formula = "0.16*v2_saliency+0.23*center_darkening+0.15*local_contrast_gain+0.11*blackhat_value+0.06*min(v2_zscore,25);clip[3.6,35.0]"
    pools = []
    decision = trace.get("decision_input", {})
    if isinstance(decision, Mapping) and isinstance(decision.get("retained_candidates"), list): pools.append(decision["retained_candidates"])
    for stage in trace.get("stages", []):
        if isinstance(stage, Mapping) and isinstance(stage.get("candidates"), list): pools.append(stage["candidates"])
        confirmation = stage.get("local_confirmation") if isinstance(stage, Mapping) else None
        if isinstance(confirmation, Mapping) and isinstance(confirmation.get("candidates"), list): pools.append(confirmation["candidates"])
    for pool in pools:
        groups: dict[str, list[float]] = {}
        for candidate in pool:
            if not isinstance(candidate, dict): continue
            name = _diagnostic_source(candidate); groups.setdefault(name, []).append(float(candidate.get("score", 0.0) or 0.0))
        for candidate in pool:
            if not isinstance(candidate, dict): continue
            name = _diagnostic_source(candidate); values = sorted(groups.get(name, [])); score = float(candidate.get("score", 0.0) or 0.0)
            rank = sum(value <= score for value in values) - 1
            candidate["diagnostic_source"] = name
            candidate["diagnostic_score_formula"] = formula
            candidate["diagnostic_score_saturated"] = bool(score >= 35.0 - 1e-9)
            candidate["diagnostic_source_percentile"] = (rank / (len(values) - 1)) if len(values) > 1 else 1.0


def selector_snapshot(trace: Mapping[str, Any], canonical_manifest: str | None = None) -> dict[str, Any]:
    """Build independent selector diagnostics without changing runtime output."""
    decision = trace.get("decision_input", {}) if isinstance(trace, Mapping) else {}
    deterministic = decision.get("deterministic_selection") if isinstance(decision, Mapping) else None
    selectors: dict[str, Any] = {
        "CURRENT_DETERMINISTIC": {
            "selector": "CURRENT_DETERMINISTIC", "status": "LIVE_DETERMINISTIC",
            "selected": _safe(deterministic),
            "track_id": (trace.get("outcome", {}) or {}).get("matched_track_id"),
        }
    }
    try:
        from src.engine.ai.confirmation_selection_shadow import from_trace
        selectors["CONFIRMATION_SELECTION_SHADOW"] = from_trace(trace)
    except Exception as exc:
        selectors["CONFIRMATION_SELECTION_SHADOW"] = {
            "selector": "CONFIRMATION_SELECTION_SHADOW", "status": "UNAVAILABLE", "error": str(exc),
        }
    canonical = trace.get("canonical_challenger")
    if canonical is None and canonical_manifest:
        try:
            from src.engine.ai.canonical_challenger import CanonicalChallenger
            pool = decision.get("retained_candidates") if isinstance(decision, Mapping) else None
            if pool is None:
                pool = next((s["candidates"] for s in reversed(trace.get("stages", [])) if s.get("candidates")), [])
            canonical = CanonicalChallenger(canonical_manifest).rank(pool)
        except Exception as exc:
            canonical = {"mode": "SHADOW", "status": "UNAVAILABLE", "error": str(exc)}
    selectors["CANONICAL_AI_SHADOW"] = canonical or {"mode": "SHADOW", "status": "UNAVAILABLE"}
    return selectors


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
                 writer_delay_s: float = 0.0, writer_workers: int = 8,
                 capture_history: dict | None = None, object_only: bool = False) -> None:
        self.root = Path(root)
        self.repo_root = Path(repo_root or Path.cwd())
        self.enabled = bool(enabled)
        self.object_only = bool(object_only)
        self.capture_history = validate_capture_history(capture_history)
        self._history_environment = None
        self._capture_event_limit_reached = False
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max(1, int(queue_maxsize)))
        self._active: dict[int, dict[str, Any]] = {}
        self._seen_events: set[int] = set()
        self._refused_events: set[int] = set()
        self._owned_capture_roots: set[Path] = set()
        self._refused_capture_roots: set[Path] = set()
        self._lock = threading.RLock()
        self._error_count = 0
        self._metadata_failures = 0
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
        if os.environ.get('SKJUTBANA_PHYSICAL_TRACE_DISABLE')=='1':
            if os.environ.get('SKJUTBANA_PHYSICAL_CAPTURE_BINDING'):
                raise ValueError('Explicit full capture and trace-disable are mutually exclusive')
            self.enabled=False
            return  # process-only gameplay; never write into saved historical roots
        values = settings if isinstance(settings, Mapping) else {}
        self.enabled = bool(values.get("physical_trace_capture_enabled", False))
        if values.get("physical_trace_root"):
            self.root = Path(str(values["physical_trace_root"]))
        if os.environ.get('SKJUTBANA_PHYSICAL_CAPTURE_BINDING'):
            # main.py verified/claimed the new binding before hardware imports.
            root=os.environ.get('SKJUTBANA_PHYSICAL_TRACE_ROOT')
            if not root:raise ValueError('capture binding has not passed startup preflight')
            self.root=Path(root);self.enabled=True
            encoded = os.environ.get('SKJUTBANA_PHYSICAL_CAPTURE_HISTORY', 'null')
            if encoded != self._history_environment:
                self.capture_history = validate_capture_history(json.loads(encoded))
                self._history_environment = encoded

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
                if job["kind"] == "finalize":
                    active = self._active[job['shot_id']]
                    active['checkpoint_saved'] = self._finalize_safely(job['shot_id'], active, timed_out=False)
                    continue
                elif job["kind"] == "array":
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
                    active = self._active.get(job.get('shot_id'))
                    if job['kind'] == 'array' and active is not None:
                        active['pending_arrays'] -= 1
                        self._request_checkpoint(job['shot_id'], active)
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
            elif meta['artifact']=='object_crop':
                active['trace'].setdefault('object_evidence_crops',[]).append(meta)
                if self.object_only:active['persisted_counts']['object']+=1

    def _request_checkpoint(self, shot_id, active):
        """One metadata job after a bounded event closes and its arrays settle.

        Called under the recorder lock by both observation and persistence, so
        closure before/after the final array write is equivalent. No live flush.
        Failed array writes still produce an explicitly incomplete trace.
        """
        if (not active['finished'] or not active.get('post_window_closed')
                or active.get('checkpoint_requested')):
            return
        # Writer failures may prevent expected == persisted. Wait for all array
        # jobs for this event, rather than falsely certifying missing artifacts.
        if active.get('pending_arrays', 0):
            return
        job = dict(kind='finalize', path=active['directory']/'trace.json', shot_id=shot_id)
        active['checkpoint_requested'] = self._enqueue(job)

    def _enqueue(self, job: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            with self._lock:
                if self._queued_bytes + int(job.get("nbytes", 0)) > self._max_queued_bytes:
                    self._record_error(job.get("shot_id"), kind="queue_byte_limit", path=Path(job["path"]), queue_drop=True)
                    return False
                self._queue.put_nowait(job)
                if job.get('kind') == 'array' and job.get('shot_id') in self._active:
                    active = self._active[job['shot_id']]
                    active['pending_arrays'] = active.get('pending_arrays', 0) + 1
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
                'capture_root_override':str(self.root) if os.environ.get('SKJUTBANA_PHYSICAL_CAPTURE_BINDING') else None,
                'capture_binding':os.environ.get('SKJUTBANA_PHYSICAL_CAPTURE_BINDING'),
                'capture_history_requested': self.capture_history,
                "settings_sha256": hashlib.sha256(_json(settings_copy).encode()).hexdigest(),
                "capture_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_identifiers": artifacts,
                "models": [{"path": p, "sha256": v} for p, v in artifacts.items() if "model" in p or "ranker" in p],
                "calibration": {"status": "snapshot_required_from_runtime"}}

    def _shot_dir(self, shot_id: int) -> Path:
        return self.root / "shots" / f"shot_{int(shot_id):08d}"

    def capture_object_crop(self, shot_id, kind, timestamp, bbox, pixels):
        """Actual scoped query inputs, separate from global replay frame arrays."""
        if not self.enabled:return
        with self._lock:
            active=self._active.get(shot_id)
            if active is None:
                self._record_error(shot_id,kind='object_crop_without_active_trace')
                return
            seq=active.get('object_crop_seq',0);active['object_crop_seq']=seq+1
            if self.object_only:
                # Query limit: <=64 regions, 3 PRE + 8 POST; <=1M pixels/frame.
                size=np.asarray(pixels).size
                total=active.get('object_pixels',0)+size
                if seq>=64*11 or size>1_000_000 or total>11_000_000:
                    self._record_error(shot_id,kind='object_crop_limit');return
                active['object_pixels']=total
                active['expected_counts']['object']+=1
            frozen=np.asarray(pixels,dtype=np.uint8).copy()
            path=active['directory']/'object_inputs'/f'{seq:04d}_{kind}.npy'
            meta=dict(artifact='object_crop',kind=kind,timestamp=float(timestamp),camera_bbox=list(bbox),
                shape=list(frozen.shape),dtype=str(frozen.dtype),seq=seq,
                sha256=hashlib.sha256(frozen.tobytes()).hexdigest(),semantics='OBJECT input only; not global replay frame')
            from src.engine.camera.camera_manager import camera_manager
            meta['acquisition'] = camera_manager.provenance_for_timestamp(timestamp)
        self._enqueue(dict(kind='array',path=path,value=frozen,nbytes=frozen.nbytes,shot_id=shot_id,meta=meta))

    def start_from_scanner(self, scanner: Any, event: Any, settings: Mapping[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        try:
            shot_id = int(getattr(event, "shot_id"))
            with self._lock:
                if shot_id in self._active or shot_id in self._refused_events:
                    return
                if self.object_only and len(self._active)>=20:
                    self._refused_events.add(shot_id)
                    print('[OBJECT-CROPS] 20-event diagnostic limit reached',flush=True)
                    return
                if self.capture_history and len(self._active) >= self.capture_history['max_events']:
                    if not self._capture_event_limit_reached:
                        self._capture_event_limit_reached = True
                        self._record_error(None, kind='capture_event_limit')
                        print(f'[PHYSICAL-TRACE] CAPTURE REFUSED: event limit reached at shot {shot_id}')
                    return
                directory = self._shot_dir(shot_id)
                capture_root = directory.parent.absolute()
                if capture_root in self._refused_capture_roots:
                    self._refused_events.add(shot_id)
                    self._seen_events.add(shot_id)
                    return
                if capture_root not in self._owned_capture_roots:
                    if directory.parent.exists() and any(directory.parent.iterdir()):
                        self._refused_capture_roots.add(capture_root)
                        self._refused_events.add(shot_id)
                        self._seen_events.add(shot_id)
                        self._record_error(shot_id, kind="existing_capture_root", path=directory.parent)
                        print(f"[PHYSICAL-TRACE] CAPTURE REFUSED: {directory.parent} contains prior events; bind a fresh session")
                        return
                    self._owned_capture_roots.add(capture_root)
                # A restarted runtime reuses shot IDs. Claim the directory before
                # enqueueing anything; an old session must never become a mixture
                # of new frames and old labels. Atomic mkdir also arbitrates two
                # recorder instances. Capture refusal does not alter detection.
                directory.parent.mkdir(parents=True, exist_ok=True)
                try:
                    directory.mkdir(exist_ok=False)
                except FileExistsError as exc:
                    self._refused_capture_roots.add(capture_root)
                    self._refused_events.add(shot_id)
                    self._seen_events.add(shot_id)
                    self._record_error(shot_id, kind="existing_shot_directory", path=directory, exc=exc)
                    print(f"[PHYSICAL-TRACE] CAPTURE REFUSED: {directory} already exists; bind a fresh session")
                    return
                trace = {"schema_version": TRACE_SCHEMA, "shot_id": shot_id,
                    "audio_trigger": _safe(getattr(event, "audio_trigger", None)),
                    "session_id": (self.root.name if os.environ.get('SKJUTBANA_PHYSICAL_CAPTURE_BINDING') else
                        str(settings.get("physical_trace_session_id", "runtime") if isinstance(settings, Mapping) else "runtime")),
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
                from src.engine.camera.aruco_calibrator import geometry_context
                trace['context']['runtime_geometry'] = geometry_context()
                # Reuse the authoritative PANG snapshot, not scanner debug fields.
                # Copy once so later registry retirement cannot change this capture.
                from src.engine.input.object_hit_v2223 import object_hit_registry_v2223
                shot_context = object_hit_registry_v2223.game_context(shot_id)
                trace['context']['shot_game_context'] = (_safe(shot_context)
                    if shot_context is not None and shot_context.get('peak_ts') == trace['peak_ts'] else None)
                active = {"trace": trace, "directory": directory, "frame_ts": set(), "settings": dict(settings or {}),
                          "map_seq": 0, "map_keys": set(), "stages_by_id": {}, "finished": False,
                          "artifact_order": 0,
                          "writer_errors": 0, "queue_drops": 0, "errors": [], "expected_counts": {"pre": 0, "post": 0, "evidence": 0},
                          "persisted_counts": {"pre": 0, "post": 0, "evidence": 0}, "post_limit_drops": 0}
                self._active[shot_id] = active
                if self.object_only:
                    active['expected_counts']['object']=active['persisted_counts']['object']=0
                    trace['capture_scope']='OBJECT crops only; no full-frame replay'
                    trace['session_id']=self.root.name
                    self._seen_events.add(shot_id)
                    return  # Never save scanner history/snapshot or global evidence.
                if self.capture_history:
                    peak = trace['peak_ts']
                    active['history_config'] = dict(self.capture_history)
                    active['post_seen_until'] = peak
                    active['coverage_last_acquisition'] = {}
                    active['post_window_closed'] = False
                    trace['capture_history'] = {
                        'requested': dict(self.capture_history),
                        'timestamp_semantics': 'CAMERA_READ_RECEIPT_NOT_EXPOSURE',
                        'source': 'existing_scanner_native_cadence_history_no_resampling',
                        'post_semantics': 'observation only; frames after decision/next event are not earlier authority inputs',
                        'available_identity_limit_per_window': 1024,
                        'coverage': {kind: dict(requested_start_ts=start, requested_end_ts=end,
                            available_frames=0, attempted_frames=0, frame_limit_drops=0,
                            first_available_ts=None, last_available_ts=None, maximum_available_gap_s=0.,
                            camera_sequence_gaps=0, sequence_discontinuities=0, unknown_acquisition_frames=0,
                            available_identities=[], available_identity_limit_drops=0)
                            for kind,start,end in [('pre_history',peak-self.capture_history['pre_seconds'],peak),
                                                  ('post',peak,peak+self.capture_history['post_seconds'])]}}
                self._seen_events.add(shot_id)
                for previous_id, previous in self._active.items():
                    if previous_id != shot_id and previous['trace']['peak_ts'] < trace['peak_ts']:
                        previous['trace'].setdefault('next_audio_peak_ts', trace['peak_ts'])
            history = list(getattr(scanner, "frame_history", []) or [])
            pre_snapshot = getattr(scanner, "pre_shot_snapshot", None)
            pre_ts = float(getattr(scanner, "pre_shot_snapshot_ts", 0.0) or 0.0)
            if isinstance(pre_snapshot, np.ndarray):
                self._save_frame(shot_id, "pre_snapshot", pre_ts, pre_snapshot)
                with self._lock:
                    trace["context"]["frame_shape"] = list(pre_snapshot.shape)
            eligible = [(float(getattr(frame, "timestamp", 0.0)), getattr(frame, "gray", None)) for frame in history]
            eligible = [(ts, gray) for ts, gray in eligible if ts <= float(getattr(event, "peak_ts", 0.0)) and isinstance(gray, np.ndarray)]
            if self.capture_history:
                coverage = trace['capture_history']['coverage']['pre_history']
                coverage['left_boundary_observed'] = bool(eligible and min(ts for ts,_ in eligible) <= coverage['requested_start_ts'])
                eligible = sorted({ts: gray for ts,gray in eligible if ts >= coverage['requested_start_ts']}.items())
                for ts,gray in eligible:
                    self._history_available(active, 'pre_history', ts)
                cap = active['history_config']['pre_max_frames']
                coverage['frame_limit_drops'] = max(0,len(eligible)-cap)
                eligible = eligible[-cap:]
                coverage['attempted_frames'] = len(eligible)
            else:
                eligible = eligible[-24:]
            for ts, gray in eligible:
                self._save_frame(shot_id, "pre_history", ts, gray)
        except Exception as exc:
            self._record_error(None, kind="capture_exception", exc=exc)

    def _history_available(self, active, kind, timestamp):
        """Bounded counters over frames actually seen, never an assumed FPS."""
        from src.engine.camera.camera_manager import camera_manager
        coverage = active['trace']['capture_history']['coverage'][kind]
        previous = coverage['last_available_ts']
        if previous is None:
            coverage['first_available_ts'] = timestamp
        else:
            coverage['maximum_available_gap_s'] = max(coverage['maximum_available_gap_s'],timestamp-previous)
        coverage['last_available_ts'] = timestamp
        coverage['available_frames'] += 1
        meta = camera_manager.provenance_for_timestamp(timestamp) or {}
        identity = (meta.get('stream_id'),meta.get('sequence'))
        if len(coverage['available_identities']) < 1024:
            coverage['available_identities'].append(dict(timestamp=timestamp,stream_id=identity[0],sequence=identity[1]))
        else:
            coverage['available_identity_limit_drops'] += 1
        prior = active['coverage_last_acquisition'].get(kind)
        if None in identity:
            coverage['unknown_acquisition_frames'] += 1
        elif prior is not None and None not in prior:
            if identity[0] == prior[0] and identity[1] > prior[1]:
                coverage['camera_sequence_gaps'] += max(0,identity[1]-prior[1]-1)
            else:
                coverage['sequence_discontinuities'] += 1
        active['coverage_last_acquisition'][kind] = identity

    def _capture_extended_post(self, sid, active, history):
        if active['post_window_closed']:
            return False
        coverage = active['trace']['capture_history']['coverage']['post']
        end = coverage['requested_end_ts']
        new_post = False
        for frame in sorted(history, key=lambda f: float(getattr(f,'timestamp',0.))):
            ts = float(getattr(frame,'timestamp',0.))
            gray = getattr(frame,'gray',None)
            if not isinstance(gray,np.ndarray):
                continue
            if active['post_seen_until'] < ts <= end:
                active['post_seen_until'] = ts
                self._history_available(active,'post',ts)
                if coverage['attempted_frames'] < active['history_config']['post_max_frames']:
                    coverage['attempted_frames'] += 1
                    active['frame_ts'].add(ts)
                    self._save_frame(sid,'post',ts,gray)
                    new_post = True
                else:
                    coverage['frame_limit_drops'] += 1
                    active['post_limit_drops'] += 1
            if ts >= end:
                coverage['right_boundary_observed'] = True
                active['post_window_closed'] = True
        return new_post

    def _save_frame(self, shot_id: int, kind: str, timestamp: float, frame: np.ndarray) -> None:
        if self.object_only:return
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
            from src.engine.camera.camera_manager import camera_manager
            meta['acquisition'] = camera_manager.provenance_for_timestamp(timestamp)
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
        if self.object_only:return
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
                    history = list(getattr(scanner, 'frame_history', []) or [])
                    if 'history_config' in active:
                        new_post = self._capture_extended_post(sid,active,history)
                        if active['finished']:
                            continue  # Extended pixels never extend the decision/track ledger.
                    for frame in ([] if 'history_config' in active else history):
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
                             "tracks": _safe(getattr(scanner, "last_stable_tracks", [])),
                             "pipeline": (_safe(getattr(scanner, "last_trace_pipeline", "UNAVAILABLE"))
                                          if getattr(scanner, "last_trace_pipeline_shot_id", sid) == sid else "UNAVAILABLE"),
                             "candidate_pool_shot_id": getattr(scanner, "last_trace_pipeline_shot_id", None),
                             "thresholds": {"combined": getattr(scanner, "last_threshold_value", None), "change": getattr(scanner, "last_change_threshold_value", None), "vote": getattr(scanner, "last_vote_threshold_value", None)},
                             "window_debug": _safe(getattr(scanner, "last_window_debug", {})), "evidence_maps": {}, "event": _safe(item)}
                    confirmation = getattr(scanner, "last_trace_confirmation", None)
                    stage["local_confirmation"] = (_safe(confirmation) if isinstance(confirmation, Mapping)
                                                   and confirmation.get("shot_id") == sid else None)
                    stage["confirmed_candidates"] = (stage["local_confirmation"]["candidates"]
                                                     if stage["local_confirmation"] is not None else None)
                    from src.engine.offline.causal_candidates import classify
                    stage['causal_observation_class'] = classify(
                        cutoff=trace.get('decision_input', {}).get('timestamp'),
                        observed_at=stage['timestamp'], evidence_at=None,
                        owner=stage['candidate_pool_shot_id'], shot_id=sid,
                        next_peak=trace.get('next_audio_peak_ts'))
                    stage['observation_semantics'] = 'shared scanner diagnostic snapshot; use decision_input for authority availability'
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
            # Terminal events may already have left scanner.audio_events. Their
            # bounded capture window still observes pixels, without retaining or
            # changing the event, scanner history, candidates or detector timing.
            with self._lock:
                history = list(getattr(scanner,'frame_history',[]) or [])
                for sid,active in self._active.items():
                    if 'history_config' in active:
                        self._capture_extended_post(sid,active,history)
                        self._request_checkpoint(sid, active)
        except Exception as exc:
            self._record_error(None, kind="capture_exception", exc=exc)

    def capture_decision(self, scanner: Any, track: Any, event: Any, settings=None) -> None:
        """Freeze deterministic authority input before any existing AI override."""
        if self.object_only:return
        if not self.enabled:
            return
        self.start_from_scanner(scanner, event, settings)
        with self._lock:
            active = self._active.get(int(event.shot_id))
            if active is not None:
                if "decision_input" in active["trace"]:
                    return
                from src.engine.track_audit import capture
                active["trace"]["decision_input"] = {
                    "complete_track_audit": {**capture(scanner, event, track), "selected_passed_readiness_for_emission": True},
                    "shot_id": int(event.shot_id), "timestamp": time.time(),
                    "deterministic_selection": _safe(track),
                    "retained_candidates": _copy_candidates(getattr(scanner, "last_candidates", [])),
                    "candidate_pool_shot_id": getattr(scanner, "last_trace_pipeline_shot_id", None),
                    "local_confirmation": (_safe(getattr(scanner, "last_trace_confirmation", None))
                        if (getattr(scanner, "last_trace_confirmation", None) or {}).get('shot_id') == int(event.shot_id) else None),
                    "semantics": "retained proposal pool at emission boundary; not all candidates passed local confirmation",
                }

    def finish(self, shot_id: int, scanner: Any, event: Any) -> None:
        if self.object_only:return
        with self._lock:
            active = self._active.get(int(shot_id))
            if active is None:
                return
            if active['finished']:
                return  # Later scanner activity must not overwrite the terminal event outcome.
            trace = active["trace"]
            from src.engine.track_audit import capture
            if "decision_input" not in trace:
                trace["terminal_track_audit_v1"] = capture(scanner, event)
            # Each terminal trace owns a frozen ledger. Release scanner copies
            # after persistence ownership transfers; overlapping pending owners stay.
            pending_owners = {e.shot_id for e in getattr(scanner, "audio_events", [])
                              if e.state == "pending" and e.shot_id != int(shot_id)}
            scanner._audit_batches = [b for b in getattr(scanner, "_audit_batches", [])
                                      if b.get("dispatch_owner") in pending_owners]
            getattr(scanner, "_audit_selections", {}).pop(int(shot_id), None)
            debug = _safe(getattr(scanner, "last_event_debug", {}))
            debug_shot = debug.get("shot_id") if isinstance(debug, Mapping) else None
            detector_e2e = (debug.get("detector_e2e_ms")
                            if isinstance(debug, Mapping) and (debug_shot is None or int(debug_shot) == int(shot_id))
                            else None)
            trace_completion = max(0.0, (time.time() - float(trace["peak_ts"])) * 1000.0)
            trace["outcome"] = {"status": str(getattr(event, "state", "finished")), "emitted": bool(getattr(event, "emitted", False)),
                "matched_track_id": getattr(event, "matched_track_id", None), "matched_hole_id": getattr(event, "matched_hole_id", None),
                "confidence": getattr(event, "confidence", None), "note": str(getattr(event, "note", "")),
                "final_camera_xy": (_safe(getattr(scanner, "last_best_candidate", None))
                    if bool(getattr(event, "emitted", False)) and (debug_shot is None or int(debug_shot) == int(shot_id)) else None),
                "rescue_used": bool(getattr(scanner, "last_window_debug", {}).get("rescue_used", False)),
                "detector_e2e_latency_ms": detector_e2e, "trace_completion_latency_ms": trace_completion,
                "runtime_event_debug": debug, "timeout": str(getattr(event, "state", "")) == "missed", "ground_truth": None}
            active["finished"] = True

    def finish_object_query(self, shot_id, result):
        """Checkpoint bounded query inputs on the existing writer; no global data."""
        if not self.object_only:return
        with self._lock:
            active=self._active.get(shot_id)
            if active is None or active['finished']:return
            active['trace']['object_query_diagnostic']=_safe(result)
            active['trace']['outcome']=dict(status=result.get('status'),ground_truth=None,
                object_decision=_safe(result.get('object_decision')))
            active['finished']=active['post_window_closed']=True
            self._request_checkpoint(shot_id,active)

    def _finalize_trace(self, shot_id: int, active: dict[str, Any], *, timed_out: bool) -> bool:
        with self._lock:
            expected, persisted = dict(active["expected_counts"]), dict(active["persisted_counts"])
            complete = (not timed_out and not active["errors"] and active["post_limit_drops"] == 0 and expected == persisted and active["finished"])
            trace = active["trace"]
            if self.object_only:
                trace['completeness']=dict(trace_complete=complete,expected=expected,persisted=persisted,
                    writer_errors=active['writer_errors'],queue_drops=active['queue_drops'],errors=active['errors'])
                path=active['directory']/'trace.json';temporary=path.with_name(path.name+'.tmp')
                temporary.write_text(_json(trace),encoding='utf-8');os.replace(temporary,path)
                return True
            if 'capture_history' in trace:
                for kind,coverage in trace['capture_history']['coverage'].items():
                    saved = sorted(f['timestamp'] for f in trace['frames'] if f['kind']==kind)
                    coverage.update(saved_frames=len(saved), first_saved_ts=saved[0] if saved else None,
                        last_saved_ts=saved[-1] if saved else None,
                        saved_span_s=saved[-1]-saved[0] if saved else None,
                        maximum_saved_gap_s=max((b-a for a,b in zip(saved,saved[1:])),default=None),
                        persistence_losses=coverage['attempted_frames']-len(saved),
                        continuity_claim='NOT_CERTIFIED; inspect boundaries, sequence gaps, unavailable metadata and writer losses')
                    coverage.setdefault('right_boundary_observed' if kind=='post' else 'left_boundary_observed',False)
                trace['capture_history']['post_window_closed'] = active['post_window_closed']
                coverage = trace['capture_history']['coverage']
                observed = (coverage['pre_history']['left_boundary_observed']
                            and coverage['post']['right_boundary_observed']
                            and all(c['available_frames'] > 0 for c in coverage.values())
                            and not any(c['frame_limit_drops'] for c in coverage.values()))
                trace['capture_history']['requested_boundaries_observed_without_frame_cap_loss'] = bool(observed)
                complete = complete and observed
            manifest = active["settings"].get("canonical_challenger_manifest")
            # Diagnostic sidecar only: never changes decision_input, selectors,
            # ground truth, candidate pools or persistent physical state.
            from src.engine.input.object_hit_v2223 import object_hit_registry_v2223
            object_snapshot = object_hit_registry_v2223.snapshot_for_shot(shot_id)
            if object_snapshot is not None and object_snapshot.object_query is not None:
                trace['object_query_diagnostic'] = _safe(object_snapshot.object_query.result)
                trace['object_gameplay_diagnostic'] = _safe(object_snapshot.object_gameplay)
            from src.engine.offline.causal_candidates import classify
            for stage in trace['stages']:
                stage['causal_observation_class'] = classify(
                    cutoff=trace.get('decision_input', {}).get('timestamp'),
                    observed_at=stage.get('timestamp'), evidence_at=None,
                    owner=stage.get('candidate_pool_shot_id'), shot_id=shot_id,
                    next_peak=trace.get('next_audio_peak_ts'))
            if manifest and "canonical_challenger" not in trace:
                try:
                    from src.engine.ai.canonical_challenger import CanonicalChallenger
                    decision = trace.get("decision_input", {})
                    pool = decision.get("retained_candidates")
                    if pool is None:
                        pool = next((s["candidates"] for s in reversed(trace["stages"]) if s.get("candidates")), [])
                    trace["canonical_challenger"] = CanonicalChallenger(manifest).rank(pool)
                    trace["canonical_challenger"]["evaluation_timing"] = "post_decision_at_trace_finalization"
                    trace["canonical_challenger"]["pool_semantics"] = "decision_boundary_retained" if decision else "last_observed_retained"
                except Exception as exc:
                    trace["canonical_challenger"] = {"mode": "SHADOW", "status": "UNAVAILABLE", "error": str(exc)}
            _annotate_score_diagnostics(trace)
            trace["selectors"] = selector_snapshot(trace, manifest)
            # Keep the historical top-level key while exposing the frozen
            # selector beside it for consumers that understand the new schema.
            trace["confirmation_selection_shadow"] = trace["selectors"]["CONFIRMATION_SELECTION_SHADOW"]
            trace["frames"].sort(key=lambda f: f.get("order", 0))
            trace["writer_errors"] = active["writer_errors"]
            trace["completeness"] = {"pre_frames_expected": expected["pre"], "pre_frames_persisted": persisted["pre"], "post_frames_expected": expected["post"], "post_frames_persisted": persisted["post"], "evidence_artifacts_expected": expected["evidence"], "evidence_artifacts_persisted": persisted["evidence"], "writer_errors": active["writer_errors"], "queue_drops": active["queue_drops"], "post_frame_limit_drops": active["post_limit_drops"], "trace_complete": complete, "completion_reason": "complete" if complete else ("flush_timeout" if timed_out else "incomplete_persistence"), "errors": list(active["errors"])}
            path = active["directory"] / "trace.json"; path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(_json(trace), encoding="utf-8"); os.replace(temporary, path); return True

    def _finalize_safely(self, shot_id, active, *, timed_out):
        try:
            return self._finalize_trace(shot_id, active, timed_out=timed_out)
        except Exception as exc:
            error = dict(kind='trace_finalize_exception', shot_id=shot_id,
                         exception_type=type(exc).__name__, message=str(exc), recorded_at=time.time())
            with self._lock:
                self._metadata_failures += 1
                active['errors'].append(error)
            print(f'[PHYSICAL-TRACE] FINALIZATION FAILED shot={shot_id}: {type(exc).__name__}: {exc}', flush=True)
            # This is an error sidecar, never a substitute trace or fabricated
            # frame ledger. Preserve any earlier successful trace checkpoint.
            try:
                path = active['directory']/'trace_error.json'
                temporary = path.with_name(path.name+'.tmp')
                temporary.write_text(_json(error), encoding='utf-8')
                os.replace(temporary, path)
            except Exception as write_exc:
                print(f'[PHYSICAL-TRACE] error sidecar failed: {write_exc}', flush=True)
            return False

    def flush(self, timeout: float = 5.0) -> bool:
        deadline = time.time() + timeout
        while self._queue.unfinished_tasks and time.time() < deadline:
            time.sleep(0.005)
        drained = self._queue.unfinished_tasks == 0
        with self._lock:
            items = list(self._active.items())
        finalized = True
        for sid, active in items:
            finalized = self._finalize_safely(sid, active, timed_out=not drained) and finalized
        return drained and finalized

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return {"queued_items": self._queue.qsize(), "persisted_items": self._persisted_items,
                    "writer_errors": self._error_count, "queue_drops": self._queue_drops,
                    "metadata_failures": self._metadata_failures,
                    "refused_existing_shots": len(self._refused_events),
                    "refused_capture_roots": len(self._refused_capture_roots),
                    "capture_event_limit_reached": self._capture_event_limit_reached,
                    "queue_high_water": self._high_water, "queue_high_water_bytes": self._high_water_bytes,
                    "queue_maxsize": self._queue.maxsize, "max_queued_bytes": self._max_queued_bytes,
                    "flush_pending": bool(self._queue.unfinished_tasks)}

    def capture_samples(self) -> list[dict[str, float | int | str]]:
        with self._lock:
            return list(self._capture_samples)

    def shutdown(self, timeout: float = 5.0) -> dict[str, int | bool]:
        """Bounded final flush for application shutdown; never waits forever."""
        finalized = self.flush(timeout)
        report = self.stats()
        report["drained"] = not bool(self._queue.unfinished_tasks)
        report["finalized"] = finalized
        return report

    def attach_ground_truth(self, shot_id: int, camera_xy: tuple[float, float], *, label_source: str = "manual_verified") -> Path:
        if int(shot_id) in self._refused_events:
            raise ValueError("Capture was refused: cannot attach a new label to an existing shot")
        path = self._shot_dir(shot_id) / "ground_truth.json"
        payload = {"schema_version": TRACE_SCHEMA, "shot_id": int(shot_id), "space": "camera", "camera_x": float(camera_xy[0]), "camera_y": float(camera_xy[1]), "label_source": str(label_source), "attached_at": time.time()}
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(_json(payload), encoding="utf-8"); return path


_RECORDER: PhysicalTraceRecorder | None = None


def get_physical_trace_recorder() -> PhysicalTraceRecorder:
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = PhysicalTraceRecorder()
    return _RECORDER


_OBJECT_RECORDER: PhysicalTraceRecorder | None = None


def get_object_crop_recorder() -> PhysicalTraceRecorder | None:
    """Explicit short gameplay diagnostic, reusing the bounded trace writer.

    A separate instance keeps ordinary trace configuration/authority untouched.
    Only object crops and the completed scoped query enter this writer.
    """
    global _OBJECT_RECORDER
    name=os.environ.get('SKJUTBANA_OBJECT_CROP_ROOT')
    if not name:return None
    from automation.physical_collection import _development_path
    root=_development_path(name)
    if not root.resolve().is_relative_to(Path('/data/skjutbana')):
        raise ValueError('Object diagnostics must be under /data/skjutbana')
    if os.environ.get('SKJUTBANA_PHYSICAL_TRACE_DISABLE')!='1' or os.environ.get('SKJUTBANA_PHYSICAL_CAPTURE_BINDING'):
        raise ValueError('Object-only crops require full tracing disabled')
    if _OBJECT_RECORDER is None:
        root.mkdir(parents=True,exist_ok=False)  # Never append into a prior run.
        _OBJECT_RECORDER=PhysicalTraceRecorder(root,enabled=True,object_only=True,
            writer_workers=1,queue_maxsize=128,max_queued_bytes=16*1024*1024)
    elif _OBJECT_RECORDER.root!=root:
        raise ValueError('Cannot rebind object diagnostics in one process')
    return _OBJECT_RECORDER
