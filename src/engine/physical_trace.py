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
DECISION_TRACE_SCHEMA = "physical-decision-trace-1"


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


_AUDIO_BLOCK_TRACE_FIELDS = (
    "stream_id", "sequence", "sample_start_index", "sample_end_index_exclusive",
    "sample_count", "sample_rate", "channels", "backend", "read_start_monotonic",
    "read_complete_monotonic", "receive_wall_ts", "read_calls", "complete_block",
    "first_sample_receipt_estimate_monotonic", "last_sample_receipt_estimate_monotonic",
    "receipt_minus_stream_elapsed_s", "timestamp_semantics", "device_timestamp",
    "upstream_buffer_age_s",
)


def _audio_block_trace_fields(meta: Mapping[str, Any], *, trigger: bool = False) -> dict[str, Any]:
    fields = list(_AUDIO_BLOCK_TRACE_FIELDS)
    if trigger:
        fields.extend(("trigger_sample_offset", "maximum_sample_offset", "trigger_sample_index",
                       "trigger_receipt_estimate_monotonic", "detection_monotonic", "event_enqueued_monotonic",
                       "pending_depth_at_enqueue", "dispatch_monotonic", "dispatch_queue_age_ms",
                       "dispatch_batch_size", "dispatch_batch_index", "callback_start_monotonic",
                       "callback_end_monotonic", "callback_duration_ms", "shot_id_assigned_monotonic",
                       "pre_source_selected_monotonic", "shot_created_monotonic",
                       "deadline_monotonic"))
    return {key: _safe(meta.get(key)) for key in fields if key in meta}


def _audio_pipe_timing_snapshot(event: Any) -> dict[str, Any]:
    """Return a bounded metadata-only PCM neighborhood for opt-in audit traces."""
    trigger = getattr(event, "audio_trigger", None)
    trigger = trigger if isinstance(trigger, Mapping) else {}
    diagnostics = trigger.get("diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    acquisition = diagnostics.get("acquisition")
    acquisition = acquisition if isinstance(acquisition, Mapping) else None
    if acquisition is None or acquisition.get("sequence") is None:
        return {
            "state": "UNRECORDED",
            "reason": "audio_event_has_no_block_acquisition_identity",
            "device_timestamp": None,
            "upstream_buffer_age_s": None,
        }
    sequence = int(acquisition["sequence"])
    stream_id = acquisition.get("stream_id")
    requested_first = max(1, sequence - 160)
    try:
        from src.engine.audio.audio_peak_detector import audio_peak_detector
        preceding = [_audio_block_trace_fields(row) for row in audio_peak_detector.acquisition_metadata_window(
            requested_first, sequence, limit=161) if row.get("stream_id") == stream_id]
    except Exception as exc:
        return {
            "state": "UNRECORDED",
            "reason": f"metadata_snapshot_failed:{type(exc).__name__}",
            "trigger_sequence": sequence,
            "device_timestamp": None,
            "upstream_buffer_age_s": None,
        }
    return {
        "state": "AVAILABLE" if acquisition else "UNRECORDED",
        "semantics": "APP_PIPE_READ_METADATA_ONLY_NOT_ADC_OR_ACOUSTIC_IMPACT_TIME",
        "trigger_sequence": sequence,
        "stream_id": stream_id,
        "trigger_block": _audio_block_trace_fields(acquisition, trigger=True),
        "preceding_and_trigger_blocks": preceding,
        "pre_window": {
            "requested_first_sequence": requested_first,
            "requested_last_sequence": sequence,
            "expected_blocks": sequence - requested_first + 1,
            "observed_blocks": len(preceding),
            "first_observed_sequence": preceding[0].get("sequence") if preceding else None,
            "last_observed_sequence": preceding[-1].get("sequence") if preceding else None,
            "complete": ([row.get("sequence") for row in preceding]
                         == list(range(requested_first, sequence + 1))),
        },
        "post_blocks": [],
        "limits": {"pre_including_trigger_max_blocks": 161, "post_max_blocks": 64,
                   "pre_history_nominal_seconds": 2.56, "block_duration_seconds": 0.016},
        "post_capture_state": "PENDING_FINALIZATION",
        "device_timestamp": None,
        "upstream_buffer_age_s": None,
    }


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


def _decision_field_state(container: Mapping[str, Any], key: str) -> dict[str, Any]:
    if key not in container:
        return {"state": "UNRECORDED", "value": None}
    value = container.get(key)
    if value is None:
        return {"state": "MISSING", "value": None}
    if isinstance(value, float) and not math.isfinite(value):
        return {"state": "MISSING", "value": None}
    return {"state": "AVAILABLE", "value": _safe(value)}


def _build_decision_trace_v1(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Project already-captured authority ledgers into a versioned, explicit view.

    Called by the existing background trace finalizer. Candidate evidence values
    remain in the canonical track audit and are referenced, not duplicated.
    """
    decision = trace.get("decision_input", {})
    if not isinstance(decision, Mapping):
        decision = {}
    audit = decision.get("complete_track_audit")
    audit_source = "decision_input/complete_track_audit"
    if not isinstance(audit, Mapping):
        audit = trace.get("terminal_track_audit_v1")
        audit_source = "terminal_track_audit_v1"
    if not isinstance(audit, Mapping):
        audit = {}
    tracks = audit.get("tracks", [])
    if not isinstance(tracks, list):
        tracks = []
    frame_identity_by_ts = {}
    saved_frames = trace.get("frames", [])
    if not isinstance(saved_frames, list):
        saved_frames = []
    for frame in saved_frames:
        if not isinstance(frame, Mapping) or frame.get("timestamp") is None:
            continue
        acquisition = frame.get("acquisition")
        if isinstance(acquisition, Mapping) and acquisition.get("stream_id") is not None and acquisition.get("sequence") is not None:
            try:
                timestamp_key = float(frame["timestamp"])
                identity = {
                    "stream_id": acquisition["stream_id"], "sequence": acquisition["sequence"],
                    "timestamp": frame["timestamp"],
                }
                previous = frame_identity_by_ts.get(timestamp_key)
                if previous is None:
                    frame_identity_by_ts[timestamp_key] = identity
                elif previous != identity:
                    frame_identity_by_ts[timestamp_key] = "AMBIGUOUS"
            except (TypeError, ValueError):
                continue
    evidence_fields = (
        "score", "timestamp", "center_darkening", "local_contrast_gain", "blackhat_value",
        "v2_zscore", "v2_saliency", "area", "radius", "circularity",
        "v2225_local_confirm", "v2225_confirm_score", "v252_fresh_physical",
        "v252_confirm_score", "v252_authority_score", "v253_authority_ok",
        "v253_novelty_score", "v253_confirm_score", "v253_distance_novelty",
        "v251_region_evidence",
    )
    candidate_rows = []
    rejected = []
    for index, track in enumerate(tracks):
        if not isinstance(track, Mapping):
            continue
        candidate = track.get("last_candidate")
        if not isinstance(candidate, Mapping):
            candidate = {}
        tid = track.get("track_id")
        oid = track.get("observation_id")
        source_ts = candidate.get("timestamp")
        if source_ts is None:
            source_ts = candidate.get("source_frame_ts")
        frame_identity = None
        try:
            frame_identity = frame_identity_by_ts.get(float(source_ts))
        except (TypeError, ValueError):
            pass
        track_xy_present = track.get("camera_x") is not None and track.get("camera_y") is not None
        candidate_xy_present = candidate.get("camera_x") is not None and candidate.get("camera_y") is not None
        candidate_ref = f"/{audit_source}/tracks/{index}/last_candidate"
        frame_identity_state = ("AMBIGUOUS" if frame_identity == "AMBIGUOUS" else
                                "AVAILABLE" if frame_identity is not None else "UNRECORDED")
        frame_identity_reason = (
            "multiple saved frame identities share the source timestamp"
            if frame_identity_state == "AMBIGUOUS" else
            None if frame_identity_state == "AVAILABLE" else
            "no saved frame with matching source timestamp and acquisition identity"
        )
        row = {
            "track_id": {"state": "AVAILABLE" if tid is not None else "UNRECORDED", "value": tid},
            "candidate_identity": {"state": "AVAILABLE" if oid is not None else "UNRECORDED", "value": oid},
            "track_camera_xy": {"state": "AVAILABLE" if track_xy_present else "UNRECORDED",
                                 "value": [track.get("camera_x"), track.get("camera_y")] if track_xy_present else None},
            "candidate_camera_xy": {"state": "AVAILABLE" if candidate_xy_present else "UNRECORDED",
                                     "value": [candidate.get("camera_x"), candidate.get("camera_y")] if candidate_xy_present else None,
                                     "semantics": "unchanged coordinate from the recorded last candidate"},
            "producer": {"state": "AVAILABLE" if track.get("source") not in (None, "UNKNOWN") else "UNRECORDED",
                         "value": track.get("source")},
            "producer_shot_id": _decision_field_state(candidate, "v2224_producer_shot_id"),
            "source_frame_ids": {"state": frame_identity_state,
                                 "value": frame_identity if frame_identity_state == "AVAILABLE" else None,
                                 "reason": frame_identity_reason},
            "source_observation_timestamp": {
                **_decision_field_state(candidate, "timestamp"),
                "semantics": "raw candidate timestamp; source_frame_ts fallback is used only for source-frame identity join",
            },
            "source_frame_timestamp": {
                **_decision_field_state(candidate, "source_frame_ts"),
                "semantics": "raw candidate source_frame_ts; does not override timestamp for source-frame identity join",
            },
            "evidence_values_ref": {"state": "AVAILABLE" if candidate else "UNRECORDED", "path": candidate_ref},
            "evidence_field_states": {key: _decision_field_state(candidate, key) for key in evidence_fields},
            "eligibility": {"state": "AVAILABLE" if "eligible" in track else "UNRECORDED",
                            "value": track.get("eligible")},
            "causal_ownership": {"state": "AVAILABLE" if track.get("causal_ownership") is not None else "UNRECORDED",
                                 "value": track.get("causal_ownership")},
            "readiness": {"state": "AVAILABLE" if isinstance(track.get("readiness_inputs"), Mapping) else "UNRECORDED",
                          "value": _safe(track.get("readiness_inputs")) if isinstance(track.get("readiness_inputs"), Mapping) else None},
            "local_confirmation": {
                "state": ("AVAILABLE" if isinstance(track.get("local_confirmation"), Mapping) and track.get("local_confirmation")
                          else "MISSING" if isinstance(track.get("local_confirmation"), Mapping) else "UNRECORDED"),
                "value": _safe(track.get("local_confirmation")) if isinstance(track.get("local_confirmation"), Mapping) else None},
            "rejection_reason": {"state": "AVAILABLE" if track.get("rejection_reason") is not None else "NOT_APPLICABLE",
                                 "value": track.get("rejection_reason")},
            "rank": {"state": "AVAILABLE" if track.get("final_rank") is not None else "NOT_APPLICABLE",
                     "value": track.get("final_rank")},
            "selected": {"state": "AVAILABLE" if "selected" in track else "UNRECORDED",
                         "value": track.get("selected")},
        }
        candidate_rows.append(row)
        if track.get("eligible") is False:
            rejected.append({"track_id": row["track_id"], "candidate_identity": row["candidate_identity"],
                             "reason": row["rejection_reason"], "candidate_ref": candidate_ref})

    selected_id = audit.get("selected_track_id")
    selected_track = next((row for row in candidate_rows
                           if row["track_id"].get("value") == selected_id and selected_id is not None), None)
    runtime = trace.get("decision_trace_runtime_v1")
    if not isinstance(runtime, Mapping):
        runtime = {}
    policy = audit.get("policy")
    selection_available = bool(audit.get("complete"))
    outcome = trace.get("outcome", {})
    if not isinstance(outcome, Mapping):
        outcome = {}
    return {
        "schema": DECISION_TRACE_SCHEMA,
        "shot_id": {"state": "AVAILABLE" if trace.get("shot_id") is not None else "UNRECORDED", "value": trace.get("shot_id")},
        "selector": {
            "state": "AVAILABLE" if selection_available else "UNRECORDED",
            "owner": {"state": "AVAILABLE" if policy is not None else "UNRECORDED", "value": policy},
            "gate": ({"state": "NOT_APPLICABLE", "value": None}
                     if "gate" in audit and audit.get("gate") is None
                     else _decision_field_state(audit, "gate")),
            "path_hops": {"state": "AVAILABLE" if "selector_path_hops" in audit else "UNRECORDED",
                          "value": audit.get("selector_path_hops", []),
                          "truncated": bool(audit.get("selector_path_hops_truncated", False))},
            "decision_observed_at": _decision_field_state(audit, "decision_observed_at"),
            "selected_track_id": {"state": "AVAILABLE" if selected_id is not None else "NOT_APPLICABLE", "value": selected_id},
            "selected_candidate_ref": selected_track["evidence_values_ref"] if selected_track else {"state": "NOT_APPLICABLE", "path": None},
            "causal_window": {
                "audio_peak": _decision_field_state(trace, "peak_ts"),
                "association_lead_s": _decision_field_state(audit, "association_lead_s"),
                "association_lag_s": _decision_field_state(audit, "association_lag_s"),
                "semantics": "track ownership and onset are source ledger values; camera timestamps are receipt/observation timing, not exposure proof",
            },
        },
        "candidate_ledger": {
            "state": "AVAILABLE" if selection_available else "UNRECORDED",
            "source": audit_source,
            "tracks": candidate_rows,
            "selector_rejections": rejected,
            "upstream_dropped_proposals": {"state": "UNRECORDED", "reason": "not part of track audit candidate pool"},
        },
        "fallback_rescue": runtime.get("rescue", {"state": "UNRECORDED"}),
        "ai": runtime.get("ai_decision", {"state": "UNRECORDED"}),
        "emission": runtime.get("emission", {
            "state": "AVAILABLE" if outcome else "UNRECORDED",
            "emitted": outcome.get("emitted"),
            "camera_xy": {"state": "NOT_APPLICABLE" if outcome and not outcome.get("emitted") else "UNRECORDED",
                          "value": None},
            "result_state": outcome.get("status"),
        }),
        "timestamps": {
            "audio_peak": _decision_field_state(trace, "peak_ts"),
            "selector_cutoff": _decision_field_state(decision, "timestamp"),
            "selector_observed_at": _decision_field_state(runtime, "selector_observed_at"),
            "ai_observed_at": _decision_field_state(runtime.get("ai_decision", {}) if isinstance(runtime.get("ai_decision"), Mapping) else {}, "observed_at"),
            "emission_observed_at": _decision_field_state(runtime.get("emission", {}) if isinstance(runtime.get("emission"), Mapping) else {}, "observed_at"),
            "clock_semantics": "application wall-clock timestamps; not camera exposure or projectile impact time",
        },
    }


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
        self.decision_trace_enabled = False
        self.object_only = bool(object_only)
        self.capture_history = validate_capture_history(capture_history)
        self._history_environment = None
        self._capture_event_limit_reached = False
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max(1, int(queue_maxsize)))
        self._active: dict[int, dict[str, Any]] = {}
        self._review_paths: set[Path] = set()
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
            self.decision_trace_enabled=False
            return  # process-only gameplay; never write into saved historical roots
        values = settings if isinstance(settings, Mapping) else {}
        self.enabled = bool(values.get("physical_trace_capture_enabled", False))
        # V1 decision details are a separate, opt-in layer. The existing trace
        # remains disabled by default and must also be enabled for persistence.
        self.decision_trace_enabled = bool(self.enabled and (
            values.get("physical_decision_trace_enabled", False)
            or os.environ.get("SKJUTBANA_PHYSICAL_DECISION_TRACE") == "1"))
        if values.get("physical_trace_root"):
            self.root = Path(str(values["physical_trace_root"]))
        if os.environ.get('SKJUTBANA_PHYSICAL_CAPTURE_BINDING'):
            # main.py verified/claimed the new binding before hardware imports.
            root=os.environ.get('SKJUTBANA_PHYSICAL_TRACE_ROOT')
            if not root:raise ValueError('capture binding has not passed startup preflight')
            self.root=Path(root);self.enabled=True
            self.decision_trace_enabled = bool(
                values.get("physical_decision_trace_enabled", False)
                or os.environ.get("SKJUTBANA_PHYSICAL_DECISION_TRACE") == "1")
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
                elif job["kind"] == "review_frame":
                    # Separate post-decision human-inspection artifact. Never
                    # append to trace.frames or expose to detector replay.
                    started = time.perf_counter()
                    array = job['value']
                    temporary = path.with_name(path.name + '.tmp')
                    with temporary.open('wb') as stream:
                        np.save(stream, array, allow_pickle=False)
                    os.replace(temporary, path)
                    metadata = dict(job['meta'], image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        pixel_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                        writer_ms=(time.perf_counter()-started)*1000., status='SAVED')
                    sidecar = path.with_suffix('.json')
                    temporary = sidecar.with_name(sidecar.name+'.tmp')
                    temporary.write_text(_json(metadata), encoding='utf-8')
                    os.replace(temporary, sidecar)
                elif job["kind"] == "array":
                    temporary = path.with_name(path.name + ".tmp")
                    with temporary.open("wb") as stream:
                        np.save(stream, np.asarray(job["value"]), allow_pickle=False)
                    os.replace(temporary, path)
                elif job["kind"] == "json":
                    temporary = path.with_name(path.name + ".tmp")
                    payload = job.get('encoded_json')
                    temporary.write_text(payload if payload is not None else _json(job["value"]), encoding="utf-8")
                    os.replace(temporary, path)
                else:
                    raise ValueError(f"unknown trace writer job kind: {job['kind']}")
                if job['kind'] == 'review_frame':
                    job['persisted'] = True  # Never append to any live trace.frames.
                else:
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

    def enqueue_review_frame(self, path, pixels, metadata):
        """Full native image via existing bounded writer, post-decision ONLY.

        Caller owns the terminal-shot/presentation barrier. A separate schema,
        role and path prevent review pixels from entering live trace.frames.
        Copy cost is measured here; hashes/NPY I/O run in the existing writer.
        """
        if (metadata.get('schema') != 'physical_gt_review_frame_v1'
                or metadata.get('live_authority') is not False
                or metadata.get('role') not in ('PRE','LIVE_POST','GT_SETTLED')):
            raise ValueError('explicit non-authoritative review frame contract required')
        return self._enqueue_inspection_frame(path, pixels, metadata)

    def enqueue_optical_frame(self, path, pixels, metadata):
        """Research optical pixels, never GT or ordinary detector trace.frames."""
        if (metadata.get('schema') != 'physical_optical_frame_v1'
                or metadata.get('live_authority') is not False
                or metadata.get('projectile_truth') is not False
                or metadata.get('shot_id') is not None
                or metadata.get('role') not in ('OPTICAL_PASSIVE','OPTICAL_BASELINE','OPTICAL_PROBE','OPTICAL_RECOVERY')):
            raise ValueError('explicit non-authoritative optical frame contract required')
        return self._enqueue_inspection_frame(path, pixels, metadata)

    def enqueue_render_frame(self, path, pixels, metadata):
        """Immutable engine image, SCREEN pixels; never camera evidence or GT."""
        if (metadata.get('schema') != 'physical_render_content_v1'
                or metadata.get('live_authority') is not False
                or metadata.get('projectile_truth') is not False
                or metadata.get('shot_id') is not None
                or metadata.get('role') != 'ENGINE_RENDER_CONTENT'
                or metadata.get('coordinate_space') != 'PYGAME_CLIENT_PIXELS'):
            raise ValueError('explicit non-authoritative screen image contract required')
        return self._enqueue_inspection_frame(path, pixels, metadata, camera=False)

    def _enqueue_inspection_frame(self, path, pixels, metadata, *, camera=True):
        """Shared bounded native image writer; callers validate separate roles."""
        path = Path(path)
        if (not path.resolve().is_relative_to(self.root.resolve()) or path.exists()
                or path.with_suffix('.json').exists()):
            raise ValueError('review artifact must be new and under recorder root')
        with self._lock:
            if path.resolve() in self._review_paths:
                raise ValueError('review path already reserved; never overwrite an enqueued frame')
            self._review_paths.add(path.resolve())
        started = time.perf_counter()
        frozen = np.asarray(pixels).copy()
        if frozen.dtype != np.uint8 or frozen.ndim != 3 or frozen.shape[2] != 3:
            raise ValueError('native BGR uint8 image required')
        meta = dict(metadata, shape=list(frozen.shape), dtype=str(frozen.dtype), color_space='BGR',
                    copy_ms=(time.perf_counter()-started)*1000.)
        meta['full_camera_size' if camera else 'render_size'] = [frozen.shape[1],frozen.shape[0]]
        return self._enqueue(dict(kind='review_frame', path=path, value=frozen,
                                  nbytes=frozen.nbytes, shot_id=metadata['shot_id'], meta=meta))

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
                if self.decision_trace_enabled:
                    # A small metadata-only copy from the existing audio ring.
                    # PCM bytes, disk I/O and event decisions are not touched.
                    trace['audio_pipe_timing_v1'] = _audio_pipe_timing_snapshot(event)
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
                is_camera_receive_time = (
                    getattr(scanner, "_pre_shot_snapshot_source", None) == "camera_ring"
                    and getattr(scanner, "_pre_shot_snapshot_timestamp_semantics", None)
                    == "APPLICATION_RECEIVE_TIME_NOT_EXPOSURE"
                )
                self._save_frame(shot_id, "pre_snapshot", pre_ts, pre_snapshot,
                                 acquisition_eligible=is_camera_receive_time)
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

    def _save_frame(self, shot_id: int, kind: str, timestamp: float, frame: np.ndarray,
                    *, acquisition_eligible: bool = True) -> None:
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
            meta['acquisition'] = camera_manager.provenance_for_timestamp(timestamp) if acquisition_eligible else None
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
                if self.decision_trace_enabled:
                    scanner.physical_decision_trace_v1_enabled = True
                    active["trace"]["decision_trace_runtime_v1"] = {
                        "schema": DECISION_TRACE_SCHEMA,
                        "shot_id": int(event.shot_id),
                        "selector_observed_at": time.time(),
                        "ai_decision": {"state": "UNRECORDED"},
                        "emission": {"state": "UNRECORDED"},
                        "rescue": self._rescue_state(int(event.shot_id)),
                    }

    @staticmethod
    def _rescue_state(shot_id: int) -> dict[str, Any]:
        try:
            from src.engine.shot_fast_v2225 import rescue_router_v2225
            return {
                "state": "AVAILABLE",
                "requested": bool(rescue_router_v2225.requested(shot_id)),
                "consumed": bool(rescue_router_v2225.was_consumed(shot_id)),
                "source": "V2.22.5 RescueRouterV2225",
            }
        except Exception as exc:
            return {"state": "UNRECORDED", "reason": type(exc).__name__}

    def record_ai_decision(self, scanner: Any, event: Any, *, mode: str, decision: Mapping[str, Any] | None) -> None:
        """Store chooser observation only; does not apply or interpret authority."""
        if not self.decision_trace_enabled:
            return
        with self._lock:
            active = self._active.get(int(getattr(event, "shot_id", 0) or 0))
            if active is None:
                return
            runtime = active["trace"].setdefault("decision_trace_runtime_v1", {
                "schema": DECISION_TRACE_SCHEMA, "shot_id": int(event.shot_id),
                "selector_observed_at": None, "rescue": self._rescue_state(int(event.shot_id))})
            value = dict(decision or {})
            proposed_xy = None
            xy_state = "NOT_APPLICABLE"
            if value.get("camera_x") is not None and value.get("camera_y") is not None:
                proposed_xy = [float(value["camera_x"]), float(value["camera_y"])]
                xy_state = "AVAILABLE"
            runtime["ai_decision"] = {
                "state": "AVAILABLE" if decision is not None else "UNRECORDED",
                "mode": str(mode) if mode is not None else None,
                "apply": bool(value.get("apply", False)) if decision is not None else None,
                "reason": value.get("reason"),
                "proposed_camera_xy": {"state": xy_state, "value": proposed_xy},
                "confidence": {"state": "AVAILABLE" if value.get("confidence") is not None else "UNRECORDED",
                               "value": value.get("confidence")},
                "observed_at": time.time(),
            }
            runtime["rescue"] = self._rescue_state(int(event.shot_id))

    def record_emission_decision(self, scanner: Any, event: Any, track: Any) -> None:
        """Record actual HitInput result after the unchanged emitter returns."""
        if not self.decision_trace_enabled:
            return
        sid = int(getattr(event, "shot_id", 0) or 0)
        emitted = bool(getattr(event, "emitted", False))
        timing = getattr(event, "audio_trigger", {}).get("timing", {})
        push_xy = timing.get("physical_decision_trace_push_camera_xy") if isinstance(timing, Mapping) else None
        push_attempt = {
            "state": "AVAILABLE" if isinstance(push_xy, (list, tuple)) and len(push_xy) == 2 else "UNRECORDED",
            "camera_xy": ([float(push_xy[0]), float(push_xy[1])]
                          if isinstance(push_xy, (list, tuple)) and len(push_xy) == 2 else None),
            "call_state": (timing.get("physical_decision_trace_push_call_state", "UNRECORDED")
                           if isinstance(timing, Mapping) else "UNRECORDED"),
            "exception_type": (timing.get("physical_decision_trace_push_exception_type")
                               if isinstance(timing, Mapping) else None),
            "semantics": "arguments supplied to HitInput.push_camera_hit; RETURNED is a successful function return, not proof of downstream gameplay handling",
        }
        xy = ([float(push_xy[0]), float(push_xy[1])]
              if emitted and isinstance(push_xy, (list, tuple)) and len(push_xy) == 2 else None)
        matched_hole = timing.get("physical_decision_trace_matched_hole") if isinstance(timing, Mapping) else None
        if isinstance(matched_hole, Mapping):
            matched_hole_value = {"hole_id": matched_hole.get("hole_id"),
                                  "camera_xy": _safe(matched_hole.get("camera_xy"))}
            matched_hole_state = "AVAILABLE"
        else:
            matched_hole_value = None
            matched_hole_state = "NOT_APPLICABLE" if getattr(event, "state", None) != "rehit" else "UNRECORDED"
        candidate_xy = None
        if track is not None and getattr(track, "camera_x", None) is not None and getattr(track, "camera_y", None) is not None:
            candidate_xy = [float(track.camera_x), float(track.camera_y)]
        with self._lock:
            active = self._active.get(sid)
            if active is None:
                return
            runtime = active["trace"].setdefault("decision_trace_runtime_v1", {
                "schema": DECISION_TRACE_SCHEMA, "shot_id": sid, "ai_decision": {"state": "UNRECORDED"},
                "rescue": self._rescue_state(sid)})
            runtime["emission"] = {
                "state": "AVAILABLE", "emitted": emitted,
                "selected_track_id": {"state": "AVAILABLE" if track is not None else "UNRECORDED",
                                      "value": getattr(track, "track_id", None)},
                "matched_track_id": {"state": "AVAILABLE" if getattr(event, "matched_track_id", None) is not None else "NOT_APPLICABLE",
                                     "value": getattr(event, "matched_track_id", None)},
                "camera_xy": {"state": "AVAILABLE" if xy is not None else ("NOT_APPLICABLE" if not emitted else "UNRECORDED"),
                              "value": xy},
                "push_attempt": push_attempt,
                "candidate_camera_xy": {"state": "AVAILABLE" if candidate_xy is not None else "UNRECORDED",
                                        "value": candidate_xy,
                                        "semantics": "selected track XY before emission substitution"},
                "matched_hole": {"state": matched_hole_state, "value": matched_hole_value,
                                 "semantics": "copied known-hole identity and XY used by re-hit branch"},
                "result_state": str(getattr(event, "state", "UNAVAILABLE")),
                "observed_at": time.time(),
                "xy_semantics": ("arguments passed to HitInput.push_camera_hit" if xy is not None
                                 else "actual push arguments not recorded"),
            }
            runtime["rescue"] = self._rescue_state(sid)

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
            current_trigger = _safe(getattr(event, "audio_trigger", None))
            if isinstance(current_trigger, Mapping):
                # Refresh only bounded metadata: AudioPeakDetector mutates this
                # acquisition record through callback completion on the main thread.
                trace["audio_trigger"] = current_trigger
                acquisition = ((current_trigger.get("diagnostics") or {}).get("acquisition")
                               if isinstance(current_trigger.get("diagnostics"), Mapping) else None)
                trace["event_timing_v1"] = {
                    "state": "AVAILABLE" if isinstance(acquisition, Mapping) else "UNRECORDED",
                    "clock": "monotonic software boundaries; no device/exposure time",
                    "audio": _safe(acquisition) if isinstance(acquisition, Mapping) else None,
                    "shot": _safe(current_trigger.get("timing", {})),
                    "shot_id": int(shot_id),
                    "created_monotonic": getattr(event, "created_monotonic", None),
                    "decision_deadline_monotonic": getattr(event, "decision_deadline_monotonic", None),
                    "lifecycle_state": getattr(event, "lifecycle_state", "UNRECORDED"),
                    "pre_source": current_trigger.get("pre_source", "UNRECORDED"),
                    "pre_timestamp": current_trigger.get("pre_timestamp"),
                    "pre_timestamp_semantics": current_trigger.get("pre_timestamp_semantics", "UNRECORDED"),
                }
                if self.decision_trace_enabled:
                    trace["audio_pipe_timing_v1"] = _audio_pipe_timing_snapshot(event)
            emitted = bool(getattr(event, "emitted", False))
            if self.decision_trace_enabled:
                event_timing = ((current_trigger.get("timing") or {})
                                if isinstance(current_trigger, Mapping) else {})
                push_xy = event_timing.get("physical_decision_trace_push_camera_xy") \
                    if isinstance(event_timing, Mapping) else None
                final_camera_xy = ({"camera_x": float(push_xy[0]), "camera_y": float(push_xy[1])}
                                   if emitted and isinstance(push_xy, (list, tuple)) and len(push_xy) == 2
                                   else None)
            else:
                final_camera_xy = (_safe(getattr(scanner, "last_best_candidate", None))
                                   if emitted and (debug_shot is None or int(debug_shot) == int(shot_id)) else None)
            trace["outcome"] = {"status": str(getattr(event, "state", "finished")), "emitted": emitted,
                "lifecycle_state": getattr(event, "lifecycle_state", "UNRECORDED"),
                "matched_track_id": getattr(event, "matched_track_id", None), "matched_hole_id": getattr(event, "matched_hole_id", None),
                "confidence": getattr(event, "confidence", None), "note": str(getattr(event, "note", "")),
                "final_camera_xy": final_camera_xy,
                "final_camera_xy_semantics": ("HitInput.push_camera_hit arguments" if self.decision_trace_enabled
                                               else "legacy scanner debug cache"),
                "rescue_used": bool(getattr(scanner, "last_window_debug", {}).get("rescue_used", False)),
                "detector_e2e_latency_ms": detector_e2e, "trace_completion_latency_ms": trace_completion,
                "runtime_event_debug": debug, "timeout": str(getattr(event, "state", "")) == "missed", "ground_truth": None}
            if self.decision_trace_enabled:
                runtime = trace.setdefault("decision_trace_runtime_v1", {
                    "schema": DECISION_TRACE_SCHEMA,
                    "shot_id": int(shot_id),
                    "ai_decision": {"state": "UNRECORDED"},
                    "rescue": self._rescue_state(int(shot_id)),
                })
                if "emission" not in runtime:
                    emitted = bool(getattr(event, "emitted", False))
                    runtime["emission"] = {
                        "state": "AVAILABLE",
                        "emitted": emitted,
                        "selected_track_id": {"state": "UNRECORDED", "value": None},
                        "camera_xy": {"state": "UNRECORDED" if emitted else "NOT_APPLICABLE", "value": None},
                        "push_attempt": {
                            "state": "AVAILABLE" if isinstance(current_trigger, Mapping) and isinstance(
                                (current_trigger.get("timing") or {}).get("physical_decision_trace_push_camera_xy"), (list, tuple)) else "UNRECORDED",
                            "camera_xy": ((current_trigger.get("timing") or {}).get("physical_decision_trace_push_camera_xy")
                                          if isinstance(current_trigger, Mapping) else None),
                            "call_state": ((current_trigger.get("timing") or {}).get("physical_decision_trace_push_call_state", "UNRECORDED")
                                           if isinstance(current_trigger, Mapping) else "UNRECORDED"),
                            "exception_type": ((current_trigger.get("timing") or {}).get("physical_decision_trace_push_exception_type")
                                               if isinstance(current_trigger, Mapping) else None),
                            "semantics": "arguments supplied to HitInput.push_camera_hit; RETURNED is a successful function return, not proof of downstream gameplay handling",
                        },
                        "result_state": str(getattr(event, "state", "finished")),
                        "observed_at": time.time(),
                        "reason": "emission_hook_not_observed" if emitted else "no_emission_observed",
                    }
                runtime["rescue"] = self._rescue_state(int(shot_id))
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

    def capture_mature_pool(self, shot_id, stage, candidates, frame_ts, available_at,
                            *, full_rescue=False, diagnostics=None):
        """Bounded additive sidecars, including late deliveries after OBJECT closes.

        Native dictionaries/order are copied, never mutated. No images or new
        detector work. Uses this recorder's existing queue/lifetime; no new owner.
        Missing/truncated exports cannot authorize a unique shadow object.
        """
        if not self.object_only or not self.enabled:return False
        with self._lock:
            active=self._active.get(shot_id)
            if active is None:return False
            seq=active.get('mature_pool_count',0)
            if active.get('mature_pool_limit_reported'):return False
            packet=dict(schema='object-mature-pool-1',shot_id=shot_id,
                peak_ts=active['trace']['peak_ts'],stage=stage,coordinate_space='camera',
                frame_ts=float(frame_ts),available_at=float(available_at),
                availability_semantics='main_thread_delivery_before_tracking; not worker frame time',
                delivery_thread=threading.current_thread().name,full_rescue=bool(full_rescue),
                input_count=len(candidates),complete=len(candidates)<=1024,
                candidates=_copy_candidates(candidates[:1024]),diagnostics=_safe(diagnostics or {}))
            encoded=_json(packet)
            if len(encoded.encode('utf-8'))>2*1024*1024:
                packet.update(candidates=[],diagnostics={},complete=False,unavailable_reason='pool_byte_limit')
                encoded=_json(packet)
            used=active.get('mature_pool_bytes',0)
            size=len(encoded.encode('utf-8'))
            limit=seq>=32 or used+size>8*1024*1024-4096  # reserve the small limit marker
            if limit:
                active['mature_pool_limit_reported']=True
                packet.update(candidates=[],diagnostics={},complete=False,unavailable_reason='event_export_limit')
                encoded=_json(packet)
            name='LIMIT.json' if limit else f'pool_{seq:04d}.json'
            active['mature_pool_count']=seq+1
            size=len(encoded.encode('utf-8'));active['mature_pool_bytes']=used+size
            return self._enqueue(dict(kind='json',path=active['directory']/'mature_candidates'/name,
                value=packet,encoded_json=encoded,nbytes=size,shot_id=shot_id,meta=dict(artifact='mature_candidates')))

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
            audio_timing = trace.get('audio_pipe_timing_v1')
            if self.decision_trace_enabled and isinstance(audio_timing, dict):
                trigger_sequence = audio_timing.get('trigger_sequence')
                if isinstance(trigger_sequence, int):
                    try:
                        from src.engine.audio.audio_peak_detector import audio_peak_detector
                        post_blocks = [_audio_block_trace_fields(row) for row in audio_peak_detector.acquisition_metadata_window(
                            trigger_sequence + 1, trigger_sequence + 64, limit=64)
                            if row.get('stream_id') == audio_timing.get('stream_id')]
                        audio_timing['post_blocks'] = post_blocks
                        observed_sequences = [row.get('sequence') for row in post_blocks]
                        audio_timing['post_window'] = {
                            'requested_first_sequence': trigger_sequence + 1,
                            'requested_last_sequence': trigger_sequence + 64,
                            'observed_blocks': len(post_blocks),
                            'first_observed_sequence': observed_sequences[0] if observed_sequences else None,
                            'last_observed_sequence': observed_sequences[-1] if observed_sequences else None,
                            'complete': observed_sequences == list(range(trigger_sequence + 1, trigger_sequence + 65)),
                        }
                        audio_timing['post_capture_state'] = (
                            'COMPLETE' if audio_timing['post_window']['complete'] else
                            'PARTIAL' if post_blocks else 'MISSING')
                        audio_timing['post_capture_reason'] = (
                            None if post_blocks else 'no_later_block_available_at_trace_finalization')
                    except Exception as exc:
                        audio_timing['post_capture_state'] = 'UNRECORDED'
                        audio_timing['post_capture_reason'] = f'metadata_snapshot_failed:{type(exc).__name__}'
                elif audio_timing.get('state') == 'UNRECORDED':
                    audio_timing['post_capture_state'] = 'UNRECORDED'
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
            if self.decision_trace_enabled:
                trace["decision_trace_v1"] = _build_decision_trace_v1(trace)
                trace.pop("decision_trace_runtime_v1", None)
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


def capture_object_mature_pool(shot_id, stage, candidates, frame_ts, *, full_rescue=False, diagnostics=None):
    """Optional main-thread observation only; never initializes capture/hardware."""
    if os.environ.get('SKJUTBANA_OBJECT_MATURE_CAPTURE')!='1' or _OBJECT_RECORDER is None:return
    available_at=time.time()
    started=time.perf_counter()
    try:
        _OBJECT_RECORDER.capture_mature_pool(int(shot_id),stage,candidates,frame_ts,available_at,
            full_rescue=full_rescue,diagnostics=diagnostics)
    except Exception as exc:
        print(f'[OBJECT-MATURE-CAPTURE] unavailable: {type(exc).__name__}: {exc}',flush=True)
    finally:
        print(f'[OBJECT-MATURE-CAPTURE] shot={shot_id} stage={stage} candidates={len(candidates)} '
              f'copy_enqueue_ms={(time.perf_counter()-started)*1000.:.3f}',flush=True)


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
