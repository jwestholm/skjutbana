"""
AI Runtime for hit detection learning.

Design principles:
- Minimal engine coupling: only reads from hit_scanner and camera_manager singletons
- All heavy logic stays inside this module
- Feature vectors are consistent between training and scoring
- Memory is bounded and persisted to content/ai/memory.json
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import cv2
except Exception:
    cv2 = None

import numpy as np


Point = Tuple[float, float]
Candidate = Dict[str, Any]

AI_DIR = Path("content/ai")
MEMORY_FILE = AI_DIR / "memory.json"
SETTINGS_FILE = AI_DIR / "settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "enabled": True,
    "mode": "train_only",  # off | train_only | advisory | blended | ai_priority | ai_only
    "top_k": 10,
    "memory_limit_positive": 400,
    "memory_limit_negative": 1200,
    "click_match_radius_px": 70.0,
    "min_confidence": 0.58,
    "override_confidence": 0.92,
    "max_negatives_per_click": 3,
    "trust_percent": 0,
    "show_overlay": True,
    "auto_learn": True,
    "gt_match_radius_px": 10.0,
    "save_hole_images": True,
    "candidate_limit": 200,
    "supplement_candidates_enabled": True,
    "supplement_min_candidates": 120,
    "supplement_peak_percentile": 99.6,
    "sampling_mode": "center_bias",  # center_bias | uniform | edge_bias | corners
    "benchmark_mode": False,  # True = eval only, no model updates during F1/F2
}

# ---- Feature keys (consistent between training and scoring) ----
FEATURE_KEYS = [
    "detector_score",
    "area",
    "radius",
    "circularity",
    "center_change",
    "local_contrast",
    "pre_shot_change",
    "change_value",
    "patch_mean",
    "patch_std",
    "edge_strength",
    "x_norm",
    "y_norm",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass(frozen=True)
class CandidateImageEvidence:
    """Small, immutable image samples tied to one detector candidate.

    Full camera frames are intentionally not stored per candidate. A compact
    patch is enough for feature extraction and prevents later frames from
    silently changing the visual evidence used by the AI.
    """

    evidence_id: int
    shot_id: int
    source_frame_ts: float
    source_shape: Tuple[int, int]
    patch_bounds: Tuple[int, int, int, int]
    source_patch: Optional[np.ndarray] = None
    pre_patch: Optional[np.ndarray] = None
    diff_patch: Optional[np.ndarray] = None


@dataclass
class AIShotContext:
    """Camera observations that belong to one and only one audio shot."""

    shot_id: int
    peak_ts: float
    created_at: float
    pre_shot_gray: Optional[np.ndarray] = None
    pre_shot_ts: float = 0.0
    candidates: List[Candidate] = field(default_factory=list)
    candidate_evidence: Dict[int, CandidateImageEvidence] = field(default_factory=dict)
    candidate_frame_ts: float = 0.0
    post_shot_gray: Optional[np.ndarray] = None
    post_shot_frames: List[Tuple[np.ndarray, float]] = field(default_factory=list)
    last_post_frame_ts: float = 0.0
    state: str = "pending"


# ======================================================================
# SimpleAIMemory — bounded online learner with feature normalization
# ======================================================================

class SimpleAIMemory:
    """
    Stores compact feature dicts and scores new candidates by comparing
    against stored positive/negative examples.

    Improvements over previous version:
    - Features stored as dicts with named keys (not positional lists)
    - Per-feature normalization using running min/max
    - Time-weighted scoring (recent memories matter more)
    - Balanced negative sampling
    """

    def __init__(self, positive_limit: int = 400, negative_limit: int = 1200) -> None:
        self.positive_limit = int(positive_limit)
        self.negative_limit = int(negative_limit)
        self.positives: List[Dict[str, Any]] = []
        self.negatives: List[Dict[str, Any]] = []
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        self.stats: Dict[str, Any] = {
            "positive_count": 0,
            "negative_count": 0,
            "total_clicks": 0,
            "last_updated": None,
            "last_import_ts": None,
            "local_updates_since_import": 0,
            "version": 2,
        }
        self.load()

    def load(self) -> None:
        if not MEMORY_FILE.exists():
            return
        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            self.positives = list(data.get("positives", []))[-self.positive_limit:]
            self.negatives = list(data.get("negatives", []))[-self.negative_limit:]
            self.feature_ranges = {
                k: (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) and len(v) == 2 else (0.0, 0.0)
                for k, v in dict(data.get("feature_ranges", {})).items()
            }
            saved_stats = data.get("stats", {})
            if isinstance(saved_stats, dict):
                self.stats.update(saved_stats)
        except Exception:
            self.positives = []
            self.negatives = []

    def save(self) -> None:
        AI_DIR.mkdir(parents=True, exist_ok=True)
        self.stats["positive_count"] = len(self.positives)
        self.stats["negative_count"] = len(self.negatives)
        self.stats["last_updated"] = time.time()
        payload = {
            "positives": self.positives[-self.positive_limit:],
            "negatives": self.negatives[-self.negative_limit:],
            "feature_ranges": self.feature_ranges,
            "stats": self.stats,
        }
        MEMORY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def export_brain(self, path: Path) -> None:
        """Export the full memory to a portable file."""
        self.save()
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        data["exported_at"] = time.time()
        data["stats"]["exported_positive"] = len(self.positives)
        data["stats"]["exported_negative"] = len(self.negatives)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def import_brain(self, path: Path) -> Dict[str, Any]:
        """Import a brain file, replacing current memory."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self.positives = list(data.get("positives", []))[-self.positive_limit:]
        self.negatives = list(data.get("negatives", []))[-self.negative_limit:]
        raw_ranges = dict(data.get("feature_ranges", {}))
        self.feature_ranges = {
            k: (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) and len(v) == 2 else (0.0, 0.0)
            for k, v in raw_ranges.items()
        }
        imported_stats = data.get("stats", {})
        self.stats["last_import_ts"] = time.time()
        self.stats["local_updates_since_import"] = 0
        self.stats["positive_count"] = len(self.positives)
        self.stats["negative_count"] = len(self.negatives)
        self.save()
        return {
            "imported_positive": len(self.positives),
            "imported_negative": len(self.negatives),
        }

    def reset(self) -> None:
        self.positives.clear()
        self.negatives.clear()
        self.feature_ranges.clear()
        self.stats["total_clicks"] = 0
        self.stats["local_updates_since_import"] = 0
        self.save()

    def add_positive(self, features: Dict[str, float], meta: Optional[Dict[str, Any]] = None) -> None:
        self._update_ranges(features)
        self.positives.append({
            "features": features,
            "meta": meta or {},
            "timestamp": time.time(),
        })
        if len(self.positives) > self.positive_limit:
            self.positives = self.positives[-self.positive_limit:]

    def add_negative(self, features: Dict[str, float], meta: Optional[Dict[str, Any]] = None) -> None:
        self._update_ranges(features)
        self.negatives.append({
            "features": features,
            "meta": meta or {},
            "timestamp": time.time(),
        })
        if len(self.negatives) > self.negative_limit:
            self.negatives = self.negatives[-self.negative_limit:]

    def _update_ranges(self, features: Dict[str, float]) -> None:
        for key, val in features.items():
            v = _safe_float(val)
            if key in self.feature_ranges:
                lo, hi = self.feature_ranges[key]
                self.feature_ranges[key] = (min(lo, v), max(hi, v))
            else:
                self.feature_ranges[key] = (v, v)

    def _normalize(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features to 0-1 range using stored min/max."""
        result = {}
        for key in FEATURE_KEYS:
            raw = _safe_float(features.get(key, 0.0))
            if key in self.feature_ranges:
                lo, hi = self.feature_ranges[key]
                span = hi - lo
                if span > 1e-9:
                    result[key] = (raw - lo) / span
                else:
                    result[key] = 0.5
            else:
                result[key] = raw
        return result

    def score(self, features: Dict[str, float]) -> float:
        """Score a candidate: 0.0 = definitely not a hit, 1.0 = definitely a hit."""
        if not self.positives and not self.negatives:
            return 0.5

        norm = self._normalize(features)
        now = time.time()

        # Use recent memories for scoring (bounded for performance)
        pos_sample = self.positives[-64:]
        neg_sample = self.negatives[-128:]

        # Vectorized distance computation for better performance
        pos_best = self._best_distance(norm, pos_sample, now)
        neg_best = self._best_distance(norm, neg_sample, now)

        # Sigmoid-ish mapping: closer to positives = higher score
        raw = 0.5 + (neg_best - pos_best) / 6.0
        return max(0.0, min(1.0, raw))

    def _best_distance(
        self, norm_features: Dict[str, float], memories: List[Dict[str, Any]], now: float
    ) -> float:
        """Find minimum weighted distance to a set of memories."""
        if not memories:
            return 4.0

        # Build feature vector once
        query = [norm_features.get(key, 0.5) for key in FEATURE_KEYS]
        best = 4.0

        for mem in memories:
            mem_features = self._normalize(mem.get("features", {}))
            total = 0.0
            for i, key in enumerate(FEATURE_KEYS):
                diff = query[i] - mem_features.get(key, 0.5)
                total += diff * diff
            dist = math.sqrt(total / len(FEATURE_KEYS))

            # Time decay: memories older than 1 hour get slightly penalized
            age_hours = (now - _safe_float(mem.get("timestamp", now))) / 3600.0
            if age_hours > 0:
                dist *= 1.0 + 0.05 * age_hours

            if dist < best:
                best = dist

        return best

    def summary(self) -> Dict[str, Any]:
        return {
            "positive_count": len(self.positives),
            "negative_count": len(self.negatives),
            "total_clicks": self.stats.get("total_clicks", 0),
            "last_updated": self.stats.get("last_updated"),
            "last_import_ts": self.stats.get("last_import_ts"),
            "local_updates_since_import": self.stats.get("local_updates_since_import", 0),
            "feature_keys": len(self.feature_ranges),
        }


# ======================================================================
# AIRuntime — main interface between AI and the rest of the engine
# ======================================================================

class AIRuntime:
    def __init__(self, storage_dir: str = "content/ai") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.settings = dict(DEFAULT_SETTINGS)
        if SETTINGS_FILE.exists():
            try:
                loaded = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.settings.update(loaded)
            except Exception:
                pass

        self.memory = SimpleAIMemory(
            positive_limit=int(self.settings.get("memory_limit_positive", 400)),
            negative_limit=int(self.settings.get("memory_limit_negative", 1200)),
        )

        # Observation state — updated every frame by bootstrap patch.
        # _shots is authoritative. The _latest_* fields are maintained for
        # backwards compatibility with the training UI.
        self._last_audio_count: int = 0
        self._shots: Dict[int, AIShotContext] = {}
        self._next_evidence_id: int = 1
        self._active_shot_id: Optional[int] = None
        self._last_completed_shot_id: Optional[int] = None
        self._latest_candidates: List[Candidate] = []
        self._latest_gray: Optional[np.ndarray] = None
        self._latest_frame_ts: float = 0.0
        self._pre_shot_gray: Optional[np.ndarray] = None
        self._pre_shot_ts: float = 0.0
        self._post_shot_gray: Optional[np.ndarray] = None
        self._post_shot_frames: List[Tuple[np.ndarray, float]] = []  # (gray, timestamp)
        self._shot_ts: float = 0.0  # Audio peak timestamp for the active shot
        self._latest_snapshot: Optional[Dict[str, Any]] = None
        self._shot_detected: bool = False

        # Funnel diagnostics
        from src.engine.ai.diagnostics import FunnelTracker
        self.funnel = FunnelTracker()

        # Session tracking
        self.session_stats: Dict[str, Any] = self._new_session_stats()

    @staticmethod
    def _new_session_stats() -> Dict[str, Any]:
        """Return a complete, fresh set of per-session counters.

        The AI training scene must use this factory instead of replacing the
        dictionary with an older, partial set of keys. Keeping the defaults in
        one place prevents new diagnostics counters from causing KeyError after
        F1/F2 starts a fresh training run.
        """
        return {
            "shots_seen": 0,
            "clicks": 0,
            "last_click_camera": None,
            "shot_contexts_created": 0,
            "stale_candidate_blocks": 0,
            "duplicate_post_frames_skipped": 0,
            "emission_syncs": 0,
            "candidate_source_patches": 0,
            "candidate_pre_patches": 0,
            "candidate_diff_patches": 0,
            "candidate_patch_misses": 0,
            "candidate_patch_fallbacks": 0,
            "supplemental_frames_processed": 0,
            "supplemental_candidates_generated": 0,
            "supplemental_candidates_merged": 0,
            "supplemental_temporal_candidates": 0,
            "supplemental_appearance_candidates": 0,
        }

    def reset_session_stats(self) -> None:
        """Reset all per-session counters without dropping newer keys."""
        self.session_stats = self._new_session_stats()


    def save_settings(self) -> None:
        AI_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")

    @property
    def candidate_limit(self) -> int:
        """Max raw hotspot candidates retained by HitScanner. Clamped to [1, 2000]."""
        try:
            val = int(self.settings.get("candidate_limit", 200))
        except (TypeError, ValueError):
            val = 200
        return max(1, min(2000, val))

    @property
    def sampling_mode(self) -> str:
        """Sampling strategy for synthetic hole placement during auto-training."""
        mode = str(self.settings.get("sampling_mode", "center_bias")).strip().lower()
        valid = ("center_bias", "center", "uniform", "full_uniform", "edge_bias", "edge", "corners", "corner")
        if mode not in valid:
            print(f"[AI] Unknown sampling_mode '{mode}', falling back to center_bias")
            return "center_bias"
        return mode

    # ------------------------------------------------------------------
    # Bootstrap hooks (called from patched HitScanner)
    # ------------------------------------------------------------------

    def observe_scanner(self, scanner, event=None) -> None:
        """Capture scanner state without allowing observations to cross shots.

        This method is called after normal scanner updates and immediately before
        emission. The latter is important: HitScanner can emit from inside its
        update method, before the ordinary post-update hook has run.
        """
        if not self.settings.get("enabled", True):
            return

        debug_frames = getattr(scanner, "debug_frames", {})
        camera_gray = debug_frames.get("camera_gray")
        camera_ts = _safe_float(getattr(scanner, "_last_frame_ts", 0.0), 0.0)
        if camera_gray is not None:
            self._latest_gray = camera_gray
            if camera_ts > 0.0:
                self._latest_frame_ts = camera_ts

        # Register every scanner event we have not seen. Passing event from the
        # emission hook guarantees that the relevant context exists before AI
        # is allowed to choose a candidate.
        scanner_events = list(getattr(scanner, "audio_events", []))
        if event is not None and all(getattr(ev, "shot_id", None) != getattr(event, "shot_id", None) for ev in scanner_events):
            scanner_events.append(event)

        for scanner_event in scanner_events:
            shot_id = int(getattr(scanner_event, "shot_id", 0) or 0)
            if shot_id <= 0:
                continue
            if shot_id not in self._shots:
                self._create_shot_context(scanner, scanner_event)

        # Mirror terminal scanner states, including shots that timed out without
        # ever reaching the emission hook.
        for scanner_event in scanner_events:
            shot_id = int(getattr(scanner_event, "shot_id", 0) or 0)
            ctx = self._shots.get(shot_id)
            scanner_state = str(getattr(scanner_event, "state", "pending") or "pending")
            if ctx is not None and ctx.state == "pending" and scanner_state != "pending":
                self.mark_shot_finished(shot_id, scanner_state)

        current_count = int(getattr(scanner, "audio_event_count", 0) or 0)
        self._last_audio_count = max(self._last_audio_count, current_count)

        pending_contexts = [ctx for ctx in self._shots.values() if ctx.state == "pending"]
        if not pending_contexts:
            self._shot_detected = False
            return

        all_candidates = [dict(c) for c in getattr(scanner, "last_candidates", [])]
        candidate_ts = max(
            (_safe_float(c.get("timestamp", camera_ts), camera_ts) for c in all_candidates),
            default=camera_ts,
        )
        frame_ts = candidate_ts or camera_ts
        if frame_ts <= 0.0:
            return

        association_lead = _safe_float(getattr(scanner, "association_lead_s", 0.08), 0.08)
        association_lag = _safe_float(getattr(scanner, "association_lag_s", 1.5), 1.5)
        eligible = [
            ctx for ctx in pending_contexts
            if ctx.peak_ts - association_lead <= frame_ts <= ctx.peak_ts + association_lag
        ]
        if not eligible:
            return

        # One detector frame must belong to one shot only. During emission the
        # explicit event wins; otherwise use the temporally closest audio peak.
        explicit_shot_id = int(getattr(event, "shot_id", 0) or 0) if event is not None else 0
        target = next((ctx for ctx in eligible if ctx.shot_id == explicit_shot_id), None)
        if target is None:
            target = min(eligible, key=lambda ctx: (abs(frame_ts - ctx.peak_ts), -ctx.peak_ts))

        # A newer detector frame replaces the previous candidate snapshot,
        # including replacing it with an empty list. This is the key stale
        # candidate fix. We may also supplement sparse/raw detector candidates
        # with AI-side hotspot proposals to improve recall on moving content.
        if frame_ts > target.candidate_frame_ts + 1e-6:
            target.candidate_evidence = {}
            source_frame_cache: Dict[float, Optional[np.ndarray]] = {}

            merged_candidates = [dict(candidate) for candidate in all_candidates]
            if self.settings.get("supplement_candidates_enabled", True):
                try:
                    min_candidates = max(0, min(self.candidate_limit, int(self.settings.get("supplement_min_candidates", 120))))
                except Exception:
                    min_candidates = min(self.candidate_limit, 120)
                supplement_budget = max(0, min_candidates - len(merged_candidates))
                if supplement_budget > 0:
                    primary_source_ts = camera_ts if camera_ts > 0.0 else frame_ts
                    primary_cache_key = round(primary_source_ts, 6)
                    if primary_cache_key not in source_frame_cache:
                        source_frame_cache[primary_cache_key] = self._resolve_candidate_source_gray(
                            scanner=scanner,
                            source_ts=primary_source_ts,
                            camera_gray=camera_gray,
                            camera_ts=camera_ts,
                        )
                    supplemental = self._generate_supplemental_candidates(
                        ctx=target,
                        source_gray=source_frame_cache.get(primary_cache_key),
                        source_ts=primary_source_ts,
                        existing_candidates=merged_candidates,
                        limit=supplement_budget,
                    )
                    if supplemental:
                        self.session_stats["supplemental_candidates_generated"] += len(supplemental)
                        merged_candidates = self._merge_candidate_lists(
                            base_candidates=merged_candidates,
                            extra_candidates=supplemental,
                            limit=self.candidate_limit,
                        )
                        self.session_stats["supplemental_candidates_merged"] += max(0, len(merged_candidates) - len(all_candidates))

            tagged: List[Candidate] = []
            for candidate in merged_candidates[: self.candidate_limit]:
                copied = dict(candidate)
                copied["_ai_shot_id"] = target.shot_id
                source_ts = _safe_float(candidate.get("timestamp", frame_ts), frame_ts)
                copied["_ai_source_frame_ts"] = source_ts

                cache_key = round(source_ts, 6)
                if cache_key not in source_frame_cache:
                    source_frame_cache[cache_key] = self._resolve_candidate_source_gray(
                        scanner=scanner,
                        source_ts=source_ts,
                        camera_gray=camera_gray,
                        camera_ts=camera_ts,
                    )

                evidence = self._build_candidate_evidence(
                    ctx=target,
                    candidate=copied,
                    source_ts=source_ts,
                    source_gray=source_frame_cache[cache_key],
                )
                if evidence is not None:
                    copied["_ai_evidence_id"] = evidence.evidence_id
                    target.candidate_evidence[evidence.evidence_id] = evidence
                tagged.append(copied)
            if not tagged and target.candidates:
                self.session_stats["stale_candidate_blocks"] += 1
            target.candidates = tagged
            target.candidate_frame_ts = frame_ts

        self._capture_unique_post_frame(target, camera_gray, camera_ts)
        self._sync_legacy_state(target)

    def _create_shot_context(self, scanner, scanner_event) -> AIShotContext:
        shot_id = int(getattr(scanner_event, "shot_id", 0) or 0)
        peak_ts = _safe_float(getattr(scanner_event, "peak_ts", time.time()), time.time())
        pre_gray, pre_ts = self._resolve_pre_shot(scanner, scanner_event, peak_ts)
        ctx = AIShotContext(
            shot_id=shot_id,
            peak_ts=peak_ts,
            created_at=time.time(),
            pre_shot_gray=pre_gray,
            pre_shot_ts=pre_ts,
        )
        self._shots[shot_id] = ctx
        self._active_shot_id = shot_id
        self._shot_detected = True
        self.session_stats["shots_seen"] += 1
        self.session_stats["shot_contexts_created"] += 1
        self._prune_shot_contexts()
        self._sync_legacy_state(ctx)
        print(f"[AI SHOT] created shot_id={shot_id} peak={peak_ts:.3f} pre={pre_ts:.3f}")
        return ctx

    def _resolve_pre_shot(self, scanner, scanner_event, peak_ts: float) -> Tuple[Optional[np.ndarray], float]:
        # HitScanner stores one authoritative snapshot for the newest event.
        scanner_events = list(getattr(scanner, "audio_events", []))
        newest_id = max((int(getattr(ev, "shot_id", 0) or 0) for ev in scanner_events), default=0)
        event_id = int(getattr(scanner_event, "shot_id", 0) or 0)
        snapshot = getattr(scanner, "pre_shot_snapshot", None)
        snapshot_ts = _safe_float(getattr(scanner, "pre_shot_snapshot_ts", 0.0), 0.0)
        if snapshot is not None and (event_id == newest_id or newest_id == 0):
            return snapshot.copy(), snapshot_ts

        # For an older overlapping event, resolve its own frame from history.
        frame_history = getattr(scanner, "frame_history", None)
        target_ts = peak_ts - 0.25
        best_frame = None
        best_delta = float("inf")
        if frame_history is not None:
            for frame in frame_history:
                if frame.timestamp > target_ts:
                    continue
                delta = abs(frame.timestamp - target_ts)
                if delta < best_delta:
                    best_delta = delta
                    best_frame = frame
        if best_frame is not None:
            return best_frame.gray.copy(), float(best_frame.timestamp)
        if self._latest_gray is not None:
            return self._latest_gray.copy(), self._latest_frame_ts or time.time()
        return None, 0.0

    def _resolve_candidate_source_gray(
        self,
        scanner,
        source_ts: float,
        camera_gray: Optional[np.ndarray],
        camera_ts: float,
    ) -> Optional[np.ndarray]:
        """Resolve the exact detector frame that produced a candidate.

        ``scanner.last_candidates`` belongs to a specific camera timestamp.  A
        later main-loop iteration may already expose another camera image, so
        we first require a timestamp match and otherwise look the frame up in
        HitScanner's grayscale history.
        """
        if camera_gray is not None and camera_ts > 0.0 and abs(camera_ts - source_ts) <= 1e-6:
            return camera_gray

        best_gray: Optional[np.ndarray] = None
        best_delta = float("inf")
        frame_history = getattr(scanner, "frame_history", None)
        if frame_history is not None:
            for frame in frame_history:
                frame_ts = _safe_float(getattr(frame, "timestamp", 0.0), 0.0)
                delta = abs(frame_ts - source_ts)
                if delta < best_delta:
                    best_delta = delta
                    best_gray = getattr(frame, "gray", None)

        # At 30 FPS neighbouring frames are about 33 ms apart. 75 ms gives
        # enough tolerance for timestamp rounding without accepting an
        # unrelated image from a moving video.
        if best_gray is not None and best_delta <= 0.075:
            return best_gray

        if camera_gray is not None and source_ts <= 0.0:
            return camera_gray
        return None

    @staticmethod
    def _crop_patch(
        gray: Optional[np.ndarray],
        x: float,
        y: float,
        radius: int = 8,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        if gray is None:
            return None, None
        ix, iy = int(round(x)), int(round(y))
        h, w = gray.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None, None

        r = max(1, int(radius))
        x0, y0 = max(0, ix - r), max(0, iy - r)
        x1, y1 = min(w, ix + r + 1), min(h, iy + r + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            return None, None
        return patch.copy(), (x0, y0, x1, y1)

    def _build_candidate_evidence(
        self,
        *,
        ctx: AIShotContext,
        candidate: Candidate,
        source_ts: float,
        source_gray: Optional[np.ndarray],
    ) -> Optional[CandidateImageEvidence]:
        if source_gray is None:
            self.session_stats["candidate_patch_misses"] += 1
            return None

        x = _safe_float(candidate.get("camera_x", 0.0))
        y = _safe_float(candidate.get("camera_y", 0.0))
        source_patch, bounds = self._crop_patch(source_gray, x, y, radius=8)
        if source_patch is None or bounds is None:
            self.session_stats["candidate_patch_misses"] += 1
            return None

        pre_patch: Optional[np.ndarray] = None
        diff_patch: Optional[np.ndarray] = None
        if ctx.pre_shot_gray is not None and ctx.pre_shot_gray.shape == source_gray.shape:
            x0, y0, x1, y1 = bounds
            candidate_pre = ctx.pre_shot_gray[y0:y1, x0:x1]
            if candidate_pre.shape == source_patch.shape and candidate_pre.size > 0:
                pre_patch = candidate_pre.copy()
                diff_patch = np.abs(
                    source_patch.astype(np.int16) - pre_patch.astype(np.int16)
                ).astype(np.uint8)

        evidence_id = self._next_evidence_id
        self._next_evidence_id += 1
        evidence = CandidateImageEvidence(
            evidence_id=evidence_id,
            shot_id=ctx.shot_id,
            source_frame_ts=float(source_ts),
            source_shape=(int(source_gray.shape[0]), int(source_gray.shape[1])),
            patch_bounds=bounds,
            source_patch=source_patch,
            pre_patch=pre_patch,
            diff_patch=diff_patch,
        )
        self.session_stats["candidate_source_patches"] += 1
        if pre_patch is not None:
            self.session_stats["candidate_pre_patches"] += 1
        if diff_patch is not None:
            self.session_stats["candidate_diff_patches"] += 1
        return evidence

    def _generate_supplemental_candidates(
        self,
        *,
        ctx: AIShotContext,
        source_gray: Optional[np.ndarray],
        source_ts: float,
        existing_candidates: Sequence[Candidate],
        limit: int,
    ) -> List[Candidate]:
        """Create extra hotspot candidates when the detector is sparse.

        This is intentionally conservative on static scenes because the native
        detector already performs well there. On moving content (video / game
        backgrounds) the detector often returns very few hotspots; this helper
        supplements them with small, multiscale blob-like peaks extracted from
        the candidate source frame itself.
        """
        if cv2 is None or source_gray is None or limit <= 0:
            return []
        if source_gray.size == 0:
            return []

        gray = source_gray
        if len(gray.shape) > 2:
            try:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            except Exception:
                return []
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        self.session_stats["supplemental_frames_processed"] += 1

        h, w = gray.shape[:2]
        median5 = cv2.medianBlur(gray, 5)
        median9 = cv2.medianBlur(gray, 9)

        # Appearance cues: small dark/bright anomalies that stand out from the
        # local neighbourhood regardless of the projected image content.
        dark_local = cv2.subtract(median5, gray).astype(np.float32)
        bright_local = cv2.subtract(gray, median5).astype(np.float32)
        dark_multi = dark_local.copy()
        bright_multi = bright_local.copy()
        for k in (3, 5, 7, 9, 13):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            dark_multi = np.maximum(dark_multi, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32))
            bright_multi = np.maximum(bright_multi, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32))
        dark_multi = np.maximum(dark_multi, cv2.subtract(median9, gray).astype(np.float32))
        bright_multi = np.maximum(bright_multi, cv2.subtract(gray, median9).astype(np.float32))

        responses: List[Tuple[str, np.ndarray, int, str]] = [
            ("supp_dark", dark_multi, 5, "dark"),
            ("supp_bright", bright_multi, 5, "bright"),
        ]

        # Temporal cue: retain local, hole-sized change but suppress broad
        # scene motion by subtracting a wider blur of the diff image.
        if ctx.pre_shot_gray is not None and ctx.pre_shot_gray.shape[:2] == gray.shape[:2]:
            absdiff = cv2.absdiff(gray, ctx.pre_shot_gray).astype(np.float32)
            small = cv2.GaussianBlur(absdiff, (0, 0), 1.0)
            large = cv2.GaussianBlur(absdiff, (0, 0), 4.0)
            temporal_local = np.clip(small - large, 0.0, 255.0)
            responses.append(("supp_temporal", temporal_local, 6, "change"))

        # Gather peaks from all response maps and merge them.
        candidates: List[Candidate] = []
        existing_points = [
            (
                _safe_float(c.get("camera_x", -1.0)),
                _safe_float(c.get("camera_y", -1.0)),
            )
            for c in existing_candidates
        ]

        per_map_limit = max(12, int(math.ceil(max(1, limit) / max(1, len(responses))) * 3))
        for source_name, response, radius_hint, polarity in responses:
            peaks = self._extract_response_peaks(
                response=response,
                source_name=source_name,
                polarity=polarity,
                gray=gray,
                pre_gray=ctx.pre_shot_gray if ctx.pre_shot_gray is not None and ctx.pre_shot_gray.shape[:2] == gray.shape[:2] else None,
                source_ts=source_ts,
                existing_points=existing_points,
                radius_hint=radius_hint,
                limit=per_map_limit,
            )
            if not peaks:
                continue
            if source_name == "supp_temporal":
                self.session_stats["supplemental_temporal_candidates"] += len(peaks)
            else:
                self.session_stats["supplemental_appearance_candidates"] += len(peaks)
            candidates.extend(peaks)

        merged = self._merge_candidate_lists(base_candidates=[], extra_candidates=candidates, limit=limit)
        return merged

    def _extract_response_peaks(
        self,
        *,
        response: np.ndarray,
        source_name: str,
        polarity: str,
        gray: np.ndarray,
        pre_gray: Optional[np.ndarray],
        source_ts: float,
        existing_points: Sequence[Tuple[float, float]],
        radius_hint: int,
        limit: int,
    ) -> List[Candidate]:
        if cv2 is None or response.size == 0 or limit <= 0:
            return []
        response32 = np.clip(response.astype(np.float32), 0.0, None)
        positive = response32[response32 > 0.5]
        if positive.size < 8:
            return []

        try:
            base_percentile = float(self.settings.get("supplement_peak_percentile", 99.6))
        except Exception:
            base_percentile = 99.6
        percentiles = [base_percentile, 99.3, 99.0, 98.7, 98.4]
        local_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, radius_hint * 2 + 1), max(5, radius_hint * 2 + 1)))
        local_max = cv2.dilate(response32, local_kernel)

        coords: List[Tuple[float, int, int]] = []
        for perc in percentiles:
            threshold = float(np.percentile(positive, min(99.99, max(50.0, perc))))
            if threshold <= 0.0:
                continue
            mask = (response32 >= threshold) & (response32 >= local_max - 1e-6)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            ranked = sorted(((float(response32[y, x]), int(x), int(y)) for x, y in zip(xs, ys)), reverse=True)
            coords = ranked
            if len(coords) >= limit:
                break
        if not coords:
            return []

        selected: List[Candidate] = []
        used_points = list(existing_points)
        min_sep = max(6.0, float(radius_hint) * 1.5)
        response_max = max(1.0, float(np.max(response32)))

        for strength, x, y in coords:
            if len(selected) >= limit:
                break
            if any(math.hypot(px - x, py - y) < min_sep for px, py in used_points):
                continue
            candidate = self._build_supplemental_candidate(
                gray=gray,
                pre_gray=pre_gray,
                x=float(x),
                y=float(y),
                source_ts=source_ts,
                source_name=source_name,
                polarity=polarity,
                radius_hint=radius_hint,
                response_strength=float(strength),
                response_max=response_max,
            )
            if candidate is None:
                continue
            selected.append(candidate)
            used_points.append((float(x), float(y)))
        return selected

    def _build_supplemental_candidate(
        self,
        *,
        gray: np.ndarray,
        pre_gray: Optional[np.ndarray],
        x: float,
        y: float,
        source_ts: float,
        source_name: str,
        polarity: str,
        radius_hint: int,
        response_strength: float,
        response_max: float,
    ) -> Optional[Candidate]:
        patch, bounds = self._crop_patch(gray, x, y, radius=max(4, radius_hint + 2))
        if patch is None or bounds is None:
            return None
        patch_mean = float(np.mean(patch))
        patch_std = float(np.std(patch))
        center_value = float(gray[int(round(y)), int(round(x))])

        change_value = 0.0
        pre_shot_change = 0.0
        if pre_gray is not None and pre_gray.shape[:2] == gray.shape[:2]:
            x0, y0, x1, y1 = bounds
            pre_patch = pre_gray[y0:y1, x0:x1]
            if pre_patch.shape == patch.shape and pre_patch.size > 0:
                diff_patch = np.abs(patch.astype(np.int16) - pre_patch.astype(np.int16)).astype(np.uint8)
                change_value = float(np.mean(diff_patch))
                pre_shot_change = float(np.max(diff_patch))

        if polarity == "bright":
            center_change = max(0.0, center_value - patch_mean) / 255.0
        elif polarity == "dark":
            center_change = max(0.0, patch_mean - center_value) / 255.0
        else:
            center_change = min(1.0, change_value / 32.0)

        local_contrast_gain = patch_std / 48.0
        strength_norm = min(1.0, max(0.0, response_strength / max(1.0, response_max)))
        temporal_bonus = min(1.0, change_value / 24.0) * 1.5 if polarity == "change" else min(1.0, change_value / 32.0)
        score = 4.0 + 7.0 * strength_norm + 1.8 * min(1.5, local_contrast_gain) + 2.4 * min(1.5, temporal_bonus)

        area = float(math.pi * (max(2.0, float(radius_hint)) ** 2))
        circularity = 0.75 if polarity != "change" else 0.65
        candidate: Candidate = {
            "camera_x": float(x),
            "camera_y": float(y),
            "timestamp": float(source_ts),
            "score": float(score),
            "area": area,
            "radius": float(max(2, radius_hint)),
            "circularity": float(circularity),
            "center_darkening": float(center_change),
            "local_contrast_gain": float(local_contrast_gain),
            "pre_shot_change": float(pre_shot_change),
            "change_value": float(change_value),
            "source": source_name,
            "_ai_supplemental": True,
        }
        return candidate

    @staticmethod
    def _merge_candidate_lists(
        *,
        base_candidates: Sequence[Candidate],
        extra_candidates: Sequence[Candidate],
        limit: int,
    ) -> List[Candidate]:
        merged: List[Candidate] = [dict(c) for c in base_candidates]
        for extra in extra_candidates:
            ex_x = _safe_float(extra.get("camera_x", 0.0))
            ex_y = _safe_float(extra.get("camera_y", 0.0))
            ex_score = _safe_float(extra.get("score", 0.0))
            replace_index: Optional[int] = None
            duplicate_index: Optional[int] = None
            for idx, current in enumerate(merged):
                cur_x = _safe_float(current.get("camera_x", 0.0))
                cur_y = _safe_float(current.get("camera_y", 0.0))
                if math.hypot(cur_x - ex_x, cur_y - ex_y) <= 8.0:
                    duplicate_index = idx
                    cur_score = _safe_float(current.get("score", 0.0))
                    if ex_score > cur_score:
                        replace_index = idx
                    break
            if duplicate_index is not None:
                if replace_index is not None:
                    merged[replace_index] = dict(extra)
                continue
            merged.append(dict(extra))

        merged.sort(key=lambda c: _safe_float(c.get("score", 0.0)), reverse=True)
        return merged[: max(1, int(limit))]

    def _candidate_evidence(self, candidate: Candidate) -> Optional[CandidateImageEvidence]:
        shot_id = int(candidate.get("_ai_shot_id", 0) or 0)
        evidence_id = int(candidate.get("_ai_evidence_id", 0) or 0)
        if shot_id <= 0 or evidence_id <= 0:
            return None
        ctx = self._shots.get(shot_id)
        if ctx is None:
            return None
        evidence = ctx.candidate_evidence.get(evidence_id)
        if evidence is None or evidence.shot_id != shot_id:
            return None
        return evidence

    def _capture_unique_post_frame(
        self,
        ctx: AIShotContext,
        camera_gray: Optional[np.ndarray],
        camera_ts: float,
    ) -> None:
        if camera_gray is None or camera_ts <= 0.0:
            return
        if camera_ts <= ctx.last_post_frame_ts + 1e-6:
            self.session_stats["duplicate_post_frames_skipped"] += 1
            return
        if camera_ts < ctx.peak_ts - 0.02:
            return

        ctx.post_shot_gray = camera_gray.copy()
        # Use camera timestamps, not main-loop time. The same camera frame can
        # therefore never be counted twice as persistence evidence.
        if len(ctx.post_shot_frames) < 8:
            if not ctx.post_shot_frames or camera_ts - ctx.post_shot_frames[-1][1] >= 0.025:
                ctx.post_shot_frames.append((camera_gray.copy(), camera_ts))
        ctx.last_post_frame_ts = camera_ts

    def _select_context(self, shot_id: Optional[int]) -> Optional[AIShotContext]:
        if shot_id is not None and shot_id in self._shots:
            return self._shots[shot_id]
        if self._active_shot_id is not None and self._active_shot_id in self._shots:
            return self._shots[self._active_shot_id]
        return None

    def _sync_legacy_state(self, ctx: AIShotContext) -> None:
        self._active_shot_id = ctx.shot_id
        self._latest_candidates = list(ctx.candidates)
        self._pre_shot_gray = ctx.pre_shot_gray
        self._pre_shot_ts = ctx.pre_shot_ts
        self._post_shot_gray = ctx.post_shot_gray
        self._post_shot_frames = list(ctx.post_shot_frames)
        self._shot_ts = ctx.peak_ts

    def _prune_shot_contexts(self) -> None:
        """Keep pending shots plus at most the newest completed observation.

        A 4K grayscale frame is several megabytes, and each context may retain
        multiple post-shot frames. Keeping many completed contexts would turn a
        state-isolation fix into a memory leak.
        """
        protected = {self._active_shot_id, self._last_completed_shot_id}
        completed = sorted(
            (ctx for ctx in self._shots.values() if ctx.state != "pending"),
            key=lambda ctx: (ctx.created_at, ctx.shot_id),
            reverse=True,
        )
        keep_completed = {ctx.shot_id for ctx in completed[:1]}
        for ctx in list(self._shots.values()):
            if ctx.state == "pending":
                continue
            if ctx.shot_id in protected or ctx.shot_id in keep_completed:
                continue
            self._shots.pop(ctx.shot_id, None)

        # Hard guard for pathological bursts: retain newest contexts, but never
        # discard the active one.
        if len(self._shots) > 6:
            ordered = sorted(self._shots.values(), key=lambda ctx: (ctx.created_at, ctx.shot_id))
            for ctx in ordered:
                if len(self._shots) <= 6:
                    break
                if ctx.shot_id == self._active_shot_id:
                    continue
                self._shots.pop(ctx.shot_id, None)

    def mark_shot_finished(self, shot_id: int, state: str = "finished") -> None:
        ctx = self._shots.get(int(shot_id))
        if ctx is None:
            return
        new_state = str(state or "finished")
        if ctx.state == new_state and ctx.state != "pending":
            return
        ctx.state = new_state
        self._last_completed_shot_id = ctx.shot_id

        pending = [item for item in self._shots.values() if item.state == "pending"]
        if pending:
            newest_pending = max(pending, key=lambda item: (item.peak_ts, item.shot_id))
            self._sync_legacy_state(newest_pending)
            self._shot_detected = True
        else:
            self._sync_legacy_state(ctx)
            self._shot_detected = False

        print(
            f"[AI SHOT] finished shot_id={ctx.shot_id} state={ctx.state} "
            f"candidates={len(ctx.candidates)} evidence={len(ctx.candidate_evidence)} "
            f"post_frames={len(ctx.post_shot_frames)}"
        )
        self._prune_shot_contexts()

    def choose_for_emission(
        self,
        default_x: float,
        default_y: float,
        shot_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Called from bootstrap when hit_scanner wants to emit a hit.
        Returns whether AI wants to override the position."""
        result: Dict[str, Any] = {
            "apply": False,
            "camera_x": float(default_x),
            "camera_y": float(default_y),
            "confidence": 0.0,
            "reason": "passthrough",
            "shot_id": shot_id,
        }

        ctx = self._select_context(shot_id)
        if ctx is None or (shot_id is not None and ctx.shot_id != int(shot_id)):
            self.session_stats["stale_candidate_blocks"] += 1
            result["reason"] = "missing_shot_context"
            return result
        self._sync_legacy_state(ctx)
        self.session_stats["emission_syncs"] += 1

        mode = str(self.settings.get("mode", "train_only"))
        if mode in {"off", "train_only", "advisory"}:
            return result

        shot_candidates = [
            candidate for candidate in ctx.candidates
            if int(candidate.get("_ai_shot_id", ctx.shot_id)) == ctx.shot_id
        ]
        if not shot_candidates:
            result["reason"] = "no_candidates_for_shot"
            return result

        # Rank candidates and pick the best. Only candidates explicitly tagged
        # with this event's shot_id are eligible.
        ranked = self.rank_candidates(shot_candidates)
        if not ranked:
            return result

        best = ranked[0]
        confidence = _safe_float(best.get("ai_score", 0.5))
        min_conf = float(self.settings.get("min_confidence", 0.58))
        override_conf = float(self.settings.get("override_confidence", 0.92))

        if confidence < min_conf:
            result["reason"] = "below_min_confidence"
            return result

        if mode == "ai_only":
            blend = 1.0
        elif mode == "ai_priority" and confidence >= override_conf:
            blend = 1.0
        elif mode == "blended":
            blend = float(self.settings.get("trust_percent", 0)) / 100.0
        else:
            blend = 0.0

        if blend <= 0.0:
            return result

        bx = _safe_float(best.get("camera_x", default_x))
        by = _safe_float(best.get("camera_y", default_y))
        final_x = (1.0 - blend) * default_x + blend * bx
        final_y = (1.0 - blend) * default_y + blend * by

        result.update({
            "apply": True,
            "camera_x": float(final_x),
            "camera_y": float(final_y),
            "confidence": float(confidence),
            "blend": float(blend),
            "reason": f"ai_{mode}",
        })
        return result

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, candidate: Candidate) -> Dict[str, float]:
        """Build a consistent feature dict from a hit_scanner candidate."""
        x = _safe_float(candidate.get("camera_x", 0.0))
        y = _safe_float(candidate.get("camera_y", 0.0))

        features: Dict[str, float] = {
            "detector_score": _safe_float(candidate.get("score", 0.0)),
            "area": _safe_float(candidate.get("area", 0.0)),
            "radius": _safe_float(candidate.get("radius", 0.0)),
            "circularity": _safe_float(candidate.get("circularity", 0.0)),
            "center_change": _safe_float(candidate.get("center_darkening", 0.0)),
            "local_contrast": _safe_float(candidate.get("local_contrast_gain", 0.0)),
            "pre_shot_change": _safe_float(candidate.get("pre_shot_change", 0.0)),
            "change_value": _safe_float(candidate.get("change_value", 0.0)),
        }

        # Patch stats must come from the detector frame that created this
        # candidate. Falling back to the latest frame is only for legacy or
        # synthetic candidates that have no shot-bound image evidence.
        evidence = self._candidate_evidence(candidate)
        patch_mean, patch_std, edge_strength = self._patch_stats(candidate, x, y)
        features["patch_mean"] = patch_mean
        features["patch_std"] = patch_std
        features["edge_strength"] = edge_strength

        # Normalized position (helps AI learn edge-of-frame bias)
        if evidence is not None:
            h, w = evidence.source_shape
            features["x_norm"] = x / max(1.0, float(w))
            features["y_norm"] = y / max(1.0, float(h))
        elif self._latest_gray is not None:
            h, w = self._latest_gray.shape[:2]
            features["x_norm"] = x / max(1.0, float(w))
            features["y_norm"] = y / max(1.0, float(h))
        else:
            features["x_norm"] = 0.0
            features["y_norm"] = 0.0

        return features

    def _patch_stats(self, candidate: Candidate, x: float, y: float) -> Tuple[float, float, float]:
        """Extract patch statistics from the candidate's own source frame."""
        evidence = self._candidate_evidence(candidate)
        if evidence is not None and evidence.source_patch is not None:
            return self._patch_stats_from_patch(evidence.source_patch)

        # A detector candidate that belongs to a concrete shot must never be
        # evaluated against a later global frame. Missing evidence is safer as
        # neutral/zero image features than silently using the wrong picture.
        if int(candidate.get("_ai_shot_id", 0) or 0) > 0:
            return 0.0, 0.0, 0.0

        if self._latest_gray is None:
            return 0.0, 0.0, 0.0

        # Backwards compatibility for training/synthetic candidates that were
        # not produced by the shot-isolated detector path.
        self.session_stats["candidate_patch_fallbacks"] = (
            int(self.session_stats.get("candidate_patch_fallbacks", 0) or 0) + 1
        )
        patch, _ = self._crop_patch(self._latest_gray, x, y, radius=8)
        if patch is None:
            return 0.0, 0.0, 0.0
        return self._patch_stats_from_patch(patch)

    @staticmethod
    def _patch_stats_from_patch(patch: np.ndarray) -> Tuple[float, float, float]:
        if patch.size == 0:
            return 0.0, 0.0, 0.0

        patch_mean = float(np.mean(patch)) / 255.0
        patch_std = float(np.std(patch)) / 64.0

        edge_strength = 0.0
        if cv2 is not None and patch.shape[0] >= 3 and patch.shape[1] >= 3:
            lap = cv2.Laplacian(patch, cv2.CV_32F)
            edge_strength = float(np.mean(np.abs(lap))) / 32.0

        return patch_mean, patch_std, edge_strength

    # ------------------------------------------------------------------
    # Hotspot persistence & noise rejection (Stage 1)
    # ------------------------------------------------------------------

    def compute_persistence(self, candidate: Candidate) -> float:
        """
        Check if a hotspot is persistent across multiple post-shot frames.
        Returns 0.0-1.0 where 1.0 = visible in all post-frames.
        A real hole persists; flicker/shadows don't.
        """
        if not self._post_shot_frames:
            return 0.5  # No data — neutral

        evidence = self._candidate_evidence(candidate)
        if evidence is not None and evidence.pre_patch is not None:
            pre_patch = evidence.pre_patch
            x0, y0, x1, y1 = evidence.patch_bounds
            source_shape = evidence.source_shape
        else:
            if self._pre_shot_gray is None:
                return 0.5
            x = _safe_float(candidate.get("camera_x", 0.0))
            y = _safe_float(candidate.get("camera_y", 0.0))
            pre_patch, bounds = self._crop_patch(self._pre_shot_gray, x, y, radius=6)
            if pre_patch is None or bounds is None:
                return 0.0
            x0, y0, x1, y1 = bounds
            source_shape = (int(self._pre_shot_gray.shape[0]), int(self._pre_shot_gray.shape[1]))

        pre_mean = float(np.mean(pre_patch))

        visible_count = 0
        compatible_count = 0
        for post_gray, _ in self._post_shot_frames:
            if post_gray.shape[:2] != source_shape:
                continue
            post_patch = post_gray[y0:y1, x0:x1]
            if post_patch.shape != pre_patch.shape or post_patch.size == 0:
                continue
            compatible_count += 1
            delta = abs(float(np.mean(post_patch)) - pre_mean)
            if delta > 3.0:  # Threshold: visible change
                visible_count += 1

        if compatible_count == 0:
            return 0.5
        return float(visible_count) / float(compatible_count)

    def existed_before_shot(self, candidate: Candidate) -> float:
        """
        Check if the hotspot evidence existed BEFORE the shot.
        Returns 0.0-1.0 where 1.0 = definitely existed before (old hole/artifact).
        """
        evidence = self._candidate_evidence(candidate)
        if (
            evidence is not None
            and evidence.pre_patch is not None
            and evidence.source_patch is not None
            and evidence.pre_patch.shape == evidence.source_patch.shape
        ):
            pre_patch = evidence.pre_patch
            post_patch = evidence.source_patch
        else:
            if self._pre_shot_gray is None or self._post_shot_gray is None:
                return 0.0

            x = _safe_float(candidate.get("camera_x", 0.0))
            y = _safe_float(candidate.get("camera_y", 0.0))
            if self._pre_shot_gray.shape != self._post_shot_gray.shape:
                return 0.0
            pre_patch, bounds = self._crop_patch(self._pre_shot_gray, x, y, radius=6)
            if pre_patch is None or bounds is None:
                return 0.0
            x0, y0, x1, y1 = bounds
            post_patch = self._post_shot_gray[y0:y1, x0:x1]
            if post_patch.shape != pre_patch.shape or post_patch.size == 0:
                return 0.0

        pre_std = float(np.std(pre_patch))
        post_std = float(np.std(post_patch))

        # High-contrast backgrounds (checker, grid) naturally have high std.
        # Only flag as "existed before" if the pre-shot patch has moderate
        # contrast AND the post-shot didn't change much.
        # On checker patterns pre_std can be 20-40 — don't penalize those.
        if pre_std > 15.0:
            # High-contrast region — need a much stronger signal to reject.
            # Only reject if pre and post are nearly identical (no new hole).
            delta_mean = float(np.mean(np.abs(post_patch.astype(np.int16) - pre_patch.astype(np.int16))))
            if delta_mean < 2.0:
                return min(1.0, 0.3 + pre_std / 80.0)
            return 0.0

        # Low-contrast region: original logic
        if pre_std > 5.0 and abs(pre_std - post_std) < pre_std * 0.5:
            return min(1.0, pre_std / 20.0)
        return 0.0

    def reject_noise_hotspots(
        self, hotspots: Sequence[Candidate]
    ) -> Tuple[List[Candidate], Dict[str, int]]:
        """
        Stage 1: Conservative noise rejection.
        Removes obvious non-holes while preserving the true hit.
        Returns (surviving_hotspots, rejection_counts).
        """
        surviving: List[Candidate] = []
        rejection_counts: Dict[str, int] = {
            "outside_viewport": 0,
            "existed_before": 0,
            "no_persistence": 0,
            "too_large_change": 0,
        }

        # Get viewport bounds for out-of-bounds rejection
        try:
            from src.engine.settings import load_viewport_rect
            vp = load_viewport_rect()
            from src.engine.input.hit_input import hit_input
        except Exception:
            vp = None

        for hs in hotspots:
            # Check if candidate is within viewport (screen space)
            if vp is not None:
                try:
                    cx = _safe_float(hs.get("camera_x", 0.0))
                    cy = _safe_float(hs.get("camera_y", 0.0))
                    sx, sy = hit_input._canonical_camera_to_screen(cx, cy)
                    if not vp.collidepoint(int(round(sx)), int(round(sy))):
                        rejection_counts["outside_viewport"] += 1
                        continue
                except Exception:
                    pass
            # Check if it existed before the shot (conservative — only reject high confidence)
            existed = self.existed_before_shot(hs)
            if existed > 0.8:
                rejection_counts["existed_before"] += 1
                continue

            # Check persistence (only if we have multi-frame data)
            if len(self._post_shot_frames) >= 2:
                persistence = self.compute_persistence(hs)
                if persistence < 0.1:  # Very low: visible in <10% of frames
                    rejection_counts["no_persistence"] += 1
                    continue

            # Check if the change region is too large (not a bullet hole)
            area = _safe_float(hs.get("area", 0.0))
            if area > 1200:  # Very large change — probably not a hole
                rejection_counts["too_large_change"] += 1
                continue

            # Enrich with persistence data
            enriched = dict(hs)
            enriched["persistence"] = self.compute_persistence(hs) if self._post_shot_frames else 0.5
            enriched["existed_before"] = existed
            surviving.append(enriched)

        return surviving, rejection_counts

    # ------------------------------------------------------------------
    # Candidate ranking (Stage 2)
    # ------------------------------------------------------------------

    def rank_candidates(
        self, candidates: Sequence[Candidate], limit: Optional[int] = None
    ) -> List[Candidate]:
        """Rank candidates using fused detector + AI score."""
        if not candidates:
            return []

        top_k = int(limit or self.settings.get("top_k", 10))
        total_memories = len(self.memory.positives) + len(self.memory.negatives)

        ranked: List[Candidate] = []
        for cand in candidates:
            features = self.extract_features(cand)
            ai_score = self.memory.score(features)
            detector_score = _safe_float(cand.get("score", 0.0))

            # Adaptive weighting: trust detector more when AI has little data,
            # shift toward AI as it accumulates memories.
            if total_memories < 20:
                ai_weight = 0.1
            elif total_memories < 100:
                ai_weight = 0.3
            elif total_memories < 300:
                ai_weight = 0.5
            else:
                ai_weight = 0.65

            det_weight = 1.0 - ai_weight
            # Normalize detector score to 0-1 range (typical range 3-20)
            det_norm = min(1.0, max(0.0, detector_score / 15.0))

            # Persistence bonus: persistent hotspots get a boost
            persistence = _safe_float(cand.get("persistence", 0.5))
            persistence_bonus = persistence * 0.15  # Up to 0.15 bonus

            # Existed-before penalty
            existed = _safe_float(cand.get("existed_before", 0.0))
            existed_penalty = existed * 0.2  # Up to 0.2 penalty

            combined = det_weight * det_norm + ai_weight * ai_score + persistence_bonus - existed_penalty

            enriched = dict(cand)
            enriched["features"] = features
            enriched["ai_score"] = float(ai_score)
            enriched["combined_score"] = float(combined)
            enriched["ai_weight"] = float(ai_weight)
            ranked.append(enriched)

        ranked.sort(key=lambda c: _safe_float(c.get("combined_score", 0.0)), reverse=True)
        for i, c in enumerate(ranked):
            c["rank"] = i + 1
        return ranked[:top_k]

    def rank_with_funnel(
        self,
        raw_hotspots: Sequence[Candidate],
        gt_xy: Optional[Point] = None,
        limit: Optional[int] = None,
        match_radius_px: Optional[float] = None,
    ) -> Tuple[List[Candidate], Optional["ShotDiagnostics"]]:
        """
        Full hotspot pipeline: reject noise → rank → select.
        Optionally tracks diagnostics against ground truth.
        Returns (ranked_hotspots, diagnostics_or_None).
        """
        from src.engine.ai.diagnostics import ShotDiagnostics

        diag = None
        radius = float(match_radius_px or self.settings.get("gt_match_radius_px", 10.0))

        if gt_xy is not None:
            diag = ShotDiagnostics(gt_xy[0], gt_xy[1], radius)
            diag.evaluate_raw_hotspots(raw_hotspots)

        # Stage 1: Noise rejection
        surviving, rejection_counts = self.reject_noise_hotspots(raw_hotspots)

        if diag is not None:
            diag.evaluate_filtered(surviving, rejection_counts)

        # Stage 2: AI ranking
        ranked = self.rank_candidates(surviving, limit=limit)

        if diag is not None:
            diag.evaluate_ai_ranked(ranked, top_k=min(10, len(ranked)))
            if ranked:
                diag.evaluate_selected(ranked[0])
            self.funnel.add(diag)

        return ranked, diag

    # ------------------------------------------------------------------
    # Training (click-based learning)
    # ------------------------------------------------------------------

    def learn_from_click(
        self,
        click_camera_xy: Point,
        shown_candidates: Sequence[Candidate],
        gray_pre: Optional[np.ndarray] = None,
        gray_post: Optional[np.ndarray] = None,
        freeze: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the AI from a user click (or just evaluate if freeze=True).

        freeze=True: measure only, no model updates (benchmark/eval mode).
        freeze=False: normal training — updates positives/negatives.
        """
        self.session_stats["clicks"] += 1
        self.session_stats["last_click_camera"] = [float(click_camera_xy[0]), float(click_camera_xy[1])]

        click_radius = float(self.settings.get("click_match_radius_px", 70.0))
        max_neg = int(self.settings.get("max_negatives_per_click", 3))

        # Find nearest candidate to click
        nearest_idx: Optional[int] = None
        nearest_dist = float("inf")
        for i, cand in enumerate(shown_candidates):
            cx = _safe_float(cand.get("camera_x", 0.0))
            cy = _safe_float(cand.get("camera_y", 0.0))
            dist = math.hypot(click_camera_xy[0] - cx, click_camera_xy[1] - cy)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = i

        positive_added = False
        neg_added = 0

        if not freeze:
            # --- TRAINING MODE: update model ---
            self.memory.stats["total_clicks"] = self.memory.stats.get("total_clicks", 0) + 1
            self.memory.stats["local_updates_since_import"] = self.memory.stats.get("local_updates_since_import", 0) + 1

            # If a candidate is close enough, use it as positive
            if nearest_idx is not None and nearest_dist <= click_radius:
                cand = shown_candidates[nearest_idx]
                features = self._ensure_features(cand)
                self.memory.add_positive(features, {"kind": "candidate_match", "distance": nearest_dist})
                positive_added = True

            # Add limited negatives — never mark near-GT candidates as negative
            safe_radius = click_radius * 2.0
            neg_candidates = []
            for i, c in enumerate(shown_candidates):
                if positive_added and i == nearest_idx:
                    continue
                cx = _safe_float(c.get("camera_x", 0.0))
                cy = _safe_float(c.get("camera_y", 0.0))
                dist = math.hypot(click_camera_xy[0] - cx, click_camera_xy[1] - cy)
                if dist <= safe_radius:
                    continue
                neg_candidates.append((i, c))
            neg_candidates.sort(key=lambda ic: _safe_float(ic[1].get("combined_score", ic[1].get("score", 0.0))))
            for i, cand in neg_candidates:
                if neg_added >= max_neg:
                    break
                features = self._ensure_features(cand)
                self.memory.add_negative(features, {"kind": "shown_other"})
                neg_added += 1

            # If no candidate matched, create synthetic positive
            if not positive_added:
                features = self._synthetic_features(click_camera_xy, gray_pre, gray_post)
                self.memory.add_positive(features, {"kind": "synthetic_click"})
                positive_added = True

            self.memory.save()

        self._shot_detected = False
        self._post_shot_frames = []
        ctx = self._select_context(self._active_shot_id)
        if ctx is not None:
            ctx.post_shot_frames = []

        return {
            "positive_added": positive_added,
            "negatives_added": neg_added,
            "nearest_index": nearest_idx,
            "nearest_distance": float(nearest_dist),
            "total_positives": len(self.memory.positives),
            "total_negatives": len(self.memory.negatives),
        }

    def study_click_area(
        self,
        click_camera_xy: Point,
        gray_pre: Optional[np.ndarray] = None,
        gray_post: Optional[np.ndarray] = None,
    ) -> None:
        """
        Study the visual area around a confirmed click point.

        Stores TWO kinds of positive examples:
        1. Diff-based: pre vs post at click point — "what changed" (needs pre-shot)
        2. Absolute: just the post-shot patch — "what a hole looks like" (works
           even when background changes completely between shots)

        This way the AI learns both:
        - temporal change signatures (useful with static backgrounds)
        - absolute hole appearance (useful with video/animation backgrounds)
        """
        if gray_post is None:
            return

        x, y = click_camera_xy
        ix, iy = int(round(x)), int(round(y))
        h, w = gray_post.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return

        r = 12
        x0, y0 = max(0, ix - r), max(0, iy - r)
        x1, y1 = min(w, ix + r + 1), min(h, iy + r + 1)

        post_patch = gray_post[y0:y1, x0:x1]
        if post_patch.size == 0:
            return

        # --- Example 1: Absolute hole appearance ---
        # "What does a hole look like?" — independent of background
        abs_features: Dict[str, float] = {k: 0.0 for k in FEATURE_KEYS}
        abs_features["patch_mean"] = float(np.mean(post_patch)) / 255.0
        abs_features["patch_std"] = float(np.std(post_patch)) / 64.0
        abs_features["x_norm"] = x / max(1.0, float(w))
        abs_features["y_norm"] = y / max(1.0, float(h))

        if cv2 is not None and post_patch.shape[0] >= 3 and post_patch.shape[1] >= 3:
            lap = cv2.Laplacian(post_patch, cv2.CV_32F)
            abs_features["edge_strength"] = float(np.mean(np.abs(lap))) / 32.0

        # Center vs ring in the post-shot — holes have distinct center
        if post_patch.shape[0] >= 2 * r and post_patch.shape[1] >= 2 * r:
            center_val = float(np.mean(post_patch[r - 2:r + 3, r - 2:r + 3]))
            ring_vals = [
                float(np.mean(post_patch[:3, :])),
                float(np.mean(post_patch[-3:, :])),
                float(np.mean(post_patch[:, :3])),
                float(np.mean(post_patch[:, -3:])),
            ]
            ring_val = sum(ring_vals) / len(ring_vals)
            abs_features["center_change"] = center_val / 255.0
            abs_features["local_contrast"] = abs(center_val - ring_val) / 255.0

        if abs_features.get("patch_std", 0.0) > 0.003:
            self.memory.add_positive(abs_features, {"kind": "hole_appearance"})

        # --- Example 2: Diff-based change signature ---
        # "What changed at this spot?" — needs pre-shot
        if gray_pre is not None and gray_pre.shape == gray_post.shape:
            pre_patch = gray_pre[y0:y1, x0:x1]
            if pre_patch.size > 0 and pre_patch.shape == post_patch.shape:
                diff_features: Dict[str, float] = {k: 0.0 for k in FEATURE_KEYS}
                delta = np.abs(post_patch.astype(np.int16) - pre_patch.astype(np.int16))
                delta_mean = float(np.mean(delta))

                diff_features["pre_shot_change"] = delta_mean / 64.0
                diff_features["change_value"] = delta_mean / 64.0
                diff_features["patch_mean"] = abs_features["patch_mean"]
                diff_features["patch_std"] = abs_features["patch_std"]
                diff_features["edge_strength"] = abs_features.get("edge_strength", 0.0)
                diff_features["x_norm"] = abs_features["x_norm"]
                diff_features["y_norm"] = abs_features["y_norm"]
                diff_features["detector_score"] = float(np.mean(pre_patch)) / 255.0

                # Center vs ring in delta
                if delta.shape[0] >= 2 * r and delta.shape[1] >= 2 * r:
                    center_delta = float(np.mean(delta[r - 2:r + 3, r - 2:r + 3]))
                    ring_delta = (float(np.mean(delta[:3, :])) + float(np.mean(delta[-3:, :]))) / 2.0
                    diff_features["center_change"] = center_delta / 64.0
                    diff_features["local_contrast"] = abs(center_delta - ring_delta) / 64.0

                if delta_mean > 0.5:
                    self.memory.add_positive(diff_features, {"kind": "click_area_diff"})

        self.memory.save()

    def _ensure_features(self, candidate: Candidate) -> Dict[str, float]:
        """Get or extract features from a candidate."""
        existing = candidate.get("features")
        if isinstance(existing, dict) and len(existing) >= len(FEATURE_KEYS) // 2:
            return {k: _safe_float(existing.get(k, 0.0)) for k in FEATURE_KEYS}
        return self.extract_features(candidate)

    def _synthetic_features(
        self,
        camera_xy: Point,
        gray_pre: Optional[np.ndarray],
        gray_post: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Build features for a click point that didn't match any candidate.

        Uses actual image data at the click point instead of hardcoded values.
        """
        x, y = camera_xy
        features: Dict[str, float] = {k: 0.0 for k in FEATURE_KEYS}

        # Use post-shot gray for patch stats (this is where the hole is)
        gray = gray_post if gray_post is not None else self._latest_gray
        if gray is not None:
            ix, iy = int(round(x)), int(round(y))
            h, w = gray.shape[:2]
            features["x_norm"] = x / max(1.0, float(w))
            features["y_norm"] = y / max(1.0, float(h))

            r = 8
            x0, y0 = max(0, ix - r), max(0, iy - r)
            x1, y1 = min(w, ix + r + 1), min(h, iy + r + 1)
            patch = gray[y0:y1, x0:x1]
            if patch.size > 0:
                features["patch_mean"] = float(np.mean(patch)) / 255.0
                features["patch_std"] = float(np.std(patch)) / 64.0
                if cv2 is not None and patch.shape[0] >= 3 and patch.shape[1] >= 3:
                    lap = cv2.Laplacian(patch, cv2.CV_32F)
                    features["edge_strength"] = float(np.mean(np.abs(lap))) / 32.0

        # Delta between pre and post at click point
        if gray_pre is not None and gray_post is not None and gray_pre.shape == gray_post.shape:
            ix, iy = int(round(x)), int(round(y))
            h, w = gray_pre.shape[:2]
            r = 6
            x0, y0 = max(0, ix - r), max(0, iy - r)
            x1, y1 = min(w, ix + r + 1), min(h, iy + r + 1)
            pre_patch = gray_pre[y0:y1, x0:x1]
            post_patch = gray_post[y0:y1, x0:x1]
            if pre_patch.size > 0 and post_patch.size > 0:
                delta = np.abs(post_patch.astype(np.int16) - pre_patch.astype(np.int16))
                features["pre_shot_change"] = float(np.mean(delta)) / 64.0
                features["change_value"] = features["pre_shot_change"]
                features["center_change"] = features["pre_shot_change"]

        return features

    # ------------------------------------------------------------------
    # Accessors for UI / settings scene
    # ------------------------------------------------------------------

    @property
    def has_new_shot(self) -> bool:
        return self._shot_detected

    @property
    def latest_candidates(self) -> List[Candidate]:
        return self._latest_candidates

    @property
    def pre_shot_gray(self) -> Optional[np.ndarray]:
        return self._pre_shot_gray

    @property
    def post_shot_gray(self) -> Optional[np.ndarray]:
        return self._post_shot_gray


# ======================================================================
# Module-level singleton
# ======================================================================

_RUNTIME: Optional[AIRuntime] = None


def get_ai_runtime(storage_dir: str = "content/ai") -> AIRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = AIRuntime(storage_dir=storage_dir)
    return _RUNTIME
