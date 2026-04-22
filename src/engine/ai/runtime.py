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
    "click_match_radius_px": 42.0,
    "min_confidence": 0.58,
    "override_confidence": 0.92,
    "max_negatives_per_click": 3,
    "trust_percent": 0,
    "show_overlay": True,
    "auto_learn": True,
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

        pos_dists = self._weighted_distances(norm, self.positives[-64:], now)
        neg_dists = self._weighted_distances(norm, self.negatives[-128:], now)

        pos_best = min(pos_dists) if pos_dists else 4.0
        neg_best = min(neg_dists) if neg_dists else 4.0

        # Sigmoid-ish mapping: closer to positives = higher score
        raw = 0.5 + (neg_best - pos_best) / 6.0
        return max(0.0, min(1.0, raw))

    def _weighted_distances(
        self, norm_features: Dict[str, float], memories: List[Dict[str, Any]], now: float
    ) -> List[float]:
        dists = []
        for mem in memories:
            mem_features = self._normalize(mem.get("features", {}))
            # Euclidean distance over shared normalized features
            total = 0.0
            count = 0
            for key in FEATURE_KEYS:
                a = norm_features.get(key, 0.5)
                b = mem_features.get(key, 0.5)
                total += (a - b) ** 2
                count += 1
            if count == 0:
                continue
            dist = math.sqrt(total / count)
            # Time decay: memories older than 1 hour get slightly penalized
            age_hours = (now - _safe_float(mem.get("timestamp", now))) / 3600.0
            time_penalty = 1.0 + 0.05 * max(0.0, age_hours)
            dists.append(dist * time_penalty)
        return dists

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

        # Observation state — updated every frame by bootstrap patch
        self._last_audio_count: int = 0
        self._latest_candidates: List[Candidate] = []
        self._latest_gray: Optional[np.ndarray] = None
        self._pre_shot_gray: Optional[np.ndarray] = None
        self._post_shot_gray: Optional[np.ndarray] = None
        self._latest_snapshot: Optional[Dict[str, Any]] = None
        self._shot_detected: bool = False

        # Session tracking
        self.session_stats: Dict[str, Any] = {
            "shots_seen": 0,
            "clicks": 0,
            "last_click_camera": None,
        }

    def save_settings(self) -> None:
        AI_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Bootstrap hooks (called from patched HitScanner)
    # ------------------------------------------------------------------

    def observe_scanner(self, scanner) -> None:
        """Called every frame after hit_scanner.update(). Captures state for AI."""
        if not self.settings.get("enabled", True):
            return

        # Reuse scanner's already-converted gray frame instead of converting again
        debug_frames = getattr(scanner, "debug_frames", {})
        camera_gray = debug_frames.get("camera_gray")
        if camera_gray is not None:
            self._latest_gray = camera_gray

        # Detect new shot by watching audio_event_count
        current_count = getattr(scanner, "audio_event_count", 0)
        if current_count > self._last_audio_count:
            self._capture_pre_shot_frame(scanner)
            self._shot_detected = True
            self.session_stats["shots_seen"] += 1
        self._last_audio_count = current_count

        # Capture all candidates (not just top 10)
        all_candidates = list(getattr(scanner, "last_candidates", []))
        if all_candidates:
            self._latest_candidates = all_candidates
            if self._shot_detected and self._latest_gray is not None:
                self._post_shot_gray = self._latest_gray.copy()

    def _capture_pre_shot_frame(self, scanner) -> None:
        """Get a frame from before the shot using scanner's frame_history."""
        frame_history = getattr(scanner, "frame_history", None)
        if frame_history is None or len(frame_history) < 2:
            self._pre_shot_gray = self._latest_gray
            return

        # Get the second-to-last frame (more likely to be pre-shot)
        audio_events = getattr(scanner, "audio_events", [])
        earliest_peak = None
        for ev in audio_events:
            if getattr(ev, "state", "") == "pending":
                ts = getattr(ev, "peak_ts", None)
                if ts is not None and (earliest_peak is None or ts < earliest_peak):
                    earliest_peak = ts

        if earliest_peak is not None:
            # Find frame before the peak
            for fr in reversed(frame_history):
                if fr.timestamp < earliest_peak - 0.04:
                    self._pre_shot_gray = fr.gray.copy()
                    return

        # Fallback: use oldest available frame
        if len(frame_history) >= 3:
            self._pre_shot_gray = frame_history[-3].gray.copy()
        elif self._latest_gray is not None:
            self._pre_shot_gray = self._latest_gray.copy()
        else:
            self._pre_shot_gray = None

    def choose_for_emission(
        self, default_x: float, default_y: float
    ) -> Dict[str, Any]:
        """Called from bootstrap when hit_scanner wants to emit a hit.
        Returns whether AI wants to override the position."""
        result: Dict[str, Any] = {
            "apply": False,
            "camera_x": float(default_x),
            "camera_y": float(default_y),
            "confidence": 0.0,
            "reason": "passthrough",
        }

        mode = str(self.settings.get("mode", "train_only"))
        if mode in {"off", "train_only", "advisory"}:
            return result

        if not self._latest_candidates:
            return result

        # Rank candidates and pick the best
        ranked = self.rank_candidates(self._latest_candidates)
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

        # Patch stats from camera frame
        patch_mean, patch_std, edge_strength = self._patch_stats(x, y)
        features["patch_mean"] = patch_mean
        features["patch_std"] = patch_std
        features["edge_strength"] = edge_strength

        # Normalized position (helps AI learn edge-of-frame bias)
        if self._latest_gray is not None:
            h, w = self._latest_gray.shape[:2]
            features["x_norm"] = x / max(1.0, float(w))
            features["y_norm"] = y / max(1.0, float(h))
        else:
            features["x_norm"] = 0.0
            features["y_norm"] = 0.0

        return features

    def _patch_stats(self, x: float, y: float) -> Tuple[float, float, float]:
        """Extract patch statistics around a point from the latest gray frame."""
        if self._latest_gray is None:
            return 0.0, 0.0, 0.0
        ix, iy = int(round(x)), int(round(y))
        h, w = self._latest_gray.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return 0.0, 0.0, 0.0

        r = 8
        x0, y0 = max(0, ix - r), max(0, iy - r)
        x1, y1 = min(w, ix + r + 1), min(h, iy + r + 1)
        patch = self._latest_gray[y0:y1, x0:x1]
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
    # Candidate ranking
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
            combined = det_weight * det_norm + ai_weight * ai_score

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

    # ------------------------------------------------------------------
    # Training (click-based learning)
    # ------------------------------------------------------------------

    def learn_from_click(
        self,
        click_camera_xy: Point,
        shown_candidates: Sequence[Candidate],
        gray_pre: Optional[np.ndarray] = None,
        gray_post: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the AI from a user click.
        - Nearest shown candidate within radius → positive
        - A few worst-scoring shown candidates → negative (balanced, not all)
        - If no candidate near click → synthetic positive from click location
        """
        self.session_stats["clicks"] += 1
        self.session_stats["last_click_camera"] = [float(click_camera_xy[0]), float(click_camera_xy[1])]
        self.memory.stats["total_clicks"] = self.memory.stats.get("total_clicks", 0) + 1
        self.memory.stats["local_updates_since_import"] = self.memory.stats.get("local_updates_since_import", 0) + 1

        click_radius = float(self.settings.get("click_match_radius_px", 42.0))
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

        # If a candidate is close enough, use it as positive
        if nearest_idx is not None and nearest_dist <= click_radius:
            cand = shown_candidates[nearest_idx]
            features = self._ensure_features(cand)
            self.memory.add_positive(features, {"kind": "candidate_match", "distance": nearest_dist})
            positive_added = True

        # Add limited negatives — pick the worst-scoring ones that aren't the match
        neg_candidates = [
            (i, c) for i, c in enumerate(shown_candidates)
            if not (positive_added and i == nearest_idx)
        ]
        # Sort by combined_score ascending (worst first) to pick the most useful negatives
        neg_candidates.sort(key=lambda ic: _safe_float(ic[1].get("combined_score", ic[1].get("score", 0.0))))
        neg_added = 0
        for i, cand in neg_candidates:
            if neg_added >= max_neg:
                break
            features = self._ensure_features(cand)
            self.memory.add_negative(features, {"kind": "shown_other"})
            neg_added += 1

        # If no candidate matched, create synthetic positive from click point
        if not positive_added:
            features = self._synthetic_features(click_camera_xy, gray_pre, gray_post)
            self.memory.add_positive(features, {"kind": "synthetic_click"})
            positive_added = True

        self.memory.save()
        self._shot_detected = False

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
        """Build features for a click point that didn't match any candidate."""
        x, y = camera_xy
        features: Dict[str, float] = {k: 0.0 for k in FEATURE_KEYS}
        features["detector_score"] = 0.0
        features["area"] = 5.0  # Typical small hole
        features["radius"] = 1.5
        features["circularity"] = 0.8

        # Use post-shot gray for patch stats
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
