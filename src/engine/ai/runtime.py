
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required for engine.ai.runtime") from exc


Point = Tuple[float, float]
Candidate = Dict[str, Any]


DEFAULT_SETTINGS: Dict[str, Any] = {
    "enabled": True,
    "mode": "train_only",
    "top_k": 10,
    "memory_limit_positive": 400,
    "memory_limit_negative": 1200,
    "click_match_radius_px": 42.0,
    "training_roi_margin_x": 0.18,
    "training_roi_margin_y": 0.16,
    "diff_threshold": 18,
    "min_blob_area": 6,
    "max_blob_area": 1500,
    "edge_reject_px": 6,
    "trust_percent": 0,
    "allow_black_background": True,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _candidate_xy(candidate: Candidate) -> Optional[Point]:
    pairs = [
        ("camera_x", "camera_y"),
        ("x", "y"),
        ("cx", "cy"),
        ("screen_x", "screen_y"),
    ]
    for kx, ky in pairs:
        if kx in candidate and ky in candidate:
            return (_safe_float(candidate[kx]), _safe_float(candidate[ky]))
    return None


class SimpleAIMemory:
    """
    Small bounded online learner.
    Stores compact feature vectors and updates immediately after the user clicks.
    This is intentionally tiny and transparent rather than a heavy NN.
    """

    def __init__(
        self,
        storage_dir: Path,
        positive_limit: int = 400,
        negative_limit: int = 1200,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.storage_dir / "memory.json"
        self.positive_limit = int(positive_limit)
        self.negative_limit = int(negative_limit)
        self.positives: List[Dict[str, Any]] = []
        self.negatives: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {
            "positive_count": 0,
            "negative_count": 0,
            "last_updated": None,
            "version": 1,
        }
        self.load()

    def load(self) -> None:
        if not self.memory_file.exists():
            return
        try:
            data = json.loads(self.memory_file.read_text(encoding="utf-8"))
            self.positives = list(data.get("positives", []))[-self.positive_limit :]
            self.negatives = list(data.get("negatives", []))[-self.negative_limit :]
            self.stats.update(dict(data.get("stats", {})))
        except Exception:
            self.positives = []
            self.negatives = []

    def save(self) -> None:
        self.stats["positive_count"] = len(self.positives)
        self.stats["negative_count"] = len(self.negatives)
        self.stats["last_updated"] = time.time()
        payload = {
            "positives": self.positives[-self.positive_limit :],
            "negatives": self.negatives[-self.negative_limit :],
            "stats": self.stats,
        }
        self.memory_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _trim(self) -> None:
        if len(self.positives) > self.positive_limit:
            self.positives = self.positives[-self.positive_limit :]
        if len(self.negatives) > self.negative_limit:
            self.negatives = self.negatives[-self.negative_limit :]

    def add_positive(self, features: Dict[str, float], meta: Optional[Dict[str, Any]] = None) -> None:
        self.positives.append({"features": features, "meta": meta or {}})
        self._trim()

    def add_negative(self, features: Dict[str, float], meta: Optional[Dict[str, Any]] = None) -> None:
        self.negatives.append({"features": features, "meta": meta or {}})
        self._trim()

    @staticmethod
    def _distance(a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = sorted(set(a.keys()) | set(b.keys()))
        if not keys:
            return 9999.0
        total = 0.0
        for key in keys:
            av = _safe_float(a.get(key, 0.0))
            bv = _safe_float(b.get(key, 0.0))
            total += (av - bv) ** 2
        return math.sqrt(total / max(1, len(keys)))

    def score(self, features: Dict[str, float]) -> float:
        """
        Returns 0..1, higher means more hit-like.
        """
        pos = [self._distance(features, row["features"]) for row in self.positives[-64:]]
        neg = [self._distance(features, row["features"]) for row in self.negatives[-128:]]
        if not pos and not neg:
            return 0.5
        pos_best = min(pos) if pos else 4.0
        neg_best = min(neg) if neg else 4.0
        # Convert relative closeness into a bounded confidence.
        raw = 0.5 + (neg_best - pos_best) / 8.0
        return max(0.0, min(1.0, raw))


class AIRuntime:
    def __init__(self, storage_dir: str = "content/ai") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.settings_path = self.storage_dir / "settings.json"
        self.settings = dict(DEFAULT_SETTINGS)
        if self.settings_path.exists():
            try:
                loaded = json.loads(self.settings_path.read_text(encoding="utf-8"))
                self.settings.update(loaded)
            except Exception:
                pass

        self.memory = SimpleAIMemory(
            storage_dir=self.storage_dir,
            positive_limit=int(self.settings.get("memory_limit_positive", 400)),
            negative_limit=int(self.settings.get("memory_limit_negative", 1200)),
        )
        self.session_stats: Dict[str, Any] = {
            "shots_seen": 0,
            "click_updates": 0,
            "last_click": None,
        }

    def save(self) -> None:
        self.settings_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
        self.memory.save()

    def training_roi_rect(self, width: int, height: int) -> Tuple[int, int, int, int]:
        margin_x = float(self.settings.get("training_roi_margin_x", 0.18))
        margin_y = float(self.settings.get("training_roi_margin_y", 0.16))
        x = int(width * margin_x)
        y = int(height * margin_y)
        w = int(width * (1.0 - margin_x * 2.0))
        h = int(height * (1.0 - margin_y * 2.0))
        return (x, y, max(1, w), max(1, h))

    def _extract_patch_features(
        self,
        gray_pre: Optional["np.ndarray"],
        gray_post: "np.ndarray",
        x: int,
        y: int,
        radius: int = 11,
    ) -> Dict[str, float]:
        h, w = gray_post.shape[:2]
        x0 = max(0, x - radius)
        y0 = max(0, y - radius)
        x1 = min(w, x + radius + 1)
        y1 = min(h, y + radius + 1)
        patch = gray_post[y0:y1, x0:x1]
        if patch.size == 0:
            return {"mean": 0.0, "std": 0.0, "center_darkness": 0.0, "delta_mean": 0.0}
        center = gray_post[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)]
        patch_mean = float(np.mean(patch))
        patch_std = float(np.std(patch))
        center_darkness = 255.0 - float(np.mean(center))
        delta_mean = 0.0
        if gray_pre is not None and gray_pre.shape == gray_post.shape:
            pre_patch = gray_pre[y0:y1, x0:x1]
            delta = np.abs(patch.astype(np.int16) - pre_patch.astype(np.int16))
            delta_mean = float(np.mean(delta))
        if cv2 is not None:
            lap = cv2.Laplacian(patch, cv2.CV_32F)
            edge = float(np.mean(np.abs(lap)))
        else:
            gy, gx = np.gradient(patch.astype(np.float32))
            edge = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
        return {
            "mean": patch_mean / 255.0,
            "std": patch_std / 64.0,
            "center_darkness": center_darkness / 255.0,
            "delta_mean": delta_mean / 64.0,
            "edge": edge / 32.0,
            "x_norm": x / max(1.0, float(w)),
            "y_norm": y / max(1.0, float(h)),
        }

    def _normalize_detector_candidate(
        self,
        candidate: Candidate,
        gray_pre: Optional["np.ndarray"],
        gray_post: Optional["np.ndarray"],
        roi_rect: Tuple[int, int, int, int],
    ) -> Optional[Candidate]:
        xy = _candidate_xy(candidate)
        if xy is None:
            return None
        x, y = int(round(xy[0])), int(round(xy[1]))
        rx, ry, rw, rh = roi_rect
        if not (rx <= x < rx + rw and ry <= y < ry + rh):
            return None
        edge_pad = int(self.settings.get("edge_reject_px", 6))
        if x < rx + edge_pad or x >= rx + rw - edge_pad or y < ry + edge_pad or y >= ry + rh - edge_pad:
            return None
        features = dict(candidate.get("features", {}))
        if gray_post is not None:
            features.update(self._extract_patch_features(gray_pre, gray_post, x, y))
        normalized = dict(candidate)
        normalized["camera_x"] = float(x)
        normalized["camera_y"] = float(y)
        normalized["features"] = features
        normalized["detector_score"] = _safe_float(
            candidate.get("score", candidate.get("detector_score", 0.0))
        )
        return normalized

    def _generate_diff_candidates(
        self,
        gray_pre: Optional["np.ndarray"],
        gray_post: Optional["np.ndarray"],
        roi_rect: Tuple[int, int, int, int],
    ) -> List[Candidate]:
        if gray_pre is None or gray_post is None or gray_pre.shape != gray_post.shape:
            return []
        rx, ry, rw, rh = roi_rect
        pre_roi = gray_pre[ry:ry + rh, rx:rx + rw]
        post_roi = gray_post[ry:ry + rh, rx:rx + rw]
        diff = np.abs(post_roi.astype(np.int16) - pre_roi.astype(np.int16)).astype(np.uint8)
        threshold = int(self.settings.get("diff_threshold", 18))
        mask = (diff >= threshold).astype(np.uint8) * 255

        if cv2 is not None:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blobs = contours
        else:
            ys, xs = np.where(mask > 0)
            blobs = []
            for x, y in zip(xs.tolist(), ys.tolist()):
                blobs.append(np.array([[[x, y]]], dtype=np.int32))

        results: List[Candidate] = []
        min_area = int(self.settings.get("min_blob_area", 6))
        max_area = int(self.settings.get("max_blob_area", 1500))
        for blob in blobs:
            if cv2 is not None:
                area = float(cv2.contourArea(blob))
                if area < min_area or area > max_area:
                    continue
                M = cv2.moments(blob)
                if not M.get("m00"):
                    continue
                local_x = int(M["m10"] / M["m00"])
                local_y = int(M["m01"] / M["m00"])
                x = rx + local_x
                y = ry + local_y
            else:
                x = rx + int(blob[0][0][0])
                y = ry + int(blob[0][0][1])
                area = 1.0
            features = self._extract_patch_features(gray_pre, gray_post, x, y)
            detector_score = features.get("delta_mean", 0.0) * 8.0 + features.get("center_darkness", 0.0) * 2.0
            results.append(
                {
                    "source": "diff",
                    "camera_x": float(x),
                    "camera_y": float(y),
                    "detector_score": float(detector_score),
                    "area": float(area),
                    "features": features,
                }
            )
        return results

    @staticmethod
    def _dedupe_candidates(candidates: Sequence[Candidate], radius_px: float = 14.0) -> List[Candidate]:
        kept: List[Candidate] = []
        for candidate in sorted(candidates, key=lambda c: _safe_float(c.get("detector_score", 0.0)), reverse=True):
            xy = _candidate_xy(candidate)
            if xy is None:
                continue
            dup = False
            for existing in kept:
                exy = _candidate_xy(existing)
                if exy is None:
                    continue
                if math.dist(xy, exy) <= radius_px:
                    dup = True
                    break
            if not dup:
                kept.append(candidate)
        return kept

    def rank_candidates(
        self,
        gray_pre: Optional["np.ndarray"],
        gray_post: Optional["np.ndarray"],
        detector_candidates: Optional[Sequence[Candidate]] = None,
        limit: Optional[int] = None,
    ) -> List[Candidate]:
        if gray_post is None:
            return []
        height, width = gray_post.shape[:2]
        roi_rect = self.training_roi_rect(width, height)

        merged: List[Candidate] = []
        for row in detector_candidates or []:
            candidate = self._normalize_detector_candidate(row, gray_pre, gray_post, roi_rect)
            if candidate is not None:
                merged.append(candidate)

        merged.extend(self._generate_diff_candidates(gray_pre, gray_post, roi_rect))
        merged = self._dedupe_candidates(merged)
        top_k = int(limit or self.settings.get("top_k", 10))

        for candidate in merged:
            features = dict(candidate.get("features", {}))
            ai_score = self.memory.score(features)
            detector_score = _safe_float(candidate.get("detector_score", 0.0))
            combined = detector_score * 0.35 + ai_score * 0.65
            candidate["ai_score"] = float(ai_score)
            candidate["combined_score"] = float(combined)

        merged.sort(key=lambda c: _safe_float(c.get("combined_score", 0.0)), reverse=True)
        return merged[:top_k]

    def learn_from_click(
        self,
        click_camera_xy: Point,
        shown_candidates: Sequence[Candidate],
        gray_pre: Optional["np.ndarray"] = None,
        gray_post: Optional["np.ndarray"] = None,
    ) -> Dict[str, Any]:
        """
        This is the important part: yes, the click trains the AI.
        - nearest shown candidate within radius becomes positive
        - other shown candidates become negatives
        - if no shown candidate is near enough, a synthetic positive sample is created
          directly from the click location
        """
        self.session_stats["click_updates"] += 1
        self.session_stats["last_click"] = [float(click_camera_xy[0]), float(click_camera_xy[1])]
        click_radius = float(self.settings.get("click_match_radius_px", 42.0))

        nearest_index: Optional[int] = None
        nearest_distance = 10_000.0
        for index, candidate in enumerate(shown_candidates):
            xy = _candidate_xy(candidate)
            if xy is None:
                continue
            dist = math.dist(click_camera_xy, xy)
            if dist < nearest_distance:
                nearest_distance = dist
                nearest_index = index

        positive_added = False
        if nearest_index is not None and nearest_distance <= click_radius:
            candidate = shown_candidates[nearest_index]
            features = dict(candidate.get("features", {}))
            features["click_distance"] = nearest_distance / max(1.0, click_radius)
            self.memory.add_positive(features, {"kind": "candidate_match"})
            positive_added = True

        # All non-winning shown candidates become negatives.
        for index, candidate in enumerate(shown_candidates):
            if positive_added and index == nearest_index:
                continue
            features = dict(candidate.get("features", {}))
            xy = _candidate_xy(candidate)
            if xy is not None:
                features["distance_to_click"] = math.dist(click_camera_xy, xy) / max(1.0, click_radius)
            self.memory.add_negative(features, {"kind": "shown_other"})

        # If nothing matched closely, create a positive sample from the clicked spot itself.
        if not positive_added and gray_post is not None:
            x = int(round(click_camera_xy[0]))
            y = int(round(click_camera_xy[1]))
            features = self._extract_patch_features(gray_pre, gray_post, x, y)
            self.memory.add_positive(features, {"kind": "synthetic_click_positive"})
            positive_added = True

        self.memory.save()
        return {
            "positive_added": positive_added,
            "nearest_index": nearest_index,
            "nearest_distance": nearest_distance,
        }


_RUNTIME: Optional[AIRuntime] = None


def get_ai_runtime(storage_dir: str = "content/ai") -> AIRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = AIRuntime(storage_dir=storage_dir)
    return _RUNTIME
