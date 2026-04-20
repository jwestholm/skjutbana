from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.engine.camera.camera_manager import camera_manager

from .model import PrototypeMemoryModel
from .settings import load_ai_settings
from .space_mapper import candidate_with_projection, project_camera_point
from .training_data import save_training_example


@dataclass
class AICandidate:
    rank: int
    camera_x: float
    camera_y: float
    detector_score: float
    ai_score: float
    fused_score: float
    feature_vector: list[float]
    source_index: int


class AIRuntime:
    def __init__(self) -> None:
        self.model = PrototypeMemoryModel()
        self.latest_prediction: dict[str, Any] | None = None
        self.latest_training_feedback: dict[str, Any] | None = None
        self._last_shot_serial = 0
        self._latest_gray: np.ndarray | None = None

    def update_observation(self, scanner) -> None:
        try:
            frame = camera_manager.get_latest_frame()
            if frame is not None:
                self._latest_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            self._latest_gray = None

        snapshot = scanner.get_debug_snapshot()
        candidates_raw = snapshot.get("candidates", []) or []
        if not candidates_raw:
            return

        ranked = self.rank_candidates(candidates_raw)
        self._last_shot_serial += 1
        self.latest_prediction = {
            "shot_serial": self._last_shot_serial,
            "timestamp": time.time(),
            "candidates": [self._candidate_to_dict(c) for c in ranked],
            "debug": {
                "state": snapshot.get("state", ""),
                "status": snapshot.get("last_status", ""),
                "tracks": snapshot.get("stable_tracks_count", 0),
            },
        }

    def rank_candidates(self, candidates_raw: list[dict[str, Any]]) -> list[AICandidate]:
        settings = load_ai_settings()
        if not bool(settings.get("enabled", True)):
            return []
        top_k = int(settings.get("top_k", 10))
        ranked: list[AICandidate] = []
        for index, raw in enumerate(candidates_raw):
            feats = self._feature_vector(raw)
            detector_score = float(raw.get("score", 0.0))
            ai_score = self.model.score(feats)
            fused = 0.45 * self._sigmoid(detector_score / 8.0) + 0.55 * ai_score
            ranked.append(
                AICandidate(
                    rank=0,
                    camera_x=float(raw.get("camera_x", raw.get("x", 0.0))),
                    camera_y=float(raw.get("camera_y", raw.get("y", 0.0))),
                    detector_score=detector_score,
                    ai_score=ai_score,
                    fused_score=float(fused),
                    feature_vector=feats,
                    source_index=index,
                )
            )
        ranked.sort(key=lambda item: item.fused_score, reverse=True)
        for i, candidate in enumerate(ranked, start=1):
            candidate.rank = i
        return ranked[:top_k]

    def choose_for_emission(self, default_x: float, default_y: float, scanner_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
        settings = load_ai_settings()
        mode = str(settings.get("mode", "train_only"))
        result = {
            "apply": False,
            "camera_x": float(default_x),
            "camera_y": float(default_y),
            "confidence": 0.0,
            "blend_percent": 0.0,
            "reason": mode,
        }
        if not bool(settings.get("enabled", True)):
            return result
        if mode in {"off", "train_only", "advisory"}:
            return result

        if scanner_snapshot is None:
            return result
        ranked = self.rank_candidates(scanner_snapshot.get("candidates", []) or [])
        if not ranked:
            return result
        best = ranked[0]
        min_conf = float(settings.get("min_confidence", 0.58))
        override_conf = float(settings.get("override_confidence", 0.92))
        if best.fused_score < min_conf:
            result["reason"] = "below_min_confidence"
            return result

        if mode == "ai_only":
            blend = 1.0
        elif mode == "ai_priority" and best.fused_score >= override_conf:
            blend = 1.0
        else:
            blend = float(settings.get("blend_percent", 0.0)) / 100.0
        final_camera_x = float((1.0 - blend) * default_x + blend * best.camera_x)
        final_camera_y = float((1.0 - blend) * default_y + blend * best.camera_y)
        projected = project_camera_point(final_camera_x, final_camera_y)
        result.update(
            {
                "apply": blend > 0.0,
                "camera_x": final_camera_x,
                "camera_y": final_camera_y,
                "screen_x": projected.screen_x,
                "screen_y": projected.screen_y,
                "game_x": projected.game_x,
                "game_y": projected.game_y,
                "confidence": float(best.fused_score),
                "blend_percent": float(blend * 100.0),
                "reason": f"rank_{best.rank}",
            }
        )
        self.latest_prediction = {
            "shot_serial": self._last_shot_serial,
            "timestamp": time.time(),
            "candidates": [self._candidate_to_dict(c) for c in ranked],
            "selected": self._candidate_to_dict(best),
            "resolved_hit": {
                "camera_x": projected.camera_x,
                "camera_y": projected.camera_y,
                "screen_x": projected.screen_x,
                "screen_y": projected.screen_y,
                "game_x": projected.game_x,
                "game_y": projected.game_y,
            },
        }
        return result

    def learn_from_click(self, click_camera_x: float, click_camera_y: float, *, clear_visuals: bool = True) -> dict[str, Any] | None:
        prediction = self.latest_prediction
        if not prediction:
            return None
        settings = load_ai_settings()
        radius = float(settings.get("click_match_radius_px", 36.0))
        candidates = prediction.get("candidates", []) or []
        best_match = None
        best_dist = None
        for cand in candidates:
            dx = float(cand.get("camera_x", 0.0)) - float(click_camera_x)
            dy = float(cand.get("camera_y", 0.0)) - float(click_camera_y)
            dist = math.hypot(dx, dy)
            if best_match is None or dist < best_dist:
                best_match = cand
                best_dist = dist
        positives = 0
        negatives = 0
        accepted = best_match is not None and best_dist is not None and best_dist <= radius
        for cand in candidates:
            feats = list(cand.get("feature_vector", []))
            if not feats:
                continue
            label = 1 if accepted and cand is best_match else 0
            if label:
                self.model.add_sample(feats, 1, weight=1.5, source="training_click")
                positives += 1
            else:
                self.model.add_sample(feats, 0, weight=1.0, source="training_click")
                negatives += 1
        self.model.save()

        click_projected = project_camera_point(click_camera_x, click_camera_y)
        payload = {
            "timestamp": time.time(),
            "shot_serial": prediction.get("shot_serial"),
            "click_camera_x": float(click_camera_x),
            "click_camera_y": float(click_camera_y),
            "click_screen_x": float(click_projected.screen_x),
            "click_screen_y": float(click_projected.screen_y),
            "accepted_candidate": best_match,
            "accepted": bool(accepted),
            "best_distance": None if best_dist is None else float(best_dist),
            "candidates": candidates,
            "model_summary": self.model.summary(),
            "positives_added": int(positives),
            "negatives_added": int(negatives),
        }
        save_training_example(payload)
        self.latest_training_feedback = payload
        if clear_visuals:
            self.clear_visual_state()
        return payload

    def clear_visual_state(self) -> None:
        self.latest_prediction = None

    def _feature_vector(self, raw: dict[str, Any]) -> list[float]:
        x = float(raw.get("camera_x", raw.get("x", 0.0)))
        y = float(raw.get("camera_y", raw.get("y", 0.0)))
        detector_score = float(raw.get("score", 0.0))
        area = float(raw.get("area", 0.0))
        radius = float(raw.get("radius", 0.0))
        circularity = float(raw.get("circularity", 0.0))
        local_contrast = float(raw.get("local_contrast_gain", raw.get("local_contrast", 0.0)))
        center_delta = float(raw.get("center_delta", raw.get("center_darkening", 0.0)))
        pre_delta = float(raw.get("pre_shot_delta", raw.get("delta_pre", 0.0)))
        recent_delta = float(raw.get("recent_delta", raw.get("delta_recent", 0.0)))
        blackhat = float(raw.get("blackhat", 0.0))
        track_hits = float(raw.get("track_hits", raw.get("hits", 0.0)))
        patch_mean, patch_std = self._patch_stats(x, y)
        return [
            detector_score,
            area,
            radius,
            circularity,
            local_contrast,
            center_delta,
            pre_delta,
            recent_delta,
            blackhat,
            track_hits,
            patch_mean,
            patch_std,
        ]

    def _patch_stats(self, x: float, y: float) -> tuple[float, float]:
        if self._latest_gray is None:
            return 0.0, 0.0
        ix = int(round(x))
        iy = int(round(y))
        h, w = self._latest_gray.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return 0.0, 0.0
        x0 = max(0, ix - 8)
        y0 = max(0, iy - 8)
        x1 = min(w, ix + 9)
        y1 = min(h, iy + 9)
        patch = self._latest_gray[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0, 0.0
        return float(np.mean(patch) / 255.0), float(np.std(patch) / 255.0)

    def _candidate_to_dict(self, candidate: AICandidate) -> dict[str, Any]:
        base = {
            "rank": int(candidate.rank),
            "camera_x": float(candidate.camera_x),
            "camera_y": float(candidate.camera_y),
            "detector_score": float(candidate.detector_score),
            "ai_score": float(candidate.ai_score),
            "fused_score": float(candidate.fused_score),
            "feature_vector": list(candidate.feature_vector),
            "source_index": int(candidate.source_index),
        }
        return candidate_with_projection(base)

    @staticmethod
    def _sigmoid(v: float) -> float:
        return float(1.0 / (1.0 + math.exp(-v)))


ai_runtime = AIRuntime()
