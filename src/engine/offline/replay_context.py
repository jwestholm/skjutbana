from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass
class ReplayFrame:
    timestamp: float
    gray: np.ndarray


@dataclass
class ReplayAudioEvent:
    shot_id: int
    peak_ts: float
    state: str = "pending"
    emitted: bool = False
    confidence: float = 0.0
    note: str = "offline_replay"
    matched_track_id: int | None = None
    matched_hole_id: str | None = None


class ReplayScannerContext:
    """Minimal HitScanner-shaped adapter consumed by CandidateGeneratorV2.

    V2.12 deliberately adapts the *input* to the live detector instead of
    cloning its algorithms.  When CandidateGeneratorV2 changes in dev, replay
    therefore exercises the same implementation.
    """

    def __init__(
        self,
        pre_frames: Sequence[np.ndarray],
        *,
        shot_id: int,
        peak_ts: float,
        pre_timestamps: Sequence[float] | None = None,
        roi_mask: np.ndarray | None = None,
        artifact_suppression_mask: np.ndarray | None = None,
        known_holes: Sequence[tuple[float, float]] | None = None,
        candidate_limit: int = 500,
        duplicate_radius_px: float = 18.0,
        ground_truth: tuple[float, float] | None = None,
    ) -> None:
        if not pre_frames:
            raise ValueError("ReplayScannerContext requires at least one pre frame")
        shape = pre_frames[0].shape[:2]
        for frame in pre_frames:
            if frame.shape[:2] != shape:
                raise ValueError("Replay pre frames must have identical shapes")
        if pre_timestamps is None:
            count = len(pre_frames)
            pre_timestamps = [peak_ts - 0.05 * float(count - index) for index in range(count)]
        if len(pre_timestamps) != len(pre_frames):
            raise ValueError("pre_timestamps length mismatch")

        self.frame_history = [
            ReplayFrame(float(ts), frame.astype(np.uint8, copy=False))
            for ts, frame in zip(pre_timestamps, pre_frames)
        ]
        self.audio_events = [ReplayAudioEvent(int(shot_id), float(peak_ts))]
        self.candidate_limit = int(candidate_limit)
        self.duplicate_radius_px = float(duplicate_radius_px)
        self.artifact_suppression_mask = artifact_suppression_mask
        self.debug_frames: dict[str, np.ndarray] = {}
        self.last_window_debug: dict[str, float] = {}
        self.shot_diag_enabled = False
        self._roi_mask = roi_mask
        self._known_holes = [(float(x), float(y)) for x, y in (known_holes or [])]
        self._detector_v2_ground_truth: dict[str, Any] | None = None
        if ground_truth is not None:
            self._detector_v2_ground_truth = {
                "shot_id": int(shot_id),
                "camera_x": float(ground_truth[0]),
                "camera_y": float(ground_truth[1]),
            }

    def _frame_roi_mask(self, shape: tuple[int, ...]) -> np.ndarray:
        height, width = int(shape[0]), int(shape[1])
        if self._roi_mask is None:
            return np.full((height, width), 255, dtype=np.uint8)
        mask = self._roi_mask
        if mask.shape[:2] != (height, width):
            raise ValueError("Replay ROI mask shape mismatch")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def _build_pre_shot_background(self) -> np.ndarray | None:
        if not self.frame_history:
            return None
        stack = np.stack([frame.gray.astype(np.float32) for frame in self.frame_history], axis=0)
        return np.median(stack, axis=0).astype(np.uint8)

    def _is_near_known_hole(self, x: float, y: float):
        if not self._known_holes:
            return None
        best = min(
            self._known_holes,
            key=lambda point: math.hypot(float(x) - point[0], float(y) - point[1]),
        )
        distance = math.hypot(float(x) - best[0], float(y) - best[1])
        return best, float(distance)
