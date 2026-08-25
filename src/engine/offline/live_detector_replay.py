from __future__ import annotations

"""Hardware-free adapter for the *current live* HitScanner V1->V2 hybrid.

The important design choice is that replay does not clone detector logic.  It
instantiates the real HitScanner, installs the same CandidateGeneratorV2 wrapper
used by the application, and only replaces the two hardware/environment inputs
that do not exist offline: the ROI mask and the pre-shot background lookup.

This module is intentionally imported lazily by offline replay.  Archive
inspection and direct-image evidence tools therefore remain usable on machines
that do not have the complete game/runtime dependencies installed.
"""

from dataclasses import dataclass
from types import MethodType
from typing import Any, Sequence

import numpy as np


class _FrozenConfig:
    """Read-only detector config that prevents replay diagnostics from writing."""

    def __init__(self, values: dict[str, Any]):
        self.values = dict(values)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass
class LiveDetectorReplayResult:
    candidates: list[dict[str, Any]]
    v1_candidates: list[dict[str, Any]]
    v2_candidates: list[dict[str, Any]]
    agreement_candidates: list[dict[str, Any]]
    telemetry: dict[str, Any]


class LiveHybridReplayDetector:
    """Replay the current live V1+V2 candidate detector without hardware."""

    def __init__(self) -> None:
        # These imports are deliberately here rather than module-global.  The
        # inspector/evidence tools can run without importing camera/audio/game
        # runtime modules, while full replay inside the repo exercises them.
        from src.engine.camera.candidate_generator_v2 import install_candidate_generator_v2
        from src.engine.camera.hit_scanner import HitScanner

        install_candidate_generator_v2(HitScanner)
        self._scanner_cls = HitScanner
        self._engine = getattr(HitScanner, "_candidate_generator_v2_engine", None)
        if self._engine is None:
            raise RuntimeError("CandidateGeneratorV2 hybrid wrapper was not installed")

        # The offline benchmark must never append to the live shot diagnostics.
        values = self._engine.config.snapshot()
        values["diagnostics_enabled"] = False
        values["debug_frames_enabled"] = False
        self._engine.config = _FrozenConfig(values)

    @staticmethod
    def _median_pre(pre_frames: Sequence[np.ndarray]) -> np.ndarray:
        stack = np.stack([frame.astype(np.float32) for frame in pre_frames], axis=0)
        return np.median(stack, axis=0).astype(np.uint8)

    def detect(
        self,
        *,
        pre_frames: Sequence[np.ndarray],
        post_frames: Sequence[np.ndarray],
        known_holes: Sequence[tuple[float, float]] = (),
        ground_truth: tuple[float, float] | None = None,
        candidate_limit: int = 500,
        post_interval_s: float = 1.0 / 30.0,
    ) -> LiveDetectorReplayResult:
        if not pre_frames or not post_frames:
            raise ValueError("Live detector replay requires pre and post frames")

        from src.engine.camera.hit_scanner import AudioShotEvent, ScanportFrame

        shape = pre_frames[0].shape[:2]
        if any(frame.shape[:2] != shape for frame in list(pre_frames) + list(post_frames)):
            raise ValueError("Live detector replay frame shape mismatch")

        try:
            self._engine.reset_runtime_state()
        except Exception:
            pass

        scanner = self._scanner_cls()
        scanner.shot_diag_enabled = False
        scanner.candidate_limit = max(1, int(candidate_limit))
        scanner.debug_frames = {}
        scanner.last_window_debug = {}
        scanner.artifact_suppression_mask = None

        peak_ts = 1.0
        # Put every pre frame inside V2's immediate pre-shot window (<=320 ms),
        # while leaving a non-zero gap to the synthetic audio peak.
        count = len(pre_frames)
        if count == 1:
            timestamps = [peak_ts - 0.12]
        else:
            timestamps = list(np.linspace(peak_ts - 0.30, peak_ts - 0.04, count))
        scanner.frame_history.clear()
        for timestamp, frame in zip(timestamps, pre_frames):
            scanner.frame_history.append(
                ScanportFrame(timestamp=float(timestamp), gray=frame.astype(np.uint8, copy=False))
            )

        scanner.audio_events.clear()
        scanner.audio_events.append(
            AudioShotEvent(
                shot_id=1,
                peak_ts=peak_ts,
                created_at=peak_ts,
            )
        )

        reference = self._median_pre(pre_frames)
        scanner.pre_shot_snapshot = reference.copy()
        scanner.pre_shot_snapshot_ts = float(timestamps[-1])
        # This is only a fallback for current V1 code.  The method below is
        # overridden to return the replay pre-shot stack directly.
        scanner.scene_reference_gray = reference.copy()
        scanner.surface_reference_gray = None

        scanner.known_holes = [
            {
                "hole_id": float(index + 1),
                "camera_x": float(x),
                "camera_y": float(y),
                "score": 10.0,
                "timestamp": 0.0,
                "hit_count": 1.0,
            }
            for index, (x, y) in enumerate(known_holes)
        ]
        if ground_truth is not None:
            scanner._detector_v2_ground_truth = {
                "shot_id": 1,
                "camera_x": float(ground_truth[0]),
                "camera_y": float(ground_truth[1]),
            }

        def offline_pre_shot_background(_self: Any) -> np.ndarray:
            return reference

        def offline_roi_mask(_self: Any, frame_shape: tuple[int, ...]) -> np.ndarray:
            height, width = int(frame_shape[0]), int(frame_shape[1])
            return np.full((height, width), 255, dtype=np.uint8)

        # These are the adapter boundary.  Everything after them is the same
        # _detect_frame_candidates implementation/wrapper used by live runtime.
        scanner._build_pre_shot_background = MethodType(offline_pre_shot_background, scanner)
        scanner._frame_roi_mask = MethodType(offline_roi_mask, scanner)

        final: list[dict[str, Any]] = []
        for index, frame in enumerate(post_frames, start=1):
            frame_ts = peak_ts + max(0.020, float(index) * float(post_interval_s))
            detected = scanner._detect_frame_candidates(
                gray=frame.astype(np.uint8, copy=False),
                frame_ts=float(frame_ts),
            )
            final = [dict(candidate) for candidate in detected]

        # Provenance is already written by the live V2 hybrid merge.  Keep
        # sub-sources separately measurable without changing the live code.
        v1 = [
            dict(candidate)
            for candidate in final
            if float(candidate.get("detector_v1", 0.0) or 0.0) > 0.0
        ]
        v2 = [
            dict(candidate)
            for candidate in final
            if float(candidate.get("detector_v2", 0.0) or 0.0) > 0.0
        ]
        agreement = [
            dict(candidate)
            for candidate in final
            if float(candidate.get("detector_agreement", 0.0) or 0.0) > 0.0
            or (
                float(candidate.get("detector_v1", 0.0) or 0.0) > 0.0
                and float(candidate.get("detector_v2", 0.0) or 0.0) > 0.0
            )
        ]

        telemetry = dict(getattr(scanner, "last_window_debug", {}) or {})
        telemetry.update(
            {
                "offline_live_hybrid": True,
                "final_candidate_count": len(final),
                "v1_provenance_count": len(v1),
                "v2_provenance_count": len(v2),
                "agreement_count": len(agreement),
            }
        )

        try:
            self._engine.reset_runtime_state()
        except Exception:
            pass

        return LiveDetectorReplayResult(
            candidates=final,
            v1_candidates=v1,
            v2_candidates=v2,
            agreement_candidates=agreement,
            telemetry=telemetry,
        )
