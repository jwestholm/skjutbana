from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


CONFIG_PATH = Path("content/ai/detector_v2.json")
DIAGNOSTICS_DIR = Path("content/ai/detector_v2")
DIAGNOSTICS_JSONL = DIAGNOSTICS_DIR / "shot_diagnostics.jsonl"

DEFAULT_CONFIG: dict[str, Any] = {
    # Master switch. False = exact legacy HitScanner candidate generator.
    "enabled": True,
    "hybrid_with_legacy": True,

    # Recent pre-shot frames. The old detector normally compares against a
    # frame ~500 ms old. V2 prefers frames immediately BEFORE the audio peak.
    "pre_stack_frames": 3,
    "pre_stack_window_s": 0.32,
    "pre_stack_min_gap_s": 0.006,
    "stable_stack_max_mad": 4.5,

    # Do not begin adding V2 evidence at the exact audio timestamp. This avoids
    # building persistent tracks from the last pre-hole frame while still being
    # short enough for normal camera frame intervals. V1 remains active during
    # this tiny delay because the hybrid always runs the legacy detector first.
    "min_event_age_s": 0.015,

    # Mild blur only. Synthetic holes are only a few screen pixels across.
    "blur_kernel": 3,
    "blur_sigma": 0.55,

    # Sub-pixel/global camera registration. It is soft-fail: low-confidence or
    # too-large shifts are ignored.
    "registration_enabled": True,
    "registration_max_shift_px": 4.0,
    "registration_min_shift_px": 0.35,
    "registration_min_response": 0.08,
    "registration_max_dimension": 720,

    # Per-pixel temporal noise model from pre-shot frames.
    "noise_floor": 1.35,
    "noise_cap": 28.0,

    # Multi-scale local contrast on the temporal difference image. Box means
    # are used instead of repeated Gaussian filters because this path runs on
    # every camera frame. Each pair is [small_window, broad_window].
    "local_contrast_scales": [[3, 9], [5, 15], [9, 27]],

    # Continuous saliency composition.
    "weight_zscore": 5.8,
    "weight_absdiff": 0.72,
    "weight_dog": 1.75,
    "weight_darkening": 0.32,

    # Candidate gate. The robust threshold is median + sigma*MAD on the already
    # noise-normalised saliency map, rather than "top 1.2% of the whole ROI".
    "robust_sigma": 3.2,
    "min_saliency": 10.0,
    "min_temporal_change": 1.8,
    "min_zscore": 1.5,
    "strong_temporal_change": 4.0,

    # Small local-max filter followed by explicit Euclidean NMS.
    "local_max_kernel": 5,
    "nms_radius_px": 5.0,

    # Spatial coverage prevents one shimmering/textured region from consuming
    # all candidate slots. Keep a few local maxima from every tile.
    "tile_columns": 8,
    "tile_rows": 6,
    "per_tile_candidates": 4,
    "global_extra_candidates": 45,
    "max_v2_candidates": 170,

    # Hybrid merge. V2 gets reserved slots so a noisy V1 list cannot crowd out
    # all high-recall V2 points before the AI gets to rank them.
    "merge_radius_px": 5.5,
    "agreement_bonus": 1.5,
    "v2_reserved_slots": 110,

    # Existing artifact mask becomes a SOFT prior for V2. A real hit may still
    # exist on a pixel that the white/black projector calibration considered an
    # "artifact".
    "artifact_inactive_weight": 0.58,
    "roi_edge_margin_px": 5,

    # Camera/projector edge residuals often dominate simple frame subtraction.
    # This is deliberately soft; true holes on printed/projected edges must
    # remain possible candidates.
    "edge_penalty_enabled": True,
    "edge_gradient_start": 24.0,
    "edge_min_weight": 0.30,

    # Temporal persistence. A real new hole remains in the same registered
    # position for several post-shot frames; random sensor noise usually does
    # not. This lets very weak but repeatable changes accumulate evidence.
    "persistence_enabled": True,
    "persistence_decay": 0.68,
    "persistence_weight": 4.0,
    "persistence_min_change": 0.65,
    "persistence_min_zscore": 0.65,
    "persistence_dilate_kernel": 3,

    # Machine-readable detector telemetry. Synthetic AI training automatically
    # contributes exact ground truth through a lazy hook.
    "diagnostics_enabled": True,
    "diagnostics_match_radius_px": 42.0,

    # Cropped visual debug maps are useful while tuning but cost allocations and
    # percentiles every camera frame. Machine-readable shot diagnostics stay on.
    "debug_frames_enabled": False,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except Exception:
        pass
    return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _odd(value: int, minimum: int = 1) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        value = result.stdout.strip()
        return value or None
    except Exception:
        return None


@dataclass
class V2FrameResult:
    candidates: list[dict[str, float]]
    telemetry: dict[str, Any]


class DetectorV2Config:
    """Small JSON-backed configuration with cheap mtime-based hot reload."""

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = path
        self.values = dict(DEFAULT_CONFIG)
        self._last_check = 0.0
        self._last_mtime: float | None = None
        self.reload(force=True)

    def reload(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_check < 1.0:
            return
        self._last_check = now

        try:
            mtime = self.path.stat().st_mtime
        except Exception:
            mtime = None

        if not force and mtime == self._last_mtime:
            return

        self._last_mtime = mtime
        values = dict(DEFAULT_CONFIG)

        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    values.update(loaded)
            except Exception as exc:
                print(f"[DETECTOR-V2] Config load failed, defaults kept: {exc}")

        self.values = values

    def get(self, key: str, default: Any = None) -> Any:
        self.reload()
        return self.values.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        self.reload()
        return dict(self.values)


class CandidateGeneratorV2:
    """
    High-recall candidate generator for tiny persistent shot holes.

    Design goals:
      * preserve the legacy detector as a fallback/baseline
      * use the frames immediately before the audio event
      * estimate per-pixel temporal noise from a pre-shot stack
      * compensate small global camera movement before differencing
      * keep the signal continuous and find local maxima instead of requiring
        the hole to survive one hard global binary threshold + contour pipeline
      * use spatial candidate quotas so one noisy region cannot monopolise all
        candidate slots
      * keep artifact and edge knowledge as soft priors, not hard deletion

    The class does NOT emit hits itself. It only supplies ordinary HitScanner
    candidate dictionaries, so existing tracking, AI ranking and hit emission
    remain unchanged.
    """

    SCHEMA_VERSION = "2.0"

    def __init__(self) -> None:
        self.config = DetectorV2Config()
        self._diagnostics: dict[int, dict[str, Any]] = {}
        self._diagnostics_written: set[int] = set()
        self._persistence_states: dict[int, dict[str, Any]] = {}
        self._shot_models: dict[int, dict[str, Any]] = {}
        self._git_commit = _git_commit()
        self._runtime_session_id = (
            time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )

    # ------------------------------------------------------------------
    # Public integration API
    # ------------------------------------------------------------------

    def generate(
        self,
        scanner: Any,
        gray: np.ndarray,
        frame_ts: float,
        legacy_candidates: list[dict[str, float]],
    ) -> V2FrameResult:
        cfg = self.config.snapshot()

        if not bool(cfg.get("enabled", True)):
            return V2FrameResult(
                candidates=list(legacy_candidates),
                telemetry={"enabled": False},
            )

        roi_mask_full = scanner._frame_roi_mask(gray.shape)

        try:
            bx, by, bw, bh = cv2.boundingRect(roi_mask_full)
        except Exception:
            bx = by = bw = bh = 0

        if bw <= 0 or bh <= 0:
            return V2FrameResult(
                candidates=list(legacy_candidates),
                telemetry={"enabled": True, "reason": "empty_roi"},
            )

        x0 = max(0, int(bx) - 2)
        x1 = min(gray.shape[1], int(bx + bw) + 2)
        y0 = max(0, int(by) - 2)
        y1 = min(gray.shape[0], int(by + bh) + 2)

        if x1 - x0 < 8 or y1 - y0 < 8:
            return V2FrameResult(
                candidates=list(legacy_candidates),
                telemetry={"enabled": True, "reason": "roi_too_small"},
            )

        roi = roi_mask_full[y0:y1, x0:x1]
        current = gray[y0:y1, x0:x1]

        pending = [
            ev for ev in scanner.audio_events
            if getattr(ev, "state", "") == "pending"
        ]
        if not pending:
            return V2FrameResult(
                candidates=list(legacy_candidates),
                telemetry={"enabled": True, "reason": "no_pending_event"},
            )

        event = min(pending, key=lambda ev: float(getattr(ev, "peak_ts", frame_ts)))
        shot_id = int(getattr(event, "shot_id", 0) or 0)
        peak_ts = float(getattr(event, "peak_ts", frame_ts))

        # V1 has already run for this frame. V2 waits a few milliseconds before
        # accumulating its deliberately permissive/persistent evidence so a
        # pre-hole camera frame cannot seed a stable false V2 track.
        min_event_age_s = max(
            0.0, _safe_float(cfg.get("min_event_age_s", 0.015), 0.015)
        )
        event_age_s = float(frame_ts - peak_ts)
        if event_age_s < min_event_age_s:
            return V2FrameResult(
                candidates=list(legacy_candidates),
                telemetry={
                    "enabled": True,
                    "reason": "waiting_post_peak",
                    "shot_id": shot_id,
                    "event_age_s": event_age_s,
                    "min_event_age_s": min_event_age_s,
                },
            )

        blur_kernel = _odd(_safe_int(cfg.get("blur_kernel", 3), 3), 1)
        blur_sigma = max(0.0, _safe_float(cfg.get("blur_sigma", 0.55), 0.55))

        model = self._shot_models.get(shot_id)
        if (
            not isinstance(model, dict)
            or tuple(model.get("bbox", ())) != (x0, y0, x1, y1)
            or abs(_safe_float(model.get("peak_ts", 0.0)) - peak_ts) > 1e-6
        ):
            pre_frames = self._collect_pre_frames(
                scanner,
                peak_ts=peak_ts,
                bbox=(x0, y0, x1, y1),
                roi=roi,
                cfg=cfg,
            )

            if not pre_frames:
                fallback = scanner._build_pre_shot_background()
                if fallback is None or fallback.shape != gray.shape:
                    return V2FrameResult(
                        candidates=list(legacy_candidates),
                        telemetry={
                            "enabled": True,
                            "reason": "no_pre_reference",
                            "shot_id": shot_id,
                        },
                    )
                pre_frames = [fallback[y0:y1, x0:x1]]

            reference, temporal_noise, stack_stats = self._build_reference_and_noise(
                pre_frames,
                roi=roi,
                cfg=cfg,
            )

            if blur_kernel > 1:
                reference_work = cv2.GaussianBlur(
                    reference, (blur_kernel, blur_kernel), blur_sigma
                )
            else:
                reference_work = reference

            # OpenCV phaseCorrelate can report a tiny shape-dependent
            # non-zero offset even for reference-vs-reference. Measure that
            # bias once per shot and subtract it from all later registrations.
            _same_image, registration_bias_info = self._register_current(
                reference_work,
                reference_work,
                roi=roi,
                cfg=cfg,
                registration_bias=(0.0, 0.0),
            )
            registration_bias = (
                float(registration_bias_info.get("raw_dx", 0.0)),
                float(registration_bias_info.get("raw_dy", 0.0)),
            )

            cached_edge_weights = (
                self._edge_weights(reference_work, cfg)
                if bool(cfg.get("edge_penalty_enabled", True))
                else None
            )

            model = {
                "bbox": (x0, y0, x1, y1),
                "peak_ts": peak_ts,
                "reference_work": reference_work,
                "temporal_noise": temporal_noise,
                "stack_stats": stack_stats,
                "pre_frame_count": len(pre_frames),
                "registration_bias": registration_bias,
                "edge_weights": cached_edge_weights,
            }
            self._shot_models[shot_id] = model
        else:
            reference_work = model["reference_work"]
            temporal_noise = model["temporal_noise"]
            stack_stats = model["stack_stats"]
            pre_frames = [None] * int(model.get("pre_frame_count", 1))

        current_u8 = current.astype(np.uint8, copy=False)

        # Very small blur only. A 5x5 Gaussian can be comparable to the whole
        # diameter of the synthetic signal.
        if blur_kernel > 1:
            current_work = cv2.GaussianBlur(
                current_u8, (blur_kernel, blur_kernel), blur_sigma
            )
        else:
            current_work = current_u8

        current_aligned, registration = self._register_current(
            reference_work,
            current_work,
            roi=roi,
            cfg=cfg,
            registration_bias=tuple(model.get("registration_bias", (0.0, 0.0))),
        )

        # V2 keeps the PRE-shot/calibration coordinate system canonical and
        # warps the current frame back onto it. That improves both temporal
        # persistence and compatibility with the existing homography/ground
        # truth, which are expressed in calibrated camera coordinates.
        current_norm, exposure_offset = self._normalise_photometry(
            reference_work,
            current_aligned,
            roi=roi,
        )

        ref_f = reference_work.astype(np.float32)
        cur_f = current_norm.astype(np.float32)

        absdiff = np.abs(ref_f - cur_f)
        darkening = np.maximum(ref_f - cur_f, 0.0)

        noise_floor = max(0.25, _safe_float(cfg.get("noise_floor", 1.35), 1.35))
        noise_cap = max(noise_floor, _safe_float(cfg.get("noise_cap", 28.0), 28.0))
        noise = np.clip(temporal_noise.astype(np.float32), noise_floor, noise_cap)
        zscore = absdiff / noise

        dog = self._multiscale_change_response(absdiff, cfg)

        saliency = (
            _safe_float(cfg.get("weight_zscore", 5.8), 5.8) * zscore
            + _safe_float(cfg.get("weight_absdiff", 0.72), 0.72) * absdiff
            + _safe_float(cfg.get("weight_dog", 1.75), 1.75) * dog
            + _safe_float(cfg.get("weight_darkening", 0.32), 0.32) * darkening
        ).astype(np.float32)

        # Suppress pixels outside the actual perspective ROI. A small inward
        # margin also removes warp-border residuals after registration.
        roi_edge_margin = max(0, _safe_int(cfg.get("roi_edge_margin_px", 5), 5))
        if roi_edge_margin > 0:
            kernel_size = _odd(roi_edge_margin * 2 + 1, 3)
            valid_mask = cv2.erode(
                roi,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (kernel_size, kernel_size),
                ),
                iterations=1,
            )
            valid = valid_mask > 0
        else:
            valid = roi > 0

        saliency[~valid] = 0.0

        # Soft projector-artifact prior.
        artifact_weight_mean = 1.0
        artifact_mask = getattr(scanner, "artifact_suppression_mask", None)
        if artifact_mask is not None and artifact_mask.shape == gray.shape:
            artifact_crop = artifact_mask[y0:y1, x0:x1]
            inactive_weight = float(
                np.clip(
                    _safe_float(cfg.get("artifact_inactive_weight", 0.58), 0.58),
                    0.05,
                    1.0,
                )
            )
            weights = np.where(artifact_crop > 0, 1.0, inactive_weight).astype(np.float32)
            saliency *= weights
            if np.any(valid):
                artifact_weight_mean = float(np.mean(weights[valid]))

        # Soft pre-existing-edge penalty. Registration errors primarily light up
        # long existing scene edges; a new hole has strong temporal evidence but
        # was not an edge in the pre-shot frame.
        edge_weight_mean = 1.0
        if bool(cfg.get("edge_penalty_enabled", True)):
            edge_weights = model.get("edge_weights")
            if not isinstance(edge_weights, np.ndarray):
                edge_weights = self._edge_weights(reference_work, cfg)
            saliency *= edge_weights
            if np.any(valid):
                edge_weight_mean = float(np.mean(edge_weights[valid]))

        persistence_max = 0.0
        if bool(cfg.get("persistence_enabled", True)):
            persistence = self._update_persistence(
                shot_id=shot_id,
                bbox=(x0, y0, x1, y1),
                absdiff=absdiff,
                zscore=zscore,
                valid=valid,
                cfg=cfg,
            )
            persistence_weight = max(
                0.0,
                _safe_float(cfg.get("persistence_weight", 4.0), 4.0),
            )
            saliency += persistence_weight * persistence
            saliency[~valid] = 0.0
            if np.any(valid):
                persistence_max = float(np.max(persistence[valid]))

        threshold, robust_stats = self._robust_threshold(
            saliency,
            valid=valid,
            cfg=cfg,
        )

        v2_candidates = self._extract_candidates(
            scanner=scanner,
            saliency=saliency,
            absdiff=absdiff,
            darkening=darkening,
            dog=dog,
            zscore=zscore,
            valid=valid,
            bbox=(x0, y0, x1, y1),
            frame_ts=frame_ts,
            threshold=threshold,
            cfg=cfg,
        )

        merged = self._merge_hybrid(
            scanner=scanner,
            legacy=list(legacy_candidates),
            v2=v2_candidates,
            cfg=cfg,
        )

        telemetry: dict[str, Any] = {
            "enabled": True,
            "schema_version": self.SCHEMA_VERSION,
            "runtime_session_id": self._runtime_session_id,
            "shot_id": shot_id,
            "frame_ts": float(frame_ts),
            "peak_ts": peak_ts,
            "bbox": [x0, y0, x1, y1],
            "pre_stack_frames": len(pre_frames),
            "pre_stack_mode": stack_stats["mode"],
            "pre_stack_instability": stack_stats["instability"],
            "exposure_offset": exposure_offset,
            "registration": registration,
            "threshold": threshold,
            "saliency_median": robust_stats["median"],
            "saliency_mad": robust_stats["mad"],
            "legacy_count": len(legacy_candidates),
            "v2_count": len(v2_candidates),
            "merged_count": len(merged),
            "artifact_weight_mean": artifact_weight_mean,
            "edge_weight_mean": edge_weight_mean,
            "persistence_max": persistence_max,
            "max_absdiff": float(absdiff[valid].max()) if np.any(valid) else 0.0,
            "max_zscore": float(zscore[valid].max()) if np.any(valid) else 0.0,
            "max_saliency": float(saliency[valid].max()) if np.any(valid) else 0.0,
        }

        # Optional visual debug views. Machine-readable diagnostics remain
        # enabled independently. Keeping these off by default saves significant
        # per-frame allocations on HD cameras.
        if bool(cfg.get("debug_frames_enabled", False)):
            scanner.debug_frames["v2_absdiff_crop"] = np.clip(
                absdiff * 8.0, 0, 255
            ).astype(np.uint8)
            scanner.debug_frames["v2_zscore_crop"] = np.clip(
                zscore * 24.0, 0, 255
            ).astype(np.uint8)
            scanner.debug_frames["v2_saliency_crop"] = self._normalise_debug_map(
                saliency, valid
            )

            peaks_debug = np.zeros_like(roi, dtype=np.uint8)
            for candidate in v2_candidates:
                px = int(round(float(candidate["camera_x"]))) - x0
                py = int(round(float(candidate["camera_y"]))) - y0
                if 0 <= px < peaks_debug.shape[1] and 0 <= py < peaks_debug.shape[0]:
                    cv2.circle(peaks_debug, (px, py), 2, 255, 1)
            scanner.debug_frames["v2_peaks_crop"] = peaks_debug

        scanner.last_window_debug.update(
            {
                "v2_enabled": 1.0,
                "v2_bbox_x0": float(x0),
                "v2_bbox_y0": float(y0),
                "v2_bbox_x1": float(x1),
                "v2_bbox_y1": float(y1),
                "v2_pre_frames": float(len(pre_frames)),
                "v2_pre_instability": float(stack_stats["instability"]),
                "v2_registration_dx": float(registration["dx"]),
                "v2_registration_dy": float(registration["dy"]),
                "v2_registration_response": float(registration["response"]),
                "v2_threshold": float(threshold),
                "v2_candidates": float(len(v2_candidates)),
                "v2_legacy_candidates": float(len(legacy_candidates)),
                "v2_merged_candidates": float(len(merged)),
            }
        )

        self._update_diagnostics(
            scanner=scanner,
            event=event,
            telemetry=telemetry,
            legacy=legacy_candidates,
            v2=v2_candidates,
            merged=merged,
            bbox=(x0, y0, x1, y1),
            absdiff=absdiff,
            zscore=zscore,
            saliency=saliency,
        )

        return V2FrameResult(candidates=merged, telemetry=telemetry)

    def flush_resolved_shots(self, scanner: Any) -> None:
        if not bool(self.config.get("diagnostics_enabled", True)):
            return

        for event in list(getattr(scanner, "audio_events", [])):
            shot_id = int(getattr(event, "shot_id", 0) or 0)
            if shot_id <= 0 or shot_id in self._diagnostics_written:
                continue
            if str(getattr(event, "state", "pending")) == "pending":
                continue

            record = self._diagnostics.get(shot_id)
            if record is None:
                continue

            record["resolved"] = {
                "state": str(getattr(event, "state", "")),
                "emitted": bool(getattr(event, "emitted", False)),
                "confidence": _safe_float(getattr(event, "confidence", 0.0)),
                "note": str(getattr(event, "note", "")),
                "matched_track_id": getattr(event, "matched_track_id", None),
                "matched_hole_id": getattr(event, "matched_hole_id", None),
            }
            record["finished_at"] = time.time()

            self._write_diagnostic(record)
            self._diagnostics_written.add(shot_id)
            self._diagnostics.pop(shot_id, None)
            self._persistence_states.pop(shot_id, None)
            self._shot_models.pop(shot_id, None)

            gt = getattr(scanner, "_detector_v2_ground_truth", None)
            if isinstance(gt, dict) and int(gt.get("shot_id", -1)) == shot_id:
                scanner._detector_v2_ground_truth = None

    def reset_runtime_state(self) -> None:
        """Drop per-shot V2 caches when HitScanner is disabled/re-armed."""
        self._diagnostics.clear()
        self._diagnostics_written.clear()
        self._persistence_states.clear()
        self._shot_models.clear()

    # ------------------------------------------------------------------
    # Reference/noise
    # ------------------------------------------------------------------

    def _collect_pre_frames(
        self,
        scanner: Any,
        *,
        peak_ts: float,
        bbox: tuple[int, int, int, int],
        roi: np.ndarray,
        cfg: dict[str, Any],
    ) -> list[np.ndarray]:
        x0, y0, x1, y1 = bbox
        max_frames = max(1, _safe_int(cfg.get("pre_stack_frames", 3), 3))
        window = max(0.05, _safe_float(cfg.get("pre_stack_window_s", 0.32), 0.32))
        min_gap = max(0.0, _safe_float(cfg.get("pre_stack_min_gap_s", 0.006), 0.006))

        newest_allowed = peak_ts - min_gap
        oldest_allowed = peak_ts - window

        selected: list[np.ndarray] = []

        for frame in reversed(list(getattr(scanner, "frame_history", []))):
            ts = float(getattr(frame, "timestamp", 0.0))
            if ts > newest_allowed:
                continue
            if ts < oldest_allowed:
                break

            gray = getattr(frame, "gray", None)
            if not isinstance(gray, np.ndarray):
                continue
            if gray.shape[0] < y1 or gray.shape[1] < x1:
                continue

            selected.append(gray[y0:y1, x0:x1])
            if len(selected) >= max_frames:
                break

        # Return chronological order. It is not required mathematically, but it
        # makes "latest" explicit and diagnostics easier to reason about.
        selected.reverse()
        return selected

    def _build_reference_and_noise(
        self,
        frames: list[np.ndarray],
        *,
        roi: np.ndarray,
        cfg: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if len(frames) == 1:
            reference = frames[-1].astype(np.uint8, copy=True)
            noise = np.full(reference.shape, float(cfg.get("noise_floor", 1.35)), dtype=np.float32)
            return reference, noise, {"mode": "single_latest", "instability": 0.0}

        # The latest three frames are enough to estimate a robust pre-shot
        # centre and noise range, and an exact 3-sample median can be computed
        # without NumPy's comparatively expensive per-pixel partition().
        selected = frames[-3:] if len(frames) >= 3 else frames

        if len(selected) >= 3:
            # Exact median of three uint8 frames without promoting three full
            # images to uint16. This is substantially faster on HD camera ROIs.
            a = selected[-3].astype(np.uint8, copy=False)
            b = selected[-2].astype(np.uint8, copy=False)
            c = selected[-1].astype(np.uint8, copy=False)

            ab_min = np.minimum(a, b)
            ab_max = np.maximum(a, b)
            low = np.minimum(ab_min, c)
            high = np.maximum(ab_max, c)
            median = np.maximum(ab_min, np.minimum(ab_max, c))
            temporal_range = cv2.absdiff(high, low).astype(np.float32)

            valid = roi > 0
            instability = 0.0
            sample_mask = valid[::4, ::4]
            if np.any(sample_mask):
                sampled_range = temporal_range[::4, ::4]
                instability = float(np.median(sampled_range[sample_mask])) * 0.5

            # Half the 3-frame range is a cheap conservative noise estimate.
            # The configured noise floor later protects perfectly static pixels.
            noise = temporal_range * 0.5

            stable_limit = _safe_float(cfg.get("stable_stack_max_mad", 4.5), 4.5)
            if instability <= stable_limit:
                reference = median
                mode = "median3"
            else:
                reference = selected[-1].astype(np.uint8, copy=True)
                mode = "latest_unstable"
        else:
            reference = selected[-1].astype(np.uint8, copy=True)
            if len(selected) == 2:
                noise = cv2.absdiff(selected[-2], selected[-1]).astype(np.float32)
                valid = roi > 0
                instability = (
                    float(np.median(noise[valid])) * 0.5
                    if np.any(valid)
                    else 0.0
                )
                mode = "latest_two_frame_noise"
            else:
                noise = np.full(
                    reference.shape,
                    float(cfg.get("noise_floor", 1.35)),
                    dtype=np.float32,
                )
                instability = 0.0
                mode = "single_latest"

        return reference, noise.astype(np.float32), {
            "mode": mode,
            "instability": round(instability, 4),
        }

    # ------------------------------------------------------------------
    # Registration / photometry
    # ------------------------------------------------------------------

    def _register_current(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        *,
        roi: np.ndarray,
        cfg: dict[str, Any],
        registration_bias: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[np.ndarray, dict[str, Any]]:
        result = {
            "enabled": bool(cfg.get("registration_enabled", True)),
            "applied": False,
            "dx": 0.0,
            "dy": 0.0,
            "raw_dx": 0.0,
            "raw_dy": 0.0,
            "bias_dx": float(registration_bias[0]),
            "bias_dy": float(registration_bias[1]),
            "response": 0.0,
        }

        if not result["enabled"]:
            return current, result

        if reference.shape != current.shape or reference.size < 64:
            return current, result

        try:
            ref = reference.astype(np.float32)
            cur = current.astype(np.float32)
            valid = roi > 0

            # High-pass images make translation estimation depend on stable
            # texture/edges instead of uniform brightness.
            ref_hp = ref - cv2.GaussianBlur(ref, (0, 0), 6.0)
            cur_hp = cur - cv2.GaussianBlur(cur, (0, 0), 6.0)
            ref_hp[~valid] = 0.0
            cur_hp[~valid] = 0.0

            max_dim = max(64, _safe_int(cfg.get("registration_max_dimension", 720), 720))
            scale = min(1.0, float(max_dim) / max(ref_hp.shape))
            if scale < 1.0:
                size = (
                    max(16, int(round(ref_hp.shape[1] * scale))),
                    max(16, int(round(ref_hp.shape[0] * scale))),
                )
                ref_pc = cv2.resize(ref_hp, size, interpolation=cv2.INTER_AREA)
                cur_pc = cv2.resize(cur_hp, size, interpolation=cv2.INTER_AREA)
            else:
                ref_pc = ref_hp
                cur_pc = cur_hp

            window = cv2.createHanningWindow(
                (ref_pc.shape[1], ref_pc.shape[0]),
                cv2.CV_32F,
            )
            (dx_small, dy_small), response = cv2.phaseCorrelate(
                ref_pc,
                cur_pc,
                window,
            )

            raw_dx = float(dx_small / max(scale, 1e-9))
            raw_dy = float(dy_small / max(scale, 1e-9))
            dx = raw_dx - float(registration_bias[0])
            dy = raw_dy - float(registration_bias[1])
            response = float(response)

            result.update(
                {
                    "raw_dx": raw_dx,
                    "raw_dy": raw_dy,
                    "dx": dx,
                    "dy": dy,
                    "response": response,
                }
            )

            max_shift = max(0.0, _safe_float(cfg.get("registration_max_shift_px", 4.0), 4.0))
            min_shift = max(0.0, _safe_float(cfg.get("registration_min_shift_px", 0.35), 0.35))
            min_response = _safe_float(cfg.get("registration_min_response", 0.08), 0.08)
            shift_magnitude = math.hypot(dx, dy)

            if (
                math.isfinite(dx)
                and math.isfinite(dy)
                and math.isfinite(response)
                and response >= min_response
                and min_shift <= shift_magnitude <= max_shift
            ):
                # phaseCorrelate(reference, current) returns the displacement
                # of CURRENT relative to REFERENCE. Warp CURRENT by the inverse
                # shift so all V2 maps stay in the calibrated pre-shot camera
                # coordinate system.
                matrix = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
                aligned = cv2.warpAffine(
                    current,
                    matrix,
                    (current.shape[1], current.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                result["applied"] = True
                return aligned, result

        except Exception:
            pass

        return current, result

    @staticmethod
    def _normalise_photometry(
        reference: np.ndarray,
        current: np.ndarray,
        *,
        roi: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        valid = roi > 0

        if not np.any(valid):
            return current, 0.0

        ref_f = reference.astype(np.float32)
        cur_f = current.astype(np.float32)

        # Median is robust to a tiny new hole and local scene details.
        offset = float(np.median(ref_f[valid] - cur_f[valid]))
        offset = float(np.clip(offset, -15.0, 15.0))

        if abs(offset) < 0.05:
            return current, offset

        normalised = np.clip(cur_f + offset, 0, 255).astype(np.uint8)
        return normalised, offset

    # ------------------------------------------------------------------
    # Saliency / candidate extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _multiscale_change_response(
        absdiff: np.ndarray,
        cfg: dict[str, Any],
    ) -> np.ndarray:
        scales = cfg.get(
            "local_contrast_scales",
            [[3, 9], [5, 15], [9, 27]],
        )
        if not isinstance(scales, list) or not scales:
            scales = [[3, 9], [5, 15], [9, 27]]

        src = absdiff.astype(np.float32)
        best = np.zeros_like(src, dtype=np.float32)

        for pair in scales:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue

            small_k = _odd(_safe_int(pair[0], 3), 1)
            broad_k = _odd(_safe_int(pair[1], 9), small_k + 2)

            if broad_k <= small_k:
                broad_k = _odd(small_k + 4, small_k + 2)

            small = cv2.boxFilter(
                src,
                cv2.CV_32F,
                (small_k, small_k),
                normalize=True,
                borderType=cv2.BORDER_REFLECT,
            )
            broad = cv2.boxFilter(
                src,
                cv2.CV_32F,
                (broad_k, broad_k),
                normalize=True,
                borderType=cv2.BORDER_REFLECT,
            )
            best = np.maximum(best, small - broad)

        return np.maximum(best, 0.0)

    @staticmethod
    def _edge_weights(
        reference: np.ndarray,
        cfg: dict[str, Any],
    ) -> np.ndarray:
        ref_f = reference.astype(np.float32)
        gx = cv2.Sobel(ref_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ref_f, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gx, gy)

        start = max(1.0, _safe_float(cfg.get("edge_gradient_start", 24.0), 24.0))
        min_weight = float(
            np.clip(_safe_float(cfg.get("edge_min_weight", 0.30), 0.30), 0.1, 1.0)
        )

        # Smoothly approaches min_weight on strong pre-existing edges.
        excess = np.maximum(gradient - start, 0.0)
        t = np.clip(excess / (start * 2.5), 0.0, 1.0)
        return (1.0 - t * (1.0 - min_weight)).astype(np.float32)

    def _update_persistence(
        self,
        *,
        shot_id: int,
        bbox: tuple[int, int, int, int],
        absdiff: np.ndarray,
        zscore: np.ndarray,
        valid: np.ndarray,
        cfg: dict[str, Any],
    ) -> np.ndarray:
        if shot_id <= 0:
            return np.zeros(absdiff.shape, dtype=np.float32)

        min_change = max(
            0.0,
            _safe_float(cfg.get("persistence_min_change", 0.65), 0.65),
        )
        min_z = max(
            0.0,
            _safe_float(cfg.get("persistence_min_zscore", 0.65), 0.65),
        )
        decay = float(
            np.clip(
                _safe_float(cfg.get("persistence_decay", 0.68), 0.68),
                0.0,
                0.98,
            )
        )

        evidence = (
            valid
            & (absdiff >= min_change)
            & (zscore >= min_z)
        ).astype(np.float32)

        dilate_kernel = _odd(
            _safe_int(cfg.get("persistence_dilate_kernel", 3), 3),
            1,
        )
        if dilate_kernel > 1:
            evidence = cv2.dilate(
                evidence,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (dilate_kernel, dilate_kernel),
                ),
            )

        state = self._persistence_states.get(shot_id)
        if (
            not isinstance(state, dict)
            or tuple(state.get("bbox", ())) != tuple(bbox)
            or not isinstance(state.get("map"), np.ndarray)
            or state["map"].shape != evidence.shape
        ):
            persistent = evidence
        else:
            previous = state["map"].astype(np.float32, copy=False)
            persistent = previous * decay + evidence

        # Bound accumulation so an event left open for a long time cannot
        # dominate all other signals.
        persistent = np.clip(persistent, 0.0, 4.0).astype(np.float32)
        persistent[~valid] = 0.0

        self._persistence_states[shot_id] = {
            "bbox": tuple(bbox),
            "map": persistent,
            "updated_at": time.time(),
        }

        # "Persistence" should mean repeated evidence. The first observation at
        # a pixel only seeds the accumulator; it does not get a free saliency
        # bonus. From the second agreeing frame onward the excess over 1.0
        # boosts the candidate. This sharply reduces random one-frame peaks.
        return np.maximum(persistent - 1.0, 0.0).astype(np.float32)

    @staticmethod
    def _robust_threshold(
        saliency: np.ndarray,
        *,
        valid: np.ndarray,
        cfg: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        values = saliency[valid]
        if values.size == 0:
            return float(cfg.get("min_saliency", 9.0)), {"median": 0.0, "mad": 0.0}

        # Sampling avoids expensive quantile work on multi-megapixel ROIs.
        if values.size > 200_000:
            stride = max(1, values.size // 200_000)
            values = values[::stride]

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_sigma = max(0.0, _safe_float(cfg.get("robust_sigma", 3.2), 3.2))
        min_saliency = max(0.0, _safe_float(cfg.get("min_saliency", 10.0), 10.0))

        # 1.4826 converts MAD to a normal-distribution sigma estimate.
        threshold = max(min_saliency, median + robust_sigma * 1.4826 * mad)

        return float(threshold), {
            "median": round(median, 4),
            "mad": round(mad, 4),
        }

    def _extract_candidates(
        self,
        *,
        scanner: Any,
        saliency: np.ndarray,
        absdiff: np.ndarray,
        darkening: np.ndarray,
        dog: np.ndarray,
        zscore: np.ndarray,
        valid: np.ndarray,
        bbox: tuple[int, int, int, int],
        frame_ts: float,
        threshold: float,
        cfg: dict[str, Any],
    ) -> list[dict[str, float]]:
        x0, y0, _, _ = bbox

        local_kernel = _odd(_safe_int(cfg.get("local_max_kernel", 5), 5), 3)
        dilated = cv2.dilate(
            saliency,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (local_kernel, local_kernel)),
        )

        min_change = max(
            0.0,
            _safe_float(cfg.get("min_temporal_change", 1.8), 1.8),
        )
        min_z = max(0.0, _safe_float(cfg.get("min_zscore", 1.5), 1.5))
        strong_change = max(
            min_change,
            _safe_float(cfg.get("strong_temporal_change", 4.0), 4.0),
        )

        evidence = (absdiff >= strong_change) | (
            (absdiff >= min_change) & (zscore >= min_z)
        )

        peak_mask = (
            valid
            & evidence
            & (saliency >= threshold)
            & (saliency >= (dilated - 1e-6))
        )

        peak_u8 = peak_mask.astype(np.uint8)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            peak_u8,
            connectivity=8,
        )

        raw_peaks: list[tuple[float, int, int]] = []

        for label in range(1, num_labels):
            sx = int(stats[label, cv2.CC_STAT_LEFT])
            sy = int(stats[label, cv2.CC_STAT_TOP])
            sw = int(stats[label, cv2.CC_STAT_WIDTH])
            sh = int(stats[label, cv2.CC_STAT_HEIGHT])

            if sw <= 0 or sh <= 0:
                continue

            component = labels[sy:sy + sh, sx:sx + sw] == label
            if not np.any(component):
                continue

            local_saliency = saliency[sy:sy + sh, sx:sx + sw]
            masked = np.where(component, local_saliency, -np.inf)
            flat_index = int(np.argmax(masked))
            py, px = np.unravel_index(flat_index, masked.shape)
            px += sx
            py += sy

            score = float(saliency[py, px])
            raw_peaks.append((score, px, py))

        if not raw_peaks:
            return []

        # High score first.
        raw_peaks.sort(key=lambda item: item[0], reverse=True)

        # Spatial quota: keep local evidence all over the ROI, not only around
        # whichever projected texture happens to create the strongest residuals.
        cols = max(1, _safe_int(cfg.get("tile_columns", 8), 8))
        rows = max(1, _safe_int(cfg.get("tile_rows", 6), 6))
        per_tile = max(1, _safe_int(cfg.get("per_tile_candidates", 4), 4))
        extra_global = max(0, _safe_int(cfg.get("global_extra_candidates", 45), 45))
        max_candidates = max(1, _safe_int(cfg.get("max_v2_candidates", 170), 170))
        nms_radius = max(0.5, _safe_float(cfg.get("nms_radius_px", 5.0), 5.0))

        tile_w = max(1.0, saliency.shape[1] / float(cols))
        tile_h = max(1.0, saliency.shape[0] / float(rows))
        tile_counts: dict[tuple[int, int], int] = {}

        chosen: list[tuple[float, int, int]] = []

        def far_enough(px: int, py: int) -> bool:
            for _score, cx, cy in chosen:
                if math.hypot(float(px - cx), float(py - cy)) < nms_radius:
                    return False
            return True

        # Pass 1: spatial coverage.
        for peak in raw_peaks:
            score, px, py = peak
            tx = min(cols - 1, int(px / tile_w))
            ty = min(rows - 1, int(py / tile_h))
            key = (tx, ty)

            if tile_counts.get(key, 0) >= per_tile:
                continue
            if not far_enough(px, py):
                continue

            chosen.append(peak)
            tile_counts[key] = tile_counts.get(key, 0) + 1
            if len(chosen) >= max_candidates:
                break

        # Pass 2: global strongest extras.
        extras_added = 0
        if len(chosen) < max_candidates and extra_global > 0:
            chosen_xy = {(px, py) for _, px, py in chosen}
            for peak in raw_peaks:
                if extras_added >= extra_global or len(chosen) >= max_candidates:
                    break

                _score, px, py = peak
                if (px, py) in chosen_xy:
                    continue
                if not far_enough(px, py):
                    continue

                chosen.append(peak)
                chosen_xy.add((px, py))
                extras_added += 1

        chosen.sort(key=lambda item: item[0], reverse=True)

        candidates: list[dict[str, float]] = []

        for peak_saliency, px, py in chosen[:max_candidates]:
            features = self._candidate_features(
                px=px,
                py=py,
                saliency=saliency,
                absdiff=absdiff,
                darkening=darkening,
                dog=dog,
                zscore=zscore,
            )

            camera_x = float(px + x0)
            camera_y = float(py + y0)

            candidate: dict[str, float] = {
                "camera_x": camera_x,
                "camera_y": camera_y,
                "area": float(features["area"]),
                "radius": float(features["radius"]),
                "circularity": float(features["circularity"]),
                "score": float(features["score"]),
                "center_darkening": float(features["center_change"]),
                "local_contrast_gain": float(features["local_contrast"]),
                "blackhat_value": float(features["dog_value"]),
                "change_value": float(features["center_change"]),
                "pre_shot_change": float(features["center_change"]),
                "timestamp": float(frame_ts),

                # Extra machine-readable fields are intentionally additive.
                # Existing AI code uses .get() and ignores unknown keys.
                "detector_v2": 1.0,
                "detector_v1": 0.0,
                "v2_saliency": float(peak_saliency),
                "v2_zscore": float(features["zscore"]),
                "v2_absdiff": float(features["absdiff"]),
                "v2_darkening": float(features["darkening"]),
                "v2_dog": float(features["dog_value"]),
            }

            self._apply_known_hole_penalty(scanner, candidate)
            candidates.append(candidate)

        candidates.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
        return candidates

    @staticmethod
    def _candidate_features(
        *,
        px: int,
        py: int,
        saliency: np.ndarray,
        absdiff: np.ndarray,
        darkening: np.ndarray,
        dog: np.ndarray,
        zscore: np.ndarray,
    ) -> dict[str, float]:
        h, w = absdiff.shape
        r = 6
        x0 = max(0, px - r)
        x1 = min(w, px + r + 1)
        y0 = max(0, py - r)
        y1 = min(h, py + r + 1)

        patch_abs = absdiff[y0:y1, x0:x1]
        patch_sal = saliency[y0:y1, x0:x1]

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist_sq = (xx - px) ** 2 + (yy - py) ** 2
        center_mask = dist_sq <= 4.0       # radius 2
        ring_mask = (dist_sq >= 16.0) & (dist_sq <= 49.0)  # radius 4..7

        center_change = (
            float(np.mean(patch_abs[center_mask]))
            if np.any(center_mask)
            else float(absdiff[py, px])
        )
        ring_change = (
            float(np.mean(patch_abs[ring_mask]))
            if np.any(ring_mask)
            else 0.0
        )
        local_contrast = max(0.0, center_change - ring_change)

        peak_abs = float(absdiff[py, px])
        blob_threshold = max(1.0, peak_abs * 0.36)
        blob = (patch_abs >= blob_threshold).astype(np.uint8)

        # Only keep the connected blob containing the peak if possible.
        local_px = px - x0
        local_py = py - y0
        area = 1.0
        radius = 1.0
        circularity = 0.75

        try:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
            label = int(labels[local_py, local_px])
            if label > 0 and label < n:
                component = (labels == label).astype(np.uint8)
                area = max(1.0, float(stats[label, cv2.CC_STAT_AREA]))

                contours, _ = cv2.findContours(
                    component,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    perimeter = float(cv2.arcLength(contour, True))
                    (_cx, _cy), enclosing_radius = cv2.minEnclosingCircle(contour)
                    radius = max(0.8, float(enclosing_radius))

                    if perimeter > 1e-6:
                        circularity = float(
                            np.clip(
                                (4.0 * math.pi * area) / (perimeter * perimeter),
                                0.05,
                                1.0,
                            )
                        )
        except Exception:
            pass

        z = float(zscore[py, px])
        dog_value = float(dog[py, px])
        dark = float(darkening[py, px])
        peak_sal = float(saliency[py, px])

        # Convert the V2 saliency scale to the detector-score scale already used
        # by HitScanner tracks / AI runtime. Keep the score permissive but sane.
        score = (
            0.16 * peak_sal
            + 0.23 * center_change
            + 0.15 * local_contrast
            + 0.11 * dog_value
            + 0.06 * min(z, 25.0)
        )
        score = float(np.clip(score, 3.6, 35.0))

        return {
            "area": area,
            "radius": radius,
            "circularity": circularity,
            "score": score,
            "center_change": center_change,
            "local_contrast": local_contrast,
            "zscore": z,
            "absdiff": peak_abs,
            "darkening": dark,
            "dog_value": dog_value,
            "patch_saliency_max": float(np.max(patch_sal)) if patch_sal.size else peak_sal,
        }

    @staticmethod
    def _apply_known_hole_penalty(scanner: Any, candidate: dict[str, float]) -> None:
        try:
            near = scanner._is_near_known_hole(
                float(candidate["camera_x"]),
                float(candidate["camera_y"]),
            )
        except Exception:
            near = None

        if near is None:
            return

        _hole, dist = near
        duplicate = max(1.0, float(getattr(scanner, "duplicate_radius_px", 18.0)))

        if dist <= duplicate * 0.5:
            candidate["score"] *= 0.15
        elif dist <= duplicate:
            candidate["score"] *= 0.4
        elif dist <= duplicate * 1.5:
            candidate["score"] *= 0.7

        candidate["near_known_hole_dist"] = float(dist)

    def _merge_hybrid(
        self,
        *,
        scanner: Any,
        legacy: list[dict[str, float]],
        v2: list[dict[str, float]],
        cfg: dict[str, Any],
    ) -> list[dict[str, float]]:
        if not bool(cfg.get("hybrid_with_legacy", True)):
            return list(v2[: int(getattr(scanner, "candidate_limit", 200))])

        merge_radius = max(0.5, _safe_float(cfg.get("merge_radius_px", 5.5), 5.5))
        agreement_bonus = max(0.0, _safe_float(cfg.get("agreement_bonus", 1.5), 1.5))
        limit = max(1, int(getattr(scanner, "candidate_limit", 200)))
        reserved = min(
            limit,
            max(0, _safe_int(cfg.get("v2_reserved_slots", 110), 110)),
        )

        merged: list[dict[str, float]] = []

        # Legacy detector stays intact, but mark provenance.
        for item in legacy:
            candidate = dict(item)
            candidate["detector_v1"] = 1.0
            candidate.setdefault("detector_v2", 0.0)
            merged.append(candidate)

        for v2_candidate in v2:
            best_index = -1
            best_dist = float("inf")

            for index, existing in enumerate(merged):
                dist = math.hypot(
                    float(existing.get("camera_x", 0.0)) - float(v2_candidate["camera_x"]),
                    float(existing.get("camera_y", 0.0)) - float(v2_candidate["camera_y"]),
                )
                if dist <= merge_radius and dist < best_dist:
                    best_dist = dist
                    best_index = index

            if best_index < 0:
                merged.append(dict(v2_candidate))
                continue

            old = merged[best_index]
            old_score = float(old.get("score", 0.0))
            new_score = float(v2_candidate.get("score", 0.0))

            # Keep the geometry/features of whichever detector has the stronger
            # evidence, but retain provenance from both.
            if new_score > old_score:
                replacement = dict(v2_candidate)
                for key, value in old.items():
                    replacement.setdefault(key, value)
                combined = replacement
            else:
                combined = dict(old)
                for key, value in v2_candidate.items():
                    combined.setdefault(key, value)

            combined["detector_v1"] = 1.0
            combined["detector_v2"] = 1.0
            combined["detector_agreement"] = 1.0
            combined["detector_agreement_distance"] = float(best_dist)
            combined["score"] = max(old_score, new_score) + agreement_bonus
            merged[best_index] = combined

        merged.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)

        if reserved <= 0:
            return merged[:limit]

        v2_ranked = [
            c for c in merged
            if float(c.get("detector_v2", 0.0)) > 0.5
        ]

        selected: list[dict[str, float]] = []
        selected_ids: set[int] = set()

        for candidate in v2_ranked[:reserved]:
            selected.append(candidate)
            selected_ids.add(id(candidate))

        for candidate in merged:
            if len(selected) >= limit:
                break
            if id(candidate) in selected_ids:
                continue
            selected.append(candidate)

        # Preserve score ordering for downstream AI while retaining the V2
        # reservation guarantee.
        selected.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)
        return selected[:limit]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _update_diagnostics(
        self,
        *,
        scanner: Any,
        event: Any,
        telemetry: dict[str, Any],
        legacy: list[dict[str, float]],
        v2: list[dict[str, float]],
        merged: list[dict[str, float]],
        bbox: tuple[int, int, int, int],
        absdiff: np.ndarray,
        zscore: np.ndarray,
        saliency: np.ndarray,
    ) -> None:
        if not bool(self.config.get("diagnostics_enabled", True)):
            return

        shot_id = int(getattr(event, "shot_id", 0) or 0)
        if shot_id <= 0:
            return

        record = self._diagnostics.setdefault(
            shot_id,
            {
                "schema_version": self.SCHEMA_VERSION,
                "runtime_session_id": self._runtime_session_id,
                "shot_id": shot_id,
                "created_at": time.time(),
                "git_commit": self._git_commit,
                "frames_seen": 0,
                "max_counts": {"legacy": 0, "v2": 0, "merged": 0},
                "signal_max": {
                    "absdiff": 0.0,
                    "zscore": 0.0,
                    "saliency": 0.0,
                },
                "registration": {
                    "applied_frames": 0,
                    "best_response": 0.0,
                    "max_abs_dx": 0.0,
                    "max_abs_dy": 0.0,
                },
                "ground_truth": None,
                "nearest_candidate_distance_px": {
                    "legacy": None,
                    "v2": None,
                    "merged": None,
                },
                "gt_signal_max": {
                    "absdiff": None,
                    "zscore": None,
                    "saliency": None,
                },
            },
        )

        record["frames_seen"] = int(record.get("frames_seen", 0)) + 1
        counts = record["max_counts"]
        counts["legacy"] = max(int(counts["legacy"]), len(legacy))
        counts["v2"] = max(int(counts["v2"]), len(v2))
        counts["merged"] = max(int(counts["merged"]), len(merged))

        signal_max = record["signal_max"]
        signal_max["absdiff"] = max(float(signal_max["absdiff"]), float(telemetry.get("max_absdiff", 0.0)))
        signal_max["zscore"] = max(float(signal_max["zscore"]), float(telemetry.get("max_zscore", 0.0)))
        signal_max["saliency"] = max(float(signal_max["saliency"]), float(telemetry.get("max_saliency", 0.0)))

        reg = telemetry.get("registration", {})
        reg_record = record["registration"]
        if bool(reg.get("applied", False)):
            reg_record["applied_frames"] = int(reg_record["applied_frames"]) + 1
        reg_record["best_response"] = max(
            float(reg_record["best_response"]),
            _safe_float(reg.get("response", 0.0)),
        )
        reg_record["max_abs_dx"] = max(
            float(reg_record["max_abs_dx"]),
            abs(_safe_float(reg.get("dx", 0.0))),
        )
        reg_record["max_abs_dy"] = max(
            float(reg_record["max_abs_dy"]),
            abs(_safe_float(reg.get("dy", 0.0))),
        )

        gt = getattr(scanner, "_detector_v2_ground_truth", None)
        if not isinstance(gt, dict) or int(gt.get("shot_id", -1)) != shot_id:
            return

        record["ground_truth"] = dict(gt)

        gt_x = _safe_float(gt.get("camera_x", -1.0), -1.0)
        gt_y = _safe_float(gt.get("camera_y", -1.0), -1.0)

        def nearest(candidates: list[dict[str, float]]) -> float | None:
            if not candidates:
                return None
            return min(
                math.hypot(
                    float(candidate.get("camera_x", 0.0)) - gt_x,
                    float(candidate.get("camera_y", 0.0)) - gt_y,
                )
                for candidate in candidates
            )

        distances = record["nearest_candidate_distance_px"]
        for key, candidates in (
            ("legacy", legacy),
            ("v2", v2),
            ("merged", merged),
        ):
            value = nearest(candidates)
            if value is None:
                continue
            previous = distances.get(key)
            distances[key] = value if previous is None else min(float(previous), value)

        x0, y0, x1, y1 = bbox
        local_x = int(round(gt_x)) - x0
        local_y = int(round(gt_y)) - y0

        if 0 <= local_x < absdiff.shape[1] and 0 <= local_y < absdiff.shape[0]:
            gx0 = max(0, local_x - 2)
            gx1 = min(absdiff.shape[1], local_x + 3)
            gy0 = max(0, local_y - 2)
            gy1 = min(absdiff.shape[0], local_y + 3)

            gt_signals = record["gt_signal_max"]

            for key, array in (
                ("absdiff", absdiff),
                ("zscore", zscore),
                ("saliency", saliency),
            ):
                patch = array[gy0:gy1, gx0:gx1]
                if patch.size == 0:
                    continue
                value = float(np.max(patch))
                previous = gt_signals.get(key)
                gt_signals[key] = value if previous is None else max(float(previous), value)

    def _write_diagnostic(self, record: dict[str, Any]) -> None:
        try:
            DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
            payload = dict(record)
            payload["detector_config"] = self.config.snapshot()

            gt = payload.get("ground_truth")
            if isinstance(gt, dict):
                match_radius = _safe_float(
                    self.config.get("diagnostics_match_radius_px", 42.0),
                    42.0,
                )
                nearest = payload.get("nearest_candidate_distance_px", {})
                payload["ground_truth_result"] = {
                    key: (
                        value is not None and float(value) <= match_radius
                    )
                    for key, value in nearest.items()
                }
                payload["ground_truth_result"]["match_radius_px"] = match_radius

            with DIAGNOSTICS_JSONL.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[DETECTOR-V2] Could not write diagnostic: {exc}")

    @staticmethod
    def _normalise_debug_map(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
        result = np.zeros(array.shape, dtype=np.uint8)
        values = array[valid]
        if values.size == 0:
            return result

        try:
            high = float(np.percentile(values, 99.5))
        except Exception:
            high = float(np.max(values))

        if high <= 1e-6:
            return result

        result[valid] = np.clip(array[valid] * (255.0 / high), 0, 255).astype(np.uint8)
        return result


# ----------------------------------------------------------------------
# Installation / compatibility wrapper
# ----------------------------------------------------------------------


def _install_ai_training_ground_truth_hook() -> None:
    """
    Lazily augment AITrainingScene once it has been imported.

    The normal training scene is otherwise untouched. The wrapper copies the
    synthetic-hole spec immediately before it is revealed, calls the original
    method, then records the exact screen+camera ground truth on HitScanner.
    This gives detector_v2 JSONL diagnostics real labels without coupling
    HitScanner itself to the training UI.
    """

    module = sys.modules.get("src.engine.scenes.ai_training")
    if module is None:
        return

    cls = getattr(module, "AITrainingScene", None)
    if cls is None:
        return

    if bool(getattr(cls, "_detector_v2_gt_hook_installed", False)):
        return

    original = getattr(cls, "_reveal_pending_synthetic_hole", None)
    if not callable(original):
        return

    def wrapped(self: Any) -> bool:
        spec_raw = getattr(self, "synthetic_pending_hole_spec", None)
        spec = dict(spec_raw) if isinstance(spec_raw, dict) else None

        result = original(self)

        if result and spec:
            try:
                from src.engine.ai.space_mapper import project_screen_point
                from src.engine.camera.hit_scanner import hit_scanner

                projected = project_screen_point(
                    float(spec["x"]),
                    float(spec["y"]),
                )

                pending = [
                    event
                    for event in hit_scanner.audio_events
                    if str(getattr(event, "state", "")) == "pending"
                ]
                if pending:
                    event = max(
                        pending,
                        key=lambda item: int(getattr(item, "shot_id", 0) or 0),
                    )

                    background = "unknown"
                    try:
                        background = self.MODE_NAMES[self.bg_mode_index]
                    except Exception:
                        pass

                    hit_scanner._detector_v2_ground_truth = {
                        "shot_id": int(getattr(event, "shot_id", 0) or 0),
                        "screen_x": float(spec["x"]),
                        "screen_y": float(spec["y"]),
                        "camera_x": float(projected.camera_x),
                        "camera_y": float(projected.camera_y),
                        "background": background,
                        "kind": str(spec.get("kind", "")),
                        "radius_px": _safe_float(spec.get("radius_px", 0.0)),
                        "strength": _safe_float(spec.get("strength", 0.0)),
                        "opacity": _safe_float(spec.get("opacity", 0.0)),
                        "recorded_at": time.time(),
                    }
            except Exception:
                # Diagnostics must never be able to break training.
                pass

        return result

    cls._reveal_pending_synthetic_hole = wrapped
    cls._detector_v2_gt_hook_installed = True


def install_candidate_generator_v2(scanner_cls: type) -> None:
    """
    Install V2 once on HitScanner.

    This is intentionally a class-level compatibility wrapper rather than a
    rewrite of the 1100+ line HitScanner. The original detector remains callable
    for every frame and is the automatic fallback if V2 ever raises.
    """

    if bool(getattr(scanner_cls, "_candidate_generator_v2_installed", False)):
        return

    original_detect: Callable[..., list[dict[str, float]]] = scanner_cls._detect_frame_candidates
    original_update: Callable[..., Any] = scanner_cls.update
    original_disable: Callable[..., Any] = scanner_cls.disable

    engine = CandidateGeneratorV2()

    def detect_wrapper(
        self: Any,
        gray: np.ndarray,
        frame_ts: float,
    ) -> list[dict[str, float]]:
        legacy = original_detect(self, gray, frame_ts)

        if not bool(engine.config.get("enabled", True)):
            return legacy

        try:
            result = engine.generate(
                scanner=self,
                gray=gray,
                frame_ts=frame_ts,
                legacy_candidates=legacy,
            )
            self.last_candidates = list(result.candidates)
            self.last_window_debug["v2_fallback"] = 0.0
            return self.last_candidates
        except Exception as exc:
            # Fail open to V1. This is a core safety invariant for the first V2
            # rollout: experimental recall improvements must not disable working
            # legacy detection.
            self.last_candidates = legacy
            self.last_window_debug["v2_fallback"] = 1.0
            self.last_window_debug["v2_error_hash"] = float(abs(hash(str(exc))) % 1_000_000)
            if bool(getattr(self, "shot_diag_enabled", False)):
                print(f"[DETECTOR-V2] fallback to legacy: {exc}")
            return legacy

    def update_wrapper(self: Any, dt: float) -> Any:
        # AITrainingScene is imported after the camera package. Install the
        # synthetic-ground-truth hook lazily once the class exists.
        _install_ai_training_ground_truth_hook()

        result = original_update(self, dt)

        try:
            engine.flush_resolved_shots(self)
        except Exception:
            pass

        return result

    def disable_wrapper(self: Any) -> Any:
        result = original_disable(self)
        engine.reset_runtime_state()
        return result

    scanner_cls._detect_frame_candidates = detect_wrapper
    scanner_cls.update = update_wrapper
    scanner_cls.disable = disable_wrapper
    scanner_cls._candidate_generator_v2_installed = True
    scanner_cls._candidate_generator_v2_engine = engine

    print("[DETECTOR-V2] Hybrid candidate generator installed")
