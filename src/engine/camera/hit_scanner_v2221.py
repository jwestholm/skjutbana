"""V2.22.1 live HitScanner optimization patch.

Goals:
- preserve the canonical full-camera coordinate plane end-to-end;
- run expensive OpenCV detection only on the calibrated playfield crop;
- remove a narrow, perspective-aware playfield edge band before contours;
- keep existing tracking/known-hole/AI/HitInput behaviour intact;
- add timing/ROI diagnostics and correct per-event shot IDs in HIT/MISS logs.

This module intentionally wraps the existing detector instead of duplicating its
candidate scoring algorithm.  The existing ``HitScanner._detect_frame_candidates``
is executed on a crop with temporary crop-local references/masks.  Returned
candidate coordinates are translated back to full camera coordinates before the
rest of the engine sees them.
"""
from __future__ import annotations

from collections import deque
import time
from typing import Any

import cv2
import numpy as np

from src.engine.camera.analysis_geometry_v2221 import (
    AnalysisGeometryV2221,
    build_full_frame_geometry_v2221,
    build_perspective_geometry_v2221,
    build_rect_fallback_geometry_v2221,
)

SCHEMA_VERSION = "2.22.1"
_INSTALLED = False

DEFAULT_EDGE_GUARD_SCREEN_PX = 12.0
DEFAULT_CROP_PADDING_CAMERA_PX = 16
DEFAULT_FALLBACK_EDGE_GUARD_CAMERA_PX = 8


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _runtime_settings() -> dict[str, Any]:
    try:
        from src.engine.ai.runtime import get_ai_runtime

        runtime = get_ai_runtime()
        settings = getattr(runtime, "settings", {})
        return settings if isinstance(settings, dict) else {}
    except Exception:
        return {}


def _rect_xywh(rect: Any) -> tuple[float, float, float, float]:
    return (
        float(getattr(rect, "x")),
        float(getattr(rect, "y")),
        float(getattr(rect, "w", getattr(rect, "width"))),
        float(getattr(rect, "h", getattr(rect, "height"))),
    )


def _build_geometry_from_live_settings(scanner: Any, shape: tuple[int, int]) -> AnalysisGeometryV2221:
    """Resolve calibrated content/playfield geometry for one camera shape."""
    from src.engine.settings import (
        load_camera_calibration,
        load_content_rect,
        load_scanport_rect,
        load_viewport_rect,
    )

    settings = _runtime_settings()
    enabled = bool(settings.get("analysis_roi_crop_v2221_enabled", True))
    if not enabled:
        return build_full_frame_geometry_v2221(shape)

    guard_screen_px = max(
        0.0,
        _safe_float(
            settings.get("analysis_playfield_edge_guard_screen_px", DEFAULT_EDGE_GUARD_SCREEN_PX),
            DEFAULT_EDGE_GUARD_SCREEN_PX,
        ),
    )
    crop_padding = max(
        0,
        _safe_int(
            settings.get("analysis_crop_padding_camera_px", DEFAULT_CROP_PADDING_CAMERA_PX),
            DEFAULT_CROP_PADDING_CAMERA_PX,
        ),
    )
    fallback_guard = max(
        0,
        _safe_int(
            settings.get("analysis_fallback_edge_guard_camera_px", DEFAULT_FALLBACK_EDGE_GUARD_CAMERA_PX),
            DEFAULT_FALLBACK_EDGE_GUARD_CAMERA_PX,
        ),
    )

    calibration = load_camera_calibration() or {}
    raw_inv = calibration.get("inverse_homography")
    content = load_content_rect()
    viewport = load_viewport_rect()
    if raw_inv is not None and content is not None and viewport is not None:
        try:
            vx, vy, _vw, _vh = _rect_xywh(viewport)
            cx, cy, cw, ch = _rect_xywh(content)
            # Same semantics as the current HitScanner ROI code: content_rect is
            # viewport-local and must be offset into absolute screen coordinates.
            screen_rect = (vx + cx, vy + cy, cw, ch)
            return build_perspective_geometry_v2221(
                shape,
                raw_inv,
                screen_rect,
                guard_screen_px=guard_screen_px,
                crop_padding_camera_px=crop_padding,
            )
        except Exception as exc:
            scanner.last_window_debug = dict(getattr(scanner, "last_window_debug", {}) or {})
            scanner.last_window_debug["v2221_geometry_homography_failed"] = 1.0
            scanner.last_window_debug["v2221_geometry_error"] = str(exc)

    scanport = load_scanport_rect()
    if scanport is not None:
        try:
            return build_rect_fallback_geometry_v2221(
                shape,
                _rect_xywh(scanport),
                guard_camera_px=fallback_guard,
                crop_padding_camera_px=min(crop_padding, 16),
            )
        except Exception:
            pass

    return build_full_frame_geometry_v2221(shape)


def _shift_known_holes_to_local(
    holes: list[dict[str, float]], geometry: AnalysisGeometryV2221
) -> list[dict[str, float]]:
    shifted: list[dict[str, float]] = []
    for hole in holes:
        item = dict(hole)
        try:
            item["camera_x"] = float(item.get("camera_x", 0.0)) - float(geometry.crop_x0)
            item["camera_y"] = float(item.get("camera_y", 0.0)) - float(geometry.crop_y0)
        except Exception:
            pass
        shifted.append(item)
    return shifted


def install_v2221_hit_scanner_patch() -> None:
    """Install V2.22.1 once; safe to call repeatedly."""
    global _INSTALLED
    if _INSTALLED:
        return

    import src.engine.camera.hit_scanner as hs_module

    HitScanner = hs_module.HitScanner
    camera_manager = hs_module.camera_manager

    original_update = HitScanner.update
    original_detect = HitScanner._detect_frame_candidates
    original_frame_roi_mask = HitScanner._frame_roi_mask
    original_recent_background = HitScanner._build_recent_background
    original_pre_shot_background = HitScanner._build_pre_shot_background

    def patched_frame_roi_mask(self, shape):
        geometry = getattr(self, "_v2221_active_geometry", None)
        if isinstance(geometry, AnalysisGeometryV2221):
            if tuple(int(v) for v in shape[:2]) == tuple(geometry.safe_mask_local.shape[:2]):
                return geometry.safe_mask_local.copy()
        return original_frame_roi_mask(self, shape)

    def patched_recent_background(self, frame_ts: float):
        geometry = getattr(self, "_v2221_active_geometry", None)
        if not isinstance(geometry, AnalysisGeometryV2221):
            return original_recent_background(self, frame_ts)
        candidates: list[np.ndarray] = []
        for fr in reversed(self.frame_history):
            age = frame_ts - fr.timestamp
            if age < self.recent_bg_min_age_s:
                continue
            if age > self.recent_bg_max_age_s:
                break
            cropped = geometry.crop_array(fr.gray)
            if cropped is not None:
                candidates.append(cropped)
            if len(candidates) >= self.max_background_frames:
                break
        if len(candidates) < 3:
            return None
        return np.median(np.stack(candidates, axis=0), axis=0).astype(np.uint8)

    def patched_pre_shot_background(self):
        geometry = getattr(self, "_v2221_active_geometry", None)
        if not isinstance(geometry, AnalysisGeometryV2221):
            return original_pre_shot_background(self)

        earliest_peak_ts = None
        for ev in self.audio_events:
            if ev.state == "pending":
                if earliest_peak_ts is None or ev.peak_ts < earliest_peak_ts:
                    earliest_peak_ts = ev.peak_ts
        if earliest_peak_ts is None:
            return None

        # Prefer the camera ring, but crop BGR first so grayscale conversion is
        # performed only on the analysis ROI rather than on the whole 4K frame.
        ring = camera_manager.get_ring_snapshot()
        if ring and len(ring) >= 5:
            target_ts = earliest_peak_ts - 0.50
            best = min(ring, key=lambda cf: abs(float(cf.timestamp) - float(target_ts)))
            try:
                bgr_crop = best.frame_bgr[
                    geometry.crop_y0:geometry.crop_y1,
                    geometry.crop_x0:geometry.crop_x1,
                ]
                if bgr_crop.size:
                    return cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
            except Exception:
                pass

        cutoff_latest = earliest_peak_ts - 0.20
        cutoff_earliest = earliest_peak_ts - 0.50
        pre_frames: list[np.ndarray] = []
        for fr in reversed(self.frame_history):
            if fr.timestamp > cutoff_latest:
                continue
            if fr.timestamp < cutoff_earliest:
                break
            cropped = geometry.crop_array(fr.gray)
            if cropped is not None:
                pre_frames.append(cropped)
            if len(pre_frames) >= 3:
                break
        if pre_frames:
            if len(pre_frames) == 1:
                return pre_frames[0]
            return np.median(np.stack(pre_frames, axis=0), axis=0).astype(np.uint8)

        scene_ref = getattr(self, "scene_reference_gray", None)
        return geometry.crop_array(scene_ref)

    def patched_detect(self, gray: np.ndarray, frame_ts: float):
        total_start = time.perf_counter()
        geometry_start = time.perf_counter()
        try:
            geometry = _build_geometry_from_live_settings(self, gray.shape[:2])
        except Exception:
            geometry = build_full_frame_geometry_v2221(gray.shape[:2])
        geometry_ms = (time.perf_counter() - geometry_start) * 1000.0

        # Full-frame fallback preserves old behaviour exactly.
        if geometry.mode == "full_frame_fallback":
            detect_start = time.perf_counter()
            result = original_detect(self, gray, frame_ts)
            detect_ms = (time.perf_counter() - detect_start) * 1000.0
            self.last_window_debug.update({
                "v2221_schema": SCHEMA_VERSION,
                "v2221_geometry_mode": geometry.mode,
                "v2221_crop_fraction": 1.0,
                "v2221_geometry_ms": geometry_ms,
                "v2221_detector_ms": detect_ms,
                "v2221_detect_wrapper_ms": (time.perf_counter() - total_start) * 1000.0,
            })
            return result

        crop = geometry.crop_array(gray)
        if crop is None or crop.size == 0:
            return original_detect(self, gray, frame_ts)

        original_scene = self.scene_reference_gray
        original_surface = self.surface_reference_gray
        original_artifact = self.artifact_suppression_mask
        original_known_holes = self.known_holes
        self._v2221_active_geometry = geometry

        try:
            # Direct reference use inside the legacy detector must see arrays in
            # the same crop-local plane as ``gray``.
            self.scene_reference_gray = geometry.crop_array(original_scene)
            self.surface_reference_gray = geometry.crop_array(original_surface)
            self.artifact_suppression_mask = geometry.crop_array(original_artifact)
            self.known_holes = _shift_known_holes_to_local(original_known_holes, geometry)

            detect_start = time.perf_counter()
            local_candidates = original_detect(self, crop, frame_ts)
            detect_ms = (time.perf_counter() - detect_start) * 1000.0

            # Translate the canonical candidate plane back to full camera XY
            # *before* tracking, AI/resolver, known-hole state, and HitInput.
            for candidate in local_candidates:
                if "camera_x" in candidate:
                    candidate["camera_x"] = float(candidate["camera_x"]) + float(geometry.crop_x0)
                if "camera_y" in candidate:
                    candidate["camera_y"] = float(candidate["camera_y"]) + float(geometry.crop_y0)
                candidate["analysis_crop_x0"] = float(geometry.crop_x0)
                candidate["analysis_crop_y0"] = float(geometry.crop_y0)
                candidate["analysis_geometry_v2221"] = 1.0
            self.last_candidates = local_candidates
        finally:
            self.scene_reference_gray = original_scene
            self.surface_reference_gray = original_surface
            self.artifact_suppression_mask = original_artifact
            self.known_holes = original_known_holes
            self._v2221_active_geometry = None

        stats = dict(getattr(self, "last_window_debug", {}) or {})
        stats.update({
            "v2221_schema": SCHEMA_VERSION,
            "v2221_geometry_mode": geometry.mode,
            "v2221_frame_pixels": float(geometry.frame_pixels),
            "v2221_crop_pixels": float(geometry.crop_pixels),
            "v2221_crop_fraction": float(geometry.crop_fraction),
            "v2221_safe_pixels": float(geometry.safe_pixels),
            "v2221_edge_guard_screen_px": float(geometry.guard_screen_px),
            "v2221_crop_x0": float(geometry.crop_x0),
            "v2221_crop_y0": float(geometry.crop_y0),
            "v2221_crop_w": float(geometry.crop_width),
            "v2221_crop_h": float(geometry.crop_height),
            "v2221_geometry_ms": float(geometry_ms),
            "v2221_detector_ms": float(detect_ms),
            "v2221_detect_wrapper_ms": float((time.perf_counter() - total_start) * 1000.0),
        })
        self.last_window_debug = stats

        # Keep one lightweight full-frame geometry debug view. Heavy detector
        # maps remain crop-local, which is intentional and saves memory/copying.
        try:
            preview = np.zeros(gray.shape[:2], dtype=np.uint8)
            outer = np.round(geometry.outer_camera_polygon).astype(np.int32)
            safe = np.round(geometry.safe_camera_polygon).astype(np.int32)
            cv2.polylines(preview, [outer], True, 128, 2)
            cv2.polylines(preview, [safe], True, 255, 2)
            self.debug_frames["roi_polygon_full_v2221"] = preview
        except Exception:
            pass

        settings = _runtime_settings()
        should_log = bool(settings.get("analysis_v2221_log", True))
        # The legacy detector increments _diag_frame_count once per processed
        # frame.  Log the first analysed frame for each audio shot.
        if should_log and int(getattr(self, "_diag_frame_count", 0)) == 1:
            raw = int(float(stats.get("raw_blobs_total", 0.0) or 0.0))
            kept = int(float(stats.get("candidates_kept", len(local_candidates)) or 0.0))
            print(
                f"[V2.22.1 ROI] shot={int(getattr(self, '_diag_shot_id', 0) or 0)} "
                f"mode={geometry.mode} crop={100.0 * geometry.crop_fraction:.1f}% "
                f"guard={geometry.guard_screen_px:.1f}px raw={raw} kept={kept} "
                f"geometry={geometry_ms:.2f}ms detector={detect_ms:.2f}ms"
            )
        return local_candidates

    def patched_resolve_audio_events(self, now_ts: float) -> int:
        emitted_now = 0
        for event in list(self.audio_events):
            if event.state != "pending":
                continue
            track = self._best_track_for_event(event)
            if track is not None:
                event.matched_track_id = track.track_id
                event.confidence = max(event.confidence, float(track.best_score))
                detector_e2e_ms = max(0.0, (float(now_ts) - float(event.peak_ts)) * 1000.0)
                self.last_event_debug = {
                    "shot_id": float(event.shot_id),
                    "track_id": float(track.track_id),
                    "score": float(track.best_score),
                    "peak_ts": float(event.peak_ts),
                    "detector_e2e_ms": float(detector_e2e_ms),
                    "v2221_detector_ms": float(self.last_window_debug.get("v2221_detector_ms", 0.0) or 0.0),
                    "v2221_crop_fraction": float(self.last_window_debug.get("v2221_crop_fraction", 1.0) or 1.0),
                }
                if self._track_is_ready(track, now_ts, event):
                    print(
                        f"[SHOT #{event.shot_id}] HIT: ({track.camera_x:.0f},{track.camera_y:.0f}) "
                        f"score={track.best_score:.2f} hits={track.hits} "
                        f"candidates={len(self.last_candidates)} e2e={detector_e2e_ms:.1f}ms"
                    )
                    if self._emit_track_result(track, event):
                        emitted_now += 1
                    continue
            if now_ts - event.peak_ts >= self.event_timeout_s:
                event.state = "missed"
                event.note = "timeout"
                n_tracks = len(self._active_tracks)
                best_info = ""
                if n_tracks > 0:
                    best_t = max(self._active_tracks.values(), key=lambda t: t.best_score)
                    best_info = (
                        f" best=({best_t.camera_x:.0f},{best_t.camera_y:.0f}) "
                        f"score={best_t.best_score:.2f} hits={best_t.hits}"
                    )
                detector_e2e_ms = max(0.0, (float(now_ts) - float(event.peak_ts)) * 1000.0)
                self.last_event_debug = {
                    "shot_id": float(event.shot_id),
                    "peak_ts": float(event.peak_ts),
                    "detector_e2e_ms": float(detector_e2e_ms),
                    "state": "missed",
                }
                print(
                    f"[SHOT #{event.shot_id}] MISS: candidates={len(self.last_candidates)} "
                    f"tracks={n_tracks}{best_info} e2e={detector_e2e_ms:.1f}ms"
                )
        return emitted_now

    def patched_update(self, dt: float):
        started = time.perf_counter()
        result = original_update(self, dt)
        elapsed = (time.perf_counter() - started) * 1000.0
        try:
            self.last_window_debug["v2221_update_total_ms"] = float(elapsed)
        except Exception:
            pass
        return result

    HitScanner._frame_roi_mask = patched_frame_roi_mask
    HitScanner._build_recent_background = patched_recent_background
    HitScanner._build_pre_shot_background = patched_pre_shot_background
    HitScanner._detect_frame_candidates = patched_detect
    HitScanner._resolve_audio_events = patched_resolve_audio_events
    HitScanner.update = patched_update

    _INSTALLED = True
    print("[V2.22.1] Perspective ROI/edge-guard HitScanner patch installed")
