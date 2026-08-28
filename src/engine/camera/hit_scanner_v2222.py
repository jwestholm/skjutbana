"""V2.22.2 fast-path cleanup patch for the live HitScanner.

Installed *after* V2.22.1 and *before* AIRuntime wraps HitScanner.

Changes are deliberately bounded and fail-open:
- hide the OS cursor only while a shot is being detected, then restore it for
  manual hit labelling; mask its pre-shot position from CV so disappearance is
  not mistaken for a hole;
- demote stale PRE-shot appearance and reject stale candidates at registered
  old holes while preserving genuine fresh re-hits;
- remove long horizontal candidate ridges in the calibrated screen plane;
- cap camera backlog ingestion during an open shot so a slow detector does not
  spend the next update converting dozens of obsolete 4K frames;
- expose concise timing / cleanup diagnostics.

Canonical coordinates stay full-camera XY.  No game/content coordinate path is
changed by this patch.
"""
from __future__ import annotations

import math
import time
from typing import Any, Sequence

import cv2
import numpy as np

from src.engine.camera.analysis_filters_v2222 import (
    apply_novelty_cleanup_v2222,
    project_camera_candidates_to_screen_v2222,
    suppress_horizontal_ridges_v2222,
)

SCHEMA_VERSION = "2.22.2"
_INSTALLED = False


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
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
    width = getattr(rect, "w", None)
    height = getattr(rect, "h", None)
    if width is None:
        width = getattr(rect, "width")
    if height is None:
        height = getattr(rect, "height")
    return (
        float(getattr(rect, "x")),
        float(getattr(rect, "y")),
        float(width),
        float(height),
    )


def _screen_rect_and_homographies() -> tuple[tuple[float, float, float, float] | None, np.ndarray | None, np.ndarray | None]:
    """Return content/playfield screen rect, camera->screen H and screen->camera H."""
    try:
        from src.engine.settings import load_camera_calibration, load_content_rect, load_viewport_rect

        calibration = load_camera_calibration() or {}
        H = calibration.get("homography")
        H_inv = calibration.get("inverse_homography")
        content = load_content_rect()
        viewport = load_viewport_rect()
        if content is None or viewport is None:
            return None, None, None
        vx, vy, _vw, _vh = _rect_xywh(viewport)
        cx, cy, cw, ch = _rect_xywh(content)
        screen_rect = (vx + cx, vy + cy, cw, ch)

        def norm(matrix):
            if matrix is None:
                return None
            arr = np.asarray(matrix, dtype=np.float32)
            return arr if arr.shape == (3, 3) and np.all(np.isfinite(arr)) else None

        return screen_rect, norm(H), norm(H_inv)
    except Exception:
        return None, None, None


def _select_backlog_frames(frames: Sequence[Any], limit: int) -> list[Any]:
    values = list(frames)
    n = len(values)
    k = max(1, int(limit))
    if n <= k:
        return values
    # Keep temporal coverage rather than simply the last k frames.  The newest
    # sample is always included and original_update still detects only on it.
    indices = np.linspace(0, n - 1, num=k, dtype=np.int32)
    unique: list[int] = []
    for idx in indices.tolist():
        if not unique or int(idx) != unique[-1]:
            unique.append(int(idx))
    if unique[-1] != n - 1:
        unique[-1] = n - 1
    return [values[i] for i in unique]


def _hide_cursor_for_shot(scanner: Any) -> None:
    settings = _runtime_settings()
    if not bool(settings.get("analysis_cursor_guard_v2222_enabled", True)):
        return
    try:
        import pygame

        visible = bool(pygame.mouse.get_visible())
        pos = tuple(int(v) for v in pygame.mouse.get_pos())
        scanner._v2222_cursor_prev_visible = visible
        scanner._v2222_cursor_screen_xy = pos
        scanner._v2222_cursor_guard_active = visible
        if visible:
            pygame.mouse.set_visible(False)
    except Exception:
        scanner._v2222_cursor_guard_active = False


def _restore_cursor(scanner: Any) -> None:
    if not bool(getattr(scanner, "_v2222_cursor_guard_active", False)):
        return
    try:
        import pygame

        pygame.mouse.set_visible(bool(getattr(scanner, "_v2222_cursor_prev_visible", True)))
    except Exception:
        pass
    scanner._v2222_cursor_guard_active = False


def _apply_cursor_mask(scanner: Any, mask: np.ndarray) -> np.ndarray:
    if not bool(getattr(scanner, "_v2222_cursor_guard_active", False)):
        return mask
    settings = _runtime_settings()
    guard = max(8.0, _finite(settings.get("analysis_cursor_guard_screen_px", 36.0), 36.0))
    pos = getattr(scanner, "_v2222_cursor_screen_xy", None)
    geometry = getattr(scanner, "_v2221_active_geometry", None)
    if pos is None or geometry is None:
        return mask
    _screen_rect, _H, H_inv = _screen_rect_and_homographies()
    if H_inv is None:
        return mask
    try:
        x, y = float(pos[0]), float(pos[1])
        g = float(guard)
        square = np.asarray(
            [[[x - g, y - g], [x + g, y - g], [x + g, y + g], [x - g, y + g]]],
            dtype=np.float32,
        )
        cam = cv2.perspectiveTransform(square.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        cam[:, 0] -= float(geometry.crop_x0)
        cam[:, 1] -= float(geometry.crop_y0)
        poly = np.round(cam).astype(np.int32)
        result = np.asarray(mask).copy()
        cv2.fillConvexPoly(result, poly, 0)
        return result
    except Exception:
        return mask


def install_v2222_hit_scanner_patch() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import src.engine.camera.hit_scanner as hs_module

    HitScanner = hs_module.HitScanner
    camera_manager = hs_module.camera_manager

    original_on_audio_peak = HitScanner._on_audio_peak
    original_disable = HitScanner.disable
    original_frame_roi_mask = HitScanner._frame_roi_mask
    original_detect = HitScanner._detect_frame_candidates
    original_resolve_audio_events = HitScanner._resolve_audio_events
    original_update = HitScanner.update

    def patched_on_audio_peak(self, ev):
        # Hide before the first post-shot camera frame is analysed. The cursor's
        # previous screen position is simultaneously excluded from the shot ROI,
        # so its disappearance cannot become a candidate.
        _hide_cursor_for_shot(self)
        return original_on_audio_peak(self, ev)

    def patched_disable(self):
        try:
            return original_disable(self)
        finally:
            _restore_cursor(self)

    def patched_frame_roi_mask(self, shape):
        mask = original_frame_roi_mask(self, shape)
        try:
            return _apply_cursor_mask(self, mask)
        except Exception:
            return mask

    def patched_detect(self, gray: np.ndarray, frame_ts: float):
        cleanup_start = time.perf_counter()
        candidates = list(original_detect(self, gray, frame_ts) or [])
        input_count = len(candidates)
        settings = _runtime_settings()

        novelty_stats = {"pre_shot_informative": False, "stale_known_removed": 0, "novelty_demoted": 0}
        if bool(settings.get("analysis_novelty_cleanup_v2222_enabled", True)) and candidates:
            candidates, novelty_stats = apply_novelty_cleanup_v2222(
                candidates,
                list(getattr(self, "known_holes", []) or []),
                duplicate_radius_px=_finite(getattr(self, "duplicate_radius_px", 18.0), 18.0),
                fresh_rehit_min=max(
                    1.0,
                    _finite(
                        settings.get(
                            "analysis_fresh_rehit_min_v2222",
                            max(5.0, _finite(getattr(self, "rehit_gain_required", 4.0), 4.0)),
                        ),
                        5.0,
                    ),
                ),
            )

        ridge_stats = {"ridge_removed": 0, "ridge_groups": 0, "ridge_preserved_fresh": 0}
        if bool(settings.get("analysis_horizontal_ridge_filter_v2222_enabled", True)) and candidates:
            screen_rect, H, _H_inv = _screen_rect_and_homographies()
            if screen_rect is not None and H is not None:
                try:
                    screen_points = project_camera_candidates_to_screen_v2222(candidates, H)
                    candidates, ridge_stats = suppress_horizontal_ridges_v2222(
                        candidates,
                        screen_points,
                        screen_rect_xywh=screen_rect,
                        band_px=max(3.0, _finite(settings.get("analysis_ridge_band_screen_px_v2222", 7.0), 7.0)),
                        min_count=max(5, _as_int(settings.get("analysis_ridge_min_candidates_v2222", 9), 9)),
                        min_span_fraction=max(
                            0.15,
                            min(0.90, _finite(settings.get("analysis_ridge_min_span_fraction_v2222", 0.35), 0.35)),
                        ),
                        fresh_preserve_min=max(
                            1.0,
                            _finite(settings.get("analysis_ridge_fresh_preserve_min_v2222", 6.0), 6.0),
                        ),
                        max_preserve_per_ridge=max(
                            1,
                            _as_int(settings.get("analysis_ridge_max_fresh_keep_v2222", 3), 3),
                        ),
                    )
                except Exception:
                    pass

        limit = max(1, _as_int(getattr(self, "candidate_limit", 200), 200))
        candidates.sort(key=lambda c: _finite(c.get("score", 0.0)), reverse=True)
        candidates = candidates[:limit]
        self.last_candidates = candidates

        stats = dict(getattr(self, "last_window_debug", {}) or {})
        stats.update({
            "v2222_schema": SCHEMA_VERSION,
            "v2222_candidates_input": float(input_count),
            "v2222_candidates_output": float(len(candidates)),
            "v2222_stale_known_removed": float(novelty_stats.get("stale_known_removed", 0)),
            "v2222_novelty_demoted": float(novelty_stats.get("novelty_demoted", 0)),
            "v2222_pre_shot_informative": 1.0 if novelty_stats.get("pre_shot_informative") else 0.0,
            "v2222_ridge_removed": float(ridge_stats.get("ridge_removed", 0)),
            "v2222_ridge_groups": float(ridge_stats.get("ridge_groups", 0)),
            "v2222_ridge_preserved_fresh": float(ridge_stats.get("ridge_preserved_fresh", 0)),
            "v2222_cleanup_ms": float((time.perf_counter() - cleanup_start) * 1000.0),
        })
        self.last_window_debug = stats

        if bool(settings.get("analysis_v2222_log", True)) and int(getattr(self, "_diag_frame_count", 0)) == 1:
            print(
                f"[V2.22.2 CLEAN] shot={int(getattr(self, '_diag_shot_id', 0) or 0)} "
                f"candidates={input_count}->{len(candidates)} "
                f"stale={int(stats['v2222_stale_known_removed'])} "
                f"demoted={int(stats['v2222_novelty_demoted'])} "
                f"ridges={int(stats['v2222_ridge_groups'])} "
                f"ridge_removed={int(stats['v2222_ridge_removed'])} "
                f"fresh_saved={int(stats['v2222_ridge_preserved_fresh'])} "
                f"cleanup={stats['v2222_cleanup_ms']:.2f}ms"
            )
        return candidates

    def patched_resolve_audio_events(self, now_ts: float):
        result = original_resolve_audio_events(self, now_ts)
        try:
            if not self._has_open_events():
                _restore_cursor(self)
        except Exception:
            pass
        return result

    def patched_update(self, dt: float):
        settings = _runtime_settings()
        max_ingest = max(0, _as_int(settings.get("analysis_ingest_max_frames_v2222", 3), 3))
        thin = False
        try:
            thin = bool(max_ingest > 0 and self.enabled and self._has_open_events())
        except Exception:
            thin = False

        batch_total = 0
        batch_used = 0
        dropped = 0
        original_pickup = None
        started = time.perf_counter()

        if thin:
            try:
                original_pickup = camera_manager.get_new_frames_since_last_pickup
                batch = list(original_pickup() or [])
                batch_total = len(batch)
                selected = _select_backlog_frames(batch, max_ingest) if batch else []
                batch_used = len(selected)
                dropped = max(0, batch_total - batch_used)
                used = False

                def one_shot_pickup():
                    nonlocal used
                    if used:
                        return []
                    used = True
                    return selected

                camera_manager.get_new_frames_since_last_pickup = one_shot_pickup
            except Exception:
                original_pickup = None

        try:
            result = original_update(self, dt)
        finally:
            if original_pickup is not None:
                try:
                    camera_manager.get_new_frames_since_last_pickup = original_pickup
                except Exception:
                    pass

        total_ms = (time.perf_counter() - started) * 1000.0
        try:
            stats = dict(getattr(self, "last_window_debug", {}) or {})
            stats.update({
                "v2222_update_total_ms": float(total_ms),
                "v2222_ingest_batch_total": float(batch_total),
                "v2222_ingest_batch_used": float(batch_used),
                "v2222_ingest_frames_dropped": float(dropped),
            })
            self.last_window_debug = stats
            if bool(settings.get("analysis_v2222_log", True)) and int(getattr(self, "_diag_frame_count", 0)) == 1:
                detector_ms = _finite(stats.get("v2221_detector_ms", 0.0), 0.0)
                cleanup_ms = _finite(stats.get("v2222_cleanup_ms", 0.0), 0.0)
                overhead_ms = max(0.0, total_ms - detector_ms - cleanup_ms)
                print(
                    f"[V2.22.2 FAST] shot={int(getattr(self, '_diag_shot_id', 0) or 0)} "
                    f"frames={batch_total}->{batch_used} drop={dropped} "
                    f"detector={detector_ms:.1f}ms cleanup={cleanup_ms:.1f}ms "
                    f"overhead={overhead_ms:.1f}ms update={total_ms:.1f}ms"
                )
        except Exception:
            pass
        return result

    HitScanner._on_audio_peak = patched_on_audio_peak
    HitScanner.disable = patched_disable
    HitScanner._frame_roi_mask = patched_frame_roi_mask
    HitScanner._detect_frame_candidates = patched_detect
    HitScanner._resolve_audio_events = patched_resolve_audio_events
    HitScanner.update = patched_update
    HitScanner._v2222_fast_cleanup_patch = True

    _INSTALLED = True
    print("[V2.22.2] cursor/novelty/ridge/backlog fast-path patch installed")


__all__ = ["SCHEMA_VERSION", "install_v2222_hit_scanner_patch"]
