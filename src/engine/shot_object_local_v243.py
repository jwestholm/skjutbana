"""V2.24.3 local-ROI integration fix.

Physical V2.24.2 testing exposed two independent integration problems:

1) V2.24.1 restricted only CandidateGeneratorV2._extract_candidates.  Legacy/V1
   proposals and V2's early waiting_post_peak path can therefore bypass the
   object-local mask.
2) settings.load_content_rect() returns an absolute viewport copy when no
   explicit content_rect exists, although HitScanner/HitInput document and use
   content_rect as viewport-local.  That can apply viewport x/y twice and move
   the detector ROI away from the game regions.

V2.24.3 fixes both without changing hit authority:

* normalise the implicit content rect to (0, 0, viewport.w, viewport.h),
* refresh HitInput calibration immediately before the shot-time object snapshot,
* move object-region restriction to HitScanner._frame_roi_mask so V1, early V2
  and normal V2 all see the same local search area,
* preserve V2.22.5 FULL-RESCUE as a genuinely global pass,
* if the configured global ROI and calibrated object regions have zero overlap,
  search the calibrated object regions first (ROI recovery) rather than silently
  reverting to global; physical PRE->POST evidence is still mandatory.

No candidate is snapped to an object and object roles never create a hit.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from src.engine.input.object_hit_v2223 import object_hit_registry_v2223
from src.engine.shot_fast_v2225 import rescue_router_v2225
from src.engine.shot_object_local_v241 import merge_camera_regions_v241

SCHEMA_VERSION = "2.24.3"
PATCH_REVISION = "r1"
_INSTALLED = False


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _runtime_settings() -> dict[str, Any]:
    try:
        from src.engine.ai.runtime import get_ai_runtime
        settings = getattr(get_ai_runtime(), "settings", {})
        return settings if isinstance(settings, dict) else {}
    except Exception:
        return {}


def _margin_px() -> float:
    return max(0.0, min(256.0, _finite(_runtime_settings().get(
        "object_local_search_margin_px_v241", 36.0
    ), 36.0)))


def _max_regions() -> int:
    try:
        value = int(_runtime_settings().get("object_local_search_max_regions_v241", 256))
    except Exception:
        value = 256
    return max(1, min(2048, value))


def _enabled() -> bool:
    return bool(_runtime_settings().get("object_local_search_enabled_v241", True))


def _log_enabled() -> bool:
    return bool(_runtime_settings().get("object_local_search_log_v241", True))


def _shot_id_from_scanner(scanner: Any) -> int:
    pending = [
        ev for ev in list(getattr(scanner, "audio_events", []) or [])
        if str(getattr(ev, "state", "")) == "pending"
    ]
    if not pending:
        return 0
    event = min(pending, key=lambda ev: float(getattr(ev, "peak_ts", 0.0) or 0.0))
    return int(getattr(event, "shot_id", 0) or 0)


def _mask_fraction(mask: np.ndarray) -> float:
    if not isinstance(mask, np.ndarray) or mask.size <= 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def _build_region_mask(shape: tuple[int, int], windows) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    for region in windows:
        x0 = max(0, min(w, int(math.floor(float(region.x0)))))
        y0 = max(0, min(h, int(math.floor(float(region.y0)))))
        x1 = max(0, min(w, int(math.ceil(float(region.x1)))))
        y1 = max(0, min(h, int(math.ceil(float(region.y1)))))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def _bounds_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _install_content_rect_fix() -> None:
    """Fix the implicit content rect without rewriting the user's settings file."""
    import src.engine.settings as settings_module

    if getattr(settings_module, "_v243_content_rect_fix", False):
        return
    previous = settings_module.load_content_rect

    def load_content_rect_v243():
        try:
            data = settings_module._load_settings_dict()
            explicit = settings_module._rect_from_value(data.get("content_rect"))
        except Exception:
            explicit = None
        if explicit is not None:
            return previous()
        viewport = settings_module.load_viewport_rect()
        if viewport is None:
            return previous()
        # content_rect is viewport-local.  The implicit full-content rectangle
        # therefore starts at local (0, 0), not viewport.x/y.
        try:
            import pygame
            return pygame.Rect(0, 0, int(viewport.w), int(viewport.h))
        except Exception:
            rect = viewport.copy()
            rect.x = 0
            rect.y = 0
            return rect

    settings_module.load_content_rect = load_content_rect_v243
    settings_module._v243_content_rect_fix = True
    settings_module._v243_previous_load_content_rect = previous

    # These modules imported load_content_rect by name before runtime patches
    # are installed, so update their bound globals too.
    try:
        import src.engine.camera.hit_scanner as hit_scanner_module
        hit_scanner_module.load_content_rect = load_content_rect_v243
    except Exception:
        pass
    try:
        import src.engine.input.hit_input as hit_input_module
        hit_input_module.load_content_rect = load_content_rect_v243
    except Exception:
        pass


def _install_calibration_refresh() -> None:
    """Make shot-time game->camera transforms use the latest saved calibration."""
    from src.engine.input.object_hit_v2223 import ObjectHitRegistryV2223
    if getattr(ObjectHitRegistryV2223, "_v243_calibration_refresh", False):
        return
    previous = ObjectHitRegistryV2223.snapshot

    def snapshot_v243(self, shot_id: int, peak_ts: float, scene: Any = None):
        try:
            from src.engine.input.hit_input import hit_input
            hit_input.reload_calibration()
        except Exception:
            pass
        return previous(self, shot_id, peak_ts, scene)

    ObjectHitRegistryV2223.snapshot = snapshot_v243
    ObjectHitRegistryV2223._v243_calibration_refresh = True
    ObjectHitRegistryV2223._v243_previous_snapshot = previous


def _remove_v241_candidate_only_mask() -> None:
    """V2.24.3 supersedes the too-low CandidateGenerator-only V2.24.1 hook."""
    try:
        from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
    except Exception:
        return
    previous = getattr(CandidateGeneratorV2, "_v241_previous_extract", None)
    if getattr(CandidateGeneratorV2, "_v241_object_local_patch", False) and callable(previous):
        # _v241_previous_extract is the V2.22.5 wrapper because install order is
        # V2.22.5 -> V2.24.1.  Restoring it keeps FAST/FULL-RESCUE intact while
        # removing only the redundant candidate-level region mask.
        CandidateGeneratorV2._extract_candidates = previous
        CandidateGeneratorV2._v243_removed_v241_candidate_mask = True


def _install_frame_roi_patch() -> None:
    from src.engine.camera.hit_scanner import HitScanner
    if getattr(HitScanner, "_v243_object_roi_patch", False):
        return

    previous_roi = HitScanner._frame_roi_mask
    previous_detect = HitScanner._detect_frame_candidates

    def frame_roi_v243(self, shape: tuple[int, int]) -> np.ndarray:
        global_mask = np.asarray(previous_roi(self, shape), dtype=np.uint8)
        sid = _shot_id_from_scanner(self)
        diag = {
            "shot_id": sid,
            "mode": "global",
            "regions": 0,
            "merged": 0,
            "global_pixels": int(np.count_nonzero(global_mask)),
            "region_pixels": 0,
            "overlap_pixels": 0,
            "selected_pixels": int(np.count_nonzero(global_mask)),
        }
        self._v243_roi_diag = diag

        if not _enabled() or sid <= 0:
            return global_mask

        # A V2.22.5 rescue must be genuinely global.  It is requested before
        # the next detector frame, so this method sees the request and restores
        # the unmodified ROI before high-recall extraction runs.
        if rescue_router_v2225.requested(sid):
            diag["mode"] = "full_rescue_global"
            self._v243_roi_diag = diag
            key = (sid, "full_rescue_global")
            if _log_enabled() and getattr(self, "_v243_last_roi_log", None) != key:
                print(
                    f"[V2.24.3 GLOBAL-RESCUE-ROI] shot={sid} "
                    f"global={_mask_fraction(global_mask) * 100.0:.1f}%"
                )
                self._v243_last_roi_log = key
            return global_mask

        snapshot = object_hit_registry_v2223.snapshot_for_shot(sid)
        camera_regions = tuple(getattr(snapshot, "camera_regions", ()) or ()) if snapshot is not None else ()
        if snapshot is None or not camera_regions:
            return global_mask

        max_regions = _max_regions()
        if len(camera_regions) > max_regions:
            diag["mode"] = "too_many_regions_global"
            self._v243_roi_diag = diag
            return global_mask

        windows = merge_camera_regions_v241(
            camera_regions,
            margin_px=_margin_px(),
            max_regions=max_regions,
        )
        if not windows:
            diag["mode"] = "invalid_regions_global"
            self._v243_roi_diag = diag
            return global_mask

        region_mask = _build_region_mask(shape, windows)
        overlap = cv2.bitwise_and(global_mask, region_mask)
        region_pixels = int(np.count_nonzero(region_mask))
        overlap_pixels = int(np.count_nonzero(overlap))
        diag.update({
            "regions": len(camera_regions),
            "merged": len(windows),
            "region_pixels": region_pixels,
            "overlap_pixels": overlap_pixels,
        })

        if overlap_pixels > 0:
            selected = overlap
            mode = "intersect"
        elif region_pixels > 0:
            # The V2.24.2 physical run reached this condition repeatedly.  The
            # calibrated object AABBs are still useful, so search them instead
            # of silently giving the whole frame back to legacy ranking.  This
            # does not invent a hit: PRE->POST physical evidence remains required,
            # and V2.22.5 can still request one global full rescue afterwards.
            selected = region_mask
            mode = "region_recovery"
        else:
            selected = global_mask
            mode = "empty_region_global"

        diag["mode"] = mode
        diag["selected_pixels"] = int(np.count_nonzero(selected))
        diag["bounds"] = _bounds_from_mask(selected)
        self._v243_roi_diag = diag
        try:
            self.debug_frames["v243_global_roi"] = global_mask.copy()
            self.debug_frames["v243_object_roi"] = region_mask.copy()
            self.debug_frames["roi_polygon"] = selected.copy()
        except Exception:
            pass

        key = (sid, mode)
        if _log_enabled() and getattr(self, "_v243_last_roi_log", None) != key:
            bounds = diag.get("bounds")
            if mode == "region_recovery":
                prefix = "[V2.24.3 ROI-RECOVERY]"
                reason = " zero_overlap=1"
            else:
                prefix = "[V2.24.3 LOCAL-ROI]"
                reason = ""
            print(
                f"{prefix} shot={sid} regions={len(camera_regions)} merged={len(windows)} "
                f"global={_mask_fraction(global_mask) * 100.0:.1f}% "
                f"region={_mask_fraction(region_mask) * 100.0:.1f}% "
                f"overlap={overlap_pixels} selected={_mask_fraction(selected) * 100.0:.1f}% "
                f"margin={_margin_px():.0f}px bounds={bounds}{reason}"
            )
            self._v243_last_roi_log = key
        return selected

    def detect_v243(self, gray: np.ndarray, frame_ts: float):
        result = previous_detect(self, gray, frame_ts)
        diag = getattr(self, "_v243_roi_diag", None)
        if isinstance(diag, dict):
            try:
                debug = getattr(self, "last_window_debug", None)
                if isinstance(debug, dict):
                    debug["v243_local_roi"] = 1.0 if diag.get("mode") in ("intersect", "region_recovery") else 0.0
                    debug["v243_roi_recovery"] = 1.0 if diag.get("mode") == "region_recovery" else 0.0
                    debug["v243_region_count"] = float(diag.get("regions", 0))
                    debug["v243_merged_count"] = float(diag.get("merged", 0))
                    debug["v243_global_pixels"] = float(diag.get("global_pixels", 0))
                    debug["v243_region_pixels"] = float(diag.get("region_pixels", 0))
                    debug["v243_overlap_pixels"] = float(diag.get("overlap_pixels", 0))
                    debug["v243_selected_pixels"] = float(diag.get("selected_pixels", 0))
            except Exception:
                pass
        return result

    HitScanner._frame_roi_mask = frame_roi_v243
    HitScanner._detect_frame_candidates = detect_v243
    HitScanner._v243_object_roi_patch = True
    HitScanner._v243_previous_frame_roi_mask = previous_roi
    HitScanner._v243_previous_detect = previous_detect


def install_v243_runtime(AppClass: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_content_rect_fix()
    _install_calibration_refresh()
    _remove_v241_candidate_only_mask()
    _install_frame_roi_patch()
    AppClass._v243_object_roi_patch = True
    _INSTALLED = True
    print(
        f"[V2.24.3] HitScanner-level object ROI installed "
        f"(margin={_margin_px():.0f}px, implicit content_rect fixed, global rescue preserved)"
    )


__all__ = [
    "SCHEMA_VERSION",
    "PATCH_REVISION",
    "install_v243_runtime",
]
