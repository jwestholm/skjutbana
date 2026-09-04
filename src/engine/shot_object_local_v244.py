"""V2.24.4 detector working-space HitRegion mapping.

Physical V2.24.3 testing proved that the object regions existed and transformed
successfully, but the object mask still became empty.  The reason is the
V2.22.1 analysis crop: HitRegions are canonical FULL-CAMERA coordinates while
HitScanner._frame_roi_mask() is called from inside the V2.22.1 crop-local
legacy detector.

V2.24.4 makes that coordinate boundary explicit:

    game -> screen -> full camera -> analysis/worker-local

The mapping uses the live V2.22.1 AnalysisGeometry when present.  It subtracts
the crop origin and also supports a scale factor if a future worker shape is not
exactly the crop size.  No hard-coded resolution or /2 scaling is used.

Hit authority is unchanged:
* regions only restrict WHERE the first physical pass searches;
* PRE->POST physical evidence / tracking / local confirmation still decide hits;
* V2.22.5 FULL-RESCUE bypasses the object mask and uses the original global ROI;
* no candidate is snapped to an object and roles never invent a hit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from src.engine.input.object_hit_v2223 import object_hit_registry_v2223
from src.engine.shot_fast_v2225 import rescue_router_v2225
from src.engine.shot_object_local_v241 import LocalSearchWindowV241, merge_camera_regions_v241

SCHEMA_VERSION = "2.24.4"
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
    return max(0.0, min(256.0, _finite(
        _runtime_settings().get("object_local_search_margin_px_v241", 36.0), 36.0
    )))


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


def _bounds_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@dataclass(frozen=True)
class WorkingSpaceMapV244:
    """Map canonical full-camera XY into the detector's current working image."""

    full_height: int
    full_width: int
    crop_x0: float
    crop_y0: float
    crop_width: float
    crop_height: float
    work_height: int
    work_width: int
    scale_x: float
    scale_y: float
    mode: str

    def camera_to_work(self, x: float, y: float) -> tuple[float, float]:
        return (
            (float(x) - float(self.crop_x0)) * float(self.scale_x),
            (float(y) - float(self.crop_y0)) * float(self.scale_y),
        )

    def camera_window_to_work(self, window: LocalSearchWindowV241) -> LocalSearchWindowV241:
        x0, y0 = self.camera_to_work(window.x0, window.y0)
        x1, y1 = self.camera_to_work(window.x1, window.y1)
        return LocalSearchWindowV241(
            min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
            tuple(window.object_ids), tuple(window.roles),
        )


def _working_space_map(scanner: Any, shape: Sequence[int]) -> WorkingSpaceMapV244:
    """Resolve the live V2.22.1 crop-local plane, with a safe direct fallback."""
    h, w = int(shape[0]), int(shape[1])
    geometry = getattr(scanner, "_v2221_active_geometry", None)

    required = (
        "frame_height", "frame_width", "crop_x0", "crop_y0",
        "crop_width", "crop_height",
    )
    if geometry is not None and all(hasattr(geometry, name) for name in required):
        full_h = max(1, int(getattr(geometry, "frame_height")))
        full_w = max(1, int(getattr(geometry, "frame_width")))
        crop_x0 = float(getattr(geometry, "crop_x0"))
        crop_y0 = float(getattr(geometry, "crop_y0"))
        crop_w = max(1.0, float(getattr(geometry, "crop_width")))
        crop_h = max(1.0, float(getattr(geometry, "crop_height")))
        # Normally work==crop in V2.22.1.  Keeping the ratio explicit makes the
        # contract safe if a later worker downsamples the crop.
        scale_x = float(w) / crop_w
        scale_y = float(h) / crop_h
        mode = str(getattr(geometry, "mode", "analysis_crop") or "analysis_crop")
        return WorkingSpaceMapV244(
            full_height=full_h,
            full_width=full_w,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            crop_width=crop_w,
            crop_height=crop_h,
            work_height=h,
            work_width=w,
            scale_x=scale_x,
            scale_y=scale_y,
            mode=mode,
        )

    # No analysis geometry means _frame_roi_mask is operating in the canonical
    # full-camera plane, so the transform is identity.
    return WorkingSpaceMapV244(
        full_height=h,
        full_width=w,
        crop_x0=0.0,
        crop_y0=0.0,
        crop_width=float(w),
        crop_height=float(h),
        work_height=h,
        work_width=w,
        scale_x=1.0,
        scale_y=1.0,
        mode="full_camera_identity",
    )


def map_camera_windows_to_work_v244(
    scanner: Any,
    shape: Sequence[int],
    windows: Iterable[LocalSearchWindowV241],
) -> tuple[WorkingSpaceMapV244, tuple[LocalSearchWindowV241, ...]]:
    mapping = _working_space_map(scanner, shape)
    return mapping, tuple(mapping.camera_window_to_work(w) for w in windows)


def _build_work_region_mask(
    shape: Sequence[int], windows: Iterable[LocalSearchWindowV241]
) -> np.ndarray:
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


def _first_region_text(camera_regions: Sequence[Any], mapping: WorkingSpaceMapV244) -> str:
    if not camera_regions:
        return "first=none"
    region = camera_regions[0]
    wid = str(getattr(region, "object_id", "?") or "?")
    try:
        cam = (
            float(region.x), float(region.y),
            float(region.x + region.width), float(region.y + region.height),
        )
        wx0, wy0 = mapping.camera_to_work(cam[0], cam[1])
        wx1, wy1 = mapping.camera_to_work(cam[2], cam[3])
        return (
            f"first={wid} camera=({cam[0]:.0f},{cam[1]:.0f},{cam[2]:.0f},{cam[3]:.0f}) "
            f"work=({wx0:.0f},{wy0:.0f},{wx1:.0f},{wy1:.0f})"
        )
    except Exception:
        return f"first={wid}"


def _install_frame_roi_patch() -> None:
    from src.engine.camera.hit_scanner import HitScanner

    if getattr(HitScanner, "_v244_working_space_roi_patch", False):
        return

    # V2.24.3 stored the ROI function that existed immediately before its own
    # wrapper.  That is exactly what we need: it preserves V2.22.1's safe
    # crop-local polygon while removing only the broken V2.24.3 camera/local mix.
    base_roi = getattr(HitScanner, "_v243_previous_frame_roi_mask", None)
    if not callable(base_roi):
        base_roi = HitScanner._frame_roi_mask

    previous_detect = HitScanner._detect_frame_candidates

    def frame_roi_v244(self, shape: tuple[int, int]) -> np.ndarray:
        global_mask = np.asarray(base_roi(self, shape), dtype=np.uint8)
        sid = _shot_id_from_scanner(self)
        mapping = _working_space_map(self, shape)
        diag: dict[str, Any] = {
            "shot_id": sid,
            "mode": "global",
            "regions": 0,
            "merged": 0,
            "global_pixels": int(np.count_nonzero(global_mask)),
            "region_pixels": 0,
            "overlap_pixels": 0,
            "selected_pixels": int(np.count_nonzero(global_mask)),
            "mapping_mode": mapping.mode,
            "full_shape": (mapping.full_height, mapping.full_width),
            "work_shape": (mapping.work_height, mapping.work_width),
            "crop": (
                mapping.crop_x0, mapping.crop_y0,
                mapping.crop_width, mapping.crop_height,
            ),
            "scale": (mapping.scale_x, mapping.scale_y),
        }
        self._v244_roi_diag = diag
        # Keep V2.24.3's detect telemetry wrapper coherent; V2.24.4 owns the
        # actual ROI now but older debug consumers may still inspect this field.
        self._v243_roi_diag = diag

        if not _enabled() or sid <= 0:
            return global_mask

        if rescue_router_v2225.requested(sid):
            diag["mode"] = "full_rescue_global"
            self._v244_roi_diag = diag
            self._v243_roi_diag = diag
            key = (sid, "full_rescue_global")
            if _log_enabled() and getattr(self, "_v244_last_roi_log", None) != key:
                print(
                    f"[V2.24.4 GLOBAL-RESCUE-ROI] shot={sid} "
                    f"work={mapping.work_width}x{mapping.work_height} "
                    f"global={_mask_fraction(global_mask) * 100.0:.1f}%"
                )
                self._v244_last_roi_log = key
            return global_mask

        snapshot = object_hit_registry_v2223.snapshot_for_shot(sid)
        camera_regions = tuple(getattr(snapshot, "camera_regions", ()) or ()) if snapshot is not None else ()
        if snapshot is None or not camera_regions:
            return global_mask

        max_regions = _max_regions()
        if len(camera_regions) > max_regions:
            diag["mode"] = "too_many_regions_global"
            return global_mask

        camera_windows = merge_camera_regions_v241(
            camera_regions,
            margin_px=_margin_px(),
            max_regions=max_regions,
        )
        if not camera_windows:
            diag["mode"] = "invalid_regions_global"
            return global_mask

        mapping, work_windows = map_camera_windows_to_work_v244(self, shape, camera_windows)
        region_mask = _build_work_region_mask(shape, work_windows)
        overlap = cv2.bitwise_and(global_mask, region_mask)
        region_pixels = int(np.count_nonzero(region_mask))
        overlap_pixels = int(np.count_nonzero(overlap))

        diag.update({
            "regions": len(camera_regions),
            "merged": len(camera_windows),
            "region_pixels": region_pixels,
            "overlap_pixels": overlap_pixels,
            "mapping_mode": mapping.mode,
            "full_shape": (mapping.full_height, mapping.full_width),
            "work_shape": (mapping.work_height, mapping.work_width),
            "crop": (mapping.crop_x0, mapping.crop_y0, mapping.crop_width, mapping.crop_height),
            "scale": (mapping.scale_x, mapping.scale_y),
            "first_region": _first_region_text(camera_regions, mapping),
        })

        if overlap_pixels > 0:
            selected = overlap
            mode = "intersect"
        elif region_pixels > 0:
            # Keep V2.24.3's explicit recovery semantics.  This is only a search
            # area, never hit authority, and FULL-RESCUE is still global.
            selected = region_mask
            mode = "region_recovery"
        else:
            selected = global_mask
            mode = "empty_region_global"

        diag["mode"] = mode
        diag["selected_pixels"] = int(np.count_nonzero(selected))
        diag["bounds"] = _bounds_from_mask(selected)
        self._v244_roi_diag = diag
        self._v243_roi_diag = diag

        try:
            self.debug_frames["v244_global_roi_work"] = global_mask.copy()
            self.debug_frames["v244_object_roi_work"] = region_mask.copy()
            self.debug_frames["roi_polygon"] = selected.copy()
        except Exception:
            pass

        key = (sid, mode)
        if _log_enabled() and getattr(self, "_v244_last_roi_log", None) != key:
            crop_x0, crop_y0, crop_w, crop_h = diag["crop"]
            scale_x, scale_y = diag["scale"]
            print(
                f"[V2.24.4 ROI-MAP] shot={sid} map={mapping.mode} "
                f"full={mapping.full_width}x{mapping.full_height} "
                f"crop=({crop_x0:.0f},{crop_y0:.0f},{crop_w:.0f},{crop_h:.0f}) "
                f"work={mapping.work_width}x{mapping.work_height} "
                f"scale=({scale_x:.4f},{scale_y:.4f}) {diag['first_region']}"
            )
            prefix = "[V2.24.4 ROI-RECOVERY]" if mode == "region_recovery" else "[V2.24.4 LOCAL-ROI]"
            print(
                f"{prefix} shot={sid} regions={len(camera_regions)} merged={len(camera_windows)} "
                f"global={_mask_fraction(global_mask) * 100.0:.1f}% "
                f"region={_mask_fraction(region_mask) * 100.0:.1f}% "
                f"overlap={overlap_pixels} selected={_mask_fraction(selected) * 100.0:.1f}% "
                f"margin={_margin_px():.0f}px bounds={diag['bounds']}"
            )
            self._v244_last_roi_log = key
        return selected

    def detect_v244(self, gray: np.ndarray, frame_ts: float):
        result = previous_detect(self, gray, frame_ts)
        diag = getattr(self, "_v244_roi_diag", None)
        if isinstance(diag, dict):
            try:
                debug = getattr(self, "last_window_debug", None)
                if isinstance(debug, dict):
                    debug["v244_local_roi"] = 1.0 if diag.get("mode") in ("intersect", "region_recovery") else 0.0
                    debug["v244_roi_recovery"] = 1.0 if diag.get("mode") == "region_recovery" else 0.0
                    debug["v244_region_count"] = float(diag.get("regions", 0))
                    debug["v244_merged_count"] = float(diag.get("merged", 0))
                    debug["v244_global_pixels"] = float(diag.get("global_pixels", 0))
                    debug["v244_region_pixels"] = float(diag.get("region_pixels", 0))
                    debug["v244_overlap_pixels"] = float(diag.get("overlap_pixels", 0))
                    debug["v244_selected_pixels"] = float(diag.get("selected_pixels", 0))
                    crop = diag.get("crop", (0.0, 0.0, 0.0, 0.0))
                    scale = diag.get("scale", (1.0, 1.0))
                    debug["v244_crop_x0"] = float(crop[0])
                    debug["v244_crop_y0"] = float(crop[1])
                    debug["v244_work_scale_x"] = float(scale[0])
                    debug["v244_work_scale_y"] = float(scale[1])
            except Exception:
                pass
        return result

    HitScanner._frame_roi_mask = frame_roi_v244
    HitScanner._detect_frame_candidates = detect_v244
    HitScanner._v244_working_space_roi_patch = True
    HitScanner._v244_base_frame_roi_mask = base_roi
    HitScanner._v244_previous_detect = previous_detect


def install_v244_runtime(AppClass: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_frame_roi_patch()
    AppClass._v244_working_space_roi_patch = True
    _INSTALLED = True
    print(
        f"[V2.24.4] full-camera -> detector working-space HitRegion mapping installed "
        f"(margin={_margin_px():.0f}px, V2.22.1 crop-aware, global rescue preserved)"
    )


__all__ = [
    "SCHEMA_VERSION",
    "PATCH_REVISION",
    "WorkingSpaceMapV244",
    "map_camera_windows_to_work_v244",
    "install_v244_runtime",
]
