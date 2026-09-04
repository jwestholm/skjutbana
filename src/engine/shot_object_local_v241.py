"""V2.24.1 object-aware local physical proposal search.

HitRegions only define WHERE the first physical proposal pass searches.
Physical PRE->POST evidence and V2.22.5 local confirmation remain mandatory.
V2.22.5's one FULL-RESCUE pass is deliberately global and bypasses this mask.
No candidate is snapped to an object and object roles never create a hit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Any, Iterable, Sequence

import numpy as np

from src.engine.input.object_hit_v2223 import (
    CameraHitRegionV240,
    ObjectShotSnapshotV2223,
    object_hit_registry_v2223,
)
from src.engine.shot_fast_v2225 import rescue_router_v2225

SCHEMA_VERSION = "2.24.1"
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


def _setting_bool(name: str, default: bool) -> bool:
    return bool(_runtime_settings().get(name, default))


def _setting_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(_runtime_settings().get(name, default))
    except Exception:
        value = int(default)
    return max(lo, min(hi, value))


def _setting_float(name: str, default: float, lo: float, hi: float) -> float:
    value = _finite(_runtime_settings().get(name, default), default)
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class ObjectLocalConfigV241:
    enabled: bool = True
    margin_px: float = 36.0
    max_regions: int = 256
    log: bool = True


_CONFIG = ObjectLocalConfigV241()


def _load_config_from_runtime() -> None:
    global _CONFIG
    _CONFIG = ObjectLocalConfigV241(
        enabled=_setting_bool("object_local_search_enabled_v241", True),
        margin_px=_setting_float("object_local_search_margin_px_v241", 36.0, 0.0, 256.0),
        max_regions=_setting_int("object_local_search_max_regions_v241", 256, 1, 2048),
        log=_setting_bool("object_local_search_log_v241", True),
    )


@dataclass(frozen=True)
class LocalSearchWindowV241:
    x0: float
    y0: float
    x1: float
    y1: float
    object_ids: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    def overlaps(self, other: "LocalSearchWindowV241") -> bool:
        return not (
            self.x1 < other.x0 or other.x1 < self.x0
            or self.y1 < other.y0 or other.y1 < self.y0
        )

    def union(self, other: "LocalSearchWindowV241") -> "LocalSearchWindowV241":
        return LocalSearchWindowV241(
            min(self.x0, other.x0), min(self.y0, other.y0),
            max(self.x1, other.x1), max(self.y1, other.y1),
            tuple(sorted(set(self.object_ids) | set(other.object_ids))),
            tuple(sorted(set(self.roles) | set(other.roles))),
        )


@dataclass(frozen=True)
class LocalMaskDiagnosticsV241:
    region_count: int
    merged_count: int
    mask_pixels: int
    original_valid_pixels: int
    local_valid_pixels: int

    @property
    def valid_fraction(self) -> float:
        if self.original_valid_pixels <= 0:
            return 0.0
        return float(self.local_valid_pixels) / float(self.original_valid_pixels)


class LocalSearchTelemetryV241:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.local_passes = 0
            self.global_no_context = 0
            self.global_invalid_context = 0
            self.global_full_rescue = 0
            self.last_shot_id = 0
            self.last_reason = "none"
            self.last_diag: LocalMaskDiagnosticsV241 | None = None

    def record(self, shot_id: int, reason: str, diag: LocalMaskDiagnosticsV241 | None = None) -> None:
        with self._lock:
            self.last_shot_id = int(shot_id or 0)
            self.last_reason = str(reason)
            self.last_diag = diag
            if reason == "local":
                self.local_passes += 1
            elif reason == "full_rescue":
                self.global_full_rescue += 1
            elif reason == "no_context":
                self.global_no_context += 1
            else:
                self.global_invalid_context += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "local_passes": self.local_passes,
                "global_no_context": self.global_no_context,
                "global_invalid_context": self.global_invalid_context,
                "global_full_rescue": self.global_full_rescue,
                "last_shot_id": self.last_shot_id,
                "last_reason": self.last_reason,
                "last_diag": self.last_diag,
            }


local_search_telemetry_v241 = LocalSearchTelemetryV241()


def _shot_id_from_scanner(scanner: Any) -> int:
    if scanner is None:
        return 0
    pending = [
        ev for ev in list(getattr(scanner, "audio_events", []) or [])
        if str(getattr(ev, "state", "")) == "pending"
    ]
    if not pending:
        return 0
    ev = min(pending, key=lambda item: float(getattr(item, "peak_ts", 0.0) or 0.0))
    return int(getattr(ev, "shot_id", 0) or 0)


def _normalise_camera_regions(
    regions: Iterable[CameraHitRegionV240], *, margin_px: float, max_regions: int
) -> list[LocalSearchWindowV241]:
    out: list[LocalSearchWindowV241] = []
    margin = max(0.0, _finite(margin_px, 0.0))
    for region in list(regions)[: max(1, int(max_regions))]:
        x = _finite(getattr(region, "x", None), float("nan"))
        y = _finite(getattr(region, "y", None), float("nan"))
        w = _finite(getattr(region, "width", None), float("nan"))
        h = _finite(getattr(region, "height", None), float("nan"))
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0.0 or h <= 0.0:
            continue
        object_id = str(getattr(region, "object_id", "") or "")
        role = str(getattr(region, "role", "target") or "target")
        out.append(LocalSearchWindowV241(
            x - margin, y - margin, x + w + margin, y + h + margin,
            (object_id,) if object_id else (), (role,) if role else (),
        ))
    return out


def merge_camera_regions_v241(
    regions: Sequence[CameraHitRegionV240] | Iterable[CameraHitRegionV240], *,
    margin_px: float = 36.0, max_regions: int = 256,
) -> tuple[LocalSearchWindowV241, ...]:
    """Expand and transitively merge overlapping camera-space AABBs."""
    source = _normalise_camera_regions(regions, margin_px=margin_px, max_regions=max_regions)
    if not source:
        return ()
    parent = list(range(len(source)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(source)):
        for j in range(i + 1, len(source)):
            if source[i].overlaps(source[j]):
                union(i, j)

    groups: dict[int, LocalSearchWindowV241] = {}
    for i, window in enumerate(source):
        root = find(i)
        groups[root] = window if root not in groups else groups[root].union(window)
    return tuple(sorted(groups.values(), key=lambda r: (r.y0, r.x0, r.y1, r.x1)))


def build_local_valid_mask_v241(
    valid: np.ndarray,
    bbox: tuple[int, int, int, int] | Sequence[int],
    windows: Sequence[LocalSearchWindowV241], *,
    source_region_count: int | None = None,
) -> tuple[np.ndarray, LocalMaskDiagnosticsV241]:
    """Intersect the existing detector-valid ROI with object search windows."""
    base = np.asarray(valid, dtype=bool)
    if base.ndim != 2:
        raise ValueError("valid mask must be 2-D")
    camera_x0, camera_y0 = int(bbox[0]), int(bbox[1])
    mask = np.zeros(base.shape, dtype=bool)
    height, width = base.shape
    for window in windows:
        lx0 = max(0, min(width, int(math.floor(window.x0 - camera_x0))))
        ly0 = max(0, min(height, int(math.floor(window.y0 - camera_y0))))
        lx1 = max(0, min(width, int(math.ceil(window.x1 - camera_x0))))
        ly1 = max(0, min(height, int(math.ceil(window.y1 - camera_y0))))
        if lx1 > lx0 and ly1 > ly0:
            mask[ly0:ly1, lx0:lx1] = True
    restricted = base & mask
    diag = LocalMaskDiagnosticsV241(
        int(source_region_count if source_region_count is not None else len(windows)),
        len(windows), int(np.count_nonzero(mask)), int(np.count_nonzero(base)),
        int(np.count_nonzero(restricted)),
    )
    return restricted, diag


def local_search_context_v241(shot_id: int) -> tuple[ObjectShotSnapshotV2223 | None, tuple[CameraHitRegionV240, ...]]:
    snapshot = object_hit_registry_v2223.snapshot_for_shot(int(shot_id or 0))
    return (None, ()) if snapshot is None else (snapshot, tuple(snapshot.camera_regions))


def _set_debug(scanner: Any, **values: float) -> None:
    try:
        debug = getattr(scanner, "last_window_debug", None)
        if isinstance(debug, dict):
            for key, value in values.items():
                debug[key] = float(value)
    except Exception:
        pass


def _install_settings_defaults() -> None:
    defaults = {
        "object_local_search_enabled_v241": True,
        "object_local_search_margin_px_v241": 36.0,
        "object_local_search_max_regions_v241": 256,
        "object_local_search_log_v241": True,
    }
    try:
        import src.engine.ai.runtime as runtime_module
        runtime_module.DEFAULT_SETTINGS.update(defaults)
        existing = getattr(runtime_module, "_RUNTIME", None)
        if existing is not None:
            for key, value in defaults.items():
                getattr(existing, "settings", {}).setdefault(key, value)
    except Exception:
        pass


def _install_candidate_region_patch() -> None:
    from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
    if getattr(CandidateGeneratorV2, "_v241_object_local_patch", False):
        return
    previous_extract = CandidateGeneratorV2._extract_candidates
    CandidateGeneratorV2._v241_previous_extract = previous_extract

    def extract_v241(self, *args, **kwargs):
        scanner = kwargs.get("scanner")
        sid = _shot_id_from_scanner(scanner)
        live_worker = threading.current_thread().name.startswith("shot-cv-v2224")
        # Keep offline/F2/replay extraction byte-for-byte on the previous path.
        if not _CONFIG.enabled or sid <= 0 or not live_worker:
            return previous_extract(self, *args, **kwargs)

        # This test MUST happen before V2.22.5 consumes the rescue request.
        if rescue_router_v2225.requested(sid):
            local_search_telemetry_v241.record(sid, "full_rescue")
            _set_debug(scanner, v241_local_active=0, v241_global_fallback=1, v241_full_rescue=1)
            if _CONFIG.log:
                print(f"[V2.24.1 GLOBAL-FALLBACK] shot={sid} reason=v2225_full_rescue")
            return previous_extract(self, *args, **kwargs)

        snapshot, camera_regions = local_search_context_v241(sid)
        if snapshot is None or not camera_regions:
            has_source = bool(snapshot and (snapshot.game_regions or snapshot.regions))
            reason = "transform_unavailable" if has_source else "no_context"
            local_search_telemetry_v241.record(sid, reason)
            _set_debug(scanner, v241_local_active=0, v241_global_fallback=1, v241_full_rescue=0)
            if _CONFIG.log and reason != "no_context":
                print(f"[V2.24.1 GLOBAL-FALLBACK] shot={sid} reason={reason}")
            return previous_extract(self, *args, **kwargs)

        if len(camera_regions) > _CONFIG.max_regions:
            local_search_telemetry_v241.record(sid, "too_many_regions")
            _set_debug(scanner, v241_local_active=0, v241_global_fallback=1)
            if _CONFIG.log:
                print(f"[V2.24.1 GLOBAL-FALLBACK] shot={sid} reason=too_many_regions count={len(camera_regions)}")
            return previous_extract(self, *args, **kwargs)

        valid, bbox = kwargs.get("valid"), kwargs.get("bbox")
        if not isinstance(valid, np.ndarray) or bbox is None:
            local_search_telemetry_v241.record(sid, "missing_extractor_geometry")
            _set_debug(scanner, v241_local_active=0, v241_global_fallback=1)
            return previous_extract(self, *args, **kwargs)

        windows = merge_camera_regions_v241(camera_regions, margin_px=_CONFIG.margin_px, max_regions=_CONFIG.max_regions)
        if not windows:
            local_search_telemetry_v241.record(sid, "invalid_regions")
            return previous_extract(self, *args, **kwargs)

        try:
            local_valid, diag = build_local_valid_mask_v241(
                valid, bbox, windows, source_region_count=len(camera_regions)
            )
        except Exception:
            local_search_telemetry_v241.record(sid, "mask_error")
            return previous_extract(self, *args, **kwargs)

        # Invalid/out-of-ROI context fails open to today's global path.
        if diag.local_valid_pixels <= 0:
            local_search_telemetry_v241.record(sid, "outside_detector_roi", diag)
            _set_debug(scanner, v241_local_active=0, v241_global_fallback=1,
                       v241_region_count=diag.region_count, v241_merged_count=diag.merged_count,
                       v241_valid_fraction=diag.valid_fraction)
            if _CONFIG.log:
                print(f"[V2.24.1 GLOBAL-FALLBACK] shot={sid} reason=outside_detector_roi regions={diag.region_count}")
            return previous_extract(self, *args, **kwargs)

        local_search_telemetry_v241.record(sid, "local", diag)
        _set_debug(scanner, v241_local_active=1, v241_global_fallback=0, v241_full_rescue=0,
                   v241_region_count=diag.region_count, v241_merged_count=diag.merged_count,
                   v241_mask_pixels=diag.mask_pixels, v241_local_valid_pixels=diag.local_valid_pixels,
                   v241_valid_fraction=diag.valid_fraction, v241_margin_px=_CONFIG.margin_px)
        if _CONFIG.log:
            print(f"[V2.24.1 LOCAL-SEARCH] shot={sid} regions={diag.region_count} merged={diag.merged_count} "
                  f"valid={diag.valid_fraction * 100.0:.1f}% margin={_CONFIG.margin_px:.0f}px")
        local_kwargs = dict(kwargs)
        local_kwargs["valid"] = local_valid
        return previous_extract(self, *args, **local_kwargs)

    CandidateGeneratorV2._extract_candidates = extract_v241
    CandidateGeneratorV2._v241_object_local_patch = True


def install_v241_runtime(AppClass: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_settings_defaults()
    _load_config_from_runtime()
    _install_candidate_region_patch()
    AppClass._v241_object_local_search_patch = True
    _INSTALLED = True
    print(f"[V2.24.1] object-aware local physical search installed "
          f"(margin={_CONFIG.margin_px:.0f}px, global rescue preserved)")


__all__ = [
    "SCHEMA_VERSION", "PATCH_REVISION", "ObjectLocalConfigV241",
    "LocalSearchWindowV241", "LocalMaskDiagnosticsV241",
    "merge_camera_regions_v241", "build_local_valid_mask_v241",
    "local_search_context_v241", "local_search_telemetry_v241",
    "install_v241_runtime",
]
