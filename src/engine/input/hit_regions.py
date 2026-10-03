"""Stable game-facing hit-region API (V2.24+).

Games should import from this module, not from the versioned V2.22.3 runtime
module.  HitRegion coordinates are always viewport-local/game-local.
"""
from __future__ import annotations

from dataclasses import replace

from src.engine.input.object_hit_v2223 import (
    CameraHitRegionV240 as CameraHitRegion,
    GameHitRegionV240 as HitRegion,
    object_hit_registry_v2223,
)


def game_to_camera_point(x: float, y: float):
    return object_hit_registry_v2223.game_to_camera_point(x, y)


def camera_to_game_point(x: float, y: float):
    return object_hit_registry_v2223.camera_to_game_point(x, y)


def game_rect_to_camera_aabb(region: HitRegion):
    return object_hit_registry_v2223.game_rect_to_camera_aabb(region)


def latest_hit_context_snapshot():
    return object_hit_registry_v2223.latest_snapshot()


def hit_context_snapshot_for_shot(shot_id: int):
    """Return the exact frozen game context for one scanner shot id."""
    return object_hit_registry_v2223.snapshot_for_shot(int(shot_id))


def object_query_regions(regions, *, top_k=3, margin_camera_px=36., owner=None,
                         surface_after_ts=None, require_presented_surface=False):
    """Request OBJECT over these shot-frozen regions. Margin never expands collision.

    Results arrive as EventBus ``physical.object_result`` and remain available
    through object_physical_result(shot_id). No HitInput event is fabricated.
    """
    if int(top_k)<1: raise ValueError('OBJECT top_k must be positive')
    config=dict(mode='OBJECT',top_k=min(3,int(top_k)),
        margin_camera_px=max(0.,min(128.,float(margin_camera_px))))
    if owner is not None: config['owner']=str(owner)
    if require_presented_surface: config['surface_after_ts']=surface_after_ts
    return tuple(replace(r,metadata={**r.metadata,'physical_query':dict(config)}) for r in regions)


def object_physical_result(shot_id):
    return object_hit_registry_v2223.object_result(int(shot_id))


__all__ = [
    "HitRegion",
    "CameraHitRegion",
    "game_to_camera_point",
    "camera_to_game_point",
    "game_rect_to_camera_aabb",
    "latest_hit_context_snapshot",
    "hit_context_snapshot_for_shot",
    "object_query_regions",
    "object_physical_result",
]
