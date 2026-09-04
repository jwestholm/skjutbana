"""Stable game-facing hit-region API (V2.24+).

Games should import from this module, not from the versioned V2.22.3 runtime
module.  HitRegion coordinates are always viewport-local/game-local.
"""
from __future__ import annotations

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


__all__ = [
    "HitRegion",
    "CameraHitRegion",
    "game_to_camera_point",
    "camera_to_game_point",
    "game_rect_to_camera_aabb",
    "latest_hit_context_snapshot",
    "hit_context_snapshot_for_shot",
]
