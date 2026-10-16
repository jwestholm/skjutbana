"""Shared distance/size LOD policy for scenery generation.

Stage 1: conservative static budget based on minimum possible distance to the
playable trench centreline. Does NOT change live GPU mesh or ballistic shapes.
Stage 2 (future): camera-dependent meshes + hysteresis for movable viewpoints.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LODLevel:
    name: str
    trunk_facets: int
    branch_facets: int
    leaf_segments: int
    leaf_rows: int
    bush_clumps: int
    twig_budget: int


NEAR = LODLevel('near', 10, 7, 12, 5, 4, 3)
MID = LODLevel('mid', 7, 6, 8, 3, 3, 1)
FAR = LODLevel('far', 6, 6, 6, 2, 1, 0)


def lod_from_min_distance(distance_m: float, radius_m: float = 1.0) -> LODLevel:
    """Conservative per-object *minimum* possible view distance, metres.

    An object touching the route stays high detail. No camera-distance popping
    as the player advances; distant props have cheaper baked meshes.
    """
    if not math.isfinite(distance_m) or not math.isfinite(radius_m):
        raise ValueError('LOD values must be finite')
    near_edge = max(0., distance_m - max(0., radius_m))
    if near_edge <= 16.: return NEAR
    if near_edge <= 48.: return MID
    return FAR


def screen_coverage_px(radius_m: float, distance_m: float, viewport_height: int,
                       vertical_fov_deg: float) -> float:
    """Budget helper for future runtime LOD; objects evaluated by screen size."""
    if radius_m <= 0 or distance_m <= 0 or viewport_height <= 0 or not 0 < vertical_fov_deg < 180:
        raise ValueError('invalid projection parameters')
    return (radius_m * viewport_height /
            (distance_m * math.tan(math.radians(vertical_fov_deg) / 2)))
