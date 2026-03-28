from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass(frozen=True)
class RangeProjectionGeometry:
    wall_distance_m: float
    viewport_physical_width_cm: float
    viewport_physical_height_cm: float


def clamp_distance_m(distance_m: float, min_distance_m: float = 5.0, max_distance_m: float = 600.0) -> float:
    return max(min_distance_m, min(max_distance_m, float(distance_m)))


def projected_size_on_wall_cm(real_size_cm: float, virtual_distance_m: float, wall_distance_m: float) -> float:
    """
    Likformiga trianglar:
    size_on_wall = real_size * wall_distance / virtual_distance
    """
    virtual_distance_m = max(0.01, float(virtual_distance_m))
    wall_distance_m = max(0.01, float(wall_distance_m))
    return float(real_size_cm) * (wall_distance_m / virtual_distance_m)


def cm_to_viewport_px_x(size_cm: float, viewport: pygame.Rect, viewport_physical_width_cm: float) -> float:
    viewport_physical_width_cm = max(0.01, float(viewport_physical_width_cm))
    return (float(size_cm) / viewport_physical_width_cm) * viewport.w


def cm_to_viewport_px_y(size_cm: float, viewport: pygame.Rect, viewport_physical_height_cm: float) -> float:
    viewport_physical_height_cm = max(0.01, float(viewport_physical_height_cm))
    return (float(size_cm) / viewport_physical_height_cm) * viewport.h


def projected_target_height_px(
    real_height_cm: float,
    virtual_distance_m: float,
    viewport: pygame.Rect,
    geometry: RangeProjectionGeometry,
) -> float:
    wall_height_cm = projected_size_on_wall_cm(
        real_size_cm=real_height_cm,
        virtual_distance_m=virtual_distance_m,
        wall_distance_m=geometry.wall_distance_m,
    )
    return cm_to_viewport_px_y(
        size_cm=wall_height_cm,
        viewport=viewport,
        viewport_physical_height_cm=geometry.viewport_physical_height_cm,
    )


def projected_lateral_offset_px(
    world_lateral_offset_m: float,
    virtual_distance_m: float,
    viewport: pygame.Rect,
    geometry: RangeProjectionGeometry,
) -> float:
    """
    En sidoförflyttning i den simulerade världen projiceras också via likformiga trianglar.
    """
    offset_on_wall_cm = projected_size_on_wall_cm(
        real_size_cm=float(world_lateral_offset_m) * 100.0,
        virtual_distance_m=virtual_distance_m,
        wall_distance_m=geometry.wall_distance_m,
    )
    return cm_to_viewport_px_x(
        size_cm=offset_on_wall_cm,
        viewport=viewport,
        viewport_physical_width_cm=geometry.viewport_physical_width_cm,
    )


def _inverse_depth_factor(distance_m: float, min_distance_m: float, max_distance_m: float) -> float:
    """
    0 = längst bort
    1 = närmast

    Vi använder invers avståndsskala för att få rörelsen i Y-led att kännas mer perspektivisk.
    """
    d = clamp_distance_m(distance_m, min_distance_m=min_distance_m, max_distance_m=max_distance_m)
    near_inv = 1.0 / max(0.01, min_distance_m)
    far_inv = 1.0 / max(0.01, max_distance_m)
    value_inv = 1.0 / max(0.01, d)

    denom = (near_inv - far_inv)
    if abs(denom) < 1e-9:
        return 0.0

    t = (value_inv - far_inv) / denom
    return max(0.0, min(1.0, t))


def projected_ground_anchor_y_px(
    distance_m: float,
    viewport: pygame.Rect,
    *,
    min_distance_m: float = 5.0,
    max_distance_m: float = 600.0,
    near_ground_y_norm: float = 0.92,
    far_ground_y_norm: float = 0.58,
) -> float:
    """
    En enkel motorfunktion som låter targetens fötter röra sig 'bakåt in i bilden'
    när den blir längre bort.

    near_ground_y_norm:
        var fötterna ska ligga i viewporten när targeten är nära

    far_ground_y_norm:
        var fötterna ska ligga när targeten är långt bort
    """
    t = _inverse_depth_factor(distance_m, min_distance_m=min_distance_m, max_distance_m=max_distance_m)
    y_norm = far_ground_y_norm + (near_ground_y_norm - far_ground_y_norm) * t
    return viewport.y + y_norm * viewport.h