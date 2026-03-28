from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass(frozen=True)
class RangeProjectionGeometry:
    wall_distance_m: float
    viewport_physical_width_cm: float
    viewport_physical_height_cm: float
    viewport_bottom_world_cm: float


def clamp_distance_m(
    distance_m: float,
    min_distance_m: float = 5.0,
    max_distance_m: float = 600.0,
) -> float:
    return max(min_distance_m, min(max_distance_m, float(distance_m)))


def _safe_distance_m(distance_m: float) -> float:
    return max(0.01, float(distance_m))


def _safe_wall_distance_m(wall_distance_m: float) -> float:
    return max(0.01, float(wall_distance_m))


def depth_ratio(
    virtual_distance_m: float,
    wall_distance_m: float,
) -> float:
    z = _safe_distance_m(virtual_distance_m)
    wall = _safe_wall_distance_m(wall_distance_m)
    return wall / z


def projected_size_on_wall_cm(
    real_size_cm: float,
    virtual_distance_m: float,
    wall_distance_m: float,
) -> float:
    """
    Likformiga trianglar:
    size_on_wall = real_size * wall_distance / virtual_distance
    """
    return float(real_size_cm) * depth_ratio(
        virtual_distance_m=virtual_distance_m,
        wall_distance_m=wall_distance_m,
    )


def cm_to_viewport_px_x(
    size_cm: float,
    viewport: pygame.Rect,
    viewport_physical_width_cm: float,
) -> float:
    viewport_physical_width_cm = max(0.01, float(viewport_physical_width_cm))
    return (float(size_cm) / viewport_physical_width_cm) * viewport.w


def cm_to_viewport_px_y(
    size_cm: float,
    viewport: pygame.Rect,
    viewport_physical_height_cm: float,
) -> float:
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
    Sidled följer samma djupfaktor som storleken.
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


def near_ground_anchor_y_norm(geometry: RangeProjectionGeometry) -> float:
    """
    Fotpunktens normerade y-läge vid väggplanet.

    Exempel:
    - viewport height = 70 cm
    - viewport bottom = 105 cm över mark
    Då ligger marken 105 cm under viewportens nederkant.
    Det blir 105 / 70 = 1.5 viewport-höjder under bilden.
    Fotpunkten blir alltså vid y_norm = 1.0 + 1.5 = 2.5
    """
    return 1.0 + (
        float(geometry.viewport_bottom_world_cm)
        / max(0.01, float(geometry.viewport_physical_height_cm))
    )


def projected_ground_anchor_y_px(
    distance_m: float,
    viewport: pygame.Rect,
    geometry: RangeProjectionGeometry,
    *,
    horizon_y_norm: float = 0.5,
) -> float:
    """
    Fotpunkten följer markplanet:
    - vid väggplanet ligger den på near_ground_anchor_y_norm(...)
    - längre bort går den mot horisonten
    """
    ratio = depth_ratio(
        virtual_distance_m=distance_m,
        wall_distance_m=geometry.wall_distance_m,
    )
    ratio = max(0.0, min(1.0, ratio))

    near_y = near_ground_anchor_y_norm(geometry)
    y_norm = horizon_y_norm + (near_y - horizon_y_norm) * ratio

    return viewport.y + (y_norm * viewport.h)