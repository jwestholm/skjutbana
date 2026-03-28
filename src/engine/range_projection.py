from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass(frozen=True)
class RangeProjectionGeometry:
    wall_distance_m: float
    viewport_physical_width_cm: float
    viewport_physical_height_cm: float


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
    """
    Gemensam djupfaktor för hela projektionen.

    Vid distance == wall_distance blir ratio == 1.0
    Vid större avstånd går ratio asymptotiskt mot 0.
    """
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


def projected_ground_anchor_y_px(
    distance_m: float,
    viewport: pygame.Rect,
    geometry: RangeProjectionGeometry,
    *,
    near_ground_y_norm: float = 0.94,
    horizon_y_norm: float = 0.43,
) -> float:
    """
    Fotpunkten rör sig mot horisonten med samma depth_ratio som resten.

    Detta är mer sammanhållet än att interpolera mellan ett near- och far-läge
    med en separat kurva.
    """
    ratio = depth_ratio(
        virtual_distance_m=distance_m,
        wall_distance_m=geometry.wall_distance_m,
    )
    ratio = max(0.0, min(1.0, ratio))

    y_norm = horizon_y_norm + (near_ground_y_norm - horizon_y_norm) * ratio
    return viewport.y + (y_norm * viewport.h)


def projected_centerline_x_px(
    distance_m: float,
    viewport: pygame.Rect,
    geometry: RangeProjectionGeometry,
    *,
    near_center_x_norm: float = 0.56,
    vanishing_x_norm: float = 0.52,
) -> float:
    """
    Baslinjen som figuren rör sig längs i bakgrunden.

    Vid nära håll ligger den vid near_center_x_norm.
    På långt håll konvergerar den mot vanishing_x_norm.
    """
    ratio = depth_ratio(
        virtual_distance_m=distance_m,
        wall_distance_m=geometry.wall_distance_m,
    )
    ratio = max(0.0, min(1.0, ratio))

    x_norm = vanishing_x_norm + (near_center_x_norm - vanishing_x_norm) * ratio
    return viewport.x + (x_norm * viewport.w)