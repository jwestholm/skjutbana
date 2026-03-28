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


def viewport_top_world_cm(geometry: RangeProjectionGeometry) -> float:
    return (
        float(geometry.viewport_bottom_world_cm)
        + float(geometry.viewport_physical_height_cm)
    )


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


def project_world_height_to_wall_cm(
    world_height_cm: float,
    virtual_distance_m: float,
    geometry: RangeProjectionGeometry,
) -> float:
    return projected_size_on_wall_cm(
        real_size_cm=world_height_cm,
        virtual_distance_m=virtual_distance_m,
        wall_distance_m=geometry.wall_distance_m,
    )


def world_height_to_screen_y_px(
    world_height_cm: float,
    virtual_distance_m: float,
    viewport: pygame.Rect,
    geometry: RangeProjectionGeometry,
) -> float:
    """
    Mappar en världshöjd över mark till en pixel-y inne i viewportens
    fysiska intervall på väggen.

    Exempel:
    - viewport_bottom_world_cm = 105
    - viewport_physical_height_cm = 70
    Då motsvarar viewporten vägghöjderna 105..175 cm över mark.
    """
    wall_height_cm = project_world_height_to_wall_cm(
        world_height_cm=world_height_cm,
        virtual_distance_m=virtual_distance_m,
        geometry=geometry,
    )

    bottom_cm = float(geometry.viewport_bottom_world_cm)
    top_cm = viewport_top_world_cm(geometry)
    span_cm = max(0.01, top_cm - bottom_cm)

    y_norm = 1.0 - ((wall_height_cm - bottom_cm) / span_cm)
    return viewport.y + (y_norm * viewport.h)


def figure_top_y_px(
    figure_height_cm: float,
    virtual_distance_m: float,
    viewport: pygame.Rect,
    geometry: RangeProjectionGeometry,
) -> float:
    return world_height_to_screen_y_px(
        world_height_cm=figure_height_cm,
        virtual_distance_m=virtual_distance_m,
        viewport=viewport,
        geometry=geometry,
    )


def figure_foot_y_px(
    virtual_distance_m: float,
    viewport: pygame.Rect,
    geometry: RangeProjectionGeometry,
) -> float:
    return world_height_to_screen_y_px(
        world_height_cm=0.0,
        virtual_distance_m=virtual_distance_m,
        viewport=viewport,
        geometry=geometry,
    )