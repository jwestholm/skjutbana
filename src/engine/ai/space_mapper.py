from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.engine.input.hit_input import hit_input
from src.engine.settings import load_content_rect, load_viewport_rect


@dataclass
class ProjectionPoint:
    camera_x: float
    camera_y: float
    screen_x: float
    screen_y: float
    viewport_x: float
    viewport_y: float
    content_x: float
    content_y: float
    content_norm_x: float
    content_norm_y: float
    game_x: float
    game_y: float


def _safe_camera_to_screen(camera_x: float, camera_y: float) -> tuple[float, float]:
    try:
        sx, sy = hit_input._canonical_camera_to_screen(float(camera_x), float(camera_y))  # type: ignore[attr-defined]
        return float(sx), float(sy)
    except Exception:
        viewport = load_viewport_rect()
        if viewport is None:
            return float(camera_x), float(camera_y)
        return float(viewport.left + camera_x), float(viewport.top + camera_y)


def _safe_screen_to_camera(screen_x: float, screen_y: float) -> tuple[float, float]:
    try:
        cx, cy = hit_input._canonical_screen_to_camera(float(screen_x), float(screen_y))  # type: ignore[attr-defined]
        return float(cx), float(cy)
    except Exception:
        viewport = load_viewport_rect()
        if viewport is None:
            return float(screen_x), float(screen_y)
        return float(screen_x - viewport.left), float(screen_y - viewport.top)


def project_camera_point(camera_x: float, camera_y: float) -> ProjectionPoint:
    screen_x, screen_y = _safe_camera_to_screen(camera_x, camera_y)
    try:
        viewport_x, viewport_y, content_x, content_y, content_norm_x, content_norm_y = hit_input._screen_to_spaces(screen_x, screen_y)  # type: ignore[attr-defined]
    except Exception:
        viewport = load_viewport_rect()
        content = load_content_rect()
        viewport_x = float(screen_x - viewport.x) if viewport is not None else float(screen_x)
        viewport_y = float(screen_y - viewport.y) if viewport is not None else float(screen_y)
        if viewport is not None and content is not None:
            abs_content_x = float(viewport.x + content.x)
            abs_content_y = float(viewport.y + content.y)
            content_x = float(screen_x - abs_content_x)
            content_y = float(screen_y - abs_content_y)
            content_norm_x = content_x / float(content.w) if content.w > 0 else 0.0
            content_norm_y = content_y / float(content.h) if content.h > 0 else 0.0
        else:
            content_x = viewport_x
            content_y = viewport_y
            content_norm_x = 0.0
            content_norm_y = 0.0
    return ProjectionPoint(
        camera_x=float(camera_x),
        camera_y=float(camera_y),
        screen_x=float(screen_x),
        screen_y=float(screen_y),
        viewport_x=float(viewport_x),
        viewport_y=float(viewport_y),
        content_x=float(content_x),
        content_y=float(content_y),
        content_norm_x=float(content_norm_x),
        content_norm_y=float(content_norm_y),
        game_x=float(viewport_x),
        game_y=float(viewport_y),
    )


def project_screen_point(screen_x: float, screen_y: float) -> ProjectionPoint:
    camera_x, camera_y = _safe_screen_to_camera(screen_x, screen_y)
    point = project_camera_point(camera_x, camera_y)
    return ProjectionPoint(
        camera_x=point.camera_x,
        camera_y=point.camera_y,
        screen_x=float(screen_x),
        screen_y=float(screen_y),
        viewport_x=point.viewport_x,
        viewport_y=point.viewport_y,
        content_x=point.content_x,
        content_y=point.content_y,
        content_norm_x=point.content_norm_x,
        content_norm_y=point.content_norm_y,
        game_x=point.game_x,
        game_y=point.game_y,
    )


def candidate_with_projection(candidate: dict[str, Any]) -> dict[str, Any]:
    camera_x = float(candidate.get("camera_x", candidate.get("x", 0.0)))
    camera_y = float(candidate.get("camera_y", candidate.get("y", 0.0)))
    projected = project_camera_point(camera_x, camera_y)
    enriched = dict(candidate)
    enriched.update(
        {
            "camera_x": projected.camera_x,
            "camera_y": projected.camera_y,
            "screen_x": projected.screen_x,
            "screen_y": projected.screen_y,
            "viewport_x": projected.viewport_x,
            "viewport_y": projected.viewport_y,
            "content_x": projected.content_x,
            "content_y": projected.content_y,
            "content_norm_x": projected.content_norm_x,
            "content_norm_y": projected.content_norm_y,
            "game_x": projected.game_x,
            "game_y": projected.game_y,
        }
    )
    return enriched
