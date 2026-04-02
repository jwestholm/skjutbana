from __future__ import annotations

import math
import time
from dataclasses import dataclass

import pygame

from src.engine.input.hit_input import HitEvent, hit_input
from src.engine.settings import (
    load_content_rect,
    load_viewport_rect,
    load_visual_hits_enabled,
    load_visual_hits_lifetime_ms,
    load_visual_hits_mode,
    load_visual_hits_radius,
    load_visual_hits_show_all_planes,
)


@dataclass
class VisualMarker:
    x: float
    y: float
    timestamp: float
    color: tuple[int, int, int]
    label: str
    style: str


class HitVisualizer:
    COLOR_MOUSE = (255, 60, 60)
    COLOR_CAMERA = (0, 200, 255)
    COLOR_VIEWPORT = (255, 220, 80)
    COLOR_CONTENT = (255, 120, 220)
    COLOR_SCANPORT = (120, 255, 120)
    COLOR_HOMOGRAPHY = (190, 120, 255)
    COLOR_DEFAULT = (255, 255, 255)

    def __init__(self):
        self.hits: list[VisualMarker] = []
        hit_input.subscribe(self._on_hit)

    def reload_settings(self):
        """
        Behålls för kompatibilitet med befintliga settings-scener.
        Inställningarna läses dynamiskt i render/update.
        """
        return None

    def clear(self):
        self.hits.clear()

    def _color_for_source(self, source: str):
        if source == "mouse":
            return self.COLOR_MOUSE
        if source == "camera":
            return self.COLOR_CAMERA
        return self.COLOR_DEFAULT

    def _append_marker(
        self,
        markers: list[VisualMarker],
        *,
        x: float,
        y: float,
        color: tuple[int, int, int],
        label: str,
        style: str,
        timestamp: float,
    ):
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        markers.append(
            VisualMarker(
                x=float(x),
                y=float(y),
                timestamp=timestamp,
                color=color,
                label=label,
                style=style,
            )
        )

    def _markers_for_event(self, event: HitEvent) -> list[VisualMarker]:
        now = time.time()
        markers: list[VisualMarker] = []
        base_color = self._color_for_source(event.source)

        if not load_visual_hits_show_all_planes():
            self._append_marker(
                markers,
                x=event.screen_x,
                y=event.screen_y,
                color=base_color,
                label=event.source,
                style="cross",
                timestamp=now,
            )
            return markers

        viewport = load_viewport_rect()
        content = load_content_rect()

        viewport_screen_x = float(viewport.x + event.viewport_x)
        viewport_screen_y = float(viewport.y + event.viewport_y)
        content_screen_x = float(content.x + event.content_x)
        content_screen_y = float(content.y + event.content_y)

        self._append_marker(
            markers,
            x=event.requested_screen_x,
            y=event.requested_screen_y,
            color=base_color,
            label="click" if event.source == "mouse" else "requested",
            style="ring",
            timestamp=now,
        )
        self._append_marker(
            markers,
            x=event.screen_x,
            y=event.screen_y,
            color=base_color,
            label="screen",
            style="cross",
            timestamp=now,
        )
        self._append_marker(
            markers,
            x=viewport_screen_x,
            y=viewport_screen_y,
            color=self.COLOR_VIEWPORT,
            label="viewport",
            style="square",
            timestamp=now,
        )
        self._append_marker(
            markers,
            x=content_screen_x,
            y=content_screen_y,
            color=self.COLOR_CONTENT,
            label="content",
            style="diamond",
            timestamp=now,
        )
        self._append_marker(
            markers,
            x=event.scanport_screen_x,
            y=event.scanport_screen_y,
            color=self.COLOR_SCANPORT,
            label="scanport",
            style="triangle",
            timestamp=now,
        )
        self._append_marker(
            markers,
            x=event.homography_screen_x,
            y=event.homography_screen_y,
            color=self.COLOR_HOMOGRAPHY,
            label="homography",
            style="x",
            timestamp=now,
        )
        return markers

    def _on_hit(self, event: HitEvent):
        if not load_visual_hits_enabled():
            return
        self.hits.extend(self._markers_for_event(event))

    def update(self, dt: float):
        del dt
        mode = load_visual_hits_mode()
        if mode == "persistent":
            return

        lifetime = load_visual_hits_lifetime_ms() / 1000.0
        now = time.time()
        self.hits = [hit for hit in self.hits if now - hit.timestamp <= lifetime]

    def _draw_cross(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        pygame.draw.circle(overlay, color, (x, y), radius, 2)
        pygame.draw.line(overlay, color, (x - radius - 4, y), (x + radius + 4, y), 2)
        pygame.draw.line(overlay, color, (x, y - radius - 4), (x, y + radius + 4), 2)

    def _draw_ring(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        pygame.draw.circle(overlay, color, (x, y), radius + 6, 2)
        pygame.draw.circle(overlay, color, (x, y), max(2, radius - 4), 1)

    def _draw_square(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        size = radius * 2
        rect = pygame.Rect(x - radius, y - radius, size, size)
        pygame.draw.rect(overlay, color, rect, 2)

    def _draw_diamond(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        points = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
        pygame.draw.polygon(overlay, color, points, 2)

    def _draw_triangle(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        points = [(x, y - radius), (x + radius, y + radius), (x - radius, y + radius)]
        pygame.draw.polygon(overlay, color, points, 2)

    def _draw_x(self, overlay: pygame.Surface, color, x: int, y: int, radius: int):
        pygame.draw.line(overlay, color, (x - radius, y - radius), (x + radius, y + radius), 2)
        pygame.draw.line(overlay, color, (x - radius, y + radius), (x + radius, y - radius), 2)

    def _draw_marker(self, overlay: pygame.Surface, marker: VisualMarker, radius: int):
        rgba = (marker.color[0], marker.color[1], marker.color[2], 255)
        x = int(round(marker.x))
        y = int(round(marker.y))

        if marker.style == "ring":
            self._draw_ring(overlay, rgba, x, y, radius)
        elif marker.style == "square":
            self._draw_square(overlay, rgba, x, y, radius)
        elif marker.style == "diamond":
            self._draw_diamond(overlay, rgba, x, y, radius)
        elif marker.style == "triangle":
            self._draw_triangle(overlay, rgba, x, y, radius)
        elif marker.style == "x":
            self._draw_x(overlay, rgba, x, y, radius)
        else:
            self._draw_cross(overlay, rgba, x, y, radius)

    def render(self, screen: pygame.Surface):
        if not load_visual_hits_enabled():
            return

        radius = load_visual_hits_radius()
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        show_all_planes = load_visual_hits_show_all_planes()
        label_font = pygame.font.Font(None, 18) if show_all_planes else None

        for hit in self.hits:
            self._draw_marker(overlay, hit, radius)
            if show_all_planes and label_font is not None:
                text = label_font.render(hit.label, True, hit.color)
                overlay.blit(text, (int(round(hit.x)) + radius + 8, int(round(hit.y)) - 10))

        screen.blit(overlay, (0, 0))


hit_visualizer = HitVisualizer()
