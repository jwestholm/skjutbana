from __future__ import annotations

from dataclasses import dataclass

import pygame

from src.engine.input.hit_input import HitEvent, hit_input
from src.engine.settings import load_visual_hits_enabled, load_visual_hits_mode


@dataclass
class VisualMarker:
    x: float
    y: float
    age: float = 0.0


class HitVisualizer:
    def __init__(self) -> None:
        self.enabled = load_visual_hits_enabled()
        self.mode = load_visual_hits_mode()
        self.duration_seconds = 2.5
        self.markers: list[VisualMarker] = []
        hit_input.subscribe(self.on_hit)

    def reload_settings(self) -> None:
        self.enabled = load_visual_hits_enabled()
        self.mode = load_visual_hits_mode()

    def on_hit(self, event: HitEvent) -> None:
        self.reload_settings()
        if not self.enabled:
            return
        self.markers.append(VisualMarker(x=event.screen_x, y=event.screen_y, age=0.0))

    def clear(self) -> None:
        self.markers.clear()

    def update(self, dt: float) -> None:
        if not self.enabled:
            self.markers.clear()
            return

        if self.mode == "persistent":
            return

        survivors: list[VisualMarker] = []
        for marker in self.markers:
            marker.age += float(dt)
            if marker.age <= self.duration_seconds:
                survivors.append(marker)
        self.markers = survivors

    def render(self, screen: pygame.Surface) -> None:
        if not self.enabled or not self.markers:
            return

        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

        for marker in self.markers:
            alpha = 255
            if self.mode == "fade":
                t = max(0.0, min(1.0, marker.age / self.duration_seconds))
                alpha = int(255 * (1.0 - t))

            color = (255, 70, 70, alpha)
            outer = (255, 255, 255, max(0, alpha // 2))

            x = int(round(marker.x))
            y = int(round(marker.y))

            pygame.draw.circle(overlay, outer, (x, y), 16, 1)
            pygame.draw.circle(overlay, color, (x, y), 11, 3)
            pygame.draw.line(overlay, color, (x - 18, y), (x + 18, y), 3)
            pygame.draw.line(overlay, color, (x, y - 18), (x, y + 18), 3)

        screen.blit(overlay, (0, 0))


hit_visualizer = HitVisualizer()