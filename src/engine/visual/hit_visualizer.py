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

    def __init__(self):

        self.enabled = load_visual_hits_enabled()
        self.mode = load_visual_hits_mode()

        self.duration = 2.5
        self.markers: list[VisualMarker] = []

        hit_input.subscribe(self.on_hit)

    def reload_settings(self):

        self.enabled = load_visual_hits_enabled()
        self.mode = load_visual_hits_mode()

    def on_hit(self, event: HitEvent):

        self.reload_settings()

        if not self.enabled:
            return

        self.markers.append(VisualMarker(event.screen_x, event.screen_y))

    def clear(self):

        self.markers.clear()

    def update(self, dt):

        if not self.enabled:
            self.markers.clear()
            return

        if self.mode == "persistent":
            return

        alive = []

        for m in self.markers:

            m.age += dt

            if m.age < self.duration:
                alive.append(m)

        self.markers = alive

    def render(self, screen):

        if not self.enabled:
            return

        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)

        for m in self.markers:

            alpha = 255

            if self.mode == "fade":
                t = min(1.0, m.age / self.duration)
                alpha = int(255 * (1 - t))

            color = (255, 60, 60, alpha)

            x = int(m.x)
            y = int(m.y)

            pygame.draw.circle(overlay, color, (x, y), 12, 3)
            pygame.draw.line(overlay, color, (x - 16, y), (x + 16, y), 3)
            pygame.draw.line(overlay, color, (x, y - 16), (x, y + 16), 3)

        screen.blit(overlay, (0, 0))


hit_visualizer = HitVisualizer()