from __future__ import annotations

import time
from dataclasses import dataclass

import pygame

from src.engine.input.hit_input import hit_input, HitEvent
from src.engine.settings import (
    load_visual_hits_enabled,
    load_visual_hits_mode,
    load_visual_hits_lifetime_ms,
    load_visual_hits_radius,
)


@dataclass
class VisualHit:
    x: float
    y: float
    timestamp: float
    source: str


class HitVisualizer:

    COLOR_MOUSE = (255, 60, 60)
    COLOR_CAMERA = (60, 255, 60)
    COLOR_DEFAULT = (255, 255, 255)

    def __init__(self):
        self.hits: list[VisualHit] = []

        hit_input.subscribe(self._on_hit)

    # --------------------------------------------------

    def clear(self):
        """Rensa alla visualiserade träffar."""
        self.hits.clear()

    # --------------------------------------------------

    def _on_hit(self, event: HitEvent):

        if not load_visual_hits_enabled():
            return

        self.hits.append(
            VisualHit(
                x=event.screen_x,
                y=event.screen_y,
                timestamp=time.time(),
                source=event.source,
            )
        )

    # --------------------------------------------------

    def update(self, dt: float):
        del dt

        mode = load_visual_hits_mode()

        if mode == "persistent":
            return

        lifetime = load_visual_hits_lifetime_ms() / 1000.0
        now = time.time()

        self.hits = [
            hit for hit in self.hits
            if now - hit.timestamp <= lifetime
        ]

    # --------------------------------------------------

    def _color_for_source(self, source: str):

        if source == "mouse":
            return self.COLOR_MOUSE

        if source == "camera":
            return self.COLOR_CAMERA

        return self.COLOR_DEFAULT

    # --------------------------------------------------

    def render(self, screen: pygame.Surface):

        if not load_visual_hits_enabled():
            return

        radius = load_visual_hits_radius()

        for hit in self.hits:

            color = self._color_for_source(hit.source)

            pygame.draw.circle(
                screen,
                color,
                (int(hit.x), int(hit.y)),
                radius,
                3,
            )


hit_visualizer = HitVisualizer()