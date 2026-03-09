from __future__ import annotations

import pygame

from src.engine.scene import Scene
from src.engine.visual.hit_visualizer import hit_visualizer
from src.engine.visual.scanner_debug_overlay import scanner_debug_overlay
from src.engine.visual.scanner_status_overlay import scanner_status_overlay
from src.engine.input.hit_input import hit_input


class OverlayScene(Scene):
    def __init__(self, inner: Scene):
        self.inner = inner

    @property
    def wants_hit_scanning(self) -> bool:
        return bool(getattr(self.inner, "wants_hit_scanning", False))

    @property
    def wants_camera_preview(self) -> bool:
        return bool(getattr(self.inner, "wants_camera_preview", False))

    def on_enter(self):
        if hasattr(self.inner, "on_enter"):
            self.inner.on_enter()

    def on_exit(self):
        hit_visualizer.clear()

        if hasattr(self.inner, "on_exit"):
            self.inner.on_exit()

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            hit_input.push_mouse_hit(event.pos[0], event.pos[1])

        return self.inner.handle_event(event)

    def update(self, dt):
        result = self.inner.update(dt)
        hit_visualizer.update(dt)
        return result

    def render(self, screen):
        self.inner.render(screen)
        hit_visualizer.render(screen)
        scanner_debug_overlay.render(screen)
        scanner_status_overlay.render(screen)