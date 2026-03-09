from __future__ import annotations

import pygame

from src.engine.scene import Scene
from src.engine.visual.hit_visualizer import hit_visualizer


class OverlayScene(Scene):
    def __init__(self, inner: Scene) -> None:
        self.inner = inner

    def on_enter(self) -> None:
        hit_visualizer.reload_settings()
        hit_visualizer.clear()
        if hasattr(self.inner, "on_enter"):
            self.inner.on_enter()

    def on_exit(self) -> None:
        if hasattr(self.inner, "on_exit"):
            self.inner.on_exit()
        hit_visualizer.clear()

    def handle_event(self, event: pygame.event.Event):
        return self.inner.handle_event(event)

    def update(self, dt: float):
        result = self.inner.update(dt)
        hit_visualizer.update(dt)
        return result

    def render(self, screen: pygame.Surface) -> None:
        self.inner.render(screen)
        hit_visualizer.render(screen)