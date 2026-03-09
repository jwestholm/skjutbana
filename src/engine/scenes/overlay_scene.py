from __future__ import annotations

import pygame

from src.engine.scene import Scene
from src.engine.visual.hit_visualizer import hit_visualizer
from src.engine.input.hit_input import hit_input


class OverlayScene(Scene):

    def __init__(self, inner: Scene):

        self.inner = inner

    def on_enter(self):

        if hasattr(self.inner, "on_enter"):
            self.inner.on_enter()

    def on_exit(self):

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