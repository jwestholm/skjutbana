from __future__ import annotations

import pygame

from src.engine.input.hit_input import hit_input
from src.engine.scene import Scene, SceneSwitch
from src.engine.visual.hit_visualizer import hit_visualizer
from src.engine.visual.scanner_debug_overlay import scanner_debug_overlay
from src.engine.visual.scanner_status_overlay import scanner_status_overlay


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
        if hasattr(self.inner, "on_exit"):
            self.inner.on_exit()

    def _return_to_previous_menu(self):
        menu_state = getattr(self, "return_menu_state", None)
        if menu_state is None:
            menu_state = getattr(self.inner, "return_menu_state", None)
        if menu_state is None:
            return None

        from src.engine.scenes.menu import MenuScene

        return SceneSwitch(MenuScene(menu_state=menu_state))

    def _accepts_mouse_simulated_hits(self) -> bool:
        explicit = getattr(self.inner, "wants_mouse_simulated_hits", None)
        if explicit is not None:
            return bool(explicit)
        return bool(getattr(self.inner, "wants_hit_scanning", False))

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            result = self._return_to_previous_menu()
            if result is not None:
                return result

        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self._accepts_mouse_simulated_hits()
        ):
            hit_input.push_mouse_hit(event.pos[0], event.pos[1])

        return self.inner.handle_event(event)

    def update(self, dt):
        result = self.inner.update(dt)
        return result

    def render(self, screen):
        self.inner.render(screen)
        # Suppress overlays when inner scene requests it (e.g. during calibration)
        if getattr(self.inner, "suppress_overlays", False):
            return
        hit_visualizer.render(screen)
        scanner_debug_overlay.render(screen)
        scanner_status_overlay.render(screen)
