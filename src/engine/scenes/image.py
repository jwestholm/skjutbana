from __future__ import annotations

import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect


class ImageScene(Scene):
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path

        self.viewport = None

        self.original: pygame.Surface | None = None
        self.scaled: pygame.Surface | None = None

        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0

        self.move_step = 20
        self.zoom_step = 0.05
        self.min_zoom = 0.05
        self.max_zoom = 4.0

        self._scaled_size: tuple[int, int] | None = None

    def on_enter(self) -> None:
        self.viewport = load_viewport_rect()

        self.original = pygame.image.load(self.image_path).convert()

        # Reset varje gång man går in i banan
        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0

        self._rebuild_scaled()

    def _rebuild_scaled(self) -> None:
        assert self.viewport is not None
        assert self.original is not None

        w = max(1, int(self.viewport.w * self.zoom))
        h = max(1, int(self.viewport.h * self.zoom))
        size = (w, h)

        if self._scaled_size != size:
            self.scaled = pygame.transform.smoothscale(self.original, size)
            self._scaled_size = size

    def _clamp_offset(self) -> None:
        # Enkel clamp så man inte tappar bort motivet helt
        assert self.viewport is not None
        max_x = self.viewport.w
        max_y = self.viewport.h
        self.offset_x = max(-max_x, min(self.offset_x, max_x))
        self.offset_y = max(-max_y, min(self.offset_y, max_y))

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())

        moved = False
        zoomed = False

        # Pan: WASD + pilar
        if event.key in (pygame.K_LEFT, pygame.K_a):
            self.offset_x -= self.move_step
            moved = True
        elif event.key in (pygame.K_RIGHT, pygame.K_d):
            self.offset_x += self.move_step
            moved = True
        elif event.key in (pygame.K_UP, pygame.K_w):
            self.offset_y -= self.move_step
            moved = True
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.offset_y += self.move_step
            moved = True

        # Zoom: + / -
        elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.zoom = min(self.max_zoom, self.zoom + self.zoom_step)
            zoomed = True
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.zoom = max(self.min_zoom, self.zoom - self.zoom_step)
            zoomed = True

        # Reset
        elif event.key == pygame.K_r:
            self.offset_x = 0
            self.offset_y = 0
            self.zoom = 1.0
            zoomed = True
            moved = True

        if moved:
            self._clamp_offset()

        if zoomed:
            self._rebuild_scaled()
            self._clamp_offset()

        return None

    def render(self, screen: pygame.Surface) -> None:
        assert self.viewport is not None
        assert self.scaled is not None

        # Resten av projektorn svart
        screen.fill((0, 0, 0))

        # Placera scaled centrerat i viewport + offset
        x = self.viewport.x + (self.viewport.w - self.scaled.get_width()) // 2 + self.offset_x
        y = self.viewport.y + (self.viewport.h - self.scaled.get_height()) // 2 + self.offset_y

        # Klippning: rita aldrig utanför viewport
        old_clip = screen.get_clip()
        screen.set_clip(self.viewport)
        screen.blit(self.scaled, (x, y))
        screen.set_clip(old_clip)