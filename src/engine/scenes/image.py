from __future__ import annotations

import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_viewport_rect


def _fit_size(src_w: int, src_h: int, dst_w: int, dst_h: int, mode: str) -> tuple[int, int]:
    if mode == "stretch":
        return (dst_w, dst_h)

    if src_w <= 0 or src_h <= 0:
        return (dst_w, dst_h)

    sx = dst_w / src_w
    sy = dst_h / src_h

    if mode == "contain":
        s = min(sx, sy)
    elif mode == "cover":
        s = max(sx, sy)
    else:
        return (dst_w, dst_h)

    w = max(1, int(src_w * s))
    h = max(1, int(src_h * s))
    return (w, h)


class ImageScene(Scene):
    def __init__(self, image_path: str, fit: str = "stretch", bg_color=(0, 0, 0)) -> None:
        self.image_path = image_path
        self.fit = (fit or "stretch").lower().strip()
        self.bg_color = tuple(bg_color)

        self.viewport = None

        self.original: pygame.Surface | None = None
        self.scaled: pygame.Surface | None = None
        self._scaled_size: tuple[int, int] | None = None

        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0

        self.move_step = 20
        self.zoom_step = 0.05
        self.min_zoom = 0.25
        self.max_zoom = 3.0

    def on_enter(self) -> None:
        self.viewport = load_viewport_rect()
        self.original = pygame.image.load(self.image_path).convert()

        # reset varje gång man går in
        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0

        self._rebuild_scaled()

    def _rebuild_scaled(self) -> None:
        assert self.viewport is not None
        assert self.original is not None

        base_w, base_h = _fit_size(
            self.original.get_width(),
            self.original.get_height(),
            self.viewport.w,
            self.viewport.h,
            self.fit,
        )

        w = max(1, int(base_w * self.zoom))
        h = max(1, int(base_h * self.zoom))
        size = (w, h)

        if self._scaled_size != size:
            self.scaled = pygame.transform.smoothscale(self.original, size)
            self._scaled_size = size

    def _clamp_offset(self) -> None:
        assert self.viewport is not None
        # generös clamp så man kan flytta runt för att slita på olika delar av tavlan
        max_x = self.viewport.w * 2
        max_y = self.viewport.h * 2
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

        # pan: WASD + pilar
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

        # zoom: +/-
        elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.zoom = min(self.max_zoom, self.zoom + self.zoom_step)
            zoomed = True
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.zoom = max(self.min_zoom, self.zoom - self.zoom_step)
            zoomed = True

        # reset
        elif event.key == pygame.K_r:
            self.offset_x = 0
            self.offset_y = 0
            self.zoom = 1.0
            moved = True
            zoomed = True

        if moved:
            self._clamp_offset()
        if zoomed:
            self._rebuild_scaled()
            self._clamp_offset()

        return None

    def render(self, screen: pygame.Surface) -> None:
        assert self.viewport is not None
        assert self.scaled is not None

        screen.fill(self.bg_color)

        x = self.viewport.x + (self.viewport.w - self.scaled.get_width()) // 2 + self.offset_x
        y = self.viewport.y + (self.viewport.h - self.scaled.get_height()) // 2 + self.offset_y

        old_clip = screen.get_clip()
        screen.set_clip(self.viewport)
        screen.blit(self.scaled, (x, y))
        screen.set_clip(old_clip)