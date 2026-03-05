from __future__ import annotations

import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.video_player import VideoPlayer
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


class VideoScene(Scene):
    def __init__(self, movie_path: str, fit: str = "stretch", bg_color=(0, 0, 0)) -> None:
        self.movie_path = movie_path
        self.fit = (fit or "stretch").lower().strip()
        self.bg_color = tuple(bg_color)

        self.player: VideoPlayer | None = None
        self.last_frame: pygame.Surface | None = None
        self.viewport = None

        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0

        self.move_step = 20
        self.zoom_step = 0.05
        self.min_zoom = 0.25
        self.max_zoom = 3.0

    def on_enter(self) -> None:
        self.viewport = load_viewport_rect()

        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0

        # target_size=None => behåll original ratio, skala i render istället
        self.player = VideoPlayer(self.movie_path, target_size=None)

    def on_exit(self) -> None:
        if self.player:
            self.player.close()

    def _clamp_offset(self) -> None:
        assert self.viewport is not None
        max_x = self.viewport.w * 2
        max_y = self.viewport.h * 2
        self.offset_x = max(-max_x, min(self.offset_x, max_x))
        self.offset_y = max(-max_y, min(self.offset_y, max_y))

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_SPACE:
            if self.player:
                self.player.toggle_pause()
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

        if moved or zoomed:
            self._clamp_offset()

        return None

    def update(self, dt: float):
        if not self.player:
            return None

        frame = self.player.update(dt)
        if frame is not None:
            self.last_frame = frame

        if self.player.finished:
            self.player.pause()

        return None

    def render(self, screen: pygame.Surface) -> None:
        assert self.viewport is not None

        screen.fill(self.bg_color)

        if not self.last_frame:
            return

        src_w, src_h = self.last_frame.get_size()
        base_w, base_h = _fit_size(src_w, src_h, self.viewport.w, self.viewport.h, self.fit)

        w = max(1, int(base_w * self.zoom))
        h = max(1, int(base_h * self.zoom))

        frame = self.last_frame
        if (w, h) != frame.get_size():
            frame = pygame.transform.smoothscale(frame, (w, h))

        x = self.viewport.x + (self.viewport.w - frame.get_width()) // 2 + self.offset_x
        y = self.viewport.y + (self.viewport.h - frame.get_height()) // 2 + self.offset_y

        old_clip = screen.get_clip()
        screen.set_clip(self.viewport)
        screen.blit(frame, (x, y))
        screen.set_clip(old_clip)