from __future__ import annotations

import pygame

from src.engine.scene import Scene, SceneSwitch
from src.engine.video_player import VideoPlayer
from src.engine.settings import load_viewport_rect


class VideoScene(Scene):
    def __init__(self, movie_path: str) -> None:
        self.movie_path = movie_path

        self.player: VideoPlayer | None = None
        self.last_frame: pygame.Surface | None = None
        self.viewport = None

        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0

        self.move_step = 20
        self.zoom_step = 0.05
        self.min_zoom = 0.5
        self.max_zoom = 2.0

    def on_enter(self) -> None:
        self.viewport = load_viewport_rect()

        # Reset varje gång man går in i banan
        self.offset_x = 0
        self.offset_y = 0
        self.zoom = 1.0

        # Basframe i viewport-storlek (zoom sker i render för minimal ändring)
        self.player = VideoPlayer(self.movie_path, (self.viewport.w, self.viewport.h))

    def on_exit(self) -> None:
        if self.player:
            self.player.close()

    def _clamp_offset(self) -> None:
        assert self.viewport is not None
        max_x = self.viewport.w
        max_y = self.viewport.h
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
            moved = True
            zoomed = True

        if moved:
            self._clamp_offset()
        if zoomed:
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

        screen.fill((0, 0, 0))

        if not self.last_frame:
            return

        frame = self.last_frame

        # Zooma frame i render (minimalt ingrepp)
        if abs(self.zoom - 1.0) > 1e-6:
            w = max(1, int(self.viewport.w * self.zoom))
            h = max(1, int(self.viewport.h * self.zoom))
            frame = pygame.transform.smoothscale(frame, (w, h))

        x = self.viewport.x + (self.viewport.w - frame.get_width()) // 2 + self.offset_x
        y = self.viewport.y + (self.viewport.h - frame.get_height()) // 2 + self.offset_y

        old_clip = screen.get_clip()
        screen.set_clip(self.viewport)
        screen.blit(frame, (x, y))
        screen.set_clip(old_clip)