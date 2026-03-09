from __future__ import annotations

import cv2
import numpy as np
import pygame

from src.engine.camera.camera_manager import camera_manager
from src.engine.camera.hit_scanner import hit_scanner
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


class CameraTestScene(Scene):
    wants_camera_preview = True

    def __init__(self, fit: str = "contain", bg_color=(0, 0, 0)) -> None:
        self.fit = (fit or "contain").lower().strip()
        self.bg_color = tuple(bg_color)

        self.viewport = None
        self.last_frame: pygame.Surface | None = None
        self.font = None
        self.small = None

    def on_enter(self) -> None:
        self.viewport = load_viewport_rect()
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 24)

        camera_manager.start()

    def on_exit(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())

        return None

    def update(self, dt: float):
        del dt

        frame_bgr = camera_manager.get_latest_frame()
        if frame_bgr is None:
            self.last_frame = None
            return None

        self.last_frame = self._frame_to_surface(frame_bgr)
        return None

    def _frame_to_surface(self, frame_bgr: np.ndarray) -> pygame.Surface:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_rgb)
        return surf.convert()

    def render(self, screen: pygame.Surface) -> None:
        assert self.viewport is not None

        screen.fill(self.bg_color)

        if self.last_frame is not None:
            src_w, src_h = self.last_frame.get_size()
            draw_w, draw_h = _fit_size(src_w, src_h, self.viewport.w, self.viewport.h, self.fit)

            frame = self.last_frame
            if frame.get_size() != (draw_w, draw_h):
                frame = pygame.transform.smoothscale(frame, (draw_w, draw_h))

            x = self.viewport.x + (self.viewport.w - frame.get_width()) // 2
            y = self.viewport.y + (self.viewport.h - frame.get_height()) // 2

            old_clip = screen.get_clip()
            screen.set_clip(self.viewport)
            screen.blit(frame, (x, y))
            screen.set_clip(old_clip)

        title = self.font.render("Kontrollera kamera", True, (240, 240, 240))
        screen.blit(title, (30, 20))

        hint = self.small.render("ESC: tillbaka till menyn", True, (200, 200, 200))
        screen.blit(hint, (30, 60))

        y = 95
        for line in camera_manager.get_status_lines():
            txt = self.small.render(line, True, (220, 220, 220))
            screen.blit(txt, (30, y))
            y += 22

        for line in hit_scanner.get_status_lines():
            txt = self.small.render(line, True, (180, 220, 180))
            screen.blit(txt, (30, y))
            y += 22