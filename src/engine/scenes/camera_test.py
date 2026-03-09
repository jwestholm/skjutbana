from __future__ import annotations

import cv2
import numpy as np
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT
from src.engine.camera.camera_manager import camera_manager
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import load_scanport_rect, save_scanport_rect


WHITE = (240, 240, 240)
SOFT_WHITE = (205, 205, 205)
ORANGE = (255, 165, 0)
PANEL_BG = (0, 0, 0, 140)


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
    """
    Justera scanport ovanpå live-kamerabild.

    Den här scenen ska ENDAST hantera kamerans analysyta.
    Den ska inte försöka visa viewport eller tidigare homography.

    Kontroller:
    - Pilar: flytta scanport
    - +/-: ändra storlek på scanport
    - ENTER: spara
    - ESC: tillbaka utan att spara
    """

    wants_camera_preview = True

    def __init__(self, fit: str = "contain", bg_color=(0, 0, 0)) -> None:
        self.fit = (fit or "contain").lower().strip()
        self.bg_color = tuple(bg_color)

        self.scanport: pygame.Rect | None = None
        self.original_scanport: pygame.Rect | None = None

        self.last_frame_bgr: np.ndarray | None = None
        self.last_frame_surface: pygame.Surface | None = None
        self.frame_draw_rect: pygame.Rect | None = None

        self.font = None
        self.small = None
        self.tiny = None

        self.move_step = 10
        self.size_step = 20

    def on_enter(self) -> None:
        self.original_scanport = load_scanport_rect()
        self.scanport = self.original_scanport.copy() if self.original_scanport else None

        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 26)
        self.tiny = pygame.font.Font(None, 22)

        camera_manager.start()

    def on_exit(self) -> None:
        pass

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())

        if self.scanport is None:
            return None

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            save_scanport_rect(self.scanport)
            from src.engine.scenes.menu import MenuScene
            return SceneSwitch(MenuScene())

        if event.key == pygame.K_LEFT:
            self.scanport.x -= self.move_step
        elif event.key == pygame.K_RIGHT:
            self.scanport.x += self.move_step
        elif event.key == pygame.K_UP:
            self.scanport.y -= self.move_step
        elif event.key == pygame.K_DOWN:
            self.scanport.y += self.move_step
        elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.scanport.w += self.size_step
            self.scanport.h += self.size_step
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.scanport.w -= self.size_step
            self.scanport.h -= self.size_step

        self._clamp_scanport_to_frame()
        return None

    def update(self, dt: float):
        del dt

        frame_bgr = camera_manager.get_latest_frame()
        if frame_bgr is None:
            self.last_frame_bgr = None
            self.last_frame_surface = None
            self.frame_draw_rect = None
            return None

        self.last_frame_bgr = frame_bgr

        if self.scanport is None:
            self.scanport = self._default_scanport_for_frame(frame_bgr)

        self._clamp_scanport_to_frame()
        self.last_frame_surface = self._frame_to_surface(frame_bgr)
        return None

    def _default_scanport_for_frame(self, frame_bgr: np.ndarray) -> pygame.Rect:
        h, w = frame_bgr.shape[:2]
        margin_x = max(40, int(w * 0.08))
        margin_y = max(30, int(h * 0.08))
        return pygame.Rect(
            margin_x,
            margin_y,
            max(100, w - margin_x * 2),
            max(100, h - margin_y * 2),
        )

    def _clamp_scanport_to_frame(self) -> None:
        if self.scanport is None or self.last_frame_bgr is None:
            return

        frame_h, frame_w = self.last_frame_bgr.shape[:2]

        self.scanport.w = max(50, min(self.scanport.w, frame_w))
        self.scanport.h = max(50, min(self.scanport.h, frame_h))

        self.scanport.x = max(0, min(self.scanport.x, frame_w - self.scanport.w))
        self.scanport.y = max(0, min(self.scanport.y, frame_h - self.scanport.h))

    def _frame_to_surface(self, frame_bgr: np.ndarray) -> pygame.Surface:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_rgb)
        return surf.convert()

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        if self.last_frame_surface is not None:
            self._render_camera_preview(screen)

        self._render_header(screen)
        self._render_footer(screen)

    def _render_camera_preview(self, screen: pygame.Surface) -> None:
        assert self.last_frame_surface is not None

        src_w, src_h = self.last_frame_surface.get_size()
        draw_w, draw_h = _fit_size(src_w, src_h, SCREEN_WIDTH, SCREEN_HEIGHT, self.fit)

        frame = self.last_frame_surface
        if frame.get_size() != (draw_w, draw_h):
            frame = pygame.transform.smoothscale(frame, (draw_w, draw_h))

        x = (SCREEN_WIDTH - draw_w) // 2
        y = (SCREEN_HEIGHT - draw_h) // 2

        self.frame_draw_rect = pygame.Rect(x, y, draw_w, draw_h)
        screen.blit(frame, (x, y))

        if self.scanport is not None and self.last_frame_bgr is not None:
            scanport_draw = self._camera_rect_to_screen_rect(
                self.scanport,
                self.last_frame_bgr.shape[:2],
            )
            pygame.draw.rect(screen, ORANGE, scanport_draw, 3)

    def _render_header(self, screen: pygame.Surface) -> None:
        panel = pygame.Surface((SCREEN_WIDTH, 110), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (0, 0))

        title = self.font.render("Justera scanport", True, WHITE)
        screen.blit(title, (28, 18))

        hint = "Pilar: flytta | +/-: storlek | ENTER: spara | ESC: tillbaka"
        hint_surf = self.small.render(hint, True, SOFT_WHITE)
        screen.blit(hint_surf, (28, 62))

        legend = "Orange ram = kamerans analysyta"
        legend_surf = self.tiny.render(legend, True, SOFT_WHITE)
        screen.blit(legend_surf, (28, 90))

    def _render_footer(self, screen: pygame.Surface) -> None:
        panel_h = 70
        panel = pygame.Surface((SCREEN_WIDTH, panel_h), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (0, SCREEN_HEIGHT - panel_h))

        y = SCREEN_HEIGHT - panel_h + 18

        if self.scanport is not None:
            scan_text = (
                f"Scanport: x={self.scanport.x} y={self.scanport.y} "
                f"w={self.scanport.w} h={self.scanport.h}"
            )
        else:
            scan_text = "Scanport: ingen frame ännu"

        scan_surf = self.small.render(scan_text, True, ORANGE)
        screen.blit(scan_surf, (28, y))

    def _camera_rect_to_screen_rect(
        self,
        rect: pygame.Rect,
        frame_shape_hw: tuple[int, int],
    ) -> pygame.Rect:
        frame_h, frame_w = frame_shape_hw
        left, top = self._camera_point_to_screen_point(rect.x, rect.y, frame_w, frame_h)
        right, bottom = self._camera_point_to_screen_point(rect.right, rect.bottom, frame_w, frame_h)

        x = min(left, right)
        y = min(top, bottom)
        w = abs(right - left)
        h = abs(bottom - top)
        return pygame.Rect(x, y, w, h)

    def _camera_point_to_screen_point(
        self,
        cx: float,
        cy: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int]:
        assert self.frame_draw_rect is not None

        sx = self.frame_draw_rect.x + int(round((cx / frame_w) * self.frame_draw_rect.w))
        sy = self.frame_draw_rect.y + int(round((cy / frame_h) * self.frame_draw_rect.h))
        return sx, sy