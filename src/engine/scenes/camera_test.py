from __future__ import annotations

import cv2
import numpy as np
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT
from src.engine.camera.camera_manager import camera_manager
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_camera_calibration,
    load_scanport_rect,
    load_viewport_rect,
    save_scanport_rect,
)


WHITE = (240, 240, 240)
SOFT_WHITE = (205, 205, 205)
ORANGE = (255, 165, 0)
GREEN = (80, 255, 120)
RED = (255, 100, 100)
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

    Kontroller:
    - Pilar: flytta scanport
    - +/-: ändra storlek på scanport
    - ENTER: spara
    - ESC: tillbaka utan att spara

    Overlay:
    - Orange ram = scanport
    - Grön streckad polygon = viewport projicerad tillbaka till kamerabilden
    """

    wants_camera_preview = True

    def __init__(self, fit: str = "contain", bg_color=(0, 0, 0)) -> None:
        self.fit = (fit or "contain").lower().strip()
        self.bg_color = tuple(bg_color)

        self.viewport = None
        self.scanport = None
        self.original_scanport = None

        self.last_frame_bgr: np.ndarray | None = None
        self.last_frame_surface: pygame.Surface | None = None
        self.frame_draw_rect: pygame.Rect | None = None

        self.font = None
        self.small = None
        self.tiny = None

        self.move_step = 10
        self.size_step = 20

    def on_enter(self) -> None:
        self.viewport = load_viewport_rect()
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

        # Scanport
        if self.scanport is not None and self.last_frame_bgr is not None:
            scanport_draw = self._camera_rect_to_screen_rect(self.scanport, self.last_frame_bgr.shape[:2])
            pygame.draw.rect(screen, ORANGE, scanport_draw, 3)

        # Viewport projicerad tillbaka till kamerabilden
        viewport_poly = self._get_viewport_polygon_in_camera_screen()
        if viewport_poly:
            self._draw_dashed_polygon(screen, viewport_poly, GREEN, dash_len=12, gap_len=8, width=2)

    def _render_header(self, screen: pygame.Surface) -> None:
        panel = pygame.Surface((SCREEN_WIDTH, 118), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (0, 0))

        title = self.font.render("Justera scanport", True, WHITE)
        screen.blit(title, (28, 18))

        hint = "Pilar: flytta | +/-: storlek | ENTER: spara | ESC: tillbaka"
        hint_surf = self.small.render(hint, True, SOFT_WHITE)
        screen.blit(hint_surf, (28, 62))

        legend = "Orange = scanport | Grön streckad = viewport projicerad till kamerabilden"
        legend_surf = self.tiny.render(legend, True, SOFT_WHITE)
        screen.blit(legend_surf, (28, 90))

    def _render_footer(self, screen: pygame.Surface) -> None:
        panel_h = 92
        panel = pygame.Surface((SCREEN_WIDTH, panel_h), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (0, SCREEN_HEIGHT - panel_h))

        y = SCREEN_HEIGHT - panel_h + 12

        if self.scanport is not None:
            scan_text = (
                f"Scanport: x={self.scanport.x} y={self.scanport.y} "
                f"w={self.scanport.w} h={self.scanport.h}"
            )
        else:
            scan_text = "Scanport: ingen frame ännu"

        scan_surf = self.small.render(scan_text, True, ORANGE)
        screen.blit(scan_surf, (28, y))
        y += 28

        if self.viewport is not None:
            viewport_text = (
                f"Viewport: x={self.viewport.x} y={self.viewport.y} "
                f"w={self.viewport.w} h={self.viewport.h}"
            )
            viewport_surf = self.small.render(viewport_text, True, GREEN)
            screen.blit(viewport_surf, (28, y))

        if self._has_valid_calibration():
            cal_text = "Kalibrering: OK"
            cal_color = GREEN
        else:
            cal_text = "Kalibrering: saknas eller är ogiltig"
            cal_color = RED

        cal_surf = self.small.render(cal_text, True, cal_color)
        cal_x = SCREEN_WIDTH - cal_surf.get_width() - 28
        screen.blit(cal_surf, (cal_x, SCREEN_HEIGHT - panel_h + 12))

    def _has_valid_calibration(self) -> bool:
        calibration = load_camera_calibration()
        if not isinstance(calibration, dict):
            return False

        homography = calibration.get("homography")
        if not isinstance(homography, list):
            return False

        try:
            H = np.array(homography, dtype=np.float32)
            return H.shape == (3, 3)
        except Exception:
            return False

    def _get_viewport_polygon_in_camera_screen(self) -> list[tuple[int, int]] | None:
        if self.last_frame_bgr is None or self.frame_draw_rect is None or self.viewport is None:
            return None

        calibration = load_camera_calibration()
        if not calibration or "homography" not in calibration:
            return None

        try:
            H_camera_to_screen = np.array(calibration["homography"], dtype=np.float32)
            if H_camera_to_screen.shape != (3, 3):
                return None
            H_screen_to_camera = np.linalg.inv(H_camera_to_screen).astype(np.float32)
        except Exception:
            return None

        vx = float(self.viewport.x)
        vy = float(self.viewport.y)
        vw = float(self.viewport.w)
        vh = float(self.viewport.h)

        screen_pts = np.array(
            [[
                [vx, vy],
                [vx + vw, vy],
                [vx + vw, vy + vh],
                [vx, vy + vh],
            ]],
            dtype=np.float32,
        )

        try:
            camera_pts = cv2.perspectiveTransform(screen_pts, H_screen_to_camera)[0]
        except Exception:
            return None

        frame_h, frame_w = self.last_frame_bgr.shape[:2]
        out: list[tuple[int, int]] = []

        for cx, cy in camera_pts:
            sx, sy = self._camera_point_to_screen_point(float(cx), float(cy), frame_w, frame_h)
            out.append((sx, sy))

        return out

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

    def _draw_dashed_polygon(
        self,
        screen: pygame.Surface,
        points: list[tuple[int, int]],
        color: tuple[int, int, int],
        dash_len: int = 10,
        gap_len: int = 6,
        width: int = 2,
    ) -> None:
        if len(points) < 2:
            return

        pts = points + [points[0]]
        for i in range(len(pts) - 1):
            self._draw_dashed_line(screen, pts[i], pts[i + 1], color, dash_len, gap_len, width)

    def _draw_dashed_line(
        self,
        screen: pygame.Surface,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        dash_len: int,
        gap_len: int,
        width: int,
    ) -> None:
        x1, y1 = start
        x2, y2 = end

        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length <= 0.0:
            return

        step = dash_len + gap_len
        count = int(length // step) + 1

        ux = dx / length
        uy = dy / length

        for i in range(count):
            seg_start = i * step
            seg_end = min(seg_start + dash_len, length)

            if seg_start >= length:
                break

            ax = int(round(x1 + ux * seg_start))
            ay = int(round(y1 + uy * seg_start))
            bx = int(round(x1 + ux * seg_end))
            by = int(round(y1 + uy * seg_end))

            pygame.draw.line(screen, color, (ax, ay), (bx, by), width)