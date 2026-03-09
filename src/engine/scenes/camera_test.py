from __future__ import annotations

import math

import cv2
import numpy as np
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    clear_camera_calibration,
    load_camera_calibration,
    load_scanport_norm,
    load_viewport_rect,
    save_camera_calibration,
    save_scanport_norm,
    scanport_norm_to_screen_rect,
    screen_rect_to_scanport_norm,
)

ORANGE = (255, 140, 0)
GREEN = (0, 255, 0)
WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
RED = (255, 120, 120)
BLACK = (0, 0, 0)
PANEL_BG = (0, 0, 0, 165)

POINT_COLORS = [
    (255, 80, 80),    # 1 - uppe vänster
    (80, 180, 255),   # 2 - uppe höger
    (255, 220, 80),   # 3 - nere höger
    (180, 100, 255),  # 4 - nere vänster
]

POINT_LABELS = ["1", "2", "3", "4"]
POINT_RADIUS = 12
POINT_HIT_RADIUS = 20

CROSS_COLOR = (255, 255, 255)
CENTER_COLOR = (180, 255, 180)
CROSS_RADIUS = 18
CROSS_THICKNESS = 4
CROSS_ARM = 28


def _draw_dashed_line(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    width: int = 2,
    dash: int = 12,
    gap: int = 8,
) -> None:
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1
    length = max(1, int((dx * dx + dy * dy) ** 0.5))

    for i in range(0, length, dash + gap):
        a = i / length
        b = min(i + dash, length) / length

        sx = int(x1 + dx * a)
        sy = int(y1 + dy * a)
        ex = int(x1 + dx * b)
        ey = int(y1 + dy * b)

        pygame.draw.line(surface, color, (sx, sy), (ex, ey), width)


def _draw_dashed_rect(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    rect: pygame.Rect,
    width: int = 2,
    dash: int = 12,
    gap: int = 8,
) -> None:
    _draw_dashed_line(surface, color, rect.topleft, rect.topright, width, dash, gap)
    _draw_dashed_line(surface, color, rect.topright, rect.bottomright, width, dash, gap)
    _draw_dashed_line(surface, color, rect.bottomright, rect.bottomleft, width, dash, gap)
    _draw_dashed_line(surface, color, rect.bottomleft, rect.topleft, width, dash, gap)


def _draw_dashed_polygon(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    points: list[tuple[int, int]],
    width: int = 2,
    dash: int = 12,
    gap: int = 8,
) -> None:
    if len(points) < 2:
        return

    for i in range(len(points)):
        a = points[i]
        b = points[(i + 1) % len(points)]
        _draw_dashed_line(surface, color, a, b, width, dash, gap)


def _draw_crosshair(
    surface: pygame.Surface,
    center: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = CROSS_RADIUS,
    arm: int = CROSS_ARM,
    thickness: int = CROSS_THICKNESS,
) -> None:
    x, y = center
    pygame.draw.circle(surface, color, center, radius, thickness)
    pygame.draw.line(surface, color, (x - arm, y), (x + arm, y), thickness)
    pygame.draw.line(surface, color, (x, y - arm), (x, y + arm), thickness)


def _clamp_rect_to_screen(rect: pygame.Rect) -> pygame.Rect:
    min_w, min_h = 100, 100
    rect.w = max(min_w, min(rect.w, SCREEN_WIDTH))
    rect.h = max(min_h, min(rect.h, SCREEN_HEIGHT))
    rect.x = max(0, min(rect.x, SCREEN_WIDTH - rect.w))
    rect.y = max(0, min(rect.y, SCREEN_HEIGHT - rect.h))
    return rect


def _rect_to_corner_points(rect: pygame.Rect) -> np.ndarray:
    return np.array(
        [
            [float(rect.left), float(rect.top)],
            [float(rect.right), float(rect.top)],
            [float(rect.right), float(rect.bottom)],
            [float(rect.left), float(rect.bottom)],
        ],
        dtype=np.float32,
    )


def _normalize_points(points: np.ndarray) -> list[list[float]]:
    out: list[list[float]] = []
    for x, y in points:
        out.append([float(x) / max(1, SCREEN_WIDTH), float(y) / max(1, SCREEN_HEIGHT)])
    return out


def _clamp_point_to_screen(x: float, y: float) -> tuple[float, float]:
    x = max(0.0, min(float(x), float(SCREEN_WIDTH - 1)))
    y = max(0.0, min(float(y), float(SCREEN_HEIGHT - 1)))
    return x, y


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class CameraTestScene(Scene):
    """
    Scanport-kalibrering + manuell hörnjustering med mus.

    Kontroller:
    - Pilar = flytta scanport
    - + / - = skala proportionellt
    - SHIFT+B / B = öka / minska bredd
    - SHIFT+H / H = öka / minska höjd
    - Mus vänsterknapp = dra färgade hörnpunkter
    - SPACE = visa / dölj kalibreringsmönster
    - TAB = växla mellan kameravy och kalibreringsmönster
    - R = återställ hörnpunkter till scanportens hörn
    - ENTER = spara scanport + hörnkalibrering
    - C = radera sparad hörnkalibrering
    - ESC = tillbaka
    """

    def __init__(self, camera_index: int = 0, bg_color=(0, 0, 0)) -> None:
        self.camera_index = camera_index
        self.bg_color = tuple(bg_color)

        self.cap: cv2.VideoCapture | None = None
        self.last_frame: pygame.Surface | None = None
        self.error_message: str | None = None
        self.status_message: str = ""

        self.font = None
        self.small = None
        self.tiny = None

        self.viewport: pygame.Rect | None = None
        self.rect: pygame.Rect | None = None
        self.detected_points: np.ndarray | None = None
        self.calibration_data: dict | None = None

        self.move_step = 10
        self.size_step = 20

        self.dragging_index: int | None = None
        self.show_pattern = False

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 26)
        self.tiny = pygame.font.Font(None, 24)

        self.error_message = None
        self.status_message = ""
        self.last_frame = None
        self.dragging_index = None
        self.show_pattern = False

        self.viewport = load_viewport_rect()
        scanport = load_scanport_norm()
        self.rect = scanport_norm_to_screen_rect(scanport)
        self.calibration_data = load_camera_calibration()

        if self.calibration_data and self.calibration_data.get("camera_points_norm"):
            pts = []
            for x, y in self.calibration_data["camera_points_norm"]:
                pts.append([x * SCREEN_WIDTH, y * SCREEN_HEIGHT])
            self.detected_points = np.array(pts, dtype=np.float32)
            self.status_message = "Kalibrering hittad. Dra punkterna med musen vid behov."
        else:
            self.detected_points = _rect_to_corner_points(self.rect)
            self.status_message = "Ingen hörnkalibrering sparad. Punkterna startar i scanportens hörn."

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap or not self.cap.isOpened():
            self.error_message = f"Kunde inte öppna kamera index {self.camera_index}"
            if self.cap:
                self.cap.release()
            self.cap = None
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def on_exit(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    def _reset_points_from_scanport(self) -> None:
        assert self.rect is not None
        self.detected_points = _rect_to_corner_points(self.rect)
        self.status_message = "Hörnpunkter återställda till scanportens hörn."

    def _save_calibration(self) -> None:
        assert self.viewport is not None
        assert self.rect is not None
        assert self.detected_points is not None

        x, y, w, h = screen_rect_to_scanport_norm(self.rect)
        save_scanport_norm(x, y, w, h)

        viewport_points = _rect_to_corner_points(self.viewport)
        homography, _ = cv2.findHomography(self.detected_points, viewport_points)

        if homography is None:
            self.status_message = "Kunde inte räkna ut transformeringen."
            return

        self.calibration_data = {
            "is_calibrated": True,
            "camera_points_norm": _normalize_points(self.detected_points),
            "viewport_points": viewport_points.tolist(),
            "homography": homography.tolist(),
        }
        save_camera_calibration(self.calibration_data)
        self.status_message = "Scanport och hörnkalibrering sparade."

    def _find_point_at_mouse(self, mouse_pos: tuple[int, int]) -> int | None:
        if self.detected_points is None:
            return None

        mx, my = mouse_pos
        best_index = None
        best_distance = float("inf")

        for idx, point in enumerate(self.detected_points):
            px, py = float(point[0]), float(point[1])
            d = _distance((mx, my), (px, py))
            if d <= POINT_HIT_RADIUS and d < best_distance:
                best_index = idx
                best_distance = d

        return best_index

    def _get_pattern_points(self) -> list[tuple[int, int]]:
        assert self.viewport is not None
        cx = self.viewport.centerx
        cy = self.viewport.centery

        return [
            (self.viewport.left, self.viewport.top),
            (self.viewport.right, self.viewport.top),
            (self.viewport.right, self.viewport.bottom),
            (self.viewport.left, self.viewport.bottom),
            (cx, cy),
        ]

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return self._go_back()

            if self.rect is None:
                return None

            mods = pygame.key.get_mods()

            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._save_calibration()
                return None

            if event.key == pygame.K_c:
                clear_camera_calibration()
                self.calibration_data = None
                self._reset_points_from_scanport()
                self.status_message = "Sparad hörnkalibrering raderad."
                return None

            if event.key == pygame.K_r:
                self._reset_points_from_scanport()
                return None

            if event.key == pygame.K_SPACE or event.key == pygame.K_TAB:
                self.show_pattern = not self.show_pattern
                if self.show_pattern:
                    self.status_message = "Kalibreringsmönster visas. Dra punkterna till mitten av de projicerade hörnkryssen."
                else:
                    self.status_message = "Tillbaka i kameravy."
                return None

            if event.key == pygame.K_LEFT:
                self.rect.x -= self.move_step
            elif event.key == pygame.K_RIGHT:
                self.rect.x += self.move_step
            elif event.key == pygame.K_UP:
                self.rect.y -= self.move_step
            elif event.key == pygame.K_DOWN:
                self.rect.y += self.move_step
            elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                self.rect.inflate_ip(self.size_step, self.size_step)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.rect.inflate_ip(-self.size_step, -self.size_step)
            elif event.key == pygame.K_b:
                if mods & pygame.KMOD_SHIFT:
                    self.rect.w += self.size_step
                else:
                    self.rect.w -= self.size_step
            elif event.key == pygame.K_h:
                if mods & pygame.KMOD_SHIFT:
                    self.rect.h += self.size_step
                else:
                    self.rect.h -= self.size_step

            self.rect = _clamp_rect_to_screen(self.rect)
            return None

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            idx = self._find_point_at_mouse(event.pos)
            if idx is not None:
                self.dragging_index = idx
                return None

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging_index = None
            return None

        if event.type == pygame.MOUSEMOTION and self.dragging_index is not None:
            if self.detected_points is None:
                return None

            x, y = _clamp_point_to_screen(event.pos[0], event.pos[1])
            self.detected_points[self.dragging_index][0] = x
            self.detected_points[self.dragging_index][1] = y
            return None

        return None

    def update(self, dt: float):
        if self.show_pattern:
            return None

        if not self.cap:
            return None

        ok, frame_bgr = self.cap.read()
        if not ok:
            self.error_message = "Kunde inte läsa bild från kameran."
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(
            frame_rgb,
            (SCREEN_WIDTH, SCREEN_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        frame_surface = np.transpose(frame_rgb, (1, 0, 2))
        self.last_frame = pygame.surfarray.make_surface(frame_surface).convert()
        return None

    def _draw_overlay(self, screen: pygame.Surface) -> None:
        assert self.rect is not None
        assert self.viewport is not None

        shade = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 120))
        pygame.draw.rect(shade, (0, 0, 0, 0), self.rect)
        screen.blit(shade, (0, 0))

        pygame.draw.rect(screen, ORANGE, self.rect, width=4)
        _draw_dashed_rect(screen, GREEN, self.viewport, width=2, dash=12, gap=8)

        if self.detected_points is not None:
            polygon_points: list[tuple[int, int]] = []
            for point in self.detected_points:
                polygon_points.append((int(point[0]), int(point[1])))

            _draw_dashed_polygon(screen, WHITE, polygon_points, width=2, dash=10, gap=6)

            for idx, point in enumerate(self.detected_points):
                px, py = int(point[0]), int(point[1])
                color = POINT_COLORS[idx]

                pygame.draw.circle(screen, color, (px, py), POINT_RADIUS, 4)
                pygame.draw.circle(screen, WHITE, (px, py), POINT_RADIUS + 4, 1)

                label = self.tiny.render(POINT_LABELS[idx], True, color)
                screen.blit(label, (px + 14, py - 12))

    def _draw_info_panel(self, screen: pygame.Surface) -> None:
        panel = pygame.Surface((980, 340), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (20, 20))

        lines = [
            ("Justera scanport och kalibrera kamerahörn", self.font, WHITE),
            ("Orange ram = scanport som analyseras", self.small, WHITE),
            ("Grön streckad ram = viewport / projicerad spelyta", self.small, WHITE),
            ("Färgade ringar 1-4 = kamerahörn som du drar till hörnkryssen", self.small, WHITE),
            ("Vit streckad linje = koppling mellan hörnpunkterna", self.tiny, SOFT_WHITE),
            ("Pilar: flytta scanport", self.tiny, SOFT_WHITE),
            ("+ / - : öka eller minska proportionellt", self.tiny, SOFT_WHITE),
            ("SHIFT+B / B : öka eller minska bredd", self.tiny, SOFT_WHITE),
            ("SHIFT+H / H : öka eller minska höjd", self.tiny, SOFT_WHITE),
            ("Mus vänsterknapp: klicka och dra en hörnpunkt", self.tiny, SOFT_WHITE),
            ("SPACE eller TAB: visa / dölj kalibreringsmönster", self.tiny, SOFT_WHITE),
            ("R: återställ hörnpunkter till scanportens hörn", self.tiny, SOFT_WHITE),
            ("ENTER: spara scanport och hörnkalibrering", self.tiny, SOFT_WHITE),
            ("C: radera sparad hörnkalibrering", self.tiny, SOFT_WHITE),
            ("ESC: tillbaka till menyn", self.tiny, SOFT_WHITE),
        ]

        y = 32
        for text, font, color in lines:
            surf = font.render(text, True, color)
            screen.blit(surf, (36, y))
            y += surf.get_height() + 4

    def _draw_status(self, screen: pygame.Surface) -> None:
        status_lines: list[tuple[str, tuple[int, int, int]]] = []

        if self.error_message:
            status_lines.append((self.error_message, RED))
        else:
            if self.rect is not None:
                status_lines.append(
                    (
                        f"Scanport px: x={self.rect.x} y={self.rect.y} w={self.rect.w} h={self.rect.h}",
                        WHITE,
                    )
                )

            if self.calibration_data and self.calibration_data.get("is_calibrated"):
                status_lines.append(("Status: kalibrerad", WHITE))
            else:
                status_lines.append(("Status: ej kalibrerad", RED))

            status_lines.append(
                ("Hörnordning: 1=uppe vänster, 2=uppe höger, 3=nere höger, 4=nere vänster", SOFT_WHITE)
            )

            if self.show_pattern:
                status_lines.append(
                    ("Kalibreringsmönster visas: dra varje färgad punkt till mitten av motsvarande projicerat kryss.", SOFT_WHITE)
                )

            if self.status_message:
                status_lines.append((self.status_message, SOFT_WHITE))

        y = SCREEN_HEIGHT - (len(status_lines) * 28) - 12
        for text, color in status_lines:
            surf = self.tiny.render(text, True, color)
            screen.blit(surf, (24, y))
            y += 28

    def _draw_pattern_view(self, screen: pygame.Surface) -> None:
        assert self.viewport is not None
        screen.fill(BLACK)

        points = self._get_pattern_points()
        for idx, point in enumerate(points):
            color = CROSS_COLOR if idx < 4 else CENTER_COLOR
            _draw_crosshair(screen, point, color)

        title = self.small.render(
            "Kalibreringsmönster - dra hörnpunkterna till mitten av de fyra hörnkryssen",
            True,
            WHITE,
        )
        screen.blit(title, (24, 24))

        info = self.tiny.render(
            "SPACE/TAB: tillbaka till kameravy    ENTER: spara    ESC: meny",
            True,
            SOFT_WHITE,
        )
        screen.blit(info, (24, 56))

    def render(self, screen: pygame.Surface) -> None:
        if self.show_pattern:
            self._draw_pattern_view(screen)
            return

        screen.fill(self.bg_color)

        if self.last_frame is not None:
            screen.blit(self.last_frame, (0, 0))

        if self.rect is not None and self.viewport is not None:
            self._draw_overlay(screen)

        self._draw_info_panel(screen)
        self._draw_status(screen)