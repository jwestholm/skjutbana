from __future__ import annotations

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
YELLOW = (255, 220, 80)
BLACK = (0, 0, 0)
PANEL_BG = (0, 0, 0, 165)

CALIBRATION_MARKER_RADIUS = 28
CALIBRATION_MARKER_THICKNESS = 6


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


def _draw_crosshair(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    center: tuple[int, int],
    radius: int = CALIBRATION_MARKER_RADIUS,
    thickness: int = CALIBRATION_MARKER_THICKNESS,
) -> None:
    x, y = center
    pygame.draw.circle(surface, color, center, radius, thickness)
    pygame.draw.line(surface, color, (x - radius - 10, y), (x + radius + 10, y), thickness)
    pygame.draw.line(surface, color, (x, y - radius - 10), (x, y + radius + 10), thickness)


class CameraTestScene(Scene):
    """
    Scanport-kalibrering + sekventiell hörnkalibrering.

    Kontroller:
    - Pilar = flytta scanport
    - + / - = skala proportionellt
    - SHIFT+B / B = öka / minska bredd
    - SHIFT+H / H = öka / minska höjd
    - ENTER = spara scanport
    - SPACE = starta hörnkalibrering
    - C = radera sparad hörnkalibrering
    - ESC = tillbaka
    """

    def __init__(self, camera_index: int = 0, bg_color=(0, 0, 0)) -> None:
        self.camera_index = camera_index
        self.bg_color = tuple(bg_color)

        self.cap: cv2.VideoCapture | None = None
        self.last_frame: pygame.Surface | None = None
        self.last_frame_rgb: np.ndarray | None = None

        self.error_message: str | None = None
        self.status_message: str = ""

        self.font = None
        self.small = None
        self.tiny = None

        self.viewport: pygame.Rect | None = None
        self.rect: pygame.Rect | None = None

        self.move_step = 10
        self.size_step = 20

        self.calibration_mode = False
        self.calibration_index = 0
        self.calibration_frames_needed = 6
        self.calibration_hits = 0
        self.current_detected_point: tuple[float, float] | None = None
        self.detected_points: np.ndarray | None = None
        self.calibration_points: list[list[float]] = []
        self.calibration_data: dict | None = None

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 26)
        self.tiny = pygame.font.Font(None, 24)

        self.error_message = None
        self.status_message = ""
        self.last_frame = None
        self.last_frame_rgb = None

        self.calibration_mode = False
        self.calibration_index = 0
        self.calibration_hits = 0
        self.current_detected_point = None
        self.calibration_points = []

        self.viewport = load_viewport_rect()
        scanport = load_scanport_norm()
        self.rect = scanport_norm_to_screen_rect(scanport)
        self.calibration_data = load_camera_calibration()

        if self.calibration_data and self.calibration_data.get("camera_points_norm"):
            pts = []
            for x, y in self.calibration_data["camera_points_norm"]:
                pts.append([x * SCREEN_WIDTH, y * SCREEN_HEIGHT])
            self.detected_points = np.array(pts, dtype=np.float32)
        else:
            self.detected_points = None

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap or not self.cap.isOpened():
            self.error_message = f"Kunde inte öppna kamera index {self.camera_index}"
            if self.cap:
                self.cap.release()
            self.cap = None
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if self.calibration_data and self.calibration_data.get("is_calibrated"):
            self.status_message = "Kalibrering hittad."
        else:
            self.status_message = "Ingen hörnkalibrering sparad."

    def on_exit(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    def _start_corner_calibration(self) -> None:
        self.calibration_mode = True
        self.calibration_index = 0
        self.calibration_hits = 0
        self.current_detected_point = None
        self.calibration_points = []
        self.status_message = "Kalibrering startad. Söker hörn 1 av 4..."

    def _abort_corner_calibration(self) -> None:
        self.calibration_mode = False
        self.calibration_index = 0
        self.calibration_hits = 0
        self.current_detected_point = None
        self.calibration_points = []
        self.status_message = "Kalibrering avbruten."

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            if self.calibration_mode:
                self._abort_corner_calibration()
                return None
            return self._go_back()

        if self.rect is None:
            return None

        mods = pygame.key.get_mods()

        if event.key == pygame.K_SPACE:
            if self.cap is not None:
                self._start_corner_calibration()
            return None

        if self.calibration_mode:
            return None

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            x, y, w, h = screen_rect_to_scanport_norm(self.rect)
            save_scanport_norm(x, y, w, h)
            self.status_message = "Scanport sparad."
            return None

        if event.key == pygame.K_c:
            clear_camera_calibration()
            self.calibration_data = None
            self.detected_points = None
            self.status_message = "Sparad hörnkalibrering raderad."
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

    def _read_camera(self) -> None:
        if not self.cap:
            return

        ok, frame_bgr = self.cap.read()
        if not ok:
            self.error_message = "Kunde inte läsa bild från kameran."
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(
            frame_rgb,
            (SCREEN_WIDTH, SCREEN_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        self.last_frame_rgb = frame_rgb
        frame_surface = np.transpose(frame_rgb, (1, 0, 2))
        self.last_frame = pygame.surfarray.make_surface(frame_surface).convert()

    def _get_calibration_marker_position(self) -> tuple[int, int]:
        assert self.viewport is not None

        points = [
            (self.viewport.left, self.viewport.top),
            (self.viewport.right, self.viewport.top),
            (self.viewport.right, self.viewport.bottom),
            (self.viewport.left, self.viewport.bottom),
        ]
        return points[self.calibration_index]

    def _detect_single_marker(self) -> tuple[float, float] | None:
        if self.last_frame_rgb is None or self.rect is None:
            return None

        x1, y1, w, h = self.rect.x, self.rect.y, self.rect.w, self.rect.h
        if w <= 0 or h <= 0:
            return None

        roi = self.last_frame_rgb[y1:y1 + h, x1:x1 + w]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_contour = None
        best_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 60:
                continue
            if area > best_area:
                best_area = area
                best_contour = contour

        if best_contour is None:
            return None

        moments = cv2.moments(best_contour)
        if moments["m00"] == 0:
            return None

        cx = (moments["m10"] / moments["m00"]) + x1
        cy = (moments["m01"] / moments["m00"]) + y1
        return (float(cx), float(cy))

    def _save_detected_calibration(self, camera_points: np.ndarray) -> None:
        assert self.viewport is not None

        viewport_points = _rect_to_corner_points(self.viewport)
        homography, _ = cv2.findHomography(camera_points, viewport_points)
        if homography is None:
            self.status_message = "Kunde inte räkna ut transformeringen."
            return

        self.calibration_data = {
            "is_calibrated": True,
            "camera_points_norm": _normalize_points(camera_points),
            "viewport_points": viewport_points.tolist(),
            "homography": homography.tolist(),
        }
        save_camera_calibration(self.calibration_data)
        self.detected_points = camera_points.copy()
        self.calibration_mode = False
        self.calibration_hits = 0
        self.current_detected_point = None
        self.status_message = "Kalibrering klar och sparad."

    def _update_corner_calibration(self) -> None:
        point = self._detect_single_marker()

        if point is None:
            self.calibration_hits = 0
            self.current_detected_point = None
            return

        self.current_detected_point = point
        self.calibration_hits += 1

        if self.calibration_hits < self.calibration_frames_needed:
            return

        self.calibration_points.append([point[0], point[1]])
        self.calibration_hits = 0
        self.current_detected_point = None

        if self.calibration_index < 3:
            self.calibration_index += 1
            self.status_message = f"Kalibrering: hörn {self.calibration_index + 1} av 4..."
            return

        camera_points = np.array(self.calibration_points, dtype=np.float32)
        self._save_detected_calibration(camera_points)

    def update(self, dt: float):
        self._read_camera()

        if self.calibration_mode and self.last_frame_rgb is not None:
            self._update_corner_calibration()

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
            for idx, point in enumerate(self.detected_points):
                px, py = int(point[0]), int(point[1])
                pygame.draw.circle(screen, WHITE, (px, py), 10, 3)
                label = self.tiny.render(str(idx + 1), True, YELLOW)
                screen.blit(label, (px + 12, py - 10))

    def _draw_info_panel(self, screen: pygame.Surface) -> None:
        panel = pygame.Surface((900, 290), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (20, 20))

        lines = [
            ("Justera scanport och kalibrera kamerahörn", self.font, WHITE),
            ("Orange ram = scanport som analyseras", self.small, WHITE),
            ("Grön streckad ram = viewport / projicerad spelyta", self.small, WHITE),
            ("Pilar: flytta scanport", self.tiny, SOFT_WHITE),
            ("+ / - : öka eller minska proportionellt", self.tiny, SOFT_WHITE),
            ("SHIFT+B / B : öka eller minska bredd", self.tiny, SOFT_WHITE),
            ("SHIFT+H / H : öka eller minska höjd", self.tiny, SOFT_WHITE),
            ("ENTER: spara scanport", self.tiny, SOFT_WHITE),
            ("SPACE: starta automatisk hörnkalibrering", self.tiny, SOFT_WHITE),
            ("C: radera sparad hörnkalibrering", self.tiny, SOFT_WHITE),
            ("ESC: tillbaka till menyn", self.tiny, SOFT_WHITE),
        ]

        y = 32
        for text, font, color in lines:
            surf = font.render(text, True, color)
            screen.blit(surf, (36, y))
            y += surf.get_height() + 5

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

            if self.calibration_mode:
                status_lines.append(
                    (f"Kalibrering pågår: hörn {self.calibration_index + 1} av 4", WHITE)
                )
            elif self.calibration_data and self.calibration_data.get("is_calibrated"):
                status_lines.append(("Status: kalibrerad", WHITE))
            else:
                status_lines.append(("Status: ej kalibrerad", RED))

            if self.status_message:
                status_lines.append((self.status_message, SOFT_WHITE))

        y = SCREEN_HEIGHT - (len(status_lines) * 28) - 12
        for text, color in status_lines:
            surf = self.tiny.render(text, True, color)
            screen.blit(surf, (24, y))
            y += 28

    def _draw_calibration_pattern(self, screen: pygame.Surface) -> None:
        assert self.viewport is not None

        screen.fill(BLACK)

        point = self._get_calibration_marker_position()
        _draw_crosshair(screen, WHITE, point)

        title = self.small.render(
            f"Kalibrering pågår - visar hörn {self.calibration_index + 1} av 4",
            True,
            WHITE,
        )
        screen.blit(title, (24, 24))

        info = self.tiny.render(
            "Håll tavlan stilla tills markeringen registrerats. ESC avbryter.",
            True,
            SOFT_WHITE,
        )
        screen.blit(info, (24, 56))

        if self.current_detected_point is not None:
            px, py = int(self.current_detected_point[0]), int(self.current_detected_point[1])
            pygame.draw.circle(screen, YELLOW, (px, py), 12, 3)

    def render(self, screen: pygame.Surface) -> None:
        if self.calibration_mode:
            self._draw_calibration_pattern(screen)
            return

        screen.fill(self.bg_color)

        if self.last_frame is not None:
            screen.blit(self.last_frame, (0, 0))

        if self.rect is not None and self.viewport is not None:
            self._draw_overlay(screen)

        self._draw_info_panel(screen)
        self._draw_status(screen)