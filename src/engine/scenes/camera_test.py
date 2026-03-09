from __future__ import annotations

import math
from dataclasses import dataclass

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
CYAN = (100, 220, 255)
YELLOW = (255, 220, 80)

SCANPORT_SHADE_ALPHA = 120

MARKER_RADIUS = 22
MARKER_THICKNESS = 5
MARKER_ARM = 34

# Hur många update-cykler vi väntar i varje steg
PHASE_SETTLE_FRAMES = 6
PHASE_CAPTURE_FRAMES = 3

# Trösklar för differensdetektion
DIFF_THRESHOLD = 35
MIN_BLOB_AREA = 40


@dataclass
class CalibrationState:
    active: bool = False
    corner_index: int = 0          # 0=TL,1=TR,2=BR,3=BL
    phase: str = "off_settle"      # off_settle, off_capture, on_settle, on_capture, done
    frame_counter: int = 0
    off_frame: np.ndarray | None = None
    on_frame: np.ndarray | None = None


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


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _to_int_pos(p: tuple[float, float]) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _expected_corner_positions(rect: pygame.Rect) -> list[tuple[float, float]]:
    return [
        (float(rect.left), float(rect.top)),
        (float(rect.right), float(rect.top)),
        (float(rect.right), float(rect.bottom)),
        (float(rect.left), float(rect.bottom)),
    ]


def _draw_crosshair(
    surface: pygame.Surface,
    center: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = MARKER_RADIUS,
    arm: int = MARKER_ARM,
    thickness: int = MARKER_THICKNESS,
) -> None:
    x, y = center
    pygame.draw.circle(surface, color, center, radius, thickness)
    pygame.draw.line(surface, color, (x - arm, y), (x + arm, y), thickness)
    pygame.draw.line(surface, color, (x, y - arm), (x, y + arm), thickness)


class CameraTestScene(Scene):
    """
    Scanport-kalibrering + automatisk hörnkalibrering via bilddifferens.

    Kontroller:
    - Pilar = flytta scanport
    - + / - = skala proportionellt
    - SHIFT+B / B = öka / minska bredd
    - SHIFT+H / H = öka / minska höjd
    - ENTER = spara scanport
    - SPACE = starta / avbryt automatisk kalibrering
    - C = radera sparad kalibrering
    - ESC = tillbaka till menyn
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
        self.calibration_data: dict | None = None

        self.move_step = 10
        self.size_step = 20

        self.calibration = CalibrationState()
        self.detected_points: list[tuple[float, float]] = []
        self.last_diff_debug: np.ndarray | None = None

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 26)
        self.tiny = pygame.font.Font(None, 24)

        self.error_message = None
        self.status_message = ""
        self.last_frame = None
        self.last_frame_rgb = None
        self.calibration = CalibrationState()
        self.detected_points = []
        self.last_diff_debug = None

        self.viewport = load_viewport_rect()
        scanport = load_scanport_norm()
        self.rect = scanport_norm_to_screen_rect(scanport)
        self.calibration_data = load_camera_calibration()

        if self.calibration_data and self.calibration_data.get("camera_points_norm"):
            pts = []
            for x, y in self.calibration_data["camera_points_norm"]:
                pts.append((x * SCREEN_WIDTH, y * SCREEN_HEIGHT))
            self.detected_points = pts
            self.status_message = "Kalibrering hittad."
        else:
            self.status_message = "Ingen hörnkalibrering sparad."

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

    def _save_scanport_only(self) -> None:
        assert self.rect is not None
        x, y, w, h = screen_rect_to_scanport_norm(self.rect)
        save_scanport_norm(x, y, w, h)
        self.status_message = "Scanport sparad."

    def _save_calibration(self, camera_points: np.ndarray) -> None:
        assert self.viewport is not None
        assert self.rect is not None

        x, y, w, h = screen_rect_to_scanport_norm(self.rect)
        save_scanport_norm(x, y, w, h)

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
            "method": "difference_blink",
        }
        save_camera_calibration(self.calibration_data)
        self.detected_points = [(float(x), float(y)) for x, y in camera_points]
        self.status_message = "Kalibrering klar och sparad."

    def _start_calibration(self) -> None:
        self.calibration = CalibrationState(active=True, corner_index=0, phase="off_settle", frame_counter=0)
        self.detected_points = []
        self.last_diff_debug = None
        self.status_message = "Kalibrering startad."

    def _abort_calibration(self) -> None:
        self.calibration = CalibrationState()
        self.last_diff_debug = None
        self.status_message = "Kalibrering avbruten."

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            return self._go_back()

        if self.rect is None:
            return None

        mods = pygame.key.get_mods()

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._save_scanport_only()
            return None

        if event.key == pygame.K_c:
            clear_camera_calibration()
            self.calibration_data = None
            self.detected_points = []
            self.status_message = "Sparad kalibrering raderad."
            return None

        if event.key == pygame.K_SPACE:
            if self.calibration.active:
                self._abort_calibration()
            else:
                self._start_calibration()
            return None

        if self.calibration.active:
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

    def _current_corner_target(self) -> tuple[float, float]:
        assert self.viewport is not None
        return _expected_corner_positions(self.viewport)[self.calibration.corner_index]

    def _diff_find_marker(self, off_frame: np.ndarray, on_frame: np.ndarray) -> tuple[float, float] | None:
        assert self.rect is not None
        target = self._current_corner_target()

        diff = cv2.absdiff(on_frame, off_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

        x, y, w, h = self.rect.x, self.rect.y, self.rect.w, self.rect.h
        roi = diff_gray[y:y + h, x:x + w]
        if roi.size == 0:
            return None

        _, thresh = cv2.threshold(roi, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        self.last_diff_debug = thresh.copy()

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, tuple[float, float]]] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_BLOB_AREA:
                continue

            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue

            cx = (m["m10"] / m["m00"]) + x
            cy = (m["m01"] / m["m00"]) + y

            d = _distance((cx, cy), target)
            candidates.append((d, (float(cx), float(cy))))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _advance_phase(self) -> None:
        c = self.calibration

        if c.phase == "off_settle":
            c.phase = "off_capture"
        elif c.phase == "off_capture":
            c.phase = "on_settle"
        elif c.phase == "on_settle":
            c.phase = "on_capture"
        elif c.phase == "on_capture":
            if c.off_frame is not None and c.on_frame is not None:
                point = self._diff_find_marker(c.off_frame, c.on_frame)
                if point is None:
                    self.status_message = f"Kunde inte hitta hörn {c.corner_index + 1}. Försök igen."
                    self._abort_calibration()
                    return

                self.detected_points.append(point)

            if c.corner_index >= 3:
                camera_points = np.array(self.detected_points, dtype=np.float32)
                self._save_calibration(camera_points)
                self.calibration = CalibrationState()
            else:
                c.corner_index += 1
                c.phase = "off_settle"
                c.frame_counter = 0
                c.off_frame = None
                c.on_frame = None
                self.status_message = f"Hittade hörn {c.corner_index} / 4. Fortsätter..."
                return

        c.frame_counter = 0

    def _update_calibration(self) -> None:
        if not self.calibration.active or self.last_frame_rgb is None:
            return

        c = self.calibration
        c.frame_counter += 1

        if c.phase == "off_settle":
            if c.frame_counter >= PHASE_SETTLE_FRAMES:
                self._advance_phase()
            return

        if c.phase == "off_capture":
            if c.frame_counter >= PHASE_CAPTURE_FRAMES:
                c.off_frame = self.last_frame_rgb.copy()
                self._advance_phase()
            return

        if c.phase == "on_settle":
            if c.frame_counter >= PHASE_SETTLE_FRAMES:
                self._advance_phase()
            return

        if c.phase == "on_capture":
            if c.frame_counter >= PHASE_CAPTURE_FRAMES:
                c.on_frame = self.last_frame_rgb.copy()
                self._advance_phase()
            return

    def update(self, dt: float):
        self._read_camera()
        if self.error_message:
            return None

        self._update_calibration()
        return None

    def _draw_live_overlay(self, screen: pygame.Surface) -> None:
        assert self.rect is not None
        assert self.viewport is not None

        shade = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, SCANPORT_SHADE_ALPHA))
        pygame.draw.rect(shade, (0, 0, 0, 0), self.rect)
        screen.blit(shade, (0, 0))

        pygame.draw.rect(screen, ORANGE, self.rect, width=4)
        _draw_dashed_rect(screen, GREEN, self.viewport, width=2, dash=12, gap=8)

        for idx, point in enumerate(self.detected_points):
            px, py = _to_int_pos(point)
            pygame.draw.circle(screen, CYAN, (px, py), 10, 3)
            label = self.tiny.render(str(idx + 1), True, CYAN)
            screen.blit(label, (px + 12, py - 10))

    def _draw_calibration_pattern(self, screen: pygame.Surface) -> None:
        assert self.viewport is not None

        if self.last_frame is not None:
            screen.blit(self.last_frame, (0, 0))
        else:
            screen.fill(self.bg_color)

        shade = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 190))
        pygame.draw.rect(shade, (0, 0, 0, 0), self.viewport)
        screen.blit(shade, (0, 0))

        pygame.draw.rect(screen, WHITE, self.viewport, width=2)

        # Visa bara EN aktiv blinkmarkör åt gången
        expected = _expected_corner_positions(self.viewport)
        for idx, pt in enumerate(expected):
            if idx < len(self.detected_points):
                pygame.draw.circle(screen, CYAN, _to_int_pos(pt), 10, 2)

        if self.calibration.active:
            current = _to_int_pos(expected[self.calibration.corner_index])

            # markören ska bara visas i on_settle / on_capture
            if self.calibration.phase in ("on_settle", "on_capture"):
                _draw_crosshair(screen, current, WHITE)

            label = self.small.render(
                f"Kalibrerar hörn {self.calibration.corner_index + 1} av 4",
                True,
                WHITE,
            )
            screen.blit(label, (24, 24))

            info = self.tiny.render(
                "Systemet blinkar markören och letar efter den förändring som uppstår i kamerabilden.",
                True,
                SOFT_WHITE,
            )
            screen.blit(info, (24, 56))

    def _draw_info_panel(self, screen: pygame.Surface) -> None:
        panel = pygame.Surface((980, 300), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (20, 20))

        lines = [
            ("Justera scanport och kalibrera kamerahörn", self.font, WHITE),
            ("Orange ram = scanport som analyseras", self.small, WHITE),
            ("Grön streckad ram = viewport / projicerad spelyta", self.small, WHITE),
            ("SPACE: starta automatisk kalibrering med blinkande hörnmarkörer", self.tiny, SOFT_WHITE),
            ("Kalibreringen visar EN markör i taget i viewporten och jämför av/på-bild", self.tiny, SOFT_WHITE),
            ("Detta fungerar bättre än QR/Aruco i rekursiv projektorbild", self.tiny, SOFT_WHITE),
            ("Pilar: flytta scanport", self.tiny, SOFT_WHITE),
            ("+ / - : öka eller minska proportionellt", self.tiny, SOFT_WHITE),
            ("SHIFT+B / B : öka eller minska bredd", self.tiny, SOFT_WHITE),
            ("SHIFT+H / H : öka eller minska höjd", self.tiny, SOFT_WHITE),
            ("ENTER: spara scanport", self.tiny, SOFT_WHITE),
            ("C: radera sparad kalibrering", self.tiny, SOFT_WHITE),
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

            if self.calibration.active:
                status_lines.append(
                    (
                        f"Kalibrering pågår: hörn {self.calibration.corner_index + 1}/4, fas={self.calibration.phase}",
                        SOFT_WHITE,
                    )
                )

            if self.status_message:
                status_lines.append((self.status_message, SOFT_WHITE))

        y = SCREEN_HEIGHT - (len(status_lines) * 28) - 12
        for text, color in status_lines:
            surf = self.tiny.render(text, True, color)
            screen.blit(surf, (24, y))
            y += 28

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        if self.last_frame is not None:
            screen.blit(self.last_frame, (0, 0))

        if self.calibration.active:
            self._draw_calibration_pattern(screen)
        else:
            if self.rect is not None and self.viewport is not None:
                self._draw_live_overlay(screen)
            self._draw_info_panel(screen)

        self._draw_status(screen)