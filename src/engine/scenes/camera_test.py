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

ARUCO_IDS = [10, 11, 12, 13]  # TL, TR, BR, BL
ARUCO_LABELS = ["TL", "TR", "BR", "BL"]
ARUCO_SIZE = 120
ARUCO_MARGIN = 30

DETECTION_STABLE_FRAMES = 10
SCANPORT_SHADE_ALPHA = 120


@dataclass
class MarkerDetection:
    marker_id: int
    center: tuple[float, float]
    corners: np.ndarray  # shape (4, 2)


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
            [float(rect.left), float(rect.top)],        # TL
            [float(rect.right), float(rect.top)],       # TR
            [float(rect.right), float(rect.bottom)],    # BR
            [float(rect.left), float(rect.bottom)],     # BL
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


def _corner_targets_from_rect(rect: pygame.Rect) -> list[tuple[float, float]]:
    return [
        (float(rect.left), float(rect.top)),
        (float(rect.right), float(rect.top)),
        (float(rect.right), float(rect.bottom)),
        (float(rect.left), float(rect.bottom)),
    ]


class CameraTestScene(Scene):
    """
    Scanport-kalibrering + automatisk viewport-kalibrering med ArUco-markörer.

    Kontroller:
    - Pilar = flytta scanport
    - + / - = skala proportionellt
    - SHIFT+B / B = öka / minska bredd
    - SHIFT+H / H = öka / minska höjd
    - ENTER = spara scanport
    - SPACE = starta / stoppa automatisk kalibrering i viewporten
    - C = radera sparad hörnkalibrering
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

        self.show_calibration_pattern = False
        self.calibration_stable_hits = 0
        self.detected_points: np.ndarray | None = None
        self.detected_marker_centers: dict[int, tuple[float, float]] = {}
        self.aruco_available = hasattr(cv2, "aruco")

        self.aruco_dict = None
        self.aruco_detector = None
        self.marker_surfaces: dict[int, pygame.Surface] = {}

    def on_enter(self) -> None:
        self.font = pygame.font.Font(None, 42)
        self.small = pygame.font.Font(None, 26)
        self.tiny = pygame.font.Font(None, 24)

        self.error_message = None
        self.status_message = ""
        self.last_frame = None
        self.last_frame_rgb = None
        self.show_calibration_pattern = False
        self.calibration_stable_hits = 0
        self.detected_marker_centers = {}

        self.viewport = load_viewport_rect()
        scanport = load_scanport_norm()
        self.rect = scanport_norm_to_screen_rect(scanport)
        self.calibration_data = load_camera_calibration()

        if self.calibration_data and self.calibration_data.get("camera_points_norm"):
            pts = []
            for x, y in self.calibration_data["camera_points_norm"]:
                pts.append([x * SCREEN_WIDTH, y * SCREEN_HEIGHT])
            self.detected_points = np.array(pts, dtype=np.float32)
            self.status_message = "Kalibrering hittad."
        else:
            self.detected_points = None
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

        if not self.aruco_available:
            self.error_message = "OpenCV saknar cv2.aruco. Installera opencv-contrib-python."
            return

        self._init_aruco()

    def on_exit(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def _go_back(self):
        from src.engine.scenes.menu import MenuScene
        return SceneSwitch(MenuScene())

    def _init_aruco(self) -> None:
        # Kompatibel med flera OpenCV-versioner
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

        if hasattr(cv2.aruco, "DetectorParameters"):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)
        else:
            self.aruco_detector = None

        self.marker_surfaces = {}
        for marker_id in ARUCO_IDS:
            marker_img = self._generate_aruco_marker(marker_id, ARUCO_SIZE)
            marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
            marker_surface = pygame.image.frombuffer(
                marker_rgb.tobytes(),
                (marker_rgb.shape[1], marker_rgb.shape[0]),
                "RGB",
            ).convert()
            self.marker_surfaces[marker_id] = marker_surface

    def _generate_aruco_marker(self, marker_id: int, size: int) -> np.ndarray:
        if hasattr(cv2.aruco, "generateImageMarker"):
            img = np.zeros((size, size), dtype=np.uint8)
            cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, size, img, 1)
            return img
        img = np.zeros((size, size), dtype=np.uint8)
        cv2.aruco.drawMarker(self.aruco_dict, marker_id, size, img, 1)
        return img

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
            "marker_ids": ARUCO_IDS,
        }
        save_camera_calibration(self.calibration_data)
        self.detected_points = camera_points.copy()
        self.status_message = "Kalibrering klar och sparad."

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
            self.detected_points = None
            self.detected_marker_centers = {}
            self.status_message = "Sparad kalibrering raderad."
            return None

        if event.key == pygame.K_SPACE:
            self.show_calibration_pattern = not self.show_calibration_pattern
            self.calibration_stable_hits = 0
            self.detected_marker_centers = {}
            if self.show_calibration_pattern:
                self.status_message = (
                    "Kalibrering aktiv: visar ArUco-markörer i viewporten och söker närmaste instans i varje hörn."
                )
            else:
                self.status_message = "Kalibrering stoppad."
            return None

        if self.show_calibration_pattern:
            # Lås scanportjustering medan kalibreringen visas
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

    def _detect_aruco_markers(self) -> list[MarkerDetection]:
        if self.last_frame_rgb is None:
            return []

        gray = cv2.cvtColor(self.last_frame_rgb, cv2.COLOR_RGB2GRAY)

        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)

        if ids is None or len(ids) == 0:
            return []

        detections: list[MarkerDetection] = []
        for idx, marker_id in enumerate(ids.flatten().tolist()):
            pts = np.array(corners[idx], dtype=np.float32).reshape(4, 2)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            detections.append(
                MarkerDetection(
                    marker_id=int(marker_id),
                    center=(cx, cy),
                    corners=pts,
                )
            )
        return detections

    def _select_best_corner_instances(
        self,
        detections: list[MarkerDetection],
    ) -> np.ndarray | None:
        assert self.viewport is not None
        targets = _corner_targets_from_rect(self.viewport)

        selected_points: list[list[float]] = []
        selected_debug: dict[int, tuple[float, float]] = {}

        for marker_id, target in zip(ARUCO_IDS, targets):
            candidates = [d for d in detections if d.marker_id == marker_id]
            if not candidates:
                return None

            best = min(candidates, key=lambda d: _distance(d.center, target))
            selected_points.append([best.center[0], best.center[1]])
            selected_debug[marker_id] = best.center

        self.detected_marker_centers = selected_debug
        return np.array(selected_points, dtype=np.float32)

    def _update_auto_calibration(self) -> None:
        if not self.show_calibration_pattern:
            return

        detections = self._detect_aruco_markers()
        points = self._select_best_corner_instances(detections)

        if points is None:
            self.calibration_stable_hits = 0
            return

        self.detected_points = points
        self.calibration_stable_hits += 1

        if self.calibration_stable_hits >= DETECTION_STABLE_FRAMES:
            self._save_calibration(points)
            self.show_calibration_pattern = False
            self.calibration_stable_hits = 0

    def update(self, dt: float):
        self._read_camera()
        if self.error_message:
            return None

        self._update_auto_calibration()
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

        if self.detected_points is not None:
            for idx, point in enumerate(self.detected_points):
                px, py = _to_int_pos((float(point[0]), float(point[1])))
                pygame.draw.circle(screen, CYAN, (px, py), 10, 3)
                label = self.tiny.render(ARUCO_LABELS[idx], True, CYAN)
                screen.blit(label, (px + 12, py - 10))

    def _draw_calibration_pattern(self, screen: pygame.Surface) -> None:
        assert self.viewport is not None

        # Behåll livebild som bakgrund men rita markörerna i viewporten
        if self.last_frame is not None:
            screen.blit(self.last_frame, (0, 0))
        else:
            screen.fill(self.bg_color)

        # Dämpa allt utanför viewporten för att göra kalibreringen tydligare
        shade = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        shade.fill((0, 0, 0, 185))
        pygame.draw.rect(shade, (0, 0, 0, 0), self.viewport)
        screen.blit(shade, (0, 0))

        pygame.draw.rect(screen, WHITE, self.viewport, width=2)

        marker_positions = [
            (self.viewport.left + ARUCO_MARGIN, self.viewport.top + ARUCO_MARGIN),
            (self.viewport.right - ARUCO_MARGIN - ARUCO_SIZE, self.viewport.top + ARUCO_MARGIN),
            (self.viewport.right - ARUCO_MARGIN - ARUCO_SIZE, self.viewport.bottom - ARUCO_MARGIN - ARUCO_SIZE),
            (self.viewport.left + ARUCO_MARGIN, self.viewport.bottom - ARUCO_MARGIN - ARUCO_SIZE),
        ]

        for marker_id, pos in zip(ARUCO_IDS, marker_positions):
            surface = self.marker_surfaces[marker_id]
            screen.blit(surface, pos)

        # Centermarkör bara som visuell referens
        pygame.draw.circle(screen, YELLOW, self.viewport.center, 18, 3)
        pygame.draw.line(
            screen,
            YELLOW,
            (self.viewport.centerx - 26, self.viewport.centery),
            (self.viewport.centerx + 26, self.viewport.centery),
            3,
        )
        pygame.draw.line(
            screen,
            YELLOW,
            (self.viewport.centerx, self.viewport.centery - 26),
            (self.viewport.centerx, self.viewport.centery + 26),
            3,
        )

        # Visar hittade punkter under pågående kalibrering
        if self.detected_points is not None:
            for idx, point in enumerate(self.detected_points):
                px, py = _to_int_pos((float(point[0]), float(point[1])))
                pygame.draw.circle(screen, CYAN, (px, py), 12, 3)
                label = self.tiny.render(ARUCO_LABELS[idx], True, CYAN)
                screen.blit(label, (px + 12, py - 10))

    def _draw_info_panel(self, screen: pygame.Surface) -> None:
        panel = pygame.Surface((1020, 300), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (20, 20))

        lines = [
            ("Justera scanport och kalibrera kamerahörn", self.font, WHITE),
            ("Orange ram = scanport som analyseras", self.small, WHITE),
            ("Grön streckad ram = viewport / projicerad spelyta", self.small, WHITE),
            ("SPACE: visa ArUco-markörer i viewporten och kör automatisk kalibrering", self.tiny, SOFT_WHITE),
            ("Systemet väljer närmaste instans av varje marker-ID i respektive hörn", self.tiny, SOFT_WHITE),
            ("Detta hanterar spegel-/loop-effekten bättre än vanliga kryss", self.tiny, SOFT_WHITE),
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

            if self.show_calibration_pattern:
                status_lines.append(
                    (
                        f"Kalibrering pågår... stabila träffar: {self.calibration_stable_hits}/{DETECTION_STABLE_FRAMES}",
                        SOFT_WHITE,
                    )
                )

            if self.detected_marker_centers:
                ids_text = ", ".join(
                    f"{marker_id}:{int(pos[0])},{int(pos[1])}"
                    for marker_id, pos in sorted(self.detected_marker_centers.items())
                )
                status_lines.append((f"Hittade markörer: {ids_text}", SOFT_WHITE))

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

        if self.show_calibration_pattern:
            self._draw_calibration_pattern(screen)
        else:
            if self.rect is not None and self.viewport is not None:
                self._draw_live_overlay(screen)
            self._draw_info_panel(screen)

        self._draw_status(screen)