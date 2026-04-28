"""
Camera → viewport calibration scene (manual, interactive).

Uses the shared ArucoCalibrator engine for detection and homography.
This scene adds the interactive UI: idle screen, ENTER to start, timeout,
status display, per-marker error visualization.
"""
from __future__ import annotations

import time

import pygame

from config import SCREEN_HEIGHT, SCREEN_WIDTH
from src.engine.camera.aruco_calibrator import ArucoCalibrator
from src.engine.camera.camera_manager import camera_manager
from src.engine.input.hit_input import hit_input
from src.engine.scene import Scene, SceneSwitch
from src.engine.scenes.menu import MenuScene
from src.engine.settings import (
    load_camera_calibration,
    load_viewport_rect,
    save_camera_calibration,
)


WHITE = (245, 245, 245)
SOFT = (205, 205, 205)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
PANEL_BG = (0, 0, 0, 170)
STATUS_OK_BG = (35, 80, 35, 170)
STATUS_ERR_BG = (95, 25, 25, 170)


class CameraViewportCalibrationScene(Scene):
    wants_hit_scanning = False
    wants_camera_preview = False

    def __init__(self, bg_color=(10, 10, 10), timeout_seconds: float = 5.0, **kwargs) -> None:
        super().__init__()
        self.bg_color = tuple(bg_color)
        self.timeout_seconds = float(timeout_seconds)
        self.kwargs = kwargs

        self.font_title: pygame.font.Font | None = None
        self.font_body: pygame.font.Font | None = None
        self.font_small: pygame.font.Font | None = None

        self.viewport_rect = load_viewport_rect()
        self.state = "idle"
        self.capture_started_at: float | None = None

        self.status_message = "Redo."
        self.status_is_error = False
        self.last_saved = load_camera_calibration() or {}

        self.last_frame_bgr = None
        self.last_detection_count = 0

        self.calibrator: ArucoCalibrator | None = None

    def on_enter(self) -> None:
        self.font_title = pygame.font.Font(None, 44)
        self.font_body = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)
        self.viewport_rect = load_viewport_rect()

        self.calibrator = ArucoCalibrator(self.viewport_rect)
        if not self.calibrator.available:
            self.status_message = "OpenCV ArUco saknas i denna installation."
            self.status_is_error = True

        camera_manager.start()

    def on_exit(self) -> None:
        pass

    def _go_back(self):
        return SceneSwitch(MenuScene(menu_state=getattr(self, "return_menu_state", None)))

    def _start_capture(self) -> None:
        if self.calibrator is None or not self.calibrator.available:
            self.status_message = "OpenCV ArUco saknas i denna installation."
            self.status_is_error = True
            return
        self.viewport_rect = load_viewport_rect()
        self.calibrator.viewport_rect = self.viewport_rect
        self.capture_started_at = time.monotonic()
        self.state = "capturing"
        self.last_detection_count = 0

    def _finish_capture(self, success: bool, message: str, result=None) -> None:
        self.state = "idle"
        self.capture_started_at = None
        self.status_message = message
        self.status_is_error = not success
        if success and result is not None and result.success:
            self.calibrator.save_and_apply(result)
            self.last_saved = result.calibration_dict

    def _clear_calibration(self) -> None:
        save_camera_calibration({})
        self.last_saved = {}
        hit_input.reload_calibration()
        self.status_message = "Sparad kamerakalibrering rensad."
        self.status_is_error = False
        self.state = "idle"

    def handle_event(self, event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            return self._go_back()

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.state == "idle":
                self._start_capture()
            return None

        if event.key == pygame.K_r and self.state == "idle":
            self._clear_calibration()
            return None

        return None

    def update(self, dt: float):
        del dt
        camera_manager.update()
        self.last_frame_bgr = camera_manager.get_latest_frame()

        if self.state != "capturing":
            return None

        if self.capture_started_at is None:
            self.capture_started_at = time.monotonic()

        elapsed = time.monotonic() - self.capture_started_at
        if elapsed >= self.timeout_seconds:
            self._finish_capture(False, "Timeout: kunde inte hitta tillräckligt många kalibreringsmarkörer.")
            return None

        if self.last_frame_bgr is None or self.calibrator is None:
            self.last_detection_count = 0
            return None

        result = self.calibrator.detect_and_calibrate(self.last_frame_bgr)
        self.last_detection_count = result.marker_count

        if not result.success:
            return None

        self._finish_capture(
            True,
            f"Kalibrering sparad. {result.marker_count} markörer ({result.inliers} inliers), "
            f"reprojection error {result.reprojection_error_px:.1f} px.",
            result=result,
        )

    def _fmt(self, key: str, fallback: str = "-") -> str:
        value = self.last_saved.get(key) if isinstance(self.last_saved, dict) else None
        if value in (None, "", [], {}):
            return fallback
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _render_idle(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)

        title = self.font_title.render("Kamera → viewport-kalibrering", True, WHITE)
        screen.blit(title, (36, 28))

        subtitle = self.font_small.render(
            "Starta bara ny kalibrering när projektorn visar tavlan på rätt skärm.",
            True,
            SOFT,
        )
        screen.blit(subtitle, (38, 70))

        panel = pygame.Surface((900, 330), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (28, 110))

        lines = [
            f"Senast kalibrerad: {self._fmt('calibrated_at', 'Aldrig')}",
            f"Metod: {self._fmt('method', '-')}",
            f"Markörer senast: {self._fmt('marker_count', '0')}",
            f"Reprojection error: {self._fmt('reprojection_error_px', '-')}",
            f"Viewport rect: {self._fmt('viewport_rect', '-')}",
            f"Kamerastorlek: {self._fmt('camera_frame_size', '-')}",
        ]

        y = 138
        for line in lines:
            surf = self.font_body.render(line, True, WHITE)
            screen.blit(surf, (48, y))
            y += 38

        help_lines = [
            "ENTER = starta ny kalibrering",
            "R = rensa sparad kalibrering",
            "ESC = tillbaka",
            f"Timeout vid kalibrering: {self.timeout_seconds:.0f} sekunder",
        ]
        y += 12
        for line in help_lines:
            surf = self.font_small.render(line, True, SOFT)
            screen.blit(surf, (48, y))
            y += 26

        status_bg = STATUS_ERR_BG if self.status_is_error else STATUS_OK_BG
        status_fg = RED if self.status_is_error else GREEN
        bar = pygame.Surface((SCREEN_WIDTH - 56, 56), pygame.SRCALPHA)
        bar.fill(status_bg)
        screen.blit(bar, (28, SCREEN_HEIGHT - 86))
        status = self.font_body.render(self.status_message or " ", True, status_fg)
        screen.blit(status, (42, SCREEN_HEIGHT - 69))

    def _render_capture(self, screen: pygame.Surface) -> None:
        if self.calibrator is not None:
            self.calibrator.render_markers(screen)

    def render(self, screen: pygame.Surface) -> None:
        self.viewport_rect = load_viewport_rect()
        if self.calibrator is not None:
            self.calibrator.viewport_rect = self.viewport_rect

        if self.font_title is None or self.font_body is None or self.font_small is None:
            return

        if self.state == "capturing":
            self._render_capture(screen)
            return

        self._render_idle(screen)
