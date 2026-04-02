from __future__ import annotations

import time
from datetime import datetime

import cv2
import numpy as np
import pygame

from config import SCREEN_HEIGHT, SCREEN_WIDTH
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

        self.aruco_available = hasattr(cv2, "aruco")
        self.aruco_dict = None
        self.aruco_detector = None
        self.detector_params = None

    def on_enter(self) -> None:
        self.font_title = pygame.font.Font(None, 44)
        self.font_body = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)
        self.viewport_rect = load_viewport_rect()

        if self.aruco_available:
            self._init_aruco()
        else:
            self.status_message = "OpenCV ArUco saknas i denna installation."
            self.status_is_error = True

        camera_manager.start()

    def on_exit(self) -> None:
        pass

    def _go_back(self):
        return SceneSwitch(MenuScene(menu_state=getattr(self, "return_menu_state", None)))

    def _init_aruco(self) -> None:
        dict_name = getattr(cv2.aruco, "DICT_4X4_50", None)
        if dict_name is None:
            self.aruco_available = False
            self.status_message = "ArUco dictionary saknas i OpenCV."
            self.status_is_error = True
            return

        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(dict_name)

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.detector_params = cv2.aruco.DetectorParameters()
        else:
            self.detector_params = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    def _detect_markers(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector is not None:
            return self.aruco_detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)

    def _marker_size_px(self) -> int:
        m = min(self.viewport_rect.w, self.viewport_rect.h)
        return max(64, min(160, int(m * 0.11)))

    def _marker_positions(self) -> dict[int, tuple[float, float]]:
        rect = self.viewport_rect
        margin_x = max(54, int(rect.w * 0.08))
        margin_y = max(54, int(rect.h * 0.08))
        return {
            0: (rect.left + margin_x, rect.top + margin_y),
            1: (rect.centerx, rect.top + margin_y),
            2: (rect.right - margin_x, rect.top + margin_y),
            3: (rect.right - margin_x, rect.centery),
            4: (rect.right - margin_x, rect.bottom - margin_y),
            5: (rect.centerx, rect.bottom - margin_y),
            6: (rect.left + margin_x, rect.bottom - margin_y),
            7: (rect.left + margin_x, rect.centery),
        }

    def _draw_marker_surface(self, marker_id: int, size: int) -> pygame.Surface:
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker_img = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, size)
        else:
            marker_img = np.zeros((size, size), dtype=np.uint8)
            cv2.aruco.drawMarker(self.aruco_dict, marker_id, size, marker_img, 1)

        rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
        rgb = np.transpose(rgb, (1, 0, 2))
        surf = pygame.surfarray.make_surface(rgb)
        return surf.convert()

    def _homography_error(self, H: np.ndarray, camera_pts: np.ndarray, viewport_pts: np.ndarray) -> float:
        projected = cv2.perspectiveTransform(camera_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        diff = projected - viewport_pts
        distances = np.sqrt(np.sum(diff * diff, axis=1))
        return float(np.mean(distances)) if len(distances) else 9999.0

    def _extract_result(self, frame_bgr, corners, ids) -> dict | None:
        if ids is None or len(ids) < 4:
            return None

        marker_positions = self._marker_positions()
        camera_points = []
        viewport_points = []

        for idx, marker_id in enumerate(ids.flatten().tolist()):
            if marker_id not in marker_positions:
                continue
            center = np.mean(corners[idx][0], axis=0).astype(np.float32)
            camera_points.append([float(center[0]), float(center[1])])
            vx, vy = marker_positions[marker_id]
            viewport_points.append([float(vx), float(vy)])

        if len(camera_points) < 4:
            return None

        cam_np = np.array(camera_points, dtype=np.float32)
        vp_np = np.array(viewport_points, dtype=np.float32)

        H, _ = cv2.findHomography(cam_np, vp_np, method=0)
        if H is None:
            return None

        H_inv, _ = cv2.findHomography(vp_np, cam_np, method=0)
        reproj = self._homography_error(H, cam_np, vp_np)

        frame_h, frame_w = frame_bgr.shape[:2]

        return {
            "method": "aruco_viewport_board",
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            "homography": H.tolist(),
            "inverse_homography": H_inv.tolist() if H_inv is not None else None,
            "marker_count": int(len(camera_points)),
            "reprojection_error_px": float(reproj),
            "camera_points": camera_points,
            "viewport_points": viewport_points,
            "camera_frame_size": [int(frame_w), int(frame_h)],
            "viewport_rect": [
                int(self.viewport_rect.x),
                int(self.viewport_rect.y),
                int(self.viewport_rect.w),
                int(self.viewport_rect.h),
            ],
        }

    def _start_capture(self) -> None:
        if not self.aruco_available:
            self.status_message = "OpenCV ArUco saknas i denna installation."
            self.status_is_error = True
            return
        self.viewport_rect = load_viewport_rect()
        self.capture_started_at = time.monotonic()
        self.state = "capturing"
        self.last_detection_count = 0

    def _finish_capture(self, success: bool, message: str, result: dict | None = None) -> None:
        self.state = "idle"
        self.capture_started_at = None
        self.status_message = message
        self.status_is_error = not success
        if success and result is not None:
            save_camera_calibration(result)
            self.last_saved = result
            hit_input.reload_calibration()

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

        if self.last_frame_bgr is None:
            self.last_detection_count = 0
            return None

        corners, ids, _ = self._detect_markers(self.last_frame_bgr)
        self.last_detection_count = 0 if ids is None else int(len(ids))

        result = self._extract_result(self.last_frame_bgr, corners, ids)
        if result is None:
            return None

        reproj = float(result.get("reprojection_error_px", 9999.0))
        if reproj > 25.0:
            return None

        self._finish_capture(
            True,
            f"Kalibrering sparad. {result['marker_count']} markörer, reprojection error {reproj:.1f} px.",
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
        screen.fill(WHITE)

        rect = self.viewport_rect
        positions = self._marker_positions()
        marker_size = self._marker_size_px()

        fill = pygame.Surface((rect.w, rect.h))
        fill.fill(WHITE)
        screen.blit(fill, rect.topleft)

        for marker_id, center in positions.items():
            surf = self._draw_marker_surface(marker_id, marker_size)
            marker_rect = surf.get_rect(center=(int(center[0]), int(center[1])))
            screen.blit(surf, marker_rect.topleft)

    def render(self, screen: pygame.Surface) -> None:
        self.viewport_rect = load_viewport_rect()

        if self.font_title is None or self.font_body is None or self.font_small is None:
            return

        if self.state == "capturing":
            self._render_capture(screen)
            return

        self._render_idle(screen)
