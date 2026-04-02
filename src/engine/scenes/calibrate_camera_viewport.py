from __future__ import annotations

import time
from datetime import datetime

import cv2
import numpy as np
import pygame

from config import SCREEN_HEIGHT, SCREEN_WIDTH
from src.engine.camera.camera_manager import camera_manager
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_camera_calibration,
    load_viewport_rect,
    save_camera_calibration,
)
from src.engine.scenes.menu import MenuScene


WHITE = (240, 240, 240)
SOFT = (190, 190, 190)
GREEN = (120, 255, 120)
RED = (255, 120, 120)
YELLOW = (255, 220, 80)
CYAN = (110, 220, 255)
PANEL_BG = (0, 0, 0, 170)
OK_BG = (30, 70, 30, 170)
ERR_BG = (90, 25, 25, 170)


class CameraViewportCalibrationScene(Scene):
    """
    Kalibrerar mapping mellan kamera och projicerad viewport via ArUco-markörer.

    Idle-läge:
    - Visar tidigare kalibreringsdata
    - ENTER startar ny kalibrering

    Capturing-läge:
    - Visar markörer i viewport-ytan
    - Läser kameran
    - Timeout efter några sekunder om projektorn är av, fel skärm används
      eller markörerna inte hittas
    """

    wants_hit_scanning = False
    wants_camera_preview = True

    def __init__(self, bg_color=(10, 10, 10), timeout_seconds: float = 5.0) -> None:
        super().__init__()
        self.bg_color = tuple(bg_color)
        self.timeout_seconds = float(timeout_seconds)

        self.font_title: pygame.font.Font | None = None
        self.font_body: pygame.font.Font | None = None
        self.font_small: pygame.font.Font | None = None

        self.state = "idle"  # idle | capturing | result
        self.status_message = "Redo."
        self.status_is_error = False
        self.capture_started_at: float | None = None

        self.viewport_rect = load_viewport_rect()
        self.last_frame_bgr: np.ndarray | None = None
        self.last_frame_surface: pygame.Surface | None = None
        self.frame_draw_rect: pygame.Rect | None = None

        self.last_detection_count = 0
        self.last_saved = load_camera_calibration() or {}

        self.aruco_available = hasattr(cv2, "aruco")
        self.aruco_dict = None
        self.aruco_detector = None
        self.detector_params = None

        self.marker_ids = [0, 1, 2, 3, 4, 5, 6, 7]
        self.marker_specs: dict[int, tuple[float, float]] = {}

        self.latest_result: dict | None = None

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------
    def _back_to_menu(self):
        return SceneSwitch(MenuScene(menu_state=getattr(self, "return_menu_state", None)))

    # ------------------------------------------------------------
    # ArUco helpers
    # ------------------------------------------------------------
    def _init_aruco(self) -> None:
        dictionary_name = getattr(cv2.aruco, "DICT_4X4_50", None)
        if dictionary_name is None:
            self.aruco_available = False
            self.status_message = "ArUco dictionary saknas i OpenCV."
            self.status_is_error = True
            return

        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_name)
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(dictionary_name)

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.detector_params = cv2.aruco.DetectorParameters()
        else:
            self.detector_params = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    def _draw_marker_surface(self, marker_id: int, size: int) -> pygame.Surface:
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker_img = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, size)
        else:
            marker_img = np.zeros((size, size), dtype=np.uint8)
            cv2.aruco.drawMarker(self.aruco_dict, marker_id, size, marker_img, 1)

        marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
        marker_rgb = np.transpose(marker_rgb, (1, 0, 2))
        surf = pygame.surfarray.make_surface(marker_rgb)
        return surf.convert()

    def _detect_markers(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.aruco_detector is not None:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.detector_params,
            )
        return corners, ids, rejected

    # ------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------
    def _build_marker_specs(self) -> dict[int, tuple[float, float]]:
        rect = self.viewport_rect
        margin_x = max(46, int(rect.w * 0.07))
        margin_y = max(46, int(rect.h * 0.07))
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

    def _marker_size_px(self) -> int:
        m = min(self.viewport_rect.w, self.viewport_rect.h)
        return max(56, min(160, int(m * 0.12)))

    def _frame_to_surface(self, frame_bgr: np.ndarray) -> pygame.Surface:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame_rgb)
        return surf.convert()

    def _camera_to_screen_preview_point(self, camera_x: float, camera_y: float, frame_w: int, frame_h: int) -> tuple[int, int]:
        if self.frame_draw_rect is None:
            return (0, 0)
        sx = self.frame_draw_rect.x + int(round((camera_x / max(1, frame_w)) * self.frame_draw_rect.w))
        sy = self.frame_draw_rect.y + int(round((camera_y / max(1, frame_h)) * self.frame_draw_rect.h))
        return sx, sy

    def _homography_error(self, H: np.ndarray, camera_pts: np.ndarray, viewport_pts: np.ndarray) -> float:
        projected = cv2.perspectiveTransform(camera_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        diff = projected - viewport_pts
        distances = np.sqrt(np.sum(diff * diff, axis=1))
        return float(np.mean(distances)) if len(distances) else 9999.0

    def _quad_from_corners(self, mapping: dict[int, np.ndarray]) -> dict[str, list[float]] | None:
        required = [0, 2, 4, 6]  # ungefär tl, tr, br, bl via våra 8 markörer
        if not all(mid in mapping for mid in required):
            return None
        return {
            "top_left": [float(mapping[0][0]), float(mapping[0][1])],
            "top_right": [float(mapping[2][0]), float(mapping[2][1])],
            "bottom_right": [float(mapping[4][0]), float(mapping[4][1])],
            "bottom_left": [float(mapping[6][0]), float(mapping[6][1])],
        }

    def _extract_result(self, frame_bgr: np.ndarray, corners, ids) -> dict | None:
        if ids is None or len(ids) < 4:
            return None

        self.marker_specs = self._build_marker_specs()

        camera_points: list[list[float]] = []
        viewport_points: list[list[float]] = []
        detected_centers: dict[int, np.ndarray] = {}

        for idx, marker_id in enumerate(ids.flatten().tolist()):
            if marker_id not in self.marker_specs:
                continue
            pts = corners[idx][0]
            center = np.mean(pts, axis=0)
            detected_centers[marker_id] = center.astype(np.float32)
            camera_points.append([float(center[0]), float(center[1])])
            vx, vy = self.marker_specs[marker_id]
            viewport_points.append([float(vx), float(vy)])

        if len(camera_points) < 4:
            return None

        cam_np = np.array(camera_points, dtype=np.float32)
        vp_np = np.array(viewport_points, dtype=np.float32)

        H, _ = cv2.findHomography(cam_np, vp_np, method=0)
        if H is None:
            return None

        H_inv, _ = cv2.findHomography(vp_np, cam_np, method=0)
        reprojection_error = self._homography_error(H, cam_np, vp_np)

        frame_h, frame_w = frame_bgr.shape[:2]

        result = {
            "method": "aruco_viewport_board",
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            "homography": H.tolist(),
            "inverse_homography": H_inv.tolist() if H_inv is not None else None,
            "marker_count": int(len(camera_points)),
            "marker_ids": sorted(int(v) for v in detected_centers.keys()),
            "reprojection_error_px": float(reprojection_error),
            "camera_points": camera_points,
            "viewport_points": viewport_points,
            "camera_frame_size": [int(frame_w), int(frame_h)],
            "viewport_rect": [
                int(self.viewport_rect.x),
                int(self.viewport_rect.y),
                int(self.viewport_rect.w),
                int(self.viewport_rect.h),
            ],
            "camera_quad": self._quad_from_corners(detected_centers),
        }
        return result

    # ------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------
    def _start_capture(self) -> None:
        if not self.aruco_available:
            self.status_message = "OpenCV ArUco saknas i denna installation."
            self.status_is_error = True
            return
        self.viewport_rect = load_viewport_rect()
        self.capture_started_at = time.monotonic()
        self.state = "capturing"
        self.status_message = "Visar kalibreringspunkter och söker markörer..."
        self.status_is_error = False
        self.latest_result = None
        self.last_detection_count = 0

    def _clear_calibration(self) -> None:
        save_camera_calibration({})
        self.last_saved = {}
        self.latest_result = None
        self.status_message = "Sparad kamerakalibrering rensad."
        self.status_is_error = False
        self.state = "idle"

    def _finish_capture(self, success: bool, result: dict | None = None, message: str = "") -> None:
        self.state = "result"
        self.capture_started_at = None
        self.latest_result = result
        self.status_message = message
        self.status_is_error = not success

        if success and result is not None:
            save_camera_calibration(result)
            self.last_saved = result

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            return self._back_to_menu()

        if event.key == pygame.K_r:
            if self.state == "capturing":
                self.state = "idle"
                self.capture_started_at = None
                self.status_message = "Kalibreringen avbröts."
                self.status_is_error = True
            else:
                self._clear_calibration()
            return None

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.state in ("idle", "result"):
                self._start_capture()
            return None

        return None

    # ------------------------------------------------------------
    # Update
    # ------------------------------------------------------------
    def update(self, dt: float):
        del dt

        camera_manager.update()
        self.last_frame_bgr = camera_manager.get_latest_frame()

        if self.last_frame_bgr is not None:
            self.last_frame_surface = self._frame_to_surface(self.last_frame_bgr)
        else:
            self.last_frame_surface = None

        if self.state != "capturing":
            return None

        if self.capture_started_at is None:
            self.capture_started_at = time.monotonic()

        elapsed = time.monotonic() - self.capture_started_at
        if elapsed >= self.timeout_seconds:
            self._finish_capture(
                success=False,
                result=None,
                message="Timeout: kunde inte hitta tillräckligt många kalibreringsmarkörer.",
            )
            return None

        if self.last_frame_bgr is None:
            self.status_message = f"Väntar på kamerabild... {elapsed:.1f}/{self.timeout_seconds:.0f} s"
            self.status_is_error = False
            return None

        corners, ids, _ = self._detect_markers(self.last_frame_bgr)
        self.last_detection_count = 0 if ids is None else int(len(ids))

        result = self._extract_result(self.last_frame_bgr, corners, ids)
        if result is not None:
            reproj = float(result.get("reprojection_error_px", 9999.0))
            if reproj <= 25.0:
                self._finish_capture(
                    success=True,
                    result=result,
                    message=f"Kalibrering sparad. {result['marker_count']} markörer, reprojection error {reproj:.1f} px.",
                )
                return None

            self.status_message = (
                f"Hittade markörer men felet är högt ({reproj:.1f} px). "
                f"Fortsätter söka... {elapsed:.1f}/{self.timeout_seconds:.0f} s"
            )
            self.status_is_error = True
            return None

        self.status_message = (
            f"Söker markörer... hittade {self.last_detection_count}/4+ "
            f"({elapsed:.1f}/{self.timeout_seconds:.0f} s)"
        )
        self.status_is_error = False
        return None

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------
    def _render_camera_preview(self, screen: pygame.Surface) -> None:
        if self.last_frame_surface is None:
            self.frame_draw_rect = None
            panel = pygame.Surface((430, 250), pygame.SRCALPHA)
            panel.fill(PANEL_BG)
            screen.blit(panel, (SCREEN_WIDTH - 450, 24))
            msg = self.font_body.render("Ingen kamerabild", True, RED)
            screen.blit(msg, (SCREEN_WIDTH - 420, 50))
            return

        preview_w = int(SCREEN_WIDTH * 0.34)
        preview_h = int(SCREEN_HEIGHT * 0.34)
        src_w, src_h = self.last_frame_surface.get_size()
        scale = min(preview_w / max(1, src_w), preview_h / max(1, src_h))
        draw_w = max(1, int(src_w * scale))
        draw_h = max(1, int(src_h * scale))

        surf = self.last_frame_surface
        if surf.get_size() != (draw_w, draw_h):
            surf = pygame.transform.smoothscale(surf, (draw_w, draw_h))

        x = SCREEN_WIDTH - draw_w - 28
        y = 28
        box = pygame.Rect(x - 10, y - 10, draw_w + 20, draw_h + 20)
        panel = pygame.Surface((box.w, box.h), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, box.topleft)
        screen.blit(surf, (x, y))
        self.frame_draw_rect = pygame.Rect(x, y, draw_w, draw_h)

        title = self.font_small.render("Kamerapreview", True, WHITE)
        screen.blit(title, (x, y - 22))

        if self.last_frame_bgr is not None and self.state == "capturing":
            corners, ids, _ = self._detect_markers(self.last_frame_bgr)
            if ids is not None:
                frame_h, frame_w = self.last_frame_bgr.shape[:2]
                for idx, marker_id in enumerate(ids.flatten().tolist()):
                    pts = corners[idx][0]
                    center = np.mean(pts, axis=0)
                    sx, sy = self._camera_to_screen_preview_point(center[0], center[1], frame_w, frame_h)
                    pygame.draw.circle(screen, CYAN, (sx, sy), 6, 2)
                    tag = self.font_small.render(str(marker_id), True, CYAN)
                    screen.blit(tag, (sx + 8, sy - 8))

    def _render_viewport_board(self, screen: pygame.Surface) -> None:
        rect = self.viewport_rect
        outline = GREEN if self.state != "capturing" else YELLOW
        pygame.draw.rect(screen, outline, rect, 3)

        if self.state != "capturing" or not self.aruco_available:
            return

        self.marker_specs = self._build_marker_specs()
        marker_size = self._marker_size_px()

        fill = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        fill.fill((255, 255, 255, 6))
        screen.blit(fill, rect.topleft)

        for marker_id, (cx, cy) in self.marker_specs.items():
            surf = self._draw_marker_surface(marker_id, marker_size)
            marker_rect = surf.get_rect(center=(int(cx), int(cy)))
            screen.blit(surf, marker_rect.topleft)

            label = self.font_small.render(str(marker_id), True, WHITE)
            lab_rect = label.get_rect(center=(marker_rect.centerx, marker_rect.bottom + 14))
            screen.blit(label, lab_rect)

    def _fmt_saved_value(self, key: str, fallback: str = "-") -> str:
        if not isinstance(self.last_saved, dict):
            return fallback
        value = self.last_saved.get(key)
        if value in (None, "", [], {}):
            return fallback
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _render_idle_info(self, screen: pygame.Surface) -> None:
        panel = pygame.Surface((760, 310), pygame.SRCALPHA)
        panel.fill(PANEL_BG)
        screen.blit(panel, (26, 88))

        title = self.font_title.render("Kamera → viewport-kalibrering", True, WHITE)
        screen.blit(title, (40, 100))

        lines = [
            f"Senast kalibrerad: {self._fmt_saved_value('calibrated_at', 'Aldrig')}",
            f"Metod: {self._fmt_saved_value('method', '-')}",
            f"Markörer senast: {self._fmt_saved_value('marker_count', '0')}",
            f"Reprojection error: {self._fmt_saved_value('reprojection_error_px', '-')}",
            f"Viewport rect: {self._fmt_saved_value('viewport_rect', '-')}",
            f"Kamerastorlek: {self._fmt_saved_value('camera_frame_size', '-')}",
        ]

        y = 152
        for line in lines:
            surf = self.font_body.render(line, True, SOFT)
            screen.blit(surf, (46, y))
            y += 34

        help_lines = [
            "ENTER = visa kalibreringspunkter och starta ny kalibrering",
            "R = rensa sparad kalibrering",
            "ESC = tillbaka",
            "Kalibreringen timeoutar automatiskt om projektorn är av eller fel skärm används.",
        ]
        y += 10
        for line in help_lines:
            surf = self.font_small.render(line, True, WHITE)
            screen.blit(surf, (46, y))
            y += 26

    def _render_status(self, screen: pygame.Surface) -> None:
        color = RED if self.status_is_error else GREEN
        bg = ERR_BG if self.status_is_error else OK_BG
        panel = pygame.Surface((SCREEN_WIDTH - 52, 54), pygame.SRCALPHA)
        panel.fill(bg)
        screen.blit(panel, (26, SCREEN_HEIGHT - 82))
        text = self.font_body.render(self.status_message or " ", True, color)
        screen.blit(text, (40, SCREEN_HEIGHT - 67))

    def render(self, screen: pygame.Surface) -> None:
        screen.fill(self.bg_color)
        self.viewport_rect = load_viewport_rect()

        if self.font_title is None or self.font_body is None or self.font_small is None:
            return

        self._render_viewport_board(screen)
        self._render_idle_info(screen)
        self._render_camera_preview(screen)
        self._render_status(screen)

        if self.state == "capturing":
            cap = self.font_body.render(
                f"Aktiv kalibrering: {self.last_detection_count} markörer hittade",
                True,
                YELLOW,
            )
            screen.blit(cap, (40, 420))

        if self.last_saved.get("camera_quad"):
            quad = self.last_saved["camera_quad"]
            quad_text = self.font_small.render(
                f"Hörn i kamera: TL {quad.get('top_left')}  TR {quad.get('top_right')}",
                True,
                SOFT,
            )
            screen.blit(quad_text, (40, 452))
