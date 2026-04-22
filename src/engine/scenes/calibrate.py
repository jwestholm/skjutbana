from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pygame

from config import LOADING_SCREEN_PATH, SCREEN_HEIGHT, SCREEN_WIDTH
from src.engine.camera.camera_manager import camera_manager
from src.engine.input.hit_input import hit_input
from src.engine.scene import Scene, SceneSwitch
from src.engine.settings import (
    load_camera_calibration,
    load_viewport_rect,
    save_camera_calibration,
    save_viewport_rect,
)


WHITE = (240, 240, 240)
SOFT_WHITE = (210, 210, 210)
BLACK = (15, 15, 15)
DARK = (40, 40, 40)
GREEN = (120, 255, 120)
RED = (255, 100, 100)
YELLOW = (255, 220, 80)
CYAN = (80, 220, 255)
PANEL_BG = (0, 0, 0, 175)


@dataclass(frozen=True)
class MarkerSpec:
    marker_id: int
    u: float
    v: float


class CalibrateViewportScene(Scene):
    """
    Kombinerad scen för:
    1) manuell justering av viewport-rect (befintlig funktion)
    2) kamerakalibrering mot viewporten via projicerade ArUco-markörer

    TAB växlar läge mellan MANUAL och CAMERA.
    """

    wants_camera_preview = True
    wants_hit_scanning = False
    wants_mouse_simulated_hits = False

    MODE_CAMERA = "camera"
    MODE_MANUAL = "manual"

    MARKER_SPECS: tuple[MarkerSpec, ...] = (
        MarkerSpec(0, 0.08, 0.08),
        MarkerSpec(1, 0.50, 0.08),
        MarkerSpec(2, 0.92, 0.08),
        MarkerSpec(3, 0.92, 0.50),
        MarkerSpec(4, 0.92, 0.92),
        MarkerSpec(5, 0.50, 0.92),
        MarkerSpec(6, 0.08, 0.92),
        MarkerSpec(7, 0.08, 0.50),
        MarkerSpec(8, 0.50, 0.50),
    )

    def __init__(self) -> None:
        self.mode = self.MODE_CAMERA

        self.bg = None
        self.overlay = None
        self.board_surface: pygame.Surface | None = None
        self.board_hash: tuple[int, int, int, int] | None = None
        self.font = None
        self.small = None
        self.tiny = None

        self.original_viewport: pygame.Rect | None = None
        self.rect: pygame.Rect | None = None
        self.move_step = 10
        self.size_step = 20

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)

        self.detected_camera_points: dict[int, tuple[float, float]] = {}
        self.detected_screen_points: dict[int, tuple[float, float]] = {}
        self.last_homography: np.ndarray | None = None
        self.last_reprojection_error_px: float | None = None
        self.last_detect_count = 0
        self.last_status = "Ingen kamerakalibrering ännu."
        self.preview_surface: pygame.Surface | None = None
        self.last_frame_size: tuple[int, int] | None = None
        self.last_saved_summary = ""

    def on_enter(self) -> None:
        bg = pygame.image.load(str(LOADING_SCREEN_PATH)).convert()
        self.bg = pygame.transform.smoothscale(bg, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.overlay.fill((0, 0, 0, 140))

        self.font = pygame.font.Font(None, 44)
        self.small = pygame.font.Font(None, 26)
        self.tiny = pygame.font.Font(None, 22)

        self.original_viewport = load_viewport_rect()
        self.rect = self.original_viewport.copy()
        self._rebuild_board_if_needed(force=True)

        calibration = load_camera_calibration() or {}
        if calibration.get("is_calibrated"):
            error = calibration.get("reprojection_error_px")
            if isinstance(error, (int, float)):
                self.last_saved_summary = f"Senast sparad reprojection error: {float(error):.2f}px"
            else:
                self.last_saved_summary = "Kalibrering hittad i settings."
        else:
            self.last_saved_summary = "Ingen tidigare kalibrering sparad."

    def handle_event(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_ESCAPE:
            return self._go_back()

        if event.key == pygame.K_TAB:
            self.mode = self.MODE_MANUAL if self.mode == self.MODE_CAMERA else self.MODE_CAMERA
            self.last_status = (
                "Bytte till manuell viewport-justering."
                if self.mode == self.MODE_MANUAL
                else "Bytte till kamerakalibrering med markörer."
            )
            return None

        if event.key == pygame.K_r:
            if self.mode == self.MODE_MANUAL:
                assert self.original_viewport is not None
                self.rect = self.original_viewport.copy()
                self.last_status = "Återställde viewport till sparat värde."
                self._rebuild_board_if_needed(force=True)
            else:
                self.detected_camera_points.clear()
                self.detected_screen_points.clear()
                self.last_homography = None
                self.last_reprojection_error_px = None
                self.last_status = "Rensade aktuell kamerakalibrering i minnet."
            return None

        if self.mode == self.MODE_MANUAL:
            return self._handle_manual_event(event)
        return self._handle_camera_event(event)

    def update(self, dt: float):
        del dt
        self._rebuild_board_if_needed()
        if self.mode == self.MODE_CAMERA:
            self._update_camera_calibration_preview()
        return None

    def render(self, screen: pygame.Surface) -> None:
        if self.mode == self.MODE_MANUAL:
            self._render_manual(screen)
        else:
            self._render_camera(screen)

    # ------------------------------------------------------------
    # Mode handlers
    # ------------------------------------------------------------
    def _handle_manual_event(self, event: pygame.event.Event):
        assert self.rect is not None

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            save_viewport_rect(self.rect)
            self.original_viewport = self.rect.copy()
            self.last_status = "Viewport sparad."
            self._rebuild_board_if_needed(force=True)
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
            self.rect.w += self.size_step
            self.rect.h += self.size_step
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.rect.w -= self.size_step
            self.rect.h -= self.size_step
        else:
            return None

        self.rect.w = max(200, self.rect.w)
        self.rect.h = max(200, self.rect.h)
        self.rect.x = max(0, self.rect.x)
        self.rect.y = max(0, self.rect.y)
        if self.rect.right > SCREEN_WIDTH:
            self.rect.x = SCREEN_WIDTH - self.rect.w
        if self.rect.bottom > SCREEN_HEIGHT:
            self.rect.y = SCREEN_HEIGHT - self.rect.h
        self._rebuild_board_if_needed(force=True)
        return None

    def _handle_camera_event(self, event: pygame.event.Event):
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.last_homography is None or len(self.detected_camera_points) < 4:
                self.last_status = "För få markörer hittade för att spara kalibrering."
                return None
            self._save_camera_calibration()
            return None
        return None

    # ------------------------------------------------------------
    # Board generation / detection
    # ------------------------------------------------------------
    def _rebuild_board_if_needed(self, force: bool = False) -> None:
        assert self.rect is not None
        key = (self.rect.x, self.rect.y, self.rect.w, self.rect.h)
        if not force and self.board_surface is not None and key == self.board_hash:
            return
        self.board_hash = key
        self.board_surface = self._build_board_surface(self.rect)

    def _screen_point_for_spec(self, rect: pygame.Rect, spec: MarkerSpec) -> tuple[float, float]:
        return (
            float(rect.x + spec.u * rect.w),
            float(rect.y + spec.v * rect.h),
        )

    def _marker_size_for_viewport(self, rect: pygame.Rect) -> int:
        size = int(min(rect.w, rect.h) * 0.12)
        return max(56, min(size, 180))

    def _build_board_surface(self, rect: pygame.Rect) -> pygame.Surface:
        board = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        board.fill((0, 0, 0))

        pygame.draw.rect(board, (255, 255, 255), rect)
        pygame.draw.rect(board, (0, 180, 0), rect, 4)

        marker_size = self._marker_size_for_viewport(rect)
        label_font = pygame.font.Font(None, 26)
        center_font = pygame.font.Font(None, 32)

        for spec in self.MARKER_SPECS:
            cx, cy = self._screen_point_for_spec(rect, spec)
            marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
            cv2.aruco.generateImageMarker(self.dictionary, spec.marker_id, marker_size, marker_img, 1)
            rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB)
            surf = pygame.image.frombuffer(rgb.tobytes(), (marker_size, marker_size), "RGB")
            top_left = (int(round(cx - marker_size / 2)), int(round(cy - marker_size / 2)))
            board.blit(surf, top_left)
            pygame.draw.rect(board, BLACK, pygame.Rect(top_left[0], top_left[1], marker_size, marker_size), 2)

            label = label_font.render(str(spec.marker_id), True, CYAN)
            label_rect = label.get_rect(center=(int(round(cx)), int(round(cy + marker_size / 2 + 16))))
            board.blit(label, label_rect)

        cross_center = (rect.centerx, rect.centery)
        pygame.draw.circle(board, RED, cross_center, 10, 2)
        pygame.draw.line(board, RED, (cross_center[0] - 18, cross_center[1]), (cross_center[0] + 18, cross_center[1]), 2)
        pygame.draw.line(board, RED, (cross_center[0], cross_center[1] - 18), (cross_center[0], cross_center[1] + 18), 2)

        title = center_font.render("Kamerakalibrering: projicerade ArUco-markörer", True, BLACK)
        board.blit(title, (rect.x + 16, max(8, rect.y - 34 if rect.y > 40 else rect.y + 8)))

        return board.convert()

    def _detect_markers(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, _rej = self.detector.detectMarkers(gray)
        else:
            corners, ids, _rej = cv2.aruco.detectMarkers(gray, self.dictionary)
        return gray, corners, ids

    def _update_camera_calibration_preview(self) -> None:
        frame_bgr = camera_manager.get_latest_frame()
        if frame_bgr is None:
            self.preview_surface = None
            self.last_status = "Ingen kamerabild tillgänglig ännu."
            return

        self.last_frame_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
        gray, corners, ids = self._detect_markers(frame_bgr)
        preview = frame_bgr.copy()
        self.detected_camera_points = {}
        self.detected_screen_points = {}
        self.last_detect_count = 0
        self.last_homography = None
        self.last_reprojection_error_px = None

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(preview, corners, ids)
            assert self.rect is not None
            wanted_ids = {spec.marker_id: spec for spec in self.MARKER_SPECS}
            for idx, marker_id_arr in enumerate(ids):
                marker_id = int(marker_id_arr[0])
                spec = wanted_ids.get(marker_id)
                if spec is None:
                    continue
                marker_corners = corners[idx][0]
                center_x = float(np.mean(marker_corners[:, 0]))
                center_y = float(np.mean(marker_corners[:, 1]))
                self.detected_camera_points[marker_id] = (center_x, center_y)
                self.detected_screen_points[marker_id] = self._screen_point_for_spec(self.rect, spec)
                cv2.circle(preview, (int(round(center_x)), int(round(center_y))), 6, (0, 255, 0), 2)
                cv2.putText(
                    preview,
                    f"{marker_id}",
                    (int(round(center_x)) + 8, int(round(center_y)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        self.last_detect_count = len(self.detected_camera_points)

        if len(self.detected_camera_points) >= 4:
            ordered_ids = sorted(self.detected_camera_points.keys())
            camera_pts = np.array([self.detected_camera_points[mid] for mid in ordered_ids], dtype=np.float32)
            screen_pts = np.array([self.detected_screen_points[mid] for mid in ordered_ids], dtype=np.float32)
            H, mask = cv2.findHomography(camera_pts, screen_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if H is not None:
                self.last_homography = H.astype(np.float32)
                projected = cv2.perspectiveTransform(camera_pts.reshape(-1, 1, 2), self.last_homography).reshape(-1, 2)
                errors = np.linalg.norm(projected - screen_pts, axis=1)
                self.last_reprojection_error_px = float(np.mean(errors)) if len(errors) else 0.0
                inliers = int(mask.sum()) if mask is not None else len(ordered_ids)
                self.last_status = (
                    f"Hittade {len(ordered_ids)} markörer. Homography klar. Inliers={inliers} error={self.last_reprojection_error_px:.2f}px"
                )
            else:
                self.last_status = f"Hittade {len(self.detected_camera_points)} markörer men homography misslyckades."
        else:
            self.last_status = f"Hittade {len(self.detected_camera_points)} markörer. Behöver minst 4."

        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        self.preview_surface = pygame.image.frombuffer(
            preview_rgb.tobytes(),
            (preview_rgb.shape[1], preview_rgb.shape[0]),
            "RGB",
        )

    def _save_camera_calibration(self) -> None:
        assert self.rect is not None
        assert self.last_homography is not None

        ordered_ids = sorted(self.detected_camera_points.keys())
        calibration = {
            "is_calibrated": True,
            "method": "aruco_viewport_v1",
            "prefer_homography": True,
            "dictionary": "DICT_4X4_50",
            "marker_ids": ordered_ids,
            "homography": self.last_homography.tolist(),
            "inverse_homography": np.linalg.inv(self.last_homography).astype(np.float32).tolist(),
            "viewport_rect": [int(self.rect.x), int(self.rect.y), int(self.rect.w), int(self.rect.h)],
            "camera_points": [
                [float(self.detected_camera_points[mid][0]), float(self.detected_camera_points[mid][1])]
                for mid in ordered_ids
            ],
            "screen_points": [
                [float(self.detected_screen_points[mid][0]), float(self.detected_screen_points[mid][1])]
                for mid in ordered_ids
            ],
            "reprojection_error_px": float(self.last_reprojection_error_px or 0.0),
            "frame_size": [
                int(self.last_frame_size[0]) if self.last_frame_size else 0,
                int(self.last_frame_size[1]) if self.last_frame_size else 0,
            ],
        }
        save_camera_calibration(calibration)
        hit_input.reload_calibration()
        self.last_saved_summary = f"Sparad kalibrering. Error {float(self.last_reprojection_error_px or 0.0):.2f}px"
        self.last_status = self.last_saved_summary

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------
    def _go_back(self):
        from src.engine.scenes.menu import MenuScene

        return SceneSwitch(MenuScene())

    def _render_manual(self, screen: pygame.Surface) -> None:
        assert self.rect is not None
        screen.blit(self.bg, (0, 0))
        screen.blit(self.overlay, (0, 0))

        title = self.font.render("Justera skjutfält / rityta (manuellt)", True, WHITE)
        screen.blit(title, (40, 28))

        hint = self.small.render(
            "TAB = byt till kamerakalibrering | Pilar = flytta | +/- = storlek | ENTER = spara viewport | ESC = tillbaka",
            True,
            SOFT_WHITE,
        )
        screen.blit(hint, (40, 74))

        pygame.draw.rect(screen, GREEN, self.rect, 4)
        info = self.small.render(
            f"viewport x={self.rect.x} y={self.rect.y} w={self.rect.w} h={self.rect.h}",
            True,
            WHITE,
        )
        screen.blit(info, (40, SCREEN_HEIGHT - 42))

        status = self.tiny.render(self.last_status, True, YELLOW)
        screen.blit(status, (40, SCREEN_HEIGHT - 68))

    def _render_camera(self, screen: pygame.Surface) -> None:
        assert self.board_surface is not None
        assert self.rect is not None

        screen.blit(self.board_surface, (0, 0))

        header = pygame.Surface((SCREEN_WIDTH, 98), pygame.SRCALPHA)
        header.fill(PANEL_BG)
        screen.blit(header, (0, 0))
        title = self.font.render("Kalibrera viewport mot kamera", True, WHITE)
        screen.blit(title, (26, 16))
        hint = self.small.render(
            "TAB = manuell viewport | ENTER = spara homography | R = rensa aktuell lösning | ESC = tillbaka",
            True,
            SOFT_WHITE,
        )
        screen.blit(hint, (28, 56))

        info_panel = pygame.Surface((530, 186), pygame.SRCALPHA)
        info_panel.fill(PANEL_BG)
        screen.blit(info_panel, (26, SCREEN_HEIGHT - 212))

        lines = [
            f"Viewport: x={self.rect.x} y={self.rect.y} w={self.rect.w} h={self.rect.h}",
            f"Markörer hittade: {self.last_detect_count} / {len(self.MARKER_SPECS)}",
            self.last_status,
            self.last_saved_summary,
            "Homography används som primär träffmapping efter sparning.",
            "Några pixels skillnad i mus-testet är normalt efter verklig kamerakalibrering.",
        ]
        if self.last_reprojection_error_px is not None:
            lines.append(f"Aktuell reprojection error: {self.last_reprojection_error_px:.2f}px")
        if self.last_frame_size is not None:
            lines.append(f"Kameraframe: {self.last_frame_size[0]}x{self.last_frame_size[1]}")

        y = SCREEN_HEIGHT - 200
        for idx, line in enumerate(lines[:7]):
            font = self.tiny if idx >= 2 else self.small
            color = GREEN if "error" in line.lower() else WHITE
            if idx == 2:
                color = YELLOW
            surf = font.render(line, True, color)
            screen.blit(surf, (42, y))
            y += 24 if font == self.tiny else 28

        if self.preview_surface is not None:
            preview = self.preview_surface
            max_w = 480
            max_h = 270
            pw, ph = preview.get_size()
            scale = min(max_w / pw, max_h / ph)
            scaled = pygame.transform.smoothscale(preview, (max(1, int(pw * scale)), max(1, int(ph * scale))))
            px = SCREEN_WIDTH - scaled.get_width() - 26
            py = SCREEN_HEIGHT - scaled.get_height() - 26
            shadow = pygame.Surface((scaled.get_width() + 12, scaled.get_height() + 12), pygame.SRCALPHA)
            shadow.fill(PANEL_BG)
            screen.blit(shadow, (px - 6, py - 6))
            screen.blit(scaled, (px, py))
            pygame.draw.rect(screen, CYAN, pygame.Rect(px, py, scaled.get_width(), scaled.get_height()), 2)
            label = self.tiny.render("Kamerabild + markerdetektion", True, CYAN)
            screen.blit(label, (px + 8, py + 8))

