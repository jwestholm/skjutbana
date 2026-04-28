"""
ArUco-based camera-to-viewport calibration engine.

Reusable module — used by both the manual calibration scene and
auto-calibration in AI training. No pygame rendering here; callers
handle display.

Usage:
    cal = ArucoCalibrator(viewport_rect)
    cal.detect_and_calibrate(frame_bgr)  # returns CalibrationResult | None
    cal.render_markers(screen)           # draws markers on a pygame surface
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pygame

from src.engine.settings import load_viewport_rect, save_camera_calibration
from src.engine.input.hit_input import hit_input


# 24 markers distributed across the viewport for robust homography.
# Positions as (u, v) fractions of viewport width/height.
MARKER_LAYOUT: Dict[int, Tuple[float, float]] = {
    # Corners
    0: (0.05, 0.05), 1: (0.95, 0.05), 2: (0.95, 0.95), 3: (0.05, 0.95),
    # Edges — top/bottom
    4: (0.30, 0.05), 5: (0.70, 0.05), 6: (0.30, 0.95), 7: (0.70, 0.95),
    # Edges — left/right
    8: (0.05, 0.30), 9: (0.05, 0.70), 10: (0.95, 0.30), 11: (0.95, 0.70),
    # Inner grid
    12: (0.25, 0.25), 13: (0.75, 0.25), 14: (0.75, 0.75), 15: (0.25, 0.75),
    # Center cross
    16: (0.50, 0.05), 17: (0.50, 0.95), 18: (0.05, 0.50), 19: (0.95, 0.50),
    # Inner center
    20: (0.50, 0.25), 21: (0.50, 0.75), 22: (0.25, 0.50), 23: (0.75, 0.50),
}


@dataclass
class CalibrationResult:
    """Result of a successful calibration attempt."""
    success: bool
    marker_count: int = 0
    inliers: int = 0
    reprojection_error_px: float = 9999.0
    per_marker_errors: Dict[str, float] = field(default_factory=dict)
    matched_ids: List[int] = field(default_factory=list)
    calibration_dict: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


class ArucoCalibrator:
    """
    Stateless calibration engine. Create one, call methods as needed.

    Parameters
    ----------
    viewport_rect : pygame.Rect
        The viewport rectangle in screen coordinates.
    min_markers : int
        Minimum markers required for calibration.
    max_reproj_error : float
        Maximum acceptable reprojection error in pixels.
    """

    def __init__(
        self,
        viewport_rect: Optional[pygame.Rect] = None,
        min_markers: int = 4,
        max_reproj_error: float = 25.0,
    ) -> None:
        self.viewport_rect = viewport_rect or load_viewport_rect()
        self.min_markers = min_markers
        self.max_reproj_error = max_reproj_error

        # ArUco state
        self.available = False
        self._aruco_dict = None
        self._aruco_detector = None
        self._aruco_params = None
        self._init_aruco()

    def _init_aruco(self) -> None:
        if not hasattr(cv2, "aruco"):
            return
        dict_name = getattr(cv2.aruco, "DICT_4X4_50", None)
        if dict_name is None:
            return
        try:
            if hasattr(cv2.aruco, "getPredefinedDictionary"):
                self._aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)
            else:
                self._aruco_dict = cv2.aruco.Dictionary_get(dict_name)

            if hasattr(cv2.aruco, "DetectorParameters"):
                self._aruco_params = cv2.aruco.DetectorParameters()
            else:
                self._aruco_params = cv2.aruco.DetectorParameters_create()

            if hasattr(cv2.aruco, "ArucoDetector"):
                self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

            self.available = True
        except Exception:
            self.available = False

    # ------------------------------------------------------------------
    # Marker positions
    # ------------------------------------------------------------------

    def marker_positions(self) -> Dict[int, Tuple[float, float]]:
        """Screen-space positions for all 24 markers based on current viewport."""
        rect = self.viewport_rect
        return {
            mid: (float(rect.x + u * rect.w), float(rect.y + v * rect.h))
            for mid, (u, v) in MARKER_LAYOUT.items()
        }

    def marker_size_px(self) -> int:
        m = min(self.viewport_rect.w, self.viewport_rect.h)
        return max(36, min(80, int(m * 0.06)))

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_markers(self, frame_bgr: np.ndarray):
        """Detect ArUco markers in a BGR camera frame.
        Returns (corners, ids, rejected) — same as cv2.aruco API."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._aruco_detector is not None:
            return self._aruco_detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(gray, self._aruco_dict, parameters=self._aruco_params)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def detect_and_calibrate(self, frame_bgr: np.ndarray) -> CalibrationResult:
        """
        One-shot: detect markers in frame, compute homography, return result.
        Does NOT save — caller decides whether to save.
        """
        if not self.available or self._aruco_dict is None:
            return CalibrationResult(success=False, message="ArUco not available")

        try:
            corners, ids, _ = self.detect_markers(frame_bgr)
        except Exception as exc:
            return CalibrationResult(success=False, message=f"Detection error: {exc}")

        n_detected = 0 if ids is None else int(len(ids))
        if n_detected < self.min_markers:
            return CalibrationResult(
                success=False, marker_count=n_detected,
                message=f"Only {n_detected} markers (need {self.min_markers})",
            )

        positions = self.marker_positions()
        camera_points: List[List[float]] = []
        viewport_points: List[List[float]] = []
        matched_ids: List[int] = []

        for idx, marker_id in enumerate(ids.flatten().tolist()):
            if marker_id not in positions:
                continue
            center = np.mean(corners[idx][0], axis=0).astype(np.float32)
            camera_points.append([float(center[0]), float(center[1])])
            vx, vy = positions[marker_id]
            viewport_points.append([float(vx), float(vy)])
            matched_ids.append(int(marker_id))

        if len(camera_points) < self.min_markers:
            return CalibrationResult(
                success=False, marker_count=len(camera_points),
                message=f"Only {len(camera_points)} matched markers",
            )

        cam_np = np.array(camera_points, dtype=np.float32)
        vp_np = np.array(viewport_points, dtype=np.float32)

        H, mask = cv2.findHomography(cam_np, vp_np, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            return CalibrationResult(success=False, message="findHomography failed")

        H_inv, _ = cv2.findHomography(vp_np, cam_np, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        # Reprojection error
        projected = cv2.perspectiveTransform(cam_np.reshape(-1, 1, 2), H).reshape(-1, 2)
        diffs = projected - vp_np
        distances = np.sqrt(np.sum(diffs * diffs, axis=1))
        reproj = float(np.mean(distances)) if len(distances) else 9999.0

        if reproj > self.max_reproj_error:
            return CalibrationResult(
                success=False, marker_count=len(camera_points),
                reprojection_error_px=reproj,
                message=f"Reproj error too high: {reproj:.1f}px",
            )

        # Per-marker errors
        per_marker_errors = {}
        for i, mid in enumerate(matched_ids):
            per_marker_errors[str(mid)] = float(distances[i])

        inliers = int(mask.sum()) if mask is not None else len(camera_points)
        frame_h, frame_w = frame_bgr.shape[:2]
        vp = self.viewport_rect

        cal_dict = {
            "method": "aruco_viewport_board",
            "prefer_homography": True,
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            "homography": H.tolist(),
            "inverse_homography": H_inv.tolist() if H_inv is not None else None,
            "marker_count": int(len(camera_points)),
            "inliers": inliers,
            "reprojection_error_px": float(reproj),
            "per_marker_errors": per_marker_errors,
            "camera_points": camera_points,
            "viewport_points": viewport_points,
            "matched_ids": matched_ids,
            "camera_frame_size": [int(frame_w), int(frame_h)],
            "viewport_rect": [int(vp.x), int(vp.y), int(vp.w), int(vp.h)],
        }

        return CalibrationResult(
            success=True,
            marker_count=len(camera_points),
            inliers=inliers,
            reprojection_error_px=reproj,
            per_marker_errors=per_marker_errors,
            matched_ids=matched_ids,
            calibration_dict=cal_dict,
            message=f"{len(camera_points)} markers, {inliers} inliers, reproj={reproj:.1f}px",
        )

    # ------------------------------------------------------------------
    # Save / apply
    # ------------------------------------------------------------------

    def save_and_apply(self, result: CalibrationResult) -> None:
        """Save calibration to settings and reload hit_input."""
        if not result.success or not result.calibration_dict:
            return
        save_camera_calibration(result.calibration_dict)
        hit_input.reload_calibration()

    # ------------------------------------------------------------------
    # Rendering helpers (pygame)
    # ------------------------------------------------------------------

    def render_markers(self, screen: pygame.Surface) -> None:
        """Draw all 24 ArUco markers on a white background within the viewport."""
        vp = self.viewport_rect
        screen.fill((245, 245, 245))
        pygame.draw.rect(screen, (245, 245, 245), vp)

        if self._aruco_dict is None:
            return

        positions = self.marker_positions()
        size = self.marker_size_px()

        for marker_id, center in positions.items():
            surf = self._make_marker_surface(marker_id, size)
            if surf is not None:
                rect = surf.get_rect(center=(int(center[0]), int(center[1])))
                screen.blit(surf, rect.topleft)

    def _make_marker_surface(self, marker_id: int, size: int) -> Optional[pygame.Surface]:
        if self._aruco_dict is None:
            return None
        try:
            if hasattr(cv2.aruco, "generateImageMarker"):
                img = cv2.aruco.generateImageMarker(self._aruco_dict, marker_id, size)
            else:
                img = np.zeros((size, size), dtype=np.uint8)
                cv2.aruco.drawMarker(self._aruco_dict, marker_id, size, img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            rgb = np.transpose(rgb, (1, 0, 2))
            return pygame.surfarray.make_surface(rgb).convert()
        except Exception:
            return None
