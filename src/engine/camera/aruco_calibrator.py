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
import hashlib
import json
import os
import re
import subprocess
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

    def rendered_marker_positions(self):
        """Pixel-center correspondence of the existing integer Pygame blit."""
        size = self.marker_size_px()
        result = {}
        for mid, (x, y) in self.marker_positions().items():
            rect = pygame.Rect(0, 0, size, size)
            rect.center = int(x), int(y)
            result[mid] = (rect.x+(size-1)/2., rect.y+(size-1)/2.)
        return result

    def validate_existing(self, frame_bgr, mapper=None, tolerance_camera_px=3.0):
        """Observe the rendered board against EXISTING mapping; never fit/save.

        Distributed IDs prevent a tiny cluster from claiming viewport coverage.
        Tolerance is a preflight requirement, not a physical-hit threshold.
        """
        mapper = mapper or hit_input
        corners, ids, _ = self.detect_markers(frame_bgr)
        observed = {} if ids is None else {
            int(mid): np.mean(corners[i][0], axis=0).tolist()
            for i, mid in enumerate(ids.flatten())}
        return self.validate_points(observed, mapper, tolerance_camera_px)

    def validate_points(self, observed, mapper, tolerance_camera_px=3.0):
        if not np.isfinite(tolerance_camera_px) or tolerance_camera_px <= 0:
            raise ValueError('positive camera-pixel tolerance required')
        rows = []
        for mid, screen in self.rendered_marker_positions().items():
            if mid not in observed:
                continue
            predicted = mapper._canonical_screen_to_camera(*screen)
            if predicted is None:
                continue
            actual = np.asarray(observed[mid], dtype=float)
            if actual.shape != (2,) or not np.isfinite(actual).all():
                continue
            delta = actual-np.asarray(predicted)
            back = mapper._canonical_camera_to_screen(*predicted)
            roundtrip = float(np.linalg.norm(np.asarray(back)-screen)) if back is not None else None
            if roundtrip is not None and not np.isfinite(roundtrip):
                roundtrip = None
            rows.append(dict(marker_id=mid, game_xy=[screen[0]-self.viewport_rect.x,screen[1]-self.viewport_rect.y],
                client_xy=list(screen), expected_camera_xy=list(predicted), observed_camera_xy=actual.tolist(),
                dx=float(delta[0]),dy=float(delta[1]),error_camera_px=float(np.linalg.norm(delta)),
                roundtrip_client_px=roundtrip))
        layout = [MARKER_LAYOUT[r['marker_id']] for r in rows]
        covered = len(rows)>=8 and all(any((u<.5)==left and (v<.5)==top for u,v in layout)
                                      for left,top in ((True,True),(False,True),(True,False),(False,False)))
        maximum = max((r['error_camera_px'] for r in rows),default=None)
        roundtrips = [r['roundtrip_client_px'] for r in rows]
        roundtrip_max = max(roundtrips,default=None) if None not in roundtrips else None
        passed = covered and maximum <= tolerance_camera_px and roundtrip_max is not None and roundtrip_max <= .05
        return dict(status='PASS' if passed else 'FAIL',scope='MEASURED_PROJECTED_GEOMETRY_ONLY',
            tolerance_camera_px=tolerance_camera_px,distributed_coverage=covered,points=rows,
            max_error_camera_px=maximum,max_roundtrip_client_px=roundtrip_max,
            calibration_modified=False,physical_hit_accuracy_claim=False)

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

        positions = self.rendered_marker_positions()
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

        # Reverse RANSAC may choose different inliers and is NOT an inverse.
        # Preserve one mathematical transform in both directions.
        try:
            H_inv = np.linalg.inv(H)
            scale = H_inv[2, 2] if abs(H_inv[2, 2]) > 1e-12 else np.linalg.norm(H_inv)
            if not np.isfinite(H_inv).all() or not np.isfinite(scale) or scale == 0:
                raise np.linalg.LinAlgError('Nonfinite homography')
            H_inv /= scale
        except (np.linalg.LinAlgError, FloatingPointError):
            return CalibrationResult(success=False, message='Singular homography')

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
            "inverse_method": "matrix_inverse_of_forward_homography",
            "render_correspondence": "integer_blit_pixel_centers",
            "setup_at_calibration": geometry_context(include_calibration=False, native_window=True),
            "reprojection_error_units": "client_pixels",
            "ransac_threshold_client_px": 3.0,
            "inlier_marker_ids": [mid for i, mid in enumerate(matched_ids) if mask is None or mask.ravel()[i]],
        }
        cal_dict['foundation_fit_quality'] = calibration_fit_quality(cal_dict)

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


def geometry_context(include_calibration=True, native_window=False):
    """Safe capture-time identity of existing owners; no desktop-to-camera guess."""
    from src.engine.settings import (load_viewport_rect, load_content_rect,
        load_scanport_rect, load_camera_transform_settings)
    context = dict(viewport=list(load_viewport_rect()),content_rect=list(load_content_rect()),
        scanport=list(load_scanport_rect()) if load_scanport_rect() is not None else None,
        camera_orientation=load_camera_transform_settings(),window=None,
        screen_space='PYGAME_CLIENT_PIXELS_NOT_DESKTOP',window_position_is_not_camera_truth=True)
    if include_calibration:
        context['calibration'] = dict(hit_input.calibration_data)
        context['runtime_homography'] = hit_input.homography.tolist() if hit_input.homography is not None else None
        context['runtime_inverse'] = hit_input.inverse.tolist() if hit_input.inverse is not None else None
    try:
        screen = pygame.display.get_surface()
        if screen is not None:
            from pygame._sdl2.video import Window
            window = Window.from_display_module()
            context['window'] = dict(client_surface_size=list(screen.get_size()),
                sdl_window_size=list(window.size),sdl_desktop_position=list(window.position),
                borderless=bool(window.borderless),display_flags=int(screen.get_flags()),
                display_index=int(window.display_index),
                driver=pygame.display.get_driver(),desktop_sizes=pygame.display.get_desktop_sizes(),
                decoration_extents='UNAVAILABLE',position_semantics='SDL_WINDOW_POSITION_NOT_CAMERA_COORDINATES')
            if native_window and context['window']['driver'] == 'x11':
                context['window'].update(x11_window_provenance())
    except Exception as exc:
        context['window_error'] = type(exc).__name__
    context['identity_sha256'] = hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    return context


def x11_window_provenance():
    """Read the display-owned client XID; never move/wrap/destroy a native window.

    Optional command-line X11 introspection, only for calibration/diagnostics.
    Failure stays explicit; neither unknown extents nor desktop XY becomes H.
    """
    info = {}
    try:
        xid = str(int(pygame.display.get_wm_info()['window']))
        env = dict(os.environ, LC_ALL='C')
        prop = subprocess.run(['xprop', '-id', xid, '_NET_FRAME_EXTENTS'],
            capture_output=True, text=True, timeout=1, env=env, check=True).stdout
        match = re.search(r'=\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)', prop)
        if match:
            left, right, top, bottom = map(int, match.groups())
            info['decoration_extents'] = dict(left=left,right=right,top=top,bottom=bottom)
        raw = subprocess.run(['xwininfo', '-id', xid], capture_output=True,
            text=True, timeout=1, env=env, check=True).stdout
        x = re.search(r'Absolute upper-left X:\s*(-?\d+)', raw)
        y = re.search(r'Absolute upper-left Y:\s*(-?\d+)', raw)
        if x and y:
            info['x11_client_origin'] = [int(x[1]), int(y[1])]
            info['position_semantics'] = 'SDL_POSITION_WITH_X11_CLIENT_ROOT_CROSSCHECK'
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        info['native_geometry_unavailable'] = type(exc).__name__
    return info


def _geometry_setup_value(value):
    """Canonicalize JSON/runtime container representation for setup comparison."""
    if isinstance(value, tuple):
        return [_geometry_setup_value(v) for v in value]
    if isinstance(value, list):
        return [_geometry_setup_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _geometry_setup_value(v) for k, v in value.items()}
    return value


def compare_geometry_setup(calibration_setup, current):
    """Compare setup provenance, not camera coordinates or arbitrary identity hash."""
    differences = []
    for key in ('viewport', 'content_rect', 'camera_orientation', 'screen_space'):
        a, b = calibration_setup.get(key), current.get(key)
        if a is None or _geometry_setup_value(a) != _geometry_setup_value(b):
            differences.append(dict(field=key, calibration=a, current=b))
    old, new = calibration_setup.get('window') or {}, current.get('window') or {}
    for key in ('client_surface_size','sdl_window_size','sdl_desktop_position',
                'borderless','display_index','driver','desktop_sizes'):
        a, b = old.get(key), new.get(key)
        if a is None or _geometry_setup_value(a) != _geometry_setup_value(b):
            differences.append(dict(field='window.'+key,calibration=a,current=b))
    # Surface flag bits include non-geometric focus/software flags. Compare only
    # modes that can change client/display mapping, not the full incidental mask.
    mask = pygame.FULLSCREEN | pygame.SCALED | pygame.RESIZABLE | pygame.NOFRAME
    a, b = old.get('display_flags'), new.get('display_flags')
    if a is None or b is None or (a & mask) != (b & mask):
        differences.append(dict(field='window.geometry_flags',calibration=a,current=b))
    for label, window in (('calibration',old),('current',new)):
        if 'x11_client_origin' in window and window['x11_client_origin'] != window.get('sdl_desktop_position'):
            differences.append(dict(field=label+'.SDL_vs_X11_client',
                calibration=window.get('sdl_desktop_position'),current=window['x11_client_origin']))
    return dict(status='SETUP_MISMATCH' if differences else 'MATCH', differences=differences,
        semantics='CLIENT_PLACEMENT_PROVENANCE_NOT_PHYSICAL_CALIBRATION')


def calibration_fit_quality(calibration, tolerance_camera_px=3.0):
    """Evaluate ALL saved correspondences in camera pixels, including RANSAC rejects.

    Legacy reprojection_error_px is a mean in CLIENT pixels with a 25px save
    ceiling. That metric never certifies the foundation's 3 CAMERA pixel bound.
    This audit neither refits nor changes a loaded calibration.
    """
    try:
        cam = np.asarray(calibration['camera_points'],np.float64).reshape(-1,2)
        client = np.asarray(calibration['viewport_points'],np.float64).reshape(-1,2)
        ids = calibration['matched_ids']
        H = np.asarray(calibration['homography'],np.float64)
        inv = np.asarray(calibration['inverse_homography'],np.float64)
        if H.shape != (3,3) or inv.shape != (3,3) or len(cam) != len(client) or len(ids) != len(cam) or len(cam) < 8:
            raise ValueError('insufficient/malformed correspondences')
        if not all(np.isfinite(a).all() for a in (cam,client,H,inv)):
            raise ValueError('nonfinite correspondences')
        predicted = cv2.perspectiveTransform(client[:,None,:],inv)[:,0,:]
        back = cv2.perspectiveTransform(predicted[:,None,:],H)[:,0,:]
        camera_error = np.linalg.norm(predicted-cam,axis=1)
        roundtrip = np.linalg.norm(back-client,axis=1)
        layout = [MARKER_LAYOUT[mid] for mid in ids if mid in MARKER_LAYOUT]
        coverage = all(any((u<.5)==left and (v<.5)==top for u,v in layout)
                       for left,top in ((True,True),(False,True),(True,False),(False,False)))
        finite = np.isfinite(camera_error).all() and np.isfinite(roundtrip).all()
        if not finite:raise ValueError('invalid projected coordinates')
        return dict(status='PASS' if coverage and max(camera_error)<=tolerance_camera_px and max(roundtrip)<=.05 else 'CALIBRATION_FIT_POOR',
            tolerance_camera_px=tolerance_camera_px,distributed_coverage=coverage,
            mean_camera_px=float(np.mean(camera_error)),max_camera_px=float(max(camera_error)),
            max_roundtrip_client_px=float(max(roundtrip)),
            per_marker_camera_px={str(mid):float(e) for mid,e in zip(ids,camera_error)},
            evaluated_all_markers=True,calibration_modified=False)
    except (KeyError,ValueError,TypeError,cv2.error) as exc:
        return dict(status='CALIBRATION_FIT_POOR',reason=str(exc),calibration_modified=False)
