"""Perspective-safe analysis ROI helpers for V2.22.1.

The live detector continues to use *full camera coordinates* as its canonical
coordinate plane.  This module only computes a smaller camera-image crop and a
safe inner playfield polygon in that crop.  A candidate detected at local crop
coordinates is translated back to full camera coordinates before tracking,
AI/resolver scoring, known-hole lookup, or HitInput homography conversion.

The edge guard is defined in screen/game pixels *before* inverse homography.
That makes the guard follow the real projected quadrilateral even when the
camera views the wall at an angle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

SCHEMA_VERSION = "2.22.1"


@dataclass(frozen=True)
class AnalysisGeometryV2221:
    frame_height: int
    frame_width: int
    crop_x0: int
    crop_y0: int
    crop_x1: int
    crop_y1: int
    outer_camera_polygon: np.ndarray
    safe_camera_polygon: np.ndarray
    safe_mask_local: np.ndarray
    mode: str
    guard_screen_px: float
    crop_padding_camera_px: int

    @property
    def crop_width(self) -> int:
        return max(0, int(self.crop_x1) - int(self.crop_x0))

    @property
    def crop_height(self) -> int:
        return max(0, int(self.crop_y1) - int(self.crop_y0))

    @property
    def crop_pixels(self) -> int:
        return int(self.crop_width * self.crop_height)

    @property
    def frame_pixels(self) -> int:
        return int(max(0, self.frame_height) * max(0, self.frame_width))

    @property
    def crop_fraction(self) -> float:
        return float(self.crop_pixels) / float(max(1, self.frame_pixels))

    @property
    def safe_pixels(self) -> int:
        return int(np.count_nonzero(self.safe_mask_local))

    def crop_array(self, array: np.ndarray | None) -> np.ndarray | None:
        if array is None:
            return None
        arr = np.asarray(array)
        if arr.ndim < 2:
            return arr
        # Crop only arrays that are still in the canonical full-camera plane.
        if arr.shape[0] == self.frame_height and arr.shape[1] == self.frame_width:
            return arr[self.crop_y0:self.crop_y1, self.crop_x0:self.crop_x1]
        return arr

    def local_to_camera(self, x: float, y: float) -> tuple[float, float]:
        return float(x) + float(self.crop_x0), float(y) + float(self.crop_y0)

    def camera_to_local(self, x: float, y: float) -> tuple[float, float]:
        return float(x) - float(self.crop_x0), float(y) - float(self.crop_y0)


def _as_rect_xywh(rect: Sequence[float] | object) -> tuple[float, float, float, float]:
    if isinstance(rect, (tuple, list)) and len(rect) >= 4:
        return float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
    return (
        float(getattr(rect, "x")),
        float(getattr(rect, "y")),
        float(getattr(rect, "w", getattr(rect, "width"))),
        float(getattr(rect, "h", getattr(rect, "height"))),
    )


def _screen_rect_points(rect_xywh: Sequence[float]) -> np.ndarray:
    x, y, w, h = (float(v) for v in rect_xywh[:4])
    return np.asarray(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )


def _shrink_rect(rect_xywh: Sequence[float], guard: float) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in rect_xywh[:4])
    max_guard = max(0.0, min(w, h) * 0.20)
    g = max(0.0, min(float(guard), max_guard))
    # Never collapse a very small viewport.  If necessary reduce the guard.
    if w - 2.0 * g < 4.0:
        g = max(0.0, (w - 4.0) * 0.5)
    if h - 2.0 * g < 4.0:
        g = min(g, max(0.0, (h - 4.0) * 0.5))
    return x + g, y + g, max(1.0, w - 2.0 * g), max(1.0, h - 2.0 * g)


def _clip_polygon(points: np.ndarray, frame_width: int, frame_height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, max(0.0, float(frame_width - 1)))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, max(0.0, float(frame_height - 1)))
    return pts


def _bbox_from_polygon(
    polygon: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    pad = max(0, int(padding))
    x0 = max(0, int(np.floor(float(np.min(pts[:, 0])))) - pad)
    y0 = max(0, int(np.floor(float(np.min(pts[:, 1])))) - pad)
    x1 = min(frame_width, int(np.ceil(float(np.max(pts[:, 0])))) + 1 + pad)
    y1 = min(frame_height, int(np.ceil(float(np.max(pts[:, 1])))) + 1 + pad)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, frame_width, frame_height
    return x0, y0, x1, y1


def _local_mask(
    polygon_camera: np.ndarray,
    *,
    crop_x0: int,
    crop_y0: int,
    crop_width: int,
    crop_height: int,
) -> np.ndarray:
    mask = np.zeros((max(1, crop_height), max(1, crop_width)), dtype=np.uint8)
    local = np.asarray(polygon_camera, dtype=np.float32).reshape(-1, 2).copy()
    local[:, 0] -= float(crop_x0)
    local[:, 1] -= float(crop_y0)
    poly_i = np.round(local).astype(np.int32)
    if len(poly_i) >= 3 and abs(float(cv2.contourArea(poly_i))) >= 4.0:
        cv2.fillConvexPoly(mask, poly_i, 255)
    return mask


def build_perspective_geometry_v2221(
    frame_shape: Sequence[int],
    inverse_homography: np.ndarray | Sequence[Sequence[float]],
    screen_rect_xywh: Sequence[float],
    *,
    guard_screen_px: float = 12.0,
    crop_padding_camera_px: int = 16,
) -> AnalysisGeometryV2221:
    """Build a perspective-aware crop and inner edge guard.

    ``inverse_homography`` maps absolute screen/projector coordinates to camera
    coordinates.  The safe guard is applied in screen pixels and transformed
    afterwards, so all four physical edges are respected under perspective.
    """
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    frame_height = int(frame_shape[0])
    frame_width = int(frame_shape[1])
    if frame_height <= 0 or frame_width <= 0:
        raise ValueError("invalid frame dimensions")

    H_inv = np.asarray(inverse_homography, dtype=np.float32)
    if H_inv.shape != (3, 3) or not np.all(np.isfinite(H_inv)):
        raise ValueError("inverse_homography must be a finite 3x3 matrix")

    rect = tuple(float(v) for v in screen_rect_xywh[:4])
    if rect[2] <= 1.0 or rect[3] <= 1.0:
        raise ValueError("screen_rect_xywh must have positive width/height")

    outer_screen = _screen_rect_points(rect).reshape(-1, 1, 2)
    safe_rect = _shrink_rect(rect, float(guard_screen_px))
    safe_screen = _screen_rect_points(safe_rect).reshape(-1, 1, 2)

    outer_cam = cv2.perspectiveTransform(outer_screen, H_inv).reshape(-1, 2)
    safe_cam = cv2.perspectiveTransform(safe_screen, H_inv).reshape(-1, 2)
    outer_cam = _clip_polygon(outer_cam, frame_width, frame_height)
    safe_cam = _clip_polygon(safe_cam, frame_width, frame_height)

    outer_i = np.round(outer_cam).astype(np.int32)
    safe_i = np.round(safe_cam).astype(np.int32)
    if abs(float(cv2.contourArea(outer_i))) < 10.0:
        raise ValueError("projected playfield polygon is too small")
    if abs(float(cv2.contourArea(safe_i))) < 4.0:
        raise ValueError("safe playfield polygon is too small")

    x0, y0, x1, y1 = _bbox_from_polygon(
        outer_cam,
        frame_width=frame_width,
        frame_height=frame_height,
        padding=int(crop_padding_camera_px),
    )
    mask = _local_mask(
        safe_cam,
        crop_x0=x0,
        crop_y0=y0,
        crop_width=x1 - x0,
        crop_height=y1 - y0,
    )
    if not np.any(mask):
        raise ValueError("safe playfield mask is empty")

    return AnalysisGeometryV2221(
        frame_height=frame_height,
        frame_width=frame_width,
        crop_x0=x0,
        crop_y0=y0,
        crop_x1=x1,
        crop_y1=y1,
        outer_camera_polygon=outer_cam,
        safe_camera_polygon=safe_cam,
        safe_mask_local=mask,
        mode="homography",
        guard_screen_px=float(max(0.0, guard_screen_px)),
        crop_padding_camera_px=max(0, int(crop_padding_camera_px)),
    )


def build_rect_fallback_geometry_v2221(
    frame_shape: Sequence[int],
    camera_rect_xywh: Sequence[float],
    *,
    guard_camera_px: int = 8,
    crop_padding_camera_px: int = 8,
) -> AnalysisGeometryV2221:
    """Fallback when no valid homography exists.

    The canonical coordinates remain camera coordinates, but the guard cannot be
    perspective-aware without calibration.  This path is deliberately marked
    ``scanport_fallback`` in diagnostics so live testing can confirm that the
    preferred homography path is active.
    """
    frame_height = int(frame_shape[0])
    frame_width = int(frame_shape[1])
    x, y, w, h = (float(v) for v in camera_rect_xywh[:4])
    x = max(0.0, min(float(frame_width - 1), x))
    y = max(0.0, min(float(frame_height - 1), y))
    w = max(1.0, min(float(frame_width) - x, w))
    h = max(1.0, min(float(frame_height) - y, h))
    outer = _screen_rect_points((x, y, w, h))
    safe_rect = _shrink_rect((x, y, w, h), float(max(0, int(guard_camera_px))))
    safe = _screen_rect_points(safe_rect)
    x0, y0, x1, y1 = _bbox_from_polygon(
        outer,
        frame_width=frame_width,
        frame_height=frame_height,
        padding=int(crop_padding_camera_px),
    )
    mask = _local_mask(
        safe,
        crop_x0=x0,
        crop_y0=y0,
        crop_width=x1 - x0,
        crop_height=y1 - y0,
    )
    return AnalysisGeometryV2221(
        frame_height=frame_height,
        frame_width=frame_width,
        crop_x0=x0,
        crop_y0=y0,
        crop_x1=x1,
        crop_y1=y1,
        outer_camera_polygon=outer,
        safe_camera_polygon=safe,
        safe_mask_local=mask,
        mode="scanport_fallback",
        guard_screen_px=float(max(0, int(guard_camera_px))),
        crop_padding_camera_px=max(0, int(crop_padding_camera_px)),
    )


def build_full_frame_geometry_v2221(frame_shape: Sequence[int]) -> AnalysisGeometryV2221:
    frame_height = int(frame_shape[0])
    frame_width = int(frame_shape[1])
    outer = _screen_rect_points((0.0, 0.0, float(frame_width - 1), float(frame_height - 1)))
    mask = np.full((frame_height, frame_width), 255, dtype=np.uint8)
    return AnalysisGeometryV2221(
        frame_height=frame_height,
        frame_width=frame_width,
        crop_x0=0,
        crop_y0=0,
        crop_x1=frame_width,
        crop_y1=frame_height,
        outer_camera_polygon=outer,
        safe_camera_polygon=outer.copy(),
        safe_mask_local=mask,
        mode="full_frame_fallback",
        guard_screen_px=0.0,
        crop_padding_camera_px=0,
    )
