"""Camera/projector geometry and physical PRE/POST evidence helpers."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


class TargetEvidenceError(ValueError):
    pass


def homography_from_calibration(calibration: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(calibration, dict):
        return None
    matrix = calibration.get("homography")
    try:
        h = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if h.shape != (3, 3) or not np.isfinite(h).all() or abs(float(np.linalg.det(h))) < 1e-12:
        return None
    return h


def camera_to_target(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    vector = np.asarray([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
    mapped = np.asarray(homography, dtype=np.float64) @ vector
    if mapped.shape != (3,) or not np.isfinite(mapped).all() or abs(float(mapped[2])) < 1e-12:
        raise TargetEvidenceError("Invalid camera-to-target projection")
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def target_to_camera(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    inverse = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    return camera_to_target(point, inverse)


def warp_to_target(image: np.ndarray, homography: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Warp a camera image to (width,height) projector/viewport coordinates."""
    try:
        import cv2
    except ImportError as exc:
        raise TargetEvidenceError("opencv-python is required for target view") from exc
    if image.ndim not in (2, 3) or image.shape[0] < 1 or image.shape[1] < 1:
        raise TargetEvidenceError(f"Unsupported image shape: {image.shape}")
    width, height = map(int, target_size)
    if width < 1 or height < 1:
        raise TargetEvidenceError("Target view dimensions must be positive")
    return cv2.warpPerspective(np.asarray(image), np.asarray(homography, dtype=np.float64), (width, height))


def register_pre_post(pre: np.ndarray, post: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Estimate a translation-only registration using downsampled camera frames."""
    try:
        import cv2
    except ImportError as exc:
        raise TargetEvidenceError("opencv-python is required for registration") from exc
    if pre.shape[:2] != post.shape[:2]:
        raise TargetEvidenceError("PRE and POST dimensions differ")
    def gray_u8(value: np.ndarray) -> np.ndarray:
        if value.ndim == 3:
            value = cv2.cvtColor(value[..., :3], cv2.COLOR_RGB2GRAY)
        return np.clip(value, 0, 255).astype(np.uint8)
    a, b = gray_u8(pre), gray_u8(post)
    factor = max(1, int(max(a.shape) / 640))
    if factor > 1:
        a = cv2.resize(a, (a.shape[1] // factor, a.shape[0] // factor), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (b.shape[1] // factor, b.shape[0] // factor), interpolation=cv2.INTER_AREA)
    shift, _ = cv2.phaseCorrelate(np.float32(a), np.float32(b))
    return np.asarray([float(shift[0] * factor), float(shift[1] * factor)]), (float(shift[0] * factor), float(shift[1] * factor))


def enhanced_difference(pre: np.ndarray, post: np.ndarray, *, register: bool = True) -> np.ndarray:
    """Return a visible uint8 darkening/change image from physical PRE/POST evidence."""
    try:
        import cv2
    except ImportError as exc:
        raise TargetEvidenceError("opencv-python is required for evidence enhancement") from exc
    if pre.shape[:2] != post.shape[:2]:
        raise TargetEvidenceError("PRE and POST dimensions differ")
    aligned = np.asarray(post)
    if register:
        shift, _ = register_pre_post(pre, post)
        matrix = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        aligned = cv2.warpAffine(aligned, matrix, (post.shape[1], post.shape[0]), borderMode=cv2.BORDER_REFLECT)
    p = pre.astype(np.float32)
    q = aligned.astype(np.float32)
    if p.ndim == 3:
        p = cv2.cvtColor(np.clip(p, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    if q.ndim == 3:
        q = cv2.cvtColor(np.clip(q, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    darkening = np.maximum(p - q, 0.0)
    change = np.abs(p - q)
    evidence = np.maximum(darkening, change * 0.65)
    if float(np.max(evidence)) <= 0:
        return np.zeros(evidence.shape, dtype=np.uint8)
    evidence = cv2.GaussianBlur(evidence.astype(np.float32), (0, 0), 1.0)
    lo, hi = np.percentile(evidence, (80.0, 99.8))
    if hi <= lo:
        lo, hi = float(evidence.min()), float(evidence.max())
    return np.clip((evidence - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)


def target_evidence_view(pre: np.ndarray, post: np.ndarray, homography: np.ndarray,
                        target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return warped POST and an RGB evidence overlay, both in target space."""
    warped_post = warp_to_target(post, homography, target_size)
    diff = enhanced_difference(pre, post)
    warped_diff = warp_to_target(diff, homography, target_size)
    if warped_post.ndim == 2:
        base = np.repeat(warped_post[..., None], 3, axis=2)
    else:
        base = warped_post[..., :3].copy()
    base = np.clip(base, 0, 255).astype(np.uint8)
    overlay = base.copy()
    strength = warped_diff.astype(np.float32) / 255.0
    overlay[..., 0] = np.clip(overlay[..., 0].astype(np.float32) + 180.0 * strength, 0, 255)
    overlay[..., 1] = np.clip(overlay[..., 1].astype(np.float32) * (1.0 - 0.55 * strength), 0, 255)
    overlay[..., 2] = np.clip(overlay[..., 2].astype(np.float32) * (1.0 - 0.55 * strength), 0, 255)
    return base, overlay.astype(np.uint8)


def load_calibration(path: Path) -> dict[str, Any] | None:
    import json
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value.get("camera_calibration") if isinstance(value, dict) else None
