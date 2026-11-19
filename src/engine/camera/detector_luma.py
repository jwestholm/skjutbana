"""Single conversion point from camera BGR to the detector's grey image (2026-11-19).

Default: cv2.COLOR_BGR2GRAY (unchanged behaviour).
H13 SKJUTBANA_DETECTOR_LUMA=L: additionally map the grey value through a CIE L* lookup table
(grey treated as sRGB-encoded neutral grey, OpenCV 8-bit Lab scale). Offline at GT positions the
L* tone curve raised the hole matched-filter SNR of REAL AR15 holes by ~34 % (226 vs 169; true Lab L
221) at ~1 ms per frame, by compressing bright-paper noise more than the darker hole signal;
projected holes were roughly unchanged (AUDIT_FINDINGS §27). Every live grey conversion of camera
frames must go through detector_gray() so PRE and POST always use the same transform.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

_v = np.arange(256) / 255.0
_lin = np.where(_v <= 0.04045, _v / 12.92, ((_v + 0.055) / 1.055) ** 2.4)
_f = np.where(_lin > 0.008856, np.cbrt(_lin), 7.787 * _lin + 16.0 / 116.0)
LUT_L = np.clip(np.round((116.0 * _f - 16.0) * 255.0 / 100.0), 0, 255).astype(np.uint8)

_MODE = os.environ.get("SKJUTBANA_DETECTOR_LUMA", "").strip().upper()


def mode() -> str:
    return "L" if _MODE == "L" else "GRAY"


def detector_gray(bgr: np.ndarray) -> np.ndarray:
    """BGR (or already grey) camera image -> uint8 detector grey."""
    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if _MODE == "L":
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return cv2.LUT(gray, LUT_L)
    return gray
