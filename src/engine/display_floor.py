"""Display black floor (opt-in, default OFF; 2026-11-20).

Pure black (#000000) leaves the paper unlit, so the camera cannot see a hole there (live AI test:
0 correct on 'black' in every arm; visibility probe: holes on black/dark content invisible).  A
minimum projected luminance lights the paper so a hole shows as a dark spot.  Applied as the very
last step before every display flip, for every game, without touching game logic.

SKJUTBANA_DISPLAY_BLACK_FLOOR=<0-255>          grey floor, e.g. 24 (per channel max(pixel, floor))
SKJUTBANA_DISPLAY_BLACK_FLOOR_RGB=r,g,b         coloured floor instead (e.g. 18,20,28 - dark blue-grey)
SKJUTBANA_DISPLAY_BLACK_FLOOR_TEXTURE=<amp>     static low-contrast texture of +-amp grey on the floor
                                               (fixed seed: never changes between frames, so it adds
                                               no temporal change for the detector)
Scenes are exempt while they run camera calibration (attribute _auto_cal_phase not None) or when
they set display_floor_exempt = True.  Test it with the projector ON; results must be reported
separately from projector-off runs.
"""
from __future__ import annotations

import os

_CACHE: dict = {}


def _config():
    rgb = os.environ.get("SKJUTBANA_DISPLAY_BLACK_FLOOR_RGB", "").strip()
    grey = os.environ.get("SKJUTBANA_DISPLAY_BLACK_FLOOR", "").strip()
    try:
        if rgb:
            col = tuple(max(0, min(255, int(v))) for v in rgb.split(","))[:3]
        elif grey:
            g = max(0, min(255, int(grey)))
            col = (g, g, g)
        else:
            return None
        amp = max(0, min(64, int(os.environ.get("SKJUTBANA_DISPLAY_BLACK_FLOOR_TEXTURE", "0") or 0)))
    except ValueError:
        return None
    if len(col) != 3 or max(col) <= 0:
        return None
    return col, amp


def apply(surface, scene=None) -> None:
    """Raise every pixel to at least the configured floor (BLEND_RGB_MAX). No-op when unset."""
    cfg = _config()
    if cfg is None or surface is None:
        return
    if scene is not None and (getattr(scene, "display_floor_exempt", False)
                              or getattr(scene, "_auto_cal_phase", None) is not None):
        return
    import pygame
    col, amp = cfg
    if amp <= 0:
        surface.fill(col, special_flags=pygame.BLEND_RGB_MAX)
        return
    key = (surface.get_size(), col, amp)
    tex = _CACHE.get(key)
    if tex is None:
        import numpy as np
        w, h = surface.get_size()
        rng = np.random.default_rng(20261120)
        # smooth blotchy texture (paper-like), fixed for the whole run
        small = rng.normal(0.0, 1.0, (max(2, h // 24), max(2, w // 24), 1))
        import cv2
        big = cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        big = big[..., None] if big.ndim == 2 else big
        arr = np.clip(np.array(col, np.float32)[None, None, :] + amp * big / max(1e-6, np.abs(big).max()), 0, 255)
        tex = pygame.surfarray.make_surface(arr.astype(np.uint8).transpose(1, 0, 2))
        _CACHE.clear()
        _CACHE[key] = tex
    surface.blit(tex, (0, 0), special_flags=pygame.BLEND_RGB_MAX)
