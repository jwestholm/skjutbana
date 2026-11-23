"""Selftest for src/engine/display_floor.py (default OFF black floor).
Run: PYTHONPATH=. python3 automation/display_floor_selftest.py
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from src.engine import display_floor  # noqa: E402


def _clear():
    for k in list(os.environ):
        if k.startswith("SKJUTBANA_DISPLAY_BLACK_FLOOR"):
            del os.environ[k]


def _surf():
    s = pygame.Surface((64, 48))
    s.fill((0, 0, 0))
    s.fill((200, 10, 5), pygame.Rect(0, 0, 8, 8))
    return s


def test_default_noop() -> None:
    _clear()
    s = _surf()
    display_floor.apply(s)
    assert s.get_at((30, 30))[:3] == (0, 0, 0) and s.get_at((1, 1))[:3] == (200, 10, 5)


def test_grey_floor_is_max() -> None:
    _clear(); os.environ["SKJUTBANA_DISPLAY_BLACK_FLOOR"] = "24"
    s = _surf(); display_floor.apply(s)
    assert s.get_at((30, 30))[:3] == (24, 24, 24)
    assert s.get_at((1, 1))[:3] == (200, 24, 24)   # per-channel max, bright content kept
    _clear()


def test_rgb_and_exempt() -> None:
    _clear(); os.environ["SKJUTBANA_DISPLAY_BLACK_FLOOR_RGB"] = "12,16,28"
    s = _surf(); display_floor.apply(s)
    assert s.get_at((30, 30))[:3] == (12, 16, 28)
    s = _surf(); display_floor.apply(s, SimpleNamespace(_auto_cal_phase="grid"))
    assert s.get_at((30, 30))[:3] == (0, 0, 0), "calibration must be exempt"
    s = _surf(); display_floor.apply(s, SimpleNamespace(display_floor_exempt=True))
    assert s.get_at((30, 30))[:3] == (0, 0, 0)
    _clear()


def test_texture_static() -> None:
    _clear(); os.environ["SKJUTBANA_DISPLAY_BLACK_FLOOR"] = "40"; os.environ["SKJUTBANA_DISPLAY_BLACK_FLOOR_TEXTURE"] = "8"
    a = _surf(); display_floor.apply(a); b = _surf(); display_floor.apply(b)
    pa, pb = pygame.surfarray.array3d(a), pygame.surfarray.array3d(b)
    assert (pa == pb).all(), "texture must not change between frames"
    assert pa[10:, 10:].min() >= 32 and pa[10:, 10:].max() <= 48
    _clear()


if __name__ == "__main__":
    pygame.init()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
