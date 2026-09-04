from __future__ import annotations
import pygame

def _load_settings_dict():
    return {}
def _rect_from_value(value):
    return None
def _sanitize_content_rect(rect):
    return rect
def load_viewport_rect():
    return pygame.Rect(100, 50, 800, 600)
def load_content_rect() -> pygame.Rect:
    rect = _rect_from_value(_load_settings_dict().get("content_rect"))
    if rect is not None:
        return _sanitize_content_rect(rect)
    # content_rect is viewport-local throughout the hit/input pipeline.
    # Defaulting to viewport.copy() incorrectly carries absolute viewport x/y
    # and HitScanner then adds viewport.x/y a second time.
    viewport = load_viewport_rect()
    return pygame.Rect(0, 0, viewport.w, viewport.h)
