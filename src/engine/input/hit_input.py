from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from src.engine.settings import (
    load_camera_calibration,
    load_content_rect,
    load_scanport_rect,
    load_viewport_rect,
)


@dataclass
class HitEvent:
    source: str

    # App/screen space
    screen_x: float
    screen_y: float

    # Viewport-local
    viewport_x: float
    viewport_y: float

    # Content-local (bild/video/tavla inne i viewport/app)
    content_x: float
    content_y: float
    content_norm_x: float
    content_norm_y: float

    # Kamera/full-frame
    camera_x: float
    camera_y: float

    timestamp: float


class HitInput:
    def __init__(self):
        self.queue: deque[HitEvent] = deque()
        self.subscribers: list[Callable[[HitEvent], None]] = []

        self.homography = None
        self.inverse = None

        self.last_mouse_hit: HitEvent | None = None
        self.last_camera_hit: HitEvent | None = None
        self.last_hit: HitEvent | None = None

        self._load_calibration()

    def _load_calibration(self):
        data = load_camera_calibration()
        if data and data.get("homography"):
            H = np.array(data["homography"], dtype=np.float32)
            self.homography = H
            try:
                self.inverse = np.linalg.inv(H).astype(np.float32)
            except Exception:
                self.inverse = None
        else:
            self.homography = None
            self.inverse = None

    def reload_calibration(self):
        self.homography = None
        self.inverse = None
        self._load_calibration()

    def subscribe(self, callback):
        if callback not in self.subscribers:
            self.subscribers.append(callback)

    def unsubscribe(self, callback):
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def _transform(self, matrix, x, y):
        p = np.array([[[x, y]]], dtype=np.float32)
        try:
            r = cv2.perspectiveTransform(p, matrix)
            return float(r[0, 0, 0]), float(r[0, 0, 1])
        except Exception:
            return None

    def _screen_to_spaces(self, screen_x: float, screen_y: float):
        viewport = load_viewport_rect()
        content_rect = load_content_rect()

        viewport_x = float(screen_x - viewport.x)
        viewport_y = float(screen_y - viewport.y)

        content_x = float(screen_x - content_rect.x)
        content_y = float(screen_y - content_rect.y)

        if content_rect.w > 0:
            content_norm_x = content_x / float(content_rect.w)
        else:
            content_norm_x = 0.0

        if content_rect.h > 0:
            content_norm_y = content_y / float(content_rect.h)
        else:
            content_norm_y = 0.0

        return (
            viewport_x,
            viewport_y,
            content_x,
            content_y,
            content_norm_x,
            content_norm_y,
        )

    def _camera_to_screen_via_scanport(self, camera_x: float, camera_y: float):
        """
        Primär mapping:
        full kamera -> scanport-lokal -> normalized -> viewport.
        """
        scanport = load_scanport_rect()
        viewport = load_viewport_rect()

        if scanport is None or scanport.w <= 0 or scanport.h <= 0:
            return None

        local_x = float(camera_x - scanport.x)
        local_y = float(camera_y - scanport.y)

        norm_x = local_x / float(scanport.w)
        norm_y = local_y / float(scanport.h)

        screen_x = float(viewport.x + norm_x * viewport.w)
        screen_y = float(viewport.y + norm_y * viewport.h)
        return screen_x, screen_y

    def _notify(self, event: HitEvent):
        self.last_hit = event

        if event.source == "mouse":
            self.last_mouse_hit = event
        elif event.source == "camera":
            self.last_camera_hit = event

        self.queue.append(event)

        for cb in list(self.subscribers):
            try:
                cb(event)
            except Exception:
                pass

    def push_mouse_hit(self, screen_x, screen_y):
        screen_x = float(screen_x)
        screen_y = float(screen_y)

        camera = None
        if self.inverse is not None:
            camera = self._transform(self.inverse, screen_x, screen_y)

        if camera is None:
            camera = (screen_x, screen_y)

        (
            viewport_x,
            viewport_y,
            content_x,
            content_y,
            content_norm_x,
            content_norm_y,
        ) = self._screen_to_spaces(screen_x, screen_y)

        event = HitEvent(
            source="mouse",
            screen_x=screen_x,
            screen_y=screen_y,
            viewport_x=viewport_x,
            viewport_y=viewport_y,
            content_x=content_x,
            content_y=content_y,
            content_norm_x=content_norm_x,
            content_norm_y=content_norm_y,
            camera_x=float(camera[0]),
            camera_y=float(camera[1]),
            timestamp=time.time(),
        )
        self._notify(event)

    def push_camera_hit(self, camera_x, camera_y):
        camera_x = float(camera_x)
        camera_y = float(camera_y)

        # 1) Primär väg: scanport -> viewport
        screen = self._camera_to_screen_via_scanport(camera_x, camera_y)

        # 2) Fallback: homography
        if screen is None and self.homography is not None:
            screen = self._transform(self.homography, camera_x, camera_y)

        # 3) Sista fallback
        if screen is None:
            screen = (camera_x, camera_y)

        screen_x = float(screen[0])
        screen_y = float(screen[1])

        (
            viewport_x,
            viewport_y,
            content_x,
            content_y,
            content_norm_x,
            content_norm_y,
        ) = self._screen_to_spaces(screen_x, screen_y)

        event = HitEvent(
            source="camera",
            screen_x=screen_x,
            screen_y=screen_y,
            viewport_x=viewport_x,
            viewport_y=viewport_y,
            content_x=content_x,
            content_y=content_y,
            content_norm_x=content_norm_x,
            content_norm_y=content_norm_y,
            camera_x=camera_x,
            camera_y=camera_y,
            timestamp=time.time(),
        )
        self._notify(event)

    def poll(self):
        if not self.queue:
            return None
        return self.queue.popleft()


hit_input = HitInput()