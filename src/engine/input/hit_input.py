from __future__ import annotations

import math
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

    # Canonical game/screen output after the normal camera pipeline.
    screen_x: float
    screen_y: float

    # Original requested screen point.
    # For source="mouse" this is where the user clicked.
    # For source="camera" this is the same as screen_x/screen_y.
    requested_screen_x: float
    requested_screen_y: float

    # Viewport-local / game-local.
    viewport_x: float
    viewport_y: float
    game_x: float
    game_y: float

    # Content-local.
    content_x: float
    content_y: float
    content_norm_x: float
    content_norm_y: float

    # Camera/full-frame.
    camera_x: float
    camera_y: float

    # Scanport-local.
    scanport_x: float
    scanport_y: float
    scanport_norm_x: float
    scanport_norm_y: float

    # Debug reprojections back into screen space.
    scanport_screen_x: float
    scanport_screen_y: float
    homography_screen_x: float
    homography_screen_y: float

    is_simulated: bool
    timestamp: float


class HitInput:
    def __init__(self):
        self.queue: deque[HitEvent] = deque()
        self.subscribers: list[Callable[[HitEvent], None]] = []
        self.homography: np.ndarray | None = None
        self.inverse: np.ndarray | None = None
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
        if matrix is None:
            return None
        p = np.array([[[float(x), float(y)]]], dtype=np.float32)
        try:
            r = cv2.perspectiveTransform(p, matrix)
            return float(r[0, 0, 0]), float(r[0, 0, 1])
        except Exception:
            return None

    def _camera_to_scanport(self, camera_x: float, camera_y: float):
        scanport = load_scanport_rect()
        if scanport is None or scanport.w <= 0 or scanport.h <= 0:
            return None

        local_x = float(camera_x - scanport.x)
        local_y = float(camera_y - scanport.y)
        norm_x = local_x / float(scanport.w)
        norm_y = local_y / float(scanport.h)
        return local_x, local_y, norm_x, norm_y

    def _screen_to_viewport_norm(self, screen_x: float, screen_y: float):
        viewport = load_viewport_rect()
        if viewport.w <= 0 or viewport.h <= 0:
            return None

        local_x = float(screen_x - viewport.x)
        local_y = float(screen_y - viewport.y)
        norm_x = local_x / float(viewport.w)
        norm_y = local_y / float(viewport.h)
        return local_x, local_y, norm_x, norm_y

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
        Primär mapping i nuvarande system:
        kamera -> scanport-lokal -> normalized -> viewport/screen.
        """
        scanport_info = self._camera_to_scanport(camera_x, camera_y)
        if scanport_info is None:
            return None

        _, _, norm_x, norm_y = scanport_info
        viewport = load_viewport_rect()
        screen_x = float(viewport.x + norm_x * viewport.w)
        screen_y = float(viewport.y + norm_y * viewport.h)
        return screen_x, screen_y

    def _camera_to_screen_via_homography(self, camera_x: float, camera_y: float):
        return self._transform(self.homography, camera_x, camera_y)

    def _screen_to_camera_via_scanport(self, screen_x: float, screen_y: float):
        """
        Invers till den primära scanport->viewport-mappningen.
        Används för att låta musklick bli ett syntetiskt kamerahit.
        """
        viewport_info = self._screen_to_viewport_norm(screen_x, screen_y)
        scanport = load_scanport_rect()
        if viewport_info is None or scanport is None or scanport.w <= 0 or scanport.h <= 0:
            return None

        _, _, norm_x, norm_y = viewport_info
        camera_x = float(scanport.x + norm_x * scanport.w)
        camera_y = float(scanport.y + norm_y * scanport.h)
        return camera_x, camera_y

    def _screen_to_camera_via_homography(self, screen_x: float, screen_y: float):
        return self._transform(self.inverse, screen_x, screen_y)

    def _canonical_camera_to_screen(self, camera_x: float, camera_y: float):
        screen = self._camera_to_screen_via_scanport(camera_x, camera_y)
        if screen is not None:
            return screen

        screen = self._camera_to_screen_via_homography(camera_x, camera_y)
        if screen is not None:
            return screen

        return float(camera_x), float(camera_y)

    def _canonical_screen_to_camera(self, screen_x: float, screen_y: float):
        camera = self._screen_to_camera_via_scanport(screen_x, screen_y)
        if camera is not None:
            return camera

        camera = self._screen_to_camera_via_homography(screen_x, screen_y)
        if camera is not None:
            return camera

        return float(screen_x), float(screen_y)

    def _build_event_from_camera(
        self,
        *,
        source: str,
        camera_x: float,
        camera_y: float,
        requested_screen_x: float | None = None,
        requested_screen_y: float | None = None,
        is_simulated: bool = False,
    ) -> HitEvent:
        canonical_screen = self._canonical_camera_to_screen(camera_x, camera_y)
        scanport_screen = self._camera_to_screen_via_scanport(camera_x, camera_y)
        homography_screen = self._camera_to_screen_via_homography(camera_x, camera_y)
        scanport_info = self._camera_to_scanport(camera_x, camera_y)

        screen_x = float(canonical_screen[0])
        screen_y = float(canonical_screen[1])

        if requested_screen_x is None:
            requested_screen_x = screen_x
        if requested_screen_y is None:
            requested_screen_y = screen_y

        (
            viewport_x,
            viewport_y,
            content_x,
            content_y,
            content_norm_x,
            content_norm_y,
        ) = self._screen_to_spaces(screen_x, screen_y)

        if scanport_info is not None:
            scanport_x, scanport_y, scanport_norm_x, scanport_norm_y = scanport_info
        else:
            scanport_x = float(camera_x)
            scanport_y = float(camera_y)
            scanport_norm_x = 0.0
            scanport_norm_y = 0.0

        if scanport_screen is None:
            scanport_screen_x = screen_x
            scanport_screen_y = screen_y
        else:
            scanport_screen_x = float(scanport_screen[0])
            scanport_screen_y = float(scanport_screen[1])

        if homography_screen is None:
            homography_screen_x = screen_x
            homography_screen_y = screen_y
        else:
            homography_screen_x = float(homography_screen[0])
            homography_screen_y = float(homography_screen[1])

        return HitEvent(
            source=source,
            screen_x=screen_x,
            screen_y=screen_y,
            requested_screen_x=float(requested_screen_x),
            requested_screen_y=float(requested_screen_y),
            viewport_x=viewport_x,
            viewport_y=viewport_y,
            game_x=viewport_x,
            game_y=viewport_y,
            content_x=content_x,
            content_y=content_y,
            content_norm_x=content_norm_x,
            content_norm_y=content_norm_y,
            camera_x=float(camera_x),
            camera_y=float(camera_y),
            scanport_x=float(scanport_x),
            scanport_y=float(scanport_y),
            scanport_norm_x=float(scanport_norm_x),
            scanport_norm_y=float(scanport_norm_y),
            scanport_screen_x=scanport_screen_x,
            scanport_screen_y=scanport_screen_y,
            homography_screen_x=homography_screen_x,
            homography_screen_y=homography_screen_y,
            is_simulated=bool(is_simulated),
            timestamp=time.time(),
        )

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
        """
        Behandla musklick som ett syntetiskt kamerahit.

        Klicket är alltså inte en separat spelgenväg, utan går:
        screen -> camera (bakåt) -> screen/game (framåt igen)
        med samma normala kamera-pipeline som riktiga träffar.
        """
        screen_x = float(screen_x)
        screen_y = float(screen_y)

        viewport = load_viewport_rect()
        if viewport is not None and not viewport.collidepoint(int(round(screen_x)), int(round(screen_y))):
            return None

        camera_x, camera_y = self._canonical_screen_to_camera(screen_x, screen_y)
        event = self._build_event_from_camera(
            source="mouse",
            camera_x=camera_x,
            camera_y=camera_y,
            requested_screen_x=screen_x,
            requested_screen_y=screen_y,
            is_simulated=True,
        )
        self._notify(event)
        return event

    def push_camera_hit(self, camera_x, camera_y):
        camera_x = float(camera_x)
        camera_y = float(camera_y)
        event = self._build_event_from_camera(
            source="camera",
            camera_x=camera_x,
            camera_y=camera_y,
            is_simulated=False,
        )
        self._notify(event)
        return event

    def poll(self):
        if not self.queue:
            return None
        return self.queue.popleft()


hit_input = HitInput()
