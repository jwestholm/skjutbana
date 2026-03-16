from __future__ import annotations

import time
from dataclasses import dataclass
from collections import deque
from typing import Callable

import cv2
import numpy as np

from src.engine.settings import load_camera_calibration, load_viewport_rect


@dataclass
class HitEvent:
    source: str
    screen_x: float
    screen_y: float
    game_x: float
    game_y: float
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
        viewport = load_viewport_rect()
        game_x = screen_x - viewport.x
        game_y = screen_y - viewport.y

        camera = None
        if self.inverse is not None:
            camera = self._transform(self.inverse, screen_x, screen_y)
        if camera is None:
            camera = (screen_x, screen_y)

        event = HitEvent(
            source="mouse",
            screen_x=float(screen_x),
            screen_y=float(screen_y),
            game_x=float(game_x),
            game_y=float(game_y),
            camera_x=float(camera[0]),
            camera_y=float(camera[1]),
            timestamp=time.time(),
        )
        self._notify(event)

    def push_camera_hit(self, camera_x, camera_y):
        viewport = load_viewport_rect()

        screen = None
        if self.homography is not None:
            screen = self._transform(self.homography, camera_x, camera_y)
        if screen is None:
            screen = (camera_x, camera_y)

        event = HitEvent(
            source="camera",
            screen_x=float(screen[0]),
            screen_y=float(screen[1]),
            game_x=float(screen[0] - viewport.x),
            game_y=float(screen[1] - viewport.y),
            camera_x=float(camera_x),
            camera_y=float(camera_y),
            timestamp=time.time(),
        )
        self._notify(event)

    def poll(self):
        if not self.queue:
            return None
        return self.queue.popleft()


hit_input = HitInput()