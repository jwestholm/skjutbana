from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
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
    def __init__(self) -> None:
        self.queue: deque[HitEvent] = deque()
        self.subscribers: list[Callable[[HitEvent], None]] = []

    def subscribe(self, callback: Callable[[HitEvent], None]) -> None:
        if callback not in self.subscribers:
            self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[HitEvent], None]) -> None:
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def _load_homography(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        data = load_camera_calibration()
        if not data or not data.get("homography"):
            return None, None

        try:
            homography = np.array(data["homography"], dtype=np.float32)
            inverse = np.linalg.inv(homography).astype(np.float32)
            return homography, inverse
        except Exception:
            return None, None

    def _transform(self, matrix: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
        point = np.array([[[x, y]]], dtype=np.float32)
        try:
            result = cv2.perspectiveTransform(point, matrix)
            return float(result[0, 0, 0]), float(result[0, 0, 1])
        except Exception:
            return None

    def _notify(self, event: HitEvent) -> None:
        self.queue.append(event)
        for callback in list(self.subscribers):
            try:
                callback(event)
            except Exception:
                pass

    def push_mouse_hit(self, screen_x: float, screen_y: float) -> None:
        homography, inverse = self._load_homography()
        viewport = load_viewport_rect()

        game_x = float(screen_x - viewport.x)
        game_y = float(screen_y - viewport.y)

        camera_point = None
        if inverse is not None:
            camera_point = self._transform(inverse, float(screen_x), float(screen_y))

        if camera_point is None:
            camera_point = (float(screen_x), float(screen_y))

        event = HitEvent(
            source="mouse",
            screen_x=float(screen_x),
            screen_y=float(screen_y),
            game_x=game_x,
            game_y=game_y,
            camera_x=float(camera_point[0]),
            camera_y=float(camera_point[1]),
            timestamp=time.time(),
        )
        self._notify(event)

    def push_camera_hit(self, camera_x: float, camera_y: float) -> None:
        homography, _ = self._load_homography()
        viewport = load_viewport_rect()

        screen_point = None
        if homography is not None:
            screen_point = self._transform(homography, float(camera_x), float(camera_y))

        if screen_point is None:
            screen_point = (float(camera_x), float(camera_y))

        event = HitEvent(
            source="camera",
            screen_x=float(screen_point[0]),
            screen_y=float(screen_point[1]),
            game_x=float(screen_point[0] - viewport.x),
            game_y=float(screen_point[1] - viewport.y),
            camera_x=float(camera_x),
            camera_y=float(camera_y),
            timestamp=time.time(),
        )
        self._notify(event)

    def poll(self) -> HitEvent | None:
        if not self.queue:
            return None
        return self.queue.popleft()


hit_input = HitInput()