from __future__ import annotations

import cv2
import numpy as np
import pygame


class VideoPlayer:
    def __init__(self, path: str, target_size: tuple[int, int] | None) -> None:
        self.path = path
        self.target_size = target_size

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Kunde inte öppna video: {path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 1e-6 else 30.0
        self.frame_time = 1.0 / self.fps

        # källa (för ratio)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.source_size = (w, h) if w > 0 and h > 0 else None

        self.playing = True
        self.finished = False
        self._acc = 0.0

    def toggle_pause(self) -> None:
        self.playing = not self.playing

    def pause(self) -> None:
        self.playing = False

    def close(self) -> None:
        if self.cap:
            self.cap.release()

    def update(self, dt: float) -> pygame.Surface | None:
        if self.finished:
            return None
        if not self.playing:
            return None

        self._acc += dt

        max_steps = 5
        steps = 0
        newest_surface = None

        while self._acc >= self.frame_time and steps < max_steps:
            self._acc -= self.frame_time
            steps += 1

            ok, frame_bgr = self.cap.read()
            if not ok:
                self.finished = True
                break

            newest_surface = self._frame_to_surface(frame_bgr)

        return newest_surface

    def _frame_to_surface(self, frame_bgr: np.ndarray) -> pygame.Surface:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))  # (w,h,3)

        surf = pygame.surfarray.make_surface(frame_rgb)

        if self.target_size is not None and surf.get_size() != self.target_size:
            surf = pygame.transform.smoothscale(surf, self.target_size)

        return surf.convert()