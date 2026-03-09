from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.engine.camera.camera_capabilities import (
    CameraCapabilities,
    apply_preferred_camera_settings,
    probe_camera_capabilities,
)


@dataclass
class CameraFrame:
    frame_bgr: np.ndarray
    timestamp: float


class CameraManager:
    def __init__(
        self,
        camera_index: int = 0,
        preferred_width: int = 1920,
        preferred_height: int = 1080,
        preferred_fps: int = 30,
    ) -> None:
        self.camera_index = camera_index
        self.preferred_width = preferred_width
        self.preferred_height = preferred_height
        self.preferred_fps = preferred_fps

        self.cap: cv2.VideoCapture | None = None
        self.latest_frame: CameraFrame | None = None
        self.last_error: str | None = None
        self.capabilities: CameraCapabilities | None = None
        self.property_apply_result: dict[str, bool] = {}
        self.running = False

    def start(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            self.running = True
            return True

        self.last_error = None

        cap = cv2.VideoCapture(self.camera_index)
        if not cap or not cap.isOpened():
            self.cap = None
            self.running = False
            self.last_error = f"Kunde inte öppna kamera index {self.camera_index}"
            return False

        self.cap = cap
        self.running = True

        self.property_apply_result = apply_preferred_camera_settings(
            self.cap,
            preferred_width=self.preferred_width,
            preferred_height=self.preferred_height,
            preferred_fps=self.preferred_fps,
        )
        self.capabilities = probe_camera_capabilities(self.cap)
        return True

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.latest_frame = None

    def update(self) -> None:
        if not self.running:
            return

        if self.cap is None or not self.cap.isOpened():
            if not self.start():
                return

        assert self.cap is not None

        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            self.last_error = "Kunde inte läsa frame från kameran."
            return

        self.latest_frame = CameraFrame(frame_bgr=frame_bgr, timestamp=time.time())
        self.last_error = None

    def get_latest_frame(self) -> np.ndarray | None:
        if self.latest_frame is None:
            return None
        return self.latest_frame.frame_bgr.copy()

    def get_latest_timestamp(self) -> float | None:
        if self.latest_frame is None:
            return None
        return self.latest_frame.timestamp

    def get_status_lines(self) -> list[str]:
        lines: list[str] = []

        if self.capabilities is not None:
            lines.extend(self.capabilities.summary_lines())

        if self.property_apply_result:
            applied = ", ".join(
                f"{k}={'ok' if v else 'no'}"
                for k, v in self.property_apply_result.items()
            )
            lines.append(f"Init props: {applied}")

        if self.last_error:
            lines.append(f"Fel: {self.last_error}")

        if self.latest_frame is not None:
            lines.append(f"Senaste frame-ts: {self.latest_frame.timestamp:.3f}")

        return lines


camera_manager = CameraManager()