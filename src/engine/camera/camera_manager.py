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
from src.engine.settings import load_camera_transform_settings


@dataclass
class CameraFrame:
    frame_bgr: np.ndarray
    timestamp: float


class CameraManager:
    def __init__(
        self,
        camera_index: int = 0,
        preferred_width: int = 3840,
        preferred_height: int = 2160,
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

        self.rotation = 0
        self.mirror_horizontal = False
        self.mirror_vertical = False
        self.reload_transform_settings()

    def reload_transform_settings(self) -> None:
        settings = load_camera_transform_settings()
        self.rotation = int(settings.get("rotation", 0))
        self.mirror_horizontal = bool(settings.get("mirror_horizontal", False))
        self.mirror_vertical = bool(settings.get("mirror_vertical", False))

    def _apply_frame_transform(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame = frame_bgr

        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.mirror_horizontal and self.mirror_vertical:
            frame = cv2.flip(frame, -1)
        elif self.mirror_horizontal:
            frame = cv2.flip(frame, 1)
        elif self.mirror_vertical:
            frame = cv2.flip(frame, 0)

        return frame

    def _transform_description(self) -> str:
        parts: list[str] = [f"rotation={self.rotation}°"]
        if self.mirror_horizontal:
            parts.append("mirror_h=on")
        if self.mirror_vertical:
            parts.append("mirror_v=on")
        return ", ".join(parts)

    def start(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            self.running = True
            return True

        self.last_error = None
        self.reload_transform_settings()

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

        frame_bgr = self._apply_frame_transform(frame_bgr)

        self.latest_frame = CameraFrame(frame_bgr=frame_bgr, timestamp=time.time())
        self.last_error = None

        # Uppdatera faktiskt negotiated capture-läge när kameran väl levererar frames.
        self.capabilities = probe_camera_capabilities(self.cap)

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
            lines.append(
                f"Kamera: {self.capabilities.width}x{self.capabilities.height} @ {self.capabilities.fps:.1f} fps"
            )
            lines.append(
                f"Backend: {self.capabilities.backend_name} | FOURCC: {self.capabilities.fourcc}"
            )

        if self.property_apply_result:
            applied = ", ".join(
                f"{k}={'ok' if v else 'no'}"
                for k, v in self.property_apply_result.items()
            )
            lines.append(f"Init props: {applied}")

        lines.append(f"Kameratransform: {self._transform_description()}")

        if self.last_error:
            lines.append(f"Fel: {self.last_error}")

        return lines


camera_manager = CameraManager()