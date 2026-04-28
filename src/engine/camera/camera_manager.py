from __future__ import annotations

import threading
import time
from collections import deque
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
    """
    Camera manager with a dedicated reader thread.

    The reader thread calls cap.read() in a tight loop, timestamping each
    frame at read time. Frames go into a thread-safe ring buffer.
    Main thread picks up frames via update() / get_latest_frame().

    This solves the camera buffering problem: timestamps reflect when
    the frame was actually read from the driver, and the ring buffer
    gives hit_scanner a dense frame history for accurate pre-shot selection.
    """

    # Ring buffer size: ~10 seconds at 30fps
    RING_BUFFER_SIZE = 300

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

        # Reader thread state
        self._reader_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ring: deque[CameraFrame] = deque(maxlen=self.RING_BUFFER_SIZE)
        self._read_count: int = 0
        self._last_pickup_count: int = 0
        self._fresh_request = threading.Event()
        self._fresh_frame: CameraFrame | None = None
        self._fresh_ready = threading.Event()

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
        parts.append(f"mirror_h={'on' if self.mirror_horizontal else 'off'}")
        parts.append(f"mirror_v={'on' if self.mirror_vertical else 'off'}")
        return ", ".join(parts)

    def start(self) -> bool:
        if self._reader_thread is not None and self._reader_thread.is_alive():
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

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.property_apply_result = apply_preferred_camera_settings(
            self.cap,
            preferred_width=self.preferred_width,
            preferred_height=self.preferred_height,
            preferred_fps=self.preferred_fps,
        )
        self.capabilities = probe_camera_capabilities(self.cap)

        # Start reader thread
        with self._lock:
            self._ring.clear()
            self._read_count = 0
            self._last_pickup_count = 0

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        return True

    def stop(self) -> None:
        self.running = False

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.latest_frame = None

    def _reader_loop(self) -> None:
        """Dedicated thread: reads frames as fast as the camera delivers them."""
        while self.running and self.cap is not None and self.cap.isOpened():
            # Check for fresh-frame request (main thread wants a guaranteed-current frame)
            if self._fresh_request.is_set():
                self._fresh_request.clear()
                # Flush buffer: grab a few times, then read
                try:
                    for _ in range(3):
                        self.cap.grab()
                    ok, frame_bgr = self.cap.retrieve()
                    if ok and frame_bgr is not None:
                        ts = time.time()
                        frame_bgr = self._apply_frame_transform(frame_bgr)
                        self._fresh_frame = CameraFrame(frame_bgr=frame_bgr, timestamp=ts)
                    else:
                        self._fresh_frame = None
                except Exception:
                    self._fresh_frame = None
                self._fresh_ready.set()
                continue

            try:
                ok, frame_bgr = self.cap.read()
            except Exception:
                break

            if not ok or frame_bgr is None:
                time.sleep(0.001)
                continue

            ts = time.time()
            frame_bgr = self._apply_frame_transform(frame_bgr)
            cf = CameraFrame(frame_bgr=frame_bgr, timestamp=ts)

            with self._lock:
                self._ring.append(cf)
                self._read_count += 1

    def update(self) -> None:
        """Main thread: pick up the latest frame from the ring buffer."""
        if not self.running:
            return

        with self._lock:
            if self._ring:
                self.latest_frame = self._ring[-1]
                self._last_pickup_count = self._read_count
            self.last_error = None

        if self.cap is not None:
            self.capabilities = probe_camera_capabilities(self.cap)

    def get_new_frames_since_last_pickup(self) -> list[CameraFrame]:
        """Return all frames added to the ring since the last call to this method.

        Used by hit_scanner to build a dense frame_history with every frame
        the camera produced, not just the latest per update cycle.
        """
        with self._lock:
            current_count = self._read_count
            ring_list = list(self._ring)

        # How many new frames since last pickup?
        new_count = current_count - self._last_pickup_count
        self._last_pickup_count = current_count

        if new_count <= 0:
            return []
        if new_count >= len(ring_list):
            return ring_list
        return ring_list[-new_count:]

    def get_latest_frame(self) -> np.ndarray | None:
        if self.latest_frame is None:
            return None
        return self.latest_frame.frame_bgr.copy()

    def capture_fresh_frame(self) -> np.ndarray | None:
        """Request a guaranteed-current frame from the camera.

        Signals the reader thread to flush the camera buffer and capture
        a fresh frame. Blocks until the frame is ready (max 500ms).
        Use for post-shot capture where you need what the camera sees RIGHT NOW.
        """
        if not self.running:
            return None
        self._fresh_frame = None
        self._fresh_ready.clear()
        self._fresh_request.set()

        # Wait for reader thread to deliver
        if self._fresh_ready.wait(timeout=0.5):
            if self._fresh_frame is not None:
                return self._fresh_frame.frame_bgr.copy()
        return None

    def get_latest_timestamp(self) -> float | None:
        if self.latest_frame is None:
            return None
        return self.latest_frame.timestamp

    def get_ring_snapshot(self) -> list[CameraFrame]:
        """Return a snapshot of the entire ring buffer (for pre-shot search)."""
        with self._lock:
            return list(self._ring)

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

        with self._lock:
            ring_len = len(self._ring)
            total_read = self._read_count
        lines.append(f"Ring buffer: {ring_len}/{self.RING_BUFFER_SIZE} | Total frames: {total_read}")

        if self.last_error:
            lines.append(f"Fel: {self.last_error}")

        return lines


camera_manager = CameraManager()
