from __future__ import annotations

import threading
import time
import uuid
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
    acquisition: dict | None = None


class CameraManager:
    """
    Camera manager with a dedicated reader thread.

    The reader thread calls cap.read() in a tight loop, timestamping each
    frame at read time. Frames go into a thread-safe ring buffer.
    Main thread picks up frames via update() / get_latest_frame().

    Public timestamps reflect read completion, NOT exposure. A reader thread
    avoids main-loop backlog but cannot prove device/USB buffering is absent.
    Optional acquisition provenance makes those limits observable.
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
        self._frame_sequence = 0
        self._stream_id = uuid.uuid4().hex
        self.buffer_request = {}

    def _timestamped_frame(self, pixels, started, completed, wall_ts, kind='read'):
        self._frame_sequence += 1
        backend = {}
        for name in ('CAP_PROP_POS_MSEC', 'CAP_PROP_POS_FRAMES', 'CAP_PROP_PTS'):
            prop = getattr(cv2, name, None)
            if prop is not None and self.cap is not None:
                try:
                    value = float(self.cap.get(prop))
                    backend[name] = value if np.isfinite(value) else None
                except Exception:
                    backend[name] = None
        metadata = dict(stream_id=self._stream_id, sequence=self._frame_sequence,
            camera_index=self.camera_index,
            backend_name=self.capabilities.backend_name if self.capabilities else None,
            read_start_monotonic=started, read_complete_monotonic=completed,
            receive_wall_ts=wall_ts, capture_kind=kind, backend_timestamps_raw=backend,
            backend_timestamp_semantics='UNVERIFIED_BACKEND_UNITS_NOT_EXPOSURE',
            exposure_monotonic=None, device_buffer_age_s=None,
            rotation=self.rotation, mirror_horizontal=self.mirror_horizontal,
            mirror_vertical=self.mirror_vertical, raw_shape=list(pixels.shape),
            buffer_request=dict(self.buffer_request))
        transformed = self._apply_frame_transform(pixels)
        metadata['frame_shape'] = list(transformed.shape)
        metadata['published_monotonic'] = time.monotonic()
        return CameraFrame(transformed, wall_ts, metadata)

    def provenance_for_timestamp(self, timestamp):
        with self._lock:
            for frame in reversed(self._ring):
                if frame.timestamp == timestamp:
                    return dict(frame.acquisition) if frame.acquisition else None
            if self._fresh_frame is not None and self._fresh_frame.timestamp == timestamp:
                return dict(self._fresh_frame.acquisition) if self._fresh_frame.acquisition else None
        return None

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
            accepted = bool(self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))
            self.buffer_request = dict(requested=1, accepted=accepted,
                reported=float(self.cap.get(cv2.CAP_PROP_BUFFERSIZE)),
                semantics='DRIVER_REPORT_NOT_PROOF_OF_ZERO_BACKLOG')
        except Exception:
            self.buffer_request = dict(requested=1, accepted=False, reported=None)

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
            self._frame_sequence = 0
            self._stream_id = uuid.uuid4().hex

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
            # Best-effort flush; neither grab nor retrieve proves exposure time.
            if self._fresh_request.is_set():
                self._fresh_request.clear()
                # Flush buffer: grab a few times, then read
                try:
                    started = time.monotonic()
                    for _ in range(3):
                        self.cap.grab()
                    ok, frame_bgr = self.cap.retrieve()
                    completed = time.monotonic()
                    if ok and frame_bgr is not None:
                        ts = time.time()
                        self._fresh_frame = self._timestamped_frame(frame_bgr, started, completed, ts, 'grab3_retrieve')
                    else:
                        self._fresh_frame = None
                except Exception:
                    self._fresh_frame = None
                self._fresh_ready.set()
                continue

            try:
                started = time.monotonic()
                ok, frame_bgr = self.cap.read()
                completed = time.monotonic()
            except Exception:
                break

            if not ok or frame_bgr is None:
                time.sleep(0.001)
                continue

            ts = time.time()
            cf = self._timestamped_frame(frame_bgr, started, completed, ts)

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
                if self.latest_frame.acquisition is not None:
                    self.latest_frame.acquisition.setdefault('first_main_pickup_monotonic', time.monotonic())
                self._last_pickup_count = self._read_count
            self.last_error = None

        # Camera capabilities are probed in start() before the reader thread
        # starts. Do not call VideoCapture.get() here while the reader thread
        # may concurrently be inside VideoCapture.read(); some native backends
        # are not safe for concurrent access to the same capture object.

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
        """Request a best-effort buffer flush and camera frame.

        Signals the reader thread to flush the camera buffer and capture
        a fresh frame. Blocks until the frame is ready (max 500ms).
        Exposure age remains unknown; this is not a calibrated timing guarantee.
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
