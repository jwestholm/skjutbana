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


def apply_frame_transform(frame_bgr: np.ndarray, rotation: int, mirror_horizontal: bool,
                          mirror_vertical: bool) -> np.ndarray:
    """Shared by live acquisition and offline source replay (same pixels)."""
    frame = frame_bgr
    if rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mirror_horizontal and mirror_vertical:
        frame = cv2.flip(frame, -1)
    elif mirror_horizontal:
        frame = cv2.flip(frame, 1)
    elif mirror_vertical:
        frame = cv2.flip(frame, 0)
    return frame


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
        self._source_recorder = None
        self._raw_mjpg = False

    def _timestamped_frame(self, pixels, started, completed, wall_ts, kind='read', decode_mode='cap_read'):
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
            receive_wall_ts=wall_ts, capture_kind=kind, decode_mode=decode_mode, backend_timestamps_raw=backend,
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
        # Opt-in: camera.exposure_time_absolute (units of 100 us). 167 = 16.7 ms
        # = one 60 Hz projector period, which removes rolling-shutter banding.
        # Unset keeps whatever exposure the device currently has.
        exposure = settings.get("exposure_time_absolute")
        self.exposure_time_absolute = None if exposure is None else int(exposure)

    def _apply_frame_transform(self, frame_bgr: np.ndarray) -> np.ndarray:
        return apply_frame_transform(frame_bgr, self.rotation, self.mirror_horizontal, self.mirror_vertical)

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
            # BUFFERSIZE=1 measured to drop every other 4K frame (25 -> ~12 fps):
            # the single V4L2 buffer is dequeued while the next frame arrives.
            # Two buffers gave 24 fps and fresher frames (63 vs 82 ms age).
            accepted = bool(self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2))
            self.buffer_request = dict(requested=2, accepted=accepted,
                reported=float(self.cap.get(cv2.CAP_PROP_BUFFERSIZE)),
                semantics='DRIVER_REPORT_NOT_PROOF_OF_ZERO_BACKLOG')
        except Exception:
            self.buffer_request = dict(requested=2, accepted=False, reported=None)

        self.property_apply_result = apply_preferred_camera_settings(
            self.cap,
            preferred_width=self.preferred_width,
            preferred_height=self.preferred_height,
            preferred_fps=self.preferred_fps,
            exposure_time_absolute=self.exposure_time_absolute,
        )
        self.capabilities = probe_camera_capabilities(self.cap)

        # Opt-in source recording: receive raw MJPG bytes and decode them here
        # with cv2.imdecode, so the exact bytes can be replayed offline through
        # the same decode + transform. Ordinary runs keep cap.read() decoding.
        self._source_recorder = None
        self._raw_mjpg = False
        try:
            from src.engine.source_recording import get_source_recorder
            recorder = get_source_recorder()
            if recorder is not None:
                self._raw_mjpg = bool(self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0))
                # One extra driver buffer of headroom for the recording writer.
                if self._raw_mjpg:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    self.buffer_request = dict(self.buffer_request, requested=3,
                                               reported=float(self.cap.get(cv2.CAP_PROP_BUFFERSIZE)))
                caps = self.capabilities
                started = recorder.start(dict(
                    camera=dict(index=self.camera_index, backend=getattr(caps, "backend_name", None),
                                width=getattr(caps, "width", None), height=getattr(caps, "height", None),
                                fps=getattr(caps, "fps", None), fourcc=getattr(caps, "fourcc", None),
                                capabilities=repr(caps), property_apply_result=self.property_apply_result,
                                buffer_request=self.buffer_request, convert_rgb_0_accepted=self._raw_mjpg),
                    transform=dict(rotation=self.rotation, mirror_horizontal=self.mirror_horizontal,
                                   mirror_vertical=self.mirror_vertical)))
                if started:
                    self._source_recorder = recorder
                elif self._raw_mjpg:
                    self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                    self._raw_mjpg = False
        except Exception as exc:
            print(f"[SOURCE-RECORD] camera hook unavailable: {type(exc).__name__}: {exc}")

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

    def stop_for_exit(self) -> None:
        """Quiesce the camera before the source writer is drained."""
        self.running = False
        reader = self._reader_thread
        if reader is not None:
            reader.join(timeout=2.0)
            if reader.is_alive():
                # Concurrent release and native read are unsafe. Leave the
                # handle owned until the process exits and continue teardown.
                raise TimeoutError("Camera reader is still in a native read")
        self.stop()

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
                    ok, raw = self.cap.retrieve()
                    completed = time.monotonic()
                    frame_bgr, payload, decode_mode = self._decode_buffer(raw) if ok else (None, None, None)
                    if ok and frame_bgr is not None:
                        ts = time.time()
                        self._fresh_frame = self._timestamped_frame(frame_bgr, started, completed, ts, 'grab3_retrieve',
                                                                    decode_mode=decode_mode)
                        self._record_source(payload, self._fresh_frame)
                    else:
                        self._fresh_frame = None
                except Exception:
                    self._fresh_frame = None
                self._fresh_ready.set()
                continue

            try:
                started = time.monotonic()
                ok, raw = self.cap.read()
                completed = time.monotonic()
            except Exception:
                break

            frame_bgr, payload, decode_mode = self._decode_buffer(raw) if ok else (None, None, None)
            if not ok or frame_bgr is None:
                time.sleep(0.001)
                continue

            ts = time.time()
            cf = self._timestamped_frame(frame_bgr, started, completed, ts, decode_mode=decode_mode)
            self._record_source(payload, cf)
            self.inject_frame(cf)

    def _decode_buffer(self, raw):
        """Return (pixels, mjpg_payload_or_None, decode_mode)."""
        if raw is None:
            return None, None, None
        if self._raw_mjpg:
            from src.engine.source_recording import looks_like_mjpg
            if looks_like_mjpg(raw):
                pixels = cv2.imdecode(raw.reshape(-1), cv2.IMREAD_COLOR)
                return pixels, raw, 'imdecode_color_from_mjpg'
            # Backend ignored CONVERT_RGB=0 for this frame: pixels, not bytes.
            return raw, None, 'backend_decoded_pixels'
        return raw, None, 'cap_read'

    def _record_source(self, payload, cf) -> None:
        recorder = getattr(self, '_source_recorder', None)
        if recorder is not None and cf is not None:
            if payload is not None:
                recorder.record_camera(payload, cf.acquisition or {})
            else:
                recorder.record_camera(cf.frame_bgr, cf.acquisition or {})  # logged as not-MJPG

    def inject_frame(self, cf: CameraFrame) -> None:
        """Publish a timestamped frame exactly as the reader thread does.

        Live acquisition and offline source replay share this entry point.
        """
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
