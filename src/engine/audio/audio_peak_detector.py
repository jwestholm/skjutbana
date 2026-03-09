from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioPeakEvent:
    timestamp: float
    peak: float
    rms: float


class AudioPeakDetector:
    """
    Enkel peak-detektor för mikrofon.

    Design:
    - använder ffmpeg som input-backend
    - försöker först PulseAudio default device
    - fallback till ALSA default
    - triggar på snabba peaks, inte på "rätt" skottljud

    Syfte:
    - INTE att ensam avgöra träff
    - bara att öppna ett extra noggrant bildanalysfönster
    """

    def __init__(self) -> None:
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_samples = 1024

        self.enabled = False
        self.running = False

        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen | None = None

        self.backend_name = "none"
        self.last_error = ""
        self.last_peak_ts = 0.0
        self.last_rms = 0.0
        self.last_peak_value = 0.0

        self.noise_floor = 0.01
        self.min_abs_peak = 0.10
        self.peak_ratio = 3.2
        self.cooldown_s = 0.20

        self._events: deque[AudioPeakEvent] = deque(maxlen=100)
        self._lock = threading.Lock()

    def start(self) -> bool:
        if self.running:
            return True

        if shutil.which("ffmpeg") is None:
            self.last_error = "ffmpeg saknas i PATH"
            self.enabled = False
            self.running = False
            return False

        self.last_error = ""
        self.enabled = True
        self.running = True

        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        self.running = False
        self.enabled = False

        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None

        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def update(self) -> None:
        """
        Håller API:t enkelt för app-loopen.
        Själva inspelningen sker i bakgrundstråd.
        """
        return None

    def get_events_since(self, since_ts: float) -> list[AudioPeakEvent]:
        with self._lock:
            return [ev for ev in self._events if ev.timestamp > since_ts]

    def get_latest_event(self) -> AudioPeakEvent | None:
        with self._lock:
            if not self._events:
                return None
            return self._events[-1]

    def get_status_lines(self) -> list[str]:
        lines = [
            f"Audio backend: {self.backend_name}",
            f"Audio peak: {self.last_peak_value:.3f} | rms: {self.last_rms:.3f}",
            f"Noise floor: {self.noise_floor:.3f}",
        ]
        if self.last_peak_ts > 0:
            age = time.time() - self.last_peak_ts
            lines.append(f"Last peak: {age:.2f}s ago")
        if self.last_error:
            lines.append(f"Audio error: {self.last_error}")
        return lines

    # ------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------

    def _thread_main(self) -> None:
        backends = [
            ("pulse", ["ffmpeg", "-hide_banner", "-loglevel", "error",
                       "-f", "pulse", "-i", "default",
                       "-ac", str(self.channels), "-ar", str(self.sample_rate),
                       "-f", "s16le", "-"]),
            ("alsa", ["ffmpeg", "-hide_banner", "-loglevel", "error",
                      "-f", "alsa", "-i", "default",
                      "-ac", str(self.channels), "-ar", str(self.sample_rate),
                      "-f", "s16le", "-"]),
        ]

        started = False

        for backend_name, cmd in backends:
            if not self.running:
                return

            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
            except Exception as exc:
                self.last_error = f"Kunde inte starta {backend_name}: {exc}"
                self.proc = None
                continue

            # prova läsa lite data
            try:
                bytes_needed = self.chunk_samples * 2
                chunk = self.proc.stdout.read(bytes_needed) if self.proc.stdout else b""
            except Exception as exc:
                self.last_error = f"Kunde inte läsa audio från {backend_name}: {exc}"
                chunk = b""

            if chunk and len(chunk) >= bytes_needed:
                self.backend_name = backend_name
                started = True
                self._process_chunk(chunk)
                break

            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None

        if not started:
            self.last_error = self.last_error or "Ingen audio-backend gav PCM-data"
            self.running = False
            self.enabled = False
            return

        while self.running and self.proc is not None:
            try:
                bytes_needed = self.chunk_samples * 2
                chunk = self.proc.stdout.read(bytes_needed) if self.proc.stdout else b""
            except Exception as exc:
                self.last_error = f"Audio read error: {exc}"
                break

            if not chunk or len(chunk) < bytes_needed:
                time.sleep(0.01)
                continue

            self._process_chunk(chunk)

        self.enabled = False
        self.running = False

        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None

    def _process_chunk(self, chunk: bytes) -> None:
        data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        if data.size == 0:
            return

        data /= 32768.0

        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(np.square(data))))

        self.last_peak_value = peak
        self.last_rms = rms

        # uppdatera noise floor försiktigt
        alpha = 0.03
        self.noise_floor = (1.0 - alpha) * self.noise_floor + alpha * rms

        now = time.time()

        is_peak = (
            peak >= self.min_abs_peak
            and peak >= max(self.min_abs_peak, self.noise_floor * self.peak_ratio)
            and (now - self.last_peak_ts) >= self.cooldown_s
        )

        if is_peak:
            self.last_peak_ts = now
            ev = AudioPeakEvent(timestamp=now, peak=peak, rms=rms)
            with self._lock:
                self._events.append(ev)


audio_peak_detector = AudioPeakDetector()