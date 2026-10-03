from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.engine.settings import (
    load_audio_peak_threshold,
    save_audio_peak_threshold,
)


@dataclass
class AudioPeakEvent:
    timestamp: float
    peak: float
    rms: float
    acquisition: dict | None = None


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

    Nytt i denna version:
    - kortare audiochunk för bättre timing
    - kortare cooldown för snabbskytte
    - peak-ts approximeras inne i chunk i stället för att alltid sättas till chunk-slut
    - enkel crest-factor-kontroll för att dämpa host/musik/fest-brus
    - peak-events dispatchas som tidigare via global pub/sub i main thread
    """

    def __init__(self) -> None:
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_samples = 256

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

        self.min_abs_peak = load_audio_peak_threshold()
        self.peak_ratio = 3.2
        self.crest_factor_required = 1.7
        self.cooldown_s = 0.08

        self._events: deque[AudioPeakEvent] = deque(maxlen=200)
        self._pending_dispatch: deque[AudioPeakEvent] = deque()
        self._sample_history: deque[float] = deque(maxlen=self.sample_rate * 2)
        self._subscribers: list[Callable[[AudioPeakEvent], None]] = []
        self._lock = threading.Lock()
        # Observability only: acoustic delay/room response are not calibrated.
        # A coincident real shot must not be discarded merely because SFX plays.
        self._engine_playbacks: deque[dict] = deque(maxlen=64)
        self._playback_audio_events: deque[dict] = deque(maxlen=128)
        self._pcm_blocks: deque[tuple[dict, bytes]] = deque(maxlen=512)
        self._reset_acquisition()

    def _reset_acquisition(self) -> None:
        self._stream_id = uuid.uuid4().hex
        self._pcm_sequence = 0
        self._sample_index = 0
        self._first_receipt = None
        self._current_acquisition = None
        self._pcm_blocks.clear()

    def acquisition_blocks_since(self, sequence: int = 0) -> list[tuple[dict, bytes]]:
        """Bounded PCM diagnostics; receipt-relative estimates are not ADC time."""
        with self._lock:
            return [(dict(meta), pcm) for meta, pcm in self._pcm_blocks
                    if meta['sequence'] > sequence]

    def _read_pcm_block(self, stream) -> tuple[bytes, dict]:
        # A pipe may return fewer bytes than requested. Preserve partial samples
        # across reads instead of silently discarding PCM and losing chronology.
        started = time.monotonic()
        needed = self.chunk_samples * 2 * self.channels
        parts = bytearray()
        reads = 0
        while len(parts) < needed:
            piece = stream.read(needed-len(parts))
            reads += 1
            if not piece:
                break
            parts.extend(piece)
        received = time.monotonic()
        return bytes(parts), dict(read_start_monotonic=started,
            read_complete_monotonic=received, receive_wall_ts=time.time(),
            read_calls=reads, complete_block=len(parts)==needed)

    def _deliver_pcm_block(self, chunk: bytes, receipt: dict) -> None:
        n = len(chunk)//(2*self.channels)
        end = receipt['read_complete_monotonic']
        if self._first_receipt is None:
            self._first_receipt = end
        self._pcm_sequence += 1
        meta = dict(receipt, stream_id=self._stream_id, sequence=self._pcm_sequence,
            sample_start_index=self._sample_index, sample_end_index_exclusive=self._sample_index+n,
            sample_count=n, sample_rate=self.sample_rate, channels=self.channels,
            backend=self.backend_name, first_sample_receipt_estimate_monotonic=end-(n-1)/self.sample_rate,
            last_sample_receipt_estimate_monotonic=end,
            receipt_minus_stream_elapsed_s=end-self._first_receipt-self._sample_index/self.sample_rate,
            timestamp_semantics='RECEIPT_ANCHORED_ESTIMATE_NOT_DEVICE_ACQUISITION',
            device_timestamp=None, upstream_buffer_age_s=None)
        self._sample_index += n
        with self._lock:
            self._pcm_blocks.append((meta, chunk))
        self._current_acquisition = meta
        try:
            self._process_chunk(chunk)
        finally:
            self._current_acquisition = None

    @staticmethod
    def _peak_sample_index(samples, trigger_threshold):
        indices = np.flatnonzero(np.abs(samples) >= trigger_threshold)
        return int(indices[0]) if indices.size else int(np.argmax(np.abs(samples)))

    def _peak_acquisition(self, samples, trigger_threshold) -> dict | None:
        if self._current_acquisition is None:
            return None  # Direct/generated chunks have no device provenance.
        meta = dict(self._current_acquisition)
        offset = self._peak_sample_index(samples, trigger_threshold)
        meta.update(trigger_sample_offset=offset, maximum_sample_offset=int(np.argmax(np.abs(samples))),
            trigger_sample_index=meta['sample_start_index']+offset,
            trigger_receipt_estimate_monotonic=meta['last_sample_receipt_estimate_monotonic']
                -(len(samples)-1-offset)/self.sample_rate,
            detection_monotonic=time.monotonic())
        return meta

    def pending_events_snapshot(self) -> tuple[AudioPeakEvent, ...]:
        with self._lock:
            return tuple(self._pending_dispatch)

    def play_feedback(self, sound, *, owner: str, shot_id: int, cue: str) -> dict:
        """Dispatch known output and record its clock; never gate microphone input.

        Start/end are API/nominal sample times, not measured acoustic times.
        No Sound/Channel reference is retained by the microphone owner.
        """
        started = time.time()
        record = dict(owner=owner, shot_id=int(shot_id), cue=cue,
            start_ts=started, nominal_end_ts=None, status='UNAVAILABLE',
            semantics='PLAYBACK_TIMING_NOT_ACOUSTIC_ATTRIBUTION')
        try:
            duration = float(sound.get_length())
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError('invalid sound duration')
            record['nominal_end_ts'] = started + duration
            channel = sound.play()
            record['status'] = 'DISPATCHED' if channel is not None else 'NO_CHANNEL'
        except Exception as exc:
            record['error'] = type(exc).__name__
        record['dispatch_ms'] = (time.time()-started)*1000.
        with self._lock:
            self._engine_playbacks.append(dict(record))
        return record

    def playback_diagnostics(self, owner: str, shot_id: int) -> dict:
        with self._lock:
            outputs = [dict(p) for p in self._engine_playbacks
                       if p['owner']==owner and p['shot_id']==shot_id]
            events = [dict(e) for e in self._playback_audio_events
                      if any(p['owner']==owner and p['shot_id']==shot_id for p in e['playbacks'])]
        return dict(outputs=outputs, nearby_audio_events=events,
            suppression='NONE', attribution='UNRESOLVED_REQUIRES_PHYSICAL_MEASUREMENT')

    def _annotate_engine_playback(self, event: AudioPeakEvent) -> None:
        with self._lock:
            # Two seconds is a diagnostic look-back, NOT a trigger holdoff.
            nearby = [dict(p, delta_start_ms=(event.timestamp-p['start_ts'])*1000.,
                           within_nominal_playback=p['start_ts']<=event.timestamp<=p['nominal_end_ts'])
                for p in self._engine_playbacks if p['status']=='DISPATCHED'
                and 0 <= event.timestamp-p['start_ts'] <= 2.]
            if nearby:
                self._playback_audio_events.append(dict(timestamp=event.timestamp,
                    peak=event.peak,rms=event.rms,playbacks=nearby))
        if nearby:
            diagnostics = dict(getattr(event, 'diagnostics', {}) or {})
            diagnostics['engine_playback_context'] = dict(playbacks=nearby,
                attribution='COINCIDENCE_NOT_PROOF', suppressed=False)
            event.diagnostics = diagnostics
            import json
            print('[AUDIO-PLAYBACK-CONTEXT] '+json.dumps(dict(timestamp=event.timestamp,
                peak=event.peak,context=diagnostics['engine_playback_context']),separators=(',',':')))

    def start(self) -> bool:
        if self.running:
            return True

        if shutil.which("ffmpeg") is None:
            self.last_error = "ffmpeg saknas i PATH"
            self.enabled = False
            self.running = False
            return False

        self.min_abs_peak = load_audio_peak_threshold()
        self.last_error = ""
        self.enabled = True
        self.running = True
        self.last_peak_ts = 0.0

        with self._lock:
            self._events.clear()
            self._pending_dispatch.clear()
            self._sample_history.clear()
            self._reset_acquisition()

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

        with self._lock:
            self._pending_dispatch.clear()

    def update(self) -> None:
        """
        Dispatcha väntande audio peak-events i main thread.

        Detta gör att subscribers (debug UI, scanner, osv) inte körs från audio-tråden.
        """
        with self._lock:
            pending_events = list(self._pending_dispatch)
            self._pending_dispatch.clear()
            subscribers = list(self._subscribers)

        for ev in pending_events:
            if ev.acquisition is not None:
                ev.acquisition['dispatch_monotonic'] = time.monotonic()
            self._annotate_engine_playback(ev)
            for callback in subscribers:
                try:
                    callback(ev)
                except Exception:
                    # Låt aldrig en trasig subscriber stoppa resten av systemet.
                    pass

    def subscribe(self, callback: Callable[[AudioPeakEvent], None]) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[AudioPeakEvent], None]) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def get_events_since(self, since_ts: float) -> list[AudioPeakEvent]:
        with self._lock:
            return [ev for ev in self._events if ev.timestamp > since_ts]

    def get_latest_event(self) -> AudioPeakEvent | None:
        with self._lock:
            if not self._events:
                return None
            return self._events[-1]

    def get_status_lines(self) -> list[str]:
        with self._lock:
            subscriber_count = len(self._subscribers)
            pending_count = len(self._pending_dispatch)

        chunk_ms = 1000.0 * float(self.chunk_samples) / float(self.sample_rate)
        lines = [
            f"Audio backend: {self.backend_name}",
            f"Audio peak: {self.last_peak_value:.3f} | rms: {self.last_rms:.3f}",
            f"Noise floor: {self.noise_floor:.3f}",
            f"Peak threshold: {self.min_abs_peak:.3f}",
            f"Peak ratio: {self.peak_ratio:.2f} | crest: {self.crest_factor_required:.2f}",
            f"Chunk: {self.chunk_samples} samp ({chunk_ms:.1f} ms) | cooldown: {self.cooldown_s:.2f}s",
            f"Audio listeners: {subscriber_count}",
            f"Pending peak events: {pending_count}",
        ]
        if self.last_peak_ts > 0:
            age = time.time() - self.last_peak_ts
            lines.append(f"Last peak: {age:.2f}s ago")
        if self.last_error:
            lines.append(f"Audio error: {self.last_error}")
        return lines

    def get_waveform_snapshot(self, max_points: int = 1200) -> np.ndarray:
        with self._lock:
            if not self._sample_history:
                return np.zeros(max_points, dtype=np.float32)
            arr = np.array(self._sample_history, dtype=np.float32)

        if arr.size <= max_points:
            return arr

        idx = np.linspace(0, arr.size - 1, max_points).astype(np.int32)
        return arr[idx]

    def get_peak_threshold(self) -> float:
        return float(self.min_abs_peak)

    def set_peak_threshold(self, value: float, persist: bool = True) -> None:
        self.min_abs_peak = float(max(0.005, min(0.95, value)))
        if persist:
            save_audio_peak_threshold(self.min_abs_peak)

    # ------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------

    def _thread_main(self) -> None:
        backends = [
            (
                "pulse",
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "pulse",
                    "-i",
                    "default",
                    "-ac",
                    str(self.channels),
                    "-ar",
                    str(self.sample_rate),
                    "-f",
                    "s16le",
                    "-",
                ],
            ),
            (
                "alsa",
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "alsa",
                    "-i",
                    "default",
                    "-ac",
                    str(self.channels),
                    "-ar",
                    str(self.sample_rate),
                    "-f",
                    "s16le",
                    "-",
                ],
            ),
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

            try:
                chunk, receipt = self._read_pcm_block(self.proc.stdout)
            except Exception as exc:
                self.last_error = f"Kunde inte läsa audio från {backend_name}: {exc}"
                chunk = b""

            if chunk and receipt['complete_block']:
                self.backend_name = backend_name
                started = True
                self._deliver_pcm_block(chunk, receipt)
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
                chunk, receipt = self._read_pcm_block(self.proc.stdout)
            except Exception as exc:
                self.last_error = f"Audio read error: {exc}"
                break

            if not chunk or not receipt['complete_block']:
                self.last_error = 'Audio EOF/truncated block; device sample chronology ended'
                break

            self._deliver_pcm_block(chunk, receipt)

        self.enabled = False
        self.running = False
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None

    def _estimate_event_timestamp(
        self,
        samples: np.ndarray,
        chunk_end_ts: float,
        trigger_threshold: float,
    ) -> float:
        trigger_index = self._peak_sample_index(samples, trigger_threshold)

        trailing_samples = max(0, int(samples.size - 1 - trigger_index))
        return float(chunk_end_ts - (trailing_samples / float(self.sample_rate)))

    def _process_chunk(self, chunk: bytes) -> None:
        data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        if data.size == 0:
            return

        data /= 32768.0
        abs_data = np.abs(data)
        peak = float(np.max(abs_data))
        rms = float(np.sqrt(np.mean(np.square(data))))

        self.last_peak_value = peak
        self.last_rms = rms

        alpha = 0.03
        self.noise_floor = (1.0 - alpha) * self.noise_floor + alpha * rms

        with self._lock:
            self._sample_history.extend(float(x) for x in data.tolist())

        now = time.time()
        trigger_threshold = max(self.min_abs_peak, self.noise_floor * self.peak_ratio)
        crest_ok = peak >= max(self.min_abs_peak, rms * self.crest_factor_required)
        is_peak = (
            peak >= trigger_threshold
            and crest_ok
            and (now - self.last_peak_ts) >= self.cooldown_s
        )

        if is_peak:
            event_ts = self._estimate_event_timestamp(
                samples=data,
                chunk_end_ts=now,
                trigger_threshold=max(self.min_abs_peak, trigger_threshold * 0.85),
            )
            self.last_peak_ts = event_ts
            ev = AudioPeakEvent(timestamp=event_ts, peak=peak, rms=rms)
            ev.acquisition = self._peak_acquisition(data, max(self.min_abs_peak, trigger_threshold * 0.85))
            ev.diagnostics = {'acquisition': ev.acquisition}
            with self._lock:
                self._events.append(ev)
                self._pending_dispatch.append(ev)


audio_peak_detector = AudioPeakDetector()
