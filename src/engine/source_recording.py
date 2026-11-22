"""Opt-in continuous source recording: raw camera MJPG bytes + raw PCM.

Records the acquisition *sources* so a physical session can be replayed offline
through the same decode/detector code. Enabled only with
``SKJUTBANA_SOURCE_RECORD=1``; root defaults to /data/skjutbana/source_sessions
(override ``SKJUTBANA_SOURCE_RECORD_ROOT``).

Session layout (one directory per process run)::

    session.json          provenance: git, OpenCV, camera caps/transforms, settings copies+hashes
    camera.bin            concatenated raw camera buffers (MJPG bytes as delivered)
    camera_index.jsonl    one line per frame: offset, length, payload kind, acquisition metadata
    audio.pcm             concatenated s16le PCM exactly as delivered to the detector
    audio_index.jsonl     one line per PCM block: offset, length, block metadata
    events.jsonl          drops, decode fallbacks, disk-guard stops
    session_end.json      written on clean stop

Recording never blocks acquisition: a bounded queue feeds one writer thread and
every dropped item is counted and logged. A disk guard refuses to start below
``START_MIN_FREE_GB`` and stops cleanly below ``STOP_MIN_FREE_GB``.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

SCHEMA = "source-session-1"
DEFAULT_ROOT = Path("/data/skjutbana/source_sessions")
START_MIN_FREE_GB = 4.0
STOP_MIN_FREE_GB = 2.0
QUEUE_MAX_ITEMS = 400          # ~10 s of 25 fps video plus audio blocks
DISK_CHECK_EVERY_BYTES = 64 * 1024 * 1024

PAYLOAD_MJPG = "MJPG_BYTES"
PAYLOAD_DECODED = "DECODED_PIXELS_NOT_RECORDED"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip()
    except Exception:
        return None


def looks_like_mjpg(buffer: Any) -> bool:
    """True only for a compressed JPEG byte buffer (SOI marker), never pixels."""
    arr = np.asarray(buffer)
    if arr.dtype != np.uint8 or arr.size < 4:
        return False
    if not (arr.ndim == 1 or (arr.ndim == 2 and min(arr.shape) == 1)):
        return False
    flat = arr.reshape(-1)
    return int(flat[0]) == 0xFF and int(flat[1]) == 0xD8


class WindowGate:
    """Shot-window recording (opt-in ``SKJUTBANA_SOURCE_RECORD_WINDOW=1``).

    Camera frames are kept in a ring of ``pre_s`` seconds and written only around triggers: every audio
    block whose peak reaches ``level`` (a fraction of full scale, default 0.5 x the detector's absolute
    threshold so near-threshold shots are kept too) and every explicit trigger (detected or injected
    shot).  A frame opens/extends nothing by itself; frames up to ``post_s`` after the last trigger are
    written, and one heartbeat frame every ``heartbeat_s`` keeps the board state between shots.  Audio is
    always recorded in full.  Pure logic so the same rule can be simulated offline on old sessions.
    """

    def __init__(self, pre_s: float = 1.5, post_s: float = 2.5, heartbeat_s: float = 2.0, level: float = 0.1):
        self.pre_s, self.post_s, self.heartbeat_s, self.level = float(pre_s), float(post_s), float(heartbeat_s), float(level)
        self.ring: list[tuple[float, Any]] = []
        self.open_until = -1e18
        self.last_written = -1e18
        self.skipped = 0
        self.triggers = 0

    @classmethod
    def from_env(cls) -> "WindowGate | None":
        if os.environ.get("SKJUTBANA_SOURCE_RECORD_WINDOW") != "1":
            return None
        env = lambda k, d: float(os.environ.get(k, d))
        level = os.environ.get("SKJUTBANA_SOURCE_RECORD_WINDOW_LEVEL")
        if level is None:
            try:
                from src.engine.settings import load_audio_peak_threshold
                level = 0.5 * float(load_audio_peak_threshold())
            except Exception:
                level = 0.1
        return cls(env("SKJUTBANA_SOURCE_RECORD_WINDOW_PRE_S", 1.5), env("SKJUTBANA_SOURCE_RECORD_WINDOW_POST_S", 2.5),
                   env("SKJUTBANA_SOURCE_RECORD_WINDOW_HEARTBEAT_S", 2.0), float(level))

    def config(self) -> dict:
        return dict(pre_s=self.pre_s, post_s=self.post_s, heartbeat_s=self.heartbeat_s, level=self.level)

    def trigger(self, ts: float) -> list:
        """Open the window at ts; returns ring items (oldest first) that must now be written."""
        self.triggers += 1
        self.open_until = max(self.open_until, ts + self.post_s)
        keep = [(t, item) for t, item in self.ring if t >= ts - self.pre_s]
        self.skipped += len(self.ring) - len(keep)
        self.ring = []
        if keep:
            self.last_written = max(self.last_written, keep[-1][0])
        return [item for _, item in keep]

    def audio(self, pcm: bytes, ts: float) -> list:
        data = np.frombuffer(pcm, dtype=np.int16)
        if data.size and float(np.abs(data.astype(np.int32)).max()) / 32768.0 >= self.level:
            return self.trigger(ts)
        return []

    def camera(self, ts: float, item: Any) -> list:
        """Items to write now, oldest first (time order is preserved).  Inside an open window the frame is
        written at once; otherwise it waits in the ring, and a frame leaving the ring is written only as a
        heartbeat (>= heartbeat_s after the last written frame)."""
        if ts <= self.open_until:
            self.last_written = ts
            return [item]
        self.ring.append((ts, item))
        out = []
        while self.ring and self.ring[0][0] < ts - self.pre_s:
            t, old = self.ring.pop(0)
            if t - self.last_written >= self.heartbeat_s:
                self.last_written = t
                out.append(old)
            else:
                self.skipped += 1
        return out

class SourceRecorder:
    def __init__(self, root: Path | str | None = None, *, repo_root: Path | None = None,
                 start_min_free_gb: float = START_MIN_FREE_GB,
                 stop_min_free_gb: float = STOP_MIN_FREE_GB,
                 queue_max_items: int = QUEUE_MAX_ITEMS) -> None:
        self.root = Path(root or os.environ.get("SKJUTBANA_SOURCE_RECORD_ROOT") or DEFAULT_ROOT)
        self.repo_root = Path(repo_root or Path.cwd())
        self.start_min_free_gb = float(start_min_free_gb)
        self.stop_min_free_gb = float(stop_min_free_gb)
        self.directory: Path | None = None
        self.state = "IDLE"          # IDLE -> RECORDING -> STOPPED | REFUSED
        self.reason: str | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=int(queue_max_items))
        self._lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.counts = dict(camera_frames=0, audio_blocks=0, camera_bytes=0, audio_bytes=0,
                           dropped_camera=0, dropped_audio=0, decoded_not_recorded=0)
        self._since_disk_check = 0
        self.window = WindowGate.from_env()

    # -- lifecycle -----------------------------------------------------
    def free_gb(self) -> float:
        probe = self.root if self.root.exists() else self.root.parent
        return shutil.disk_usage(probe).free / 1e9

    def start(self, provenance: Mapping[str, Any] | None = None) -> bool:
        with self._lock:
            if self.state == "RECORDING":
                # Later sources (camera after audio, or vice versa) add provenance.
                if provenance:
                    self.add_provenance(provenance)
                return True
            if self.state != "IDLE":
                return False
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                free = self.free_gb()
            except OSError as exc:
                self.state, self.reason = "REFUSED", f"root_unavailable:{exc}"
                print(f"[SOURCE-RECORD] refused: {self.reason}")
                return False
            if free < self.start_min_free_gb:
                self.state = "REFUSED"
                self.reason = f"disk_free_{free:.1f}GB_below_{self.start_min_free_gb:.1f}GB"
                print(f"[SOURCE-RECORD] refused: {self.reason}")
                return False
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            self.directory = self.root / f"{stamp}_{uuid.uuid4().hex[:8]}"
            self.directory.mkdir(parents=False, exist_ok=False)
            self._write_session_json({}, free)
            if provenance:
                self.add_provenance(provenance)
            self._camera_bin = open(self.directory / "camera.bin", "ab")
            self._camera_idx = open(self.directory / "camera_index.jsonl", "a", encoding="utf-8")
            self._audio_bin = open(self.directory / "audio.pcm", "ab")
            self._audio_idx = open(self.directory / "audio_index.jsonl", "a", encoding="utf-8")
            self._events = open(self.directory / "events.jsonl", "a", encoding="utf-8")
            self.state = "RECORDING"
            if self.window is not None:
                self._events.write(json.dumps(dict(kind="window_mode", wall=time.time(), **self.window.config())) + "\n")
                self._events.flush()
                print(f"[SOURCE-RECORD] shot-window mode {self.window.config()}")
            self._thread = threading.Thread(target=self._run, name="source-recorder", daemon=True)
            self._thread.start()
            atexit.register(self.stop)
            print(f"[SOURCE-RECORD] recording to {self.directory} (free {free:.1f} GB)")
            return True

    def _write_session_json(self, provenance: Mapping[str, Any], free_gb: float) -> None:
        import cv2
        settings = {}
        for rel in ("content/settings.json", "content/ai/settings.json"):
            path = self.repo_root / rel
            entry: dict[str, Any] = {"path": rel, "sha256": _sha256(path)}
            try:
                entry["content"] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                entry["content"] = None
                entry["error"] = type(exc).__name__
            settings[rel] = entry
        session = dict(
            schema=SCHEMA,
            created_wall=time.time(), created_monotonic=time.monotonic(),
            git=dict(commit=_git(self.repo_root, "rev-parse", "HEAD"),
                     branch=_git(self.repo_root, "branch", "--show-current"),
                     dirty=bool(_git(self.repo_root, "status", "--porcelain"))),
            opencv_version=cv2.__version__, numpy_version=np.__version__,
            disk_free_gb_at_start=round(free_gb, 2),
            audio_format=dict(encoding="s16le"),
            camera_payload_semantics="raw buffer returned by cap.read() with CAP_PROP_CONVERT_RGB=0; "
                                     "replay must decode with cv2.imdecode(IMREAD_COLOR) then apply the "
                                     "recorded rotation/mirror transform",
            clocks="receive_wall_ts=time.time(); *_monotonic=time.monotonic(); same host",
            settings=settings,
            provenance=_json_safe(dict(provenance)),
        )
        (self.directory / "session.json").write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")

    def add_provenance(self, provenance: Mapping[str, Any]) -> None:
        """Append a provenance section (one JSON line) without rewriting session.json."""
        if self.directory is None:
            return
        row = dict(wall=time.time(), provenance=_json_safe(dict(provenance)))
        with open(self.directory / "provenance.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    def stop(self, reason: str = "clean_stop") -> None:
        with self._stop_lock:
            self._stop(reason, drain=False)

    def stop_for_exit(self) -> None:
        """Drain before publishing session_end; the supervisor bounds stalled I/O."""
        with self._stop_lock:
            self._stop("application_exit", drain=True)

    def _stop(self, reason: str, *, drain: bool) -> None:
        with self._lock:
            if self.state != "RECORDING":
                return
            self.state = "STOPPING"
            self.reason = reason
        writer = self._thread
        if writer is not None and writer is not threading.current_thread():
            while writer.is_alive():
                try:
                    self._queue.put(None, timeout=2.0)
                    break
                except queue.Full:
                    if not drain:
                        break
            writer.join(timeout=None if drain else 10.0)
        remaining = 0
        while True:   # anything still queued was not written: count it, never hide it
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            remaining += 1
            if item[0] == "camera":
                self.counts["dropped_camera"] += 1
            elif item[0] == "audio":
                self.counts["dropped_audio"] += 1
        self.counts["unwritten_at_stop"] = remaining
        if self.window is not None:
            self.counts["window_skipped_frames"] = self.window.skipped + len(self.window.ring)
            self.counts["window_triggers"] = self.window.triggers
        self._close_files()
        if self.directory is not None:
            end = dict(stopped_wall=time.time(), reason=reason, counts=dict(self.counts))
            (self.directory / "session_end.json").write_text(json.dumps(end, indent=2) + "\n", encoding="utf-8")
            print(f"[SOURCE-RECORD] stopped ({reason}) {self.counts}")
        self.state = "STOPPED"

    def _close_files(self) -> None:
        for name in ("_camera_bin", "_camera_idx", "_audio_bin", "_audio_idx", "_events"):
            handle = getattr(self, name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

    # -- producers (never block) ---------------------------------------
    def record_camera(self, payload: Any, metadata: Mapping[str, Any]) -> None:
        if self.state != "RECORDING":
            return
        if not looks_like_mjpg(payload):
            self.counts["decoded_not_recorded"] += 1
            self._event("camera_payload_not_mjpg", sequence=metadata.get("sequence"))
            return
        data = np.asarray(payload).reshape(-1).tobytes()
        try:
            self._queue.put_nowait(("camera", data, dict(metadata)))
        except queue.Full:
            self.counts["dropped_camera"] += 1
            self._event("camera_dropped_queue_full", sequence=metadata.get("sequence"))

    def record_audio(self, pcm: bytes, metadata: Mapping[str, Any]) -> None:
        if self.state != "RECORDING":
            return
        try:
            self._queue.put_nowait(("audio", bytes(pcm), dict(metadata)))
        except queue.Full:
            self.counts["dropped_audio"] += 1
            self._event("audio_dropped_queue_full", sequence=metadata.get("sequence"))

    def note_trigger(self, wall_ts: float | None = None, source: str = "shot") -> None:
        """A shot (detected or injected) happened: in shot-window mode the frames around it are kept."""
        if self.state != "RECORDING" or self.window is None:
            return
        try:
            self._queue.put_nowait(("trigger", b"", dict(wall=float(wall_ts if wall_ts is not None else time.time()),
                                                         source=source)))
        except queue.Full:
            self.counts["dropped_trigger"] = self.counts.get("dropped_trigger", 0) + 1

    def _event(self, kind: str, **fields: Any) -> None:
        try:
            self._queue.put_nowait(("event", b"", dict(kind=kind, wall=time.time(), **fields)))
        except queue.Full:
            pass

    # -- writer --------------------------------------------------------
    def _write_camera(self, data: bytes, meta: Mapping[str, Any]) -> None:
        offset = self._camera_bin.tell()
        self._camera_bin.write(data)
        row = dict(offset=offset, length=len(data), payload=PAYLOAD_MJPG, acquisition=_json_safe(meta))
        self._camera_idx.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.counts["camera_frames"] += 1
        self.counts["camera_bytes"] += len(data)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            kind, data, meta = item
            try:
                if self.window is not None and kind in ("camera", "audio", "trigger"):
                    if kind == "camera":
                        ts = float(meta.get("receive_wall_ts") or time.time())
                        for d, m in self.window.camera(ts, (data, meta)):
                            self._write_camera(d, m)
                        continue
                    else:
                        ts = float(meta.get("wall") if kind == "trigger" else (meta.get("receive_wall_ts") or time.time()))
                        held = self.window.trigger(ts) if kind == "trigger" else self.window.audio(data, ts)
                        for d, m in held:
                            self._write_camera(d, m)
                        if kind == "trigger":
                            continue
                if kind == "camera":
                    self._write_camera(data, meta)
                elif kind == "audio":
                    offset = self._audio_bin.tell()
                    self._audio_bin.write(data)
                    row = dict(offset=offset, length=len(data), block=_json_safe(meta))
                    self._audio_idx.write(json.dumps(row, separators=(",", ":")) + "\n")
                    self.counts["audio_blocks"] += 1
                    self.counts["audio_bytes"] += len(data)
                else:
                    self._events.write(json.dumps(_json_safe(meta), separators=(",", ":")) + "\n")
                self._since_disk_check += len(data)
                if self._since_disk_check >= DISK_CHECK_EVERY_BYTES:
                    self._since_disk_check = 0
                    for handle in (self._camera_bin, self._camera_idx, self._audio_bin, self._audio_idx, self._events):
                        handle.flush()
                    free = self.free_gb()
                    if free < self.stop_min_free_gb:
                        self._events.write(json.dumps(dict(kind="disk_guard_stop", free_gb=round(free, 2),
                                                           wall=time.time())) + "\n")
                        threading.Thread(target=self.stop, args=(f"disk_free_{free:.1f}GB",), daemon=True).start()
                        break
            except Exception as exc:  # never crash acquisition because of recording
                try:
                    self._events.write(json.dumps(dict(kind="writer_error", error=repr(exc), wall=time.time())) + "\n")
                except Exception:
                    pass


_RECORDER: SourceRecorder | None = None
_RECORDER_LOCK = threading.Lock()


def source_recording_enabled() -> bool:
    return os.environ.get("SKJUTBANA_SOURCE_RECORD") == "1"


def get_source_recorder() -> SourceRecorder | None:
    """Process-wide recorder, or None when recording is not enabled."""
    global _RECORDER
    if not source_recording_enabled():
        return None
    with _RECORDER_LOCK:
        if _RECORDER is None:
            _RECORDER = SourceRecorder()
        return _RECORDER


# ---------------------------------------------------------------------------
# Reader (used by offline replay; identical decode contract)
# ---------------------------------------------------------------------------
class SourceSessionReader:
    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.session = json.loads((self.directory / "session.json").read_text(encoding="utf-8"))
        if self.session.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported source session schema: {self.session.get('schema')}")

    @staticmethod
    def _rows(path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def camera_index(self) -> list[dict[str, Any]]:
        return list(self._rows(self.directory / "camera_index.jsonl"))

    def audio_index(self) -> list[dict[str, Any]]:
        return list(self._rows(self.directory / "audio_index.jsonl"))

    def iter_camera_frames(self, *, decode: bool = True) -> Iterator[tuple[dict[str, Any], Any]]:
        """Yield (index_row, frame). Frames get the recorded transform, as live."""
        import cv2
        from src.engine.camera.camera_manager import apply_frame_transform
        from src.engine.camera_bin_store import open_camera_bin
        with open_camera_bin(self.directory) as handle:
            for row in self.camera_index():
                handle.seek(int(row["offset"]))
                data = handle.read(int(row["length"]))
                if not decode:
                    yield row, data
                    continue
                pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                acq = row.get("acquisition") or {}
                if pixels is not None:
                    pixels = apply_frame_transform(pixels, int(acq.get("rotation", 0) or 0),
                                                   bool(acq.get("mirror_horizontal")),
                                                   bool(acq.get("mirror_vertical")))
                yield row, pixels

    def iter_audio_blocks(self) -> Iterator[tuple[dict[str, Any], bytes]]:
        with open(self.directory / "audio.pcm", "rb") as handle:
            for row in self.audio_index():
                handle.seek(int(row["offset"]))
                yield row, handle.read(int(row["length"]))
