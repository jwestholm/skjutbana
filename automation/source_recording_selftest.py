"""Deterministic source-recording checks; never opens camera or microphone."""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.camera.camera_manager import CameraManager, apply_frame_transform
from src.engine.source_recording import SourceRecorder, SourceSessionReader, looks_like_mjpg


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def _jpeg(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.reshape(1, -1)  # same shape V4L2 returns with CONVERT_RGB=0


def test_roundtrip_bit_exact():
    with tempfile.TemporaryDirectory() as tmp:
        rec = SourceRecorder(Path(tmp) / "sessions", repo_root=ROOT, start_min_free_gb=0.0, stop_min_free_gb=0.0)
        check(rec.start({"camera": {"test": True}}), "recorder starts with explicit thresholds")
        payloads = [_jpeg(i) for i in range(5)]
        for i, buf in enumerate(payloads):
            rec.record_camera(buf, dict(sequence=i + 1, receive_wall_ts=100.0 + i * 0.04, rotation=180,
                                        mirror_horizontal=False, mirror_vertical=False))
        pcm = [np.arange(256, dtype=np.int16).tobytes(), (np.arange(256, dtype=np.int16) * -1).tobytes()]
        for i, block in enumerate(pcm):
            rec.record_audio(block, dict(sequence=i + 1, sample_rate=16000, channels=1))
        decoded = np.zeros((10, 10, 3), np.uint8)
        rec.record_camera(decoded, dict(sequence=99))
        rec.stop("selftest")
        reader = SourceSessionReader(rec.directory)
        frames = list(reader.iter_camera_frames())
        check(len(frames) == 5, "all MJPG frames persisted and indexed")
        for (row, pixels), buf in zip(frames, payloads):
            expected = apply_frame_transform(cv2.imdecode(buf.reshape(-1), cv2.IMREAD_COLOR), 180, False, False)
            if not np.array_equal(pixels, expected):
                raise AssertionError("replayed frame differs from live decode+transform")
        check(True, "replay decode+transform is bit-exact with live path")
        raw = [data for _row, data in reader.iter_camera_frames(decode=False)]
        check(raw == [b.reshape(-1).tobytes() for b in payloads], "raw camera bytes are preserved exactly")
        audio = [data for _row, data in reader.iter_audio_blocks()]
        check(audio == pcm, "PCM blocks are preserved exactly and in order")
        end = json.loads((rec.directory / "session_end.json").read_text())
        check(end["counts"]["decoded_not_recorded"] == 1, "decoded pixels are never recorded as MJPG")
        events = [json.loads(l) for l in (rec.directory / "events.jsonl").read_text().splitlines() if l]
        check(any(e["kind"] == "camera_payload_not_mjpg" for e in events), "non-MJPG payload is logged")
        session = json.loads((rec.directory / "session.json").read_text())
        check("content/settings.json" in session["settings"] and session["opencv_version"],
              "session records settings copies and decoder version")


def test_disk_guard_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        rec = SourceRecorder(Path(tmp) / "s", repo_root=ROOT, start_min_free_gb=1e9)
        check(not rec.start({}) and rec.state == "REFUSED" and "disk_free" in rec.reason,
              "disk guard refuses to start when free space is too low")


def test_queue_full_drops_are_counted():
    with tempfile.TemporaryDirectory() as tmp:
        rec = SourceRecorder(Path(tmp) / "s", repo_root=ROOT, start_min_free_gb=0.0, queue_max_items=1)
        rec.state = "RECORDING"  # no writer thread: queue cannot drain
        rec.record_camera(_jpeg(1), {"sequence": 1})
        rec.record_camera(_jpeg(2), {"sequence": 2})
        rec.record_audio(b"\0" * 512, {"sequence": 1})
        check(rec.counts["dropped_camera"] == 1 and rec.counts["dropped_audio"] == 1,
              "full queue drops are counted, never block acquisition")


def test_mjpg_detection_and_decode_modes():
    check(looks_like_mjpg(_jpeg(3)) and not looks_like_mjpg(np.zeros((4, 4, 3), np.uint8)),
          "MJPG byte buffers are distinguished from pixel arrays")
    cm = CameraManager.__new__(CameraManager)
    cm._raw_mjpg = True
    pixels, payload, mode = cm._decode_buffer(_jpeg(4))
    check(mode == "imdecode_color_from_mjpg" and payload is not None and pixels.ndim == 3,
          "raw mode decodes MJPG bytes with imdecode")
    pixels, payload, mode = cm._decode_buffer(np.zeros((4, 4, 3), np.uint8))
    check(mode == "backend_decoded_pixels" and payload is None, "backend pixel fallback is labelled")
    cm._raw_mjpg = False
    _pixels, payload, mode = cm._decode_buffer(np.zeros((4, 4, 3), np.uint8))
    check(mode == "cap_read" and payload is None, "ordinary mode keeps cap.read pixels")


def main():
    test_roundtrip_bit_exact()
    test_disk_guard_refuses()
    test_queue_full_drops_are_counted()
    test_mjpg_detection_and_decode_modes()
    print("ALL SOURCE RECORDING SELFTESTS PASSED")


if __name__ == "__main__":
    main()
