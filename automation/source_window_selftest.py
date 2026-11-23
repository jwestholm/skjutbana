"""Selftest + offline estimate for shot-window source recording (SKJUTBANA_SOURCE_RECORD_WINDOW).

    python3 -m automation.source_window_selftest                 # unit checks
    python3 -m automation.source_window_selftest SESSION_DIR...  # simulate on recorded sessions (no writes)
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from src.engine.source_recording import SourceRecorder, WindowGate


def unit():
    g = WindowGate(pre_s=1.0, post_s=2.0, heartbeat_s=3.0, level=0.5)
    written = []
    quiet = np.zeros(100, np.int16).tobytes(); loud = (np.ones(100) * 30000).astype(np.int16).tobytes()
    for i in range(300):                                 # 30 fps for 10 s, shot at 5.0 s
        t = i / 30
        if abs(t - 5.0) < 1e-9:
            written += g.audio(loud, t)
        written += g.camera(t, t)
        g.audio(quiet, t)
    assert written == sorted(written), "time order"
    win = [t for t in written if 4.0 <= t <= 7.0]
    assert len(win) >= 89, len(win)                      # full window 4.0..7.0 kept
    hb = [t for t in written if t < 4.0 or t > 7.0]
    assert 1 <= len(hb) <= 3 and all(b - a >= 3.0 - 1e-9 for a, b in zip(hb, hb[1:])), hb
    # recorder end-to-end: only window frames reach camera.bin, audio always complete
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.update(SKJUTBANA_SOURCE_RECORD_WINDOW="1", SKJUTBANA_SOURCE_RECORD_WINDOW_LEVEL="0.5",
                          SKJUTBANA_SOURCE_RECORD_WINDOW_PRE_S="0.5", SKJUTBANA_SOURCE_RECORD_WINDOW_POST_S="0.5",
                          SKJUTBANA_SOURCE_RECORD_WINDOW_HEARTBEAT_S="100")
        r = SourceRecorder(tmp, start_min_free_gb=0, stop_min_free_gb=0)
        assert r.start()
        jpg = np.frombuffer(b"\xff\xd8" + b"x" * 50, np.uint8)
        base = 1000.0
        for i in range(60):
            t = base + i * 0.05
            r.record_camera(jpg, dict(sequence=i, receive_wall_ts=t))
            r.record_audio(quiet, dict(sequence=i, receive_wall_ts=t))
            if i == 30:
                r.note_trigger(t, "injected")
            time.sleep(0.002)
        r.stop()
        d = r.directory
        cam = [json.loads(l)["acquisition"]["sequence"] for l in open(d / "camera_index.jsonl")]
        aud = sum(1 for _ in open(d / "audio_index.jsonl"))
        assert aud == 60, aud
        assert cam == sorted(cam) and cam[0] == 0 and 20 <= cam[1] <= 21 and 39 <= cam[-1] <= 40, cam  # first frame = heartbeat
        assert json.loads(open(d / "events.jsonl").readline())["kind"] == "window_mode"
        for k in [k for k in os.environ if k.startswith("SKJUTBANA_SOURCE_RECORD_WINDOW")]:
            del os.environ[k]
    print("source window selftest OK")


def simulate(dirs, **kw):
    from src.engine.camera_bin_store import open_camera_bin  # noqa: F401  (packed sessions are fine: index only)
    for sd in dirs:
        d = Path(sd)
        g = WindowGate(**kw)
        cam = [json.loads(l) for l in open(d / "camera_index.jsonl")]
        aud = [json.loads(l) for l in open(d / "audio_index.jsonl")]
        ev = [(r["acquisition"].get("receive_wall_ts"), "c", r["length"]) for r in cam]
        pcm = open(d / "audio.pcm", "rb")
        for r in aud:
            pcm.seek(r["offset"]); ev.append((r["block"].get("receive_wall_ts"), "a", pcm.read(r["length"])))
        ev.sort(key=lambda e: (e[0] or 0))
        kept = total = 0
        for t, k, x in ev:
            if t is None:
                continue
            if k == "a":
                kept += sum(g.audio(x, t))
            else:
                total += x
                kept += sum(g.camera(t, x))
        gt = d / "ground_truth.jsonl"
        miss = 0
        if gt.exists():
            print(d.name, "note: gt shots not checked against windows here")
        print(json.dumps(dict(session=d.name, raw_gb=round(total / 1e9, 2), kept_gb=round(kept / 1e9, 2),
                              kept_frac=round(kept / max(total, 1), 3), triggers=g.triggers, **g.config())), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        simulate(sys.argv[1:], level=float(os.environ.get("LEVEL", "0.1")))
    else:
        unit()
