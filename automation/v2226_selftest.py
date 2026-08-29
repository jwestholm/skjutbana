from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.shot_track_v2226 import (
    SCHEMA_VERSION,
    TrackConfigV2226,
    _install_audio_telemetry_patch,
    update_tracks_frame_unique_v2226,
)


def check(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(label)
    print(f"[PASS] {label}")


def test_frame_unique_tracking() -> None:
    from src.engine.camera.hit_scanner import HitScanner

    scanner = HitScanner()
    scanner._active_tracks.clear()
    scanner._next_track_id = 1
    scanner.track_merge_radius_px = 12.0
    scanner.track_confirm_frames = 3

    frame1 = 100.000
    same_frame_cluster = [
        {"camera_x": 100.0, "camera_y": 200.0, "score": 30.0, "source": "v1"},
        {"camera_x": 101.0, "camera_y": 200.5, "score": 29.0, "source": "v2"},
        {"camera_x": 98.5, "camera_y": 201.0, "score": 27.0, "source": "bank"},
        {"camera_x": 103.0, "camera_y": 199.0, "score": 25.0, "source": "ridge"},
    ]
    diag1 = update_tracks_frame_unique_v2226(
        scanner, same_frame_cluster, frame1,
        config=TrackConfigV2226(track_log=False),
    )
    check(len(scanner._active_tracks) == 1, "same-frame nearby proposals share one track")
    track = next(iter(scanner._active_tracks.values()))
    check(track.hits == 1, "four same-frame candidates count as exactly one temporal hit")
    check(int(getattr(track, "v2226_same_frame_support", 0)) == 4, "same-frame agreement is retained as support")
    check(int(diag1["same_frame_support"]) == 3, "duplicate same-frame supports are counted separately")

    # Even if the scanner is accidentally called twice with the identical
    # physical camera timestamp, hits must not increase.
    update_tracks_frame_unique_v2226(
        scanner,
        [{"camera_x": 100.5, "camera_y": 200.2, "score": 31.0}],
        frame1,
        config=TrackConfigV2226(track_log=False),
    )
    check(track.hits == 1, "reprocessing the identical camera frame cannot fake persistence")

    # A genuinely later LOCAL-CONFIRM frame is a second observation.
    frame2 = 100.050
    confirm = [{
        "camera_x": 100.0,
        "camera_y": 200.0,
        "score": 35.0,
        "v2225_local_confirm": 1.0,
    }]
    diag2 = update_tracks_frame_unique_v2226(
        scanner, confirm, frame2,
        config=TrackConfigV2226(track_log=False),
    )
    check(track.hits == 2, "later local-confirm frame increments temporal hits once")
    check(int(diag2["temporal_matches"]) == 1, "later-frame confirmation is reported as temporal evidence")
    check(abs(track.last_seen_ts - frame2) < 1e-9, "track last_seen timestamp follows the confirming frame")

    # A second spatially separate hole proposal must remain a separate track.
    update_tracks_frame_unique_v2226(
        scanner,
        [{"camera_x": 160.0, "camera_y": 240.0, "score": 22.0}],
        frame2,
        config=TrackConfigV2226(track_log=False),
    )
    check(len(scanner._active_tracks) == 2, "spatially separate candidate remains a separate track")


def _pcm_chunk(peak: float, samples: int = 256) -> bytes:
    arr = np.zeros(samples, dtype=np.float32)
    arr[samples // 2] = float(peak)
    arr = np.clip(arr, -0.999, 0.999)
    return (arr * 32767.0).astype(np.int16).tobytes()


def test_audio_near_miss() -> None:
    _install_audio_telemetry_patch()
    from src.engine.audio.audio_peak_detector import AudioPeakDetector

    detector = AudioPeakDetector()
    detector.min_abs_peak = 0.50
    detector.peak_ratio = 3.2
    detector.crest_factor_required = 1.7
    detector.cooldown_s = 0.08
    detector.noise_floor = 0.01
    detector.last_peak_ts = 0.0

    # 0.40 is deliberately strong but below the absolute 0.50 gate. It must
    # remain rejected while becoming visible in telemetry.
    out = StringIO()
    with redirect_stdout(out):
        detector._process_chunk(_pcm_chunk(0.40))
    text = out.getvalue()
    decision = getattr(detector, "v2226_last_audio_decision", {})
    check(len(detector._pending_dispatch) == 0, "near-miss telemetry does not create a false audio event")
    check(decision.get("kind") == "near_miss", "strong rejected transient is retained as near-miss telemetry")
    check("abs" in decision.get("reasons", ()), "near-miss records exact absolute-threshold rejection")
    check("NEAR-MISS" in text, "near-miss is visible in terminal telemetry")

    # A clear peak must retain the historical trigger semantics.
    out = StringIO()
    with redirect_stdout(out):
        detector._process_chunk(_pcm_chunk(0.80))
    check(len(detector._pending_dispatch) == 1, "strong peak still creates exactly one normal AudioPeakEvent")
    check(getattr(detector, "v2226_last_audio_decision", {}).get("kind") == "trigger", "raw trigger decision is recorded")
    check("TRIGGER" in out.getvalue(), "raw trigger is visible before main-thread dispatch")


def test_files_and_entrypoint() -> None:
    main_text = (ROOT / "main.py").read_text(encoding="utf-8")
    check("from src.engine.shot_track_v2226 import install_v2226_runtime" in main_text, "main.py imports V2.22.6")
    check("install_v2226_runtime(App)" in main_text, "main.py installs V2.22.6")
    check(main_text.index("install_v2225_runtime(App)") < main_text.index("install_v2226_runtime(App)"), "V2.22.6 installs after V2.22.5")
    module_text = (ROOT / "src/engine/shot_track_v2226.py").read_text(encoding="utf-8")
    check("same-frame agreement" in module_text, "tracking semantics are documented in runtime module")
    check("AUDIO-RAW" in module_text, "audio raw/near-miss telemetry is present")


def main() -> None:
    print("V2.22.6 FRAME-UNIQUE TRACKING + AUDIO TELEMETRY SELFTEST")
    print("=========================================================")
    check(SCHEMA_VERSION == "2.22.6", "schema is 2.22.6")
    test_frame_unique_tracking()
    test_audio_near_miss()
    test_files_and_entrypoint()
    print("\nAll V2.22.6 selftests passed.")


if __name__ == "__main__":
    main()
