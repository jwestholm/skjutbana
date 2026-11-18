"""Replay a recorded source session through the installed LIVE shot path.

  python3 -m automation.bench replay SESSION_DIR --output DIR [--use-current-settings]

The session's own recorded settings/calibration are used by default (portable
replay). Camera MJPG bytes are decoded with the same imdecode+transform as live
recording mode and injected at their recorded receive times; PCM blocks are
delivered at their recorded receipt times. Optional session sidecars:

  ground_truth.jsonl     {"wall_ts": <approx shot time>, "camera_x":..,"camera_y":..}  or  {"wall_ts":..,"kind":"no_impact"}
  synthetic_audio.jsonl  {"wall_ts": .., "amplitude": ..}   (mixed into recorded PCM, e.g. optical-hole sessions)

Evidence level: RECORDED_SOURCE_LIVE_PATH (real sources, virtual scheduling).
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

EVIDENCE_LEVEL = "RECORDED_SOURCE_LIVE_PATH"


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def replay(session_dir: Path, output: Path, *, use_current_settings: bool = False,
           worker_cost_s: float = 0.15, tolerance_px: float = 8.0, tail_s: float = 4.5,
           pulse_offset_s: float = 0.0, export_candidates: bool = False,
           label_min_age_s: float = 0.15) -> dict[str, Any]:
    from src.engine.source_recording import SourceSessionReader
    from src.engine.offline.virtual_runtime import VirtualClock, VirtualRuntime, use_recorded_settings
    output.mkdir(parents=True, exist_ok=False)
    reader = SourceSessionReader(session_dir)
    session = reader.session
    if not use_current_settings:
        settings = session.get("settings") or {}
        use_recorded_settings((settings.get("content/settings.json") or {}).get("content"),
                              (settings.get("content/ai/settings.json") or {}).get("content"),
                              Path(tempfile.mkdtemp(prefix="bench_replay_settings_")))
    cam_rows = reader.camera_index()
    audio_rows = reader.audio_index()
    if not cam_rows or not audio_rows:
        raise SystemExit("Session has no camera or audio data")
    first_wall = min(cam_rows[0]["acquisition"]["receive_wall_ts"], audio_rows[0]["block"]["receive_wall_ts"])
    first_mono = min(cam_rows[0]["acquisition"]["read_complete_monotonic"],
                     audio_rows[0]["block"]["read_complete_monotonic"])
    clock = VirtualClock(wall0=first_wall - 1.0, mono0=first_mono - 1.0)
    runtime = VirtualRuntime(clock=clock, worker_cost_s=worker_cost_s)
    # pulse_offset_s shifts synthetic shot sounds relative to the projector flip
    # (rig property: the projector shows a dot ~30-50 ms after flip, while a real
    # muzzle report reaches the mic ~18 ms before impact).
    synthetic = sorted((float(r["wall_ts"]) + float(pulse_offset_s), float(r["amplitude"]))
                       for r in _rows(session_dir / "synthetic_audio.jsonl"))
    truth = _rows(session_dir / "ground_truth.jsonl")

    events: list[tuple[float, str, Any]] = []
    for row in cam_rows:
        events.append((float(row["acquisition"]["receive_wall_ts"]) - clock.wall0, "camera", row))
    for row in audio_rows:
        events.append((float(row["block"]["receive_wall_ts"]) - clock.wall0, "audio", row))
    events.sort(key=lambda e: (e[0], e[1] != "audio"))
    end_t = events[-1][0] + tail_s

    with clock.installed():
        runtime.install()
        from src.engine.audio.audio_peak_detector import audio_peak_detector as audio
        from src.engine.camera.camera_manager import CameraFrame, camera_manager as cam
        from src.engine.camera.camera_manager import apply_frame_transform
        from src.engine.camera.hit_scanner import hit_scanner
        import cv2
        export = runtime.enable_candidate_export() if export_candidates else None
        runtime.start_scanner()
        seen_events: dict[int, Any] = {}
        cam_file = open(session_dir / "camera.bin", "rb")
        pcm_file = open(session_dir / "audio.pcm", "rb")
        step = runtime.step_interval_s
        next_step = 0.0
        index = 0
        frames = blocks = 0
        try:
            while next_step <= end_t:
                while index < len(events) and events[index][0] <= next_step:
                    t, kind, row = events[index]
                    clock.advance_to(t)
                    if kind == "camera":
                        cam_file.seek(int(row["offset"]))
                        data = cam_file.read(int(row["length"]))
                        acq = dict(row["acquisition"])
                        pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if pixels is not None:
                            pixels = apply_frame_transform(pixels, int(acq.get("rotation", 0) or 0),
                                                           bool(acq.get("mirror_horizontal")),
                                                           bool(acq.get("mirror_vertical")))
                            cam.inject_frame(CameraFrame(pixels, float(acq["receive_wall_ts"]), acq))
                            frames += 1
                    else:
                        meta = dict(row["block"])
                        pcm_file.seek(int(row["offset"]))
                        chunk = pcm_file.read(int(row["length"]))
                        if synthetic:
                            chunk = _mix_pulses(chunk, meta, synthetic)
                        receipt = {k: meta[k] for k in ("read_start_monotonic", "read_complete_monotonic",
                                                        "receive_wall_ts", "read_calls", "complete_block") if k in meta}
                        audio._deliver_pcm_block(chunk, receipt)
                        blocks += 1
                    index += 1
                clock.advance_to(next_step)
                runtime.step()
                for ev in list(hit_scanner.audio_events):
                    seen_events[int(ev.shot_id)] = (float(ev.peak_ts), ev)
                next_step += step
        finally:
            cam_file.close()
            pcm_file.close()
        emissions = [dict(camera_x=e.camera_x, camera_y=e.camera_y, wall_ts=e.wall) for e in runtime.emissions]
        audio_events = int(hit_scanner.audio_event_count)
        hit_scanner.disable()

    scored = score_truth(truth, emissions, tolerance_px) if truth else None
    if export is not None:
        rows = label_session_candidates(export, truth, seen_events, tolerance_px, pulse_offset_s,
                                        label_min_age_s, session_dir.name)
        with open(output / "candidates.jsonl", "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    result = dict(evidence_level=EVIDENCE_LEVEL, session=str(session_dir), frames=frames, pcm_blocks=blocks,
                  used_recorded_settings=not use_current_settings, pulse_offset_s=pulse_offset_s, audio_events=audio_events,
                  emissions=emissions, truth=truth, score=scored,
                  limitations=["virtual worker cost is a model, not measured CPU latency",
                               "AI shadow and trace writing disabled"])
    (output / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _mix_pulses(chunk: bytes, meta: dict[str, Any], pulses: list[tuple[float, float]]) -> bytes:
    sr = int(meta.get("sample_rate", 16000))
    samples = np.frombuffer(chunk, np.int16).astype(np.float32) / 32768.0
    end_wall = float(meta["receive_wall_ts"])
    start_wall = end_wall - samples.size / sr
    t = start_wall + np.arange(samples.size) / sr
    for wall, amplitude in pulses:
        if start_wall - 0.05 <= wall <= end_wall:
            dt = t - wall
            mask = dt >= 0
            samples[mask] += amplitude * np.exp(-dt[mask] / 0.004)
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()


def label_session_candidates(logged, truth, seen_events, tolerance_px, pulse_offset_s, min_age_s, name):
    """Out-of-band labels for a recorded session (truth never reaches the detector).

    Audio events map to the truth shot whose (offset) pulse time is within 50 ms.
    Candidates from frames received < min_age_s after the truth time are left
    UNLABELLED (exposure time is unknown), never guessed.
    """
    impacts = [t for t in truth if t.get("camera_x") is not None]
    event_truth = {}
    for sid, (peak, _ev) in seen_events.items():
        best = min(impacts, key=lambda t: abs(t["wall_ts"] + pulse_offset_s - peak), default=None)
        if best is not None and abs(best["wall_ts"] + pulse_offset_s - peak) <= 0.05:
            event_truth[sid] = best
    producer_of = {id(t): sid for sid, t in event_truth.items()}
    rows = [dict(kind="shot", scenario=f"session:{name}", seed=0, shot_index=i,
                 group=(f"{name}:{producer_of[id(t)]}" if id(t) in producer_of else f"{name}:noevent{i}"))
            for i, t in enumerate(impacts)]
    for entry in logged:
        c = entry["candidate"]
        producer = c.get("v2224_producer_shot_id")
        t = event_truth.get(int(producer)) if producer is not None else None
        if t is not None and entry["frame_ts"] < t["wall_ts"] + min_age_s:
            continue   # unknown exposure: do not label
        dist = None if t is None else math.hypot(float(c["camera_x"]) - t["camera_x"], float(c["camera_y"]) - t["camera_y"])
        rows.append(dict(kind="candidate", scenario=f"session:{name}", seed=0, group=f"{name}:{producer}",
                         producer_has_impact=t is not None, label=bool(dist is not None and dist <= tolerance_px),
                         distance_px=dist, candidate=c))
    return rows


def score_truth(truth: list[dict[str, Any]], emissions: list[dict[str, Any]], tolerance_px: float) -> dict[str, Any]:
    used: set[int] = set()
    rows = []
    for shot in sorted(truth, key=lambda r: r["wall_ts"]):
        if shot.get("kind") == "no_impact" or shot.get("camera_x") is None:
            hit = [j for j, e in enumerate(emissions) if j not in used and 0 <= e["wall_ts"] - shot["wall_ts"] <= 2.5]
            rows.append(dict(wall_ts=shot["wall_ts"], kind="no_impact", correct=not hit))
            used.update(hit)
            continue
        best, best_d = None, math.inf
        for j, e in enumerate(emissions):
            if j in used or e["wall_ts"] < shot["wall_ts"]:
                continue
            d = math.hypot(e["camera_x"] - shot["camera_x"], e["camera_y"] - shot["camera_y"])
            if d < best_d:
                best, best_d = j, d
        ok = best is not None and best_d <= tolerance_px
        if ok:
            used.add(best)
        rows.append(dict(wall_ts=shot["wall_ts"], kind="impact", correct=ok,
                         error_px=None if best is None else best_d))
    return dict(cases=len(rows), correct=sum(r["correct"] for r in rows),
                unmatched_emissions=len(emissions) - len(used), rows=rows)
