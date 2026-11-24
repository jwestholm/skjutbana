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
import threading
from time import monotonic_ns
from pathlib import Path
from typing import Any

import numpy as np

EVIDENCE_LEVEL = "RECORDED_SOURCE_LIVE_PATH"


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def profile_executor(executor) -> list[dict[str, Any]]:
    """Observe this replay executor while preserving its schedule and futures."""
    timings = []
    original_submit = executor.submit

    def submit(fn, /, *args, **kwargs):
        row = dict(submit_ns=monotonic_ns(), start_ns=None, finish_ns=None, thread=None)

        def timed_job():
            row["thread"] = threading.current_thread().name
            row["start_ns"] = monotonic_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                row["finish_ns"] = monotonic_ns()

        future = original_submit(timed_job)
        timings.append(row)
        return future

    executor.submit = submit
    return timings


def replay(session_dir: Path, output: Path, *, use_current_settings: bool = False,
           worker_cost_s: float = 0.15, tolerance_px: float = 8.0, tail_s: float = 4.5,
           pulse_offset_s: float = 0.0, export_candidates: bool = False,
           label_min_age_s: float = 0.15, wall_clock_profile: bool = False) -> dict[str, Any]:
    from src.engine.source_recording import SourceSessionReader
    from src.engine.offline.virtual_runtime import VirtualClock, VirtualRuntime, use_recorded_settings
    # never overwrite: the dir may only pre-exist holding the disk-guard run marker
    from automation.disk_guard import assert_output_directory_available
    assert_output_directory_available(output)
    output.mkdir(parents=True, exist_ok=True)
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
    _install_region_hint(truth)

    events: list[tuple[float, str, Any]] = []
    for row in cam_rows:
        events.append((float(row["acquisition"]["receive_wall_ts"]) - clock.wall0, "camera", row))
    for row in audio_rows:
        events.append((float(row["block"]["receive_wall_ts"]) - clock.wall0, "audio", row))
    events.sort(key=lambda e: (e[0], e[1] != "audio"))
    end_t = events[-1][0] + tail_s

    with clock.installed():
        runtime.install()
        host_timings = profile_executor(runtime.executor) if wall_clock_profile else None
        from src.engine.audio.audio_peak_detector import audio_peak_detector as audio
        from src.engine.camera.camera_manager import CameraFrame, camera_manager as cam
        from src.engine.camera.camera_manager import apply_frame_transform
        from src.engine.camera.hit_scanner import hit_scanner
        import cv2
        export = runtime.enable_candidate_export() if export_candidates else None
        runtime.start_scanner()
        seen_events: dict[int, Any] = {}
        from src.engine.camera_bin_store import open_camera_bin
        cam_file = open_camera_bin(session_dir)
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
    if host_timings is not None:
        profile = dict(version=1, clock="host_monotonic_ns", jobs=host_timings)
        (output / "wall_clock_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
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
    # CO2 shots give two audio peaks 66-94 ms apart.  Truth sits on the first peak while the
    # detector work (and its candidates) may be owned by the second one (causal ownership), so
    # an unmatched event within CO2_PAIR_S of a matched one inherits its truth, and the shot
    # group uses the event that actually produced candidates (2026-11-19: physical 04f1be81 and
    # taped da06f87a had 0 labelled candidates otherwise).
    CO2_PAIR_S = 0.12
    matched = sorted((seen_events[s][0], s) for s in event_truth)
    for sid, (peak, _ev) in seen_events.items():
        if sid in event_truth:
            continue
        near = [(abs(mpeak - peak), msid) for mpeak, msid in matched if abs(mpeak - peak) <= CO2_PAIR_S]
        if near:
            event_truth[sid] = event_truth[min(near)[1]]
    producers = {e["candidate"].get("v2224_producer_shot_id") for e in logged}
    producer_of = {}
    for sid, t in sorted(event_truth.items(), key=lambda kv: (kv[0] in producers, -seen_events[kv[0]][0])):
        producer_of[id(t)] = sid   # last write wins: a producing event, else the earliest
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


def _install_region_hint(truth) -> None:
    """SKJUTBANA_REPLAY_REGION_HINT_PX=<screen px radius>: a game target around an aim point drawn
    within radius/2 of each GT hole (seeded by SKJUTBANA_REPLAY_REGION_SEED, default 7), visible from
    2 s before to 2 s after the shot — the OBJECT-path experiment (measurement only)."""
    import os as _os
    import random as _random
    from src.engine.offline import virtual_runtime as _vr
    _vr.REGION_TARGETS.clear()
    try:
        radius = float(_os.environ.get("SKJUTBANA_REPLAY_REGION_HINT_PX", "0") or 0)
    except ValueError:
        radius = 0.0
    if radius <= 0:
        return
    from src.engine.input.hit_input import hit_input
    from src.engine.settings import load_viewport_rect
    vp = load_viewport_rect()
    rng = _random.Random(int(_os.environ.get("SKJUTBANA_REPLAY_REGION_SEED", "7")))
    for i, t in enumerate(sorted((r for r in truth if r.get("camera_x") is not None), key=lambda r: r["wall_ts"])):
        sx, sy = hit_input.camera_to_screen(float(t["camera_x"]), float(t["camera_y"]))
        ang, dist = rng.uniform(0, 6.283185307), rng.uniform(0, radius / 2.0)
        _vr.REGION_TARGETS.append(dict(id=f"replay_{i}", game_x=sx - vp.x + dist * __import__("math").cos(ang),
                                       game_y=sy - vp.y + dist * __import__("math").sin(ang), radius=radius,
                                       t0=float(t["wall_ts"]) - 2.0, t1=float(t["wall_ts"]) + 2.0))
    print(f"[REPLAY] region hint: {len(_vr.REGION_TARGETS)} targets, radius {radius} screen px", flush=True)


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
