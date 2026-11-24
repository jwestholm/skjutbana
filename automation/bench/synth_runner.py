"""Run AR-15 synthetic scenarios through the installed LIVE shot path.

Runs in a dedicated process (virtual clock patches ``time``). Each scenario gets
a fresh scanner/audio/worker state; the virtual clock only moves forward.
Results are evidence level "synthetic live-path": real detector/scheduler code,
synthetic sources. Never a physical accuracy claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time as _real_time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.engine.offline.ar15_scenarios import (SCENARIOS, FrameRenderer, PhysicsAssumptions, ViewportGeometry,
                                              generate, load_settings, synth_pcm)
from src.engine.offline.virtual_runtime import VirtualRuntime, make_pcm_receipt

EVIDENCE_LEVEL = "SYNTHETIC_LIVE_PATH"


def _reset_runtime_state() -> None:
    from src.engine.audio.audio_peak_detector import audio_peak_detector as audio
    from src.engine.camera.camera_manager import camera_manager as cam
    from src.engine.camera.hit_scanner import hit_scanner
    from src.engine.shot_async_v2224 import async_detector_v2224 as worker
    from src.engine.shot_critical_v2223 import shot_critical_controller_v2223 as controller
    from src.engine.shot_fast_v2225 import rescue_router_v2225
    hit_scanner.disable()
    hit_scanner.reset_hole_map()
    hit_scanner.scene_reference_gray = None
    hit_scanner.surface_reference_gray = None
    with cam._lock:
        cam._ring.clear()
        cam._read_count = 0
        cam._last_pickup_count = 0
    cam.latest_frame = None
    with audio._lock:
        audio._pending_dispatch.clear()
        audio._events.clear()
    audio.last_peak_ts = 0.0
    audio.noise_floor = 0.01
    with worker._lock:
        worker._futures.clear()
        worker._ready.clear()
    worker._submitted_frame_ts.clear()
    worker._last_applied_frame_ts.clear()
    controller.last_seen_peak_ts = 0.0
    try:
        rescue_router_v2225.reset()
    except Exception:
        pass


def _camera_frame(runtime: VirtualRuntime, pixels: np.ndarray, sequence: int, exposure_end_mono: float):
    from src.engine.camera.camera_manager import CameraFrame
    clock = runtime.clock
    meta = dict(stream_id="synthetic", sequence=sequence, camera_index=-1, backend_name="SYNTHETIC",
                read_start_monotonic=clock.mono() - 0.02, read_complete_monotonic=clock.mono(),
                receive_wall_ts=clock.wall(), capture_kind="synthetic", decode_mode="synthetic",
                backend_timestamps_raw=dict(CAP_PROP_POS_MSEC=exposure_end_mono * 1000.0),
                backend_timestamp_semantics="SYNTHETIC_EXPOSURE_END_MONOTONIC",
                exposure_monotonic=exposure_end_mono, rotation=0, mirror_horizontal=False,
                mirror_vertical=False, frame_shape=list(pixels.shape), published_monotonic=clock.mono())
    return CameraFrame(pixels, clock.wall(), meta)


def run_scenario(runtime: VirtualRuntime, name: str, seed: int, base_gray: np.ndarray,
                 geometry: ViewportGeometry, phys: PhysicsAssumptions) -> dict[str, Any]:
    from src.engine.audio.audio_peak_detector import audio_peak_detector as audio
    from src.engine.camera.camera_manager import camera_manager as cam
    from src.engine.camera.hit_scanner import hit_scanner
    _reset_runtime_state()
    scenario = generate(name, seed, geometry, phys)
    pcm = synth_pcm(scenario, phys, seed)
    renderer = FrameRenderer(base_gray, scenario, geometry, phys, seed)
    rng = np.random.default_rng(seed + 31337)
    clock = runtime.clock
    T0 = clock.now() + 0.001

    # Timelines (scenario-relative seconds).
    period = 1.0 / phys.fps
    frames = []
    k = 1
    while k * period < scenario.duration_s:
        exp_end = k * period
        recv = exp_end + phys.camera_receive_latency_s + float(rng.normal(0, phys.camera_latency_jitter_s))
        frames.append((max(recv, exp_end), exp_end, k))
        k += 1
    blocks = []
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    n_blocks = len(pcm_i16) // phys.chunk_samples
    for i in range(n_blocks):
        end_t = (i + 1) * phys.chunk_samples / phys.sample_rate
        blocks.append((end_t + float(rng.uniform(*phys.pcm_delivery_latency_s)), i))
    # parec delivers in order: enforce monotonic receipt.
    last = 0.0
    for j, (t, i) in enumerate(blocks):
        last = max(last, t)
        blocks[j] = (last, i)

    runtime.start_scanner()
    seen_events: dict[int, Any] = {}
    emissions_before = len(runtime.emissions)
    fi = bi = 0
    t = 0.0
    step = runtime.step_interval_s
    while t <= scenario.duration_s:
        # Deliver sources that arrived up to t, each at its own virtual time.
        while True:
            next_audio = blocks[bi][0] if bi < len(blocks) else math.inf
            next_frame = frames[fi][0] if fi < len(frames) else math.inf
            nxt = min(next_audio, next_frame)
            if nxt > t:
                break
            clock.advance_to(T0 + nxt)
            if next_audio <= next_frame:
                _, i = blocks[bi]
                chunk = pcm_i16[i * phys.chunk_samples:(i + 1) * phys.chunk_samples].tobytes()
                audio._deliver_pcm_block(chunk, make_pcm_receipt(clock, phys.chunk_samples, phys.sample_rate, 0.0))
                bi += 1
            else:
                recv, exp_end, seq = frames[fi]
                pixels = renderer.render(exp_end)
                cam.inject_frame(_camera_frame(runtime, pixels, seq, clock.mono0 + T0 + exp_end))
                fi += 1
        clock.advance_to(T0 + t)
        runtime.step()
        for ev in list(hit_scanner.audio_events):   # events are pruned later; keep every one seen
            seen_events[int(ev.shot_id)] = ev
        t += step

    emitted = [dict(camera_x=e.camera_x, camera_y=e.camera_y, t=e.virtual_t - T0)
               for e in runtime.emissions[emissions_before:]]
    events = [dict(shot_id=int(ev.shot_id), peak_t=float(ev.peak_ts) - (clock.wall0 + T0),
                   state=str(ev.state), lifecycle=str(getattr(ev, "lifecycle_state", "")),
                   note=str(getattr(ev, "note", "") or ""))
              for ev in sorted(seen_events.values(), key=lambda e: e.shot_id)]
    audio_event_count = int(hit_scanner.audio_event_count)
    hit_scanner.disable()
    return dict(scenario=scenario.to_dict(), emissions=emitted, audio_events=events,
                audio_event_count=audio_event_count,
                worker_jobs=int(runtime.executor.executed))


def label_candidates(logged: list[dict[str, Any]], result: dict[str, Any], geometry: ViewportGeometry,
                     tolerance_mm: float, clock) -> list[dict[str, Any]]:
    """Label exported candidates with the scenario truth (never fed to the detector).

    Each audio event is mapped to the impact shot whose muzzle report it
    corresponds to (|peak - arrival| <= 50 ms); echoes/false audio map to none.
    A candidate is positive iff it lies within tolerance of ITS producer shot's
    hole and its frame was exposed after that impact.
    """
    scenario = result["scenario"]
    impacts = [s for s in scenario["shots"] if s["kind"] == "impact"]
    muzzle = sorted(t for t, _ in scenario["audio_pulses"])
    event_shot: dict[int, dict[str, Any]] = {}
    for ev in result["audio_events"]:
        best = None
        for shot in impacts:
            arrivals = [t for t in muzzle if shot["fire_t"] <= t <= shot["impact_t"] + 0.01]
            if arrivals and abs(ev["peak_t"] - arrivals[0]) <= 0.05:
                best = shot
        if best is not None:
            event_shot[int(ev["shot_id"])] = best
    scale = geometry.px_per_mm(0.5, 0.5)
    tol_px = tolerance_mm * scale
    rows = []
    # One row per impact shot (incl. shots with no audio event / no candidates),
    # so evaluation counts no-proposal shots as misses.
    producer_of = {id(shot): sid for sid, shot in event_shot.items()}
    shot_rows = []
    for index, shot in enumerate(impacts):
        sid = producer_of.get(id(shot))
        shot_rows.append(dict(kind="shot", scenario=scenario["name"], seed=scenario["seed"], shot_index=index,
                              group=(f"{scenario['name']}:{scenario['seed']}:{sid}" if sid is not None
                                     else f"{scenario['name']}:{scenario['seed']}:noevent{index}")))
    for entry in logged:
        c = entry["candidate"]
        producer = c.get("v2224_producer_shot_id")
        shot = event_shot.get(int(producer)) if producer is not None else None
        dist = None
        positive = False
        if shot is not None and c.get("camera_x") is not None:
            dist = math.hypot(float(c["camera_x"]) - shot["camera_x"], float(c["camera_y"]) - shot["camera_y"])
            positive = dist <= tol_px
        rows.append(dict(kind="candidate", scenario=scenario["name"], seed=scenario["seed"],
                         group=f"{scenario['name']}:{scenario['seed']}:{producer}",
                         producer_has_impact=shot is not None, label=bool(positive),
                         distance_px=dist, candidate=c))
    return shot_rows + rows


def score_scenario(result: dict[str, Any], geometry: ViewportGeometry, tolerance_mm: float) -> list[dict[str, Any]]:
    """Per-case outcomes. Impact shots: correct iff a unique emission lands within
    tolerance after the shot fired. No-impact scenarios: correct iff nothing emitted."""
    scenario = result["scenario"]
    emissions = list(result["emissions"])
    used: set[int] = set()
    cases = []
    impacts = [s for s in scenario["shots"] if s["kind"] == "impact"]
    key_base = f"{scenario['name']}:{scenario['seed']}"
    for index, shot in enumerate(impacts):
        scale = geometry.px_per_mm(0.5, 0.5)
        tol_px = tolerance_mm * scale
        best, best_d = None, math.inf
        for j, e in enumerate(emissions):
            if j in used or e["t"] < shot["fire_t"]:
                continue
            d = math.hypot(e["camera_x"] - shot["camera_x"], e["camera_y"] - shot["camera_y"])
            if d < best_d:
                best, best_d = j, d
        correct = best is not None and best_d <= tol_px
        if correct:
            used.add(best)
        cases.append(dict(key=f"{key_base}:{index}", scenario=scenario["name"], kind="impact",
                          correct=bool(correct), error_px=None if best is None else float(best_d),
                          error_mm=None if best is None else float(best_d / scale),
                          latency_s=None if not correct else float(emissions[best]["t"] - shot["impact_t"])))
    extra = [e for j, e in enumerate(emissions) if j not in used]
    if not impacts:
        cases.append(dict(key=f"{key_base}:none", scenario=scenario["name"], kind="no_impact",
                          correct=not emissions, false_emissions=len(emissions)))
    elif extra:
        cases.append(dict(key=f"{key_base}:extra", scenario=scenario["name"], kind="false_emission",
                          correct=False, false_emissions=len(extra)))
    else:
        cases.append(dict(key=f"{key_base}:extra", scenario=scenario["name"], kind="false_emission",
                          correct=True, false_emissions=0))
    return cases


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except Exception:
        return None


def _load_one_background(spec: str) -> np.ndarray:
    import cv2
    p = Path(spec)
    if p.is_dir():   # a recorded source session: use its first decoded camera frame
        from src.engine.source_recording import SourceSessionReader
        for _row, pixels in SourceSessionReader(p).iter_camera_frames():
            if pixels is not None:
                return cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"No decodable frame in session {p}")
    img = np.load(p) if p.suffix == ".npy" else cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read background {p}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def load_backgrounds(spec: str | None) -> tuple[list[np.ndarray], list[str]]:
    """Comma-separated 4K camera frames (.npy/.png/.jpg) and/or source-session dirs."""
    if not spec:
        raise ValueError("--background is required (4K camera frame(s) or source session dir(s))")
    paths = [s.strip() for s in spec.split(",") if s.strip()]
    return [_load_one_background(p) for p in paths], paths


def load_base_gray(path: str | None) -> tuple[np.ndarray, str]:
    images, paths = load_backgrounds(path)
    return images[0], paths[0]


def run(args) -> Path:
    out = Path(args.output)
    # never overwrite: the dir may only pre-exist holding the disk-guard run marker
    from automation.disk_guard import assert_output_directory_available
    assert_output_directory_available(out)
    out.mkdir(parents=True, exist_ok=True)
    settings = load_settings(args.settings)
    geometry = ViewportGeometry(settings)
    phys = PhysicsAssumptions(**json.loads(args.physics)) if args.physics else PhysicsAssumptions()
    backgrounds, background_sources = load_backgrounds(args.background)
    scenarios = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    manifest = dict(evidence_level=EVIDENCE_LEVEL, variant=args.variant, created=_real_time.time(),
                    git=dict(commit=_git("rev-parse", "HEAD"), dirty=bool(_git("status", "--porcelain"))),
                    env={k: v for k, v in os.environ.items() if k.startswith("SKJUTBANA_")},
                    python=sys.version, platform=platform.platform(), physics=asdict(phys),
                    tolerance_mm=float(args.tolerance_mm), scenarios=scenarios, seeds=seeds,
                    background=background_sources,
                    background_sha256=hashlib.sha256(b"".join(hashlib.sha256(b.tobytes()).digest() for b in backgrounds)).hexdigest(),
                    ai_settings_path="content/ai/settings.json",
                    ai_settings_sha256=(hashlib.sha256(Path("content/ai/settings.json").read_bytes()).hexdigest()
                                        if Path("content/ai/settings.json").exists() else None),
                    settings_path=str(args.settings),
                    settings_sha256=hashlib.sha256(Path(args.settings).read_bytes()).hexdigest(),
                    limitations=["synthetic sources: hole renderer and audio pulses are models, not physics",
                                 "virtual worker cost is a fixed model, not measured CPU latency",
                                 "AI shadow worker and physical trace writing disabled"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    runtime = VirtualRuntime(worker_cost_s=float(args.worker_cost_s))
    cases: list[dict[str, Any]] = []
    with runtime.clock.installed():
        runtime.install()
        export = runtime.enable_candidate_export() if getattr(args, "export_candidates", False) else None
        for name in scenarios:
            for seed in seeds:
                base = backgrounds[seed % len(backgrounds)]
                mark = len(export) if export is not None else 0
                result = run_scenario(runtime, name, seed, base, geometry, phys)
                if export is not None:
                    rows = label_candidates(export[mark:], result, geometry, float(args.tolerance_mm), runtime.clock)
                    with open(out / "candidates.jsonl", "a") as handle:
                        for row in rows:
                            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                scored = score_scenario(result, geometry, float(args.tolerance_mm))
                cases.extend(scored)
                with open(out / "scenarios.jsonl", "a") as handle:
                    handle.write(json.dumps(dict(result=result, cases=scored)) + "\n")
                ok = sum(c["correct"] for c in scored)
                print(f"[bench] {name:12s} seed={seed:<5d} cases={len(scored)} correct={ok} "
                      f"emissions={len(result['emissions'])} audio_events={result['audio_event_count']}", flush=True)
    summary = summarize(cases)
    (out / "results.json").write_text(json.dumps(dict(manifest=manifest, summary=summary, cases=cases), indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return out


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    from src.engine.offline.paired_stats import wilson_interval
    out: dict[str, Any] = {}
    for group in sorted({c["scenario"] for c in cases}) + ["ALL"]:
        rows = [c for c in cases if group == "ALL" or c["scenario"] == group]
        impacts = [c for c in rows if c["kind"] == "impact"]
        ok = sum(c["correct"] for c in rows)
        hits = sum(c["correct"] for c in impacts)
        errors = sorted(c["error_mm"] for c in impacts if c["correct"] and c["error_mm"] is not None)
        lat = sorted(c["latency_s"] for c in impacts if c["correct"] and c["latency_s"] is not None)
        out[group] = dict(cases=len(rows), correct=ok, rate=ok / len(rows) if rows else None,
                          rate_ci95=wilson_interval(ok, len(rows)),
                          impacts=len(impacts), impact_hits=hits,
                          false_emission_cases=sum(1 for c in rows if c["kind"] != "impact" and not c["correct"]),
                          median_error_mm=errors[len(errors) // 2] if errors else None,
                          median_latency_s=lat[len(lat) // 2] if lat else None)
    return out
