"""Deterministic checks for the offline bench building blocks (no hardware)."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.offline.ar15_scenarios import (FrameRenderer, PhysicsAssumptions, ViewportGeometry, generate,
                                              synth_pcm)
from src.engine.offline.paired_stats import exact_sign_test_p, paired_compare, wilson_interval
from src.engine.offline.virtual_runtime import VirtualClock, VirtualExecutor


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


SETTINGS = {
    "camera_calibration": {"inverse_homography": [[2.0, 0, 1500], [0, 1.2, 1000], [0, 0, 1]]},
    "viewport": [20, 10, 760, 580],
    "range_projection": {"viewport_physical_width_cm": 90.0, "viewport_physical_height_cm": 70.0},
}


def test_paired_stats():
    check(abs(exact_sign_test_p(0, 6) - 0.03125) < 1e-12, "exact sign test p for 0/6 discordant")
    check(exact_sign_test_p(0, 0) == 1.0, "no discordant pairs gives p=1")
    base = {f"k{i}": False for i in range(20)}
    cand = {f"k{i}": True for i in range(20)}
    check(paired_compare(base, cand)["verdict"] == "BETTER", "all fixed is BETTER")
    check(paired_compare(cand, base)["verdict"] == "WORSE", "all broken is WORSE")
    same = paired_compare(base, base)
    check(same["verdict"] == "NO_SIGNIFICANT_DIFFERENCE" and same["p_value"] == 1.0, "identical arms show no difference")
    try:
        paired_compare({"a": True}, {"b": True})
        raise AssertionError("mismatched cases accepted")
    except ValueError:
        check(True, "different case sets are refused")
    lo, hi = wilson_interval(19, 20)
    check(0.7 < lo < 0.95 < hi <= 1.0, "Wilson interval brackets 95% for 19/20")


def test_virtual_clock_and_executor():
    real = time.time
    clock = VirtualClock(wall0=1000.0, mono0=50.0)
    with clock.installed():
        clock.advance_to(2.0)
        check(time.time() == 1002.0 and time.monotonic() == 52.0, "virtual wall/monotonic domains are separate")
        ex = VirtualExecutor(clock, cost_s=0.15)
        seen = []
        f1 = ex.submit(lambda: seen.append(("a", time.monotonic())) or "A")
        f2 = ex.submit(lambda: seen.append(("b", time.monotonic())) or "B")
        check(not f1.done() and not f2.done(), "jobs are not done before modelled cost")
        clock.advance_to(2.15)
        ex.run_due()
        check(f1.done() and f1.result() == "A" and not f2.done(), "single worker finishes jobs FIFO")
        clock.advance_to(2.30)
        ex.run_due()
        check(f2.result() == "B", "second job finishes one cost later")
        check(seen == [("a", 52.0), ("b", 52.15)], "job code observes the clock frozen at its start")
        try:
            clock.advance_to(1.0)
            raise AssertionError("time moved backwards")
        except ValueError:
            check(True, "virtual time cannot move backwards")
    check(time.time is real, "real time functions are restored")


def test_scenarios_and_renderer():
    geometry = ViewportGeometry(SETTINGS)
    phys = PhysicsAssumptions(jpeg_quality=None, sensor_noise_sigma=0.0)
    a = generate("double", 7, geometry, phys)
    b = generate("double", 7, geometry, phys)
    check(a.to_dict() == b.to_dict(), "scenario generation is deterministic per seed")
    check(len(a.shots) == 2 and all(s.impact_t > s.fire_t for s in a.shots), "impact follows firing")
    gap = a.shots[1].fire_t - a.shots[0].fire_t
    check(phys.double_interval_s[0] <= gap <= phys.double_interval_s[1], "double-shot interval within model")
    arrivals = sorted(t for t, _ in a.audio_pulses)
    check(arrivals[0] < a.shots[0].impact_t, "muzzle sound reaches the mic before the bullet hits")
    pcm = synth_pcm(a, phys, 7)
    check(np.abs(pcm).max() > 0.5 and abs(len(pcm) / phys.sample_rate - a.duration_s) < 1e-3, "PCM contains the pulses")
    base = np.full((600, 900), 150, np.uint8)
    single = generate("single", 3, ViewportGeometry({**SETTINGS, "camera_calibration": {
        "inverse_homography": [[0.9, 0, 30], [0, 0.8, 30], [0, 0, 1]]}}), phys)
    shot = single.shots[0]
    renderer = FrameRenderer(base, single, ViewportGeometry({**SETTINGS, "camera_calibration": {
        "inverse_homography": [[0.9, 0, 30], [0, 0.8, 30], [0, 0, 1]]}}), phys, 3)
    x, y = int(round(shot.camera_x)), int(round(shot.camera_y))
    before = renderer.render(shot.impact_t - 1e-3)[..., 0].astype(int)
    after = renderer.render(shot.impact_t + phys.exposure_s + 1e-3)[..., 0].astype(int)
    partial = renderer.render(shot.impact_t + phys.exposure_s / 2)[..., 0].astype(int)
    check(before[y, x] == 150, "hole is absent when exposure ended before impact")
    check(after[y, x] < 150 - 10, "hole is visible after a full exposure")
    check(after[y, x] < partial[y, x] < 150, "impact during exposure gives a partial hole")


def test_compare_refuses_different_experiments():
    from automation.bench.__main__ import compare
    with tempfile.TemporaryDirectory() as tmp:
        for name, settings_hash in (("a", "x"), ("b", "y")):
            d = Path(tmp) / name
            d.mkdir()
            manifest = dict(evidence_level="SYNTHETIC_LIVE_PATH", physics={}, tolerance_mm=5, background_sha256="z",
                            seeds=[1], scenarios=["single"], settings_sha256=settings_hash)
            (d / "results.json").write_text(json.dumps(dict(manifest=manifest, cases=[
                dict(key="single:1:0", scenario="single", correct=True)])))
        try:
            compare(Path(tmp) / "a", Path(tmp) / "b", quiet=True)
            raise AssertionError("compared different settings")
        except SystemExit:
            check(True, "compare refuses runs with different calibration/settings")


def main():
    test_paired_stats()
    test_virtual_clock_and_executor()
    test_scenarios_and_renderer()
    test_compare_refuses_different_experiments()
    print("ALL BENCH SELFTESTS PASSED")


if __name__ == "__main__":
    main()
