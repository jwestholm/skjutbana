"""Selftest for src/engine/precision_flash.py (policy, state machine, measurement).

  python3 -m automation.precision_flash_selftest
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from src.engine.precision_flash import (FlashConfig, PrecisionFlash, find_new_holes, resolve_policy,
                                        select_plateau)

RECT = (0, 0, 200, 120)


@dataclass
class _Frame:
    frame_bgr: np.ndarray
    timestamp: float


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _board(level, holes=(), partial_rows=None):
    img = np.full((120, 200), float(level), np.float32)
    # projector fall-off so the ratio has to normalise it
    img *= np.linspace(0.85, 1.0, 200)[None, :]
    for x, y in holes:
        img[y - 1:y + 2, x - 1:x + 2] *= 0.55
    if partial_rows is not None:
        img[partial_rows:] = 60.0
    noise = np.random.default_rng(int(level) + len(holes)).normal(0, 1.0, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


class _Camera:
    def __init__(self):
        self.frames = []

    def get_ring_snapshot(self):
        return list(self.frames)


class _Scene:
    def __init__(self, precision=None, flash="auto", mode=None):
        self.hit_precision, self.precision_flash, self.shot_input_mode = precision, flash, mode


class PolicyTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(resolve_policy(_Scene(), env={})[0], False)
        self.assertEqual(resolve_policy(_Scene("region"), env={})[:2], (False, "OBJECT"))
        self.assertEqual(resolve_policy(_Scene(mode="SHOT_ONLY"), env={})[1], "SHOT")
        self.assertEqual(resolve_policy(_Scene("exact"), env={})[:2], (True, "EXACT"))
        for level in ("SHOT", "HIT_MISS", "OBJECT", "ZONE"):
            self.assertFalse(resolve_policy(_Scene(level), env={})[0], level)

    def test_zone_series_enables_flash(self):
        scene = _Scene("zone")
        scene.hit_aggregation = "series"
        self.assertEqual(resolve_policy(scene, env={})[:2], (True, "ZONE"))

    def test_game_can_force_off_and_on(self):
        self.assertFalse(resolve_policy(_Scene("exact", "off"), env={})[0])
        self.assertTrue(resolve_policy(_Scene("region", "on"), env={})[0])
        self.assertFalse(resolve_policy(_Scene("exact", "off"), env={"SKJUTBANA_PRECISION_FLASH": "on"})[0])

    def test_env_global_off(self):
        self.assertFalse(resolve_policy(_Scene("exact"), env={"SKJUTBANA_PRECISION_FLASH": "off"})[0])

    def test_unknown_values_are_safe(self):
        enabled, precision, pref = resolve_policy(_Scene("ultra", "maybe"), env={})
        self.assertEqual((enabled, precision, pref), (False, "OBJECT", "auto"))


class MeasurementTests(unittest.TestCase):
    def test_new_hole_found_old_holes_ignored(self):
        cfg = FlashConfig(edge_margin_px=5)
        ref = _board(230, holes=[(50, 40)]).astype(np.float32)
        post = (_board(222, holes=[(50, 40), (120, 70)]).astype(np.float32))   # global drift too
        holes, _ = find_new_holes(ref, post, cfg)
        self.assertEqual(len(holes), 1)
        self.assertAlmostEqual(holes[0]["x"], 120, delta=1.0)
        self.assertAlmostEqual(holes[0]["y"], 70, delta=1.0)

    def test_no_change_no_hole(self):
        ref = _board(230).astype(np.float32)
        holes, _ = find_new_holes(ref, _board(230).astype(np.float32), FlashConfig(edge_margin_px=5))
        self.assertEqual(holes, [])

    def test_plateau_rejects_rolling_shutter_transition(self):
        frames = [_board(230), _board(230, partial_rows=60), _board(231)]
        idx, info = select_plateau(frames, RECT, baseline=100.0, cfg=FlashConfig())
        self.assertEqual(idx, [0, 2])
        idx, info = select_plateau([_board(102)], RECT, baseline=100.0, cfg=FlashConfig())
        self.assertEqual((idx, info["reason"]), ([], "FLASH_NOT_SEEN"))


class StateMachineTests(unittest.TestCase):
    def _controller(self):
        clock, cam, emitted, logs = _Clock(), _Camera(), [], []
        fl = PrecisionFlash(FlashConfig(edge_margin_px=5), clock=clock, camera=cam,
                            emit=lambda x, y: emitted.append((x, y)), scanport=RECT, log=logs.append)
        return fl, clock, cam, emitted

    def _run_flash(self, fl, clock, cam, holes, steps=200):
        """Advance until the flash has been measured; camera sees the flash frames."""
        for _ in range(steps):
            clock.t += 0.02
            was = fl.state
            fl.tick()
            if fl.state == fl.FLASHING or was == fl.FLASHING:
                cam.frames.append(_Frame(_board(230, holes), clock.t + 0.005))
            else:
                cam.frames.append(_Frame(_board(90, holes), clock.t + 0.005))
            if fl.state == fl.MEASURING and fl._worker is not None:
                fl._worker.join(2.0)
            if was == fl.MEASURING and fl.state != fl.MEASURING:
                return

    def test_disabled_does_nothing(self):
        fl, clock, cam, emitted = self._controller()
        fl.sync_policy(_Scene("region"))
        fl.on_peak(clock.t)
        self.assertFalse(fl.hold_ordinary_hit(10, 10))
        for _ in range(100):
            clock.t += 0.02
            fl.tick()
        self.assertEqual(fl.flash_count, 0)

    def test_reference_then_one_flash_per_burst(self):
        fl, clock, cam, emitted = self._controller()
        fl.sync_policy(_Scene("exact"))
        self._run_flash(fl, clock, cam, [])
        self.assertIsNotNone(fl.reference)
        self.assertEqual(fl.flash_count, 1)
        # double shot: two peaks 0.2 s apart -> ONE flash, two holes
        fl.on_peak(clock.t); clock.t += 0.2; fl.on_peak(clock.t)
        self.assertTrue(fl.hold_ordinary_hit(1, 2))            # ordinary hit withheld
        self._run_flash(fl, clock, cam, [(60, 40), (140, 80)])
        self.assertEqual(fl.flash_count, 2)
        self.assertEqual(len(emitted), 2)
        self.assertEqual(sorted((round(x), round(y)) for x, y in emitted), [(60, 40), (140, 80)])
        self.assertTrue(fl.hold_ordinary_hit(1, 2))            # late ordinary hit dropped (grace)

    def test_failed_reference_backs_off(self):
        fl, clock, cam, emitted = self._controller()
        fl.sync_policy(_Scene("exact"))
        for _ in range(int(10.0 / 0.02)):                      # camera never sees a flash
            clock.t += 0.02
            fl.tick()
            cam.frames.append(_Frame(_board(90), clock.t))
            if fl._worker is not None:
                fl._worker.join(2.0)
        self.assertEqual(fl.flash_count, 3)                   # max 3 attempts, then give up (no flicker)
        self.assertIsNone(fl.reference)

    def test_invalid_flash_releases_ordinary_hits(self):
        fl, clock, cam, emitted = self._controller()
        fl.sync_policy(_Scene("exact"))
        self._run_flash(fl, clock, cam, [])
        fl.on_peak(clock.t)
        self.assertTrue(fl.hold_ordinary_hit(33, 44))
        for _ in range(120):                                  # flash never seen by the camera
            clock.t += 0.02
            was = fl.state
            fl.tick()
            cam.frames.append(_Frame(_board(90), clock.t))
            if fl._worker is not None:
                fl._worker.join(2.0)
            if was == fl.MEASURING and fl.state == fl.IDLE:
                break
        self.assertEqual(emitted, [(33.0, 44.0)])

    def test_game_force_off_releases_and_stops(self):
        fl, clock, cam, emitted = self._controller()
        fl.sync_policy(_Scene("exact"))
        self._run_flash(fl, clock, cam, [])
        fl.on_peak(clock.t)
        fl.hold_ordinary_hit(5, 6)
        fl.sync_policy(_Scene("exact", "off"))
        self.assertEqual(emitted, [(5.0, 6.0)])
        self.assertFalse(fl.hold_ordinary_hit(7, 8))

    def test_partial_flash_result_releases_unmatched_ordinary_hit(self):
        fl, clock, cam, emitted = self._controller()
        fl.sync_policy(_Scene("exact"))
        self._run_flash(fl, clock, cam, [])
        fl.on_peak(clock.t); clock.t += 0.2; fl.on_peak(clock.t)
        fl.hold_ordinary_hit(61, 41)      # same shot as the flash hole -> replaced
        fl.hold_ordinary_hit(150, 20)     # hole the flash cannot see -> must be released
        self._run_flash(fl, clock, cam, [(60, 40)])
        self.assertEqual(len(emitted), 2)
        self.assertIn((150.0, 20.0), emitted)
        self.assertNotIn((61.0, 41.0), emitted)

    def test_peak_during_flash_gets_its_own_flash(self):
        fl, clock, cam, emitted = self._controller()
        fl.sync_policy(_Scene("exact"))
        self._run_flash(fl, clock, cam, [])
        fl.on_peak(clock.t)
        while fl.state != fl.FLASHING:
            clock.t += 0.02
            fl.tick()
            cam.frames.append(_Frame(_board(90, [(60, 40)]), clock.t))
        fl.on_peak(clock.t + 0.05)        # second shot lands during the flash
        self.assertIsNotNone(fl.next_burst)
        self._run_flash(fl, clock, cam, [(60, 40)])
        self.assertEqual(fl.state, fl.ARMED)                  # second burst waits for its own flash
        self._run_flash(fl, clock, cam, [(60, 40), (140, 80)])
        self.assertEqual(sorted((round(x), round(y)) for x, y in emitted), [(60, 40), (140, 80)])

    def test_flash_frames_hidden_from_ordinary_detection(self):
        fl, clock, cam, emitted = self._controller()
        self.assertFalse(fl.excludes_frame(clock.t))
        fl.sync_policy(_Scene("exact"))
        self._run_flash(fl, clock, cam, [])
        on, off = fl.flash_windows[-1]
        self.assertTrue(fl.excludes_frame(on + 0.1))
        self.assertTrue(fl.excludes_frame(off + 0.2))
        self.assertFalse(fl.excludes_frame(off + 0.5))
        self.assertFalse(fl.excludes_frame(on - 0.01))

    def test_double_shot_holes_get_distinct_shot_ids_and_peak_times(self):
        from src.engine.shot_context_v250 import current_shot_context_v250
        fl, clock, cam, emitted = self._controller()
        contexts = []
        fl.emit = lambda x, y: contexts.append(current_shot_context_v250())
        fl.sync_policy(_Scene("exact"))
        self._run_flash(fl, clock, cam, [])
        p1 = clock.t; fl.on_peak(p1); clock.t += 0.2; p2 = clock.t; fl.on_peak(p2)
        self._run_flash(fl, clock, cam, [(60, 40), (140, 80)])
        self.assertEqual(len(contexts), 2)
        self.assertNotEqual(contexts[0].shot_id, contexts[1].shot_id)
        self.assertEqual(sorted(c.peak_ts for c in contexts), [p1, p2])
        self.assertIsNone(current_shot_context_v250())       # context restored

    def test_series_flash_measures_whole_series_once(self):
        fl, clock, cam, emitted = self._controller()
        scene = _Scene("ZONE"); scene.hit_aggregation = "series"
        results = []
        scene.on_precision_series = results.append
        fl.sync_policy(scene)
        self._run_flash(fl, clock, cam, [])                    # reference
        for _ in range(3):
            fl.on_peak(clock.t); clock.t += 0.6
            for _ in range(40):                                 # no per-burst flash in series mode
                clock.t += 0.02; fl.tick()
        self.assertEqual(fl.flash_count, 1)
        self.assertFalse(fl.hold_ordinary_hit(1, 1))           # ordinary hits stay live
        self.assertTrue(fl.flush_series())
        self._run_flash(fl, clock, cam, [(40, 30), (100, 60), (160, 90)])
        self.assertEqual(fl.flash_count, 2)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]["holes"]), 3)
        self.assertEqual(emitted, [])                          # series result is not re-emitted as hits

    def test_duplicate_peak_registration_is_ignored(self):
        fl, clock, cam, emitted = self._controller()
        fl.sync_policy(_Scene("exact"))
        fl.on_peak(5.0); fl.on_peak(5.0)
        self.assertEqual(fl.burst.peaks, [5.0])


if __name__ == "__main__":
    unittest.main(verbosity=1)
