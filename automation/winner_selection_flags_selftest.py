"""Unit checks for the opt-in winner-selection flags (H6/H7a/H7b/H8a/H8b/H9a/H9b).

Never opens camera, microphone or a game window. Run:
    python3 -m automation.winner_selection_flags_selftest
"""
from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib

from src.engine import winner_selection_flags as wsf
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
from src.engine.camera.hit_scanner import AudioShotEvent, HitScanner, HoleTrack
from src.engine.shot_track_v2226 import update_tracks_frame_unique_v2226

HS_MODULE = importlib.import_module("src.engine.camera.hit_scanner")   # module, not the singleton


def capture_push(pushed):
    return patch.object(HS_MODULE.hit_input, "push_camera_hit", lambda x, y: pushed.append((x, y)))


FLAGS = ("SKJUTBANA_EVIDENCE_OVER_TIMING", "SKJUTBANA_BEST_READY_TOPK", "SKJUTBANA_UNCAPPED_RANK",
         "SKJUTBANA_HOLE_SHAPE_PRIOR", "SKJUTBANA_SIGNED_DARKENING_PRIOR", "SKJUTBANA_NO_OLD_HOLE_SNAP",
         "SKJUTBANA_SOFT_KNOWN_HOLE_PENALTY", "SKJUTBANA_HOLE_RADIUS_PX")


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in FLAGS}
    env.update(extra)
    return patch.dict(os.environ, env, clear=True)


def reference_best(scanner, event):
    """Verbatim copy of the pre-flag default selector key."""
    best, best_metric = None, None
    for track in scanner._active_tracks.values():
        producer_sid = (getattr(track, "last_candidate", {}) or {}).get("v2224_producer_shot_id")
        if producer_sid is not None and int(producer_sid) != int(event.shot_id):
            continue
        onset_dt = track.first_seen_ts - event.peak_ts
        if onset_dt < -scanner.association_lead_s or onset_dt > scanner.association_lag_s:
            continue
        if track.emitted and event.matched_track_id != track.track_id:
            continue
        metric = (abs(onset_dt), -track.best_score)
        if best is None or metric < best_metric:
            best, best_metric = track, metric
    return best


def track(tid, first, score, *, hits=1, last=None, sid=1, emitted=False, x=100.0, y=100.0, **cand):
    c = {"v2224_producer_shot_id": sid, "score": score}
    c.update(cand)
    return HoleTrack(track_id=tid, camera_x=x, camera_y=y, created_at=first, first_seen_ts=first,
                     last_seen_ts=first if last is None else last, hits=hits, best_score=score,
                     emitted=emitted, last_candidate=c)


def scanner_with(*tracks):
    s = HitScanner()
    s._active_tracks = {t.track_id: t for t in tracks}
    return s


class DefaultsUnchanged(unittest.TestCase):
    def test_selector_identical_on_random_pools(self):
        rng = random.Random(7)
        with clean_env():
            for _ in range(400):
                peak = 10.0
                tracks = []
                for tid in range(1, rng.randint(1, 7)):
                    first = peak + rng.choice([-0.2, -0.05, 0.0, 0.05, 0.1, 0.1, 0.4, 2.0])
                    tracks.append(track(tid, first, rng.choice([5.0, 12.0, 35.0, 36.5]),
                                        hits=rng.randint(1, 4), sid=rng.choice([1, 1, 2, None]),
                                        emitted=rng.random() < 0.15))
                s = scanner_with(*tracks)
                ev = AudioShotEvent(1, peak, peak)
                ev.matched_track_id = rng.choice([None, 1, 2])
                expected = reference_best(s, ev)
                got = s._best_track_for_event(ev)
                self.assertIs(got, expected)
                self.assertNotIn("winner_selection_flags", ev.audio_trigger)

    def test_no_new_fields_when_off(self):
        with clean_env():
            s = HitScanner()
            update_tracks_frame_unique_v2226(s, [{"camera_x": 10.0, "camera_y": 10.0, "score": 20.0,
                                                  "v26_rank_excess": 9.0}], 1.0)
            t = next(iter(s._active_tracks.values()))
            self.assertFalse(hasattr(t, "rank_value"))
            feats = features()
            self.assertNotIn("rank_raw", feats)
            self.assertEqual(feats["score"], 35.0)

    def test_known_hole_penalty_default_factors(self):
        with clean_env():
            for dist, factor in ((5.0, 0.15), (12.0, 0.4)):
                c = {"camera_x": 0.0, "camera_y": 0.0, "score": 10.0}
                CandidateGeneratorV2._apply_known_hole_penalty(FakeHoleScanner(dist), c)
                self.assertEqual(c["score"], 10.0 * factor)
                self.assertEqual(set(c), {"camera_x", "camera_y", "score", "near_known_hole_dist"})

    def test_emit_snaps_to_old_hole_by_default(self):
        with clean_env():
            xy, ev = emit_near_old_hole()
            self.assertEqual(xy, (100.0, 100.0))
            self.assertEqual(ev.state, "rehit")


class FakeHoleScanner:
    duplicate_radius_px = 18.0

    def __init__(self, dist):
        self.dist = dist

    def _is_near_known_hole(self, x, y):
        return ({"camera_x": 0.0, "camera_y": 0.0}, self.dist)


def features():
    a = np.zeros((40, 40), np.float32)
    big = a.copy()
    big[18:23, 18:23] = 200.0
    return CandidateGeneratorV2._candidate_features(px=20, py=20, saliency=big, absdiff=big, darkening=big,
                                                    dog=big, zscore=big)


def emit_near_old_hole():
    s = HitScanner()
    s.known_holes = [{"hole_id": 1.0, "camera_x": 100.0, "camera_y": 100.0, "score": 30.0, "hit_count": 1.0}]
    t = track(5, 1.0, 20.0, x=106.0, y=100.0, pre_shot_change=20.0)
    ev = AudioShotEvent(1, 1.0, 1.0)
    pushed = []
    with capture_push(pushed):
        s._emit_track_result(t, ev)
    return pushed[0], ev


class EachFlagDirection(unittest.TestCase):
    def pool(self):
        # Track 1: earliest onset, weaker. Track 2: later onset, stronger.
        return (track(1, 10.00, 12.0, hits=2), track(2, 10.05, 30.0, hits=2))

    def test_h8a_evidence_over_timing(self):
        ev = AudioShotEvent(1, 10.0, 10.0)
        with clean_env():
            self.assertEqual(scanner_with(*self.pool())._best_track_for_event(ev).track_id, 1)
        with clean_env(SKJUTBANA_EVIDENCE_OVER_TIMING="1"):
            ev = AudioShotEvent(1, 10.0, 10.0)
            self.assertEqual(scanner_with(*self.pool())._best_track_for_event(ev).track_id, 2)
            self.assertEqual(ev.audio_trigger["winner_selection_flags"]["changed_by"], ["H8a"])
        with clean_env(SKJUTBANA_EVIDENCE_OVER_TIMING="1"):   # tie-break: hits, first_seen, track_id
            a, b = track(3, 10.1, 20.0, hits=2), track(4, 10.0, 20.0, hits=3)
            self.assertEqual(scanner_with(a, b)._best_track_for_event(AudioShotEvent(1, 10.0, 10.0)).track_id, 4)
            a, b = track(3, 10.1, 20.0, hits=2), track(4, 10.0, 20.0, hits=2)
            self.assertEqual(scanner_with(a, b)._best_track_for_event(AudioShotEvent(1, 10.0, 10.0)).track_id, 4)

    def test_h6_uncapped_rank(self):
        with clean_env(SKJUTBANA_UNCAPPED_RANK="1"):
            feats = features()
            self.assertEqual(feats["score"], 35.0)
            self.assertGreater(feats["rank_raw"], 35.0)
            # Two capped candidates on the same frame: higher raw evidence ranks first.
            s = HitScanner()
            update_tracks_frame_unique_v2226(s, [
                {"camera_x": 10.0, "camera_y": 10.0, "score": 35.0, "v26_rank_excess": 1.0},
                {"camera_x": 80.0, "camera_y": 80.0, "score": 35.0, "v26_rank_excess": 20.0}], 10.0)
            for t in s._active_tracks.values():
                self.assertEqual(t.best_score, 35.0)
            ev = AudioShotEvent(1, 10.0, 10.0)
            best = s._best_track_for_event(ev)
            self.assertEqual((best.camera_x, best.rank_value), (80.0, 55.0))
        with clean_env():
            # Default: equal key -> first inserted (the one with lower excess).
            s2 = scanner_with(*s._active_tracks.values())
            self.assertEqual(s2._best_track_for_event(AudioShotEvent(1, 10.0, 10.0)).camera_x, 10.0)
        with clean_env(SKJUTBANA_UNCAPPED_RANK="1"):   # penalty scales the excess too
            c = {"camera_x": 0.0, "camera_y": 0.0, "score": 35.0, "v26_rank_raw": 50.0, "v26_rank_excess": 15.0}
            CandidateGeneratorV2._apply_known_hole_penalty(FakeHoleScanner(5.0), c)
            self.assertAlmostEqual(wsf.candidate_rank_value(c), 50.0 * 0.15)

    def test_h7a_shape_prior(self):
        hole = {"score": 20.0, "area": 9.0, "circularity": 0.9}       # r_eq 1.7 px
        blob = {"score": 20.0, "area": 400.0, "circularity": 0.2}     # r_eq 11 px, irregular
        with clean_env(SKJUTBANA_HOLE_SHAPE_PRIOR="1"):
            self.assertEqual(wsf.candidate_rank_value(hole), 20.0)
            self.assertAlmostEqual(wsf.candidate_rank_value(blob), 6.0)
            s = HitScanner()
            update_tracks_frame_unique_v2226(s, [dict(blob, camera_x=10.0, camera_y=10.0, score=24.0),
                                                 dict(hole, camera_x=80.0, camera_y=80.0)], 10.0)
            self.assertEqual(s._best_track_for_event(AudioShotEvent(1, 10.0, 10.0)).camera_x, 80.0)
        with clean_env(SKJUTBANA_HOLE_SHAPE_PRIOR="1", SKJUTBANA_HOLE_RADIUS_PX="5,15"):
            self.assertLess(wsf.candidate_rank_value(hole), 20.0)

    def test_h7b_signed_darkening(self):
        dark = {"score": 20.0, "v2_darkening": 30.0, "v2_absdiff": 30.0}
        bright = {"score": 24.0, "v2_darkening": 0.0, "v2_absdiff": 30.0}
        with clean_env(SKJUTBANA_SIGNED_DARKENING_PRIOR="1"):
            self.assertEqual(wsf.candidate_rank_value(dark), 20.0)
            self.assertAlmostEqual(wsf.candidate_rank_value(bright), 24.0 * 0.3)
            self.assertEqual(wsf.candidate_rank_value({"score": 9.0}), 9.0)
            s = HitScanner()
            update_tracks_frame_unique_v2226(s, [dict(bright, camera_x=10.0, camera_y=10.0),
                                                 dict(dark, camera_x=80.0, camera_y=80.0)], 10.0)
            self.assertEqual(s._best_track_for_event(AudioShotEvent(1, 10.0, 10.0)).camera_x, 80.0)

    def test_h8b_best_ready_topk(self):
        # Top track (earliest onset) not ready; second is ready.
        def pool():
            return (track(1, 10.00, 30.0, hits=1), track(2, 10.02, 20.0, hits=3, last=10.5))
        with clean_env():
            s = scanner_with(*pool())
            s._wsf_resolve_now_ts = 10.6
            self.assertEqual(s._best_track_for_event(AudioShotEvent(1, 10.0, 10.0)).track_id, 1)
        with clean_env(SKJUTBANA_BEST_READY_TOPK="2"):
            s = scanner_with(*pool())
            s._wsf_resolve_now_ts = 10.6
            ev = AudioShotEvent(1, 10.0, 10.0)
            self.assertEqual(s._best_track_for_event(ev).track_id, 2)
            self.assertEqual(ev.audio_trigger["winner_selection_flags"]["changed_by"], ["H8b"])
            # Through the base resolve: emits the ready track in the same call.
            s = scanner_with(*pool())
            ev = AudioShotEvent(1, 10.0, 10.0)
            s.audio_events.append(ev)
            pushed = []
            with capture_push(pushed):
                self.assertEqual(s._resolve_audio_events(10.6), 1)
            self.assertEqual(ev.matched_track_id, 2)
        with clean_env(SKJUTBANA_BEST_READY_TOPK="2"):   # top ready -> top (no latency/no change)
            s = scanner_with(track(1, 10.0, 30.0, hits=3, last=10.5), track(2, 10.02, 20.0, hits=3, last=10.5))
            s._wsf_resolve_now_ts = 10.6
            self.assertEqual(s._best_track_for_event(AudioShotEvent(1, 10.0, 10.0)).track_id, 1)

    def test_h8b_never_other_shot_or_emitted(self):
        with clean_env(SKJUTBANA_BEST_READY_TOPK="5", SKJUTBANA_EVIDENCE_OVER_TIMING="1"):
            other = track(2, 10.01, 35.0, hits=4, last=10.5, sid=2)
            emitted = track(3, 10.01, 34.0, hits=4, last=10.5, emitted=True)
            late = track(4, 12.5, 33.0, hits=4, last=12.9)
            own = track(1, 10.00, 10.0, hits=1)
            s = scanner_with(other, emitted, late, own)
            s._wsf_resolve_now_ts = 10.6
            ev = AudioShotEvent(1, 10.0, 10.0)
            self.assertEqual(s._best_track_for_event(ev).track_id, 1)   # not ready, but only eligible
            ev2 = AudioShotEvent(1, 10.0, 10.0)
            s2 = scanner_with(other, emitted, late)
            s2._wsf_resolve_now_ts = 10.6
            self.assertIsNone(s2._best_track_for_event(ev2))

    def test_h9a_no_old_hole_snap(self):
        with clean_env(SKJUTBANA_NO_OLD_HOLE_SNAP="1"):
            xy, ev = emit_near_old_hole()
            self.assertEqual(xy, (106.0, 100.0))
            self.assertEqual(ev.state, "matched")
            self.assertIn("winner_selection_no_old_hole_snap", ev.audio_trigger["timing"])

    def test_h9b_soft_penalty(self):
        with clean_env(SKJUTBANA_SOFT_KNOWN_HOLE_PENALTY="1"):
            for dist, factor in ((5.0, 0.6), (12.0, 0.8)):
                c = {"camera_x": 0.0, "camera_y": 0.0, "score": 10.0}
                CandidateGeneratorV2._apply_known_hole_penalty(FakeHoleScanner(dist), c)
                self.assertAlmostEqual(c["score"], 10.0 * factor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
