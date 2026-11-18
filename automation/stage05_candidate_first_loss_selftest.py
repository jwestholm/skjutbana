"""Stage 05 characterization tests for two candidate-to-emission boundaries.

These tests use synthetic tracks only. They document current control flow and do
not claim that either branch caused an observed physical XY error.
"""
import unittest
from unittest.mock import patch

from src.engine.camera.hit_scanner import HitScanner, AudioShotEvent, HoleTrack
from src.engine.input.hit_input import hit_input


class CandidateFirstLossTests(unittest.TestCase):
    def test_close_candidate_is_emitted_at_old_hole_xy(self):
        scanner = HitScanner()
        scanner.known_holes = [dict(hole_id=1, camera_x=100.0, camera_y=100.0,
                                    score=10.0, first_seen_ts=1.0,
                                    last_seen_ts=1.0, hit_count=1)]
        event = AudioShotEvent(1, 10.0, 10.0)
        track = HoleTrack(1, 106.0, 108.0, 10.1, 10.1, 10.3,
                          hits=3, best_score=20.0, last_candidate={
                              "camera_x": 106.0, "camera_y": 108.0,
                              "score": 20.0,
                          })
        emitted = []
        with patch.object(hit_input, "push_camera_hit",
                          side_effect=lambda x, y: emitted.append((x, y))):
            self.assertTrue(scanner._emit_track_result(track, event))

        # A 10 px candidate is inside the 12 px rehit radius. Current code
        # records a rehit and publishes the stored hole coordinate instead.
        self.assertEqual(event.state, "rehit")
        self.assertEqual(emitted, [(100.0, 100.0)])
        self.assertEqual((track.camera_x, track.camera_y), (106.0, 108.0))

    def test_low_score_candidate_is_snapped_across_outer_duplicate_radius(self):
        scanner = HitScanner()
        scanner.known_holes = [dict(hole_id=1, camera_x=100.0, camera_y=100.0,
                                    score=10.0, first_seen_ts=1.0,
                                    last_seen_ts=1.0, hit_count=1)]
        event = AudioShotEvent(3, 10.0, 10.0)
        track = HoleTrack(2, 117.5, 100.0, 10.1, 10.1, 10.3,
                          hits=3, best_score=11.0, last_candidate={
                              "camera_x": 117.5, "camera_y": 100.0,
                              "score": 11.0,
                          })
        emitted = []
        with patch.object(hit_input, "push_camera_hit",
                          side_effect=lambda x, y: emitted.append((x, y))):
            self.assertTrue(scanner._emit_track_result(track, event))

        # The weak-score branch applies anywhere _is_near_known_hole accepts
        # the match (18 px), even beyond the 12 px radius branch.
        self.assertEqual(event.state, "rehit")
        self.assertEqual(emitted, [(100.0, 100.0)])
        self.assertAlmostEqual(((117.5 - 100.0) ** 2) ** 0.5, 17.5)

    def test_unready_onset_winner_blocks_ready_runner_up(self):
        scanner = HitScanner()
        event = AudioShotEvent(2, 100.0, 100.0)
        scanner.audio_events.append(event)
        closer = HoleTrack(1, 20.0, 30.0, 100.02, 100.02, 100.02,
                           hits=1, best_score=10.0)
        ready = HoleTrack(2, 80.0, 30.0, 100.10, 100.10, 100.30,
                          hits=3, best_score=9.0)
        scanner._active_tracks = {1: closer, 2: ready}

        self.assertTrue(scanner._track_is_ready(ready, 100.30, event))
        self.assertIs(scanner._best_track_for_event(event), closer)
        self.assertEqual(scanner._resolve_audio_events(100.30), 0)
        self.assertEqual(event.state, "pending")


if __name__ == "__main__":
    unittest.main()
