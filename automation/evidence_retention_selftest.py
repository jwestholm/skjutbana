"""Regression checks for bounded evidence preservation; fixtures are not physical validation."""
import copy
import unittest
import numpy as np

from automation.evidence_retention_research import bounded_diverse, observation_coordinates
from automation.accuracy_verifier_research import event_predictions
from src.engine.offline.physical_event_truth import event_truth
from src.engine.camera.hit_scanner import HitScanner
from src.engine.shot_track_v2226 import update_tracks_frame_unique_v2226
from src.engine.track_audit import capture, record_selection
from src.engine.camera.hit_scanner import AudioShotEvent
from src.engine.offline.track_replay import current_exact_replay


class Tests(unittest.TestCase):
    def test_same_frame_coordinates_survive_for_research_without_extra_temporal_hits(self):
        scanner = HitScanner(); scanner.physical_trace_capture_enabled = True
        scanner.last_trace_pipeline_shot_id = 1
        event = AudioShotEvent(1, 10., 10.); scanner.audio_events.append(event)
        first = dict(camera_x=100., camera_y=100., score=9., v2224_producer_shot_id=1)
        useful = dict(camera_x=90., camera_y=102., score=2., v2224_producer_shot_id=1)
        update_tracks_frame_unique_v2226(scanner, [first, useful], 10.02)
        track = next(iter(scanner._active_tracks.values()))
        self.assertEqual(track.hits, 1)
        self.assertEqual((track.camera_x, track.camera_y), (100., 100.))
        update_tracks_frame_unique_v2226(scanner, [dict(first, v2225_local_confirm=1)], 10.15)
        self.assertEqual(track.hits, 2)
        record_selection(scanner, event, track)
        snapshot = capture(scanner, event, track)
        frozen = copy.deepcopy(snapshot)
        alternatives = observation_coordinates(snapshot)
        self.assertEqual([(c['camera_x'], c['camera_y']) for c in alternatives], [(90., 102.)])
        self.assertFalse(alternatives[0]['ready'])
        self.assertEqual(snapshot, frozen)
        self.assertEqual(current_exact_replay(snapshot)['track_id'], track.track_id)

    def test_budget_and_strata_protect_sparse_regions(self):
        candidates = [dict(camera_x=10.+i/100, camera_y=10., score=1000-i, radius=2) for i in range(250)]
        sparse = dict(camera_x=90., camera_y=90., score=-100, radius=4)
        candidates.append(sparse); original = copy.deepcopy(candidates)
        result = bounded_diverse(candidates, (0, 0), (100, 100), 200, head=150)
        self.assertEqual(len(result), 200)
        self.assertIn(sparse, result)
        self.assertEqual(candidates, original)
        self.assertEqual(result, bounded_diverse(candidates, (0, 0), (100, 100), 200, head=150))

    def test_diversity_is_translation_invariant(self):
        cs = [dict(camera_x=x, camera_y=y, score=i, radius=2) for i, (x, y) in enumerate([(5, 5), (40, 20), (80, 80)])]
        a = bounded_diverse(cs, (0, 0), (100, 100), 2)
        translated = [dict(c, camera_x=c['camera_x']+500, camera_y=c['camera_y']+700) for c in cs]
        b = bounded_diverse(translated, (500, 700), (100, 100), 2)
        self.assertEqual([c['score'] for c in a], [c['score'] for c in b])

    def test_no_observation_from_another_producer_or_ineligible_parent(self):
        snapshot = dict(event=dict(shot_id=1), tracks=[dict(track_id=1, eligible=True, camera_x=0, camera_y=0),
                                                     dict(track_id=2, eligible=False, camera_x=10, camera_y=10)],
            associations=[dict(records=[dict(track_id=1, observation_id='wrong', candidate=dict(camera_x=1, camera_y=1, v2224_producer_shot_id=2)),
                                        dict(track_id=2, observation_id='ineligible', candidate=dict(camera_x=11, camera_y=11))])])
        self.assertEqual(observation_coordinates(snapshot), [])

    def test_invalid_budgets_refused(self):
        with self.assertRaises(ValueError):
            bounded_diverse([], (0, 0), (10, 10), 0)
        with self.assertRaises(ValueError):
            bounded_diverse([], (0, 0), (10, 10), 10, head=11)
        for budget in (0, True, 1.5):
            with self.assertRaises(ValueError):
                observation_coordinates({}, budget)

    def test_future_observation_cannot_enter_alternate_pool(self):
        snapshot = dict(event=dict(shot_id=1), tracks=[dict(track_id=1, eligible=True, camera_x=0, camera_y=0)],
            associations=[dict(frame_ts=11., records=[dict(track_id=1, observation_id='future',
                candidate=dict(camera_x=1, camera_y=1, v2224_producer_shot_id=1))])])
        self.assertEqual(observation_coordinates(snapshot, cutoff=10.), [])

    def test_negative_rank_key_is_ordering_not_an_abstention_threshold(self):
        truth = event_truth('SINGLE_IMPACT', [dict(camera_x=100., camera_y=100.)], reason='numerical fixture')
        candidate = dict(index=0, camera_x=109., camera_y=100., pool='current', ready=True,
                         gt_distance=9., track_id=1)
        dataset = dict(records=[candidate], events=[dict(event='fixture', session='fixture', truth=truth, indices=[0])])
        row = event_predictions(dataset, np.array([-1.]), ['fixture'], threshold=-float('inf'))[0]
        self.assertTrue(row['result']['hits']['10'])
        self.assertEqual(row['selected']['track_id'], 1)


if __name__ == '__main__':
    unittest.main()
