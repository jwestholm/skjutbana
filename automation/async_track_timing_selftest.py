"""Integration regression: async result timestamps survive frame-unique tracking."""
import unittest
from src.engine.camera.hit_scanner import HitScanner,AudioShotEvent
from src.engine.shot_track_v2226 import _install_frame_unique_tracking_patch

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_frame_unique_tracking_patch()

    def test_delayed_result_keeps_camera_onset_and_association(self):
        scanner=HitScanner();scanner._v2224_async_waiting=False
        scanner._v2224_result_frame_ts=100.122
        scanner._update_tracks([{'camera_x':20,'camera_y':30,'score':40}],101.970)
        track=next(iter(scanner._active_tracks.values()))
        self.assertAlmostEqual(track.first_seen_ts,100.122)
        self.assertEqual(scanner._v2224_result_frame_ts,0)
        event=AudioShotEvent(1,100.0,100.0)
        self.assertIs(scanner._best_track_for_event(event),track)
        scanner._update_tracks([{'camera_x':20,'camera_y':30,'score':42,'v2225_local_confirm':1}],102.0)
        self.assertEqual(track.hits,2)
        self.assertAlmostEqual(track.last_seen_ts,102.0)

    def test_waiting_is_not_a_negative_observation(self):
        scanner=HitScanner();scanner._update_tracks([{'camera_x':20,'camera_y':30,'score':40}],100.1)
        track=next(iter(scanner._active_tracks.values()));before=track.missed_frames
        scanner._v2224_async_waiting=True
        scanner._update_tracks([],101.0)
        self.assertIn(track.track_id,scanner._active_tracks)
        self.assertEqual(track.missed_frames,before)

    def test_completed_empty_result_consumes_its_timestamp(self):
        from src.engine.shot_async_v2224 import tracking_frame_timestamp
        scanner=HitScanner();scanner._v2224_result_frame_ts=100.2;scanner._v2224_async_waiting=False
        self.assertEqual(tracking_frame_timestamp(scanner,[],101.9),100.2)
        self.assertEqual(scanner._v2224_result_frame_ts,0)
        self.assertEqual(tracking_frame_timestamp(scanner,[{}],102),102)

if __name__=='__main__':unittest.main()
