"""Deterministic observed-change semantics, not physical validation."""
import unittest
import threading
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from src.engine.offline.clean_physical_change import (
    causal_local_onset, local_contrast, persistent_local_support)


class CausalPreTests(unittest.TestCase):
    def test_frame_loader_excludes_future_and_preserves_real_snapshot(self):
        from automation.causal_pre_onset_research import load_observations
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = np.full((100, 100), 80, np.uint8)
            np.save(root/'before.npy', image)
            frames = [dict(timestamp=1., order=0, kind='pre_history', path='before.npy', shape=[100,100], dtype='uint8'),
                      dict(timestamp=1., order=1, kind='pre_snapshot', path='before.npy', shape=[100,100], dtype='uint8'),
                      dict(timestamp=3., order=2, kind='post', path='deliberately_absent.npy', shape=[100,100], dtype='uint8')]
            times, pixels, snap, entries = load_observations(root/'trace.json', {'frames':frames}, [(50,50)], 2., 0., {})
            self.assertEqual(times, [1.])
            np.testing.assert_array_equal(pixels[0], snap[1])
            self.assertEqual(entries[0]['timestamp'], 1.)

    def test_duplicate_receipt_with_conflicting_pixels_is_rejected(self):
        from automation.causal_pre_onset_research import load_observations
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, value in [('a',80), ('b',100)]:
                np.save(root/(name+'.npy'), np.full((100,100), value, np.uint8))
            frames = [dict(timestamp=1., order=j, kind=kind, path=name+'.npy', shape=[100,100], dtype='uint8')
                      for j,(kind,name) in enumerate([('pre_history','a'),('pre_snapshot','b')])]
            with self.assertRaisesRegex(ValueError, 'Snapshot/history'):
                load_observations(root/'trace.json', {'frames':frames}, [(50,50)], 2., 0., {})

    def test_installed_camera_pickup_retains_provenance_and_cursor(self):
        from src.engine.camera.camera_manager import CameraManager
        from src.engine.shot_critical_v2223 import _install_camera_manager_fast_update
        original = CameraManager.update
        installed = getattr(CameraManager, '_v2223_fast_update_patch', False)
        try:
            _install_camera_manager_fast_update()
            frame = SimpleNamespace(timestamp=10., acquisition={'sequence': 7})
            camera = SimpleNamespace(running=True, _lock=threading.Lock(), _ring=[frame],
                                     _last_pickup_count=3, _read_count=7)
            with patch('time.monotonic', return_value=15.):
                CameraManager.update(camera)
            self.assertEqual(frame.acquisition.get('first_main_pickup_monotonic'), 15.)
            with patch('time.monotonic', return_value=16.):
                CameraManager.update(camera)
            self.assertEqual(frame.acquisition['first_main_pickup_monotonic'], 15.)
            self.assertEqual(camera._last_pickup_count, 3)
            self.assertEqual(camera.latest_frame.timestamp, 10.)
        finally:
            CameraManager.update = original
            CameraManager._v2223_fast_update_patch = installed

    def run_signal(self, values):
        times = np.arange(len(values), dtype=float)*.08
        return causal_local_onset(values, times, cutoff=times[-1])

    def test_contaminated_pre_is_replaced_by_actual_preceding_frame(self):
        values = [0]*5 + [-9]*5
        r = self.run_signal(values)
        self.assertEqual(r['state'], 'ONSET_FOUND')
        self.assertEqual(r['pre_index'], 4)
        self.assertEqual(r['bracket'], [.32, .4])
        self.assertEqual(persistent_local_support(values[7], values[-3:], r['noise'])['support'], 0)
        self.assertEqual(persistent_local_support(values[r['pre_index']], values[-3:], r['noise'])['support'], 12)
        self.assertEqual(r['physical_state'], 'UNCERTAIN')

    def test_both_polarities_and_repeatability(self):
        for sign in (-1, 1):
            a = self.run_signal([0]*4 + [sign*6]*4)
            self.assertEqual(a, self.run_signal([0]*4 + [sign*6]*4))
            self.assertEqual(a['state'], 'ONSET_FOUND')

    def test_unchanged_old_hole_is_not_onset(self):
        patches = np.full((8, 49, 49), 150., dtype=float)
        patches[:, 23:26, 23:26] = 20
        self.assertEqual(self.run_signal(local_contrast(patches))['state'], 'NO_SUPPORTED_ONSET')

    def test_adjacent_new_hole_preserves_old_hole(self):
        patches = np.full((8, 49, 49), 150., dtype=float)
        patches[:, 22:25, 21:23] = 20  # Existing damage stays in every frame.
        patches[4:, 23:26, 25:28] = 20  # Adjacent physical change.
        r = self.run_signal(local_contrast(patches))
        self.assertEqual(r['state'], 'ONSET_FOUND')
        self.assertEqual(r['pre_index'], 3)
        self.assertEqual(patches[r['pre_index'], 22, 21], 20)
        self.assertEqual(patches[r['pre_index'], 24, 26], 150)

    def test_uniform_illumination_cancels(self):
        patches = np.full((8, 49, 49), 80.)
        patches[4:] += 20
        self.assertEqual(self.run_signal(local_contrast(patches))['state'], 'NO_SUPPORTED_ONSET')

    def test_small_noise_and_no_change(self):
        for x in ([0]*9, [.1, -.2, .2, 0, -.1, .1]):
            self.assertEqual(self.run_signal(x)['state'], 'NO_SUPPORTED_ONSET')

    def test_two_steps_are_ambiguous(self):
        self.assertEqual(self.run_signal([0]*4+[8]*4+[16]*4)['state'], 'ONSET_AMBIGUOUS')

    def test_reversion_is_ambiguous(self):
        self.assertEqual(self.run_signal([0]*4+[8]*4+[0]*4)['state'], 'ONSET_AMBIGUOUS')

    def test_unstable_before_does_not_manufacture_pre(self):
        r = self.run_signal([0, 8, -8, 8, -8, 0, 8, -8])
        self.assertEqual(r['state'], 'ONSET_AMBIGUOUS')
        self.assertIsNone(r['pre_index'])

    def test_insufficient_history_and_late_onset(self):
        self.assertEqual(self.run_signal([0, 0, 0, 8, 8])['state'], 'INSUFFICIENT_HISTORY')
        self.assertEqual(self.run_signal([0]*6+[8]*2)['state'], 'ONSET_AMBIGUOUS')

    def test_receipt_cutoff_and_order_rejected(self):
        for t, cutoff in (([0, 1, 1, 2, 3, 4], 5), ([1, 0, 2, 3, 4, 5], 6),
                          ([0, .08, .16, .24, .32, .4], .39)):
            with self.assertRaises(ValueError):
                causal_local_onset([0]*6, t, cutoff=cutoff)

    def test_missing_sequence_and_gaps_are_not_duplicates(self):
        t = np.arange(8)*.08
        r = causal_local_onset([0]*4+[8]*4, t, cutoff=t[-1], sequences=[1,2,3,5,6,7,8,9])
        self.assertEqual(r['state'], 'ONSET_FOUND')
        with self.assertRaises(ValueError):
            causal_local_onset([0]*8, t, cutoff=t[-1], sequences=[1,2,3,3,4,5,6,7])
        t[4:] += .3
        self.assertEqual(causal_local_onset([0]*4+[8]*4, t, cutoff=t[-1])['state'], 'INSUFFICIENT_HISTORY')

    def test_nonfinite_and_invalid_patches_rejected(self):
        with self.assertRaises(ValueError):
            self.run_signal([0]*5+[np.nan])
        with self.assertRaises(ValueError):
            local_contrast(np.zeros((3, 48, 49)))
        with self.assertRaises(ValueError):
            persistent_local_support(0, [1, 2], .75)


if __name__ == '__main__':
    unittest.main()
