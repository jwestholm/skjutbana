"""Measurement contracts; synthetic fixtures make no physical accuracy claim."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from automation.kompetensprov_selection_research import ordinal_steps, metrics, xy, clean
from src.engine.offline.clean_physical_change import causal_local_onset
from src.engine import shot_region_proposal_v251 as region
from src.engine.shot_cross_thread_novelty_v253 import balance_confirmed_v253, _candidate_full_camera_xy


class MeasurementTests(unittest.TestCase):
    def test_amplitude_adapter_matches_frozen_reference(self):
        rng = np.random.default_rng(252)
        signals = [np.zeros(12), np.r_[np.zeros(6), np.ones(6)*-8],
                   np.r_[np.zeros(5), np.ones(5)*-8, np.zeros(5)],
                   np.r_[np.zeros(5), np.ones(5)*-8, np.ones(5)*-16]]
        signals += [rng.normal(0, 3, 50) for _ in range(30)]
        mapping = {'ONSET_FOUND':'UNIQUE_PERSISTENT_OBSERVED_STEP',
                   'NO_SUPPORTED_ONSET':'NO_SUPPORTED_OBSERVED_STEP', 'ONSET_AMBIGUOUS':'AMBIGUOUS'}
        for signal in signals:
            times = np.arange(len(signal))*.08
            ref = causal_local_onset(signal, times, cutoff=times[-1])
            result = ordinal_steps(signal)
            self.assertEqual(result['candidates'], ref['candidates'])
            self.assertEqual(result['state'], mapping[ref['state']])
            self.assertEqual(result['causal_certification'], 'UNAVAILABLE')
            self.assertNotIn('bracket', result)

    def test_missing_time_never_becomes_physical_certification(self):
        result = ordinal_steps(np.r_[np.zeros(6),np.ones(6)*8])
        self.assertNotIn('physical_state', result)
        self.assertNotIn('onset_timestamp', result)
        self.assertEqual(ordinal_steps([1,2])['state'], 'INSUFFICIENT_SAMPLES')
        with self.assertRaises(ValueError): ordinal_steps([np.nan]*10)

    def test_inputs_unmodified_and_repeatable(self):
        signal = np.r_[np.zeros(6), np.ones(6)*-8]; original=signal.copy()
        self.assertEqual(ordinal_steps(signal), ordinal_steps(signal))
        np.testing.assert_array_equal(signal,original)
        c = dict(camera_x=7.25,camera_y=8.5,timestamp=123,score=4)
        self.assertEqual(xy(c),[1548.25,995.5])
        self.assertNotIn('timestamp',clean(c)); self.assertEqual(c['timestamp'],123)
        self.assertEqual(metrics([c],xy(c))['nearest'],0)

    def test_registered_balance_cap_and_no_xy_movement(self):
        group=region.RegionGroupV251('g',('g',),(),0,0,200,200)
        candidates=[dict(camera_x=float(i*5),camera_y=40.,score=i,
                    v251_region_group='g',v251_registered_region_proposal=1.,v251_region_evidence=i)
                    for i in range(20)]
        for cap in (8,16,32):
            with patch.object(region,'_per_region_proposals',return_value=cap):
                kept=region._balance_merged_candidates(candidates,(group,),shot_id=1)
            self.assertEqual(len(kept), min(20,cap))
            self.assertTrue(all((c['camera_x'],c['camera_y']) in
                            [(d['camera_x'],d['camera_y']) for d in candidates] for c in kept))
        self.assertNotIn('v251_shot_id',candidates[0])

    def test_confirm_two_is_downstream_cap(self):
        candidates=[dict(camera_x=float(i),camera_y=0.,v251_region_group='g',
                        v253_authority_ok=1.,v253_novelty_score=float(i),
                        v2225_confirm_center_abs=2.,v2225_confirm_compact=1.) for i in range(8)]
        kept=balance_confirmed_v253(candidates)
        self.assertEqual([c['camera_x'] for c in kept],[7.,6.])
        self.assertNotIn('v253_confirm_score',candidates[0])

    def test_novelty_metadata_uses_same_full_camera_mapping(self):
        scanner=SimpleNamespace(_v244_roi_diag=dict(crop=(1541,987,1483,711),
                                scale=(1.,1.),work_shape=(711,1483)))
        c=dict(camera_x=346.,camera_y=456.)
        self.assertEqual(list(_candidate_full_camera_xy(scanner,c)),xy(c))
        self.assertEqual(xy(c),[1887.,1443.])


if __name__ == '__main__':
    unittest.main()
