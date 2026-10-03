"""Synthetic measurement contracts; no physical accuracy inference."""
import copy
import unittest
import numpy as np

from automation.chronological_pre_research import transition_maps,temporal,bounded_keep,CAPACITY
from automation.kompetensprov_selection_research import prepare_maps
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2


def candidate(i, *, born=0, supported=False, x=None, evidence=None):
    return dict(uid=str(i),camera_x=float(i*10 if x is None else x),camera_y=40.,born=born,
        score=float(i),v251_region_evidence=float(i if evidence is None else evidence),
        temporal=dict(supported=supported,support=float(i) if supported else 0.))


class Tests(unittest.TestCase):
    def test_maps_match_existing_generator_array_exactly(self):
        rng=np.random.default_rng(260927)
        ref=rng.integers(35,200,(96,112),dtype=np.uint8)
        previous=[np.clip(ref.astype(int)+rng.integers(-3,4,ref.shape),0,255).astype(np.uint8) for _ in range(3)]
        current=previous[-1].copy();current[45:49,52:56]-=20
        mask=np.zeros(ref.shape,np.uint8);mask[8:-8,8:-8]=255
        before=[im.copy() for im in previous+[current,mask]]
        _,_,expected=prepare_maps(previous,current,mask,1,10.,9.)
        actual,_=transition_maps(CandidateGeneratorV2(),previous,current,mask)
        for key in ('saliency','absdiff','darkening','dog','zscore','valid'):
            np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
        self.assertEqual(actual['bbox'],expected['bbox'])
        for im,old in zip(previous+[current,mask],before):np.testing.assert_array_equal(im,old)

    def test_prefix_cannot_see_future_step(self):
        x=np.r_[np.zeros(6),np.ones(6)*-8]
        self.assertFalse(temporal(x[:6])['supported'])
        self.assertFalse(temporal(x[:8])['supported'])
        self.assertTrue(temporal(x[:9])['supported'])
        self.assertEqual(temporal(x)['onset_ordinal'],6)
        self.assertEqual(temporal(x)['certification'],'ORDINAL_ONLY_TIMING_UNKNOWN')

    def test_actual_times_validate_gaps_and_order(self):
        x=np.r_[np.zeros(6),np.ones(6)*-8];times=np.arange(12)*.08
        a=temporal(x,times,np.arange(12));b=temporal(x)
        self.assertEqual(a['support'],b['support'])
        times[6:]+=1
        self.assertFalse(temporal(x,times)['supported'])
        with self.assertRaises(ValueError):temporal(x,times[::-1])

    def test_no_onset_and_reversion_stay_uncertain(self):
        self.assertFalse(temporal(np.zeros(12))['supported'])
        self.assertFalse(temporal(np.r_[np.zeros(6),np.ones(6)*8,np.zeros(6)])['supported'])
        multiple=temporal(np.r_[np.zeros(6),np.ones(6)*-10,np.ones(6)*-6])
        self.assertFalse(multiple['supported'])
        self.assertGreater(len(multiple['choices']),1)

    def test_capacity_and_supported_retention(self):
        cs=[candidate(i,born=10) for i in range(32)]+[candidate(100,born=0,supported=True)]
        before=copy.deepcopy(cs);kept,reasons=bounded_keep(cs,10)
        self.assertEqual(len(kept),CAPACITY);self.assertEqual(kept[0]['uid'],'100')
        self.assertEqual(cs,before);self.assertEqual(len(reasons),len(cs)-CAPACITY)
        self.assertEqual((kept,reasons),bounded_keep(cs,10))

    def test_pending_lifetime_and_supported_retirement(self):
        c=candidate(1,born=2)
        self.assertTrue(bounded_keep([c],4)[0])
        self.assertEqual(bounded_keep([c],5)[1]['1'],'unsupported_lifetime_expired')
        c['temporal']['supported']=True
        self.assertTrue(bounded_keep([c],49)[0])
        c['temporal']['supported']=False
        self.assertFalse(bounded_keep([c],49)[0])

    def test_spatial_dedupe_preserves_observed_coordinate(self):
        a=candidate(1,born=0,x=100,evidence=5)
        b=candidate(2,born=1,x=102,evidence=6)
        kept,reasons=bounded_keep([a,b],1)
        self.assertEqual(len(kept),1);self.assertEqual(kept[0]['camera_x'],102.)
        self.assertEqual(kept[0]['born'],1)
        self.assertEqual(reasons['1'],'spatial_representative_replaced')


if __name__=='__main__':unittest.main()
