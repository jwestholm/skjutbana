"""Information-loss, causal observation and unchanged selector regressions."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from automation import physical_ai_dataset as ds,physical_ai_selector_gap as gap
from automation import physical_ai_event_selector as selector
from automation import physical_ai_temporal_representation as repair
from src.engine.offline.common_verifier import query_physical_witness,query_temporal_observation


def witness(signal,producer='CLASSIC',clocked=True):
    frames=[dict(frame_id=f'frame{i}',ordinal=i,receipt_timestamp=i*.05 if clocked else None,camera_sequence=i if clocked else None) for i in range(len(signal))]
    return query_physical_witness(signal,frames,event_id='KOMP:1',candidate_id='a',coordinate_version='v1',camera_xy=[20.,30.],source_provenance=dict(producer=producer),cutoff_frame_id=frames[-1]['frame_id'],cutoff_ts=frames[-1]['receipt_timestamp']).witnesses[0]


class RepresentationTests(unittest.TestCase):
    def test_lost_distinct_unsupported_signal_preserved_without_certifying_onset(self):
        a=witness(np.linspace(0,-6,12));b=witness(np.linspace(0,-1,12))
        self.assertEqual(a['state'],'UNKNOWN');self.assertEqual(b['state'],'UNKNOWN')
        np.testing.assert_array_equal(ds.vector(dict(physical_witness=a)),ds.vector(dict(physical_witness=b)))
        before=copy.deepcopy(a);va,ma=repair.observation_vector(a);vb,mb=repair.observation_vector(b)
        self.assertNotEqual(va,vb);self.assertEqual(a,before);self.assertLess(ma['signed_change'],mb['signed_change'])
        meta=query_temporal_observation(a).metadata;self.assertFalse(meta['onset_certified']);self.assertFalse(meta['clean_pre_certified']);self.assertEqual(meta['physical_state'],'UNKNOWN')
    def test_source_coordinate_id_gt_independence(self):
        w=witness(np.linspace(0,-6,12));changed=copy.deepcopy(w)
        changed.update(camera_xy=[99.,101.],candidate_id='different',coordinate_version='different',event_id='D02:1',source_provenance=dict(producer='VISION',score=999),gt_xy=[1,1])
        self.assertEqual(repair.observation_vector(w),repair.observation_vector(changed))
    def test_short_history_explicit_missing(self):
        v,r=repair.observation_vector(witness([0,1,2,3,4]));self.assertEqual(v,[0,0,1,1]);self.assertIsNone(r['signed_change']);self.assertFalse(r['available'])
    def test_no_future_or_reordered_frames(self):
        w=witness([0]*3+[-6]*3);w['availability']['receipt_cutoff']=.20
        with self.assertRaises(ValueError):repair.observation_vector(w)
        w=witness([0]*3+[-6]*3);w['frames'][2]['receipt_timestamp']=.21
        with self.assertRaises(ValueError):repair.observation_vector(w)
        w=witness([0]*3+[-6]*3);w['frames'][2]['ordinal']=1
        with self.assertRaises(ValueError):repair.observation_vector(w)
    def test_ordinal_is_not_fake_clock(self):
        v,r=repair.observation_vector(witness([0]*3+[-6]*3,clocked=False));self.assertEqual(r['certification'],'ORDINAL_ONLY_TIMING_UNKNOWN');self.assertIsNone(r['receipt_span']);self.assertEqual(v[:2],[-6,8])
    def test_revocation_not_overridden(self):
        w=witness([0]*3+[-6]*3+[0]*3);old=copy.deepcopy(w);repair.observation_vector(w);self.assertEqual(w,old);self.assertEqual(w['state'],'UNKNOWN')
    def test_additive_brightness_offset_and_reversal(self):
        a=repair.observation_vector(witness([0]*3+[-6]*3))[0]
        b=repair.observation_vector(witness([50]*3+[44]*3))[0];self.assertEqual(a,b)
        c=repair.observation_vector(witness([0]*3+[-6,6,-6]))[0];self.assertEqual(c[1],0)
    def test_projection_missing_and_permutation(self):
        raw=np.zeros((3,len(ds.FEATURE_NAMES)),np.float32);extra=np.array([[-5,5,0,0],[0,0,1,1],[-2,2,0,0]],np.float32)
        m=repair.matrix(raw,extra);rel=selector.relative_features(m,repair.PHYSICAL)
        np.testing.assert_array_equal(selector.relative_features(m[[2,0,1]],repair.PHYSICAL),rel[[2,0,1]])
        n=len(repair.PHYSICAL);self.assertGreater(m[1,n+len(gap.PHYSICAL)],0)
        changed=raw.copy();changed[:,ds.FEATURE_NAMES.index('score')]=999
        np.testing.assert_array_equal(repair.matrix(changed,extra),m)
    def test_feature_schema_bound_checkpoint(self):
        cfg=dict(kind='linear',hidden=12,epochs=2,learning_rate=.003,l2=.01,seed=2);samples=[(np.ones((2,3),np.float32),np.array([True,False]))]
        with tempfile.TemporaryDirectory() as tmp:
            model,_=selector.fit(samples,cfg,tmp,{},feature_names=('a','b','c'))
            self.assertEqual(model.feature_names,('a','b','c'))
            with self.assertRaises(ValueError):selector.fit(samples,cfg,tmp,{},feature_names=('x','b','c'))
    def test_s03_refused_and_output_guard(self):
        with self.assertRaises(PermissionError):repair.build('/data/skjutbana/S03/forbidden')
        with self.assertRaises(ValueError):repair.build('/tmp/no-output')

if __name__=='__main__':unittest.main()
