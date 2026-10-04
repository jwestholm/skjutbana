"""Diagnostic isolation and preprocessing parity; no accuracy claims."""
import inspect
import sys
import unittest
import numpy as np
import cv2
from automation import chronological_information_audit as a
from src.engine.ai.training_v223.evidence_patch_v2235 import build_registered_evidence_channels,EVIDENCE_CHANNEL_NAMES
from src.engine.offline.clean_physical_change import temporal_step_evidence


class InformationAuditTests(unittest.TestCase):
    def frames(self,times):
        return [dict(frame_id=f'/tmp/audit_frame_{i}.npy',ordinal=i,receipt_timestamp=t,camera_sequence=None) for i,t in enumerate(times)]
    def images(self):
        rng=np.random.default_rng(4);pre=rng.integers(80,120,(80,100),dtype=np.uint8);post=pre.copy();post[40:45,50:55]=20
        return [pre]*3+[post]*3
    def test_s03_rejected_before_open(self):
        fs=self.frames([1.,2.]);fs[0]['frame_id']='/data/skjutbana/S03/never_open.npy'
        with self.assertRaises(PermissionError):a.validate_frames(fs,2.)
    def test_receipt_order(self):
        with self.assertRaises(ValueError):a.validate_frames(self.frames([2.,1.]),2.)
    def test_cutoff(self):
        with self.assertRaises(ValueError):a.validate_frames(self.frames([1.,2.]),1.5)
    def test_ordinals_not_timestamps(self):
        a.validate_frames(self.frames([None,None]),None)
        with self.assertRaises(ValueError):a.validate_frames(self.frames([None,None]),2.)
    def test_mixed_clock_refused(self):
        with self.assertRaises(ValueError):a.validate_frames(self.frames([None,2.]),2.)
    def test_sequence_order(self):
        fs=self.frames([1.,2.]);fs[0]['camera_sequence']=10;fs[1]['camera_sequence']=9
        with self.assertRaises(ValueError):a.validate_frames(fs,2.)
    def test_duplicate_identity(self):
        fs=self.frames([1.,2.]);fs[1]['frame_id']=fs[0]['frame_id']
        with self.assertRaises(ValueError):a.validate_frames(fs,2.)
    def test_no_truth_parameter_in_maps(self):
        self.assertEqual(list(inspect.signature(a.stage_maps).parameters),['images'])
        self.assertEqual(list(inspect.signature(a.transition_maps).parameters),['before','after','registered'])
    def test_stage_parity_with_existing_builder(self):
        ims=self.images();st,_=a.stage_maps(ims)
        actual,_=build_registered_evidence_channels(ims[0],ims[-3:])
        np.testing.assert_array_equal(actual,np.stack([st['5_uint8'][k] for k in EVIDENCE_CHANNEL_NAMES]))
    def test_deterministic_and_no_mutation(self):
        ims=self.images();before=ims[0].copy();x,_=a.stage_maps(ims);y,_=a.stage_maps(ims)
        for stage in x:
            for key in x[stage]:
                # Existing OpenCV magnitude arithmetic varies a few float32 ULPs
                # with allocation/SIMD alignment; integer model inputs must match.
                if stage=='5_uint8':np.testing.assert_array_equal(x[stage][key],y[stage][key])
                else:np.testing.assert_allclose(x[stage][key],y[stage][key],rtol=2e-6,atol=4*255*np.finfo(np.float32).eps)
        np.testing.assert_array_equal(ims[0],before)
    def test_instrumentation_restored(self):
        previous=sys.getprofile();a.stage_maps(self.images());self.assertIs(previous,sys.getprofile())
    def test_gt_diagnosis_cannot_change_maps(self):
        x,_=a.transition_maps(*[im for im in self.images()[::3]])
        before=x['dark'].copy();a.rank_map(x['dark'],[52,42],np.array([100,200]));a.rank_map(x['dark'],[20,20],np.array([100,200]))
        np.testing.assert_array_equal(before,x['dark'])
    def test_camera_origin_no_refinement(self):
        m=np.zeros((50,50),np.float32);m[20,25]=10
        r=a.rank_map(m,[25,20],np.array([100,200]));self.assertEqual(r['peak_xy'],[125,220]);self.assertEqual(r['raw_rank'],1)
    def test_quantization_ties_explicit(self):
        m=np.zeros((50,50),np.float32);m[2,2]=1;m[20,25]=1
        r=a.rank_map(m,[25,20],np.array([0,0]));self.assertEqual(r['strict_rank'],1);self.assertEqual(r['tie_rank_end'],2);self.assertEqual(r['raw_rank'],2)
    def test_registration_same_reference_grid(self):
        rng=np.random.default_rng(12);before=rng.integers(0,255,(160,180),dtype=np.uint8)
        after=cv2.warpAffine(before,np.float32([[1,0,3],[0,1,2]]),(180,160),borderMode=cv2.BORDER_REFLECT)
        raw,_=a.transition_maps(before,after);reg,meta=a.transition_maps(before,after,True)
        self.assertEqual(raw['dark'].shape,reg['dark'].shape);self.assertEqual(meta['applied'],1.)
        self.assertLess(reg['absolute'][10:-10,10:-10].mean(),raw['absolute'][10:-10,10:-10].mean())
    def test_onset_reuses_ordinal_adapter(self):
        r=temporal_step_evidence([0,0,0,-10,-10,-10,-10]);self.assertTrue(r['supported']);self.assertEqual(r['certification'],'ORDINAL_ONLY_TIMING_UNKNOWN')
    def test_cohort_rule_frozen_numeric(self):
        events=[dict(event=f'S01:{i}',session='S01',truth='HIT',baseline_xy=[0,0],gt_xy=[0,0]) for i in (10,2,1)]
        self.assertEqual(a.cohort(events)[-1],'S01:1')
    def test_persistent_probe_no_truth(self):
        self.assertEqual(list(inspect.signature(a.persistent_maps).parameters),['images','times'])
    def test_persistent_step_and_retained_prior_hole(self):
        pre=np.full((20,20),100,np.uint8);pre[2,2]=20;post=pre.copy();post[10,10]=50
        m,windows=a.persistent_maps([pre,post,post,post])
        self.assertEqual(m['dark'][10,10],50);self.assertEqual(m['unsigned'][2,2],0)
        self.assertEqual(windows,[dict(before=0,onset=1,available=3)])
    def test_transient_rejected_and_input_unchanged(self):
        pre=np.full((20,20),100,np.uint8);post=pre.copy();post[10,10]=50
        ims=[pre,post,pre,pre];old=np.stack(ims).copy();m,_=a.persistent_maps(ims)
        self.assertFalse(np.any(m['unsigned']));np.testing.assert_array_equal(np.stack(ims),old)
    def test_probe_requires_causal_persistence_frames(self):
        with self.assertRaises(ValueError):a.persistent_maps(self.images()[:3])
        m,w=a.persistent_maps(self.images()[:4],[0.,.1,.2,.6]);self.assertEqual(w,[])
    def test_probe_time_order(self):
        with self.assertRaises(ValueError):a.persistent_maps(self.images()[:4],[0.,.1,.1,.2])
    def test_probe_deterministic(self):
        x,w=a.persistent_maps(self.images());y,v=a.persistent_maps(self.images())
        self.assertEqual(w,v)
        for name in x:np.testing.assert_array_equal(x[name],y[name])
    def test_primary_prefix_does_not_use_later_frames(self):
        pre=np.full((60,60),100,np.uint8);post=pre.copy();post[30,30]=0
        history=[pre]*4+[post]*3
        x,_=a.persistent_maps(history[:4]);y,_=a.persistent_maps(history)
        self.assertFalse(np.any(x['unsigned']));self.assertEqual(y['dark'][30,30],100)
    def test_fixed_peak_budget(self):
        m=np.zeros((100,100),np.float32);m[::8,::8]=1
        xy,_=a.peaks.raw_peaks(m);retained,_=a.peaks.nms_trace(xy,6,32)
        self.assertEqual(len(retained),32)


if __name__=='__main__':unittest.main()
