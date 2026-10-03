import unittest
import numpy as np
from src.engine.offline.common_verifier import (CommonFrameContext,extract_patch_context,
    patch_representation,CommonVerifier,coherent_change_evidence,coherent_event_decision)

class Disabled(CommonVerifier):
    def score_track(self,track,context):return {'score':1}

class CommonVerifierTests(unittest.TestCase):
    def setUp(self):
        self.pre=np.full((40,40),100,np.uint8);self.post=self.pre.copy();self.post[18:23,18:23]=50
        self.ctx=CommonFrameContext(self.pre,(self.post,self.post),1.5,(1.2,1.8))
    def test_cutoff_and_patch_shapes(self):
        b=extract_patch_context(self.ctx,(20,20));self.assertEqual(len(b['pairs']),1);self.assertEqual(b['pairs'][0]['signed'].shape,(33,33))
    def test_representations_deterministic(self):
        b=extract_patch_context(self.ctx,(20,20));
        for k in ('difference','gradient','pca','texture'):self.assertTrue(np.array_equal(patch_representation(b,k),patch_representation(b,k)))
    def test_disabled_verifier_is_empty(self):
        self.assertEqual(Disabled().verify_event([],self.ctx),[])


class CoherentEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.names=[f'r{r}_{k}' for r in (2,4,8) for k in
                    ('abs_late','center_ring_ratio','spatial_consistency','pre_std')]

    def test_joint_evidence_cannot_borrow_support_from_another_scale(self):
        raw=np.array([[10,10,0,0, 0,0,1,0, 0,0,0,0],
                      [4,4,1,0, 0,0,0,0, 0,0,0,0]],float)
        score,_=coherent_change_evidence(np.log1p(raw),self.names)
        self.assertEqual(score[0],0)
        self.assertAlmostEqual(score[1],.8)

    def test_pre_variability_softens_but_does_not_veto_rehit(self):
        raw=np.tile([4.,4.,1.,0.],(2,3));raw[1,3::4]=100
        encoded=np.log1p(raw)
        plain,_=coherent_change_evidence(encoded,self.names)
        contextual,_=coherent_change_evidence(encoded,self.names,pre_context=True)
        self.assertEqual(plain[0],plain[1])
        self.assertGreater(contextual[0],contextual[1]);self.assertGreater(contextual[1],0)

    def test_source_truth_xy_and_static_texture_do_not_enter_score(self):
        names=self.names+['source','gt_x','camera_x','old_hole','pre_texture']
        x=np.column_stack([np.log1p(np.tile([4.,4.,1.,0.],(3,3))),np.zeros((3,5))])
        a,_=coherent_change_evidence(x,names,pre_context=True)
        x[:,-5:]=np.nan
        b,_=coherent_change_evidence(x,names,pre_context=True)
        np.testing.assert_array_equal(a,b)

    def test_permutation_equivariance_and_immutable_inputs(self):
        raw=np.tile([4.,4.,1.,0.],(3,3));raw[:,0]=[1,3,9]
        x=np.log1p(raw);saved=x.copy();p=[2,0,1]
        a,_=coherent_change_evidence(x,self.names)
        b,_=coherent_change_evidence(x[p],self.names)
        np.testing.assert_array_equal(a[p],b);np.testing.assert_array_equal(x,saved)

    def test_no_candidates_or_no_change_is_uncertain_not_no_physical_hit(self):
        for scores,coords in [([],[]),([0],[[1,2]])]:
            d=coherent_event_decision(scores,coords,abstain=True)
            self.assertEqual(d['state'],'UNCERTAIN');self.assertIsNone(d['selected'])

    def test_distinct_competing_damage_is_uncertain(self):
        d=coherent_event_decision([.8,.5],[[0,0],[100,0]],abstain=True)
        self.assertEqual(d['state'],'UNCERTAIN')
        self.assertEqual(coherent_event_decision([.8,.5],[[0,0],[5,0]],abstain=True)['state'],'PHYSICAL_HIT')
        self.assertEqual(coherent_event_decision([.8,.2],[[0,0],[100,0]],abstain=True)['selected'],0)

    def test_decision_translation_invariance(self):
        a=coherent_event_decision([.8,.5],[[0,0],[100,0]],abstain=True)
        b=coherent_event_decision([.8,.5],[[987,421],[1087,421]],abstain=True)
        self.assertEqual(a,b)

    def test_malformed_evidence_is_rejected(self):
        x=np.zeros((1,len(self.names)))
        for v in (np.nan,-1.,100.):
            bad=x.copy();bad[0,0]=v
            with self.assertRaises(ValueError):coherent_change_evidence(bad,self.names)
        bad=x.copy();bad[0,2]=np.log1p(2)
        with self.assertRaises(ValueError):coherent_change_evidence(bad,self.names)
        with self.assertRaises(ValueError):coherent_change_evidence(x,self.names[:-1])
        with self.assertRaises(ValueError):coherent_event_decision([np.nan],[[0,0]])

    def test_summary_does_not_reward_uncertainty_as_correct_no_hit(self):
        from automation.physical_selection_pass import summarize_group,comparison
        from src.engine.offline.physical_event_truth import event_truth,score_event
        truth=event_truth('NO_PHYSICAL_SHOT',[])
        r=dict(truth=truth,result=score_event(truth,None),decision=dict(state='UNCERTAIN'),
               oracle={str(t):False for t in (5,10,20,42)},ranks={str(t):None for t in (5,10,20,42)},
               candidate_count=0,unready_at_recorded_decision=False)
        r['comparison']=comparison(r,r)
        s=summarize_group([r])
        self.assertEqual(s['no_hit_correct_decisions'],0);self.assertEqual(s['no_hit_uncertain'],1)
        self.assertEqual(s['false_emissions'],0)

    def test_ledger_missing_stage_is_unavailable_not_negative_evidence(self):
        from automation.physical_selection_pass import pool_measurement
        from src.engine.offline.physical_event_truth import event_truth
        truth=event_truth('SINGLE_IMPACT',[dict(camera_x=20.,camera_y=20.)])
        self.assertIsNone(pool_measurement([],truth,available=False)['oracle']['42'])
        self.assertFalse(pool_measurement([],truth)['oracle']['42'])

    def test_missing_proposal_export_uses_verified_associations_with_ownership(self):
        from automation.physical_selection_pass import causal_observations
        def observation(x,owner):
            return dict(candidate=dict(camera_x=x,camera_y=0.,timestamp=1.,v2224_producer_shot_id=owner))
        d=dict(stage_positions={},snapshot=dict(event=dict(shot_id=2),associations=[
            dict(frame_ts=1.,records=[observation(5.,2),observation(8.,1)]),
            dict(frame_ts=3.,records=[observation(7.,2)])]))
        result=causal_observations(d,dict(cutoff=2.))
        self.assertEqual([c['camera_x'] for c in result],[5.])

    def test_regression_preserved_at_each_camera_tolerance(self):
        from automation.physical_selection_pass import comparison
        from src.engine.offline.physical_event_truth import event_truth,score_event
        truth=event_truth('SINGLE_IMPACT',[dict(camera_x=0.,camera_y=0.)])
        current=dict(result=score_event(truth,dict(camera_x=9.146,camera_y=0.)))
        alternative=dict(result=score_event(truth,dict(camera_x=197.077,camera_y=0.)))
        self.assertEqual(comparison(current,alternative),
                         {'5':'WRONG_TO_WRONG','10':'REGRESSION','20':'REGRESSION','42':'REGRESSION'})

if __name__=='__main__':unittest.main()
