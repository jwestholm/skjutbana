"""Regression tests for offline evidence semantics; fixtures are not physical validation."""
import copy
import unittest

import numpy as np

from automation.physical_counterfactual_research import owned_history, observed_medoid, SESSIONS
from automation.physical_counterfactual_models import late_reserve, training_groups, model_columns, train_pairwise
from automation.physical_counterfactual_forensics import auc, contributions, summarize_pool, loss_reason
from automation.physical_change_step_research import step_statistics


class Tests(unittest.TestCase):
    def test_history_rejects_future_and_other_producer(self):
        candidate=lambda producer:dict(camera_x=1.,camera_y=2.,v2224_producer_shot_id=producer)
        snap=dict(event=dict(shot_id=7),associations=[
            dict(frame_ts=10.,records=[dict(track_id=1,candidate=candidate(7)),dict(track_id=2,candidate=candidate(8))]),
            dict(frame_ts=12.,records=[dict(track_id=3,candidate=candidate(7))])])
        self.assertEqual(list(owned_history(snap,11.)),[1])

    def test_medoid_is_observed_and_frame_duplicates_do_not_vote_twice(self):
        row=lambda t,x:dict(frame_ts=t,candidate=dict(camera_x=x,camera_y=0.))
        rows=[row(1.,0.),row(1.,10.),row(2.,0.)]
        expected=observed_medoid(rows,dict(camera_x=20.,camera_y=0.))
        self.assertEqual(expected,(0.,0.))
        self.assertEqual(observed_medoid(rows+[row(1.,10.)]*100,{}),expected)

    def test_medoid_does_not_manufacture_midpoint(self):
        rows=[dict(frame_ts=t,candidate=dict(camera_x=x,camera_y=y)) for t,x,y in [(1,0,0),(2,10,0),(3,5,8)]]
        self.assertIn(observed_medoid(rows,{}),[(0.,0.),(10.,0.),(5.,8.)])

    def reserve_fixture(self):
        records=[dict(index=i,camera_x=30.+30*(i%10),camera_y=30.+30*(i//10),
                      physical_priority=float(i%7),selectable=True,gt_distance=0.) for i in range(90)]
        event=dict(context=dict(crop=[0,0,400,400]),pools=dict(current=[0,1],history=list(range(90))))
        names=[f'{family}_clean_r4_{key}' for family in ('persistent','pre_centered','piecewise') for key in ('mean','inner_ring','noise')]
        matrix=np.arange(90*9,dtype=float).reshape(90,9)%13
        return records,event,names,matrix

    def test_reserves_preserve_current_and_obey_budget_without_gt_or_source_scores(self):
        records,event,names,matrix=self.reserve_fixture()
        for mode in ('physical','weak_strata','clean_families'):
            baseline=late_reserve(event,records,matrix,names,mode,budget=12,cap=14)
            audit={}
            self.assertEqual(baseline,late_reserve(event,records,matrix,names,mode,budget=12,cap=14,audit=audit))
            self.assertEqual(audit['stopped_by'],'budget_exhausted')
            self.assertEqual(baseline[:2],[0,1]);self.assertLessEqual(len(baseline),14)
            changed=[dict(r,gt_distance=1000-r['index'],score=10000-r['index'],source='different') for r in records]
            self.assertEqual(baseline,late_reserve(event,changed,matrix,names,mode,budget=12,cap=14))
            self.assertEqual(len(baseline),len(set(baseline)))

    def test_reserve_translation_invariance_and_immutability(self):
        records,event,names,matrix=self.reserve_fixture();before=copy.deepcopy((records,event))
        a=late_reserve(event,records,matrix,names,'weak_strata')
        moved=[dict(r,camera_x=r['camera_x']+1000,camera_y=r['camera_y']+600) for r in records]
        translated=copy.deepcopy(event);translated['context']['crop'][:2]=[1000,600]
        self.assertEqual(a,late_reserve(translated,moved,matrix,names,'weak_strata'))
        self.assertEqual((records,event),before)

    def test_refuse_cap_that_would_drop_current(self):
        records,event,names,matrix=self.reserve_fixture()
        with self.assertRaises(ValueError):late_reserve(event,records,matrix,names,'physical',cap=1)

    def training_fixture(self):
        records=[];events=[]
        for s in SESSIONS:
            ids=[]
            for d,selectable in [(3.,True),(100.,True),(0.,False)]:
                i=len(records);ids.append(i)
                records.append(dict(index=i,session=s,ready=True,gt_distance=d,selectable=selectable))
            events.append(dict(event=s+':1',session=s,truth=dict(state='SINGLE_IMPACT'),pools=dict(current=ids)))
        return dict(records=records,events=events)

    def test_whole_session_holdout_and_no_gt_only_training_coordinates(self):
        data=self.training_fixture()
        for hold in SESSIONS:
            groups=training_groups(data,[s for s in SESSIONS if s!=hold])
            self.assertEqual({g['session'] for g in groups},set(SESSIONS)-{hold})
            for g in groups:
                for i in g['positive']+g['negative']:
                    self.assertTrue(data['records'][i]['selectable'])
                    self.assertNotEqual(data['records'][i]['session'],hold)
        with self.assertRaises(ValueError):training_groups(data,['H10'])
        with self.assertRaises(ValueError):training_groups(data,['S03'])

    def test_no_impact_training_has_no_positive_coordinates(self):
        data=self.training_fixture();data['events'][0]['truth']['state']='NO_PHYSICAL_SHOT'
        g=training_groups(data,['D01'])[0]
        self.assertEqual(g['positive'],[]);self.assertEqual(len(g['negative']),2)

    def test_held_out_features_and_labels_cannot_change_fit(self):
        data=self.training_fixture();matrix=np.tile(np.array([[2.,1.],[0.,0.],[99.,99.]]),(4,1))
        sessions=[s for s in SESSIONS if s!='D01']
        a=train_pairwise(data,matrix,sessions)
        changed=copy.deepcopy(data);other=matrix.copy();other[:3]=123456.
        for record in changed['records'][:3]:record['gt_distance']=9999.
        b=train_pairwise(changed,other,sessions)
        for key in ('mu','sd','beta'):np.testing.assert_array_equal(a[key],b[key])

    def test_feature_family_ablation_keeps_history_separate(self):
        names=['r4_persistence','pre_rms_clean_r4_mean','history_medoid_offset','representative_offset']
        self.assertEqual(model_columns(names,'PAIR_CORE'),[0])
        self.assertEqual(model_columns(names,'PAIR_CLEAN'),[0,1])
        self.assertEqual(model_columns(names,'PAIR_HISTORY'),[0,1,2,3])

    def test_unknown_stage_is_not_measured_zero_recall(self):
        result=summarize_pool([],dict(camera_x=0,camera_y=0),available=False)
        self.assertIsNone(result['oracle']['5']);self.assertIsNone(result['count'])

    def test_tiny_raw_contour_uses_recorded_area_bound_not_guessed_threshold(self):
        event=dict(truth=dict(impacts=[dict(camera_x=0.,camera_y=0.)]))
        details=dict(stage_positions=dict(raw_contours_reconstructed=[dict(camera_x=1.,camera_y=0.,area=0.)]),
            stage_details=[dict(upstream=dict(rejected_blobs=[dict(reason='area (2 vs 2.0-900.0)')]))])
        result=loss_reason('raw_mask_geometry','legacy_precap',5,event,details,[])
        self.assertEqual(result['rule'],'Legacy contour area filter');self.assertEqual(result['min_area'],2.)
        details['stage_details']=[]
        self.assertEqual(loss_reason('raw_mask_geometry','legacy_precap',5,event,details,[])['rule'],'UNAVAILABLE')

    def test_pair_contributions_sum_to_actual_linear_margin(self):
        matrix=np.array([[2.,3.],[1.,7.]])
        model=dict(mu=np.array([0.,1.]),sd=np.array([1.,2.]),beta=np.array([4.,.5,-.3]))
        row=contributions(model,[0,1],matrix,['a','b'],0,1)
        self.assertAlmostEqual(row['correct_minus_wrong_logit'],1.1)

    def test_auc_ties_and_direction(self):
        self.assertEqual(auc([3,4],[1,2]),1.)
        self.assertEqual(auc([1,2],[3,4]),0.)
        self.assertEqual(auc([1],[1]),.5)

    def test_early_persistent_step_is_supported_without_audio_timing_veto(self):
        s=step_statistics([0,0,5,5,5,5],[-.5,-.4,-.3,-.2,.1,.2],0.)
        self.assertEqual(s['step_amplitude'],5.);self.assertEqual(s['step_onset_ms'],-300.)
        self.assertEqual(s['step_persistent_fraction'],1.)

    def test_step_polarity_and_constant_offset_invariance(self):
        x=np.array([1,1,7,7,7,7]);times=np.arange(6)
        a=step_statistics(x,times,2.)
        for shifted in (-x,x+100):
            b=step_statistics(shifted,times,2.)
            self.assertEqual(a['step_snr'],b['step_snr']);self.assertEqual(a['step_onset_ms'],b['step_onset_ms'])

    def test_constant_or_isolated_transient_is_not_a_persistent_step(self):
        for x in ([0,0,0,0,0,0],[0,0,9,0,0,0]):
            self.assertEqual(step_statistics(x,np.arange(6),2.)['step_snr'],0.)

    def test_step_requires_ordered_matching_samples(self):
        with self.assertRaises(ValueError):step_statistics([0]*6,[0,1,1,2,3,4],2.)
        with self.assertRaises(ValueError):step_statistics([0]*4,[0,1,2,3],2.)


if __name__=='__main__':unittest.main()
