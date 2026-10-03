"""Pure software regressions; fixture coordinates are not physical validation."""
import copy
import unittest

import numpy as np

from src.engine.offline.observation_hypotheses import observed_hypotheses
from automation.physical_hypothesis_research import coverage_reserve, group_order


class HypothesisTests(unittest.TestCase):
    def fixture(self):
        def row(x, owner=7, confirm=False):
            return dict(track_id=1,observation_id=str(x),producer_shot_id=owner,
                candidate=dict(camera_x=x,camera_y=0.,v2224_producer_shot_id=owner,
                    v2225_local_confirm=int(confirm),v2225_confirm_compact=4.))
        return dict(event=dict(shot_id=7),tracks=[dict(track_id=1,eligible=True,camera_x=10.,camera_y=0.)],
            associations=[dict(frame_ts=1.,observed_at=1.1,records=[row(10,confirm=True),row(0)]),
                dict(frame_ts=2.,observed_at=2.1,records=[row(0,confirm=True),row(11)]),
                dict(frame_ts=3.,observed_at=4.1,records=[row(22)]),
                dict(frame_ts=3.,observed_at=3.1,records=[row(33,owner=8)]),
                dict(frame_ts=5.,observed_at=5.1,records=[row(44)])])

    def test_xy_is_observed_and_witnesses_preserve_owner_cutoff_and_identity(self):
        s=self.fixture();result=observed_hypotheses(s,4.)
        self.assertEqual({h['camera_x'] for h in result['hypotheses']},{0,10,11})
        self.assertEqual(result['rejected'],dict(future_frame=1,future_delivery=1,other_producer=1,nonfinite_xy=0))
        for h in result['hypotheses']:
            for w in h['witnesses']:
                self.assertEqual(w['producer_shot_id'],7)
                self.assertLessEqual(w['frame_ts'],h['cutoff'])
                self.assertLessEqual(w['observed_at'],h['cutoff'])

    def test_confirmation_is_never_inherited_from_parent_xy(self):
        s=self.fixture();r=observed_hypotheses(s,1.5)['hypotheses']
        self.assertIsNotNone(next(h for h in r if h['camera_x']==10)['local_confirmation'])
        self.assertIsNone(next(h for h in r if h['camera_x']==0)['local_confirmation'])
        r=observed_hypotheses(s,4.)['hypotheses']
        self.assertIsNotNone(next(h for h in r if h['camera_x']==0)['local_confirmation'])
        self.assertIsNone(next(h for h in r if h['camera_x']==11)['local_confirmation'])

    def test_bounded_output_and_no_input_mutation(self):
        s=self.fixture();before=copy.deepcopy(s)
        r=observed_hypotheses(s,4.,extra_budget=1)
        self.assertEqual(len(r['hypotheses']),2)
        self.assertEqual(s,before)
        self.assertTrue(all(len(h['witnesses'])<=3 for h in r['hypotheses']))
        with self.assertRaises(ValueError):observed_hypotheses(s,4.,extra_budget=-1)

    def test_duplicate_same_frame_witness_does_not_create_temporal_support(self):
        s=self.fixture();s['associations'][0]['records']*=100
        r=observed_hypotheses(s,1.5)
        self.assertTrue(all(len(h['witnesses'])==1 for h in r['hypotheses']))

    def test_all_explicit_ownership_fields_are_checked(self):
        s=self.fixture();s['associations'][0]['records'][1]['dispatch_owner']=8
        r=observed_hypotheses(s,1.5)
        self.assertEqual(len(r['hypotheses']),1)

    def test_track_limit_refuses_silent_data_loss(self):
        s=self.fixture();s['tracks'].append(dict(track_id=2,eligible=True,camera_x=20,camera_y=0))
        with self.assertRaises(ValueError):observed_hypotheses(s,4.,max_tracks=1)

    def test_joint_aggregation_does_not_reward_duplicate_coordinates(self):
        score=np.array([3.,1.,2.5]);groups={1:[0,1],2:[2]}
        for mode in ['mean','max','logmeanexp','current_identity']:
            order,_=group_order(groups,score,mode,2)
            self.assertEqual(order,group_order({1:[0,1,0,1],2:[2]},score,mode,2)[0])
        self.assertEqual(group_order(groups,score,'max',2)[0][0],0)
        self.assertEqual(group_order(groups,score,'mean',2)[0][0],2)
        self.assertEqual(group_order(groups,score,'current_identity',2)[0][0],2)

    def test_coverage_is_bounded_gt_free_and_translation_invariant(self):
        records=[dict(camera_x=float(i),camera_y=float(i%3),selectable=True) for i in range(20)]
        event=dict(context=dict(crop=[0,0,100,50]),pools=dict(current=[0,1],raw_plausible=list(range(20))))
        a=coverage_reserve(event,records,budget=4,cap=6)[0]
        self.assertEqual(a[:2],[0,1]);self.assertEqual(len(a),6)
        changed=[dict(r,gt_distance=100-i,score=i*100,camera_x=r['camera_x']+100) for i,r in enumerate(records)]
        self.assertEqual(a,coverage_reserve(event,changed,budget=4,cap=6)[0])
        self.assertEqual(coverage_reserve(event,records,budget=0)[0],[0,1])


if __name__=='__main__':unittest.main()
