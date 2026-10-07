"""Focused evidence-only audit and event-loss contract tests."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from automation import physical_ai_dataset as ds
from automation import physical_ai_selector_gap as gap,physical_ai_event_selector as selector


class SelectorTests(unittest.TestCase):
    def event(self,truth='HIT'):
        return dict(event='KOMP:1',session='KOMP',truth=truth,gt_xy=[0,0] if truth=='HIT' else None,
                    candidates=[dict(id=str(i),xy=xy) for i,xy in enumerate(([0,0],[3,4],[5.01,0]))],indices=[0,1,2])
    def test_multi_positive_boundary(self):
        p=gap.positive_mask(self.event());np.testing.assert_array_equal(p,[1,1,0]);np.testing.assert_array_equal(selector.target(p),[.5,.5,0,0])
    def test_controls_and_missing_proposal_target_unknown(self):
        np.testing.assert_array_equal(selector.target(gap.positive_mask(self.event('NO_IMPACT'))),[0,0,0,1])
        e=self.event();e['gt_xy']=[100,100];np.testing.assert_array_equal(selector.target(gap.positive_mask(e)),[0,0,0,1])
        self.assertIsNone(gap.positive_mask(self.event('UNKNOWN')))
    def test_no_source_score_xy_gt_input(self):
        raw=np.zeros((3,len(ds.FEATURE_NAMES)),np.float32);before=gap.physical_matrix(raw)
        for n in ds.FAMILIES['producer_history']:
            raw[:,ds.FEATURE_NAMES.index(n)]=999
            raw[:,ds.FEATURE_NAMES.index(n+'__missing')]=1
        np.testing.assert_array_equal(before,gap.physical_matrix(raw))
        for n in selector.NAMES:self.assertFalse(any(t in n for t in ('score','camera','gt_','source','event_id','session')))
        with self.assertRaises(ValueError):gap.physical_matrix(np.zeros((3,2)))
    def test_event_relative_is_permutation_equivariant_and_no_mutation(self):
        raw=np.arange(3*len(gap.NAMES),dtype=np.float32).reshape(3,-1);raw[:,len(gap.PHYSICAL):]=0;old=raw.copy()
        a=selector.relative_features(raw);b=selector.relative_features(raw[[2,0,1]])
        np.testing.assert_array_equal(b,a[[2,0,1]]);np.testing.assert_array_equal(old,raw)
        np.testing.assert_array_equal(a,selector.relative_features(raw))
    def test_identical_candidates_cannot_be_distinguished(self):
        x=np.zeros((2,len(gap.NAMES)),np.float32);v=selector.relative_features(x);np.testing.assert_array_equal(v[0],v[1])
    def test_train_only_direction_and_scaler(self):
        e=self.event();fold=dict(train=['KOMP'],dev=['D02'],held=['S01']);x=np.zeros((3,len(gap.NAMES)),np.float32)
        a=gap.train_statistics([e],x,fold)
        self.assertEqual(a[0].shape,(len(gap.NAMES),))
        with self.assertRaises(ValueError):gap.train_statistics([dict(e,session='S01')],x,fold)
    def test_no_held_train_rows(self):
        fold=dict(train=['KOMP'],dev=['D02'],held=['S01']);data=dict(nested_training=[dict(fold=fold,events=[dict(self.event(),session='S01')])])
        with self.assertRaises(ValueError):gap.train_events(data,fold)
    def test_loss_gradient_finite_difference(self):
        scores=np.array([.3,-.2,1.1]);positive=np.array([True,True,False]);loss,g=selector.loss_gradient(scores,positive)
        for i in range(3):
            a=scores.copy();b=scores.copy();a[i]+=1e-5;b[i]-=1e-5
            numerical=(selector.loss_gradient(a,positive)[0]-selector.loss_gradient(b,positive)[0])/2e-5
            self.assertAlmostEqual(numerical,g[i],places=5)
        self.assertGreater(loss,0)
    def test_fixed_unknown_tie_and_no_gt_in_prediction(self):
        from src.engine.ai.training_v223.model import RankModelV223
        model=RankModelV223('linear',selector.NAMES,np.zeros(len(selector.NAMES)),np.ones(len(selector.NAMES)),dict(w=np.zeros(len(selector.NAMES)),b=np.zeros(1)))
        index,scores=selector.predict(model,np.zeros((3,len(gap.NAMES))))
        self.assertIsNone(index)
    def test_determinism_checkpoint_resume_scaler_and_identity(self):
        rng=np.random.default_rng(3);samples=[(rng.normal(size=(3,len(selector.NAMES))).astype(np.float32),np.array([1,0,1],bool)),(rng.normal(size=(2,len(selector.NAMES))).astype(np.float32),np.zeros(2,bool))]
        cfg=dict(kind='mlp',hidden=4,epochs=4,learning_rate=.003,l2=.01,seed=2)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);calls=[0]
            def stop():calls[0]+=1;return calls[0]>2
            with self.assertRaises(InterruptedError):selector.fit(samples,cfg,root/'resumed',{'fold':'train'},stop)
            a,_=selector.fit(samples,cfg,root/'resumed',{'fold':'train'});b,_=selector.fit(samples,cfg,root/'fresh',{'fold':'train'})
            for k in a.arrays:np.testing.assert_array_equal(a.arrays[k],b.arrays[k])
            np.testing.assert_array_equal(a.mean,np.concatenate([x for x,_ in samples]).mean(0))
            with self.assertRaises(ValueError):selector.fit(samples,cfg,root/'resumed',{'fold':'held'})
    def test_s03_and_output_guard(self):
        with self.assertRaises(PermissionError):gap.positive_mask(dict(self.event(),session='S03'))
        with self.assertRaises(PermissionError):gap.audit('/data/skjutbana/S03/forbidden')
        with self.assertRaises(ValueError):ds.output_root('/tmp/not-a-lab-output')
    def test_protected_and_control_scoreboard(self):
        e=dict(self.event(),protected=True);c=dict(self.event('NO_IMPACT'),event='KOMP:2');pred={e['event']:None,c['event']:[0,0]}
        m=ds.evaluate([e,c],pred);self.assertEqual(m['protected_failures'],['KOMP:1']);self.assertEqual(m['control_false_emissions'],1);self.assertEqual(m['unknown'],1)

if __name__=='__main__':unittest.main()
