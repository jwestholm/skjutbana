"""One-column ablation, same model and UNKNOWN semantics."""
import inspect
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from automation import physical_ai_locality_ablation as ablation
from automation import physical_ai_event_selector as selector


class AblationTests(unittest.TestCase):
    def matrix(self):
        x=np.random.default_rng(17).normal(size=(7,34)).astype(np.float32)
        x[:,17:]=0.;x[2,33]=np.log(2.);x[4,18]=np.log(2.)
        return x

    def test_exactly_one_column_removed(self):
        x=self.matrix();old=selector.relative_features(x,ablation.PHYSICAL)
        np.testing.assert_array_equal(ablation.project(x),old[:,:-1])
        self.assertEqual(ablation.NAMES,ablation.WITH_NAMES[:-1])
        self.assertNotIn('core_ring_concentration__event_midrank',ablation.NAMES)

    def test_locality_does_not_affect_retained_ranks(self):
        x=self.matrix();a=ablation.project(x);x[:,16]*=-100;x[:,33]=1
        b=ablation.project(x)
        np.testing.assert_array_equal(a[:,34:],b[:,34:])
        self.assertEqual(list(inspect.signature(ablation.project).parameters),['physical'])

    def test_no_locality_rank_computation_in_inference(self):
        with patch.object(selector,'relative_features',wraps=selector.relative_features) as called:
            ablation.project(self.matrix())
        self.assertEqual(called.call_count,1)
        self.assertEqual(called.call_args.args[0].shape[1],32)

    def test_permutation_and_missingness(self):
        x=self.matrix();order=[5,1,0,6,3,4,2]
        np.testing.assert_array_equal(ablation.project(x[order]),ablation.project(x)[order])
        np.testing.assert_array_equal(ablation.project(x)[:,:34],x)

    def test_schema_and_nonfinite_refusal(self):
        with self.assertRaises(ValueError):ablation.project(np.zeros((2,33)))
        x=self.matrix();x[0,0]=np.nan
        with self.assertRaises(ValueError):ablation.project(x)

    def test_same_fixed_zero_unknown_and_rankmodel(self):
        model=selector.RankModelV223('linear',ablation.NAMES,np.zeros(50),np.ones(50),dict(w=np.zeros(50),b=np.zeros(1)))
        self.assertIsNone(ablation.predict(model,self.matrix())[0])
        model.arrays['b'][0]=.1;self.assertEqual(ablation.predict(model,self.matrix())[0],0)
        model.feature_names=ablation.WITH_NAMES
        with self.assertRaises(ValueError):ablation.predict(model,self.matrix())

    def test_ablation_fit_deterministic_and_resumable(self):
        x=ablation.project(self.matrix());samples=[(x,np.array([True,True,False,False,False,False,False])),(x,np.zeros(7,bool))]
        cfg=dict(kind='linear',hidden=12,epochs=2,learning_rate=.003,l2=.01,seed=1729)
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            first,_=selector.fit(samples,cfg,a,{},feature_names=ablation.NAMES)
            second,_=selector.fit(samples,cfg,b,{},feature_names=ablation.NAMES)
            resumed,_=selector.fit(samples,cfg,a,{},feature_names=ablation.NAMES)
            for k in first.arrays:
                np.testing.assert_array_equal(first.arrays[k],second.arrays[k])
                np.testing.assert_array_equal(first.arrays[k],resumed.arrays[k])

    def test_forbidden_sources_and_output(self):
        with self.assertRaises(PermissionError):ablation.run('/data/skjutbana/S03/forbidden')
        with self.assertRaises(ValueError):ablation.run('/tmp/not-a-data-output')


if __name__=='__main__':unittest.main()
