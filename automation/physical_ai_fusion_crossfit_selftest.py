"""Nested exclusion and interruption/index regressions; synthetic fixtures only."""
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from automation import physical_ai_dataset as ds,physical_ai_lab as lab
from automation import physical_ai_fusion as fusion,physical_ai_lab_index as index
from automation import physical_ai_fusion_crossfit as cf,physical_ai_vision as vision
from automation.physical_ai_fusion_selftest import dataset,proposal


class CrossfitTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.data,self.x=dataset(self.root/'data')
    def tearDown(self):self.temp.cleanup()
    def config(self,*_):return dict(lab.config_for(2,0),hidden=4,epochs=3,threshold=.5)
    def metrics(self):return ds.evaluate(self.data['events'],{e['event']:None for e in self.data['events']})
    def test_legacy_missing_fusion_status_readonly(self):
        root=self.root/'legacy';root.mkdir();state=dict(champion='old',history=['old'],metrics=self.metrics())
        ds.atomic_json(root/'champion.json',state);ds.atomic_json(root/'manifest.json',dict(dataset=str(self.root/'data'),dataset_hash=self.data['dataset_hash']))
        ds.atomic_json(root/'journal.json',[dict(trial='trial_000000')]);before=(root/'champion.json').read_bytes()
        with patch.object(lab,'completed',side_effect=AssertionError('No full result scan')):s=fusion.status(root)
        self.assertEqual(s['trials'],1);self.assertEqual(s['scope'],self.data['fusion']);self.assertEqual(before,(root/'champion.json').read_bytes())
    def test_sigint_checkpoint_status_resume(self):
        root=self.root/'run';real_train=lab.train_fusion_model;checks=[0];sent=[False]
        def interrupted(*args,**kw):
            stop=kw['stop']
            def signal_after_checkpoint():
                checks[0]+=1
                if checks[0]==2:sent[0]=True;os.kill(os.getpid(),signal.SIGINT)
                return stop()
            return real_train(*args,**dict(kw,stop=signal_after_checkpoint))
        old_handler=signal.getsignal(signal.SIGINT)
        with patch.object(fusion,'config_for',side_effect=self.config),patch.object(lab,'train_fusion_model',side_effect=interrupted):
            lab.run(root,self.root/'data',hours=.02,max_trials=1,adapter=fusion)
        self.assertTrue(sent[0]);self.assertEqual(signal.getsignal(signal.SIGINT),old_handler)
        self.assertTrue((root/'trials/trial_000000/fold_00/training_checkpoint.npz').exists())
        self.assertEqual(fusion.status(root)['trials'],0)
        with patch.object(fusion,'config_for',side_effect=self.config):lab.run(root,None,hours=None,max_trials=1,resume=True,adapter=fusion)
        self.assertEqual(fusion.status(root)['trials'],1)
    def test_second_sigint_during_finalization_is_deferred(self):
        root=self.root/'run';original=ds.atomic_json;sent=[False]
        def write(path,value):
            if Path(path).name=='champion.json' and not sent[0]:
                sent[0]=True;os.kill(os.getpid(),signal.SIGINT)
            return original(path,value)
        with patch.object(ds,'atomic_json',side_effect=write):lab.run(root,self.root/'data',hours=.01,max_trials=1,adapter=fusion)
        self.assertTrue(sent[0]);self.assertIn('scope',fusion.status(root))
        self.assertIn('fusion',ds.read(root/'champion.json'))
    def test_incremental_views_bounded_and_recover_committed_result(self):
        root=self.root/'index';root.mkdir();fusion.refresh(root,self.data);m=self.metrics()
        for i in range(205):
            r=dict(trial=f'trial_{i:06d}',status='COMPLETE',metrics=m,finished_at=float(i))
            ds.atomic_json(root/'trials'/r['trial']/'result.json',r)
            if i<204:fusion.publish(root,r,self.data)
        # The last result was durable when publication was interrupted.
        fusion.refresh(root,self.data);state=fusion.status(root);self.assertEqual(state['trials'],205)
        before=ds.digest(root/'champion.json');fusion.refresh(root,self.data);self.assertEqual(before,ds.digest(root/'champion.json'))
        b=ds.read(root/'leaderboard.json');self.assertLessEqual(len(b['recent']),100);self.assertLessEqual(len(b['top']),20)
        self.assertLess((root/'leaderboard.json').stat().st_size,180000);self.assertLess((root/'journal.json').stat().st_size,256)
        self.assertEqual(ds.read(root/'journal.json')['completed_trials'],205)
        with patch.object(lab,'completed',side_effect=AssertionError('No growing scan')):self.assertEqual(fusion.status(root)['trials'],205)
    def test_atomic_index_wins_over_stale_champion_view(self):
        root=self.root/'run';root.mkdir();fusion.refresh(root,self.data)
        ds.atomic_json(root/'champion.json',dict(broken=True))
        self.assertEqual(fusion.status(root)['champion'],'frozen_exact_witness_baseline')
    def test_nested_plan_excludes_all_outer_and_predicted_sessions(self):
        for outer in ds.folds(['KOMP','D02','S01','S02','POST_FIX']):
            for predicted in outer['train']:
                r=cf.nesting(outer,predicted)
                self.assertFalse(set(r['train'])&set(outer['held']+outer['dev']+[predicted]))
                for target in r['train']:
                    singleton=set(r['train'])-{target};self.assertEqual(len(singleton),1)
                    self.assertFalse(singleton&set(outer['held']+outer['dev']+[predicted,target]))
    def test_s03_refused_before_data_access(self):
        with self.assertRaises(PermissionError):cf.nesting(dict(train=['S01','S02','S03'],held=['KOMP'],dev=[]),'S01')
        with self.assertRaises(PermissionError):cf.training_only(self.data,['S03'])
        with self.assertRaises(PermissionError):cf.load(self.root/'S03')
    def test_transitive_lineage_guard(self):
        outer=dict(train=['KOMP','D02','S01'],held=['S02'],dev=['POST_FIX']);e=self.data['events'][0]
        p=proposal(e,['D02','S01']);p['reranker_upstream_models']=['one','two']
        models=dict(base=dict(training_sessions=['D02','S01']),reranker=dict(training_sessions=['D02','S01'],upstream=['one','two']),one=dict(training_sessions=['D02']),two=dict(training_sessions=['S01']))
        cf.assert_chain(p,e,outer,models)
        models['one']['training_sessions']=['S02']
        with self.assertRaises(ValueError):cf.assert_chain(p,e,outer,models)
    def test_train_sampler_never_sees_held_labels_or_maps(self):
        selected=cf.training_only(self.data,['KOMP']);self.assertEqual({e['session'] for e in selected['events']},{'KOMP'})
        rows={e['event']:dict(event=e['event'],origin=[0,0]) for e in self.data['events']}
        data=dict(self.data,vision_rows=rows);cfg=dict(seed=1,stride=1,channels='persistent',positive_radius=6.5)
        fold=dict(train=['KOMP'],held=['S01'],dev=['S02'])
        def maps(cache,row,stride,representation):
            self.assertTrue(row['event'].startswith('KOMP:'));return np.arange(8*64*64,dtype=np.uint8).reshape(8,64,64)
        with patch.object(vision,'event_maps',side_effect=maps):
            a,norm=vision.training_samples(cf.training_only(data,fold['train']),self.root,fold,cfg)
            poisoned=copy.deepcopy(data)
            for e in poisoned['events']:
                if e['session']!='KOMP':e['gt_xy']='must not be read'
            b,other=vision.training_samples(cf.training_only(poisoned,fold['train']),self.root,fold,cfg)
        self.assertEqual(norm,other)
        for x,y in zip(a,b):np.testing.assert_array_equal(x.patches,y.patches)
    def test_protected_regression_still_refused(self):
        a=self.metrics();b=copy.deepcopy(a);b['selected']['5']=6;b['protected_failures']=['KOMP:6']
        self.assertFalse(fusion.compare(b,a)[0])


if __name__=='__main__':unittest.main()
