"""Lab safeguards and restart tests; synthetic fixtures are not physical metrics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from automation import physical_ai_dataset as ds, physical_ai_lab as lab
from src.engine.ai.training_v223.model import train_fusion_model, RankModelV223
from src.engine.ai.training_v223.evidence_model_v2235 import hard_negative_indices


def fixture(root):
    root.mkdir(); events=[]; matrices=[]
    for i,s in enumerate(('KOMP','S01','S02')):
        for j in range(2):
            start=len(matrices); x=np.zeros((3,len(ds.FEATURE_NAMES)),np.float32)
            x[:,0]=[2,-2,-3]; matrices.extend(x)
            events.append(dict(event=f'{s}:{j+1}',session=s,source_session=f'physical_{s}',truth='HIT',
                               gt_xy=[10.,10.],indices=list(range(start,start+3)),
                               candidates=[dict(id=str(k),xy=xy) for k,xy in enumerate(([10.,10.],[100.,100.],[130.,120.]))],
                               source_root=str(root/s),frames=[],baseline_xy=None,protected=False,
                               status='SYNTHETIC_TEST_FIXTURE_NOT_PHYSICAL_VALIDATION'))
    np.save(root/'features.npy',np.asarray(matrices)); ds.atomic_json(root/'provenance.json',[])
    data=dict(schema=dict(revision=ds.SCHEMA,names=list(ds.FEATURE_NAMES)),events=events,
              sessions=['KOMP','S01','S02'],folds=ds.folds(['KOMP','S01','S02']),source_hashes={},
              feature_sha256=ds.digest(root/'features.npy'),provenance_sha256=ds.digest(root/'provenance.json'))
    data['dataset_hash']=ds.canonical_hash(data); ds.atomic_json(root/'dataset.json',data)
    return data,np.asarray(matrices)


class LabTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.data,self.x=fixture(self.root/'data')
    def tearDown(self): self.tmp.cleanup()
    def save_modified(self,data):
        data=copy.deepcopy(data); data.pop('dataset_hash',None); data['dataset_hash']=ds.canonical_hash(data)
        ds.atomic_json(self.root/'data/dataset.json',data)
    def config(self):
        return dict(seed=12,hidden=4,epochs=3,learning_rate=.01,l2=.001,threshold=.5,hard_negative_count=2,excluded_families=[])
    def test_s03_names_refused(self):
        for name in ('S03','s03.json','/tmp/session_2026_S03_data/trace.json','validation_UNTOUCHED'):
            with self.subTest(name=name),self.assertRaises(PermissionError): ds.guard(name)
    def test_s03_symlink_refused(self):
        alias=self.root/'alias'; alias.symlink_to(self.root/'S03'/'missing')
        with self.assertRaises(PermissionError): ds.read(alias)
    def test_s03_manifest_refused_before_array_load(self):
        d=copy.deepcopy(self.data);d['events'][0]['source_root']='/tmp/S03/never_open'
        self.save_modified(d)
        with patch.object(np,'load',side_effect=AssertionError('must not load')),self.assertRaises(PermissionError): ds.load(self.root/'data')
    def test_s03_runner_name(self):
        with self.assertRaises(PermissionError): lab.location('/data/skjutbana','S03')
    def test_manifest_tamper(self):
        d=copy.deepcopy(self.data);d['events'][0]['gt_xy']=[2,3]
        ds.atomic_json(self.root/'data/dataset.json',d)
        with self.assertRaises(ValueError): ds.load(self.root/'data')
    def test_feature_tamper(self):
        np.save(self.root/'data/features.npy',self.x+1)
        with self.assertRaises(ValueError): ds.load(self.root/'data')
    def test_source_alias_leakage(self):
        d=copy.deepcopy(self.data);d['events'][2]['source_session']=d['events'][0]['source_session']
        self.save_modified(d)
        with self.assertRaises(ValueError): ds.load(self.root/'data')
    def test_session_folds_disjoint(self):
        held=[]
        for f in self.data['folds']:
            self.assertFalse(set(f['train'])&set(f['dev']+f['held']))
            self.assertFalse(set(f['dev'])&set(f['held']));held+=f['held']
        self.assertEqual(sorted(held),sorted(self.data['sessions']))
    def test_no_gt_in_vectors(self):
        c=dict(area=2,physical_witness=dict(measurements={},state='UNKNOWN',revocation=dict(revoked_now=False)))
        a=ds.vector(c);c.update(gt_xy=[999,999],camera_xy=[100,3],gt_distance=0,session='truth')
        self.assertEqual(a,ds.vector(c))
        self.assertEqual(a[len(ds.RAW_NAMES)+ds.RAW_NAMES.index('noise')],1)
    def test_unknown_truth_not_negative(self):
        e=copy.deepcopy(self.data['events'][0]);e['truth']='UNKNOWN';e['gt_xy']=None
        self.assertEqual(lab.samples([e],self.x),[])
        m=ds.evaluate([e],{e['event']:None});self.assertEqual(m['controls'],0)
    def test_unknown_not_control_rejection(self):
        e=copy.deepcopy(self.data['events'][0]);e['truth']='NO_IMPACT';e['gt_xy']=None
        m=ds.evaluate([e],{e['event']:None})
        self.assertEqual(m['validated_control_rejections'],0);self.assertEqual(m['control_unknown'],1)
    def test_oracle_not_selected(self):
        e=self.data['events'][0];m=ds.evaluate([e],{e['event']:[100,100]})
        self.assertEqual(m['oracle']['5'],1);self.assertEqual(m['selected']['5'],0)
    def test_metrics_missing_predictions_refused(self):
        with self.assertRaises(ValueError): ds.evaluate(self.data['events'],{})
    def test_upstream_in_sample_refused(self):
        e=self.data['events'][0];f=dict(train=['S01','S02'],dev=['POST_FIX'],held=['KOMP'])
        p=dict(event=e['event'],session=e['session'],training_sessions=['S01','KOMP'],model_sha256='h',dataset_hash='d')
        with self.assertRaises(ValueError): ds.validate_upstream(p,e,f)
    def test_upstream_nested_outer_leakage(self):
        e=self.data['events'][2];f=dict(train=['S01','S02'],dev=['POST_FIX'],held=['KOMP'])
        p=dict(event=e['event'],session=e['session'],training_sessions=['S02','KOMP'],model_sha256='h',dataset_hash='d')
        with self.assertRaises(ValueError): ds.validate_upstream(p,e,f)
        p['training_sessions']=['S02'];ds.validate_upstream(p,e,f)
    def test_miner_preserves_stable_order(self):
        np.testing.assert_array_equal(hard_negative_indices([1,4,4,8],[1,99,100,20],per_shot=2),[1,2])
    def test_resume_matches_uninterrupted(self):
        s=lab.samples(self.data['events'],self.x);kw=dict(feature_names=ds.FEATURE_NAMES,epochs=5,hidden=4,seed=8)
        baseline,_=train_fusion_model(s,**kw)
        count=[0]
        def stop(): count[0]+=1;return count[0]>2
        with self.assertRaises(InterruptedError): train_fusion_model(s,checkpoint=self.root/'ck.npz',stop=stop,**kw)
        resumed,_=train_fusion_model(s,checkpoint=self.root/'ck.npz',**kw)
        for k in baseline.arrays: np.testing.assert_array_equal(baseline.arrays[k],resumed.arrays[k])
    def test_checkpoint_config_rejected(self):
        s=lab.samples(self.data['events'],self.x);kw=dict(feature_names=ds.FEATURE_NAMES,epochs=2,hidden=4,checkpoint=self.root/'ck.npz')
        train_fusion_model(s,**kw)
        with self.assertRaises(ValueError):train_fusion_model(s,seed=999,**kw)
    def test_model_roundtrip_and_no_mutation(self):
        before=self.x.copy();m,_=train_fusion_model(lab.samples(self.data['events'],self.x),feature_names=ds.FEATURE_NAMES,epochs=2)
        m.save(self.root/'model');re=RankModelV223.load(self.root/'model')
        np.testing.assert_array_equal(m.score_matrix(self.x),re.score_matrix(self.x));np.testing.assert_array_equal(self.x,before)
    def test_fold_resume_no_retrain(self):
        out=self.root/'trial';out.mkdir();cfg=self.config()
        a=lab.train_trial(out,self.data,self.x,cfg,lambda:False)
        with patch.object(lab,'train_fusion_model',side_effect=AssertionError('completed folds immutable')):
            b=lab.train_trial(out,self.data,self.x,cfg,lambda:False)
        self.assertEqual(a,b)
    def test_protected_regression_blocks_champion(self):
        old=ds.evaluate(self.data['events'],{e['event']:None for e in self.data['events']})
        new=copy.deepcopy(old);new['selected']['5']=6;new['protected_failures']=['KOMP:6']
        self.assertFalse(lab.champion_compare(new,old)[0])
    def test_new_catastrophe_blocks_champion(self):
        old=ds.evaluate(self.data['events'],{e['event']:None for e in self.data['events']})
        new=copy.deepcopy(old);new['selected']['5']=4;new['per_event'][0]['error']=101
        self.assertFalse(lab.champion_compare(new,old)[0])
    def test_champion_history_survives_resume(self):
        root=self.root/'experiment'
        with patch.object(lab,'config_for',side_effect=lambda *_:self.config()):
            lab.run(root,self.root/'data',hours=.02,max_trials=1)
            history={p.name:p.read_bytes() for p in (root/'champions').glob('*.json')}
            lab.run(root,None,hours=None,max_trials=2,resume=True)
        self.assertTrue(history)
        for name,value in history.items():self.assertEqual(value,(root/'champions'/name).read_bytes())
        self.assertEqual(len(lab.completed(root)),2)
    def test_exclusive_worker_lock(self):
        with lab.exclusive(self.root),self.assertRaises(RuntimeError):
            with lab.exclusive(self.root):pass
    def test_atomic_journal_ignores_partial_temp(self):
        p=self.root/'trials/trial_000000';p.mkdir(parents=True);(p/'result.json.tmp').write_text('{')
        self.assertEqual(lab.completed(self.root),[])
    def test_ablations_remove_family_and_missingness(self):
        cfg=self.config();cfg['excluded_families']=['chronological'];names,_=lab.columns(cfg)
        self.assertNotIn('support',names);self.assertNotIn('support__missing',names)
    def test_vision_session_overlap_refused(self):
        from automation.physical_ai_vision import train
        with self.assertRaises(ValueError):train(self.root,self.root/'out',['S01'],['S01'])
    def test_vision_own_peaks(self):
        from src.engine.ai.training_v223.heatmap_model_v2236 import peak_camera_xy
        score=np.zeros((8,8),np.float32);score[3,4]=10
        xy=peak_camera_xy(score,top_k=1)
        np.testing.assert_array_equal(xy,[[17.5,13.5]])
    def test_random_configs_deterministic(self):
        self.assertEqual(lab.config_for(4,9),lab.config_for(4,9));self.assertNotEqual(lab.config_for(4,9),lab.config_for(4,10))
    def test_source_hash_path_refused(self):
        d=copy.deepcopy(self.data);d['source_hashes']['/tmp/S03/trace.json']='unused'
        self.save_modified(d)
        with self.assertRaises(PermissionError):ds.load(self.root/'data')
    def test_direct_runner_refuses_forbidden_before_creation(self):
        path=self.root/'S03'
        with self.assertRaises(PermissionError):lab.run(path,self.root/'data',hours=.1)
        self.assertFalse(path.exists())
    def test_code_changed_resume_refused(self):
        root=self.root/'experiment'
        with patch.object(lab,'config_for',side_effect=lambda *_:self.config()):
            lab.run(root,self.root/'data',hours=.02,max_trials=1)
        before=(root/'champion.json').read_bytes()
        with patch.object(lab,'code_identity',return_value={'changed':True}),self.assertRaises(ValueError):
            lab.run(root,None,hours=None,resume=True)
        self.assertEqual(before,(root/'champion.json').read_bytes())
    def test_queued_ablation_actually_retrains(self):
        root=self.root/'experiment'
        with patch.object(lab,'config_for',side_effect=lambda *_:self.config()):
            lab.run(root,self.root/'data',hours=.02,max_trials=1)
        jobs=lab.queue_ablations(root);self.assertEqual(len(jobs),len(ds.FAMILIES))
        with patch.object(lab,'train_trial',wraps=lab.train_trial) as train:
            lab.run_ablations(root,.02)
            self.assertEqual(train.call_count,len(ds.FAMILIES))
        results=list((root/'ablations').glob('*/*/result.json'));self.assertEqual(len(results),len(ds.FAMILIES))
        self.assertIn('unique_regressions',ds.read(results[0]))


if __name__=='__main__':unittest.main()
