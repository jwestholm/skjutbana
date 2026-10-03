"""Bounded Vision lab safeguards. Fixtures are not physical validation."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from automation import physical_ai_dataset as ds, physical_ai_lab as lab
from automation import physical_ai_vision as vision
from automation.physical_ai_lab_selftest import fixture
from src.engine.ai.training_v223.heatmap_model_v2236 import (
    init_heatmap_model, train_physical_proposals,
)


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        data,_=fixture(self.root/'data');self.cache=self.root/'maps';self.cache.mkdir()
        rng=np.random.default_rng(4);rows=[]
        for e in data['events']:
            name=e['event'].replace(':','_')+'.npy'
            maps=rng.integers(0,256,(8,64,64),dtype=np.uint8)
            np.save(self.cache/name,maps)
            rows.append(dict(event=e['event'],session=e['session'],path=name,sha256=ds.digest(self.cache/name),
                             origin=[0,0],shape=[64,64],source_hashes={},reference_frames=[],post_frames=[]))
        self.manifest=dict(schema='physical-vision-proposals-1',native_stride=1,dataset=str(self.root/'data'),
                           dataset_hash=data['dataset_hash'],events=rows)
        self.save_manifest();self.data,_=vision.load(self.cache)
        self.cfg=dict(vision.config_for(2361,0),hidden=2,epochs=2,stride=1,loss_kind='bce')

    def tearDown(self):self.tmp.cleanup()

    def save_manifest(self):
        value=copy.deepcopy(self.manifest);value['cache_hash']=ds.canonical_hash(value)
        ds.atomic_json(self.cache/'vision.json',value)

    def test_training_never_loads_held_or_dev(self):
        fold=self.data['folds'][0]
        with patch.object(vision,'event_maps',wraps=vision.event_maps) as load:
            samples,norm=vision.training_samples(self.data,self.cache,fold,self.cfg)
        self.assertEqual({s.session_id for s in samples},set(fold['train']))
        self.assertEqual(norm['training_sessions'],fold['train'])
        self.assertTrue(all(c.args[1]['session'] in fold['train'] for c in load.call_args_list))

    def test_scaler_unchanged_by_held_pixels_and_labels(self):
        fold=self.data['folds'][0]
        _,a=vision.training_samples(self.data,self.cache,fold,self.cfg)
        for e in self.data['events']:
            if e['session'] not in fold['train']:
                np.save(self.cache/self.data['vision_rows'][e['event']]['path'],np.zeros((8,64,64),np.uint8))
                e['gt_xy']=[999,999]
        _,b=vision.training_samples(self.data,self.cache,fold,self.cfg)
        self.assertEqual(a,b)

    def test_overlapping_fold_refused(self):
        fold=copy.deepcopy(self.data['folds'][0]);fold['train']+=fold['held']
        with self.assertRaises(ValueError):vision.training_samples(self.data,self.cache,fold,self.cfg)

    def test_unknown_not_trained_as_negative(self):
        data=copy.deepcopy(self.data);fold=data['folds'][0]
        e=next(e for e in data['events'] if e['session'] in fold['train'])
        unknown=dict(e,event='extra',truth='UNKNOWN',gt_xy=None)
        data['events'].append(unknown)  # No map: must be skipped before image access.
        samples,norm=vision.training_samples(data,self.cache,fold,self.cfg)
        self.assertNotIn('extra',norm['training_events'])
        self.assertNotIn('extra',[s.shot_id for s in samples])

    def test_explicit_no_impact_trained_as_negative(self):
        data=copy.deepcopy(self.data);fold=data['folds'][0]
        e=next(e for e in data['events'] if e['session'] in fold['train']);e.update(truth='NO_IMPACT',gt_xy=None)
        samples,_=vision.training_samples(data,self.cache,fold,self.cfg)
        sample=next(s for s in samples if s.shot_id==e['event'])
        self.assertTrue(np.isinf(sample.distances).all())

    def test_s03_cache_name_refused_before_read(self):
        with self.assertRaises(PermissionError):vision.load(self.root/'S03')

    def test_s03_session_refused_before_array(self):
        self.manifest['events'][0]['session']='S03';self.save_manifest()
        # Classic feature loading is permitted; the forbidden map is never opened.
        original=np.load
        def guarded(path,*args,**kwargs):
            self.assertNotEqual(Path(path),self.cache/self.manifest['events'][0]['path'])
            return original(path,*args,**kwargs)
        with patch.object(np,'load',side_effect=guarded),self.assertRaises(PermissionError):vision.load(self.cache)

    def test_s03_source_path_refused(self):
        self.manifest['events'][0]['source_hashes']={'/tmp/S03/never_open':'x'};self.save_manifest()
        with self.assertRaises(PermissionError):vision.load(self.cache)

    def test_s03_runner_name_refused(self):
        with self.assertRaises(PermissionError):vision.lab_main(['vision-status','--experiment','S03'])

    def test_outputs_must_be_under_data(self):
        with self.assertRaises(ValueError):vision.lab_main(['vision-status','--experiment','x','--root',str(self.root)])

    def test_cache_tamper_refused(self):
        np.save(self.cache/self.manifest['events'][0]['path'],np.zeros((8,64,64),np.uint8))
        with self.assertRaises(ValueError):vision.load(self.cache)

    def test_missing_event_refused(self):
        self.manifest['events'].pop();self.save_manifest()
        with self.assertRaises(ValueError):vision.load(self.cache)

    def test_legacy_downsampled_cache_refused(self):
        self.manifest['native_stride']=4;self.save_manifest()
        with self.assertRaises(ValueError):vision.load(self.cache)

    def test_own_peak_and_native_resolution(self):
        score=np.zeros((16,16),np.float32);score[3,7]=10
        p=vision.proposals(score,stride=1,origin=[101,202],nms_radius_px=6,limit=1)
        self.assertEqual(p[0]['camera_xy'],[108,205])
        self.assertEqual(p[0]['confidence'],10)
        # No GT or candidate bank argument exists in this inference API.
        p=vision.proposals(score,stride=4,origin=[101,202],nms_radius_px=6,limit=1)
        self.assertEqual(p[0]['camera_xy'],[130.5,215.5])

    def test_top_k_bounded_and_nested(self):
        score=np.random.default_rng(19).normal(size=(64,64)).astype(np.float32)
        pp=vision.proposals(score,stride=1,origin=[0,0],nms_radius_px=6)
        self.assertEqual(len(pp),8)
        for k in vision.BUDGETS:
            self.assertEqual(vision.proposals(score,stride=1,origin=[0,0],nms_radius_px=6,limit=k),pp[:k])
        with self.assertRaises(ValueError):vision.proposals(score,stride=1,origin=[0,0],nms_radius_px=6,limit=1000)

    def test_nonfinite_peak_refused(self):
        with self.assertRaises(ValueError):vision.proposals(np.full((3,3),np.nan),stride=1,origin=[0,0],nms_radius_px=6)

    def test_union_keeps_exact_nearby_identity_and_bank(self):
        classic=[dict(xy=[9.,9.])];before=copy.deepcopy(classic)
        pp=[dict(camera_xy=[9.,9.]),dict(camera_xy=[10.,10.]),dict(camera_xy=[100.,100.])]
        xy,counts=vision.union(classic,pp)
        self.assertEqual(xy,[[9.,9.],[10.,10.],[100.,100.]])
        self.assertEqual(counts,dict(added=2,exact_duplicates=1,near_duplicates_r3_5=1))
        self.assertEqual(classic,before)

    def outputs(self,events):
        return {e['event']:dict(proposals=[dict(camera_xy=[10.,10.],confidence=.2)],diagnostic={}) for e in events}

    def test_union_metrics_and_control_semantics(self):
        events=copy.deepcopy(self.data['events']);events[0]['candidates']=[]
        events[1].update(truth='NO_IMPACT',gt_xy=None);events[2].update(truth='UNKNOWN',gt_xy=None)
        m=vision.evaluate_union(events,self.outputs(events))['1']
        self.assertEqual(m['impacts'],4);self.assertEqual(m['classic_oracle']['5'],3)
        self.assertEqual(m['combined_oracle']['5'],4);self.assertEqual(m['unique_saves'],[events[0]['event']])
        self.assertEqual(len(m['controls']),1);self.assertEqual(len(m['unknown_truth']),1)
        self.assertEqual(m['controls'][0]['proposals'],1)

    def test_metrics_refuse_foreign_or_unbounded(self):
        with self.assertRaises(ValueError):vision.evaluate_union(self.data['events'],{})
        pp=self.outputs(self.data['events']);pp[self.data['events'][0]['event']]['proposals']*=9
        with self.assertRaises(ValueError):vision.evaluate_union(self.data['events'],pp)

    def test_oof_lineage_and_leakage_refusal(self):
        fold=self.data['folds'][0];e=next(e for e in self.data['events'] if e['session'] in fold['held'])
        p=dict(event=e['event'],source='VISION',held_session=e['session'],training_sessions=fold['train'],
               data_hash='cache',model_sha256='model',model_id='trial/fold',code={'sha':'test'})
        vision.validate_lineage(p,e,fold,'cache')
        for key,value in [('training_sessions',fold['train']+fold['held']),('event','foreign'),('model_sha256',''),('data_hash','wrong')]:
            with self.subTest(key=key),self.assertRaises(ValueError):vision.validate_lineage(dict(p,**{key:value}),e,fold,'cache')
        with self.assertRaises(PermissionError):vision.validate_lineage(dict(p,training_sessions=['S03']),e,fold,'cache')

    def test_deterministic_config(self):
        self.assertEqual(vision.config_for(1,5),vision.config_for(1,5))
        self.assertNotEqual(vision.config_for(1,5),vision.config_for(1,6))

    def test_resume_exact_adam_rng_and_no_mutation(self):
        samples,_=vision.training_samples(self.data,self.cache,self.data['folds'][0],self.cfg)
        before=[s.patches.copy() for s in samples];kwargs=dict(epochs=4,learning_rate=.003,l2=.001,seed=12,hard_count=8)
        a=init_heatmap_model('spatial_conv',hidden=2,seed=7)
        history=train_physical_proposals(a,samples,**kwargs)
        b=init_heatmap_model('spatial_conv',hidden=2,seed=7);calls=[0]
        def stop():calls[0]+=1;return calls[0]>2
        with self.assertRaises(InterruptedError):train_physical_proposals(b,samples,checkpoint=self.root/'ck.npz',stop=stop,**kwargs)
        resumed=init_heatmap_model('spatial_conv',hidden=2,seed=999)
        h=train_physical_proposals(resumed,samples,checkpoint=self.root/'ck.npz',**kwargs)
        self.assertEqual(history,h)
        for k in a.arrays:np.testing.assert_array_equal(a.arrays[k],resumed.arrays[k])
        for s,v in zip(samples,before):np.testing.assert_array_equal(s.patches,v)

    def test_checkpoint_identity_refused(self):
        samples,_=vision.training_samples(self.data,self.cache,self.data['folds'][0],self.cfg)
        model=init_heatmap_model('spatial_conv',hidden=2)
        kwargs=dict(epochs=1,learning_rate=.003,l2=.001,seed=12,checkpoint=self.root/'ck.npz')
        train_physical_proposals(model,samples,identity='a',**kwargs)
        with self.assertRaises(ValueError):train_physical_proposals(model,samples,identity='b',**kwargs)

    def test_fold_resume_and_reproduction(self):
        path=self.root/'trial';path.mkdir()
        before=copy.deepcopy(self.data['events'])
        result=vision.train_trial(path,self.data,self.cache,self.cfg,lambda:False)
        with patch.object(vision,'training_samples',side_effect=AssertionError('no retraining')):
            resumed=vision.train_trial(path,self.data,self.cache,self.cfg,lambda:False)
        self.assertEqual(result,resumed);self.assertEqual(self.data['events'],before)
        other=self.root/'other';other.mkdir()
        second=vision.train_trial(other,self.data,self.cache,self.cfg,lambda:False)
        self.assertEqual(result['metrics'],second['metrics'])
        for e in result['outputs']:
            self.assertEqual([p['camera_xy'] for p in result['outputs'][e]['proposals']],
                             [p['camera_xy'] for p in second['outputs'][e]['proposals']])

    def test_champion_history_preserved_runner_resume(self):
        path=self.root/'run'
        with patch.object(vision,'config_for',return_value=self.cfg):
            lab.run(path,self.cache,hours=.03,max_trials=1,adapter=vision)
            history={p:p.read_bytes() for p in (path/'champions').glob('*/*.json')}
            lab.run(path,None,hours=None,max_trials=2,resume=True,adapter=vision)
        self.assertEqual(len(history),3);self.assertEqual(len(lab.completed(path)),2)
        for p,value in history.items():self.assertEqual(p.read_bytes(),value)
        self.assertEqual(set(ds.read(path/'champion.json')['tracks']),{'1','3','8'})

    def test_patch_and_map_normalization_equivalent(self):
        _,norm=vision.training_samples(self.data,self.cache,self.data['folds'][0],self.cfg)
        maps=vision.event_maps(self.cache,next(iter(self.data['vision_rows'].values())),1)
        points=np.asarray([[10,10],[0,0],[63,63]])
        patches=vision.normalize(vision.extract_grid_patches(maps,points),norm,patches=True)
        model=init_heatmap_model('spatial_conv',hidden=2,seed=11)
        score=model.score_map(vision.normalize(maps,norm))
        np.testing.assert_allclose(model.score_patches(patches),score[points[:,1],points[:,0]],atol=1e-6)


if __name__=='__main__':unittest.main()
