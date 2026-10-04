"""Representation/mining tests; synthetic fixtures are not physical validation."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from automation import physical_ai_dataset as ds,physical_ai_lab as lab
from automation import physical_ai_vision as v,physical_ai_vision_peaks as p,physical_ai_vision_repr as r
from automation import physical_ai_vision_selftest as fixtures
from src.engine.ai.training_v223.heatmap_model_v2236 import init_heatmap_model


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.VisionTests();self.f.setUp()
        self.data=self.f.data;self.cache=self.f.cache;self.root=self.f.root
        self.fold=self.data['folds'][0];self.cfg=dict(r.config_for(2381,1),hidden=2,epochs=2,refine_epochs=2)
    def tearDown(self):self.f.tearDown()
    def inputs(self):
        samples,norm=v.training_samples(self.data,self.cache,self.fold,self.cfg)
        model=init_heatmap_model('spatial_conv',hidden=2,seed=7)
        return model,samples,norm

    def test_native_no_copy_or_change(self):
        a=np.arange(8*32*32,dtype=np.uint8).reshape(8,32,32)
        self.assertIs(v.representation_maps(a),a)

    def test_context_retains_native_high_resolution(self):
        a=np.zeros((8,64,64),np.uint8);a[:,30,30]=255;before=a.copy()
        b=v.representation_maps(a,'context9')
        np.testing.assert_array_equal(a,before);np.testing.assert_array_equal(a[[2,4,5,7]],b[[2,4,5,7]])
        self.assertGreater(np.count_nonzero(b[0]),1);self.assertLess(b[0].max(),255)
        np.testing.assert_array_equal(b,v.representation_maps(a,'context9'))

    def test_no_truth_parameter_in_representation(self):
        import inspect
        self.assertEqual(list(inspect.signature(v.representation_maps).parameters),['maps','representation'])
        with self.assertRaises(ValueError):v.representation_maps(np.zeros((8,8,8),np.uint8),'gt')

    def test_mining_reads_train_only(self):
        model,samples,norm=self.inputs()
        with patch.object(v,'event_maps',wraps=v.event_maps) as load:
            augmented,journal=v.mine_full_map_samples(model,self.data,self.cache,self.fold,self.cfg,norm,samples)
        self.assertTrue(journal);self.assertTrue(all(c.args[1]['session'] in self.fold['train'] for c in load.call_args_list))
        for before,after,j in zip(samples,augmented,journal):
            self.assertLessEqual(j['count'],64);self.assertEqual(len(after.patches),len(before.patches)+j['count'])
            self.assertTrue(np.all(after.distances[len(before.distances):]>42))
            self.assertFalse(set(j['training_sessions'])&set(j['held_sessions']))

    def test_mining_deterministic_and_input_immutable(self):
        model,samples,norm=self.inputs();before=[s.patches.copy() for s in samples]
        a,ja=v.mine_full_map_samples(model,self.data,self.cache,self.fold,self.cfg,norm,samples)
        b,jb=v.mine_full_map_samples(model,self.data,self.cache,self.fold,self.cfg,norm,samples)
        self.assertEqual(ja,jb)
        for aa,bb,s,old in zip(a,b,samples,before):
            np.testing.assert_array_equal(aa.patches,bb.patches);np.testing.assert_array_equal(s.patches,old)

    def test_held_truth_and_pixels_cannot_change_mining_or_scaler(self):
        model,samples,norm=self.inputs();_,a=v.mine_full_map_samples(model,self.data,self.cache,self.fold,self.cfg,norm,samples)
        for e in self.data['events']:
            if e['session'] not in self.fold['train']:
                e['gt_xy']=[999,999];np.save(self.cache/self.data['vision_rows'][e['event']]['path'],np.zeros((8,64,64),np.uint8))
        ss,nn=v.training_samples(self.data,self.cache,self.fold,self.cfg)
        self.assertEqual(norm,nn)
        _,b=v.mine_full_map_samples(model,self.data,self.cache,self.fold,self.cfg,nn,ss);self.assertEqual(a,b)

    def test_unknown_skipped_before_loading(self):
        model,samples,norm=self.inputs();self.data['events'].append(dict(event='missing',session=self.fold['train'][0],truth='UNKNOWN'))
        _,journal=v.mine_full_map_samples(model,self.data,self.cache,self.fold,self.cfg,norm,samples)
        self.assertNotIn('missing',[j['event'] for j in journal])

    def test_native_control_only_negative(self):
        e=next(e for e in self.data['events'] if e['session'] in self.fold['train']);e.update(truth='NO_IMPACT',gt_xy=None)
        model,samples,norm=self.inputs();aa,j=v.mine_full_map_samples(model,self.data,self.cache,self.fold,self.cfg,norm,samples)
        s=next(s for s in aa if s.shot_id==e['event']);self.assertTrue(np.isinf(s.distances).all())

    def test_mining_overlap_and_bound_guards(self):
        model,samples,norm=self.inputs()
        for fold,cfg in ((dict(self.fold,train=self.fold['train']+self.fold['held']),self.cfg),(self.fold,dict(self.cfg,full_map_negatives=65))):
            with self.assertRaises(ValueError):v.mine_full_map_samples(model,self.data,self.cache,fold,cfg,norm,samples)

    def test_s03_refused_before_read(self):
        for path in ('S03','/data/skjutbana/S03/cache'):
            with self.assertRaises(PermissionError):r.load(path)
        model,samples,norm=self.inputs()
        with self.assertRaises(PermissionError):
            v.mine_full_map_samples(model,self.data,self.cache,dict(self.fold,held=['S03']),self.cfg,norm,samples)

    def test_output_root_guard(self):
        with self.assertRaises(ValueError):r.main(['vision-repr-status','--experiment','x','--root',str(self.root)])
        with self.assertRaises(PermissionError):r.main(['vision-repr-status','--experiment','S03'])

    def test_fixed_paired_configs(self):
        a,b=r.config_for(2381,0),r.config_for(2381,1)
        self.assertEqual({k:v for k,v in a.items() if k!='full_map_negatives'},{k:v for k,v in b.items() if k!='full_map_negatives'})
        for i in range(20):
            c=r.config_for(2381,i);self.assertEqual(c,r.config_for(2381,i));self.assertEqual(c['budgets'],[1,3,8]);self.assertLessEqual(c['full_map_negatives'],64)

    def test_resume_restores_warmup_before_mining(self):
        model,samples,norm=self.inputs();path=self.root/'fit';path.mkdir()
        kw=dict(seed=7,identity={'fixture':'exact'},stop=None)
        a=v.fit_proposal_model(model,self.data,self.cache,self.fold,self.cfg,norm,samples,path,**kw)
        arrays={k:x.copy() for k,x in model.arrays.items()}
        model2=init_heatmap_model('spatial_conv',hidden=2,seed=7)
        b=v.fit_proposal_model(model2,self.data,self.cache,self.fold,self.cfg,norm,samples,path,**kw)
        self.assertEqual(a,b)
        for k in arrays:np.testing.assert_array_equal(arrays[k],model2.arrays[k])

    def test_interrupted_refinement_exact_resume(self):
        model,samples,norm=self.inputs();path=self.root/'interrupted';path.mkdir()
        calls=[0]
        def stop():calls[0]+=1;return calls[0]>2+len(samples)+1
        with self.assertRaises(InterruptedError):
            v.fit_proposal_model(model,self.data,self.cache,self.fold,self.cfg,norm,samples,path,seed=7,identity={},stop=stop)
        restored=init_heatmap_model('spatial_conv',hidden=2,seed=7)
        v.fit_proposal_model(restored,self.data,self.cache,self.fold,self.cfg,norm,samples,path,seed=7,identity={})
        other=self.root/'full';other.mkdir();full=init_heatmap_model('spatial_conv',hidden=2,seed=7)
        v.fit_proposal_model(full,self.data,self.cache,self.fold,self.cfg,norm,samples,other,seed=7,identity={})
        for k in full.arrays:np.testing.assert_array_equal(full.arrays[k],restored.arrays[k])

    def test_pool_metrics_union_exact_coordinates_no_mutation(self):
        event=dict(event='KOMP:1',session='KOMP',truth='HIT',gt_xy=[1.,2.],candidates=[dict(xy=[10,20])])
        before=copy.deepcopy(event);shard=dict(held={'KOMP:1':{p.ORDERS[1]:[dict(camera_xy=[1,2])]}})
        m=r.pool_metrics(dict(events=[event],sessions=['KOMP']),[shard]);self.assertEqual(m['combined']['5'],1);self.assertEqual(event,before)
        shard['held']['KOMP:1'][p.ORDERS[1]]*=33
        with self.assertRaises(ValueError):r.pool_metrics(dict(events=[event],sessions=['KOMP']),[shard])

    def test_protected_seven_including_new_save(self):
        self.assertEqual(set(r.PROTECTED),set(p.PROTECTED)|{'POST_FIX:3'})

    def test_fixed_pool_final_caps(self):
        rng=np.random.default_rng(4);maps=rng.integers(0,255,(8,64,64),dtype=np.uint8)
        for rep in ('native','context9'):
            transformed=v.representation_maps(maps,rep)
            pool=p.extract_pool(transformed[2].astype(np.float32),maps,stride=1,origin=[100,200],nms_radius_px=6,order=p.ORDERS[1])
            output,_=p.select(pool,np.arange(len(pool)),order=p.ORDERS[1],nms_radius_px=6)
            self.assertLessEqual(len(pool),32);self.assertEqual(len(output[:3]),3)
            self.assertTrue(all(q['camera_xy'] in [x['camera_xy'] for x in pool] for q in output))
            self.assertTrue(all(q['nms_rank']==q['pool_rank']<=32 for q in output))

    def test_actual_nested_runner_resume_champion_preservation(self):
        # Extend existing synthetic fixture to five allowed session identities.
        original=ds.read(self.root/'data/dataset.json');x=np.load(self.root/'data/features.npy')
        for session in ('D02','POST_FIX'):
            for j in (1,2):
                e=copy.deepcopy(original['events'][j-1]);start=len(x);x=np.concatenate([x,x[:3]])
                e.update(event=f'{session}:{j}',session=session,source_session='fixture_'+session,indices=list(range(start,start+3)))
                original['events'].append(e)
                row=copy.deepcopy(self.f.manifest['events'][j-1]);old=self.cache/row['path']
                row.update(event=e['event'],session=session,path=e['event'].replace(':','_')+'.npy')
                np.save(self.cache/row['path'],np.load(old));row['sha256']=ds.digest(self.cache/row['path']);self.f.manifest['events'].append(row)
        np.save(self.root/'data/features.npy',x);original['feature_sha256']=ds.digest(self.root/'data/features.npy')
        original['sessions']=sorted({e['session'] for e in original['events']});original['folds']=ds.folds(original['sessions'])
        original.pop('dataset_hash');original['dataset_hash']=ds.canonical_hash(original);ds.atomic_json(self.root/'data/dataset.json',original)
        self.f.manifest['dataset_hash']=original['dataset_hash'];self.f.save_manifest();data,cache=v.load(self.cache)
        classic_hash=ds.digest(self.root/'data/dataset.json')
        base=self.root/'base';trial=base/'trials/trial_000000';trial.mkdir(parents=True)
        cfg=dict(self.cfg,epochs=1,refine_epochs=0,full_map_negatives=0)
        first=v.train_trial(trial,data,cache,cfg,lambda:False);first.update(config=cfg,dataset_hash=data['dataset_hash'])
        ds.atomic_json(trial/'result.json',first);ds.atomic_json(base/'manifest.json',dict(dataset=str(cache)))
        with patch.object(ds,'output_root',side_effect=lambda x:Path(x)):
            p.prepare(base,'trial_000000',self.root/'peaks')
        pdata,shards=p.load(self.root/'peaks');rr=self.root/'baseline';rr.mkdir()
        rcfg=dict(p.config_for(2371,1),epochs=1)
        reference=p.train_trial(rr,pdata,shards,rcfg,lambda:False)
        reference.update(status='COMPLETE',config=rcfg,dataset_hash=pdata['dataset_hash']);ds.atomic_json(rr/'result.json',reference)
        with patch.object(ds,'output_root',side_effect=lambda x:Path(x)):
            r.prepare(cache,self.root/'peaks',rr/'result.json',self.root/'input')
        cfg=dict(cfg,full_map_negatives=8,refine_epochs=1)
        root=self.root/'run';sampler=v.training_samples;seen=[]
        def spy(d,c,fold,config):
            seen.append(copy.deepcopy(fold));return sampler(d,c,fold,config)
        with patch.object(ds,'output_root',side_effect=lambda x:Path(x)),patch.object(r,'config_for',return_value=cfg),patch.object(v,'training_samples',side_effect=spy),patch.object(r,'comparison',return_value=dict(accepted=True,reason='storage fixture')):
            lab.run(root,self.root/'input',hours=.05,max_trials=1,adapter=r)
            self.assertEqual(len(seen),20)
            self.assertTrue(all(not set(f['train'])&set(f['held']+f['dev']) for f in seen))
            saved={q:q.read_bytes() for q in (root/'champions').glob('*.json')};self.assertTrue(saved)
            lab.run(root,None,hours=None,max_trials=2,resume=True,adapter=r)
        self.assertEqual(ds.digest(self.root/'data/dataset.json'),classic_hash)
        for q,b in saved.items():self.assertEqual(q.read_bytes(),b)
        result=ds.read(root/'trials/trial_000000/result.json')
        self.assertEqual(result['status'],'COMPLETE')
        for output in result['outputs'].values():
            self.assertLessEqual(len(output['proposals'][:3]),3)
            for q in output['proposals']:
                self.assertNotIn(q['held_session'],q['vision_training_sessions'])
                self.assertNotIn(q['held_session'],q['reranker_training_sessions'])
                self.assertEqual(q['mining_training_sessions'],q['vision_training_sessions'])


if __name__=='__main__':unittest.main()
