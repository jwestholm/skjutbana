"""Peak/nested-OOF safeguards; synthetic fixtures confer no physical accuracy."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from automation import physical_ai_dataset as ds,physical_ai_lab as lab,physical_ai_vision as v
from automation import physical_ai_vision_peaks as p
from automation import physical_ai_vision_selftest as fixtures
from src.engine.ai.training_v223.model import train_fusion_model


class PeakTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        rng=np.random.default_rng(9);self.score=rng.normal(size=(64,64)).astype(np.float32)
        self.maps=rng.integers(0,256,size=(8,64,64),dtype=np.uint8)
    def tearDown(self):self.tmp.cleanup()
    def pool(self,order=p.ORDERS[0]):
        return p.extract_pool(self.score,self.maps,stride=1,origin=[10,20],nms_radius_px=6,order=order)
    def event_shard(self):
        fold=dict(train=['D02','S01','S02'],dev=['POST_FIX'],held=['KOMP'])
        events=[];train={}
        for s in fold['train']:
            eid=s+':1';e=dict(event=eid,session=s,truth='HIT',gt_xy=[20,30]);events.append(e)
            modes={}
            for order in p.ORDERS:
                peaks=self.pool(order)
                for peak in peaks:peak.update(event=eid,held_session=s,training_sessions=[x for x in fold['train'] if x!=s],
                                              model_sha256='test',base_data_hash='test')
                modes[order]=peaks
            train[eid]=modes
        return dict(events=events),dict(fold=fold,train=train)

    def test_below_topk_vs_nms_suppressed(self):
        score=np.zeros((80,80),np.float32)
        for i in range(9):score[4+i*8,4]=20-i
        score[70,60]=1
        a=p.audit_peak(score,stride=1,origin=[0,0],nms_radius_px=6,gt_xy=[60,70])['nearest']
        self.assertEqual(a['raw_rank'],10);self.assertEqual(a['post_nms_rank'],10);self.assertEqual(a['failure_class'],'BELOW_TOPK')
        score[70,56]=2
        a=p.audit_peak(score,stride=1,origin=[0,0],nms_radius_px=6,gt_xy=[60,70])['nearest']
        self.assertEqual(a['failure_class'],'NMS_SUPPRESSED');self.assertEqual(a['suppressor']['camera_xy'],[56,70])
        self.assertTrue(a['raw_below_top8'])

    def test_nms_trace_matches_existing_output(self):
        for stride in (1,2,4):
            a=p.audit_peak(self.score,stride=stride,origin=[10,20],nms_radius_px=6,gt_xy=[15,25])
            b=v.proposals(self.score,stride=stride,origin=[10,20],nms_radius_px=6)
            self.assertEqual(a['top8_xy'],[q['camera_xy'] for q in b])

    def test_internal_pool_bound_both_orders(self):
        for order in p.ORDERS:self.assertEqual(len(self.pool(order)),32)

    def test_nms_before_pool_includes_deeper_raw_peaks(self):
        peaks=self.pool(p.ORDERS[1]);self.assertGreater(peaks[-1]['raw_rank'],32)
        self.assertEqual(len(peaks),32)

    def test_final_budget_and_no_new_xy(self):
        for order in p.ORDERS:
            peaks=self.pool(order);scores=np.arange(32,dtype=float)
            output,_=p.select(peaks,scores,order=order,nms_radius_px=6)
            self.assertLessEqual(len(output),8)
            self.assertTrue(all(q['camera_xy'] in [x['camera_xy'] for x in peaks] for q in output))
            for k in v.BUDGETS:self.assertLessEqual(len(output[:k]),k)

    def test_gt_and_metadata_do_not_enter_feature_vector(self):
        peaks=self.pool();a=p.feature_matrix(peaks)
        for q in peaks:q.update(gt_xy=[999,999],gt_distance=0,target_distance=0,game_score=10,model_id='other',session='other')
        b=p.feature_matrix(peaks);self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_extraction_deterministic_no_mutation(self):
        before=self.maps.copy();scores=self.score.copy();a=self.pool();b=self.pool()
        self.assertEqual(a,b);np.testing.assert_array_equal(before,self.maps);np.testing.assert_array_equal(scores,self.score)

    def test_nms_tie_order_deterministic(self):
        a=np.array([[0.,0.],[3.,0.],[8.,0.]])
        ids,supp=p.nms_trace(a,6);np.testing.assert_array_equal(ids,[0,2]);self.assertEqual(supp[1],0)

    def test_complete_nms_vs_bounded_prefix(self):
        pts,_=p.raw_peaks(self.score);full,_=p.nms_trace(pts,6);first,_=p.nms_trace(pts,6,32)
        np.testing.assert_array_equal(full[:32],first)

    def test_lineage_refuses_in_sample_and_outer_leak(self):
        data,shard=self.event_shard();e=data['events'][0];peak=shard['train'][e['event']][p.ORDERS[0]][0]
        p.validate_upstream(peak,e,shard['fold'],training=True)
        for sessions in (['D02','S01','S02'],['S01','KOMP'],['S01','POST_FIX']):
            with self.assertRaises(ValueError):p.validate_upstream(dict(peak,training_sessions=sessions),e,shard['fold'],training=True)

    def test_train_sampler_excludes_held_and_unknown(self):
        data,shard=self.event_shard();cfg=p.config_for(2371,0)
        data['events'].append(dict(event='KOMP:1',session='KOMP',truth='HIT',gt_xy=[0,0]))
        data['events'][1].update(truth='UNKNOWN',gt_xy=None)
        _,samples,ids=p.samples(data,shard,cfg)
        self.assertEqual(ids,['D02:1','S02:1']);self.assertEqual(len(samples),2)
        shard['train']['KOMP:1']=shard['train']['D02:1']
        with self.assertRaises(ValueError):p.samples(data,shard,cfg)

    def test_native_control_is_negative(self):
        data,shard=self.event_shard();data['events'][0].update(truth='NO_IMPACT',gt_xy=None)
        _,samples,_=p.samples(data,shard,p.config_for(2371,0))
        self.assertTrue(np.isinf(samples[0][1]).all())

    def test_scaler_mining_and_resume_train_only(self):
        data,shard=self.event_shard();cfg=p.config_for(2371,0)
        names,samples,ids=p.samples(data,shard,cfg)
        kw=dict(feature_names=names,hidden=8,epochs=4,seed=9,hard_negative_count=4)
        from src.engine.ai.training_v223 import evidence_model_v2235 as mining
        with patch.object(mining,'hard_negative_indices',wraps=mining.hard_negative_indices) as mine:
            a,_=train_fusion_model(samples,**kw)
        self.assertGreater(mine.call_count,0)
        np.testing.assert_array_equal(a.mean,np.concatenate([x for x,_ in samples]).mean(0))
        calls=[0]
        def stop():calls[0]+=1;return calls[0]>2
        with self.assertRaises(InterruptedError):train_fusion_model(samples,checkpoint=self.root/'ck.npz',stop=stop,**kw)
        b,_=train_fusion_model(samples,checkpoint=self.root/'ck.npz',**kw)
        for k in a.arrays:np.testing.assert_array_equal(a.arrays[k],b.arrays[k])

    def test_deterministic_ranking(self):
        peaks=self.pool();scores=np.arange(32)*.1
        self.assertEqual(p.select(peaks,scores,order=p.ORDERS[0],nms_radius_px=6),p.select(peaks,scores,order=p.ORDERS[0],nms_radius_px=6))

    def test_fixed_smoke_order_and_seed(self):
        a=p.config_for(2371,0);b=p.config_for(2371,1)
        self.assertEqual({k:v for k,v in a.items() if k!='order'},{k:v for k,v in b.items() if k!='order'})
        self.assertEqual(p.config_for(2371,7),p.config_for(2371,7))

    def test_protected_saves_explicit(self):
        self.assertEqual(set(p.PROTECTED),{'KOMP:1','KOMP:5','S02:5','S02:11','S02:12','S02:13'})
        data=dict(peak_manifest=dict(baseline_metrics={'3':dict(unique_saves=list(p.PROTECTED))}))
        self.assertFalse(p._compare(dict(unique_saves=list(p.PROTECTED)[1:]),{},'3',data)['accepted'])

    def test_union_correct_and_classic_unchanged(self):
        classic=[dict(xy=[10,20])];snapshot=copy.deepcopy(classic)
        xy,counts=v.union(classic,[dict(camera_xy=[10,20]),dict(camera_xy=[11,20])])
        self.assertEqual(classic,snapshot);self.assertEqual(xy,[[10,20],[11,20]])
        self.assertEqual(counts['exact_duplicates'],1)

    def test_s03_refusal(self):
        for value in ('S03','/tmp/session_S03_x','/data/skjutbana/S03/cache'):
            with self.assertRaises(PermissionError):p.load(value)
        data,shard=self.event_shard();e=data['events'][0];q=shard['train'][e['event']][p.ORDERS[0]][0]
        with self.assertRaises(PermissionError):p.validate_upstream(dict(q,training_sessions=['S03']),e,shard['fold'],training=True)

    def test_output_root_guard(self):
        with self.assertRaises(ValueError):p.main(['peak-status','--experiment','x','--root',str(self.root)])
        with self.assertRaises(PermissionError):p.main(['peak-status','--experiment','S03'])

    def test_real_nested_training_and_cache_resume(self):
        # Five allowed session NAMES with synthetic pixels, never physical files.
        f=fixtures.VisionTests();f.setUp()
        try:
            original=ds.read(f.root/'data/dataset.json');x=np.load(f.root/'data/features.npy')
            for s in ('D02','POST_FIX'):
                for j in (1,2):
                    e=copy.deepcopy(original['events'][j-1]);start=len(x);x=np.concatenate([x,x[:3]])
                    e.update(event=f'{s}:{j}',session=s,source_session='fixture_'+s,indices=list(range(start,start+3)))
                    original['events'].append(e)
                    row=copy.deepcopy(f.manifest['events'][j-1]);row.update(event=e['event'],session=s,path=e['event'].replace(':','_')+'.npy')
                    np.save(f.cache/row['path'],self.maps);row['sha256']=ds.digest(f.cache/row['path']);f.manifest['events'].append(row)
            np.save(f.root/'data/features.npy',x);original['feature_sha256']=ds.digest(f.root/'data/features.npy')
            original['sessions']=sorted({e['session'] for e in original['events']});original['folds']=ds.folds(original['sessions'])
            original.pop('dataset_hash');original['dataset_hash']=ds.canonical_hash(original);ds.atomic_json(f.root/'data/dataset.json',original)
            f.manifest['dataset_hash']=original['dataset_hash'];f.save_manifest();data,cache=v.load(f.cache)
            exp=f.root/'base';trial=exp/'trials/trial_000000';trial.mkdir(parents=True)
            cfg=dict(f.cfg,epochs=1);base=v.train_trial(trial,data,cache,cfg,lambda:False)
            base.update(config=cfg,dataset_hash=data['dataset_hash']);ds.atomic_json(trial/'result.json',base)
            ds.atomic_json(exp/'manifest.json',dict(dataset=str(f.cache)))
            dest=f.root/'nested';seen=[];sampler=v.training_samples
            def spy(d,c,fold,cfg):
                samples,norm=sampler(d,c,fold,cfg);seen.append(copy.deepcopy(fold))
                self.assertFalse(set(fold['train'])&set(fold['held']+fold['dev']))
                self.assertTrue(all(s.session_id in fold['train'] for s in samples))
                return samples,norm
            with patch.object(ds,'output_root',side_effect=lambda x:Path(x)),patch.object(v,'training_samples',side_effect=spy):
                p.prepare(exp,'trial_000000',dest)
            self.assertEqual(len(seen),15);self.assertTrue(all(len(x['train'])==2 for x in seen))
            with patch.object(ds,'output_root',side_effect=lambda x:Path(x)),patch.object(v,'training_samples',side_effect=AssertionError('no retrain')):
                p.prepare(exp,'trial_000000',dest,resume=True)
            d,shards=p.load(dest);run=f.root/'rerank';run.mkdir();rcfg=dict(p.config_for(2371,1),epochs=2)
            r=p.train_trial(run,d,shards,rcfg,lambda:False)
            # Missing protected IDs in this small fixture are not regressions.
            self.assertEqual(r['delta']['1']['protected_lost'],[])
            with patch.object(p,'train_fusion_model',side_effect=AssertionError('immutable completed folds')):
                self.assertEqual(r,p.train_trial(run,d,shards,rcfg,lambda:False))
            self.assertEqual(set(r['outputs']),{e['event'] for e in original['events']})
            for o in r['outputs'].values():
                for q in o['proposals']:
                    self.assertNotIn(q['held_session'],q['vision_training_sessions'])
                    self.assertNotIn(q['held_session'],q['reranker_training_sessions'])
            # Exercise actual runner resume; force comparison acceptance only in
            # synthetic fixture to test immutable champion storage, not accuracy.
            eroot=f.root/'runner'
            with patch.object(p,'config_for',return_value=rcfg),patch.object(p,'_compare',return_value=dict(accepted=True,reason='storage fixture')):
                lab.run(eroot,dest,hours=.03,max_trials=1,adapter=p)
                before={a:a.read_bytes() for a in (eroot/'champions').glob('*/*.json')}
                lab.run(eroot,None,hours=None,max_trials=2,resume=True,adapter=p)
            self.assertEqual(len(before),3)
            for path,value in before.items():self.assertEqual(path.read_bytes(),value)
        finally:f.tearDown()


if __name__=='__main__':unittest.main()
