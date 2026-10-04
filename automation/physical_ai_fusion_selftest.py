"""Small synthetic plumbing tests, never physical validation."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from automation import physical_ai_fusion as f,physical_ai_dataset as ds,physical_ai_lab as lab
from automation.physical_ai_lab_selftest import fixture
from src.engine.ai.canonical_challenger import transform_matrix
from src.engine.ai.training_v223.model import RankModelV223


def proposal(event,train,xy=(10.,10.)):
    return dict(event=event['event'],held_session=event['session'],source='VISION',camera_xy=list(xy),
                training_sessions=train,vision_training_sessions=train,reranker_training_sessions=train,
                model_sha256='base',reranker_model_sha256='reranker')


def dataset(root):
    data,x=fixture(root);events=[];rows=[]
    for e in data['events']:
        fold=next(z for z in data['folds'] if e['session'] in z['held'])
        pp=[proposal(e,fold['train'],(20.+i,20.)) for i in range(3)]
        item=f.union_event(e,pp);item['indices']=list(range(len(rows),len(rows)+6))
        rows.extend(x[e['indices']]);rows.extend(np.full((3,len(ds.FEATURE_NAMES)),999,np.float32));events.append(item)
    np.save(root/'features.npy',np.asarray(rows,np.float32))
    data.update(events=events,feature_sha256=ds.digest(root/'features.npy'),fusion=dict(revision=f.REVISION))
    data.pop('dataset_hash');data['dataset_hash']=ds.canonical_hash(data);ds.atomic_json(root/'dataset.json',data)
    return data,np.asarray(rows,np.float32)


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.data,self.x=dataset(self.root/'data')
    def tearDown(self):self.temp.cleanup()
    def test_s03_refused(self):
        e=copy.deepcopy(self.data['events'][0]);p=e['vision_top3'][0];p['training_sessions']=['S03']
        with self.assertRaises(PermissionError):f.validate_proposal(p,e)
        with self.assertRaises(PermissionError):lab.location(lab.ROOT,'S03')
    def test_oof_lineage_accepted(self):
        e=self.data['events'][0];f.validate_proposal(e['vision_top3'][0],e)
    def test_own_session_lineage_refused(self):
        e=self.data['events'][0];p=copy.deepcopy(e['vision_top3'][0]);p['reranker_training_sessions']=[e['session']]
        with self.assertRaises(ValueError):f.validate_proposal(p,e)
    def test_outer_leakage_filtered(self):
        fold=self.data['folds'][0];before=[e for e in self.data['events'] if e['session'] in fold['train']]
        after=f.training_filter(before,fold)
        self.assertTrue(all(len(e['indices'])==e['classic_count'] for e in after))
        with self.assertRaises(ValueError):f.training_filter(self.data['events'],fold)
    def test_genuinely_nested_training_row_accepted(self):
        e=self.data['events'][0];p=proposal(e,['D02'])
        self.assertTrue(f.safe_training_proposal(p,e,dict(train=[e['session'],'D02'],held=['S02'],dev=['S01'])))
    def test_union_exact_and_immutable(self):
        e=dict(self.data['events'][0]);e['candidates']=e['candidates'][:3];old=copy.deepcopy(e)
        pp=[proposal(e,['S02'],e['candidates'][0]['xy']),proposal(e,['S02'],(5.,7.)),proposal(e,['S02'],(5.,7.))];saved=copy.deepcopy(pp)
        result=f.union_event(e,pp)
        self.assertEqual(e,old);self.assertEqual(pp,saved);self.assertEqual(len(result['candidates']),4)
        self.assertEqual(result['candidates'][:3],old['candidates'])
    def test_no_cap_inflation(self):
        e=self.data['events'][0]
        with self.assertRaises(ValueError):f.union_event(e,e['vision_top3']*2)
    def test_selected_metric_not_oracle(self):
        pred={e['event']:e['candidates'][1]['xy'] for e in self.data['events']};m=ds.evaluate(self.data['events'],pred)
        self.assertEqual(m['oracle']['5'],6);self.assertEqual(m['selected']['5'],0)
    def test_controls_unknown(self):
        e=copy.deepcopy(self.data['events'][0]);e.update(truth='NO_IMPACT',gt_xy=None)
        self.assertEqual(ds.evaluate([e],{e['event']:[1,1]})['control_false_emissions'],1)
        m=ds.evaluate([e],{e['event']:None});self.assertEqual(m['unknown'],1);self.assertEqual(m['validated_control_rejections'],0)
    def test_train_only_scaler_mining_determinism_checkpoint(self):
        cfg=dict(f.config_for(1,0),hidden=4,epochs=2)
        one=self.root/'one';two=self.root/'two';one.mkdir();two.mkdir()
        result=f.train_trial(one,self.data,self.x,cfg,lambda:False)
        repeated=f.train_trial(one,self.data,self.x,cfg,lambda:False)
        independent=f.train_trial(two,self.data,self.x,cfg,lambda:False)
        self.assertEqual(result['predictions'],repeated['predictions']);self.assertEqual(result['predictions'],independent['predictions'])
        fold=self.data['folds'][0];train=f.training_filter([e for e in self.data['events'] if e['session'] in fold['train']],fold)
        indices=[i for e in train for i in e['indices']];_,keep=lab.columns(cfg)
        expected=transform_matrix(self.x[:,keep],'signed_log')[indices].mean(axis=0)
        model=RankModelV223.load(one/'fold_00');np.testing.assert_array_equal(model.mean,expected)
        # The same restricted sample list is the sole input to scaler AND miner.
        self.assertEqual(result['folds'][0]['training_rows'],{e['event']:len(e['indices']) for e in train})
    def test_resume_status_and_history(self):
        out=self.root/'run';lab.run(out,self.root/'data',hours=.02,max_trials=1,adapter=f)
        h=ds.digest(out/'trials/trial_000000/result.json');old=ds.read(out/'champion.json')['history']
        lab.run(out,None,hours=None,max_trials=2,resume=True,adapter=f)
        self.assertEqual(h,ds.digest(out/'trials/trial_000000/result.json'))
        status=f.status(out);self.assertEqual(status['trials'],2);self.assertEqual(status['history'][:len(old)],old)
        self.assertIn('oracle',status['metrics']);self.assertFalse(status['live_authority'])
    def test_output_restricted(self):
        with self.assertRaises(ValueError):ds.output_root('/tmp/fusion_output')
    def test_seed_and_threshold_fixed(self):
        self.assertEqual(f.config_for(1,2),f.config_for(1,2));self.assertEqual(f.config_for(1,2)['threshold'],.5)


if __name__=='__main__':unittest.main()
