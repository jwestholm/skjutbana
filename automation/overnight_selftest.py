"""Tests for conditional metrics, shadow isolation and physical audit provenance."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from automation.conditional_ranking import metrics
from automation.physical_session_audit import pool_summary,audit
from src.engine.ai.canonical_challenger import CanonicalChallenger,digest,transform_matrix
from src.engine.ai.training_v223.model import RankModelV223
from src.engine.ai.training_v223.schema import CandidateTrainingRow,ShotTrainingRecord

class Tests(unittest.TestCase):
    def test_conditional_denominator_and_unavailable(self):
        def r(s,xs):
            v=ShotTrainingRecord('session',s,'synthetic',0,0,0,[CandidateTrainingRow(str(i),x,0,{}) for i,x in enumerate(xs)]);v.finalize_labels();return v
        result=metrics([r('1',[100,1]),r('2',[100,200])],[[0,1],[0,1]])['tolerances']['5']
        self.assertEqual(result['oracle'],1);self.assertEqual(result['conditional_top3'],1);self.assertEqual(result['conditional_mrr'],.5)
        self.assertEqual(result['conditional_top1'],0);self.assertEqual(pool_summary(None,None)['availability'],'UNAVAILABLE')
        with self.assertRaises(ValueError):metrics([r('1',[1,100])],[[0,0]])
    def test_transform_ties(self):
        np.testing.assert_array_equal(transform_matrix([[1,3],[1,1],[3,2]],'within_shot_percentile'),np.array([[0,2/3],[0,0],[2/3,1/3]],dtype=np.float32))
    def test_shadow_hashes_and_no_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);model=RankModelV223('linear',('area',),np.zeros(1),np.ones(1),{'w':np.ones(1),'b':np.zeros(1)})
            model.save(p/'model');m={'mode':'SHADOW','status':'OFFLINE_CHALLENGER','model_directory':'model','transform':'identity','hashes':{f:digest(p/'model'/f) for f in ['model.json','model.npz']}}
            (p/'manifest.json').write_text(json.dumps(m));ranker=CanonicalChallenger(p/'manifest.json');c=[{'camera_x':1,'camera_y':2,'area':3},{'camera_x':2,'camera_y':3,'area':7}];before=copy.deepcopy(c)
            ranked=ranker.rank(c);self.assertEqual(ranked['order'],[1,0]);self.assertEqual(before,c);self.assertNotIn('apply',ranked);self.assertEqual(ranker.rank([])['order'],[])
            (p/'model/model.json').write_text('{}')
            with self.assertRaises(ValueError):CanonicalChallenger(p/'manifest.json')
    def test_nonphysical_requires_explicit_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);sp=p/'shots/shot_00000006';sp.mkdir(parents=True)
            (sp/'trace.json').write_text(json.dumps({'shot_id':6,'peak_ts':1,'stages':[],'outcome':{}}))
            self.assertEqual(audit(p)['shots'][0]['physical_state'],'unresolved')
            self.assertEqual(audit(p,{'6':{'state':'NO_PHYSICAL_SHOT','label_shot_id':None}})['shots'][0]['physical_state'],'NO_PHYSICAL_SHOT')

    def test_worker_trace_transport_is_independent(self):
        from types import SimpleNamespace
        from src.engine.shot_async_v2224 import AsyncDetectorV2224, DetectorJobResultV2224
        pipeline={'filtered_candidates':[{'camera_x':4,'camera_y':5}]}
        result=DetectorJobResultV2224(3,2,1,0,0,0,[],{},{},0,0,0,trace_pipeline=pipeline)
        scanner=SimpleNamespace(physical_trace_capture_enabled=True)
        AsyncDetectorV2224.apply_result(scanner,result)
        self.assertEqual(scanner.last_trace_pipeline_shot_id,3)
        pipeline['filtered_candidates'][0]['camera_x']=99
        self.assertEqual(scanner.last_trace_pipeline['filtered_candidates'][0]['camera_x'],4)

    def test_trace_miss_does_not_borrow_previous_hit(self):
        from types import SimpleNamespace
        from src.engine.physical_trace import PhysicalTraceRecorder
        from automation.physical_trace_selftest import TraceTests
        with tempfile.TemporaryDirectory() as d:
            recorder=PhysicalTraceRecorder(Path(d),enabled=True);scanner=TraceTests().scanner()
            event=SimpleNamespace(shot_id=2,peak_ts=1,state='missed',emitted=False,confidence=0,note='timeout')
            scanner.last_event_debug={'shot_id':1,'detector_e2e_ms':100}
            recorder.start_from_scanner(scanner,event);recorder.finish(2,scanner,event);recorder.flush()
            outcome=json.loads((Path(d)/'shots/shot_00000002/trace.json').read_text())['outcome']
            self.assertIsNone(outcome['final_camera_xy']);self.assertIsNone(outcome['detector_e2e_latency_ms'])

    def test_health_rejects_missing_artifacts(self):
        from automation.physical_test import health
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'shots/shot_00000001';p.mkdir(parents=True)
            self.assertFalse(health(root)['healthy'])
            (p/'trace.json').write_text(json.dumps({'completeness':{'trace_complete':True},'peak_ts':1,'frames':[{'kind':'post','path':'missing.npy','shape':[2,2],'dtype':'uint8'}]}))
            self.assertFalse(health(root)['healthy'])

if __name__=='__main__':unittest.main()
