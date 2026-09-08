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

    def test_confirmation_shadow_is_frozen_and_non_authoritative(self):
        from src.engine.ai.confirmation_selection_shadow import CONFIG,CONFIG_HASH,select
        import hashlib
        encoded=json.dumps(CONFIG,sort_keys=True,separators=(',',':'),allow_nan=False)
        self.assertEqual(CONFIG_HASH,hashlib.sha256(encoded.encode()).hexdigest())
        candidates=[{'camera_x':1,'camera_y':2,'score':40,'v2225_confirm_center_abs':2,'v2225_confirm_darkening':1,'v2225_confirm_compact':.2},
                    {'camera_x':8,'camera_y':9,'score':1,'v2225_confirm_center_abs':4,'v2225_confirm_darkening':2,'v2225_confirm_compact':.1}]
        before=copy.deepcopy(candidates);result=select(candidates)
        self.assertEqual(candidates,before);self.assertEqual(result['selected']['camera_x'],8)
        self.assertEqual(select(candidates),select(copy.deepcopy(candidates)))
        self.assertEqual(select([{'camera_x':1,'camera_y':2}])['status'],'UNAVAILABLE')

    def test_trace_keeps_three_selectors_independent(self):
        from src.engine.physical_trace import selector_snapshot
        trace={'decision_input':{'deterministic_selection':{'camera_x':10,'camera_y':11}},
               'outcome':{'matched_track_id':7},
               'stages':[{'local_confirmation':{'candidates':[{'camera_x':20,'camera_y':21,'v2225_confirm_center_abs':3,'v2225_confirm_darkening':2,'v2225_confirm_compact':.1}]}}],
               'canonical_challenger':{'mode':'SHADOW','status':'OFFLINE_CHALLENGER','order':[0],'candidates':[{'input_index':0,'camera_x':30,'camera_y':31}]}}
        selectors=selector_snapshot(trace)
        self.assertEqual(set(selectors),{'CURRENT_DETERMINISTIC','CONFIRMATION_SELECTION_SHADOW','CANONICAL_AI_SHADOW'})
        self.assertEqual(selectors['CURRENT_DETERMINISTIC']['selected']['camera_x'],10)
        self.assertEqual(selectors['CONFIRMATION_SELECTION_SHADOW']['selected']['camera_x'],20)
        self.assertEqual(selectors['CANONICAL_AI_SHADOW']['candidates'][0]['camera_x'],30)

    def test_finalized_physical_trace_persists_three_selectors(self):
        from src.engine.physical_trace import PhysicalTraceRecorder
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); recorder=PhysicalTraceRecorder(root,enabled=True)
            trace={'shot_id':1,'stages':[{'local_confirmation':{'candidates':[{'camera_x':2,'camera_y':3,'v2225_confirm_center_abs':2}]}}],
                   'decision_input':{'deterministic_selection':{'camera_x':4,'camera_y':5}},
                   'outcome':{'matched_track_id':9},'frames':[]}
            active={'expected_counts':{'pre':0,'post':0,'evidence':0},'persisted_counts':{'pre':0,'post':0,'evidence':0},
                    'errors':[],'post_limit_drops':0,'finished':True,'trace':trace,'settings':{},'directory':root/'shots/shot_00000001',
                    'writer_errors':0,'queue_drops':0}
            self.assertTrue(recorder._finalize_trace(1,active,timed_out=False))
            saved=json.loads((root/'shots/shot_00000001/trace.json').read_text())
            self.assertEqual(set(saved['selectors']),{'CURRENT_DETERMINISTIC','CONFIRMATION_SELECTION_SHADOW','CANONICAL_AI_SHADOW'})
            self.assertEqual(saved['outcome']['matched_track_id'],9)
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

    def test_audio_diagnostics_are_bound_to_each_trigger(self):
        from unittest.mock import patch
        from src.engine.audio.audio_peak_detector import AudioPeakDetector
        from src.engine.shot_track_v2226 import _install_audio_telemetry_patch
        _install_audio_telemetry_patch();detector=AudioPeakDetector();detector.min_abs_peak=.1
        samples=np.zeros(256,dtype=np.int16);samples[128]=30000
        with patch('src.engine.shot_track_v2226.time.time',return_value=100.0):detector._process_chunk(samples.tobytes())
        first=detector.get_latest_event();frozen=copy.deepcopy(first.diagnostics)
        with patch('src.engine.shot_track_v2226.time.time',return_value=100.10066):detector._process_chunk(samples.tobytes())
        second=detector.get_latest_event()
        self.assertEqual(len(detector._events),2)
        self.assertAlmostEqual(second.timestamp-first.timestamp,.10066)
        self.assertEqual(first.diagnostics,frozen)
        self.assertAlmostEqual(second.diagnostics['previous_peak_ts'],first.timestamp)
        self.assertEqual(second.diagnostics['cooldown_s'],.08)

    def test_helper_prepare_restore_preserves_unrelated_settings(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import automation.physical_test as helper
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);settings=root/'content/ai/settings.json';settings.parent.mkdir(parents=True)
            settings.write_text(json.dumps({'mode':'advisory','user_setting':4}))
            model=RankModelV223('linear',('area',),np.zeros(1),np.ones(1),{'w':np.ones(1),'b':np.zeros(1)})
            model.save(root/'model');manifest=root/'challenger.json'
            manifest.write_text(json.dumps({'mode':'SHADOW','status':'OFFLINE_CHALLENGER','model_directory':'model','transform':'identity','hashes':{f:digest(root/'model'/f) for f in ['model.json','model.npz']}}))
            active=root/'evaluation_runs/active.json'
            with patch.multiple(helper,ROOT=root,SETTINGS=settings,ACTIVE=active), patch('socket.create_connection',side_effect=OSError):
                helper.start(SimpleNamespace(challenger=manifest,prepare_only=True))
                state=json.loads(active.read_text());self.assertTrue(Path(state['root']).is_dir())
                changed=json.loads(settings.read_text());self.assertEqual(changed['mode'],'advisory')
                changed['user_setting']=9;settings.write_text(json.dumps(changed))
                with patch('sys.argv',['physical_test','stop']):helper.main()
                self.assertEqual(json.loads(settings.read_text()),{'mode':'advisory','user_setting':9})

    def test_decision_snapshot_and_shadow_do_not_mutate_track(self):
        from types import SimpleNamespace
        from src.engine.physical_trace import PhysicalTraceRecorder
        from automation.physical_trace_selftest import TraceTests
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);model=RankModelV223('linear',('area',),np.zeros(1),np.ones(1),{'w':np.ones(1),'b':np.zeros(1)})
            model.save(p/'model');manifest=p/'challenger.json'
            manifest.write_text(json.dumps({'mode':'SHADOW','status':'OFFLINE_CHALLENGER','model_directory':'model','transform':'identity','hashes':{f:digest(p/'model'/f) for f in ['model.json','model.npz']}}))
            scanner=TraceTests().scanner();event=SimpleNamespace(shot_id=1,peak_ts=1,state='matched',emitted=True)
            track=SimpleNamespace(camera_x=2,camera_y=3,best_score=4,track_id=1)
            recorder=PhysicalTraceRecorder(p/'traces',enabled=True)
            recorder.capture_decision(scanner,track,event,{'canonical_challenger_manifest':str(manifest)})
            track.camera_x=99;recorder.finish(1,scanner,event);recorder.flush()
            trace=json.loads((p/'traces/shots/shot_00000001/trace.json').read_text())
            self.assertEqual(trace['decision_input']['deterministic_selection']['camera_x'],2)
            self.assertEqual(trace['canonical_challenger']['mode'],'SHADOW')
            self.assertEqual(trace['canonical_challenger']['evaluation_timing'],'post_decision_at_trace_finalization')
            self.assertEqual(track.camera_x,99)

if __name__=='__main__':unittest.main()
