"""Causal measurement regressions, including exact physical false-event timing."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src.engine.offline.causal_candidates import analyze_trace, classify, overlay_candidates

class Tests(unittest.TestCase):
    def test_exact_false_event_patterns(self):
        for sid,peak,decision,frame,next_peak,later in (
            (6,1788885626.4964387,1788885627.3162303,1788885627.174298,1788885627.9803684,1788885627.982724),
            (14,1788885752.6620774,1788885753.3337026,1788885753.2575214,1788885754.1525483,1788885754.43405)):
            wrong={'camera_x':100.,'camera_y':100.,'timestamp':frame}
            future={'camera_x':0.,'camera_y':0.,'timestamp':later}
            trace={'shot_id':sid,'peak_ts':peak,'decision_input':{'timestamp':decision,'retained_candidates':[wrong]},
                   'stages':[{'timestamp':later+.2,'candidate_pool_shot_id':sid+1,'candidates':[future]}]}
            before=copy.deepcopy(trace)
            a=analyze_trace(trace,next_peak,{'camera_x':0.,'camera_y':0.})
            self.assertFalse(a['causal_oracle']['hits']['42'])
            self.assertTrue(a['posthoc_all_observed_oracle']['hits']['5'])
            self.assertEqual(a['counts_candidate_versions']['CROSS_EVENT_OR_FUTURE'],1)
            self.assertEqual(len(list(overlay_candidates(trace,next_peak=next_peak))),1)
            self.assertEqual(len(list(overlay_candidates(trace,next_peak=next_peak,include_later=True))),2)
            self.assertEqual(trace,before)

    def test_frame_before_cutoff_but_worker_delivered_late(self):
        self.assertEqual(classify(cutoff=10,observed_at=11,evidence_at=9,owner=1,shot_id=1),'POST_DECISION')
        self.assertEqual(classify(cutoff=10,observed_at=None,evidence_at=9,owner=None,shot_id=1),'UNKNOWN')
        self.assertEqual(classify(cutoff=None,observed_at=11,evidence_at=9,owner=1,shot_id=1),'UNKNOWN')

    def test_future_features_not_backdated_same_xy(self):
        c={'camera_x':1,'camera_y':2,'timestamp':9,'score':3}
        d={**c,'timestamp':11,'score':35}
        t={'shot_id':1,'decision_input':{'timestamp':10,'retained_candidates':[c]},'stages':[
            {'timestamp':12,'candidate_pool_shot_id':1,'candidates':[c,d]}]}
        self.assertEqual(analyze_trace(t)['counts_candidate_versions'],
                         {'CAUSALLY_AVAILABLE':1,'POST_DECISION':1,'CROSS_EVENT_OR_FUTURE':0,'UNKNOWN':0})

    def test_synchronous_confirmation_proof(self):
        c={'camera_x':1,'camera_y':2,'timestamp':9}
        t={'shot_id':1,'decision_input':{'timestamp':10,'deterministic_selection':{'last_seen_ts':9}},
           'stages':[{'timestamp':11,'candidate_pool_shot_id':1,
                      'local_confirmation':{'shot_id':1,'frame_ts':9,'candidates':[c]}}]}
        self.assertEqual(analyze_trace(t)['counts_candidate_versions']['CAUSALLY_AVAILABLE'],1)
        t['stages'][0]['local_confirmation']['frame_ts']=10.5
        self.assertEqual(analyze_trace(t)['counts_candidate_versions']['POST_DECISION'],1)

    def test_terminal_outcome_is_not_overwritten(self):
        from src.engine.physical_trace import PhysicalTraceRecorder
        with tempfile.TemporaryDirectory() as directory:
            recorder=PhysicalTraceRecorder(root=Path(directory))
            event=SimpleNamespace(state='matched',emitted=False,shot_id=6,peak_ts=1)
            scanner=SimpleNamespace(last_event_debug={'shot_id':6},last_window_debug={})
            recorder._active[6]={'trace':{'peak_ts':1},'finished':False}
            recorder.finish(6,scanner,event)
            original=copy.deepcopy(recorder._active[6]['trace']['outcome'])
            scanner.last_event_debug={'shot_id':7}
            recorder.finish(6,scanner,event)
            self.assertEqual(recorder._active[6]['trace']['outcome'],original)
            recorder._active.clear()
            recorder.shutdown()

    def test_existing_crop_reference_mismatch_is_reproducible(self):
        # Characterization of the unchanged runtime bug. Correct handling must
        # crop history before supplying it to the V2 collector.
        from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
        engine=CandidateGeneratorV2.__new__(CandidateGeneratorV2)
        frame=np.zeros((60,80),np.uint8);frame[30:50,40:70]=200
        scanner=SimpleNamespace(frame_history=[SimpleNamespace(timestamp=9.9,gray=frame)], _v2221_active_geometry=SimpleNamespace(crop_x0=40,crop_y0=30))
        kwargs={'peak_ts':10.,'bbox':(0,0,30,20),'roi':np.ones((20,30),np.uint8),'cfg':{}}
        good=engine._collect_pre_frames(scanner,**kwargs)[0]
        self.assertEqual(float(good.mean()),200.)

    def test_pending_event_cannot_consume_next_event_frame(self):
        from src.engine.camera.hit_scanner import HitScanner, AudioShotEvent
        from src.engine.shot_fast_v2225 import LocalConfirmManagerV2225, local_confirm_candidates_v2225
        from src.engine.shot_track_v2226 import _install_frame_unique_tracking_patch
        _install_frame_unique_tracking_patch()
        scanner = HitScanner()
        first = AudioShotEvent(6, 100., 100.)
        later = AudioShotEvent(7, 101.4839297, 101.4839297)
        scanner.audio_events.extend([first, later])
        pre = np.full((64, 64), 120, np.uint8)
        candidate = {'camera_x':32., 'camera_y':32., 'score':35., 'timestamp':100.04}
        scanner._update_tracks([candidate], 100.04)
        manager = LocalConfirmManagerV2225()
        manager.start(6, 100.04, [candidate], pre)
        self.assertIsNone(manager.active_waiting(scanner, 101.4862853))

    def test_temporal_research_preserves_new_change_near_old_structure(self):
        from automation.causal_temporal_research import temporal_score
        pre = np.full((64, 64), 120., np.float32)
        pre[29:34, 29:34] = 20.
        self.assertEqual(temporal_score(np.abs(pre-pre), 31, 31), 0.)
        post = pre.copy(); post[29:34, 34:39] = 20.
        self.assertGreater(temporal_score(np.abs(pre-post), 36, 31), 0.)

    def test_export_adds_causal_fields_without_changing_legacy_pool(self):
        import json
        from automation.physical_trace_export import export
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shot = root/'shots'/'shot_00000001'; shot.mkdir(parents=True)
            old = {'camera_x':100.,'camera_y':100.,'timestamp':1.1}
            later = {'camera_x':0.,'camera_y':0.,'timestamp':2.1}
            trace = {'shot_id':1,'peak_ts':1.,'decision_input':{'timestamp':1.5,'retained_candidates':[old]},
                     'stages':[{'timestamp':2.2,'candidate_pool_shot_id':1,'candidates':[later]}]}
            (shot/'trace.json').write_text(json.dumps(trace))
            (shot/'ground_truth.json').write_text(json.dumps({'camera_x':0.,'camera_y':0.,'quality':'precise'}))
            export(root, root/'export.json')
            row = json.loads((root/'export.json').read_text())['shots'][0]
            self.assertEqual(row['retained'],[later])
            self.assertEqual(row['causally_available_candidates_v1'],[old])
            self.assertFalse(row['causal_candidate_audit_v1']['causal_oracle']['hits']['42'])

if __name__=='__main__':unittest.main()
