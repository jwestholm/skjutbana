"""Instrumented-vs-uninstrumented runtime and fail-loud replay regressions."""
import copy, json, tempfile, unittest
from pathlib import Path
from src.engine.camera.hit_scanner import HitScanner, AudioShotEvent
from src.engine.shot_track_v2226 import update_tracks_frame_unique_v2226 as update
from src.engine.track_audit import capture, record_selection, track_state
from src.engine.offline.track_replay import current_exact_replay


def candidate(x,score=10,**extra):
    return dict(camera_x=float(x),camera_y=30.,score=float(score),timestamp=100.1,**extra)

class Tests(unittest.TestCase):
    def scanner(self,enabled=True):
        s=HitScanner();s.physical_trace_capture_enabled=enabled;s.last_trace_pipeline_shot_id=1
        s.audio_events.append(AudioShotEvent(1,100,100));return s

    def test_more_than_eight_exact_and_no_behavior_change(self):
        snapshots=[]
        for enabled in (False,True):
            s=self.scanner(enabled);cs=[candidate(i*20,30-i,detector_v1=1) for i in range(30)]
            update(s,cs,100.1);update(s,[{**c,'score':c['score']+3,'v2225_local_confirm':1} for c in cs],100.6)
            selected=s._best_track_for_event(s.audio_events[0]);s._refresh_debug_views()
            snapshots.append([(t.track_id,t.camera_x,t.camera_y,t.best_score,t.hits) for t in s._active_tracks.values()])
            if enabled:
                audit=capture(s,s.audio_events[0],selected)
                self.assertEqual(len(audit['tracks']),30);self.assertEqual(len(s.last_stable_tracks),8)
                self.assertEqual(current_exact_replay(audit)['track_id'],selected.track_id)
                self.assertEqual(sum(len(b['records']) for b in audit['associations']),60)
                bad=copy.deepcopy(audit);bad['selected_track_id']=29
                with self.assertRaises(AssertionError):current_exact_replay(bad)
        self.assertEqual(*snapshots)

    def test_association_source_history_score_and_coordinate_anchor(self):
        s=self.scanner();update(s,[candidate(20,40,v2225_fast_extract=1),candidate(23,5,detector_v1=1)],100.1)
        update(s,[candidate(24,6,detector_v1=1,v2225_local_confirm=1)],100.6)
        t=next(iter(s._active_tracks.values()));selected=s._best_track_for_event(s.audio_events[0]);a=capture(s,s.audio_events[0],selected)
        records=[r for b in a['associations'] for r in b['records']]
        self.assertEqual([r['reason'] for r in records],['no_track_within_merge_radius','same_frame_support','later_frame_nearest_track'])
        self.assertEqual(records[1]['before']['xy'],records[1]['after']['xy'])
        self.assertEqual(records[2]['association_distance'],4)
        self.assertAlmostEqual(t.camera_x,21.4);self.assertEqual(t.best_score,40)
        self.assertEqual(t.last_candidate['score'],6)
        self.assertEqual([h['source'] for h in a['tracks'][0]['source_history']],['FAST','V1'])
        self.assertEqual(a['tracks'][0]['observation_id'],records[2]['observation_id'])

    def test_cross_event_and_rejected_reason(self):
        s=self.scanner();update(s,[candidate(20,40,v2224_producer_shot_id=2),candidate(60,5,v2224_producer_shot_id=1)],100.1)
        selected=s._best_track_for_event(s.audio_events[0]);a=capture(s,s.audio_events[0],selected)
        self.assertEqual(a['tracks'][0]['rejection_reason'],'producer_shot_id_mismatch')
        self.assertEqual(current_exact_replay(a)['camera_x'],60)

    def test_actual_result_transport_rejects_both_false_event_patterns(self):
        from src.engine.shot_async_v2224 import AsyncDetectorV2224, DetectorJobResultV2224
        from unittest.mock import patch
        for sid,boundary in ((6,101.4839297),(14,101.4904709)):
            s=self.scanner();s.audio_events.clear();s.last_trace_pipeline_shot_id=sid
            e=AudioShotEvent(sid,100,100);s.audio_events.extend([e,AudioShotEvent(sid+1,boundary,boundary)])
            update(s,[candidate(20,10,v2224_producer_shot_id=sid)],100.1)
            result=DetectorJobResultV2224(sid+1,boundary+.07,boundary,0,0,0,[candidate(20,30)],{},{},0,0,0)
            with patch('src.engine.shot_async_v2224._setting_bool',return_value=False):
                AsyncDetectorV2224.apply_result(s,result)
            self.assertEqual(result.candidates[0]['v2224_producer_shot_id'],sid+1)
            self.assertEqual(s.last_candidates[0]['v2224_producer_shot_id'],sid+1)
            update(s,result.candidates,result.frame_ts)
            self.assertIsNone(s._best_track_for_event(e))
            snap=capture(s,e)
            self.assertEqual(snap['tracks'][0]['rejection_reason'],'producer_shot_id_mismatch')
            self.assertEqual(snap['tracks'][0]['causal_ownership'],'CROSS_EVENT_OR_FUTURE')
            self.assertIsNone(current_exact_replay(snap))

    def test_late_preboundary_worker_result_is_still_eligible(self):
        from src.engine.shot_async_v2224 import AsyncDetectorV2224, DetectorJobResultV2224, tracking_frame_timestamp
        from unittest.mock import patch
        s=self.scanner();e=s.audio_events[0];s.audio_events.append(AudioShotEvent(2,101.48,101.48))
        update(s,[candidate(20,10,v2224_producer_shot_id=1)],100.1)
        result=DetectorJobResultV2224(1,100.2,100,0,0,0,[candidate(20,12)],{},{},0,0,0)
        with patch('src.engine.shot_async_v2224._setting_bool',return_value=False):AsyncDetectorV2224.apply_result(s,result)
        ts=tracking_frame_timestamp(s,result.candidates,101.9)
        update(s,result.candidates,ts)
        chosen=s._best_track_for_event(e)
        self.assertIsNotNone(chosen);self.assertEqual(chosen.last_seen_ts,100.2)
        self.assertEqual(current_exact_replay(capture(s,e,chosen))['camera_x'],20)

    def test_snapshot_immutable_after_new_observation(self):
        s=self.scanner();update(s,[candidate(20)],100.1);t=s._best_track_for_event(s.audio_events[0]);a=capture(s,s.audio_events[0],t);before=copy.deepcopy(a)
        update(s,[candidate(24,60)],100.6);self.assertEqual(a,before)

    def test_complete_pool_persisted_and_export_checks_identity(self):
        from src.engine.physical_trace import PhysicalTraceRecorder
        from automation.physical_trace_export import export
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);s=self.scanner();e=s.audio_events[0]
            update(s,[candidate(i*20,30-i) for i in range(20)],100.1)
            t=s._best_track_for_event(e)
            recorder=PhysicalTraceRecorder(root,enabled=True)
            recorder.capture_decision(s,t,e,{})
            before=copy.deepcopy(recorder._active[1]['trace']['decision_input'])
            update(s,[candidate(24,100)],100.6)
            recorder.capture_decision(s,next(iter(s._active_tracks.values())),e,{})
            self.assertEqual(recorder._active[1]['trace']['decision_input'],before)
            e.state='matched';recorder.finish(1,s,e);self.assertTrue(recorder.flush())
            data=json.loads((root/'shots/shot_00000001/trace.json').read_text())
            self.assertEqual(len(data['decision_input']['complete_track_audit']['tracks']),20)
            export(root,root/'export.json')
            self.assertEqual(json.loads((root/'export.json').read_text())['shots'][0]['current_exact_replay_v1']['status'],'MATCH')
            recorder.shutdown()

    def test_absent_information_fails_loudly(self):
        with self.assertRaises(ValueError):current_exact_replay({'complete':False})

    def test_eligibility_is_distinct_from_local_confirmation_and_readiness(self):
        s=self.scanner();update(s,[candidate(20,40),candidate(60,5)],100.1)
        update(s,[candidate(60,6,v2225_local_confirm=1)],100.6)
        t=s._best_track_for_event(s.audio_events[0]);a=capture(s,s.audio_events[0],t)
        self.assertEqual(t.camera_x,20);self.assertFalse(s._track_is_ready(t,100.6,s.audio_events[0]))
        self.assertTrue(a['tracks'][0]['eligible']);self.assertFalse(a['tracks'][0]['local_confirmed'])

    def test_candidate_order_and_same_frame_duplicates(self):
        states=[]
        for cs in ([candidate(20,10),candidate(25,5)], [candidate(25,5),candidate(20,10)]):
            s=self.scanner();update(s,cs,100.1);states.append([(t.camera_x,t.hits) for t in s._active_tracks.values()])
        self.assertEqual(*states);self.assertEqual(states[0],[(20.,1)])
        # Equal-score ties retain input order; they legitimately choose a different representative.
        s=self.scanner();update(s,[candidate(25,10),candidate(20,10)],100.1)
        self.assertEqual(next(iter(s._active_tracks.values())).camera_x,25)

    def test_alternate_policy_keys_and_gate(self):
        s=self.scanner();update(s,[candidate(20,v251_confirm_score=2),candidate(60,v251_confirm_score=3)],100.1)
        e=s.audio_events[0];t=list(s._active_tracks.values())[1]
        record_selection(s,e,t,'V251');self.assertEqual(current_exact_replay(capture(s,e,t))['camera_x'],60)
        record_selection(s,e,None,'V253',gate='await_registered_authority')
        self.assertIsNone(current_exact_replay(capture(s,e)))

if __name__=='__main__':unittest.main()
