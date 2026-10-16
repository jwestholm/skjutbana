"""Native mature-candidate shadow boundaries and additive export; generated data only."""
from copy import deepcopy
import atexit
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace as S
from unittest.mock import patch

from automation.object_candidate_reconciliation import gameplay, overlap, run
from automation.object_hit_replay_selftest import capture
from automation.object_physical_replay import mature_candidate_bridge
from automation.object_physical_query_selftest import snapshot
from automation.physical_collection import prepare_can_play
import automation.physical_trace_selftest as trace_fixture
from src.engine.input.hit_regions import HitRegion, object_hit_regions
from src.engine.physical_trace import PhysicalTraceRecorder, capture_object_mature_pool


def candidate(x=124.25,y=90.75,**extra):
    return dict(camera_x=x,camera_y=y,timestamp=2.1,v2224_producer_shot_id=1,
        v252_evidence_ready=1.,v252_fresh_physical=1.,v2225_local_confirm=1.,
        v252_evidence_best_dx=0.,v252_evidence_best_dy=0.,
        v2225_confirm_best_dx=0.,v2225_confirm_best_dy=0.,**extra)


def packet(cs=None,**extra):
    return dict(shot_id=1,peak_ts=2.,frame_ts=2.1,coordinate_space='camera',
        stage='LOCAL_CONFIRMATION',full_rescue=False,available_at=2.15,complete=True,
        candidates=[candidate()] if cs is None else cs,**extra)


def frozen(boxes=None):
    return snapshot(game=object_hit_regions(boxes or [HitRegion('can',100,60,60,80)]))


class BridgeTests(unittest.TestCase):
    def call(self,p=None,s=None,**kwargs):
        return mature_candidate_bridge(s or frozen(),lambda x,y:(x,y),p or packet(),2.2,**kwargs)

    def test_exact_xy_source_and_no_mutation(self):
        p=packet([candidate(score=999,detector_v1=1,producer_note='native')]);before=deepcopy(p)
        r=self.call(p);self.assertEqual(r['state'],'OBJECT_HIT')
        self.assertEqual(r['rows'][0]['camera_xy'],[124.25,90.75]);self.assertEqual(p,before)
        self.assertEqual(r['rows'][0]['candidate'],p['candidates'][0]);self.assertEqual(r['rows'][0]['source'],'V1')

    def test_outside_shape_even_huge_score_never_creates_object(self):
        r=self.call(packet([candidate(x=163,score=1e9)]))
        self.assertEqual(r['supported_object_ids'],[]);self.assertEqual(r['state'],'UNKNOWN')

    def test_two_objects_abstain_without_ordering(self):
        s=frozen([HitRegion('a',100,60,60,80),HitRegion('b',200,60,60,80)])
        r=self.call(packet([candidate(score=1),candidate(x=220,score=1e9)]),s)
        self.assertEqual(r['supported_object_ids'],['a','b']);self.assertEqual(r['state'],'UNKNOWN')

    def test_overlaps_retain_all_and_roles_do_not_weight(self):
        s=frozen([HitRegion('target',100,60,60,80),HitRegion('no_shoot',110,60,60,80)])
        self.assertEqual(self.call(s=s)['supported_object_ids'],['no_shoot','target'])
        self.assertEqual(self.call(s=s)['state'],'UNKNOWN')

    def test_empty_control_no_invented_hit(self):
        r=self.call(packet([]));self.assertEqual(r['state'],'UNKNOWN');self.assertFalse(r['no_physical_impact_established'])

    def test_frozen_not_current_geometry(self):
        s=frozen();moved=replace(s,game_regions=object_hit_regions([HitRegion('can',200,60,60,80)]))
        self.assertEqual(self.call(s=s)['object_id'],'can');self.assertEqual(self.call(s=moved)['state'],'UNKNOWN')

    def test_mouse_has_no_physical_authority(self):
        self.assertEqual(self.call(s=replace(frozen(),shot_id=None))['state'],'UNKNOWN')

    def test_late_delivery_or_frame_or_wrong_owner_abstains(self):
        for key,value in [('available_at',2.25),('frame_ts',2.3),('peak_ts',1.0),('shot_id',3),('coordinate_space','working')]:
            p=packet();p[key]=value;self.assertEqual(self.call(p)['state'],'UNKNOWN',key)
        for key,value in [('v2224_producer_shot_id',2),('timestamp',2.25),('timestamp',1.99),('v251_shot_id',3),('v253_peak_ts',1.0)]:
            p=packet();p['candidates'][0][key]=value;self.assertEqual(self.call(p)['state'],'UNKNOWN',key)

    def test_missing_owner_and_missing_confirmation_are_unknown(self):
        for field in ('v2224_producer_shot_id','timestamp','v252_fresh_physical','v2225_local_confirm','v252_evidence_best_dx'):
            p=packet();del p['candidates'][0][field]
            self.assertEqual(self.call(p)['state'],'UNKNOWN',field)
        for field in ('frame_ts','available_at','coordinate_space','peak_ts'):
            p=packet();del p[field]
            self.assertEqual(self.call(p)['state'],'UNKNOWN',field)

    def test_inactive_objects_are_not_eligible_and_no_support_retention(self):
        s=frozen();s=replace(s,game_regions=tuple(replace(r,enabled=False) for r in s.game_regions))
        self.assertEqual(self.call(s=s)['available_object_ids'],[])
        self.assertEqual(self.call()['state'],'OBJECT_HIT')
        self.assertEqual(self.call(packet([]))['state'],'UNKNOWN')

    def test_emitted_track_state_is_not_local_confirmation(self):
        p=packet();p['candidates'][0].pop('v2225_local_confirm')
        p['candidates'][0]['state']='confirmed'
        self.assertEqual(self.call(p)['state'],'UNKNOWN')

    def test_neighborhood_proof_cannot_transfer_to_exact_shape(self):
        for field in ('v252_evidence_best_dx','v2225_confirm_best_dy'):
            p=packet();p['candidates'][0][field]=2
            self.assertEqual(self.call(p)['state'],'UNKNOWN')

    def test_rescue_proposal_only_truncated_and_future_event(self):
        for key,value in [('full_rescue',True),('complete',False),('stage','WORKER_DELIVERY')]:
            p=packet();p[key]=value;self.assertEqual(self.call(p)['state'],'UNKNOWN',key)
        self.assertEqual(self.call(next_peak=2.05)['state'],'UNKNOWN')

    def test_overlap_metrics_no_coordinate_mutation(self):
        self.assertEqual(overlap([[1,2],[9,10]],[[1,2],[50,50]]),dict(
            a_count=2,b_count=2,exact_shared=1,a_without_b_within5=1,b_without_a_within5=1))


class ExportTests(unittest.TestCase):
    def test_bounded_additive_writer_after_object_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder=PhysicalTraceRecorder(directory,enabled=True,object_only=True,writer_workers=1)
            atexit.unregister(recorder.flush)  # temp fixture is explicitly flushed below
            scanner=trace_fixture.TraceTests().scanner();ev=S(shot_id=1,peak_ts=2.)
            recorder.start_from_scanner(scanner,ev)
            recorder.finish_object_query(1,dict(status='UNCERTAIN',object_decision={'state':'UNKNOWN'}))
            self.assertTrue(recorder.flush());trace=Path(directory)/'shots/shot_00000001/trace.json';before=trace.read_bytes()
            cs=[candidate()];original=deepcopy(cs)
            for _ in range(34):recorder.capture_mature_pool(1,'LOCAL_CONFIRMATION',cs,2.1,2.15)
            self.assertTrue(recorder.flush());self.assertEqual(cs,original);self.assertEqual(trace.read_bytes(),before)
            folder=trace.parent/'mature_candidates';self.assertEqual(len(list(folder.iterdir())),33)
            self.assertEqual(json.loads((folder/'pool_0000.json').read_text())['candidates'],original)
            self.assertFalse(json.loads((folder/'LIMIT.json').read_text())['complete'])

    def test_candidate_and_byte_caps_are_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            r=PhysicalTraceRecorder(directory,enabled=True,object_only=True,writer_workers=1)
            atexit.unregister(r.flush)
            r.start_from_scanner(trace_fixture.TraceTests().scanner(),S(shot_id=1,peak_ts=2.))
            r.finish_object_query(1,dict(status='UNCERTAIN',object_decision={'state':'UNKNOWN'}))
            r.capture_mature_pool(1,'LOCAL_CONFIRMATION',[candidate()]*1025,2.1,2.15)
            r.capture_mature_pool(1,'LOCAL_CONFIRMATION',[],2.1,2.15,diagnostics={'oversize':'x'*(2*1024*1024)})
            self.assertTrue(r.flush())
            folder=Path(directory)/'shots/shot_00000001/mature_candidates'
            first=json.loads((folder/'pool_0000.json').read_text());second=json.loads((folder/'pool_0001.json').read_text())
            self.assertEqual(len(first['candidates']),1024);self.assertFalse(first['complete'])
            self.assertFalse(second['complete']);self.assertEqual(second['unavailable_reason'],'pool_byte_limit')
            self.assertLess((folder/'pool_0001.json').stat().st_size,2048)

    def test_async_export_does_not_change_consumed_pool(self):
        from src.engine.shot_async_v2224 import AsyncDetectorV2224, DetectorJobResultV2224
        states=[];native=[]
        for flag in ('0','1'):
            s=S();r=DetectorJobResultV2224(1,2.1,2.,0,0,0,[candidate(score=4),candidate(x=220,score=2)],{},{},0,0,0)
            with patch.dict(os.environ,{'SKJUTBANA_OBJECT_MATURE_CAPTURE':flag}),redirect_stdout(io.StringIO()),\
                 patch('src.engine.physical_trace.capture_object_mature_pool') as export:
                AsyncDetectorV2224.apply_result(s,r)
                if flag=='1':
                    export.assert_called_once();self.assertEqual(export.call_args.args[2],r.candidates)
                else:export.assert_not_called()
            states.append(vars(s));native.append(r.candidates)
        self.assertEqual(states[0],states[1]);self.assertEqual(native[0],native[1])

    def test_off_is_noop_and_export_exceptions_do_not_escape(self):
        with patch.dict(os.environ,{'SKJUTBANA_OBJECT_MATURE_CAPTURE':'0'}),patch('src.engine.physical_trace._OBJECT_RECORDER') as r:
            capture_object_mature_pool(1,'WORKER_DELIVERY',[],2.1);r.capture_mature_pool.assert_not_called()
        with patch.dict(os.environ,{'SKJUTBANA_OBJECT_MATURE_CAPTURE':'1'}),patch('src.engine.physical_trace._OBJECT_RECORDER') as r:
            r.capture_mature_pool.side_effect=ValueError('fixture')
            with redirect_stdout(io.StringIO()):capture_object_mature_pool(1,'WORKER_DELIVERY',[],2.1)

    def test_launcher_process_only_and_requires_crops(self):
        settings=Path('content/ai/settings.json').read_bytes()
        with tempfile.TemporaryDirectory(dir='/data/skjutbana') as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):prepare_can_play(root/'bad',mature_candidates=True)
            path,env=prepare_can_play(root/'good',object_crops=True,mature_candidates=True)
            self.assertEqual(env['SKJUTBANA_OBJECT_MATURE_CAPTURE'],'1')
            self.assertFalse(json.loads(path.read_text())['full_frame_capture'])
        self.assertEqual(settings,Path('content/ai/settings.json').read_bytes())


class AnalyzerTests(unittest.TestCase):
    def test_deterministic_analysis_missing_pool_is_unavailable(self):
        with tempfile.TemporaryDirectory(dir='/data/skjutbana') as directory:
            root=Path(directory);source=root/'source';capture(source)
            (source/'operator_ground_truth.json').write_text(json.dumps(dict(shots=[dict(shot_id=1,physical_truth='OBJECT_HIT',object_id='can')])))
            a=gameplay(source,{});b=gameplay(source,{})
            self.assertEqual(a,b);self.assertTrue(a['rows'][0]['local_gt_supported'])
            self.assertIsNone(a['rows'][0]['source_overlap']);self.assertEqual(a['rows'][0]['mature_pool_status'],'UNAVAILABLE_NOT_RECORDED')

    def test_no_operator_truth_is_unknown_not_planned_aim(self):
        with tempfile.TemporaryDirectory(dir='/data/skjutbana') as directory:
            source=Path(directory)/'source';capture(source)
            r=gameplay(source,{})['rows'][0]
            self.assertEqual(r['truth_state'],'UNKNOWN');self.assertIsNone(r['local_false_supported_objects'])
            self.assertIsNone(r['gt_object'])

    def test_saved_pool_respects_cutoff_and_capture_completeness(self):
        with tempfile.TemporaryDirectory(dir='/data/skjutbana') as directory:
            source=Path(directory)/'source';trace=capture(source)
            path=source/'object_crops/shots/shot_00000001/trace.json'
            folder=path.parent/'mature_candidates';folder.mkdir()
            p=packet();p['available_at']=trace['object_query_diagnostic']['decision_ts']-.001
            p['frame_ts']=p['candidates'][0]['timestamp']=2.065
            (folder/'pool_0000.json').write_text(json.dumps(p))
            r=gameplay(source,{})['rows'][0]
            self.assertEqual(r['mature_at_cutoff']['state'],'OBJECT_HIT')
            p['available_at']+=1.;(folder/'pool_0001.json').write_text(json.dumps(p))
            self.assertEqual(gameplay(source,{})['rows'][0]['mature_at_cutoff'],r['mature_at_cutoff'])
            trace['completeness']['trace_complete']=False;path.write_text(json.dumps(trace))
            self.assertEqual(gameplay(source,{})['rows'][0]['mature_at_cutoff']['state'],'UNKNOWN')

    def test_no_files_read_for_forbidden_source_or_outside_output(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('no reads')):
            for forbidden in ('/data/skjutbana/S03','/data/skjutbana/session_S03_x'):
                with self.assertRaises(PermissionError):run([Path(forbidden)],Path('/data/skjutbana/never-created'))
            with self.assertRaises(ValueError):run([Path('/data/skjutbana/allowed')],Path('/tmp/outside'))
            with self.assertRaises(ValueError):run([],Path('/data/skjutbana/frozen/output'),Path('/data/skjutbana/frozen'))


class PairedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/data/skjutbana');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.capture=self.root/'source';self.trace=capture(self.capture)
        self.folder=self.capture/'object_crops/shots/shot_00000001/mature_candidates';self.folder.mkdir()
        self.pool=packet();self.pool['frame_ts']=self.pool['candidates'][0]['timestamp']=2.065
        self.write_pool(self.pool)

    def write_pool(self,p,name='pool_0000.json'):
        (self.folder/name).write_text(json.dumps(p))

    def analyze(self,name='out',truth=None):
        from automation.object_candidate_reconciliation import paired_analysis
        with redirect_stdout(io.StringIO()):return paired_analysis(self.capture,self.root/name,truth)

    def test_same_hashes_same_results_no_source_mutation(self):
        import hashlib
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.capture.rglob('*') if p.is_file()}
        a=self.analyze('a');b=self.analyze('b')
        for file in ('summary.json','stage_counts.json','local_objects.json','candidates.jsonl','input_manifest.json'):
            self.assertEqual((a/file).read_bytes(),(b/file).read_bytes())
        self.assertTrue(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()))
        self.assertFalse((a/'oracle_summary.json').exists())

    def test_candidate_native_xy_order_full_rescue_and_frozen_membership(self):
        p=packet([candidate(),candidate(x=400)],full_rescue_marker='audit')
        p['full_rescue']=True;self.write_pool(p)
        out=self.analyze();rows=[json.loads(l) for l in (out/'candidates.jsonl').read_text().splitlines()]
        self.assertEqual([r['camera_xy'] for r in rows],[[124.25,90.75],[400,90.75]])
        self.assertEqual([r['inside_object_ids'] for r in rows],[['can'],[]])
        self.assertTrue(all(r['full_rescue'] for r in rows))
        self.assertTrue(all('global_rescue_or_scope_unknown' in r['strict_reasons'] for r in rows))
        self.assertEqual(json.loads((out/'stage_counts.json').read_text())[0]['strict_exact_objects_at_delivery'],[])

    def test_partial_event_export_log_is_not_candidate_pool_or_physical_label(self):
        p=self.capture/'runtime.log'
        p.write_text(p.read_text()+'[V2.22.3 SHOT] shot=2 dispatch=3ms objects=7\n'
                     '[OBJECT-MATURE-CAPTURE] shot=2 stage=WORKER_DELIVERY candidates=29 copy_enqueue_ms=0.005\n')
        out=self.analyze();rows=json.loads((out/'summary.json').read_text())['shots']
        self.assertEqual(len(rows),2);self.assertEqual(rows[1]['mature_pool_status'],'UNAVAILABLE_NOT_ANALYZABLE')
        self.assertEqual(rows[1]['physical_truth'],'UNKNOWN');self.assertIsNone(rows[1]['object_state'])
        stages=json.loads((out/'stage_counts.json').read_text())
        self.assertTrue(all(r['candidate_count'] is None for r in stages if r['shot_id']==2))

    def test_no_physical_shot_not_target_miss_intent_not_truth(self):
        from automation.object_candidate_reconciliation import operator_truth_rows
        rows=operator_truth_rows(dict(shots=[dict(shot_id=1,physical_truth='TARGET_MISS',physical_discharge='yes'),
            dict(shot_id=2,physical_truth='NO_PHYSICAL_SHOT',physical_discharge='no'),
            dict(shot_id=3,intended_object='can')]),[1,2,3],'hash')
        self.assertEqual([r['physical_truth'] for r in rows.values()],['TARGET_MISS','NO_PHYSICAL_SHOT','UNKNOWN'])
        self.assertIsNone(rows[3].get('object_id'));self.assertEqual(rows[3]['intended_object'],'can')
        with self.assertRaises(ValueError):operator_truth_rows(dict(capture_play_sha256='other'),[1],'hash')
        with self.assertRaises(ValueError):operator_truth_rows(dict(shots=[dict(shot_id=1,physical_truth='NO_PHYSICAL_SHOT',physical_discharge='yes')]),[1],'hash')

    def test_missing_stage_and_truncated_pool_abstain(self):
        p=self.pool;p['complete']=False;self.write_pool(p)
        out=self.analyze();stages=json.loads((out/'stage_counts.json').read_text())
        lc=next(r for r in stages if r['stage']=='LOCAL_CONFIRMATION')
        self.assertEqual(lc['availability'],'INCOMPLETE_EXPORT');self.assertEqual(lc['strict_exact_objects_at_delivery'],[])
        worker=next(r for r in stages if r['stage']=='WORKER_DELIVERY')
        self.assertEqual(worker['availability'],'UNAVAILABLE');self.assertIsNone(worker['candidate_count'])

    def test_output_overlap_and_validation_refused_before_source_read(self):
        from automation.object_candidate_reconciliation import paired_analysis
        with patch.object(Path,'read_text',side_effect=AssertionError('no data access')):
            with self.assertRaises(PermissionError):paired_analysis('/data/skjutbana/S03',self.root/'out')
            with self.assertRaises(ValueError):paired_analysis(self.capture,self.capture/'out')

    def test_recorded_crop_hash_mismatch_refused(self):
        p=self.capture/'object_crops/shots/shot_00000001/0.npy'
        import numpy as np
        a=np.load(p);a[0,0]+=1;np.save(p,a)
        with self.assertRaisesRegex(ValueError,'hash'):self.analyze()

    def test_interrupted_trace_without_final_query_is_explicitly_unavailable(self):
        path=self.capture/'object_crops/shots/shot_00000001/trace.json'
        self.trace.pop('object_query_diagnostic');path.write_text(json.dumps(self.trace))
        out=self.analyze();row=json.loads((out/'summary.json').read_text())['shots'][0]
        self.assertEqual(row['trace_status'],'INCOMPLETE_NO_QUERY_RESULT')
        self.assertIsNone(row['object_state']);self.assertEqual(row['physical_truth'],'UNKNOWN')


if __name__=='__main__':unittest.main()
