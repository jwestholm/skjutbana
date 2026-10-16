"""Synthetic review-capture lifecycle tests; never opens a physical data session."""
from collections import deque
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from automation import physical_gt_capture as g
from automation import physical_trace_label_crops as c
from automation.physical_collection import prepare_can_play
from automation.physical_capture_console import CaptureConsole
from automation.physical_trace_label import _run_target_ui, _pygame_surface


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/data/skjutbana');self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name); self.capture=self.base/'capture'
        with redirect_stdout(io.StringIO()):
            self.play,self.env=prepare_can_play(self.capture,gt_review=True,gt_shots=3)
            self.r=g.ReviewCapture(self.env['SKJUTBANA_GT_CAPTURE_ROOT'])
        self.addCleanup(self.r.close)
        self.now=1002.
        clock=NS(time=lambda:self.now,monotonic=lambda:self.now)
        self.patch=patch.object(g,'time',clock);self.patch.start();self.addCleanup(self.patch.stop)
        self.frame_seq=0
        self.camera=NS(_ring=deque(),_lock=threading.Lock(),_read_count=0,_last_pickup_count=0,
            latest_frame=None,update=Mock())
        self.camera.get_ring_snapshot=lambda:list(self.camera._ring)
        self.add_frame(999.);self.add_frame(1000.1);self.add_frame(1000.2)
        self.event=NS(shot_id=7,peak_ts=1000.,state='matched',audio_trigger=self.audio_trigger(1000.))
        self.scanner=NS(audio_events=[self.event],enabled=True,update=Mock(side_effect=AssertionError('no live work in review')))
        self.scanner.disable=Mock(side_effect=lambda:setattr(self.scanner,'enabled',False))
        self.scanner.enable=Mock(side_effect=lambda:setattr(self.scanner,'enabled',True))
        self.query=NS(final=True,published=True)
        self.registry=NS(snapshot_for_shot=lambda sid:NS(object_query=self.query))
        self.worker=NS(has_pending=Mock(return_value=False))
        self.audio=NS(pending_events_snapshot=Mock(return_value=()),stop=Mock(),start=Mock(),last_peak_ts=1000.)
        self.controller=NS(last_seen_peak_ts=1000.)
        self.scene=NS(render=Mock(),update=Mock(side_effect=AssertionError('no game simulation in review')))
        self.app=NS(scene=self.scene,screen=NS(fill=Mock()),clock=NS(tick=Mock()),running=True)
        self.app.quit=lambda:setattr(self.app,'running',False)
        self.pg=NS(event=NS(get=lambda:[]),display=NS(flip=Mock()),QUIT=1,KEYDOWN=2,K_ESCAPE=27)

    def add_frame(self,t):
        self.frame_seq+=1
        frame=NS(timestamp=t,frame_bgr=np.full((9,13,3),self.frame_seq,np.uint8),
            acquisition=dict(read_start_monotonic=t,read_complete_monotonic=t,published_monotonic=t,receive_wall_ts=t,
                sequence=self.frame_seq,stream_id='fixture',
                frame_shape=[9,13,3],raw_shape=[9,13,3],rotation=0,mirror_horizontal=True,mirror_vertical=False))
        self.camera._ring.append(frame);self.camera._read_count+=1;self.camera.latest_frame=frame
        return frame

    @staticmethod
    def audio_trigger(peak):
        return dict(diagnostics=dict(acquisition=dict(trigger_receipt_estimate_monotonic=peak,
            timestamp_semantics='RECEIPT_ANCHORED_ESTIMATE_NOT_DEVICE_ACQUISITION',upstream_buffer_age_s=None)))

    def observe(self):
        with redirect_stdout(io.StringIO()):
            self.r.freeze_pre(self.scanner,self.camera)
            self.r.observe(self.app,self.scanner,self.camera,self.registry,self.worker,self.audio,self.controller)

    def step(self,delta,frame=False):
        self.now+=delta
        if frame:self.add_frame(self.now)
        with redirect_stdout(io.StringIO()):self.r.tick(self.pg)

    def complete(self):
        self.observe();self.step(.51)
        for _ in g.STATES:self.step(.61,True)
        self.step(1.01,True);self.r.close()
        return c.load_gt_review(self.capture)

    def test_live_closure_barrier_no_copy_no_write_no_pause_until_both_final(self):
        before=(self.r.root/'manifest.json').read_bytes()
        with patch.object(self.r.writer,'enqueue_review_frame',side_effect=AssertionError('live copy/write')):
            for attr,value in [('state','pending')]:
                setattr(self.event,attr,value);self.observe();self.event.state='matched'
            self.query.final=False;self.observe();self.query.final=True
            self.query.published=False;self.observe();self.query.published=True
            self.worker.has_pending.return_value=True;self.observe();self.worker.has_pending.return_value=False
            self.audio.pending_events_snapshot.return_value=(object(),);self.observe()
        self.assertFalse(self.r.active);self.scanner.disable.assert_not_called();self.audio.stop.assert_not_called()
        self.assertEqual((self.r.root/'manifest.json').read_bytes(),before)
        self.assertEqual(self.r.writer.stats()['persisted_items'],0)

    def test_second_pending_shot_blocks_review(self):
        self.scanner.audio_events.append(NS(shot_id=8,peak_ts=1001.,state='pending'))
        self.observe();self.assertFalse(self.r.active)
        self.scanner.audio_events[-1].state='missed';self.observe()
        row=self.r.manifest['shots'][0]
        self.assertEqual(row['status'],'UNAVAILABLE');self.assertEqual(row['frames'],[])
        self.assertEqual(row['reason'],'ANOTHER_EVENT_BEFORE_REVIEW')

    def test_full_frame_roundtrip_roles_states_hashes_and_live_results_unchanged(self):
        initial=deepcopy(vars(self.event));m=self.complete();shot=m['shots'][0]
        self.assertEqual(vars(self.event),initial)
        self.assertEqual(shot['shot_id'],7);self.assertEqual(shot['availability'],'AVAILABLE')
        entries=[f['crops'][0] for f in shot['frames']]
        self.assertEqual([e['role'] for e in entries],['PRE','LIVE_POST','LIVE_POST','GT_SETTLED','GT_SETTLED','GT_SETTLED'])
        self.assertEqual([e['projector_state'] for e in entries[-3:]],list(g.STATES))
        for e in entries:
            self.assertEqual(c.source_image(m,e).shape,(9,13,3));self.assertEqual(e['crop_origin'],[0,0])
            self.assertEqual(e['full_camera_size'],[13,9]);self.assertFalse(e['live_authority'])
            meta=json.loads((self.capture/e['metadata_path']).read_text())
            self.assertGreaterEqual(meta['copy_ms'],0);self.assertGreaterEqual(meta['writer_ms'],0)
        self.assertFalse(self.r.writer._active)
        self.assertFalse(any(self.r.root.rglob('trace.json')))

    def test_pre_reuses_ring_with_hard_guard_not_latest_before_peak(self):
        old=self.add_frame(999.5)
        self.add_frame(999.95);self.add_frame(1000.01)
        self.observe();self.r.writer.flush()
        meta=json.loads((self.r.root/'shots/shot_00000007/pre_0.json').read_text())
        self.assertEqual(meta['frame_timestamp'],old.timestamp)
        self.assertEqual(meta['delta_ms_to_audio'],-500.)
        self.assertEqual(meta['selection_rule'],g.PRE_RULE)
        self.assertTrue(meta['pre_cutoff_verified'])
        self.assertFalse(meta['causal_pre_certified'],'receipt clocks cannot certify physical impact time')
        self.assertIsNone(meta['physical_impact_timestamp'])
        self.assertEqual(meta['acquisition'],old.acquisition)

    def test_newer_than_cutoff_can_never_be_written_even_via_direct_save(self):
        self.observe()
        for delta in (-.499999,-.05,0.,.1):
            frame=self.add_frame(self.event.peak_ts+delta)
            with self.assertRaisesRegex(ValueError,'PRE refused'):
                self.r.save_frame(frame,'PRE','unsafe_pre')
        self.r.writer.flush()
        self.assertFalse((self.r.root/'shots/shot_00000007/unsafe_pre.npy').exists())

    def test_old_receive_but_late_publication_or_bad_clock_is_not_pre(self):
        for damage in ('late_publish','bad_wall','missing_clock','missing_audio','nan_clock'):
            with self.subTest(damage=damage):
                frame=deepcopy(self.camera._ring[0]);audio=self.event.audio_trigger['diagnostics']['acquisition'].copy()
                if damage=='late_publish':frame.acquisition['published_monotonic']=999.9
                elif damage=='bad_wall':frame.acquisition['receive_wall_ts']=1000.01
                elif damage=='missing_clock':frame.acquisition.pop('read_complete_monotonic')
                elif damage=='missing_audio':audio={}
                elif damage=='nan_clock':frame.acquisition['published_monotonic']=float('nan')
                self.assertIsNone(g.select_review_pre([frame],1000.,audio))

    def test_no_old_frame_is_unavailable_but_neutral_capture_unchanged(self):
        self.camera._ring=deque(f for f in self.camera._ring if f.timestamp>=1000.)
        self.add_frame(999.95)
        m=self.complete()
        row=self.r.manifest['shots'][0]
        self.assertEqual(row['pre_selection']['status'],'UNAVAILABLE')
        self.assertFalse(row['pre_selection']['causal_pre_certified'])
        self.assertIn('PRE',row['missing'])
        self.assertFalse((self.r.root/'shots/shot_00000007/pre_0.npy').exists())
        self.assertEqual([f['projector_state'] for f in row['frames'] if f['role']=='GT_SETTLED'],list(g.STATES))
        self.assertEqual(m['shots'][0]['availability'],'AVAILABLE')

    def test_missing_audio_clock_cannot_claim_available_pre(self):
        self.event.audio_trigger={}
        self.observe();self.r.writer.flush()
        self.assertIn('PRE',self.r.current['missing'])
        self.assertEqual(self.r.current['pre_selection']['status'],'UNAVAILABLE')
        self.assertFalse((self.r.root/'shots/shot_00000007/pre_0.npy').exists())

    def test_legacy_pre_is_not_retroactively_certified(self):
        self.complete()
        for name in ('config.json','manifest.json'):
            p=self.r.root/name;doc=json.loads(p.read_text())
            doc.pop('pre_selection_rule');doc.pop('pre_guard_ms');p.write_text(json.dumps(doc))
        m=c.load_gt_review(self.capture)
        proof=m['shots'][0]['pre_causality'][0]
        self.assertFalse(proof['causal_pre_certified'])
        self.assertFalse(proof['pre_cutoff_verified'])
        self.assertEqual(proof['reason'],'LEGACY_PRE_WITHOUT_SAFETY_GUARD')
        from automation.physical_gt_audit import audit
        self.assertEqual(audit(self.capture)['shots'][0]['pre'][0]['status'],'UNQUALIFIED')

    def test_reader_rechecks_saved_pre_cutoff_not_claimed_flag(self):
        self.complete()
        side=self.r.root/'shots/shot_00000007/pre_0.json'
        meta=json.loads(side.read_text())
        meta['frame_timestamp']=999.99
        meta['acquisition']['receive_wall_ts']=999.99
        side.write_text(json.dumps(meta))
        path=self.r.root/'manifest.json';m=json.loads(path.read_text())
        m['shots'][0]['frames'][0]['metadata_sha256']=hashlib.sha256(side.read_bytes()).hexdigest()
        path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'PRE cutoff/provenance|timing provenance'):
            c.load_gt_review(self.capture)

    def test_30fps_sequence_boundary_and_selection_deterministic(self):
        frames=[self.add_frame(1000.+i/30.) for i in range(-40,2)]
        audio=self.event.audio_trigger['diagnostics']['acquisition']
        for ordered in (frames,list(reversed(frames)),frames[::2]+frames[1::2]):
            selected=g.select_review_pre(ordered,1000.,audio)
            self.assertEqual(selected.timestamp,999.5)
        self.assertIsNone(g.select_review_pre([self.add_frame(999.501)],1000.,audio))

    def test_event_start_reference_survives_ring_eviction_without_live_copy_or_io(self):
        old=self.camera._ring[0]
        self.event.state='pending'
        with patch.object(self.r.writer,'enqueue_review_frame',side_effect=AssertionError('no live write')):
            self.r.freeze_pre(self.scanner,self.camera)
        frozen=self.r.frozen_pre[7]
        self.assertTrue(np.shares_memory(frozen.frame_bgr,old.frame_bgr))
        self.assertFalse(frozen.frame_bgr.flags.writeable)
        self.assertTrue(old.frame_bgr.flags.writeable)
        self.assertEqual(self.r.writer.stats()['persisted_items'],0)
        self.camera._ring.clear();self.now+=20;self.add_frame(self.now)
        self.event.state='matched';self.observe();self.r.writer.flush()
        meta=json.loads((self.r.root/'shots/shot_00000007/pre_0.json').read_text())
        self.assertEqual(meta['frame_timestamp'],old.timestamp)
        self.assertNotIn(7,self.r.frozen_pre)
        self.assertEqual(self.r.current['pre_selection']['binding_stage'],'FIRST_DISPATCHED_EVENT_OBSERVATION_REFERENCE_ONLY')

    def test_missing_at_event_start_never_retries_later_or_borrows_another_shot(self):
        self.camera._ring.clear();self.event.state='pending'
        self.r.freeze_pre(self.scanner,self.camera)
        self.add_frame(999.);self.event.state='matched';self.observe()
        self.assertIn('PRE',self.r.current['missing'])
        self.assertEqual(self.r.current['pre_selection']['reason'],'NO_QUALIFIED_OLD_RING_FRAME_AT_EVENT_START')

    def test_timing_contract_roundtrip_all_roles_and_manual_label_no_causal_claim(self):
        m=self.complete();s=m['shots'][0]
        for entry in self.r.manifest['shots'][0]['frames']:
            t=entry['timing'];self.assertEqual(t['schema'],g.TIMING_SCHEMA)
            self.assertFalse(t['causal_pre_certified']);self.assertFalse(t['exposure_timestamp_available'])
            self.assertIsNone(t['exposure_timestamp']);self.assertIsNone(t['device_sequence'])
            self.assertIsNotNone(t['frame_source_id']);self.assertIn('read_end_monotonic',t)
            if entry['role']!='PRE':self.assertIsNone(t['pre_cutoff_verified'])
        e=s['frames'][-1]['crops'][0];row=c.make_label(m,s,e,c.crop_click((5,4),(0,0),1,e),[s['frames'][0]['crops'][0]])
        self.assertFalse(row['review_truth_basis']['causal_pre_certified'])
        doc=c.save_row(self.base/'gt.json',m,row)
        doc['shots'][0]['review_truth_basis']['causal_pre_certified']=True
        with self.assertRaisesRegex(ValueError,'cannot certify'):c.validate_document(doc,m)

    def test_independent_settled_observation_without_pre_is_explicit_and_unknown_legal(self):
        self.camera._ring=deque(f for f in self.camera._ring if f.timestamp>=1000.)
        m=self.complete();s=m['shots'][0];e=s['frames'][-1]['crops'][0]
        click=c.crop_click((5,4),(0,0),1,e)
        with self.assertRaisesRegex(ValueError,'inspect'):c.make_label(m,s,e,click,[])
        row=c.make_label(m,s,e,click,[],independent_observation='Synthetic operator observation; no physical claim')
        doc=c.save_row(self.base/'gt.json',m,row)
        self.assertEqual(doc['shots'][0]['source'],e)
        self.assertFalse(doc['shots'][0]['review_truth_basis']['causal_pre_certified'])
        self.assertEqual(doc['shots'][0]['pre_inspected'],[])
        doc=c.save_row(self.base/'gt.json',m,c.unknown_row(7,'NOT_IDENTIFIABLE'),1)
        self.assertEqual(doc['shots'][0]['physical_truth_status'],'UNKNOWN')

    def test_audit_determinism_hashes_unavailable_neutral_and_forbidden_paths(self):
        from automation.physical_gt_audit import audit
        self.observe();self.step(.51);self.step(.61,True);self.r.close()
        m=c.load_gt_review(self.capture);before=deepcopy(m['input_hashes'])
        a=audit(self.capture);self.assertEqual(a,audit(self.capture))
        self.assertEqual(a['shots'][0]['settled']['WHITE']['status'],'AVAILABLE')
        self.assertEqual(a['shots'][0]['settled']['BLACK']['status'],'UNAVAILABLE')
        self.assertEqual(a['shots'][0]['capture_status'],'INTERRUPTED')
        self.assertFalse(a['shots'][0]['pre'][0]['causal_pre_certified'])
        self.assertEqual(before,a['input_hashes']);c.assert_inputs(m)
        with patch.object(Path,'resolve',side_effect=AssertionError('no forbidden lookup')):
            with self.assertRaises(PermissionError):audit('/data/skjutbana/S03')

    def test_neutral_pixels_never_reenter_ring_or_scanner_and_scene_retained(self):
        self.complete()
        self.assertEqual(list(self.camera._ring),[]);self.assertIsNone(self.camera.latest_frame)
        self.assertEqual(self.camera._last_pickup_count,self.camera._read_count)
        self.scanner.update.assert_not_called();self.scene.update.assert_not_called()
        self.scanner.disable.assert_called_once();self.scanner.enable.assert_called_once()
        self.audio.stop.assert_called_once();self.audio.start.assert_called_once()
        self.assertIs(self.app.scene,self.scene);self.assertEqual(self.app.screen.fill.call_args_list,
            [unittest.mock.call(rgb) for rgb in g.STATES.values()])
        self.assertFalse(self.r.active)

    def test_stale_or_unproven_camera_frame_not_used_for_background(self):
        self.observe();self.step(.51)
        self.step(.7)  # only pre/live frames in ring
        self.assertEqual(self.r.phase,'WHITE')
        bad=self.add_frame(self.now);bad.acquisition.pop('read_start_monotonic')
        self.step(.1);self.assertEqual(self.r.phase,'WHITE')
        self.step(2.)
        self.assertEqual(self.r.phase,'BLACK');self.assertIn('WHITE',self.r.current['missing'])
        self.step(.61,True);self.step(.61,True);self.step(1.01,True);self.r.close()
        m=c.load_gt_review(self.capture)
        self.assertEqual(m['shots'][0]['capture_status'],'PARTIAL')
        self.assertNotIn('WHITE',[f['crops'][0]['projector_state'] for f in m['shots'][0]['frames']])

    def test_interrupt_preserves_completed_frames_and_explicit_missing(self):
        self.observe();self.step(.51);self.step(.61,True);self.r.close()
        m=c.load_gt_review(self.capture);s=m['shots'][0]
        self.assertEqual(s['capture_status'],'INTERRUPTED')
        self.assertEqual(s['missing'],['BLACK','MID_GRAY'])
        self.assertEqual([f['role'] for f in s['frames']].count('GT_SETTLED'),1)
        self.audio.start.assert_not_called()

    def test_no_pre_means_unknown_and_no_fabricated_baseline(self):
        self.camera._ring=deque(f for f in self.camera._ring if f.timestamp>=1000.)
        m=self.complete();s=m['shots'][0];e=s['frames'][-1]['crops'][0]
        self.assertIn('PRE',s['missing'])
        with self.assertRaises(ValueError):c.make_label(m,s,e,c.crop_click((1,1),(0,0),1,e),[])
        doc=c.save_row(self.base/'labels.json',m,c.unknown_row(7,'NO_PRE'))
        self.assertIsNone(doc['shots'][0]['camera_xy'])

    def test_source_file_hash_and_metadata_mismatch_refused(self):
        m=self.complete();e=m['shots'][0]['frames'][-1]['crops'][0]
        p=self.capture/e['path'];a=np.load(p);a[0,0,0]^=1;np.save(p,a)
        with self.assertRaisesRegex(ValueError,'hash'):c.load_gt_review(self.capture)
        with self.assertRaisesRegex(ValueError,'hash'):c.source_image(m,e)

    def test_role_tamper_cannot_turn_neutral_into_live_post(self):
        self.complete();p=self.r.root/'manifest.json';m=json.loads(p.read_text())
        m['shots'][0]['frames'][-1]['role']='LIVE_POST';p.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'role'):c.load_gt_review(self.capture)

    def test_review_label_original_gt_only_history_unknown_and_source_unchanged(self):
        m=self.complete();s=m['shots'][0];pre=[s['frames'][0]['crops'][0]];e=s['frames'][-1]['crops'][0]
        before=deepcopy(m['input_hashes']);click=c.crop_click((12,8),(0,0),1,e)
        row=c.make_label(m,s,e,click,pre);path=self.base/'labels.json'
        doc=c.save_row(path,m,row,0)
        self.assertEqual(doc['schema'],c.REVIEW_LABEL_SCHEMA);self.assertEqual(doc['shots'][0]['camera_xy'],[12.,8.])
        self.assertEqual(doc['shots'][0]['clicked_view'],'ORIGINAL_GT_SETTLED')
        self.assertEqual(json.loads(path.read_text()),doc)
        doc=c.save_row(path,m,c.unknown_row(7,'NO_VISIBLE_HOLE'),1)
        self.assertEqual(doc['history'][-1]['previous'],row)
        self.assertEqual(doc['shots'][0]['physical_event'],'UNKNOWN')
        with self.assertRaisesRegex(ValueError,'GT_SETTLED'):
            c.make_label(m,s,s['frames'][1]['crops'][0],click,pre)
        self.assertEqual(m['input_hashes'],before);c.assert_inputs(m)

    def test_repeated_shots_bound_by_real_id_not_plan_and_limit(self):
        self.observe();self.step(.51)
        for _ in g.STATES:self.step(.61,True)
        self.step(1.01,True)
        self.event=NS(shot_id=12,peak_ts=self.now+.1,state='matched')
        self.scanner.audio_events=[self.event];self.add_frame(self.now)
        self.now+=.5;self.add_frame(self.now);self.observe()
        self.assertEqual([s['shot_id'] for s in self.r.manifest['shots']],[7,12])
        self.assertTrue(all(e['path'].startswith('shots/shot_00000012/') for e in self.r.current['frames']))

    def test_ui_no_overlays_unknown_and_manual_original_rgb_click(self):
        import pygame
        m=self.complete();s=m['shots'][0];e=s['frames'][-1]['crops'][0]
        scale,_,origin=c.view_transform(e,(800,490));origin=(origin[0],origin[1]+70)
        pos=tuple(round(origin[i]+[5,4][i]*scale[i]) for i in (0,1))
        def key(k):return pygame.event.Event(pygame.KEYDOWN,key=k)
        events=[[key(pygame.K_c)],[key(pygame.K_4)],[key(pygame.K_3)],
                [pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=pos)],
                [key(pygame.K_RETURN)],[key(pygame.K_q)]]
        path=self.base/'labels.json'
        with patch.dict(os.environ,SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy'), \
             patch.object(pygame.event,'get',side_effect=events), \
             patch('automation.physical_trace_label.overlay_candidates',side_effect=AssertionError('no overlays')):
            _run_target_ui(c.label_shots(m),(800,600),False,None,(300,300),paired=(m,path))
        doc=c.validate_document(json.loads(path.read_text()),m)
        self.assertEqual(doc['shots'][0]['clicked_view'],'ORIGINAL_GT_SETTLED')
        self.assertFalse(doc['shots'][0]['candidate_overlay'])
        with self.assertRaises(ValueError):
            _run_target_ui(c.label_shots(m),(800,600),True,None,(300,300),paired=(m,path))

    def test_rgb_display_preserves_original_channels_no_stretch(self):
        import pygame
        a=np.full((3,5,3),[11,99,202],np.uint8)
        surface=_pygame_surface(pygame,a,(5,3),original=True)
        self.assertEqual(tuple(pygame.surfarray.array3d(surface)[0,0]),(202,99,11))

    def test_candidate_files_not_opened_and_contact_sheet_has_no_truth(self):
        m=self.complete();(self.capture/'mature_candidates.json').write_text('INVALID CANDIDATE DATA')
        before=deepcopy(m['input_hashes'])
        self.assertEqual(c.load_gt_review(self.capture),m)
        out=c.prepare_package(m,self.base/'package')
        self.assertTrue((out/'shot_007/tile_01.png').exists())
        self.assertEqual(json.loads((out/'truth_status.json').read_text())['shots'][0]['physical_event'],'UNKNOWN')
        self.assertEqual(before,m['input_hashes']);c.assert_inputs(m)

    def test_default_launcher_clears_stale_opt_in_and_no_settings_write(self):
        path=Path('content/ai/settings.json');before=hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.dict(os.environ,SKJUTBANA_GT_CAPTURE_ROOT='stale'):
            p,env=prepare_can_play(self.base/'ordinary')
        self.assertNotIn('SKJUTBANA_GT_CAPTURE_ROOT',env)
        self.assertFalse((p.parent/'gt_review').exists())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

    def test_console_research_instructions_and_warnings_visible(self):
        view=CaptureConsole(self.capture,io.StringIO());out=io.StringIO()
        with redirect_stdout(out):
            view.observe('[GT-CAPTURE] shot=7 WHITE DO NOT SHOOT\n',0,40)
            view.observe('[GT-CAPTURE WARNING] shot=7 missing WHITE\n',40,42)
        self.assertIn('DO NOT SHOOT',out.getvalue());self.assertIn('WARNING',out.getvalue())

    def test_s03_guard_before_access(self):
        with patch.object(Path,'resolve',side_effect=AssertionError('no lookup')):
            with self.assertRaises(ValueError):g.checked('/data/skjutbana/S03')

    def test_review_reopen_refused_and_no_original_overwrite(self):
        self.complete();before=(self.r.root/'manifest.json').read_bytes()
        with self.assertRaises(ValueError):g.ReviewCapture(self.r.root)
        self.assertEqual((self.r.root/'manifest.json').read_bytes(),before)

    def test_missing_restored_frame_never_rearms_neutral_camera(self):
        self.observe();self.step(.51)
        for _ in g.STATES:self.step(.61,True)
        self.step(3.01)
        self.assertFalse(self.app.running);self.scanner.enable.assert_not_called()
        self.audio.start.assert_not_called();self.r.close()
        self.assertIn('RESTORED_CAMERA_FRAME',self.r.manifest['shots'][0]['missing'])

    def test_interrupted_before_live_closure_explicit_unavailable(self):
        self.event.state='pending';self.observe();self.r.close()
        m=c.load_gt_review(self.capture)
        self.assertEqual(m['shots'][0]['availability'],'UNAVAILABLE')
        self.assertEqual(m['shots'][0]['frames'],[])
        self.assertEqual(c.new_document(m)['shots'][0]['physical_event'],'UNKNOWN')

    def test_atomic_writer_does_not_mutate_or_overwrite_source(self):
        self.observe();self.r.writer.flush()
        row=self.r.current; entry=row['frames'][0]
        p=self.r.root/entry['path'];before=p.read_bytes()
        frame=self.camera._ring[0]
        frame.frame_bgr[:]=252
        self.assertEqual(p.read_bytes(),before)
        meta=json.loads(p.with_suffix('.json').read_text())
        with self.assertRaises(ValueError):self.r.writer.enqueue_review_frame(p,frame.frame_bgr,meta)
        with self.assertRaises(ValueError):
            self.r.writer.enqueue_review_frame(self.r.root/'wrong.npy',frame.frame_bgr,dict(meta,live_authority=True))

    def test_paired_analyzer_uses_valid_review_gt_without_relabeling_candidates(self):
        from automation.object_candidate_reconciliation import exact_label_analysis
        m=self.complete();s=m['shots'][0];e=s['frames'][-1]['crops'][0]
        row=c.make_label(m,s,e,c.crop_click((5,4),(0,0),1,e),[s['frames'][0]['crops'][0]])
        label=self.base/'labels.json';c.save_row(label,m,row)
        out=self.base/'analysis';out.mkdir();shots=[dict(shot_id=7)]
        with redirect_stdout(io.StringIO()):
            result=exact_label_analysis(self.capture,out,label,[7],{},[],[],shots,{})
        self.assertEqual(result['valid_shot_ids'],[7]);self.assertEqual(shots[0]['exact_camera_gt'],[5.,4.])
        oracle=json.loads((out/'oracle_summary.json').read_text())
        self.assertTrue(all(r['labelled_available_events']==0 for r in oracle['oracle']))

    def test_writer_drain_failure_stops_before_next_live_event(self):
        self.observe();self.step(.51)
        for _ in g.STATES:self.step(.61,True)
        with patch.object(self.r.writer,'flush',return_value=False):self.step(1.01,True)
        self.assertFalse(self.app.running);self.scanner.enable.assert_not_called()
        self.audio.start.assert_not_called()

    def test_actual_run_loop_opt_in_pending_preserves_live_order_and_active_bypasses(self):
        # Execute the installed run function's real source, with device services
        # replaced (no camera/projector/audio acquisition in this integration test).
        tree=ast.parse(Path('src/engine/shot_async_v2224.py').read_text())
        run=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_v2224')
        def exercise(enabled,active_after=False):
            calls=[]
            def mark(name,result=None):
                return lambda *a,**kw:(calls.append(name),result)[1]
            scanner=NS(update=mark('scanner.update'),disable=mark('scanner.disable'))
            audio=NS(last_peak_ts=0.,update=mark('audio.update'),stop=mark('audio.stop'))
            camera=NS(update=mark('camera.update'),stop=mark('camera.stop'))
            registry=NS(advance_object_queries=mark('objects.advance'))
            controller=NS(**{name:mark(name) for name in ('pending_audio','begin_pending_audio','prepare_object_snapshot',
                'discover_scanner_events','note_stage','enrich_spatial_context','evaluate_object_shadow','update_finished',
                'should_defer_scene_work','mark_visible_after_flip')})
            scene=NS(handle_event=mark('scene.event'),update=mark('scene.update'),render=mark('scene.render'),on_exit=mark('scene.exit'))
            app=NS(running=True,scene=scene,screen=None,clock=NS(tick=mark('clock',16)),
                _post_automation_events=mark('automation'),_update_window_caption=mark('caption'),
                communication_server=NS(stop=mark('communication.stop')))
            app.quit=lambda:setattr(app,'running',False)
            loop=iter([[],[NS(type=1)]])
            pg=NS(event=NS(get=lambda:next(loop)),display=NS(flip=mark('flip')),QUIT=1,quit=mark('pygame.quit'))
            review=NS(active=False,close=mark('gt.close'),freeze_pre=mark('gt.freeze_pre'))
            def observe(*args):
                calls.append('gt.observe');review.active=active_after
            review.observe=observe
            def tick(*args):calls.append('gt.tick');app.quit()
            review.tick=tick
            modules={
                'pygame':pg,'config':NS(FPS=60),'src.engine.app':NS(AUTOMATION_EVENT=10),
                'src.engine.audio.audio_peak_detector':NS(audio_peak_detector=audio),
                'src.engine.camera.camera_manager':NS(camera_manager=camera),
                'src.engine.camera.hit_scanner':NS(hit_scanner=scanner),
                'src.engine.output.led_service':NS(led_service=NS(stop=mark('led.stop'))),
                'src.engine.shot_critical_v2223':NS(shot_critical_controller_v2223=controller),
                'src.engine.input.object_hit_v2223':NS(object_hit_registry_v2223=registry)}
            env=dict(os.environ);env.pop('SKJUTBANA_GT_CAPTURE_ROOT',None)
            if enabled:env['SKJUTBANA_GT_CAPTURE_ROOT']='fixture'
            namespace=dict(time=time,os=os,async_detector_v2224=object(),_shutdown_workers=mark('workers.stop'))
            exec(compile(ast.Module(body=[deepcopy(run)],type_ignores=[]),'<real run loop>','exec'),namespace)
            with patch.dict(os.environ,env,clear=True),patch.dict('sys.modules',modules),patch.object(g,'from_environment',return_value=review) as create:
                namespace['run_v2224'](app)
                self.assertEqual(create.call_count,int(enabled))
            return calls
        normal=exercise(False);waiting=exercise(True)
        self.assertEqual(normal,[v for v in waiting if not v.startswith('gt.')])
        self.assertGreater(waiting.index('gt.observe'),waiting.index('mark_visible_after_flip'))
        self.assertGreater(waiting.index('gt.freeze_pre'),waiting.index('audio.update'))
        self.assertLess(waiting.index('gt.freeze_pre'),waiting.index('objects.advance'))
        paused=exercise(True,True)
        self.assertEqual(paused.count('scanner.update'),1);self.assertEqual(paused.count('audio.update'),1)
        self.assertEqual(paused.count('scene.update'),1);self.assertEqual(paused.count('gt.tick'),1)


if __name__=='__main__':unittest.main()
