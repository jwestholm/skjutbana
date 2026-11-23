"""Synthetic engine-render provenance; no camera, projector or physical labels."""
import os
import ast
import time
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pygame
from automation import physical_render_history as r
from automation.physical_collection import prepare_can_play


# Synthetic captures use controlled free space; live disk policy is unchanged.
from automation.trace_test_support import setUpModule, tearDownModule


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/data/skjutbana/evaluation_runs')
        self.addCleanup(self.tmp.cleanup);self.capture=Path(self.tmp.name)/'capture'
        _,env=prepare_can_play(self.capture,object_crops=True,render_history=True)
        self.rec=r.RenderHistoryCapture(env['SKJUTBANA_RENDER_HISTORY_ROOT']);self.addCleanup(self.rec.close)
        self.screen=pygame.Surface((64,48));self.screen.fill((10,20,30));self.scene=S()
        self.registry=S(game_context=lambda sid:dict(shot_id=sid,peak_ts=100.,game_hit_regions=[{'object_id':'fixture'}]))
        self.mapper=S(homography=np.eye(3),calibration_data={'fixture':True})
        self.event=S(shot_id=1,peak_ts=100.,audio_trigger={'diagnostics':{'acquisition':{
            'trigger_receipt_estimate_monotonic':100.,'timestamp_semantics':'RECEIPT_ANCHORED_ESTIMATE_NOT_DEVICE_ACQUISITION'}}})
        self.scanner=S(audio_events=[self.event])

    def present(self,t=99.9,scene=None):
        self.rec.presented(self.screen,self.scene if scene is None else scene,t-.001,t,t+1000)

    def observe(self,t=100.01):
        with patch.object(r.time,'monotonic',return_value=t):
            self.rec.observe(self.scanner,self.registry,self.mapper,self.scene)

    def finish(self):
        self.rec.poll(101.)
        self.rec.close()
        return json.loads((self.rec.root/'shot_00000001.json').read_text())

    def test_immutable_pixels_original_bgr(self):
        self.present();pixels=self.rec.ring[0][1];self.screen.fill((255,0,0))
        np.testing.assert_array_equal(pixels[0,0],[30,20,10]);self.assertFalse(pixels.flags.writeable)

    def test_native_conversion_preserves_odd_size_orientation_and_colors(self):
        rng=np.random.default_rng(1729)
        for flags in [0,pygame.SRCALPHA]:
            screen=pygame.Surface((13,17),flags,32)
            for x in range(13):
                for y in range(17):screen.set_at((x,y),tuple(rng.integers(0,256,4)))
            reference=pygame.surfarray.array3d(screen).transpose(1,0,2)[:,:,::-1]
            np.testing.assert_array_equal(r.copy_render_pixels(screen),reference)

    def test_binding_has_no_hash_copy_or_disk_io(self):
        self.present();original=self.rec.ring[0][1]
        with (patch.object(r.hashlib,'sha256',side_effect=AssertionError('hash in event')),
                patch.object(r,'atomic',side_effect=AssertionError('disk in event'))):
            self.observe()
        self.assertIs(self.rec.pending[0]['frames'][0][1],original)

    def test_frame_roundtrip_identity_and_role_separation(self):
        self.present();self.observe();row=self.finish()
        self.assertEqual(row['status'],'AVAILABLE_CPU_HISTORY');self.assertFalse(row['physical_projection_identity_certified'])
        self.assertEqual(row['frames'][0]['relative_to_audio_estimate'],'BEFORE')
        md=json.loads((self.rec.root/row['frames'][0]['path']).with_suffix('.json').read_text())
        self.assertNotIn('full_camera_size',md);self.assertEqual(md['render_size'],[64,48])
        self.assertEqual(md['coordinate_space'],'PYGAME_CLIENT_PIXELS');self.assertFalse(md['projectile_truth'])
        self.assertEqual(r.audit(self.capture)['shots'][0]['presentations'],1)

    def test_unchanged_content_dedup_multiple_presentations(self):
        self.present(99.8);self.present(99.9);self.observe();self.present(100.1)
        row=self.finish();self.assertEqual(len(row['frames']),3)
        self.assertEqual(len({f['path'] for f in row['frames']}),1)
        self.assertEqual([f['render_id'] for f in row['frames']],[1,2,3])

    def test_source_mutation_after_event_does_not_move_binding(self):
        self.present();self.observe();self.mapper.homography[0,2]=100
        row=self.finish();self.assertEqual(row['calibration']['camera_to_screen'][0][2],0)

    def test_external_and_shot_only_never_capture_parent(self):
        for scene in [S(shot_input_mode='SHOT_ONLY'),S(process=object()),S(inner=S(game=S(shot_input_mode='SHOT_ONLY')))]:
            self.present(scene=scene);self.assertFalse(self.rec.ring)
            self.assertEqual(r.renderer_status(scene)['status'],'UNAVAILABLE')
        self.scene=S(inner=S(game=S(shot_input_mode='SHOT_ONLY')));self.observe();row=self.finish()
        self.assertEqual(row['status'],'UNAVAILABLE');self.assertEqual(row['frames'],[])

    def test_ring_and_pinned_event_memory_bounds(self):
        with patch.object(r,'MAX_BYTES',64*48*3*2),patch.object(r,'MAX_FRAMES',2):
            for t in [99.7,99.8,99.9]:self.present(t)
            self.assertEqual(len(self.rec.ring),2);self.observe();self.present(100.1)
            self.assertEqual(len(self.rec.pending[0]['frames']),2)
            self.assertIn('EVENT_IMAGE_MEMORY_BOUND',self.rec.pending[0]['row']['missing'])

    def test_unknown_audio_clock_remains_unavailable(self):
        self.event.audio_trigger={};self.present();self.observe();row=self.finish()
        self.assertIsNone(row['frames'][0]['delta_ms_to_audio_estimate'])
        self.assertEqual(row['frames'][0]['relative_to_audio_estimate'],'UNAVAILABLE')

    def test_scene_switch_does_not_bind_old_scene(self):
        self.present();self.scene=S();self.observe();row=self.finish()
        self.assertEqual(row['frames'],[]);self.assertIn('PRESENTED_HISTORY_UNAVAILABLE',row['missing'])

    def test_interruption_retains_frames_marks_missing_post(self):
        self.present();self.observe();self.rec.close()
        row=json.loads((self.rec.root/'shot_00000001.json').read_text())
        self.assertEqual(row['status'],'INCOMPLETE');self.assertEqual(len(row['frames']),1)
        self.assertIn('INTERRUPTED_POST_PRESENTATION_WINDOW',row['missing'])

    def test_future_presentation_not_in_bound_window(self):
        self.present();self.observe();self.present(100.5)
        row=self.finish();self.assertEqual(len(row['frames']),1)

    def test_repeated_event_does_not_overwrite(self):
        self.present();self.observe();self.observe();self.assertEqual(len(self.rec.pending),1)
        row=self.finish();path=self.rec.root/'shot_00000001.json';before=path.read_bytes()
        self.rec.close();self.assertEqual(path.read_bytes(),before)

    def test_missing_geometry_explicit(self):
        self.registry.game_context=lambda sid:None
        self.present();self.observe();row=self.finish()
        self.assertEqual(row['status'],'INCOMPLETE');self.assertIn('FROZEN_SHOT_GEOMETRY_UNAVAILABLE',row['missing'])

    def test_reject_gt_or_optical_as_render(self):
        for schema in ['physical_gt_review_frame_v1','physical_optical_frame_v1']:
            with self.assertRaises(ValueError):self.rec.writer.enqueue_render_frame(self.rec.root/'bad.npy',np.zeros((3,3,3),np.uint8),{'schema':schema})

    def test_write_failure_visible_and_keeps_shot_identity(self):
        self.present();self.observe()
        with patch.object(self.rec.writer,'enqueue_render_frame',return_value=False),redirect_stdout(io.StringIO()) as out:
            self.rec.poll(101.);self.rec.close()
        self.assertIn('WARNING',out.getvalue())
        m=json.loads((self.rec.root/'manifest.json').read_text())
        self.assertEqual(m['shots'][0]['shot_id'],1);self.assertEqual(m['status'],'INCOMPLETE')
        self.assertEqual(r.audit(self.capture)['shots'][0]['status'],'UNAVAILABLE')

    def test_completed_sidecar_recovered_before_manifest_close(self):
        self.present();self.observe();self.rec.poll(101.)
        self.rec.futures[0].result()
        self.assertEqual(json.loads((self.rec.root/'manifest.json').read_text())['shots'],[])
        self.assertEqual(r.audit(self.capture)['shots'][0]['shot_id'],1)

    def test_metadata_tampering_refused(self):
        self.present();self.observe();row=self.finish()
        path=(self.rec.root/row['frames'][0]['path']).with_suffix('.json')
        md=json.loads(path.read_text());md['projectile_truth']=True;path.write_text(json.dumps(md))
        with self.assertRaises(ValueError):r.audit(self.capture)

    def test_dispatch_surface_not_claimed_pre_audio(self):
        self.present(100.005);self.observe();row=self.finish()
        self.assertEqual(row['presentations_before_audio_estimate'],0)
        self.assertEqual(row['frames'][0]['relative_to_audio_estimate'],'AFTER')
        self.assertIsNone(row['selected_render_at_physical_impact'])

    def test_backlog_keeps_unavailable_event_record(self):
        self.present()
        self.scanner.audio_events=[S(**{**self.event.__dict__,'shot_id':i}) for i in [1,2,3]]
        self.observe();self.rec.close()
        rows=r.audit(self.capture)['shots'];self.assertEqual([s['shot_id'] for s in rows],[1,2,3])
        self.assertIn('BACKGROUND_BACKLOG_BOUND',rows[2]['missing'])

    def test_diagnostic_error_does_not_escape_live_hook(self):
        with patch.object(self.rec,'presented',side_effect=RuntimeError('test')),redirect_stdout(io.StringIO()) as out:
            self.rec.call('presented')
        self.assertTrue(self.rec.failed);self.assertIn('WARNING',out.getvalue())

    def test_hash_tampering_refused(self):
        self.present();self.observe();row=self.finish()
        (self.rec.root/row['frames'][0]['path']).write_bytes(b'not an image')
        with self.assertRaises(ValueError):r.audit(self.capture)

    def test_audit_deterministic_read_only(self):
        self.present();self.observe();self.finish()
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.rec.root.rglob('*') if p.is_file()}
        self.assertEqual(r.audit(self.capture),r.audit(self.capture))
        self.assertEqual(before,{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before})

    def test_forbidden_source_rejected_before_read(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('must not read')):
            with self.assertRaises(ValueError):r.audit('/data/skjutbana/S03')

    def test_pygame_capture_main_thread_only(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            with self.assertRaises(RuntimeError):pool.submit(self.present).result()

    def test_default_capture_has_no_render_env_and_incompatible_mode_refused(self):
        with patch.dict(os.environ,{'SKJUTBANA_RENDER_HISTORY_ROOT':'inherited'}):
            _,env=prepare_can_play(Path(self.tmp.name)/'default')
        self.assertNotIn('SKJUTBANA_RENDER_HISTORY_ROOT',env)
        for kwargs in [dict(render_history=True),dict(render_history=True,object_crops=True,gt_review=True)]:
            with self.assertRaises(ValueError):prepare_can_play(Path(self.tmp.name)/'refused',**kwargs)

    def test_installed_loop_keeps_authority_order_and_default_disabled(self):
        """Execute the actual installed loop with inert device services; no hardware."""
        tree=ast.parse(Path('src/engine/shot_async_v2224.py').read_text())
        run=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_v2224')
        timing=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_loop_timing')
        sequences=[]
        for enabled in [False,True]:
            steps=[]
            mark=lambda name:lambda *a,**kw:steps.append(name)
            app=S(running=True,clock=S(tick=lambda fps:16),screen=self.screen,
                _post_automation_events=Mock(),_update_window_caption=Mock(),communication_server=S(stop=Mock()))
            app.scene=S(update=lambda dt:None,render=mark('render'),on_exit=Mock())
            def visible(*args):steps.append('visible');app.running=False
            controller=S(last_seen_peak_ts=0,pending_audio=lambda a:False,begin_pending_audio=lambda *a:1,
                prepare_object_snapshot=mark('freeze'),discover_scanner_events=mark('discover'),
                note_stage=Mock(),enrich_spatial_context=Mock(),evaluate_object_shadow=mark('object'),
                update_finished=mark('final'),should_defer_scene_work=lambda *a:False,mark_visible_after_flip=visible)
            audio=S(last_peak_ts=0,update=mark('audio'),stop=Mock())
            scanner=S(update=mark('scanner'),disable=Mock())
            camera=S(update=mark('camera'),stop=Mock())
            registry=S(advance_object_queries=mark('query'))
            adapter=S(call=lambda method,*a:steps.append(method),close=Mock())
            factory=Mock(return_value=adapter)
            modules={'pygame':S(event=S(get=lambda:[]),display=S(flip=mark('flip')),quit=Mock()),
                'config':S(FPS=60),'src.engine.app':S(AUTOMATION_EVENT=10),
                'src.engine.audio.audio_peak_detector':S(audio_peak_detector=audio),
                'src.engine.camera.camera_manager':S(camera_manager=camera),
                'src.engine.camera.hit_scanner':S(hit_scanner=scanner),
                'src.engine.output.led_service':S(led_service=S(stop=Mock())),
                'src.engine.shot_critical_v2223':S(shot_critical_controller_v2223=controller),
                'src.engine.input.object_hit_v2223':S(object_hit_registry_v2223=registry),
                'src.engine.input.hit_input':S(hit_input=self.mapper),
                'automation.physical_render_history':S(from_environment=factory)}
            ns=dict(time=time,os=os,async_detector_v2224=object(),_shutdown_workers=Mock(),
                _LOOP_STATS=dict(work=[],dt=[]))
            exec(compile(ast.Module(body=[deepcopy(timing),deepcopy(run)],type_ignores=[]),'<installed loop>','exec'),ns)
            with patch.dict(os.environ,{'SKJUTBANA_RENDER_HISTORY_ROOT':'fixture'} if enabled else {},clear=True),patch.dict('sys.modules',modules):
                ns['run_v2224'](app)
            if enabled:
                factory.assert_called_once();adapter.close.assert_called_once()
                self.assertLess(steps.index('freeze'),steps.index('observe'))
                self.assertLess(steps.index('observe'),steps.index('query'))
                self.assertLess(steps.index('flip'),steps.index('presented'))
            else:factory.assert_not_called()
            sequences.append([s for s in steps if s not in ('observe','presented')])
        self.assertEqual(sequences[0],sequences[1])


if __name__=='__main__':unittest.main()
