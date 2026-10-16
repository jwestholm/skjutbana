"""Deterministic optical instrument tests. No camera/projector/projectile validation."""
from concurrent.futures import Future
from contextlib import redirect_stdout
from copy import deepcopy
import ast
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pygame

from automation import physical_optical_lab as lab
from automation.physical_optical_runtime import OpticalController
from src.engine.events.event_bus import event_bus


def mapping(available=True):
    return dict(status='AVAILABLE' if available else 'UNAVAILABLE',viewport=[4,6,80,64],content=[0,0,80,64],screen_size=[96,80],
        camera_to_screen=[[1,0,-8],[0,1,-6],[0,0,1]] if available else None,
        screen_to_camera=[[1,0,8],[0,1,6],[0,0,1]] if available else None)


def transition(t=1):
    return dict(presented_monotonic=t,transition_id=1,render_frame_id=1,state='MID_GRAY',command_issued_wall_ts=t,main_applied_monotonic=t)


def frame(seq=1,t=2,pixels=None):
    return NS(timestamp=t,frame_bgr=np.full((16,24,3),127,np.uint8) if pixels is None else pixels,
        acquisition=dict(stream_id='fixture',sequence=seq,read_start_monotonic=t,read_complete_monotonic=t+.001,
            published_monotonic=t+.002,receive_wall_ts=t,backend_name='SYNTHETIC',backend_timestamps_raw=None))


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/data/skjutbana/evaluation_runs');self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)

    def run_fixture(self,mode='passive',**kwargs):
        c=lab.configuration(mode,**kwargs)
        r=lab.OpticalRun(self.base/'run',c,mapping());self.addCleanup(lambda:r.finish('INTERRUPTED'))
        return r

    def test_constant_and_exact_histogram_statistics(self):
        s=lab.TileStats(8)
        for v in (98,99,100,101,102):s.add(np.full((16,24,3),v,np.uint8))
        maps=s.maps();np.testing.assert_allclose(maps['mean'],100);np.testing.assert_allclose(maps['stddev'],np.sqrt(2))
        np.testing.assert_array_equal(maps['mad_1dn'],1);np.testing.assert_array_equal(maps['minimum'],98)
        self.assertLess(s.memory_bytes(),100000)

    def test_known_noise_drift_and_flicker_series(self):
        rng=np.random.default_rng(42);s=lab.TileStats(8)
        for v in rng.normal(128,3,2000):s.add(np.full((8,8,3),round(v),np.uint8))
        self.assertAlmostEqual(float(s.maps()['stddev'][0,0]),3,delta=.15)
        r=self.run_fixture()
        for i,v in enumerate((100,110,100,110,100,110)):
            r.add(frame(i+1,2+i,np.full((16,24,3),v,np.uint8)),'OPTICAL_PASSIVE',transition())
        m=r.finish();self.assertEqual(m['passive']['brightness_drift'],10)
        self.assertAlmostEqual(m['passive']['brightness_lag1_correlation'],-1)

    def test_geometry_response_and_software_roundtrip(self):
        a=np.full((100,120,3),127,np.uint8);b=a.copy();b[30:40,50:60]=255
        plan=dict(screen_center=[46.5,28.5]);r=lab.response_measure(a,b,a,mapping(),plan)
        np.testing.assert_allclose(r['measured']['camera_xy'],[54.5,34.5])
        self.assertAlmostEqual(r['geometry']['software_roundtrip_screen_px'],0)
        self.assertAlmostEqual(r['geometry']['observed_roundtrip_game_px'],0)
        # Intentionally displaced projected edge stays displaced, never snapped.
        b=a.copy();b[30:40,53:58]=255;b[30:40,58:63]=0
        r=lab.response_measure(a,b,a,mapping(),plan)
        self.assertAlmostEqual(r['geometry']['observed_roundtrip_game_px'],3,delta=.02)

    def test_missing_mapping_and_zero_response_unavailable(self):
        a=np.full((20,20,3),127,np.uint8);b=a.copy();b[3:5,3:5]=255
        r=lab.response_measure(a,b,a,mapping(False),dict(screen_center=[4,4]))
        self.assertEqual(r['geometry']['status'],'UNAVAILABLE')
        self.assertEqual(lab.response_measure(a,a,a,mapping(),dict(screen_center=[4,4]))['status'],'UNAVAILABLE')

    def test_mapping_snapshot_uses_existing_homography_no_fallback(self):
        m=NS(homography=np.eye(3),calibration_data={'method':'fixture'},_prefers_homography=lambda:True)
        snap=lab.snapshot_mapping(m,(5,6,80,60),(0,0,80,60),(96,80))
        m.homography[0,2]=20;self.assertEqual(snap['camera_to_screen'][0][2],0)
        m._prefers_homography=lambda:False
        self.assertEqual(lab.snapshot_mapping(m,(0,0,80,60),(0,0,80,60),(96,80))['status'],'UNAVAILABLE')

    def test_plan_game_viewport_screen_and_render(self):
        for p in lab.plans(mapping()):
            x,y,w,h=p['screen_rect'];self.assertGreaterEqual(x,4);self.assertGreaterEqual(y,6)
            np.testing.assert_allclose(np.subtract(p['screen_center'],p['game_rendered_center']),[4,6])
            surface=pygame.Surface((96,80));lab.render_probe(surface,p)
            a=pygame.surfarray.array3d(surface);self.assertEqual(a[0,0,0],127)
            self.assertTrue(np.any(a[x:x+w,y:y+h]!=127))
        bad=mapping();bad['viewport']=[-1,0,80,64]
        with self.assertRaises(ValueError):lab.plans(bad)

    def test_barrier_requires_real_read_start_not_receive_only(self):
        f=frame(t=1.5)
        self.assertTrue(lab.qualified_frame(f,transition(),.5))
        f.acquisition['read_start_monotonic']=1.499
        self.assertFalse(lab.qualified_frame(f,transition(),.5))
        del f.acquisition['read_start_monotonic']
        self.assertFalse(lab.qualified_frame(f,transition(),.5))

    def test_full_frame_writer_provenance_hash_and_no_truth(self):
        r=self.run_fixture();f=frame();original=f.frame_bgr.copy()
        record=r.add(f,'OPTICAL_PASSIVE',transition());m=r.finish()
        s=m['samples'][0];p=r.root/s['path'];meta=json.loads(p.with_suffix('.json').read_text())
        self.assertEqual(lab.digest(p),s['image_sha256']);np.testing.assert_array_equal(np.load(p),original)
        self.assertEqual(meta['acquisition'],f.acquisition);self.assertEqual(meta['pixel_sha256'],record['pixel_sha256'])
        self.assertFalse(meta['projectile_truth']);self.assertFalse(meta['live_authority']);self.assertIsNone(meta['shot_id'])
        self.assertEqual(meta['role'],'OPTICAL_PASSIVE');self.assertNotIn('camera_xy',meta)

    def test_gt_writer_cannot_accept_optical_roles_or_inverse(self):
        r=self.run_fixture();f=frame();meta=lab.frame_record(f,'OPTICAL_PASSIVE',transition(),'test',1)
        with self.assertRaises(ValueError):r.writer.enqueue_review_frame(r.root/'bad.npy',f.frame_bgr,meta)
        meta.update(schema='physical_gt_review_frame_v1',role='GT_SETTLED',shot_id=1)
        with self.assertRaises(ValueError):r.writer.enqueue_optical_frame(r.root/'bad.npy',f.frame_bgr,meta)

    def test_partial_interrupted_and_no_camera_readable(self):
        r=self.run_fixture();r.add(frame(),'OPTICAL_PASSIVE',transition());m=r.finish('INTERRUPTED','fixture')
        self.assertEqual(m['status'],'INTERRUPTED');self.assertEqual(m['samples'][0]['status'],'SAVED')
        self.assertTrue((r.root/'timing.csv').exists())
        q=lab.OpticalRun(self.base/'empty',lab.configuration('passive'),mapping())
        self.assertEqual(q.finish()['status'],'UNAVAILABLE')

    def test_missing_probe_and_missing_calibration_geometry_deterministic(self):
        r=self.run_fixture('probes');r.add(frame(),'OPTICAL_BASELINE',transition(),lab.plans(mapping())[0]);r.finish('INTERRUPTED')
        before={str(p):lab.digest(p) for p in r.root.rglob('*') if p.is_file()}
        a=lab.geometry_audit(r.root,self.base/'audit_a');b=lab.geometry_audit(r.root,self.base/'audit_b')
        self.assertEqual(a,b);self.assertTrue(all(p['status']=='UNAVAILABLE' for p in a['probes']))
        self.assertEqual(before,{str(p):lab.digest(p) for p in r.root.rglob('*') if p.is_file()})
        s=r.root/r.samples[0]['path'];s.write_bytes(b'corruption')
        with self.assertRaisesRegex(ValueError,'hash'):lab.geometry_audit(r.root,self.base/'audit_bad')

    def test_source_sequence_duplicate_and_stream_change_refused(self):
        r=self.run_fixture();r.add(frame(),'OPTICAL_PASSIVE',transition())
        with self.assertRaisesRegex(ValueError,'duplicate'):r.add(frame(),'OPTICAL_PASSIVE',transition())
        f=frame(2);f.acquisition['stream_id']='other'
        with self.assertRaisesRegex(ValueError,'stream'):r.add(f,'OPTICAL_PASSIVE',transition())

    def test_sample_budget_and_save_all(self):
        r=self.run_fixture()
        for i in range(15):r.add(frame(i+1,2+i),'OPTICAL_PASSIVE',transition())
        self.assertEqual(len(r.finish()['samples']),2)
        q=lab.OpticalRun(self.base/'all',lab.configuration('passive',save_all_frames=True),mapping())
        for i in range(3):q.add(frame(i+1,2+i),'OPTICAL_PASSIVE',transition())
        self.assertEqual(len(q.finish()['samples']),3)

    def test_source_guard_and_new_directory_only(self):
        with patch.object(Path,'resolve',side_effect=AssertionError('no lookup')):
            with self.assertRaises(ValueError):lab.checked('/data/skjutbana/S03')
        self.run_fixture()
        with self.assertRaises(FileExistsError):lab.OpticalRun(self.base/'run',lab.configuration('passive'),mapping())
        with self.assertRaises(ValueError):lab.output_path('/data/skjutbana/optical_wrong_root')

    def test_synthetic_probes_all_kinds_no_hardware_and_hash_replay(self):
        from src.engine.camera.camera_manager import CameraManager
        with patch.object(CameraManager,'start',side_effect=AssertionError('no device')):
            m=lab.synthetic_run(self.base/'synthetic',lab.configuration('probes',synthetic=True))
        self.assertEqual(m['status'],'COMPLETED');self.assertEqual(len(m['probes']),20);self.assertEqual(m['frames_processed'],60)
        a=lab.geometry_audit(self.base/'synthetic',self.base/'geometry')
        self.assertLess(max(p['geometry']['observed_roundtrip_game_px'] for p in a['probes']),.02)
        self.assertTrue(a['synthetic'])

    def test_runtime_main_thread_guard(self):
        ctrl=OpticalController(NS(),NS(),mapping=mapping());self.addCleanup(ctrl.close)
        ctrl.owner=-1
        with self.assertRaisesRegex(RuntimeError,'main thread'):ctrl.command(NS())
        ctrl.owner=__import__('threading').get_ident()

    def test_research_process_reuses_bounded_camera_ring(self):
        from collections import deque
        import threading
        images=[frame(i+1,2+i) for i in range(12)]
        camera=NS(_ring=deque(images,maxlen=300),_lock=threading.Lock())
        ctrl=OpticalController(NS(),camera,mapping=mapping());self.addCleanup(ctrl.close)
        self.assertEqual(camera._ring.maxlen,8);self.assertEqual(len(camera._ring),8)
        self.assertIs(camera._ring[-1],images[-1])

    def test_failed_finalization_still_closes_research_worker(self):
        ctrl=OpticalController(NS(),NS(),mapping=mapping());ctrl.pool.shutdown()
        ctrl.pool=NS(shutdown=Mock());ctrl.active=True
        ctrl.run=NS(finish=Mock(side_effect=OSError('fixture write failure')))
        with self.assertRaises(OSError):ctrl.close()
        self.assertFalse(ctrl.active);ctrl.pool.shutdown.assert_called_once_with(wait=True)

    def test_runtime_event_driven_lifecycle_no_live_calls(self):
        # Actual renderer/controller and writer; only camera and clock synthetic.
        import automation.physical_optical_runtime as runtime
        clock=NS(now=10.)
        class InlinePool:
            def submit(self, fn, *args):
                f=Future()
                try:f.set_result(fn(*args))
                except BaseException as exc:f.set_exception(exc)
                return f
            def shutdown(self,wait=True):pass
        ring=[];seq=0
        def update():
            nonlocal seq
            seq+=1;ring.append(frame(seq,clock.now))
        def tick(_):clock.now+=.1
        app=NS(screen=pygame.Surface((96,80)),clock=NS(tick=tick),_post_automation_events=Mock(),
            _handle_automation_event=Mock(),quit=Mock(),scene=NS(update=Mock(side_effect=AssertionError('game must not run'))))
        camera=NS(update=update,get_ring_snapshot=lambda:list(ring))
        ctrl=OpticalController(app,camera,mapping=mapping());ctrl.pool.shutdown();ctrl.pool=InlinePool();self.addCleanup(ctrl.close)
        events=[];event_bus.subscribe(events.append);self.addCleanup(lambda:event_bus.unsubscribe(events.append))
        command=NS(command='opticalLabStart',args=dict(run_id='test',output_dir=str(self.base/'runtime'),issued_wall_ts=1,
            config=lab.configuration('passive',duration_s=.3,settle_s=.1)),reply_success=Mock())
        pg=NS(display=NS(flip=Mock()),event=NS(get=lambda:[]),QUIT=1,KEYDOWN=2,K_ESCAPE=27)
        with patch.object(runtime,'time',NS(monotonic=lambda:clock.now,time=lambda:clock.now)):
            ctrl.command(command)
            for _ in range(20):
                ctrl.tick(pg,123)
                if not ctrl.active:break
        self.assertFalse(ctrl.active);app.scene.update.assert_not_called();command.reply_success.assert_called_once()
        self.assertIn('opticalLab.presented',[e['event'] for e in events]);self.assertIn('opticalLab.completed',[e['event'] for e in events])
        self.assertEqual(ctrl.run.manifest['status'],'COMPLETED')
        self.assertGreater(ctrl.run.frames,0)

    def test_runtime_guard_before_ordinary_pipeline_and_audio_start(self):
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'src/engine/shot_async_v2224.py').read_text())
        run=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_v2224')
        loop=next(n for n in ast.walk(run) if isinstance(n,ast.While))
        guard=loop.body[0];self.assertIn('_optical_lab',ast.unparse(guard));self.assertIn('continue',ast.unparse(guard))
        self.assertNotIn('hit_scanner',ast.unparse(guard));self.assertNotIn('audio_peak_detector.update',ast.unparse(guard))
        app=ast.parse((root/'src/engine/app.py').read_text())
        init=next(n for n in ast.walk(app) if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        conditional=next(n for n in ast.walk(init) if isinstance(n,ast.If) and ast.unparse(n.test)=='not optical')
        self.assertIn('audio_peak_detector.start()',ast.unparse(conditional))

    def test_actual_installed_loop_never_enters_live_path_in_optical_mode(self):
        import os,time
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'src/engine/shot_async_v2224.py').read_text())
        run=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_v2224')
        bad=lambda:Mock(side_effect=AssertionError('live authority reached'))
        app=NS(running=True,scene=NS(on_exit=Mock()))
        app.quit=lambda:setattr(app,'running',False)
        optical=NS(tick=Mock(side_effect=lambda *args:app.quit()),close=Mock())
        audio=NS(last_peak_ts=0,update=bad(),stop=Mock())
        scanner=NS(update=bad(),disable=Mock())
        camera=NS(update=bad(),stop=Mock())
        controller=NS(begin_pending_audio=bad(),prepare_object_snapshot=bad())
        app.communication_server=NS(stop=Mock())
        modules={'pygame':NS(quit=Mock()),'config':NS(FPS=60),'src.engine.app':NS(AUTOMATION_EVENT=10),
            'src.engine.audio.audio_peak_detector':NS(audio_peak_detector=audio),
            'src.engine.camera.camera_manager':NS(camera_manager=camera),
            'src.engine.camera.hit_scanner':NS(hit_scanner=scanner),
            'src.engine.output.led_service':NS(led_service=NS(stop=Mock())),
            'src.engine.shot_critical_v2223':NS(shot_critical_controller_v2223=controller),
            'automation.physical_optical_runtime':NS(OpticalController=Mock(return_value=optical))}
        ns=dict(time=time,os=os,async_detector_v2224=object(),_shutdown_workers=Mock())
        exec(compile(ast.Module(body=[deepcopy(run)],type_ignores=[]),'<installed loop>','exec'),ns)
        with patch.dict(os.environ,{'SKJUTBANA_OPTICAL_LAB':'1'},clear=True),patch.dict('sys.modules',modules):
            ns['run_v2224'](app)
        optical.tick.assert_called_once();optical.close.assert_called_once()
        audio.update.assert_not_called();scanner.update.assert_not_called();camera.update.assert_not_called()

    def test_existing_tcp_queue_and_event_listener_roundtrip(self):
        import time,socket
        try:
            probe=socket.socket();probe.close()
        except PermissionError:
            self.skipTest('sandbox forbids sockets; command queue/JSON parser tested separately')
        from concurrent.futures import ThreadPoolExecutor
        from src.engine.communication.communication_server import CommunicationServer
        from src.engine.communication.tcp_network_handler import EventListener,TcpNetworkHandler
        from src.engine.app import App
        server=CommunicationServer(port=0);server.start();self.addCleanup(server.stop)
        deadline=time.monotonic()+2
        # Existing server has no startup event. Only this test waits for bind;
        # experiment orchestration uses actual command/event acknowledgements.
        while server._server_socket is None and time.monotonic()<deadline:time.sleep(.005)
        self.assertIsNotNone(server._server_socket)
        port=server._server_socket.getsockname()[1]
        with EventListener(port=port) as listener,ThreadPoolExecutor(max_workers=1) as pool:
            request=pool.submit(TcpNetworkHandler(port=port).send_command,'opticalLabCancel',{'run_id':'fixture'})
            cmd=server._commands.get(timeout=2)
            def on_command(command):
                command.reply_success({'status':'CANCELLING'})
                event_bus.emit('opticalLab.completed',{'run_id':'fixture','status':'INTERRUPTED'},source='OpticalLab')
            app=NS(_optical_lab=NS(active=True,command=on_command))
            App._handle_automation_event(app,NS(automation_command=cmd))
            self.assertTrue(request.result(timeout=2)['success'])
            event=listener.wait_for_event('opticalLab.completed',timeout=2)
            self.assertEqual(event['data']['run_id'],'fixture')

    def test_existing_command_queue_and_listener_json_without_network(self):
        from src.engine.communication.communication_server import CommunicationServer,AutomationCommand
        from src.engine.communication.tcp_network_handler import EventListener
        from src.engine.app import App
        server=CommunicationServer();cmd=AutomationCommand(dict(command='opticalLabCancel',args={'run_id':'fixture'}))
        server._commands.put(cmd);events=[];event_bus.subscribe(events.append)
        try:
            def on_command(c):
                c.reply_success({'status':'CANCELLING'})
                event_bus.emit('opticalLab.completed',{'run_id':c.args['run_id'],'status':'INTERRUPTED'})
            app=NS(_optical_lab=NS(active=True,command=on_command))
            for queued in server.poll_commands():App._handle_automation_event(app,NS(automation_command=queued))
            self.assertTrue(cmd.wait_for_response(timeout=.1)['success'])
        finally:event_bus.unsubscribe(events.append)
        listener=EventListener()
        listener._socket=NS(gettimeout=lambda:None,settimeout=lambda _:None,
            recv=lambda n:(json.dumps(events[-1])+'\n').encode())
        self.assertEqual(listener.wait_for_event('opticalLab.completed',timeout=.1)['data']['run_id'],'fixture')

    def test_dropped_raw_frame_explicit_incomplete(self):
        r=self.run_fixture()
        with patch.object(r.writer,'enqueue_optical_frame',return_value=False):
            r.add(frame(),'OPTICAL_PASSIVE',transition())
        result=r.finish()
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertEqual(result['samples'][0]['status'],'DROPPED')

    def test_native_rate_backpressure_not_silently_complete(self):
        r=self.run_fixture();r.add(frame(),'OPTICAL_PASSIVE',transition())
        result=r.finish(diagnostics={'processor_backpressure_frames':2})
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertEqual(result['reason'],'PROCESSOR_BACKPRESSURE_NATIVE_RATE_NOT_SUSTAINED')


if __name__=='__main__':unittest.main()
