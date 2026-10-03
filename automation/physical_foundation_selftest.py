"""Foundation software semantics only; no physical hardware or settings writes."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch, Mock

import cv2
import numpy as np
import pygame

from src.engine.audio.audio_peak_detector import AudioPeakDetector
from src.engine.camera.camera_manager import CameraManager
from src.engine.camera.aruco_calibrator import (ArucoCalibrator, geometry_context,
    calibration_fit_quality, compare_geometry_setup)
from automation.physical_foundation_check import (analyse_sync, foundation_gate, write_new,
    geometry_report, next_creation_position, create_geometry_window)
from automation.physical_foundation_research import temporal_steps
from automation.physical_collection import launch_can


def project(matrix, point):
    p=matrix@np.array([*point,1.]);return tuple(p[:2]/p[2])


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.cal=ArucoCalibrator(pygame.Rect(20,10,760,580))
        self.h=np.array([[.5,0,-500],[0,.5,-250],[0,0,1.]])
        self.inv=np.linalg.inv(self.h)
        self.mapper=S(_canonical_screen_to_camera=lambda x,y:project(self.inv,(x,y)),
            _canonical_camera_to_screen=lambda x,y:project(self.h,(x,y)))

    def observed(self):
        return {mid:project(self.inv,p) for mid,p in self.cal.rendered_marker_positions().items()}

    def test_known_points_and_roundtrip(self):
        report=self.cal.validate_points(self.observed(),self.mapper)
        self.assertEqual(report['status'],'PASS');self.assertEqual(len(report['points']),24)
        self.assertEqual(report['max_error_camera_px'],0)
        for row in report['points']:
            np.testing.assert_allclose(np.array(row['client_xy'])-row['game_xy'],[20,10])

    def test_wrong_translation_scaling_or_inverse_fails(self):
        obs=self.observed()
        for delta,scale in (([0,18],1),([0,0],1.03)):
            bad={mid:(np.array(p)*scale+delta).tolist() for mid,p in obs.items()}
            self.assertEqual(self.cal.validate_points(bad,self.mapper)['status'],'FAIL')
        wrong=S(_canonical_screen_to_camera=self.mapper._canonical_screen_to_camera,
            _canonical_camera_to_screen=lambda x,y:tuple(np.array(project(self.h,(x,y)))+[4,0]))
        self.assertEqual(self.cal.validate_points(obs,wrong)['status'],'FAIL')

    def test_partial_corner_cluster_cannot_pass(self):
        self.assertEqual(self.cal.validate_points({0:self.observed()[0]},self.mapper)['status'],'FAIL')

    def test_unavailable_reverse_mapping_fails_with_serializable_report(self):
        mapper=S(_canonical_screen_to_camera=self.mapper._canonical_screen_to_camera,
            _canonical_camera_to_screen=lambda x,y:None)
        report=self.cal.validate_points(self.observed(),mapper)
        self.assertEqual(report['status'],'FAIL')
        json.dumps(report,allow_nan=False)

    def test_calibration_uses_one_fit_and_mathematical_inverse(self):
        points=self.observed();ids=np.array(list(points),np.int32).reshape(-1,1)
        corners=[np.array([[np.array(p)+d for d in ((-10,-10),(10,-10),(10,10),(-10,10))]],np.float32) for p in points.values()]
        with patch.object(self.cal,'detect_markers',return_value=(corners,ids,[])),patch.object(cv2,'findHomography',wraps=cv2.findHomography) as fit:
            result=self.cal.detect_and_calibrate(np.zeros((2160,3840,3),np.uint8))
        self.assertTrue(result.success);self.assertEqual(fit.call_count,1)
        h=np.array(result.calibration_dict['homography']);inv=np.array(result.calibration_dict['inverse_homography'])
        np.testing.assert_allclose(h@inv/(h@inv)[2,2],np.eye(3),atol=1e-10)
        self.assertEqual(result.calibration_dict['render_correspondence'],'integer_blit_pixel_centers')

    def test_rendered_pixel_center_is_actual_integer_blit(self):
        for mid,p in self.cal.marker_positions().items():
            size=self.cal.marker_size_px();r=pygame.Rect(0,0,size,size);r.center=tuple(map(int,p))
            self.assertEqual(self.cal.rendered_marker_positions()[mid],(r.x+(size-1)/2,r.y+(size-1)/2))

    def test_window_position_recorded_not_added_to_camera_mapping(self):
        pygame.display.init();pygame.display.set_mode((800,600))
        try:
            from pygame._sdl2.video import Window
            window=Window.from_display_module();window.position=(120,80)
            a=geometry_context();camera=self.mapper._canonical_screen_to_camera(200,200)
            window.position=(2130,50);b=geometry_context()
            self.assertNotEqual(a['identity_sha256'],b['identity_sha256'])
            self.assertEqual(camera,self.mapper._canonical_screen_to_camera(200,200))
            self.assertEqual(b['screen_space'],'PYGAME_CLIENT_PIXELS_NOT_DESKTOP')
            self.assertFalse(b['window']['borderless'])
        finally:pygame.display.quit()

    def test_peak_public_timestamp_derivation_unchanged(self):
        audio=AudioPeakDetector();data=np.zeros(256);data[10]=.8;data[100]=.9
        self.assertEqual(audio._estimate_event_timestamp(data,20.,.7),20.-245/16000)
        self.assertEqual(audio._peak_sample_index(data,.95),100)

    def test_short_reads_preserve_all_bytes_including_partial_samples(self):
        payload=np.arange(256,dtype=np.int16).tobytes()
        class Fragmented(io.BytesIO):
            def read(self,n):return super().read(min(n,17))
        audio=AudioPeakDetector();actual,meta=audio._read_pcm_block(Fragmented(payload))
        self.assertEqual(actual,payload);self.assertTrue(meta['complete_block']);self.assertGreater(meta['read_calls'],1)
        partial,meta=audio._read_pcm_block(io.BytesIO(payload[:31]))
        self.assertFalse(meta['complete_block']);self.assertEqual(len(partial),31)

    def test_buffered_receipt_is_not_rewritten_as_acquisition(self):
        audio=AudioPeakDetector();audio.min_abs_peak=.5
        data=np.zeros(256,np.int16);data[10]=30000
        receipt=dict(read_start_monotonic=10.,read_complete_monotonic=10.01,receive_wall_ts=100.,read_calls=1,complete_block=True)
        with patch('time.time',return_value=200.):audio._deliver_pcm_block(data.tobytes(),receipt)
        event=audio.get_latest_event();self.assertAlmostEqual(event.timestamp,200.-245/16000)
        self.assertEqual(event.acquisition['sample_start_index'],0)
        self.assertEqual(event.acquisition['trigger_sample_offset'],10)
        self.assertAlmostEqual(event.acquisition['trigger_receipt_estimate_monotonic'],10.01-245/16000)
        self.assertIsNone(event.acquisition['upstream_buffer_age_s'])

    def test_installed_audio_patch_keeps_acquisition_before_dispatch(self):
        from src.engine.shot_track_v2226 import _install_audio_telemetry_patch
        _install_audio_telemetry_patch()
        audio=AudioPeakDetector();audio.min_abs_peak=.5
        data=np.zeros(256,np.int16);data[9]=30000
        receipt=dict(read_start_monotonic=10.,read_complete_monotonic=10.02,receive_wall_ts=100.,read_calls=2,complete_block=True)
        audio._deliver_pcm_block(data.tobytes(),receipt)
        event=audio.pending_events_snapshot()[0]
        self.assertEqual(event.diagnostics['acquisition']['trigger_sample_offset'],9)
        audio.update();self.assertIn('dispatch_monotonic',event.acquisition)

    def test_pcm_sequence_gaps_and_stream_reset_are_visible(self):
        audio=AudioPeakDetector();audio._process_chunk=Mock()
        for i in range(514):audio._deliver_pcm_block(bytes(512),dict(read_start_monotonic=10+i*.016,read_complete_monotonic=10.01+i*.016))
        blocks=audio.acquisition_blocks_since();self.assertEqual(len(blocks),512)
        self.assertEqual(blocks[0][0]['sequence'],3);self.assertEqual(blocks[-1][0]['sample_end_index_exclusive'],514*256)
        sid=audio._stream_id;audio._reset_acquisition();self.assertNotEqual(sid,audio._stream_id)

    def test_camera_provenance_and_public_timestamp(self):
        camera=CameraManager();camera.cap=S(get=lambda p:0.)
        pixels=np.zeros((24,32,3),np.uint8)
        a=camera._timestamped_frame(pixels,10.,10.08,100.)
        b=camera._timestamped_frame(pixels,10.09,10.17,100.09)
        self.assertEqual(a.timestamp,100.);self.assertEqual(b.acquisition['sequence'],2)
        self.assertEqual(a.acquisition['read_complete_monotonic'],10.08)
        self.assertIsNone(a.acquisition['exposure_monotonic'])
        self.assertEqual(a.acquisition['frame_shape'],list(a.frame_bgr.shape))
        camera._ring.append(a);self.assertEqual(camera.provenance_for_timestamp(100.)['sequence'],1)
        camera._fresh_frame=b;self.assertEqual(camera.provenance_for_timestamp(100.09)['sequence'],2)
        self.assertIsNone(camera.provenance_for_timestamp(99.))

    def test_event_specific_step_uses_preceding_segment_not_oldest(self):
        images=np.full((12,64,64),140,np.float32);images[6:,29:34,29:34]=80
        rows=temporal_steps(images,np.ones((64,64),np.uint8),list(range(12)))
        self.assertTrue(rows);self.assertEqual(rows[0]['suggested_pre_ts'],5)
        self.assertEqual(rows[0]['appearance_bracket'],[5,6]);self.assertEqual(rows[0]['cutoff_ts'],11)
        self.assertEqual(temporal_steps(images[:6],np.ones((64,64),np.uint8),list(range(6))),[])

    def test_sync_reports_receipt_offset_and_missing_signal(self):
        commands=[dict(flip_complete_monotonic=t) for t in (2.,7.,12.)]
        audio=[];visual=[]
        for i,t in enumerate(np.arange(0,16,.02)):
            on=any(0<=t-c['flip_complete_monotonic']-.2<.2 for c in commands)
            audio.append(dict(read_complete_monotonic=t,rms=.5 if on else .01,sample_start_index=i*320,sample_end_index_exclusive=(i+1)*320))
            bright=any(0<=t-c['flip_complete_monotonic']-.1<.6 for c in commands)
            visual.append(dict(read_complete_monotonic=t,mean=200 if bright else 30,sequence=i))
        r=analyse_sync(audio,visual,commands);self.assertEqual(r['status'],'CHARACTERIZED_RECEIPT_ONLY')
        self.assertAlmostEqual(r['median_audio_minus_visual_ms'],100,delta=21)
        self.assertFalse(r['calibrated_device_offset']);self.assertFalse(r['pre_selection_demonstrated'])
        self.assertEqual(analyse_sync([],visual,commands)['status'],'INCOMPLETE')

    def test_failed_or_missing_foundations_block_launch_before_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'foundation preflight required'):launch_can(root/'run',root/'traces')
            self.assertFalse((root/'run').exists())
            path=root/'blocked.json';write_new(path,foundation_gate({}, {}, {}))
            with self.assertRaisesRegex(ValueError,'BLOCKED'):launch_can(root/'run',root/'traces',path)
            self.assertFalse((root/'traces').exists())

    def test_receipt_characterization_alone_never_passes_preflight(self):
        context=dict(identity_sha256='same')
        geo=dict(schema='foundation-geometry-1',status='PASS',unchanged_setup=True,context=context)
        timing=dict(status='CHARACTERIZED_RECEIPT_ONLY',unchanged_setup=True,context=context)
        result=foundation_gate(geo,timing,{})
        self.assertFalse(result['ready_to_shoot'])
        self.assertTrue(any('PRE' in r for r in result['reasons']))

    def test_changed_window_or_calibration_identity_blocks_preflight(self):
        context=dict(identity_sha256='measured')
        geo=dict(schema='foundation-geometry-1',status='PASS',unchanged_setup=True,context=context)
        timing=dict(status='CHARACTERIZED_RECEIPT_ONLY',unchanged_setup=True,context=context)
        result=foundation_gate(geo,timing,{},dict(identity_sha256='moved_window'))
        self.assertIn('current setup differs from measured geometry',result['reasons'])

    def test_cli_failure_is_nonzero_and_invalid_sync_never_opens_hardware(self):
        from automation.physical_foundation_check import main, run_live
        with patch('sys.argv',['check','geometry','--output','unused']),patch(
                'automation.physical_foundation_check.run_live',return_value=dict(status='FAIL')):
            self.assertEqual(main(),2)
        with tempfile.TemporaryDirectory() as tmp,patch('pygame.init') as start:
            with self.assertRaises(ValueError):run_live('sync',Path(tmp)/'no_output',repetitions=0)
            start.assert_not_called()

    def geometry_fixture(self):
        from copy import deepcopy
        setup=dict(viewport=[20,10,760,580],content_rect=[0,0,760,580],
            camera_orientation=dict(rotation=180,mirror_horizontal=False,mirror_vertical=False),
            screen_space='PYGAME_CLIENT_PIXELS_NOT_DESKTOP',window=dict(client_surface_size=[800,600],
                sdl_window_size=[800,600],sdl_desktop_position=[2135,95],borderless=False,
                display_index=1,driver='x11',desktop_sizes=[[1920,1080],[1024,768]],display_flags=0))
        context=deepcopy(setup)
        points=self.observed()
        context['calibration']=dict(setup_at_calibration=setup,homography=self.h.tolist(),
            inverse_homography=self.inv.tolist(),camera_points=list(points.values()),
            viewport_points=list(self.cal.rendered_marker_positions().values()),
            matched_ids=list(points),camera_frame_size=[3840,2160])
        return context

    def test_measured_window_request_correction_has_no_magic_offset(self):
        self.assertEqual(next_creation_position([2135,95],[2135,95],[2135,132]),[2135,58])
        self.assertEqual(next_creation_position([900,200],[900,200],[908,224]),[892,176])
        self.assertEqual(next_creation_position([900,200],[900,200],[900,200]),[900,200])

    def test_window_recreation_is_bounded_and_never_uses_position_setter(self):
        screen=Mock();actuals=iter(([100,130],[100,100]))
        state={}
        def mode(size):
            state['actual']=next(actuals);return screen
        def context(**kwargs):return dict(window=dict(sdl_desktop_position=state['actual']))
        with patch('pygame.display.set_mode',side_effect=mode) as create,patch('pygame.display.flip'),\
             patch('pygame.display.quit') as close,patch('pygame.display.init'),patch('pygame.display.set_caption'),\
             patch('pygame.event.get',return_value=[]),patch('pygame.time.wait'),\
             patch('automation.physical_foundation_check.geometry_context',side_effect=context),\
             patch('time.monotonic',side_effect=(i*.1 for i in range(100))),patch.dict(os.environ,{},clear=False):
            _,report=create_geometry_window((800,600),[100,100])
        self.assertEqual(create.call_count,2);close.assert_called_once()
        self.assertTrue(report['verified']);self.assertEqual(report['attempts'][1]['creation_request'],[100,70])

    def test_window_client_display_and_orientation_mismatch(self):
        from copy import deepcopy
        original=self.geometry_fixture();setup=original['calibration']['setup_at_calibration']
        self.assertEqual(compare_geometry_setup(setup,original)['status'],'MATCH')
        for key,value in [('sdl_desktop_position',[2135,132]),('client_surface_size',[800,637]),
                          ('sdl_window_size',[1024,768]),('display_index',0),('borderless',True)]:
            changed=deepcopy(original);changed['window'][key]=value
            self.assertEqual(compare_geometry_setup(setup,changed)['status'],'SETUP_MISMATCH')
        changed=deepcopy(original);changed['camera_orientation']['rotation']=0
        self.assertEqual(compare_geometry_setup(setup,changed)['status'],'SETUP_MISMATCH')
        changed=deepcopy(original);changed['window']['x11_client_origin']=[2135,132]
        self.assertEqual(compare_geometry_setup(setup,changed)['status'],'SETUP_MISMATCH')

    def test_geometry_classification_priority_and_matching_transform_unchanged(self):
        from copy import deepcopy
        c=self.geometry_fixture();before=json.dumps(c,sort_keys=True);frames=[dict(status='PASS')]*3
        self.assertEqual(geometry_report(c,c,frames)['status'],'PASS')
        self.assertEqual(json.dumps(c,sort_keys=True),before)
        self.assertEqual(geometry_report(c,c,[dict(status='FAIL')]*3)['status'],'PHYSICAL_GEOMETRY_FAIL')
        poor=deepcopy(c);poor['calibration']['camera_points'][0]=[2000,2000]
        self.assertEqual(geometry_report(poor,poor,frames)['status'],'CALIBRATION_FIT_POOR')
        poor['window']['sdl_desktop_position'][1]+=37
        result=geometry_report(poor,poor,frames)
        self.assertEqual(result['status'],'SETUP_MISMATCH')
        self.assertEqual(result['calibration_fit']['status'],'CALIBRATION_FIT_POOR')
        self.assertFalse(result['measurement_valid_for_calibrated_setup'])

    def test_fit_audit_includes_rejected_markers_in_camera_units(self):
        c=self.geometry_fixture()['calibration'];self.assertEqual(calibration_fit_quality(c)['status'],'PASS')
        # A mean client score / RANSAC inlier count cannot excuse a bad marker.
        c['camera_points'][0]=np.array(c['camera_points'][0])+[4,0]
        c['reprojection_error_px']=.05;c['inliers']=23
        result=calibration_fit_quality(c)
        self.assertEqual(result['status'],'CALIBRATION_FIT_POOR');self.assertAlmostEqual(result['max_camera_px'],4)
        self.assertTrue(result['evaluated_all_markers'])

    def test_main_loop_never_accesses_capture_backend(self):
        from src.engine.camera.camera_manager import CameraFrame
        camera=CameraManager();camera.running=True;camera.cap=Mock()
        camera._ring.append(CameraFrame(np.zeros((2,2,3),np.uint8),1.))
        import importlib
        module=importlib.import_module('src.engine.camera.camera_manager')
        with patch.object(module,'probe_camera_capabilities') as probe:
            camera.update();camera.update();probe.assert_not_called()
        camera.cap.get.assert_not_called();camera.cap.read.assert_not_called()
        self.assertIsNotNone(camera.latest_frame)

    def test_capabilities_are_probed_before_reader_start(self):
        import importlib
        module=importlib.import_module('src.engine.camera.camera_manager')
        camera=CameraManager();cap=Mock();cap.get.return_value=1.;order=[]
        thread=Mock();thread.start.side_effect=lambda:order.append('reader_start')
        with patch.object(module.cv2,'VideoCapture',return_value=cap),\
             patch.object(module,'apply_preferred_camera_settings'),\
             patch.object(module,'probe_camera_capabilities',side_effect=lambda c:order.append('probe')),\
             patch.object(module.threading,'Thread',return_value=thread):
            self.assertTrue(camera.start())
        self.assertEqual(order,['probe','reader_start'])

    def test_setup_mismatch_stops_before_camera_start(self):
        from automation.physical_foundation_check import run_live
        c=self.geometry_fixture();c['window']['sdl_desktop_position'][1]+=37
        with tempfile.TemporaryDirectory() as tmp,patch('pygame.init'),patch('pygame.quit'),\
             patch('pygame.display.set_caption'),patch('automation.physical_foundation_check.create_geometry_window',return_value=(Mock(),{})),\
             patch('automation.physical_foundation_check.geometry_context',return_value=c),\
             patch.object(CameraManager,'start') as start:
            result=run_live('geometry',Path(tmp)/'check',window_position=[2135,95])
            start.assert_not_called();self.assertEqual(result['status'],'SETUP_MISMATCH')

    def test_x11_provenance_reads_client_origin_and_real_decorations(self):
        from src.engine.camera.aruco_calibrator import x11_window_provenance
        outputs=[S(stdout='_NET_FRAME_EXTENTS(CARDINAL) = 5, 5, 37, 5'),
            S(stdout='Absolute upper-left X: 2135\nAbsolute upper-left Y: 95\n')]
        with patch('pygame.display.get_wm_info',return_value=dict(window=123)),\
             patch('src.engine.camera.aruco_calibrator.subprocess.run',side_effect=outputs) as call:
            p=x11_window_provenance()
        self.assertEqual(p['decoration_extents'],dict(left=5,right=5,top=37,bottom=5))
        self.assertEqual(p['x11_client_origin'],[2135,95])
        self.assertEqual([c.args[0][0] for c in call.call_args_list],['xprop','xwininfo'])

    def test_changed_frame_dimensions_are_setup_mismatch(self):
        c=self.geometry_fixture()
        frames=[dict(status='PASS',acquisition=dict(frame_shape=[1080,1920,3]))]*3
        self.assertEqual(geometry_report(c,c,frames)['status'],'SETUP_MISMATCH')


if __name__=='__main__':unittest.main()
