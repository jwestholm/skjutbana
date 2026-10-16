"""Synthetic independent-click contracts; never label or inspect physical holes."""
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from automation import physical_trace_label_crops as c
from automation.physical_trace_label import AnnotationDataError, _run_target_ui, _pygame_surface
from automation.object_hit_replay_selftest import capture
from automation.object_candidate_reconciliation import paired_analysis, saved_pool_oracle
from automation.object_candidate_reconciliation_selftest import packet, candidate


class CropLabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/data/skjutbana');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'capture';t=capture(self.source)
        t['coordinate_space']='camera'
        self.trace=self.source/'object_crops/shots/shot_00000001/trace.json'
        self.trace.write_text(json.dumps(t))
        log=self.source/'runtime.log';log.write_text(log.read_text()+'[V2.22.3 SHOT] shot=6\n')
        self.m=c.load_capture(self.source);self.shot=self.m['shots'][0]
        self.post=next(f['crops'][0] for f in reversed(self.shot['frames']) if f['kind']=='post')
        self.pre=[f['crops'][0] for f in self.shot['frames'] if f['kind']=='pre']
        self.path=self.root/'labels/exact_camera_ground_truth.json'

    def row(self):
        click=c.crop_click((10,20),(0,0),(1.,1.),self.post)
        return c.make_label(self.m,self.shot,self.post,click,self.pre)

    def test_explicit_crop_origin(self):
        e=dict(crop_origin=[100,200],crop_size=[101,77],full_camera_size=[1000,1000])
        r=c.crop_click((10,20),(0,0),(1.,1.),e)
        self.assertEqual(r['camera_xy'],[110,220]);self.assertEqual(r['crop_local_xy'],[10,20])

    def test_odd_size_resized_zoom_pan_transform(self):
        e=dict(crop_origin=[101,203],crop_size=[103,77],full_camera_size=[1000,1000])
        scale,size,origin=c.view_transform(e,(751,511),2.3,(-40,32))
        r=c.crop_click(tuple(origin[i]+[25.2,30.5][i]*scale[i] for i in (0,1)),origin,scale,e)
        np.testing.assert_allclose(r['camera_xy'],[126.2,233.5],rtol=0,atol=1e-12)

    def test_boundaries_and_outside_refused(self):
        e=dict(crop_origin=[100,200],crop_size=[101,77],full_camera_size=[1000,1000])
        self.assertEqual(c.crop_click((0,0),(0,0),1.,e)['camera_xy'],[100,200])
        self.assertEqual(c.crop_click((100,76),(0,0),1.,e)['camera_xy'],[200,276])
        for xy in [(-.1,1),(101,76),(0,77),(float('nan'),1),(1,float('inf'))]:
            with self.assertRaises(ValueError):c.crop_click(xy,(0,0),1.,e)

    def test_missing_origin_refuses_label(self):
        e=deepcopy(self.post);del e['crop_origin']
        with self.assertRaises(ValueError):c.crop_click((1,1),(0,0),1.,e)
        t=json.loads(self.trace.read_text());t['object_evidence_crops'][0].pop('camera_bbox');self.trace.write_text(json.dumps(t))
        with self.assertRaises(ValueError):c.load_capture(self.source)

    def test_integral_float_bounds_preserve_coordinates_fractional_refused(self):
        t=json.loads(self.trace.read_text());t['object_evidence_crops'][0]['camera_bbox']=list(map(float,t['object_evidence_crops'][0]['camera_bbox']))
        self.trace.write_text(json.dumps(t));m=c.load_capture(self.source)
        self.assertEqual(m['shots'][0]['tiles'],self.shot['tiles'])
        t['object_evidence_crops'][0]['camera_bbox'][0]+=.5;self.trace.write_text(json.dumps(t))
        with self.assertRaises(ValueError):c.load_capture(self.source)

    def test_save_reload_history_exact_and_source_untouched(self):
        before=deepcopy(self.m['input_hashes'])
        row=self.row();one=c.save_row(self.path,self.m,row,0)
        self.assertEqual(json.loads(self.path.read_text()),one)
        self.assertEqual(c.validate_document(one,c.load_capture(self.source)),one)
        two=c.save_row(self.path,self.m,c.unknown_row(1,'NOT_LABELABLE'),1)
        self.assertEqual(two['history'][-1]['previous'],row)
        self.assertEqual(two['shots'][0]['physical_truth_status'],'UNKNOWN')
        three=c.save_row(self.path,self.m,row,2)
        self.assertEqual(three['revision'],3);self.assertEqual(len(three['history']),3)
        self.assertTrue(all(c.digest(p)==h for p,h in before.items()))

    def test_save_refuses_changed_source_file(self):
        p=self.source/self.post['path'];a=np.load(p);a[0,0]^=1;np.save(p,a)
        with self.assertRaisesRegex(ValueError,'changed'):c.save_row(self.path,self.m,self.row())
        self.assertFalse(self.path.exists())
        with self.assertRaisesRegex(ValueError,'hash'):c.load_capture(self.source)

    def test_source_frame_binding_and_transform_tampering_refused(self):
        doc=c.new_document(self.m);doc['shots'][0]=self.row()
        for field,value in [('camera_xy',[1.,2.]),('trace_sha256','wrong'),('candidate_overlay',True),
                            ('clicked_view','DIFF'),('operator_method','DETECTOR'),('pre_inspected',[])]:
            bad=deepcopy(doc);bad['shots'][0][field]=value
            with self.assertRaises(ValueError):c.validate_document(bad,self.m)
        bad=deepcopy(doc);bad['shots'][0]['source']['source_image_sha256']='wrong'
        with self.assertRaises(ValueError):c.validate_document(bad,self.m)

    def test_only_post_and_inspected_pre_can_create_exact(self):
        with self.assertRaises(ValueError):c.make_label(self.m,self.shot,self.pre[0],{},self.pre)
        with self.assertRaises(ValueError):c.make_label(self.m,self.shot,self.post,{},[])

    def test_unavailable_six_unknown_not_a_negative(self):
        self.assertEqual(self.m['shots'][1]['availability'],'UNAVAILABLE')
        self.assertEqual(c.ui_frames(c.label_shots(self.m)[1],0,-1,-1),(None,None,0,0))
        doc=c.new_document(self.m)
        self.assertEqual(doc['shots'][1]['physical_event'],'UNKNOWN')
        bad=self.row();bad['shot_id']=bad['event_id']=6;doc['shots'][1]=bad
        with self.assertRaises(ValueError):c.validate_document(doc,self.m)

    def test_candidate_data_not_loaded_or_exposed(self):
        t=json.loads(self.trace.read_text())
        t['object_query_diagnostic']={'candidates':'POISON','camera_xy':[9e9,9e9]}
        t['stages']=[{'heatmap':'POISON'}];self.trace.write_text(json.dumps(t))
        folder=self.trace.parent/'mature_candidates';folder.mkdir();(folder/'pool_0000.json').write_text('INVALID JSON NEVER READ')
        with patch('automation.physical_trace_label.overlay_candidates',side_effect=AssertionError('no overlays')):
            m=c.load_capture(self.source);shots=c.label_shots(m)
        self.assertNotIn('POISON',json.dumps(shots));self.assertNotIn('causal_overlay',json.dumps(shots))
        self.assertFalse(any('mature_candidates' in p for p in m['input_hashes']))

    def test_overlays_cannot_be_enabled(self):
        with self.assertRaisesRegex(ValueError,'forbids'):
            _run_target_ui(c.label_shots(self.m),(800,600),True,None,(300,300),paired=(self.m,self.path))

    def test_original_pixels_not_contrast_stretched(self):
        import pygame
        with patch.dict(os.environ,SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy'):
            pygame.init()
            try:
                a=np.array([[100,101],[102,103]],np.uint8)
                s=_pygame_surface(pygame,a,(2,2),original=True)
                np.testing.assert_array_equal(pygame.surfarray.array3d(s)[...,0].T,a)
            finally:pygame.quit()

    def test_s03_refused_before_io(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('no reads')):
            with self.assertRaises(PermissionError):c.load_capture('/data/skjutbana/S03')

    def test_separate_output_and_revision_guard(self):
        with self.assertRaises(ValueError):c.save_row(self.source/'gt.json',self.m,self.row())
        c.save_row(self.path,self.m,self.row())
        with self.assertRaisesRegex(ValueError,'another viewer'):c.save_row(self.path,self.m,self.row(),0)

    def test_atomic_interruption_preserves_previous_label(self):
        c.save_row(self.path,self.m,self.row());old=self.path.read_bytes()
        with patch.object(c.os,'replace',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):c.save_row(self.path,self.m,c.unknown_row(1))
        self.assertEqual(self.path.read_bytes(),old)
        self.assertFalse(list(self.path.parent.glob('.exact_gt_*')))

    def test_frame_grouping_tiles_not_independent_observations(self):
        t=json.loads(self.trace.read_text());entry=deepcopy(t['object_evidence_crops'][0])
        p=self.trace.parent/'copy.npy';a=np.zeros((2,3),np.uint8);np.save(p,a)
        entry.update(path='copy.npy',camera_bbox=[0,0,3,2],shape=[2,3],sha256=hashlib.sha256(a.tobytes()).hexdigest())
        t['object_evidence_crops'].append(entry);self.trace.write_text(json.dumps(t))
        m=c.load_capture(self.source)
        self.assertEqual(len(m['shots'][0]['frames']),len(self.shot['frames']))
        self.assertEqual(len(m['shots'][0]['frames'][0]['crops']),2)

    def test_contact_sheets_and_manifest_unknown_no_source_mutation(self):
        out=c.prepare_package(self.m,self.root/'package')
        self.assertTrue((out/'shot_006/UNAVAILABLE.txt').exists())
        self.assertTrue((out/'shot_001/tile_01.png').exists())
        self.assertEqual(json.loads((out/'truth_status.json').read_text())['shots'][0]['physical_truth_status'],'UNKNOWN')
        self.assertFalse((out/'exact_camera_ground_truth.json').exists());c.assert_inputs(self.m)
        with self.assertRaises(FileExistsError):c.prepare_package(self.m,out)

    def test_full_paired_ui_click_save_unknown_reopen(self):
        import pygame
        post=self.post
        scale,_,origin=c.view_transform(post,(800,490))
        origin=(origin[0],origin[1]+70)
        pos=tuple(round(origin[i]+[10,20][i]*scale[i]) for i in (0,1))
        def key(k):return pygame.event.Event(pygame.KEYDOWN,key=k)
        events=[[key(pygame.K_c)],[key(pygame.K_4)],[key(pygame.K_3)],
                [pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=pos)],
                [key(pygame.K_RETURN)],[key(pygame.K_RIGHT)],[key(pygame.K_u)],[key(pygame.K_q)]]
        with patch.dict(os.environ,SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy'), \
             patch.object(pygame.event,'get',side_effect=events), \
             patch('automation.physical_trace_label.overlay_candidates',side_effect=AssertionError('no candidates')):
            _run_target_ui(c.label_shots(self.m),(800,600),False,None,(300,300),paired=(self.m,self.path))
        doc=c.validate_document(json.loads(self.path.read_text()),self.m)
        self.assertEqual(doc['shots'][0]['physical_truth_status'],'EXACT_CAMERA_XY')
        self.assertFalse(doc['shots'][0]['candidate_overlay'])
        self.assertEqual(doc['shots'][1]['reason'],'IMAGE_DATA_UNAVAILABLE')
        c.assert_inputs(self.m)

    def test_paired_analyzer_valid_exact_gt_and_deterministic_oracle(self):
        # Synthetic fixture labels only; physical capture never receives a test label.
        folder=self.trace.parent/'mature_candidates';folder.mkdir()
        row=self.row();x,y=row['camera_xy']
        p=packet([candidate(x=x+3,y=y)]);p['frame_ts']=p['candidates'][0]['timestamp']=2.065
        (folder/'pool_0000.json').write_text(json.dumps(p))
        c.save_row(self.path,self.m,row)
        with redirect_stdout(io.StringIO()):
            a=paired_analysis(self.source,self.root/'a',exact_camera_gt=self.path)
            b=paired_analysis(self.source,self.root/'b',exact_camera_gt=self.path)
        self.assertEqual((a/'oracle_summary.json').read_bytes(),(b/'oracle_summary.json').read_bytes())
        data=json.loads((a/'oracle_summary.json').read_text())
        lc=next(r for r in data['oracle'] if r['stage']=='LOCAL_CONFIRMATION')
        self.assertEqual(lc['oracle_at5'],1)
        worker=next(r for r in data['oracle'] if r['stage']=='WORKER_DELIVERY')
        self.assertEqual(worker['labelled_available_events'],0)
        self.assertIn('UNAVAILABLE_POOL',(a/'candidate_nearest_gt.csv').read_text())
        self.assertEqual(json.loads((a/'truth_status.json').read_text())['shots'][0]['physical_truth'],'UNKNOWN')

    def test_incomplete_gt_rejected_unknown_produces_no_oracle(self):
        doc=c.new_document(self.m);self.path.parent.mkdir();self.path.write_text(json.dumps(doc))
        with redirect_stdout(io.StringIO()):out=paired_analysis(self.source,self.root/'unknown',exact_camera_gt=self.path)
        self.assertFalse((out/'oracle_summary.json').exists())
        doc['shots'][0]=self.row();del doc['shots'][0]['source'];self.path.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):paired_analysis(self.source,self.root/'invalid',exact_camera_gt=self.path)

    def test_oracle_not_selected_and_native_order_not_score_rank(self):
        r=saved_pool_oracle([(100,100),(3,4)],(0,0))
        self.assertEqual(r['nearest_candidate_order'],2);self.assertEqual(r['oracle_at5'],1)
        self.assertEqual(r['nearest_distance_px'],5.)
        self.assertEqual(saved_pool_oracle([],None)['status'],'UNAVAILABLE_GT')
        self.assertEqual(saved_pool_oracle([], (0,0))['oracle_at5'],0)

    def test_current_emission_metrics_remain_separate_from_oracle(self):
        row=self.row();c.save_row(self.path,self.m,row)
        log=self.source/'runtime.log'
        log.write_text(log.read_text()+'[OBJECT-HIT-CURRENT] '+json.dumps(dict(shot_id=1,
            exact_coordinate_result=dict(camera_xy=row['camera_xy'],emitted=True,state='matched')))+'\n')
        # Rebind synthetic fixture after adding independently recorded CURRENT log.
        self.m=c.load_capture(self.source)
        with redirect_stdout(io.StringIO()):out=paired_analysis(self.source,self.root/'selected',exact_camera_gt=self.path)
        result=json.loads((out/'oracle_summary.json').read_text())
        self.assertEqual(result['selected_current_separate'][0]['selected_at5'],1)
        self.assertNotIn('selected_at5',result['oracle'][0])

    def test_no_physical_labels_from_prepare_and_ui_quit(self):
        import pygame
        with patch.dict(os.environ,SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy'), \
             patch.object(pygame.event,'get',return_value=[pygame.event.Event(pygame.QUIT)]):
            _run_target_ui(c.label_shots(self.m),(800,600),False,None,(300,300),paired=(self.m,self.path))
        self.assertFalse(self.path.exists())

    def test_no_s03_crop_reference_even_resolve_is_not_called(self):
        with patch.object(Path,'resolve',side_effect=AssertionError('no filesystem traversal')):
            with self.assertRaises(PermissionError):c.checked('/data/skjutbana/S03/image.npy')

    def test_pre_or_blink_view_click_cannot_save(self):
        import pygame
        def key(k):return pygame.event.Event(pygame.KEYDOWN,key=k)
        click=pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=(400,300))
        events=[[key(pygame.K_4)],[click],[key(pygame.K_RETURN)],[key(pygame.K_b)],[click],[key(pygame.K_RETURN)],[key(pygame.K_q)]]
        with patch.dict(os.environ,SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy'), \
             patch.object(pygame.event,'get',side_effect=events):
            _run_target_ui(c.label_shots(self.m),(800,600),False,None,(300,300),paired=(self.m,self.path))
        self.assertFalse(self.path.exists())


if __name__=='__main__':unittest.main()
