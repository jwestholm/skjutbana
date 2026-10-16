"""GT-review CLI/UI regression using saved shot_XXXXXXXX/{white,...}.npy.

All clicks and saved labels in this file use generated pixels, never physical GT.
"""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
import unittest
from unittest.mock import patch

from automation import physical_gt_capture_selftest as fixture
from automation import physical_trace_label as label
from automation import physical_trace_label_crops as crops


class GTLabelStartupTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing writer/lifecycle fixture to match the real capture
        # layout and provenance, rather than inventing another image schema.
        self.fx=fixture.ReviewTests();self.fx.setUp();self.addCleanup(self.fx.doCleanups)
        f=self.fx
        for sid in (1,2,3):
            f.event.shot_id=sid
            if sid>1:
                f.add_frame(f.now)
                f.event.peak_ts=f.now+.7
                f.event.audio_trigger=f.audio_trigger(f.event.peak_ts)
                f.now+=.8;f.add_frame(f.now)
                f.now+=.05;f.add_frame(f.now)
            f.observe();f.step(.51)
            for _ in range(3):f.step(.61,True)
            f.step(1.01,True)
        f.r.close()
        self.root=f.capture;self.output=f.base/'labels/exact_camera_ground_truth.json'
        self.manifest=crops.load_gt_review(self.root)

    def cli(self,events,extra=()):
        import pygame
        argv=['physical_trace_label','--gt-review','--capture',str(self.root),'--output',str(self.output)]
        argv.extend(extra)
        out=io.StringIO()
        with patch.dict(os.environ,SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy'), \
             patch('sys.argv',argv),patch.object(pygame.event,'get',side_effect=events), \
             patch.object(label,'overlay_candidates',side_effect=AssertionError('candidate overlay forbidden')), \
             patch.object(label,'load_calibration',side_effect=AssertionError('no present-day calibration')), \
             patch.object(pygame.display,'set_mode',wraps=pygame.display.set_mode) as opened, \
             patch.object(pygame.display,'flip',wraps=pygame.display.flip) as presented, \
             redirect_stdout(out):
            label.main()
        self.assertEqual(opened.call_count,1)
        self.assertGreater(presented.call_count,0)
        crops.assert_inputs(self.manifest)
        return out.getvalue()

    def test_manifest_discovers_all_three_saved_shot_directories(self):
        self.assertEqual([s['shot_id'] for s in self.manifest['shots']],[1,2,3])
        self.assertEqual(len(crops.label_shots(self.manifest)),3)
        for sid,shot in enumerate(self.manifest['shots'],1):
            self.assertEqual(shot['availability'],'AVAILABLE')
            self.assertEqual(len(shot['frames']),6)
            for filename in ('pre_0','live_post_0','live_post_1','white','black','mid_gray'):
                for suffix in ('json','npy'):
                    self.assertTrue((self.root/f'gt_review/shots/shot_{sid:08d}/{filename}.{suffix}').exists())

    def test_real_cli_opens_and_all_original_roles_are_navigable(self):
        import pygame
        def key(k):return [pygame.event.Event(pygame.KEYDOWN,key=k)]
        events=[]
        for _ in (1,2,3):
            events += [key(pygame.K_c),key(pygame.K_4),key(pygame.K_3)]
            events += [key(pygame.K_UP) for _ in range(5)]
            events += [key(pygame.K_RIGHT)]
        events += [key(pygame.K_q)]
        viewed=[]
        def original(m,e):
            viewed.append((int(e['frame_id'].split(':')[0]),e['role'],e['projector_state'],e['path']))
            return read_original(m,e)
        read_original=crops.source_image
        with patch.object(crops,'source_image',side_effect=original):out=self.cli(events)
        self.assertIn('UI initialized: SDL=dummy',out)
        self.assertIn('WARNING: SDL dummy driver has no visible window',out)
        for sid in (1,2,3):
            self.assertIn(f'Shot {sid}: AVAILABLE PRE=1 LIVE_POST=2 GT_SETTLED=3',out)
        for sid in (1,2,3):
            seen=[v for v in viewed if v[0]==sid]
            self.assertEqual({v[1] for v in seen},{'PRE','LIVE_POST','GT_SETTLED'})
            self.assertEqual({v[2] for v in seen if v[1]=='GT_SETTLED'},{'WHITE','BLACK','MID_GRAY'})
            self.assertTrue(any(v[3].endswith('live_post_0.npy') for v in seen))
            self.assertTrue(any(v[3].endswith('live_post_1.npy') for v in seen))
        self.assertFalse(self.output.exists())

    def test_offscreen_sdl_cannot_silently_appear_to_be_a_visible_window(self):
        import pygame
        with patch.object(pygame.display,'get_driver',return_value='offscreen'):
            out=self.cli([[pygame.event.Event(pygame.KEYDOWN,key=pygame.K_q)]])
        self.assertIn('WARNING: SDL offscreen driver has no visible window',out)
        self.assertFalse(self.output.exists())

    def test_click_draws_pending_marker_enter_saves_xy_unknown_and_reopen(self):
        import pygame
        def key(k):return [pygame.event.Event(pygame.KEYDOWN,key=k)]
        # Synthetic original full camera is 13x9; actual CLI viewport is1280x900.
        e=self.manifest['shots'][0]['frames'][-1]['crops'][0]
        scale,_,origin=crops.view_transform(e,(1280,790));origin=(origin[0],origin[1]+70)
        click_pos=tuple(round(origin[i]+[5.,4.][i]*scale[i]) for i in (0,1))
        expected=crops.crop_click(click_pos,origin,scale,e)['camera_xy']
        markers=[];draw_line=pygame.draw.line
        def draw(*args,**kwargs):
            if args[1]==(0,255,0):markers.append(args[2:4])
            return draw_line(*args,**kwargs)
        def enter_after_marker():
            self.assertGreaterEqual(len(markers),2)
            self.assertFalse(self.output.exists(),'click alone must not write GT')
            return key(pygame.K_RETURN)
        events=iter([key(pygame.K_4),key(pygame.K_3),
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=click_pos)],
            enter_after_marker,key(pygame.K_RIGHT),key(pygame.K_u),key(pygame.K_q)])
        def next_event():
            value=next(events)
            return value() if callable(value) else value
        with patch.object(pygame.draw,'line',side_effect=draw):self.cli(next_event)
        doc=crops.validate_document(json.loads(self.output.read_text()),self.manifest)
        self.assertEqual(doc['shots'][0]['camera_xy'],expected)
        self.assertEqual(doc['shots'][0]['clicked_view'],'ORIGINAL_GT_SETTLED')
        self.assertFalse(doc['shots'][0]['candidate_overlay'])
        self.assertEqual(doc['shots'][1]['physical_truth_status'],'UNKNOWN')
        self.assertEqual(doc['shots'][2]['physical_truth_status'],'UNKNOWN')
        before=self.output.read_bytes()
        self.cli([key(pygame.K_q)])
        self.assertEqual(self.output.read_bytes(),before)

    def test_empty_inventory_is_an_explicit_error_not_silent_ui_exit(self):
        manifest={**self.manifest,'shots':[]}
        argv=['physical_trace_label','--gt-review','--capture',str(self.root),'--output',str(self.output)]
        err=io.StringIO()
        with patch('sys.argv',argv),patch.object(crops,'load_gt_review',return_value=manifest), \
             patch.object(label,'_run_target_ui') as ui,redirect_stdout(io.StringIO()),redirect_stderr(err), \
             self.assertRaises(SystemExit) as raised:
            label.main()
        self.assertEqual(raised.exception.code,2)
        self.assertIn('manifest contains no shots',err.getvalue())
        ui.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_explicit_independent_observation_cli_preserves_noncausal_label_basis(self):
        import pygame
        entry=self.manifest['shots'][0]['frames'][-1]['crops'][0]
        scale,_,origin=crops.view_transform(entry,(1280,790));origin=(origin[0],origin[1]+70)
        pos=tuple(round(origin[i]+[5.,4.][i]*scale[i]) for i in (0,1))
        key=lambda k:[pygame.event.Event(pygame.KEYDOWN,key=k)]
        self.cli([[pygame.event.Event(pygame.MOUSEBUTTONDOWN,button=1,pos=pos)],
                  key(pygame.K_RETURN),key(pygame.K_q)],
                 ['--independent-hole-observation','Synthetic observation for UI test'])
        doc=crops.validate_document(json.loads(self.output.read_text()),self.manifest)
        row=doc['shots'][0]
        self.assertEqual(row['pre_inspected'],[])
        self.assertEqual(row['review_truth_basis']['method'],'INDEPENDENT_PHYSICAL_OBSERVATION')
        self.assertFalse(row['review_truth_basis']['causal_pre_certified'])


if __name__=='__main__':unittest.main()
