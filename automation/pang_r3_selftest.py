"""R3 palette, display state and saved-run audit contracts. No physical validation."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ['SKJUTBANA_PHYSICAL_TRACE_DISABLE']='1'
import contextlib,hashlib,io,json,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image
import pygame
from automation.light_room_palette import camera_level,ensure_floor,colorize,EXPORT_FLOOR
from automation.recolor_light_room import recolor
from automation.pang_session_audit import spot_change,review_spots
from automation.light_room_selftest import part_point
from content.games.light_targets_game import LightTargetsGame,BREAK_FPS,RESULT_DWELL_S
from src.engine.input.hit_regions import object_hit_registry_v2223


class PaletteTests(unittest.TestCase):
    def test_supplied_camera_proxy_and_floor(self):
        self.assertAlmostEqual(float(camera_level((160,160,160))),96.5395,places=2)
        self.assertLess(float(camera_level((170,170,170))),120)
        self.assertGreater(float(camera_level((188,188,188))),120)
        grid=np.array(np.meshgrid(*[np.linspace(0,255,8)]*3)).reshape(3,-1).T
        copy=grid.copy();protected=ensure_floor(grid)
        self.assertGreaterEqual(float(camera_level(protected).min()),EXPORT_FLOOR)
        self.assertTrue(np.array_equal(grid,copy))
    def test_color_state_preserves_alpha_and_is_distinct(self):
        rgba=np.full((30,30,4),230,np.uint8);rgba[:,:,3]=0;rgba[4:26,4:26,3]=255
        image=Image.fromarray(rgba)
        for kind in ('vase','jar','bowl','lamp','bottle'):
            normal=np.asarray(colorize(image,kind));hit=np.asarray(colorize(image,kind,hit=True));mask=rgba[:,:,3]>0
            self.assertTrue(np.array_equal(normal[:,:,3],rgba[:,:,3]))
            self.assertTrue(np.array_equal(hit[:,:,3],rgba[:,:,3]))
            self.assertGreaterEqual(float(camera_level(normal[:,:,:3][mask]).min()),EXPORT_FLOOR)
            self.assertGreater(float(np.abs(normal[:,:,:3].astype(float)-hit[:,:,:3]).mean()),30)
    def test_recolor_is_versioned_and_keeps_geometry_and_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'base';src.mkdir();out=root/'new'
            a=np.full((12,12,4),220,np.uint8);a[:,:,3]=0;a[2:10,2:10,3]=255
            Image.fromarray(a).save(src/'body.png');Image.fromarray(a).save(src/'frame.png')
            Image.fromarray(a[:,:,3]).save(src/'mask.png');Image.fromarray(a[:,:,:3]).save(src/'room.png')
            shape=[[[.2,.2],[.8,.2],[.8,.8],[.2,.8]]]
            manifest=dict(schema='light-room-parts-2',objects={'vase':dict(parts={'body':dict(sprite='body.png',mask='mask.png',frames=['frame.png'],polygons=shape)})},files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('*.png')})
            (src/'manifest.json').write_text(json.dumps(manifest));before=(src/'body.png').read_bytes()
            recolor(src,out);result=json.loads((out/'manifest.json').read_text())
            self.assertEqual(result['objects']['vase']['parts']['body']['polygons'],shape)
            self.assertEqual((out/'mask.png').read_bytes(),(src/'mask.png').read_bytes())
            self.assertEqual((src/'body.png').read_bytes(),before)
            with self.assertRaises(FileExistsError):recolor(src,out)
            with self.assertRaises(ValueError):recolor(out,root/'invalid_second_pass')


class DisplayTests(unittest.TestCase):
    def setUp(self):
        pygame.init();self.stream=io.StringIO();self.redirect=contextlib.redirect_stdout(self.stream);self.redirect.__enter__();self.addCleanup(self.redirect.__exit__,None,None,None)
        p=patch.object(object_hit_registry_v2223,'cancel_object_queries');p.start();self.addCleanup(p.stop)
        self.g=LightTargetsGame('',pygame.Rect(0,0,760,580));self.g.on_enter();self.addCleanup(self.g.on_exit)
        self.g.rng.seed(11);self.screen=pygame.Surface((760,580))
    def ready(self):
        self.g.start();self.g.render(self.screen);self.g.on_frame_presented(1.);self.g.presented_mono-=2;self.g.update(0);self.g.ready_wall_ts=0.;self.g.render(self.screen)
    def test_more_objects_varied_sizes_and_composited_proxy_floor(self):
        self.ready();self.assertEqual(len(self.g.items),8)
        heights=[i['geometry'].height for i in self.g.items];self.assertGreater(max(heights)-min(heights),30)
        image=pygame.surfarray.array3d(self.screen).transpose(1,0,2)
        self.assertGreaterEqual(float(camera_level(image[104:]).min()),120)
        self.assertEqual(set(self.g.entity_states().values()),{'intact'})
    def test_partial_and_broken_states_are_explicit(self):
        self.ready();obj,xy=part_point(self.g,'vase','flower');entity=obj.entity_id
        self.g._on_hit(NS(is_simulated=True,game_x=xy[0],game_y=xy[1],shot_id=None))
        self.assertEqual(self.g.entity_states()[entity],'hit');self.assertIn('TRÄFF',self.g.feedback)
        self.g.render(self.screen);self.g._reaction_start_mono-=2;self.g.update(0)
        self.g.phase='live';obj,xy=part_point(self.g,'vase','body')
        self.g._on_hit(NS(is_simulated=True,game_x=xy[0],game_y=xy[1],shot_id=None))
        self.assertEqual(self.g.entity_states()[entity],'broken')
        self.g.render(self.screen);rgb=pygame.surfarray.array3d(self.screen).transpose(1,0,2)
        self.assertGreaterEqual(float(camera_level(rgb[104:]).min()),120)
    def test_animation_is_fast_without_shortening_protected_post(self):
        self.assertEqual(12/BREAK_FPS,.6);self.assertLess(RESULT_DWELL_S,1.)
        self.ready();self.g.active_peak=100.
        with patch('content.games.light_targets_game.time.time',return_value=100.5),patch('content.games.light_targets_game.time.monotonic',return_value=20.):
            self.g._finish('UNKNOWN',1,dict(reason='test'),None)
        self.assertGreaterEqual(self.g.reaction_at,20.7)
        self.assertEqual(self.g._cue,'unknown')
        self.g._finish('MISS',2,None,None,simulated=True);self.assertEqual(self.g._cue,'miss');self.assertIn('MISS',self.g.feedback)


class AuditTests(unittest.TestCase):
    def test_review_only_white_spot_and_causal_change(self):
        white=np.full((120,180,3),200,np.uint8);cv2.circle(white,(70,60),3,(20,20,20),-1)
        yy,xx=np.mgrid[0:120,0:180];grid=np.stack((xx,yy),axis=2).astype(float)
        spots=review_spots(white,grid,lambda x,y:(x,y));self.assertEqual(len(spots),1)
        self.assertLess(np.linalg.norm(np.array(spots[0]['camera_xy'])-(70,60)),1)
        pre=np.full((120,180),200.,np.float32);post=pre.copy();post[59:62,69:72]=100
        self.assertGreater(spot_change(pre,post,(70,60))['signed_spot_dn'],50)
        self.assertEqual(spot_change(pre,pre,(70,60))['signed_spot_dn'],0)

if __name__=='__main__':unittest.main()
