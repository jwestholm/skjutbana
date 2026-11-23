"""Visible parts, persistent room damage and causal presentation; no physical accuracy claim."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ['SKJUTBANA_PHYSICAL_TRACE_DISABLE']='1'
import hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch,Mock
import cv2
import numpy as np
from PIL import Image
import pygame
from content.games.light_targets_game import LightTargetsGame,OBJECTS_PER_ROOM,BREAK_FPS,RESULT_DWELL_S
from automation.light_room_palette import camera_level
from content.games.light_room_art import ASSET_ROOT
from content.games.light_room_audio import cue_wave
from src.engine.game_objects.geometry import shape_contains
from src.engine.input.hit_regions import object_hit_registry_v2223


def part_point(game,kind,part):
    obj=next(o for o in game.manager.objects if o.metadata['sprite_kind']==kind and o.part_id==part)
    spec=game.art.objects[kind]['parts'][part]
    mask=np.asarray(Image.open(ASSET_ROOT/spec['mask'])).copy()
    _,_,_,point=cv2.minMaxLoc(cv2.distanceTransform(mask,cv2.DIST_L2,5))
    h,w=mask.shape;g=obj.geometry
    return obj,(g.x+point[0]/w*g.width,g.y+point[1]/h*g.height)


class RoomTests(unittest.TestCase):
    def setUp(self):
        pygame.init();self.cancel=patch.object(object_hit_registry_v2223,'cancel_object_queries');self.cancel.start();self.addCleanup(self.cancel.stop)
        self.g=LightTargetsGame('',pygame.Rect(20,10,760,580));self.g.on_enter();self.addCleanup(self.g.on_exit)
        self.g.rng.seed(11);self.screen=pygame.Surface((800,620))
    def live(self):
        self.g.start();self.g.render(self.screen);self.g.on_frame_presented(99.)
        self.g.presented_mono-=2;self.g.update(0);self.g.ready_wall_ts=0.;self.g.render(self.screen)
        self.assertEqual(self.g.phase,'live')
    def finish_animation(self):
        self.g.reaction_at-=2;self.g.render(self.screen);self.g._reaction_start_mono-=2;self.g.update(0)
    def test_assets_partition_palette_and_hashes(self):
        m=self.g.art.manifest
        for name,digest in m['files_sha256'].items():self.assertEqual(hashlib.sha256((ASSET_ROOT/name).read_bytes()).hexdigest(),digest)
        self.assertTrue({'vase','jar','bowl','bottle','lamp'}<=set(m['objects']))
        for name,spec in m['objects'].items():
            total=None
            for part,p in spec['parts'].items():
                img=np.asarray(Image.open(ASSET_ROOT/p['sprite']).convert('RGBA'));mask=np.asarray(Image.open(ASSET_ROOT/p['mask']))>0
                self.assertTrue(np.array_equal(mask,img[:,:,3]>0));self.assertGreaterEqual(float(camera_level(img[:,:,:3][mask]).min()),124)
                hit=np.asarray(Image.open(ASSET_ROOT/p['hit_sprite']).convert('RGBA'))
                self.assertTrue(np.array_equal(hit[:,:,3],img[:,:,3]))
                self.assertGreaterEqual(float(camera_level(hit[:,:,:3][mask]).min()),124)
                h,w=mask.shape;polygon=np.zeros((h,w),np.uint8)
                for poly in p['polygons']:cv2.fillPoly(polygon,[np.rint(np.array(poly)*(w,h)).astype(np.int32)],1)
                self.assertTrue(np.array_equal(polygon>0,mask),(name,part))
                total=mask.astype(int) if total is None else total+mask
                self.assertEqual(len(p['frames']),12)
            self.assertLessEqual(total.max(),1,'visible-part masks must not invent overlapping depth')
        room=np.asarray(Image.open(ASSET_ROOT/'room.png'))
        if m.get('rooms'):
            # R7 intentionally restores natural contrast outside active targets.
            self.assertLess(float(camera_level(room[:,:,:3]).min()),120)
        else:self.assertGreaterEqual(float(camera_level(room[:,:,:3]).min()),124)
    def test_layouts_use_real_supports_without_overlap_or_hud_collision(self):
        for size in ((640,480),(760,580),(1280,720)):
            game=LightTargetsGame('',pygame.Rect(0,0,*size));game.on_enter()
            try:
                game.rng.seed(11);anchors=set();supports=set()
                for _ in range(4):
                    game._next_target();self.assertEqual(len(game.items),OBJECTS_PER_ROOM,size)
                    for i,a in enumerate(game.items):
                        ga=a['geometry'];anchors.add(a['anchor']['id']);supports.add(a['anchor']['support']);self.assertGreaterEqual(ga.y,104)
                        for b in game.items[i+1:]:self.assertFalse(pygame.Rect(ga.aabb).colliderect(pygame.Rect(b['geometry'].aabb)),size)
                self.assertGreaterEqual(len(anchors),8);self.assertTrue({'shelf','sill','table','floor'}<=supports)
            finally:game.on_exit()
    def test_pre_wait_and_stale_audio(self):
        self.g.start();self.g.render(self.screen);self.g.on_frame_presented(100.)
        with patch('content.games.light_targets_game.time.monotonic',return_value=self.g.presented_mono+1.69):self.g.update(0)
        self.assertEqual(self.g.phase,'settling');self.assertEqual(self.g.get_hit_regions(),())
        with patch('content.games.light_targets_game.time.monotonic',return_value=self.g.presented_mono+1.71),patch('content.games.light_targets_game.time.time',return_value=102.):self.g.update(0)
        self.g.on_object_shot_frozen(NS(shot_id=1,peak_ts=101.));self.assertIsNone(self.g.active_shot)
        self.g.on_object_shot_frozen(NS(shot_id=2,peak_ts=102.1));self.assertEqual(self.g.active_shot,2)
    def test_vase_parts_have_distinct_consequences_and_points(self):
        for part,want,points in [('flower',{'flower'},200),('neck',{'neck','flower'},150),('body',{'body','neck','flower'},100)]:
            self.live();obj,xy=part_point(self.g,'vase',part);entity=obj.entity_id
            self.g._on_hit(NS(is_simulated=True,game_x=xy[0],game_y=xy[1],shot_id=None))
            self.assertEqual({o.part_id for o in self.g.manager.objects if o.entity_id==entity and not o.active},want)
            self.assertEqual(self.g.scores[0],points)
            self.assertTrue(all(o.active for o in self.g.manager.objects if o.entity_id!=entity))
    def test_real_manager_reaction_and_sound_wait_for_post_and_flip(self):
        self.live();regions=self.g.get_hit_regions();obj,xy=part_point(self.g,'vase','body')
        snap=NS(shot_id=42,peak_ts=100.,game_regions=regions,object_gameplay={});self.g.manager._snapshot_provider=lambda sid:snap
        before=pygame.surfarray.array3d(self.screen).copy();self.g.on_object_shot_frozen(snap)
        result=dict(final=True,shot_id=42,mode='OBJECT',status='PHYSICAL_HIT',result_semantics='OBJECT_HIT',object_decision=dict(state='OBJECT_HIT',object_id=obj.object_id,hypothesis=dict(confirmed=True,game_xy=xy,camera_xy=xy)))
        with patch('content.games.light_targets_game.hit_context_snapshot_for_shot',return_value=snap),patch('content.games.light_targets_game.time.time',return_value=100.5),patch('content.games.light_targets_game.time.monotonic',return_value=20.),patch('content.games.light_targets_game.audio_peak_detector.play_feedback') as sound:
            self.g._on_object_event(dict(event='physical.object_result',data=result));self.g.render(self.screen);self.g.on_frame_presented(100.51)
            self.assertTrue(np.array_equal(before,pygame.surfarray.array3d(self.screen)));sound.assert_not_called()
            self.assertNotIn('reaction_presented_ts',snap.object_gameplay)
        with patch('content.games.light_targets_game.time.monotonic',return_value=20.71),patch('content.games.light_targets_game.hit_context_snapshot_for_shot',return_value=snap),patch('content.games.light_targets_game.audio_peak_detector.play_feedback',return_value={'status':'TEST'}) as sound:
            self.g.render(self.screen);self.g.on_frame_presented(101.21);self.g.on_frame_presented(101.3)
            sound.assert_called_once();self.assertEqual(snap.object_gameplay['reaction_presented_ts'],101.21)
        self.assertEqual(self.g.scores[0],100);self.assertEqual(len(self.g._effect_ids),3)
        old_layout=self.g.layout_id;remaining={o.object_id for o in self.g.manager.objects if o.active}
        self.finish_animation();self.assertEqual(self.g.phase,'settling');self.assertEqual(self.g.layout_id,old_layout)
        self.assertEqual({o.object_id for o in self.g.manager.objects if o.active},remaining)
    def test_unknown_never_breaks_but_has_distinct_feedback(self):
        self.live();before=pygame.surfarray.array3d(self.screen).copy()
        for _ in range(3):self.g.update(.1);self.g.render(self.screen)
        self.assertTrue(np.array_equal(before,pygame.surfarray.array3d(self.screen)))
        self.g._finish('UNKNOWN',None,dict(reason='test'),None,simulated=True);self.g.render(self.screen)
        with patch('content.games.light_targets_game.audio_peak_detector.play_feedback',return_value={'status':'TEST'}) as sound:
            self.g.on_frame_presented(10.);sound.assert_called_once()
            self.assertEqual(sound.call_args.kwargs['cue'],'unknown')
        self.assertFalse(self.g.broken_ids);self.assertTrue(all(o.visible for o in self.g.manager.objects))
    def test_audio_is_quiet_distinct_and_can_be_muted(self):
        waves=[cue_wave(c,16000) for c in ('shatter','topple','petal','miss','unknown')]
        self.assertTrue(all(w.dtype==np.int16 and np.abs(w.astype(int)).max()<=.161*32767 for w in waves))
        self.assertEqual(len({len(w) for w in waves}),5)
        self.g.handle_event(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_m));self.assertFalse(self.g.sound_enabled)
    def test_part_ambiguity_never_uses_depth_in_preview(self):
        self.live();body=next(o for o in self.g.manager.objects if o.metadata['sprite_kind']=='vase' and o.part_id=='body')
        neck=next(o for o in self.g.manager.objects if o.entity_id==body.entity_id and o.part_id=='neck')
        b,n=body.shape_snapshot(),neck.shape_snapshot();points=b['points'] if b['kind']=='polygon' else b['parts'][0]['points']
        point=next((p for p in points if shape_contains(n,*p)),None);self.assertIsNotNone(point)
        self.g._on_hit(NS(is_simulated=True,game_x=point[0],game_y=point[1],shot_id=None))
        self.assertEqual(self.g.scores[0],0);self.assertTrue(all(o.active for o in self.g.manager.objects))
    def test_dense_rendered_room_targets_and_negative_controls(self):
        from automation.object_physical_query_selftest import frame,snapshot
        from src.engine.camera.object_physical_query import ObjectPhysicalQuery
        from src.engine.input.object_hit_v2223 import CameraHitRegionV240
        self.live();self.g.presented_ts=1.;game=self.g.get_hit_regions()
        cam=tuple(CameraHitRegionV240(r.object_id,r.x,r.y,r.width,r.height) for r in game);snap=snapshot(cam,game)
        pixels=pygame.surfarray.array3d(self.screen)[20:780,10:590];pre=np.ascontiguousarray(pixels.transpose(1,0,2)[:,:,::-1])
        cases=[]
        for kind,part in [('vase','body'),('vase','neck'),('vase','flower'),('jar','body'),('bottle','body'),('lamp','shade')]:
            obj,xy=part_point(self.g,kind,part);cases.append((xy,obj.object_id))
        cases.extend([(None,None),((740,550),None)])
        with patch.dict(os.environ,{'SKJUTBANA_OBJECT_DENSE_MODEL':''}):
            for xy,expected in cases:
                post=pre.copy()
                if xy is not None:cv2.circle(post,tuple(round(v) for v in xy),3,(30,30,30),-1)
                q=ObjectPhysicalQuery(snap,[frame(pre,t) for t in (1.8,1.9,1.94)],camera_to_game=lambda x,y:(x,y))
                for t in (2.36,2.40,2.44):result=q.observe(frame(post,t),delivered_ts=t+.001)
                decision=result['object_decision']
                if expected:self.assertEqual(decision.get('object_id'),expected)
                else:self.assertNotEqual(decision['state'],'OBJECT_HIT')
    def test_clear_room_bonus_and_refill(self):
        self.live();self.g.shots_per_player=40;layout=self.g.layout_id
        # Books and TV have independent parts. Clear actual active shapes instead
        # of assuming that every entity has a body/base that destroys it in one hit.
        for _ in range(40):
            active=[o for o in self.g.manager.objects if o.active]
            obj=min(active,key=lambda o:(o.part_id not in ('body','base','stand'),o.object_id))
            kind=obj.metadata['sprite_kind'];spec=self.g.art.objects[kind]['parts'][obj.part_id]
            mask=np.asarray(Image.open(ASSET_ROOT/spec['mask']));_,_,_,point=cv2.minMaxLoc(cv2.distanceTransform(mask,cv2.DIST_L2,5))
            h,w=mask.shape;g=obj.geometry;xy=(g.x+point[0]/w*g.width,g.y+point[1]/h*g.height)
            self.g.phase='live';self.g.ready_wall_ts=0.
            self.g._on_hit(NS(is_simulated=True,game_x=xy[0],game_y=xy[1],shot_id=None))
            self.assertEqual(self.g.phase,'result')
            clear=self.g._clear_bonus==500
            if clear:self.assertIn('RUMMET RENSAT',self.g.feedback)
            self.finish_animation()
            if clear:break
        else:self.fail('room did not clear after all parts were exercised')
        self.assertEqual(self.g.layout_id,layout+1)
        self.assertTrue(all(o.active for o in self.g.manager.objects))

    def test_impact_side_changes_only_after_acceptance(self):
        sides=[]
        for direction in (-1,1):
            self.live();obj,xy=part_point(self.g,'vase','body')  # vase is guaranteed; jar now varies by furnishing
            xy=(xy[0]+direction*obj.geometry.width*.15,xy[1])
            self.g._on_hit(NS(is_simulated=True,game_x=xy[0],game_y=xy[1],shot_id=None))
            self.assertEqual(self.g.phase,'result');self.assertTrue(self.g._effect_ids)
            sides.append(self.g.effect_sides[obj.object_id])
        self.assertEqual(sides,[-1,1])

    def test_resting_debris_never_occludes_a_hittable_part(self):
        self.live();item=max(self.g.items,key=lambda i:i['geometry'].bottom)
        part=next(iter(item['parts']));broken=item['parts'][part];g=item['geometry']
        target=next(i for i in self.g.items if i is not item)
        obj=next(o for o in self.g.manager.objects if o.entity_id==target['entity_id'])
        spec=self.g.art.objects[target['kind']]['parts'][obj.part_id]
        mask=np.asarray(Image.open(ASSET_ROOT/spec['mask']));_,_,_,point=cv2.minMaxLoc(cv2.distanceTransform(mask,cv2.DIST_L2,5))
        h,w=mask.shape;tg=obj.geometry;xy=(round(tg.x+point[0]/w*tg.width),round(tg.y+point[1]/h*tg.height))
        before=pygame.Surface((760,580));after=pygame.Surface((760,580));self.g.art.render(before,self.g.items)
        fragment=pygame.Surface((760,580),pygame.SRCALPHA);fragment.fill((255,200,200,255))
        with patch.object(self.g.art,'frames',return_value=([fragment],(-round(g.x),-round(g.y)))):
            self.g.art.render(after,self.g.items,{broken})
        self.assertEqual(before.get_at(xy),after.get_at(xy))

    def test_part_identity_scoring(self):
        from automation.light_targets_score import score
        with tempfile.TemporaryDirectory() as td:
            log=Path(td)/'run.log';truth=Path(td)/'truth'
            log.write_text('[LAMP-RESULT] '+json.dumps(dict(source='OBJECT',owner='a',shot_id=1,outcome='HIT',object_id='vase_1:neck',game_version='ceramic-room-parts-2'))+' logger tail')
            truth.write_text(json.dumps(dict(shot_id=1,expected='HIT',truth_object='vase_1:flower')))
            out=score(log,truth);self.assertEqual(out['wrong_objects'],1);self.assertEqual(out['correct'],0)
            truth.write_text(json.dumps(dict(shot_id=1,expected='HIT')))
            with self.assertRaises(ValueError):score(log,truth)


class ReviewTests(unittest.TestCase):
    def test_review_checks_exact_pre_and_keeps_truth_out_of_decision(self):
        from unittest.mock import Mock
        from automation.light_targets_review import review
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);session=root/'session';session.mkdir()
            (session/'session.json').write_text(json.dumps(dict(settings={'content/settings.json':dict(content=dict(
                viewport=[0,0,760,580],camera_calibration=dict(homography=np.eye(3).tolist())))})))
            log=root/'run.log';log.write_text('fixture')
            truth=root/'truth';truth.write_text(json.dumps(dict(kind='shot',shot_id=1,truth_object='vase')))
            frames=NS(ts=np.array([1.8]),cache={},fh=NS(close=Mock()))
            shot=dict(peak_ts=2.,pre_frame_ts=[1.8],object_id='vase')
            with patch('automation.light_targets_review.Frames',return_value=frames),patch('automation.light_targets_review.load_shots',return_value={1:shot}),patch('automation.light_targets_review.decide',return_value=('vase',[],'hit')) as decide:
                out=review(session,log,truth);self.assertEqual(out['correct'],1)
                self.assertEqual(out['negative_controls'],0)
                # Changing GT changes only evaluation; the detector receives identical inputs.
                truth.write_text(json.dumps(dict(kind='shot',shot_id=1,truth_object=None)))
                out=review(session,log,truth);self.assertEqual(out['false_awards'],1)
                self.assertEqual(decide.call_args_list[0].args[:2],decide.call_args_list[1].args[:2])
                frames.ts=np.array([1.7])
                with self.assertRaises(ValueError):review(session,log,truth)

if __name__=='__main__':unittest.main()
