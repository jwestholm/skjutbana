"""Deterministic OBJECT lifecycle/audio/property checks; NOT physical validation."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')

from dataclasses import replace
from types import SimpleNamespace as S
import unittest
import json
import io
import contextlib
from unittest.mock import Mock, patch
import numpy as np
import pygame

from automation import object_physical_query_selftest as fixture
from content.games import can_shootout_game as can
from src.engine.audio.audio_peak_detector import AudioPeakDetector, AudioPeakEvent
from src.engine.game_objects import ObjectManager, ObjectGeometry, make_breakable_object
from src.engine.input.object_hit_v2223 import ObjectHitRegistryV2223
from src.engine.input.object_hit_v2223 import CameraHitRegionV240, GameHitRegionV240
from src.engine.input.hit_regions import object_query_regions
from src.engine.camera.object_physical_query import ObjectPhysicalQuery
from src.engine.shot_critical_v2223 import ShotCriticalControllerV2223


class Reproductions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init();pygame.display.set_mode((800,600))

    def test_pending_audio_snapshot_binds_oldest_not_latest_peak(self):
        detector=AudioPeakDetector();detector.last_peak_ts=2.1
        detector._pending_dispatch.extend([AudioPeakEvent(2.,.8,.1),AudioPeakEvent(2.1,.8,.1)])
        controller=ShotCriticalControllerV2223()
        with patch.object(controller,'_set_wait_cursor'):
            self.assertEqual(controller.begin_pending_audio(S(),detector),2.)

    def test_destroyed_target_cannot_award_again_from_second_frozen_shot(self):
        manager=ObjectManager();obj=manager.add(make_breakable_object('can',ObjectGeometry(100,60,60,80)))
        snaps={sid:fixture.snapshot(game=(obj.get_hit_region(),),sid=sid) for sid in (1,2)}
        manager._snapshot_provider=snaps.get
        def result(sid):return dict(mode='OBJECT',status='PHYSICAL_HIT',shot_id=sid,
            hypotheses=[dict(game_xy=[130,100],confirmed=True,rank=1)])
        self.assertIsNotNone(manager.resolve_object_query(result(1),can.GAME_PROJECTILE)[0])
        self.assertFalse(obj.hittable)
        self.assertEqual(manager.resolve_object_query(result(2),can.GAME_PROJECTILE),(None,None))

    def test_restart_clears_old_comparison_frame(self):
        game=can.ShootStuffGame('',pygame.Rect(0,0,800,600));game.on_enter();self.addCleanup(game.on_exit)
        game._comparison_shot=17;game._comparison_frame=pygame.Surface((800,600))
        game.handle_event(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_r))
        self.assertIsNone(game._comparison_shot)
        self.assertIsNone(game._comparison_frame)


class Sequences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init();pygame.display.set_mode((800,600))

    def setUp(self):
        self.now=10.
        self.registry=ObjectHitRegistryV2223();self.audio=AudioPeakDetector()
        for target,attribute,value in [(can.time,'time',lambda:self.now),
            (can.time,'monotonic',lambda:self.now),
            (can,'object_hit_registry_v2223',self.registry),
            (can,'hit_context_snapshot_for_shot',self.registry.snapshot_for_shot),
            (can,'audio_peak_detector',self.audio)]:
            p=patch.object(target,attribute,value);p.start();self.addCleanup(p.stop)
        self.game=can.ShootStuffGame('',pygame.Rect(0,0,800,600));self.game.on_enter()
        self.addCleanup(self.game.on_exit)
        self.game.manager._snapshot_provider=self.registry.snapshot_for_shot
        self.screen=pygame.Surface((800,600));self.surface=np.full((600,800),140,np.uint8)
        self.sounds={s.target_id:Mock() for s in can.TARGET_SPECS}
        for sound in self.sounds.values():sound.get_length.return_value=.22
        self.game._hit_sounds=self.sounds.copy();self.sid=0
        self.live()

    def live(self):
        self.now+=2.;self.game.phase_until=self.now-.01;self.game.update(.02)
        self.assertEqual(self.game.phase,'live')
        self.game.render(self.screen);self.game.on_frame_presented(self.now)
        self.now+=.15

    def target(self,name):
        return next(o for o in self.game.manager.objects
            if can.TARGET_SPECS[o.metadata['spec_index']].target_id==name)

    def freeze(self):
        self.sid+=1
        self.game.manager._snapshot_provider=self.registry.snapshot_for_shot
        regions=self.game.get_hit_regions()
        snap=replace(fixture.snapshot(game=regions,sid=self.sid),peak_ts=self.now,
            camera_regions=tuple(CameraHitRegionV240(r.object_id,r.x,r.y,r.width,r.height) for r in regions),
            object_camera_to_game=lambda x,y:(x,y))
        self.registry._snapshots[self.sid]=snap
        if any('physical_query' in r.metadata for r in regions):self.game.on_object_shot_frozen(snap)
        return snap

    def shoot(self,name=None,xy=None,uncertain=False,level=30):
        snap=self.freeze();peak=self.now
        pre=[fixture.frame(self.surface.copy(),peak-d) for d in (.10,.065,.04)]
        post=self.surface.copy()
        if name is not None:xy=self.target(name).geometry.center
        if xy is not None:
            x,y=map(int,xy);post[y-2:y+3,x-2:x+3]=level
        frames=pre+([] if uncertain else [fixture.frame(post,peak+.033),fixture.frame(post,peak+.066)])
        camera=S(get_ring_snapshot=lambda:frames)
        self.registry.begin_object_query(snap.shot_id,S(known_holes=[]),camera)
        self.now=peak+(.36 if uncertain else .075)
        self.registry.advance_object_queries(S(),camera,self.now)
        if not snap.object_query.final:
            self.now=peak+.36;self.registry.advance_object_queries(S(),camera,self.now)
        self.game.render(self.screen);self.game.on_frame_presented(self.now)
        self.surface=post
        return snap

    def test_two_distinct_can_shots(self):
        a=self.shoot('soda_can');self.live();b=self.shoot('gold_can')
        self.assertEqual((self.game.shots_used[1],self.game.hits[1]),(2,2))
        self.assertNotEqual(a.object_gameplay['exact_collision'],b.object_gameplay['exact_collision'])
        self.sounds['soda_can'].play.assert_called_once();self.sounds['gold_can'].play.assert_called_once()

    def test_hit_then_unchanged_surface(self):
        self.shoot('soda_can');self.live();b=self.shoot()
        self.assertEqual(b.object_query.result['status'],'NO_PHYSICAL_HIT')
        self.assertEqual((self.game.shots_used[1],self.game.hits[1]),(2,1))

    def test_hit_then_adjacent_new_change_is_not_vetoed(self):
        x,y=self.target('soda_can').geometry.center
        self.shoot('soda_can');self.game._start_match();self.live();b=self.shoot(xy=(x+7,y))
        self.assertEqual(b.object_query.result['status'],'PHYSICAL_HIT')

    def test_removed_target_location_is_not_invented_as_active_search(self):
        x,y=self.target('soda_can').geometry.center
        self.shoot('soda_can');self.live();b=self.shoot(xy=(x+7,y))
        self.assertEqual(b.object_query.result['status'],'NO_PHYSICAL_HIT')
        self.assertNotIn('soda_can',[r.object_id for r in b.object_query.regions])

    def test_animation_then_next_pre_uses_presented_live_revision(self):
        self.shoot('soda_can');self.assertTrue(self.game.knocked_targets)
        self.live();self.assertEqual(self.game.knocked_targets,[]);self.assertEqual(self.game.markers,[])
        boundary=self.game._live_presented_ts;b=self.shoot('gold_can')
        self.assertTrue(all(t>boundary for t in b.object_query.pre_times))
        self.assertEqual(self.game.hits[1],2)

    def test_stale_transition_pre_is_never_used(self):
        snap=self.freeze();pre=np.full((600,800),140,np.uint8)
        snap.game_regions=tuple(replace(r,metadata={**r.metadata,'physical_query':{
            **r.metadata['physical_query'],'surface_after_ts':self.now-.02}}) for r in snap.game_regions)
        camera=S(get_ring_snapshot=lambda:[fixture.frame(pre,self.now-.04)])
        self.registry.begin_object_query(snap.shot_id,S(),camera)
        self.assertEqual(snap.object_query.result['status'],'UNCERTAIN')
        self.assertEqual(snap.object_query.pre_times,[])

    def test_no_presented_live_frame_is_uncertain(self):
        self.game._live_presented_ts=None;b=self.shoot('soda_can')
        self.assertEqual(b.object_query.result['status'],'UNCERTAIN');self.assertEqual(self.game.hits[1],0)

    def test_object_then_current_and_duplicate_score_once(self):
        snap=self.shoot('rubber_duck');self.registry.publish_object_comparisons(S(audio_events=[]))
        self.game._on_object_event(dict(event='physical.object_result',data=snap.object_query.result))
        self.game._on_hit(S(is_simulated=False,shot_id=snap.shot_id,camera_x=1,camera_y=2,
            game_x=1,game_y=2,timestamp=self.now,source='camera'))
        self.assertEqual((self.game.shots_used[1],self.game.hits[1]),(1,1))
        self.assertTrue(snap.object_current['projection_changed_before_current_finished'])
        self.assertEqual(snap.object_gameplay['duplicate_object_callbacks'],1)
        self.sounds['rubber_duck'].play.assert_called_once()

    def test_miss_near_exact_shape_consumes_without_sound(self):
        obj=self.target('soda_can');b=self.shoot(xy=(obj.geometry.x-8,obj.geometry.y-8))
        self.assertEqual(b.object_query.result['status'],'PHYSICAL_HIT')
        self.assertEqual((self.game.shots_used[1],self.game.hits[1]),(1,0))
        self.assertTrue(all(s.play.call_count==0 for s in self.sounds.values()))

    def test_uncertain_consumes_without_sound(self):
        b=self.shoot(uncertain=True)
        self.assertEqual(b.object_query.result['status'],'UNCERTAIN')
        self.assertEqual((self.game.shots_used[1],self.game.hits[1]),(1,0))
        self.assertTrue(all(s.play.call_count==0 for s in self.sounds.values()))

    def test_restart_retires_pending_query_and_late_result(self):
        old=self.freeze();pre=fixture.frame(self.surface,self.now-.04)
        self.registry.begin_object_query(old.shot_id,S(),S(get_ring_snapshot=lambda:[pre]))
        self.game.handle_event(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_r))
        self.assertEqual(old.object_query.result['reason'],'game_round_or_distance_changed')
        self.assertEqual(old.object_query.tiles,[]);self.assertEqual(self.game._pending_object_shots,{})
        self.live();self.game._on_object_event(dict(event='physical.object_result',data=old.object_query.result))
        self.assertEqual(self.game.shots_used[1],0)

    def test_distance_change_freezes_new_size(self):
        old=self.freeze();old_width=old.game_regions[0].width
        self.game.handle_event(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_KP_PLUS))
        self.live();new=self.freeze()
        self.assertEqual(self.game.distance_m,25);self.assertLess(new.game_regions[0].width,old_width)
        self.assertEqual(old.game_regions[0].width,old_width)

    def test_late_previous_player_cannot_score_current_player(self):
        old=self.freeze();self.game._start_player(2);self.live()
        self.game._on_object_event(dict(event='physical.object_result',data=dict(shot_id=old.shot_id,
            mode='OBJECT',status='PHYSICAL_HIT',hypotheses=[],peak_ts=old.peak_ts)))
        self.assertEqual(self.game.shots_used,{1:0,2:0});self.assertEqual(self.game.player,2)

    def test_two_supported_objects_abstain_in_opt_in_game(self):
        snap=self.freeze();points=[self.target(n).geometry.center for n in ('soda_can','gold_can')]
        q=ObjectPhysicalQuery(snap,[fixture.frame(self.surface,self.now-.04)],camera_to_game=lambda x,y:(x,y))
        post=self.surface.copy()
        for x,y in points:
            x,y=int(x),int(y);post[y-2:y+3,x-2:x+3]=30
        q.observe(fixture.frame(post,self.now+.033),delivered_ts=self.now+.04)
        r=q.observe(fixture.frame(post,self.now+.066),delivered_ts=self.now+.07)
        self.assertGreaterEqual(len(r['hypotheses']),2)
        self.game._on_object_event(dict(event='physical.object_result',data=r))
        self.assertEqual(r['object_decision']['state'],'UNKNOWN')
        self.assertEqual(self.game.hits[1],0)
        self.assertEqual(sum(s.play.call_count for s in self.sounds.values()),0)

    def test_normal_render_stays_frozen_before_decision(self):
        self.game.markers.append(can.ImpactMarker(120,90,True,'fixture',self.now))
        self.game.render(self.screen);before=pygame.image.tostring(self.screen,'RGB')
        self.freeze();self.now+=.10;self.game.render(self.screen)
        self.assertEqual(pygame.image.tostring(self.screen,'RGB'),before)

    def test_wrappers_do_not_project_debug_or_global_markers(self):
        from src.engine.scenes.overlay_scene import OverlayScene
        wrapper=OverlayScene(self.game)
        with patch('src.engine.scenes.overlay_scene.hit_visualizer.render') as hit,patch('src.engine.scenes.overlay_scene.scanner_debug_overlay.render') as debug,patch('src.engine.scenes.overlay_scene.scanner_status_overlay.render') as status:
            wrapper.render(self.screen)
        hit.assert_not_called();debug.assert_not_called();status.assert_not_called()

    def test_actual_registry_freeze_and_wrapper_presentation_hooks(self):
        from src.engine.scenes.game import GameScene
        from src.engine.scenes.overlay_scene import OverlayScene
        scene=GameScene('','');scene.game=self.game
        wrapped=OverlayScene(scene)
        with patch.object(self.registry,'_load_viewport_xywh',return_value=(0,0,800,600)),patch.object(self.registry,'_camera_transformers',return_value=('fixture',lambda x,y:(x,y),lambda x,y:(x,y))),patch.object(can.hit_input,'frozen_camera_to_screen',return_value=lambda x,y:(x,y)):
            snap=self.registry.snapshot(42,self.now,wrapped)
        self.assertIn(42,self.game._pending_object_shots)
        self.assertEqual(snap.object_camera_to_game(123,456),(123,456))
        original=snap.game_regions[0].metadata['v250_object_snapshot']['shape']
        self.game.manager.objects[0].geometry.move(200,0)
        self.assertEqual(snap.game_regions[0].metadata['v250_object_snapshot']['shape'],original)
        wrapped.on_frame_presented(self.now)
        self.assertTrue(scene.owns_hit_presentation)

    def test_final_query_ignores_later_animation_frames(self):
        snap=self.shoot('soda_can');before=json.dumps(snap.object_query.result,sort_keys=True)
        animation=np.zeros_like(self.surface)
        snap.object_query.observe(fixture.frame(animation,self.now+.03),delivered_ts=self.now+.04)
        self.assertEqual(json.dumps(snap.object_query.result,sort_keys=True),before)

    def test_pending_second_query_holds_first_shot_reaction(self):
        first=self.freeze();self.now+=.01;second=self.freeze()
        physical=dict(mode='OBJECT',status='NO_PHYSICAL_HIT',shot_id=first.shot_id,peak_ts=first.peak_ts,hypotheses=[])
        self.game._on_object_event(dict(event='physical.object_result',data=physical))
        before=pygame.image.tostring(self.screen,'RGB');self.game.render(self.screen)
        self.assertEqual(pygame.image.tostring(self.screen,'RGB'),before)
        self.assertIn(second.shot_id,self.game._pending_object_shots)

    def test_exit_reentry_releases_sounds_and_subscribes_once(self):
        self.game.on_exit()
        self.assertEqual(self.game._hit_sounds,{})
        for sound in self.sounds.values():sound.stop.assert_called_once()
        self.game.on_enter();self.game.on_enter()
        from src.engine.events import event_bus
        self.assertEqual(event_bus._subscribers.count(self.game._on_object_event),1)

    def test_old_same_hole_after_restart_is_not_new_damage(self):
        self.shoot('soda_can');self.game._start_match();self.live()
        snap=self.shoot('soda_can')
        self.assertEqual(snap.object_query.result['status'],'NO_PHYSICAL_HIT')
        self.assertEqual(self.game.hits[1],0)

    def test_full_match_streak_and_perfect_bonus_once(self):
        # This test isolates GAME scoring/round flow with a fixed camera mapping.
        # The actual uniform old-hole registration produces competing hypotheses;
        # its correct UNKNOWN result is preserved in the separate regression.
        register=patch('src.engine.camera.object_physical_query.CandidateGeneratorV2._register_current',
            side_effect=lambda _,post,pre,**kw:(pre,{'applied':False,'fixture':'fixed_registration'}))
        register.start();self.addCleanup(register.stop)
        for player in (1,2):
            expected=0
            for i,name in enumerate(('soda_can','rubber_duck','gold_can','beans_xxl','sardines')):
                spec=next(s for s in can.TARGET_SPECS if s.target_id==name)
                expected+=spec.points+min(can.MAX_STREAK_BONUS,i*can.STREAK_BONUS_STEP)
                self.shoot(name,level=30 if player==1 else 0) # explicit new visible change at old XY
                if i<4:self.live()
            self.game._finalize_current_player();self.game._finalize_current_player()
            self.assertEqual(self.game.score[player],expected+can.PERFECT_FIVE_BONUS)
            self.assertEqual(self.game.shots_used[player],5);self.assertEqual(self.game.hits[player],5)
            if player==1:self.game._advance_after_player_result();self.live()
        self.game._advance_after_player_result();self.assertEqual(self.game.phase,'game_over')

    def test_uniform_old_hole_registration_ambiguity_is_not_gameplay_truth(self):
        for name in ('soda_can','rubber_duck','gold_can','beans_xxl','sardines'):
            self.shoot(name);self.live() if self.game.shots_used[1]<5 else None
        self.game._start_player(2);self.live()
        self.shoot('soda_can',level=0);self.live()
        snap=self.shoot('rubber_duck',level=0)
        self.assertGreater(len(snap.object_query.result['object_decision']['supported_object_ids']),1)
        self.assertEqual(snap.object_query.result['object_decision']['state'],'UNKNOWN')
        self.assertEqual(self.game.hits[2],1)

    def test_playback_audio_is_visible_but_not_misclassified_or_scored(self):
        for name in ('rubber_duck','soda_can'):
            with self.subTest(cue=name):
                if self.game.phase!='live':self.live()
                snap=self.shoot(name);used=self.game.shots_used[1]
                self.now+=.01
                raw=AudioPeakEvent(self.now,.8,.1);self.audio._pending_dispatch.append(raw)
                received=[];self.audio.subscribe(received.append)
                with contextlib.redirect_stdout(io.StringIO()):self.audio.update()
                self.audio.unsubscribe(received.append)
                self.assertEqual(received,[raw]) # a simultaneous genuine shot must remain representable
                self.assertFalse(raw.diagnostics['engine_playback_context']['suppressed'])
                self.assertEqual(self.game.shots_used[1],used)
                self.assertTrue(self.audio.playback_diagnostics(self.game._query_owner,snap.shot_id)['nearby_audio_events'])

    def test_explicit_mute_prevents_output_without_trigger_holdoff(self):
        with patch.object(can,'CAN_SFX_ENABLED',False):snap=self.shoot('rubber_duck')
        self.sounds['rubber_duck'].play.assert_not_called()
        self.assertEqual(snap.object_gameplay['sfx']['status'],'MUTED_BY_EXPLICIT_OPTION')
        self.assertEqual(self.audio.cooldown_s,.08)

    def test_joined_record_has_frozen_shapes_times_and_no_duplicate_line(self):
        snap=self.shoot('soda_can')
        self.game._on_hit(S(is_simulated=False,shot_id=snap.shot_id,camera_x=1,camera_y=2,
            game_x=1,game_y=2,timestamp=self.now,source='camera'))
        scanner=S(audio_events=[S(shot_id=snap.shot_id,state='confirmed',emitted=True)])
        log=io.StringIO()
        with contextlib.redirect_stdout(log):
            self.registry.publish_object_comparisons(scanner);self.registry.publish_object_comparisons(scanner)
        lines=log.getvalue().splitlines();self.assertEqual(len(lines),1)
        record=json.loads(lines[0].split(' ',1)[1])
        self.assertEqual(len(record['frozen_regions']),7)
        self.assertEqual(record['gameplay']['player'],1)
        self.assertIsNotNone(record['gameplay']['reaction_presented_ts'])
        self.assertIsNotNone(record['gameplay']['sfx']['start_ts'])
        self.assertEqual(record['semantic'],'OBJECT_HIT')
        self.assertIsNotNone(record['diagnostic_xy']['camera_xy'])
        self.assertEqual(record['exact_coordinate_result']['camera_xy'],[1.,2.])
        self.assertNotIn('candidate_ledger',record)


class Properties(unittest.TestCase):
    def test_region_permutations_overlap_and_actual_pixel_invariants(self):
        rng=np.random.default_rng(1509)
        for case in range(32):
            regions=[CameraHitRegionV240(str(i),int(rng.integers(45,145)),int(rng.integers(45,145)),80,80) for i in range(7)]
            pre=np.full((300,300),140,np.uint8);post=pre.copy();post[128:133,128:133]=30
            results=[]
            for order in (regions,list(reversed(regions)),[regions[i] for i in rng.permutation(7)]):
                q=ObjectPhysicalQuery(fixture.snapshot(order),[fixture.frame(pre,1.94)],top_k=1+case%5)
                r=fixture.hit_result(q,post)
                xy=[h['camera_xy'] for h in r['hypotheses']];results.append((r['status'],xy,q.read_pixels))
                self.assertLessEqual(len(xy),min(3,1+case%5))
                for x,y in xy:
                    self.assertEqual(x,int(x));self.assertEqual(y,int(y));self.assertNotEqual(pre[int(y),int(x)],post[int(y),int(x)])
            self.assertEqual(results[0],results[1]);self.assertEqual(results[0],results[2])

    def test_random_crop_offsets_scales_viewports_roundtrip(self):
        from src.engine.input.object_hit_v2223 import transform_game_rect_to_camera_aabb,transform_camera_point_to_game
        rng=np.random.default_rng(2509)
        for _ in range(80):
            vx,vy=rng.uniform(-200,200,2);sx,sy=rng.uniform(.5,3,2);dx,dy=rng.uniform(300,600,2)
            x,y=rng.uniform(20,100,2);vp=(vx,vy,800,600)
            region=GameHitRegionV240('r',x,y,30,40)
            cam=transform_game_rect_to_camera_aabb(region,vp,lambda a,b:((a-vx)*sx+dx,(b-vy)*sy+dy))
            game=transform_camera_point_to_game(cam.x,cam.y,vp,lambda a,b:((a-dx)/sx+vx,(b-dy)/sy+vy))
            np.testing.assert_allclose(game,[x,y],atol=1e-8)

    def test_object_order_and_exact_boundary(self):
        from src.engine.game_objects import HitShapeSpec
        for order in (('a','b'),('b','a')):
            manager=ObjectManager()
            for name in order:manager.add(make_breakable_object(name,ObjectGeometry(100,60,60,80),shape=HitShapeSpec.ellipse()))
            snap=fixture.snapshot(game=manager.get_hit_regions());manager._snapshot_provider=lambda sid:snap
            result=dict(mode='OBJECT',shot_id=1,status='PHYSICAL_HIT',hypotheses=[
                dict(game_xy=[100,60],confirmed=True),dict(game_xy=[130,60],confirmed=True,rank=2)])
            resolution,h=manager.resolve_object_query(result,can.GAME_PROJECTILE)
            self.assertEqual(len(resolution.interactions),1);self.assertEqual(resolution.interactions[0].object_id,'b')

    def test_invalid_zero_bound_is_rejected(self):
        with self.assertRaises(ValueError):object_query_regions([],top_k=0)
        with self.assertRaises(ValueError):fixture.query(top_k=0)

    def test_cancel_is_owner_scoped_bounded_and_has_no_physical_hypotheses(self):
        registry=ObjectHitRegistryV2223()
        for sid,owner in ((1,'a'),(2,'b')):
            q,_=fixture.query();snap=fixture.snapshot(game=object_query_regions([GameHitRegionV240('r',100,60,60,80)],owner=owner),sid=sid)
            snap.object_query=q;registry._snapshots[sid]=snap
        registry.cancel_object_queries('a','reset')
        self.assertEqual(registry._snapshots[1].object_query.result['hypotheses'],[])
        self.assertFalse(registry._snapshots[2].object_query.final)

    def test_next_shot_frames_never_confirm_previous_query(self):
        registry=ObjectHitRegistryV2223();q,pre=fixture.query()
        first=fixture.snapshot();first.object_query=q;registry._snapshots[1]=first
        registry._snapshots[2]=replace(fixture.snapshot(sid=2),peak_ts=2.05)
        post=pre.copy();post[89:94,122:127]=30
        camera=S(get_ring_snapshot=lambda:[fixture.frame(post,2.03),fixture.frame(post,2.065)])
        with patch('src.engine.events.event_bus.emit'):
            registry.advance_object_queries(S(),camera,2.08)
        self.assertEqual(q.frames,1);self.assertEqual(q.result['status'],'UNCERTAIN')

    def test_region_and_concurrency_budgets_release_arrays(self):
        pre=np.full((240,360),140,np.uint8)
        regions=[CameraHitRegionV240(str(i),100,60,60,80) for i in range(65)]
        q=ObjectPhysicalQuery(fixture.snapshot(regions),[fixture.frame(pre,1.94)])
        self.assertEqual(q.result['reason'],'region_budget_exceeded');self.assertEqual(q.tiles,[])
        registry=ObjectHitRegistryV2223();camera=S(get_ring_snapshot=lambda:[fixture.frame(pre,1.94)])
        for sid in range(1,6):
            snap=fixture.snapshot(game=object_query_regions([GameHitRegionV240('can',100,60,60,80)]),sid=sid)
            registry._snapshots[sid]=snap;registry.begin_object_query(sid,S(),camera)
        self.assertEqual(registry._snapshots[5].object_query.result['reason'],'concurrent_query_budget_exceeded')

    def test_input_frame_read_error_is_uncertain_without_hypothesis(self):
        q,pre=fixture.query();snap=fixture.snapshot();snap.object_query=q
        registry=ObjectHitRegistryV2223();registry._snapshots[1]=snap
        with patch.object(q,'observe',side_effect=ValueError('fixture')),patch('src.engine.events.event_bus.emit'):
            registry.advance_object_queries(S(),S(get_ring_snapshot=lambda:[fixture.frame(pre,2.03)]),2.04)
        self.assertEqual(q.result['status'],'UNCERTAIN');self.assertEqual(q.tiles,[])

    def test_processing_deadline_aborts_without_late_hit(self):
        q,pre=fixture.query();post=pre.copy();post[89:94,122:127]=30
        clock=[0.];original=q._evidence
        def expensive(*args):
            value=original(*args);clock[0]+=.4;return value
        with patch('time.perf_counter',side_effect=lambda:clock[0]),patch.object(q,'_evidence',side_effect=expensive):
            result=q.observe(fixture.frame(post,2.03),delivered_ts=2.04)
        self.assertEqual(result['reason'],'processing_deadline_exceeded')
        self.assertEqual(result['status'],'UNCERTAIN');self.assertEqual(result['hypotheses'],[])
        self.assertEqual(q.tiles,[])

    def test_late_delivered_causal_frame_does_not_start_cv(self):
        q,pre=fixture.query()
        with patch('src.engine.camera.object_physical_query._gray',side_effect=AssertionError('late work')):
            result=q.observe(fixture.frame(pre,2.03),delivered_ts=2.4)
        self.assertEqual(result['status'],'UNCERTAIN');self.assertEqual(q.frames,0)


class SoundTechnical(unittest.TestCase):
    def test_sample_ranges_duration_reproducibility_and_levels(self):
        rms=[]
        for rate in (16000,22050,44100,48000):
            for spec in can.TARGET_SPECS:
                wave=can.hit_sound_wave(spec.target_id,rate)
                self.assertEqual(wave.dtype,np.int16);self.assertTrue(np.array_equal(wave,can.hit_sound_wave(spec.target_id,rate)))
                self.assertLessEqual(len(wave)/rate,.221);self.assertGreater(len(wave)/rate,.17)
                self.assertLessEqual(int(np.max(np.abs(wave))),7865)
                self.assertEqual(wave[0],0);rms.append(float(np.sqrt(np.mean(wave.astype(float)**2))))
        self.assertLess(max(rms)/min(rms),4.)

    def test_no_channel_failure_and_audio_unavailable_do_not_arm_guard(self):
        audio=AudioPeakDetector();sound=Mock();sound.get_length.return_value=.18;sound.play.return_value=None
        r=audio.play_feedback(sound,owner='test',shot_id=1,cue='can');self.assertEqual(r['status'],'NO_CHANNEL')
        sound.play.side_effect=pygame.error('gone');r=audio.play_feedback(sound,owner='test',shot_id=2,cue='duck')
        self.assertEqual(r['status'],'UNAVAILABLE');self.assertEqual(audio.last_peak_ts,0.)

    def test_trigger_decision_identical_with_playback_annotation(self):
        chunk=np.zeros(256,np.int16);chunk[128]=30000
        for active in (False,True):
            audio=AudioPeakDetector();audio.min_abs_peak=.1
            if active:
                sound=Mock();sound.get_length.return_value=.22
                with patch('time.time',return_value=1.99):audio.play_feedback(sound,owner='x',shot_id=1,cue='duck')
            with patch('time.time',return_value=2.):audio._process_chunk(chunk.tobytes())
            received=[];audio.subscribe(received.append)
            with contextlib.redirect_stdout(io.StringIO()):audio.update()
            self.assertEqual(len(received),1);self.assertAlmostEqual(received[0].timestamp,2.-127/16000.)
            self.assertEqual(audio.cooldown_s,.08)

    def test_live_telemetry_path_preserves_two_real_peaks_during_sfx(self):
        from src.engine.shot_track_v2226 import _install_audio_telemetry_patch
        _install_audio_telemetry_patch()
        audio=AudioPeakDetector();audio.min_abs_peak=.1
        sound=Mock();sound.get_length.return_value=.22
        with patch('time.time',return_value=2.):audio.play_feedback(sound,owner='x',shot_id=1,cue='duck')
        chunk=np.zeros(256,np.int16);chunk[128]=30000
        with contextlib.redirect_stdout(io.StringIO()):
            for now in (2.02,2.12):
                with patch('time.time',return_value=now):audio._process_chunk(chunk.tobytes())
            got=[];audio.subscribe(got.append);audio.update()
        self.assertEqual(len(got),2)
        self.assertTrue(all(not e.diagnostics['engine_playback_context']['suppressed'] for e in got))


if __name__=='__main__':
    unittest.main()
