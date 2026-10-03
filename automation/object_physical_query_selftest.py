"""OBJECT software regressions. Generated images are NOT physical validation."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as S
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pygame

from src.engine.camera.object_physical_query import ObjectPhysicalQuery
from src.engine.game_objects import ObjectManager, ObjectGeometry, HitShapeSpec, make_breakable_object
from src.engine.input.hit_input import HitInput
from src.engine.input.hit_regions import object_query_regions
from src.engine.input.object_hit_v2223 import (ObjectHitRegistryV2223,
    ObjectShotSnapshotV2223, CameraHitRegionV240, GameHitRegionV240)
from content.games import can_shootout_game as can


def frame(image,ts):
    return S(frame_bgr=image,timestamp=ts)


def snapshot(regions=None,game=(),sid=1):
    return ObjectShotSnapshotV2223(sid,2.,2.,(),game_regions=game,
        camera_regions=tuple(regions or [CameraHitRegionV240('can',100,60,60,80)]),
        transform_available=True,transform_method='fixture',viewport_xywh=(0,0,360,240))


def query(pre=None,regions=None,holes=(),**kwargs):
    pre=np.full((240,360),140,np.uint8) if pre is None else pre
    return ObjectPhysicalQuery(snapshot(regions),[frame(pre,1.94)],
        camera_to_game=lambda x,y:(x,y),known_holes=holes,**kwargs),pre


def hit_result(q,post):
    q.observe(frame(post,2.03),delivered_ts=2.032)
    return q.observe(frame(post,2.065),delivered_ts=2.067)


class QueryTests(unittest.TestCase):
    def test_object_evaluation_precedes_global_top_k(self):
        # Three stronger changes OUTSIDE targets must not starve another target.
        # All four are real changed pixels in this software fixture, not labels.
        manager=ObjectManager()
        for i,x in enumerate((55,165,275,385)):
            manager.add(make_breakable_object(str(i),ObjectGeometry(x,90,30,40)))
        game=tuple(manager.get_hit_regions())
        regions=tuple(CameraHitRegionV240(r.object_id,r.x,r.y,r.width,r.height) for r in game)
        snap=snapshot(regions,game)
        manager._snapshot_provider=lambda sid:snap
        pre=np.full((240,480),140,np.uint8);post=pre.copy()
        for x in (70,180,290):post[142:147,x-2:x+3]=20
        post[107:112,398:403]=100
        q=ObjectPhysicalQuery(snap,[frame(pre,1.94)],camera_to_game=lambda x,y:(x,y))
        r=hit_result(q,post)
        resolved,hypothesis=manager.resolve_object_query(r,can.GAME_PROJECTILE)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.interactions[0].object_id,'3')
        self.assertLessEqual(len(r['hypotheses']),3)
        self.assertGreater(hypothesis['global_evidence_rank'],3)
        self.assertEqual(len(r['per_object']),4)
        self.assertEqual(manager.resolve_object_query(r,can.GAME_PROJECTILE),(None,None))

    def test_duck_visible_beak_is_in_frozen_shape(self):
        from src.engine.game_objects.geometry import shape_contains,shape_bounds
        game=can.ShootStuffGame('',pygame.Rect(0,0,800,600))
        spec=next(s for s in can.TARGET_SPECS if s.target_id=='rubber_duck')
        shape=game._shape_for_spec(spec).snapshot(ObjectGeometry(0,0,115,100))
        self.assertTrue(shape_contains(shape,119,26))
        self.assertGreater(shape_bounds(shape)[2],115)

    def test_merged_crop_reserves_components_for_each_object(self):
        # More than 24 stronger components around one target, weaker new change
        # at another. Their context windows overlap and are scanned only once.
        regions=[CameraHitRegionV240('a',45,40,90,150),CameraHitRegionV240('b',150,90,30,30)]
        q,pre=query(regions=regions);self.assertEqual(len(q.tiles),1)
        post=pre.copy()
        for x in (48,62,76,90,104):
            for y in (48,70,92,114,136,158):post[y:y+3,x:x+3]=20
        post[102:107,161:166]=100
        r=hit_result(q,post)
        b=next(o for o in r['per_object'] if o['object_id']=='b')
        self.assertTrue(any(c['confirmed'] and 150<=c['camera_xy'][0]<=180 for c in b['candidates']))
        self.assertLessEqual(q.candidate_count,48)  # 24 per frame, unchanged.
        a=next(o for o in r['per_object'] if o['object_id']=='a')
        self.assertGreater(a['production']['raw_components'],a['production']['retained_components'])

    def test_nearby_dedupe_cannot_erase_inside_shape_observation(self):
        manager=ObjectManager();obj=manager.add(make_breakable_object('a',ObjectGeometry(100,60,30,30)))
        q=ObjectPhysicalQuery(snapshot([CameraHitRegionV240('a',100,60,30,30)],(obj.get_hit_region(),)),
            [frame(np.full((240,360),140,np.uint8),1.94)])
        def c(x,inside,score):return dict(camera_xy=[x,75],confirmed=True,evidence_score=score,
            regions=[dict(object_id='a',inside_exact_shape=inside)])
        selected=q._select_hypotheses([c(131,False,100),c(129,True,5)],{})
        self.assertEqual(selected[0]['camera_xy'],[129,75])
        self.assertEqual(len(selected),2)

    def test_order_independence_and_physical_requirement_before_shape(self):
        rng=np.random.default_rng(915)
        for _ in range(12):
            order=rng.permutation(7)
            regions=[CameraHitRegionV240(str(i),30+35*i,80,12,25) for i in order]
            q,_=query(regions=regions,top_k=1)
            candidates=[dict(camera_xy=[36+35*i,90],confirmed=i!=6,evidence_score=float(70-i),
                regions=[dict(object_id=str(i),inside_exact_shape=i in (5,6))]) for i in range(7)]
            rng.shuffle(candidates)
            selected=q._select_hypotheses(candidates,{})
            self.assertEqual(selected[0]['camera_xy'],[211,90])
            self.assertEqual(len(q.per_object),7)
            self.assertFalse(next(r for r in q.per_object if r['object_id']=='6')['candidates'][0]['confirmed'])

    def test_duck_drawn_interior_matches_shape_at_multiple_sizes(self):
        import cv2
        from src.engine.game_objects.geometry import shape_contains
        game=can.ShootStuffGame('',pygame.Rect(0,0,800,600));spec=can.TARGET_SPECS[4]
        for w,h in ((115,100),(31,27),(23,20),(11,10)):
            geometry=ObjectGeometry(20.2,20.2,w+.1,h+.1)
            shape=game._shape_for_spec(spec,geometry).snapshot(geometry)
            surf=pygame.Surface((180,150));game._draw_duck(surf,pygame.Rect(20,20,w,h),spec)
            rgb=pygame.surfarray.array3d(surf)
            solid=np.any(rgb!=0,axis=2)&np.any(rgb!=np.array([55,53,40]),axis=2)
            # Exclude the one-pixel raster boundary, not an added collision margin.
            interior=cv2.erode(solid.astype(np.uint8),np.ones((3,3),np.uint8))
            for x,y in np.argwhere(interior):self.assertTrue(shape_contains(shape,x+.5,y+.5))
            self.assertFalse(shape_contains(shape,20,20))  # transparent corner

    def test_crop_capture_preserves_actual_pre_post_and_geometry(self):
        saved=[]
        q,pre=query(capture_crop=lambda *args:saved.append(deepcopy(args)))
        post=pre.copy();post[89:94,122:127]=30;hit_result(q,post)
        self.assertEqual([s[1] for s in saved],['pre','post','post'])
        for sid,kind,ts,bbox,pixels in saved:
            x0,y0,x1,y1=map(int,bbox)
            np.testing.assert_array_equal(pixels,(pre if kind=='pre' else post)[y0:y1,x0:x1])
            self.assertEqual(sid,1)

    def test_actual_observed_coordinates_and_two_frames(self):
        q,pre=query();post=pre.copy();post[89:94,122:127]=30
        provisional=q.observe(frame(post,2.03),delivered_ts=2.032)
        self.assertEqual(provisional['status'],'UNCERTAIN')
        r=q.observe(frame(post,2.065),delivered_ts=2.067)
        self.assertEqual(r['status'],'PHYSICAL_HIT')
        for h in r['hypotheses']:
            x,y=h['camera_xy'];self.assertEqual(x,int(x));self.assertEqual(y,int(y))
            self.assertNotEqual(pre[int(y),int(x)],post[int(y),int(x)])
            self.assertTrue(h['confirmed']);self.assertIsNotNone(h['previous_same_xy_evidence'])
        self.assertEqual(q.tiles,[])

    def test_unchanged_old_hole_and_no_truth_update(self):
        pre=np.full((240,360),140,np.uint8);pre[89:94,122:127]=30
        holes=[dict(camera_x=124.,camera_y=91.,hit_count=3)]
        original=deepcopy(holes);q,_=query(pre=pre,holes=holes)
        q.observe(frame(pre,2.03),delivered_ts=2.032)
        r=q.observe(frame(pre,2.21),delivered_ts=2.212)
        self.assertEqual(r['status'],'NO_PHYSICAL_HIT');self.assertEqual(r['hypotheses'],[])
        self.assertEqual(holes,original)

    def test_beside_old_hole_is_allowed(self):
        pre=np.full((240,360),140,np.uint8);pre[89:94,122:127]=30
        q,_=query(pre=pre,holes=[dict(camera_x=124.,camera_y=91.)])
        post=pre.copy();post[90:94,130:134]=35
        r=hit_result(q,post);self.assertEqual(r['status'],'PHYSICAL_HIT')
        self.assertLess(r['hypotheses'][0]['nearest_known_hole_px'],15.)

    def test_same_hole_enlargement_is_allowed(self):
        pre=np.full((240,360),140,np.uint8);pre[90:93,123:126]=30
        q,_=query(pre=pre,holes=[dict(camera_x=124.,camera_y=91.)])
        post=pre.copy();post[88:95,121:128]=30
        r=hit_result(q,post);self.assertEqual(r['status'],'PHYSICAL_HIT')
        x,y=map(int,r['hypotheses'][0]['camera_xy'])
        self.assertNotEqual(pre[y,x],post[y,x])

    def test_second_shot_uses_new_surface_not_startup(self):
        q,pre=query();post=pre.copy();post[89:94,122:127]=30
        self.assertEqual(hit_result(q,post)['status'],'PHYSICAL_HIT')
        next_q,_=query(pre=post)
        next_q.observe(frame(post,2.03),delivered_ts=2.032)
        self.assertEqual(next_q.observe(frame(post,2.21),delivered_ts=2.212)['status'],'NO_PHYSICAL_HIT')

    def test_no_unrelated_board_scan(self):
        q,pre=query();post=pre.copy();post[210:225,315:330]=0
        q.observe(frame(post,2.03),delivered_ts=2.032)
        r=q.observe(frame(post,2.21),delivered_ts=2.212)
        self.assertEqual(r['status'],'NO_PHYSICAL_HIT')
        self.assertLess(r['processed_crop_pixels'],pre.size*.4)
        self.assertLess(r['search_fraction'],.3)

    def test_overlap_union_and_contributors(self):
        regions=[CameraHitRegionV240('a',100,60,60,80),CameraHitRegionV240('b',115,65,60,80)]
        q,pre=query(regions=regions);self.assertEqual(len(q.tiles),1)
        post=pre.copy();post[89:94,122:127]=30
        r=hit_result(q,post);self.assertEqual(len(r['hypotheses']),1)
        self.assertEqual({a['object_id'] for a in r['hypotheses'][0]['regions']},{'a','b'})

    def test_default_and_explicit_top_k_bound(self):
        for k in (3,100):
            q,pre=query(regions=[CameraHitRegionV240('a',50,40,220,160)],top_k=k)
            post=pre.copy()
            for x in (80,120,160,200,240):post[100:105,x:x+5]=20
            r=hit_result(q,post);self.assertEqual(len(r['hypotheses']),3)
            self.assertEqual([h['rank'] for h in r['hypotheses']],[1,2,3])

    def test_persistence_cannot_transfer_to_another_xy(self):
        q,pre=query();a=pre.copy();b=pre.copy();a[80:85,110:115]=20;b[120:125,150:155]=20
        q.observe(frame(a,2.03),delivered_ts=2.032)
        r=q.observe(frame(b,2.21),delivered_ts=2.212)
        self.assertEqual(r['status'],'UNCERTAIN')
        self.assertFalse(any(h['confirmed'] for h in r['hypotheses']))

    def test_duplicate_or_future_frame_cannot_confirm(self):
        q,pre=query();post=pre.copy();post[89:94,122:127]=30
        q.observe(frame(post,2.03),delivered_ts=2.032)
        q.observe(frame(post,2.03),delivered_ts=2.04)
        q.observe(frame(post,2.1),delivered_ts=2.04)
        self.assertEqual(q.frames,1);self.assertFalse(q.final)

    def test_recent_pre_activity_is_context_not_hit_veto(self):
        pre=np.full((240,360),140,np.uint8);earlier=pre.copy();earlier[89:94,122:127]=160
        q=ObjectPhysicalQuery(snapshot(),[frame(earlier,1.88),frame(earlier,1.91),frame(pre,1.94)])
        post=pre.copy();post[89:94,122:127]=30
        r=hit_result(q,post);self.assertTrue(r['pre_activity_context'])
        self.assertEqual(r['status'],'PHYSICAL_HIT')

    def test_pre_change_without_resolved_step_is_uncertain_not_veto(self):
        pre=np.full((240,360),140,np.uint8);latest=pre.copy();latest[89:94,122:127]=30
        q=ObjectPhysicalQuery(snapshot(),[frame(pre,1.88),frame(pre,1.91),frame(latest,1.94)])
        q.observe(frame(latest,2.03),delivered_ts=2.032)
        self.assertEqual(q.observe(frame(latest,2.21),delivered_ts=2.212)['status'],'UNCERTAIN')

    def test_broad_change_is_not_confirmed_local_damage(self):
        q,pre=query();post=np.full_like(pre,240)
        r=hit_result(q,post)
        if not q.final:r=q.finish_deadline(2.4)
        self.assertEqual(r['status'],'UNCERTAIN')

    def test_no_static_surface_fallback_and_budget_failure(self):
        q=ObjectPhysicalQuery(snapshot(),[])
        self.assertEqual(q.result['status'],'UNCERTAIN')
        pre=np.zeros((1200,1600),np.uint8)
        with patch('src.engine.camera.object_physical_query._gray',side_effect=AssertionError('must not inspect')):
            q=ObjectPhysicalQuery(snapshot([CameraHitRegionV240('a',0,0,1600,1200)]),[frame(pre,1.9)])
        self.assertEqual(q.result['reason'],'pixel_budget_exceeded')

    def test_nonzero_camera_crop_mapping_and_region_freeze(self):
        pre=np.full((600,900),140,np.uint8);region=CameraHitRegionV240('can',600,350,60,60)
        q,_=query(pre=pre,regions=[region]);post=pre.copy();post[378:383,628:633]=30
        r=hit_result(q,post);self.assertEqual(r['status'],'PHYSICAL_HIT')
        x,y=r['hypotheses'][0]['camera_xy'];self.assertTrue(628<=x<633 and 378<=y<383)
        self.assertEqual(region.x,600)

    def test_engine_result_is_not_global_truth_or_probability(self):
        q,pre=query();post=pre.copy();post[89:94,122:127]=30
        r=hit_result(q,post)
        self.assertFalse(r['global_coordinate_authority']);self.assertFalse(r['truth_or_learning_input'])
        self.assertEqual(r['hypotheses'][0]['confidence_semantics'],'uncalibrated_evidence_strength')

    def test_geometry_change_returns_uncertain(self):
        q,pre=query()
        self.assertEqual(q.observe(frame(pre[:100],2.03),delivered_ts=2.04)['status'],'UNCERTAIN')
        q=ObjectPhysicalQuery(snapshot(),[frame(pre[:100],1.9),frame(pre,1.94)])
        self.assertEqual(q.result['reason'],'pre_camera_geometry_changed')

    def test_source_frames_remain_byte_identical(self):
        q,pre=query();post=pre.copy();post[89:94,122:127]=30
        before_pre=pre.tobytes();before_post=post.tobytes()
        hit_result(q,post)
        self.assertEqual(pre.tobytes(),before_pre);self.assertEqual(post.tobytes(),before_post)


class IntegrationTests(unittest.TestCase):
    def test_frozen_homography_and_scanport(self):
        h=HitInput();h.homography=np.array([[2,0,10],[0,3,20],[0,0,1]],float)
        frozen=h.frozen_camera_to_screen('homography');h.homography[:]=np.eye(3)
        self.assertEqual(frozen(4,5),(18.,35.))
        with patch('src.engine.input.hit_input.load_scanport_rect',return_value=pygame.Rect(100,200,400,200)),patch('src.engine.input.hit_input.load_viewport_rect',return_value=pygame.Rect(10,20,800,400)):
            frozen=h.frozen_camera_to_screen('scanport')
        self.assertEqual(frozen(300,300),(410.,220.))

    def test_api_marks_copies_only(self):
        regions=(GameHitRegionV240('a',1,2,3,4),)
        marked=object_query_regions(regions)
        self.assertNotIn('physical_query',regions[0].metadata)
        self.assertEqual(marked[0].metadata['physical_query']['top_k'],3)

    def test_unrequested_snapshots_do_no_work(self):
        registry=ObjectHitRegistryV2223();registry._snapshots[1]=snapshot()
        camera=Mock();registry.begin_object_query(1,S(),camera)
        registry.advance_object_queries(S(),camera,2.1)
        camera.get_ring_snapshot.assert_not_called()

    def test_runtime_publication_once_without_global_result_or_frame_cursor(self):
        q,pre=query();post=pre.copy();post[89:94,122:127]=30
        snap=snapshot();snap.object_query=q
        registry=ObjectHitRegistryV2223();registry._snapshots[1]=snap
        camera=Mock();camera.get_ring_snapshot.return_value=[frame(post,2.03),frame(post,2.065)]
        with patch('src.engine.events.event_bus.emit') as emit:
            registry.advance_object_queries(S(),camera,2.08)
            registry.advance_object_queries(S(),camera,2.09)
        emit.assert_called_once();self.assertEqual(emit.call_args.args[0],'physical.object_result')
        camera.get_new_frames_since_last_pickup.assert_not_called()

    def test_recent_surface_is_after_previous_shot(self):
        registry=ObjectHitRegistryV2223();pre=np.full((240,360),140,np.uint8)
        snap=snapshot(game=object_query_regions([GameHitRegionV240('can',100,60,60,80)]))
        registry._snapshots[1]=replace(snap,shot_id=1,peak_ts=1.8)
        registry._snapshots[2]=replace(snap,shot_id=2)
        camera=S(get_ring_snapshot=lambda:[frame(pre,1.6),frame(pre,1.85),frame(pre,1.94)])
        registry.begin_object_query(2,S(known_holes=[]),camera)
        self.assertTrue(all(t>1.8 for t in registry._snapshots[2].object_query.pre_times))

    def test_object_exception_cannot_break_global_loop(self):
        q,pre=query();snap=snapshot();snap.object_query=q
        registry=ObjectHitRegistryV2223();registry._snapshots[1]=snap
        camera=S(get_ring_snapshot=lambda:[frame(pre,2.03)])
        with patch.object(q,'observe',side_effect=RuntimeError('fixture')),patch('src.engine.events.event_bus.emit'):
            registry.advance_object_queries(S(),camera,2.04)
        self.assertEqual(q.result['status'],'UNCERTAIN')

    def test_prepare_exception_cannot_break_global_loop(self):
        registry=ObjectHitRegistryV2223();pre=np.full((240,360),140,np.uint8)
        snap=snapshot(game=object_query_regions([GameHitRegionV240('can',100,60,60,80)]))
        registry._snapshots[1]=snap
        camera=S(get_ring_snapshot=lambda:[frame(pre,1.94)])
        with patch.object(ObjectPhysicalQuery,'_prepare',side_effect=RuntimeError('fixture')):
            registry.begin_object_query(1,S(known_holes=[]),camera)
        self.assertEqual(snap.object_query.result['status'],'UNCERTAIN')
        self.assertEqual(snap.object_query.result['reason'],'object_prepare_error:RuntimeError')

    def test_object_diagnostics_do_not_change_global_metrics(self):
        from src.engine.physical_trace import selector_snapshot
        trace=dict(shot_id=1,stages=[],outcome=dict(emitted=False))
        expected=selector_snapshot(trace)
        trace['object_query_diagnostic']=dict(status='PHYSICAL_HIT',hypotheses=[dict(camera_xy=[1,2])])
        trace['object_gameplay_diagnostic']=dict(accepted=True)
        self.assertEqual(selector_snapshot(trace),expected)

    def test_exact_shape_and_no_margin_award_then_one_hypothesis(self):
        manager=ObjectManager();obj=manager.add(make_breakable_object('can',ObjectGeometry(100,60,60,80),shape=HitShapeSpec.ellipse()))
        snap=snapshot(game=(obj.get_hit_region(),));manager._snapshot_provider=lambda sid:snap
        obj.geometry.move(300,0)  # frozen shape is authoritative
        result=dict(mode='OBJECT',shot_id=1,status='PHYSICAL_HIT',hypotheses=[
            dict(game_xy=[101,61],confirmed=True,rank=1), # AABB corner outside ellipse
            dict(game_xy=[130,100],confirmed=True,rank=2)])
        resolution,h=manager.resolve_object_query(result,can.GAME_PROJECTILE)
        self.assertEqual(h['rank'],2);self.assertEqual(len(resolution.interactions),1)
        self.assertEqual((resolution.game_x,resolution.game_y),(130,100))
        self.assertEqual(manager.resolve_object_query(result,can.GAME_PROJECTILE),(None,None))

    def test_near_miss_uncertain_missing_snapshot_never_damage(self):
        for status,xy,frozen in [('PHYSICAL_HIT',[99,100],True),('UNCERTAIN',[130,100],True),('PHYSICAL_HIT',[130,100],False)]:
            manager=ObjectManager();obj=manager.add(make_breakable_object('can',ObjectGeometry(100,60,60,80)))
            manager._snapshot_provider=lambda sid:snapshot(game=(obj.get_hit_region(),)) if frozen else None
            r=manager.resolve_object_query(dict(mode='OBJECT',shot_id=1,status=status,hypotheses=[dict(game_xy=xy,confirmed=True)]))
            self.assertEqual(r,(None,None));self.assertTrue(obj.hittable)


class CanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init();pygame.display.set_mode((800,600))

    def make_game(self,target='rubber_duck'):
        game=can.ShootStuffGame('',pygame.Rect(0,0,800,600));game.on_enter();game.phase='live'
        self.addCleanup(game.on_exit)
        obj=next(o for o in game.manager.objects if can.TARGET_SPECS[o.metadata['spec_index']].target_id==target)
        snap=snapshot(game=game.get_hit_regions())
        game.manager._snapshot_provider=lambda sid:snap
        sound=Mock();sound.get_length.return_value=.22;game._hit_sounds={target:sound}
        result=dict(mode='OBJECT',shot_id=1,status='PHYSICAL_HIT',peak_ts=2.,hypotheses=[
            dict(game_xy=list(obj.geometry.center),camera_xy=list(obj.geometry.center),confirmed=True,rank=1)])
        return game,obj,snap,sound,result

    def test_duck_and_can_sound_once_and_no_duplicate_shot(self):
        for target in ('rubber_duck','soda_can'):
            game,obj,snap,sound,result=self.make_game(target)
            with patch.object(can,'hit_context_snapshot_for_shot',return_value=snap):
                game._on_object_event(dict(event='physical.object_result',data=result))
                game._on_object_event(dict(event='physical.object_result',data=result))
            sound.play.assert_called_once();self.assertEqual(game.shots_used[1],1)
            self.assertEqual(game.hits[1],1);self.assertFalse(snap.object_gameplay['diagnostic_assist'])

    def test_miss_rejected_uncertain_no_sound(self):
        for status in ('NO_PHYSICAL_HIT','UNCERTAIN','PHYSICAL_HIT'):
            game,obj,snap,sound,result=self.make_game()
            result['status']=status;result['hypotheses'][0]['game_xy']=[-5,-5]
            with patch.object(can,'hit_context_snapshot_for_shot',return_value=snap):
                game._on_object_event(dict(event='physical.object_result',data=result))
            sound.play.assert_not_called();self.assertEqual(game.hits[1],0)

    def test_current_callback_is_comparison_only(self):
        game,obj,snap,sound,result=self.make_game();snap.object_query=S(result=result)
        x,y=obj.geometry.center
        with patch.object(can,'hit_context_snapshot_for_shot',return_value=snap):
            game._on_hit(S(is_simulated=False,shot_id=1,camera_x=x,camera_y=y,game_x=x,game_y=y,timestamp=2.4,source='camera'))
        self.assertEqual(game.shots_used[1],0);sound.play.assert_not_called()
        self.assertEqual(snap.object_current['camera_xy'],[x,y])

    def test_ring_to_event_bus_to_game_then_current_is_one_decision(self):
        game,obj,snap,sound,_=self.make_game('soda_can')
        game._live_presented_ts=1.8
        snap=replace(snap,game_regions=game.get_hit_regions())
        registry=ObjectHitRegistryV2223()
        snap=replace(snap,camera_regions=tuple(CameraHitRegionV240(r.object_id,r.x,r.y,r.width,r.height)
            for r in snap.game_regions),object_camera_to_game=lambda x,y:(x,y))
        registry._snapshots[1]=snap;game.manager._snapshot_provider=lambda sid:snap
        pre=np.full((600,800),140,np.uint8);post=pre.copy()
        x,y=map(int,obj.geometry.center);post[y-2:y+3,x-2:x+3]=30
        camera=S(get_ring_snapshot=lambda:[frame(pre,1.94),frame(post,2.03),frame(post,2.065)])
        with patch.object(can,'hit_context_snapshot_for_shot',return_value=snap):
            registry.begin_object_query(1,S(known_holes=[]),camera)
            registry.advance_object_queries(S(),camera,2.08)
            game._on_hit(S(is_simulated=False,shot_id=1,camera_x=3,camera_y=4,
                game_x=3,game_y=4,timestamp=2.8,source='camera'))
        self.assertEqual(snap.object_query.result['status'],'PHYSICAL_HIT')
        self.assertEqual(game.hits[1],1);self.assertEqual(game.shots_used[1],1)
        sound.play.assert_called_once();self.assertEqual(snap.object_current['camera_xy'],[3.,4.])

    def test_comparison_display_holds_until_current_finishes(self):
        game,obj,snap,sound,result=self.make_game()
        screen=pygame.Surface((800,600));game.render(screen);before=pygame.image.tostring(screen,'RGB')
        with patch.object(can,'OBJECT_COMPARE_HOLD_VISUALS',True),patch.object(can,'hit_context_snapshot_for_shot',return_value=snap),patch.object(can.time,'time',return_value=2.1):
            game.get_hit_regions()
            game._on_object_event(dict(event='physical.object_result',data=result))
            game.render(screen)
            self.assertEqual(pygame.image.tostring(screen,'RGB'),before)
            snap.object_current['state']='confirmed';game.render(screen)
            self.assertNotEqual(pygame.image.tostring(screen,'RGB'),before)
        sound.play.assert_called_once()

    def test_audio_unavailable_and_lost_mixer_are_safe(self):
        with patch.object(pygame.mixer,'get_init',return_value=None):self.assertEqual(can.build_hit_sounds(),{})
        game,obj,snap,sound,result=self.make_game();sound.play.side_effect=pygame.error('device lost')
        game._play_hit_sound(can.TARGET_SPECS[obj.metadata['spec_index']],1)

    def test_sound_generation_cached_and_distinct(self):
        with patch.object(can,'hit_sound_wave',wraps=can.hit_sound_wave) as generate:
            sounds=can.build_hit_sounds()
        self.assertEqual(len(sounds),7);self.assertEqual(generate.call_count,7)
        wave=can.hit_sound_wave('rubber_duck',22050)
        self.assertFalse(np.array_equal(wave,can.hit_sound_wave('soda_can',22050)))
        self.assertEqual(wave.dtype,np.int16)

    def test_match_controls_and_projected_size_preserved(self):
        game,obj,snap,sound,result=self.make_game()
        self.assertEqual(game.distance_m,20);self.assertEqual(can.SHOTS_PER_PLAYER,5)
        width=game._projected_size_px(can.TARGET_SPECS[0])[0]
        game._change_distance(5);self.assertEqual(game.distance_m,25)
        self.assertLess(game._projected_size_px(can.TARGET_SPECS[0])[0],width)
        self.assertEqual(game.shots_used,{1:0,2:0});self.assertEqual(len(game.manager.objects),7)


if __name__=='__main__':
    unittest.main()
