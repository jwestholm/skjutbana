"""Local glow contracts: timing, masks, causality, OBJECT semantics and lifecycle. No hardware."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ['SKJUTBANA_SOURCE_RECORD']='0'
import unittest
from dataclasses import replace
from types import SimpleNamespace as N
from unittest.mock import patch
import numpy as np
from src.engine.precision_flash import FlashConfig,PrecisionFlash,resolve_policy,find_new_holes
from src.engine.precision_object_glow import (GlowConfig,glow_mask,object_highlight_mask,
    measurement_circles,measure_glow,ObjectGlowQuery,decide_glow,ObjectGlow,distance_mm)
from src.engine.input.object_hit_v2223 import GameHitRegionV240

SIZE=(200,140)

def region(oid='one',x=70,y=45,w=30,h=35,entity=None):
    shape=dict(kind='rect',x=x,y=y,width=w,height=h)
    return GameHitRegionV240(oid,x,y,w,h,metadata=dict(v250_object_snapshot=dict(shape=shape,entity_id=entity or oid)))

def projection(p):return np.asarray(p,dtype=float).reshape(-1,2)

def frames(on,off,mask,holes=(),*,normal=45,seed=3):
    rng=np.random.default_rng(seed);result=[]
    for ts in np.arange(on-.40,off+.35,.04):
        # Explicit projector/exposure/receive delay; plateau candidates must sit
        # fully inside this interval, including a partial rolling transition.
        start=ts-.066-.0167;end=ts-.066
        glow=start>=on+.06 and end<=off+.02
        image=np.full(mask.shape,normal,dtype=float)
        if glow:image[mask>0]=170.
        for x,y in holes:
            yy,xx=np.mgrid[:len(image),:image.shape[1]]
            image*=1.-.30*np.exp(-((xx-x)**2+(yy-y)**2)/(2*1.35**2))
        image=np.clip(image+rng.normal(0,.35,image.shape),0,255).astype(np.uint8)
        result.append(N(timestamp=float(ts),frame_bgr=image))
    return result

def evidence(holes=()):
    regions=[region()];mask=glow_mask(regions,SIZE,12);cfg=GlowConfig()
    ref=measure_glow(frames(10,10.4,mask),10,10.4,None,mask,(0,0,*SIZE),projection,cfg,FlashConfig(edge_margin_px=3),is_reference=True)
    assert ref['valid'],ref
    post=measure_glow(frames(20,20.18,mask,holes),20,20.18,ref['image'],mask,(0,0,*SIZE),projection,cfg,FlashConfig(edge_margin_px=3),camera_valid=ref['camera_valid'])
    return ref,post

class PolicyTests(unittest.TestCase):
    def test_opt_in_and_off_wins(self):
        s=N(hit_precision='OBJECT',precision_flash='object_glow')
        self.assertTrue(resolve_policy(s,env={})[0]);self.assertFalse(resolve_policy(s,env={'SKJUTBANA_PRECISION_FLASH':'off'})[0])
        s.precision_flash='off';self.assertFalse(resolve_policy(s,env={'SKJUTBANA_PRECISION_FLASH':'on'})[0])
    def test_cadence_has_a_hard_lower_bound_and_nan_is_rejected(self):
        self.assertEqual(GlowConfig.from_scene(N(precision_glow=dict(min_interval_s=.01))).min_interval_s,2.)
        with self.assertRaises(ValueError):GlowConfig.from_scene(N(precision_glow=dict(radius_px=float('nan'))))

class MeasurementTests(unittest.TestCase):
    def test_median_reference_and_post_find_new_hole(self):
        ref,post=evidence([(85,60)]);self.assertTrue(post['valid'],post)
        self.assertGreaterEqual(ref['plateau_frames'],5);self.assertGreaterEqual(post['plateau_frames'],3)
        self.assertEqual(len(post['holes']),1)
        self.assertAlmostEqual(post['holes'][0]['camera_x'],85,delta=.5)
    def test_no_hole_is_unknown_not_a_miss(self):
        _,post=evidence();q=ObjectGlowQuery(N(shot_id=1,peak_ts=19.,game_regions=[region()],camera_regions=[]),lambda:21.)
        decision,_=decide_glow(q,post['holes'],projection,(0,0,*SIZE),(1,1),post)
        self.assertEqual(decision['state'],'UNKNOWN')
    def test_near_miss_is_measured_without_expanding_collision(self):
        _,post=evidence([(114,60)]);self.assertEqual(len(post['holes']),1)
        q=ObjectGlowQuery(N(shot_id=1,peak_ts=19.,game_regions=[region()],camera_regions=[]),lambda:21.)
        decision,_=decide_glow(q,post['holes'],projection,(0,0,*SIZE),(2,2),post)
        self.assertEqual(decision['state'],'NO_OBJECT_HIT');self.assertAlmostEqual(decision['distance_mm'],28,delta=1.)
    def test_outside_mask_and_old_holes_do_not_create_candidates(self):
        r=np.full((100,160),170,np.float32);p=r.copy();p[48:53,20:25]=80
        mask=np.zeros_like(r,np.uint8);mask[15:85,60:140]=1
        r[40:44,90:94]=p[40:44,90:94]=60
        found,_=find_new_holes(r,p,FlashConfig(edge_margin_px=2),valid_mask=mask)
        self.assertEqual(found,[])
    def test_two_projector_frames_are_not_assumed_to_be_camera_frames(self):
        mask=glow_mask([region()],SIZE,12)
        result=measure_glow(frames(10,10+.033,mask),10,10+.033,None,mask,(0,0,*SIZE),projection,GlowConfig(),FlashConfig(),is_reference=True)
        self.assertFalse(result['valid']);self.assertEqual(result['reason'],'insufficient_causal_plateau')
    def test_one_pixel_old_hole_shift_is_unknown_not_a_new_hit(self):
        mask=glow_mask([region()],SIZE,12);cfg=GlowConfig();fc=FlashConfig(edge_margin_px=3)
        ref=measure_glow(frames(10,10.4,mask,[(85,60)]),10,10.4,None,mask,(0,0,*SIZE),projection,cfg,fc,is_reference=True)
        post=measure_glow(frames(20,20.18,mask,[(86,60)]),20,20.18,ref['image'],mask,(0,0,*SIZE),projection,cfg,fc,camera_valid=ref['camera_valid'])
        self.assertFalse(post['valid']);self.assertEqual(post['reason'],'bipolar_change_camera_or_scene_motion')

    def test_future_shot_cutoff_cannot_enter_reference(self):
        mask=glow_mask([region()],SIZE,12)
        result=measure_glow(frames(10,10.4,mask),10,10.4,None,mask,(0,0,*SIZE),projection,GlowConfig(),FlashConfig(),cutoff=10.19,is_reference=True)
        self.assertFalse(result['valid'])
    def test_overlap_and_multiple_holes_abstain(self):
        _,post=evidence([(85,60)]);q=ObjectGlowQuery(N(shot_id=1,peak_ts=19.,game_regions=[region(),region('two')],camera_regions=[]),lambda:21.)
        decision,_=decide_glow(q,post['holes'],projection,(0,0,*SIZE),(1,1),post)
        self.assertEqual(decision['state'],'UNKNOWN')
        decision,_=decide_glow(q,post['holes']*2,projection,(0,0,*SIZE),(1,1),post)
        self.assertEqual(decision['state'],'UNKNOWN')

class LifecycleTests(unittest.TestCase):
    def make_glow(self):
        clock=N(t=1000.);parent=PrecisionFlash(clock=lambda:clock.t,monotonic=lambda:clock.t,log=lambda *_:None)
        glow=ObjectGlow(parent);glow.enabled=True;glow.mask=glow_mask([region()],SIZE,12)
        glow.highlight_mask=object_highlight_mask([region()],SIZE);glow.circles=measurement_circles([region()],0);glow.viewport=(0,0,*SIZE)
        glow.cfg=GlowConfig();glow.scene=N();return clock,parent,glow
    def test_flash_timing_starts_at_presented_flip_and_rest_is_enforced(self):
        clock,parent,g=self.make_glow();g.state='REFERENCE_WAIT';g.tick();self.assertEqual(g.state,'SHOWING')
        clock.t+=.3;g.tick();self.assertIsNone(g.on)
        g.presented(clock.t);clock.t+=.41;g.tick();self.assertEqual(g.state,'REMOVING')
        g.presented(clock.t);self.assertEqual(g.state,'CAPTURING');off=g.off
        g.state='REFERENCE_WAIT';clock.t=off+1.99;g.tick();self.assertEqual(g.state,'REFERENCE_WAIT')
        clock.t=off+2.01;g.tick();self.assertEqual(g.state,'SHOWING')
    def test_disable_cancels_queries_and_reenable_can_get_reference(self):
        clock,parent,g=self.make_glow();g.key=('old',);q=ObjectGlowQuery(N(shot_id=1,peak_ts=999.,game_regions=[],camera_regions=[]),lambda:clock.t)
        g.queries=[q];g.disable('off');self.assertTrue(q.final);self.assertEqual(q.result['object_decision']['state'],'UNKNOWN');self.assertIsNone(g.key)
    def test_legacy_camera_hits_are_not_released_by_glow(self):
        clock,parent,g=self.make_glow();parent._glow=g;parent.enabled=True;parent.mode='object_glow'
        out=[];parent.emit=lambda *p:out.append(p)
        self.assertTrue(parent.hold_ordinary_hit(40,50));parent._release_held('invalid');self.assertEqual(out,[])
    def test_render_affects_only_mask_and_snapshot_is_frozen(self):
        import pygame
        clock,parent,g=self.make_glow();g.state='SHOWING';surface=pygame.Surface(SIZE);surface.fill((40,60,80));g.render(surface)
        rgb=pygame.surfarray.array3d(surface).transpose(1,0,2)
        self.assertTrue(np.all(rgb[g._presentation_mask==0]==(40,60,80)))
        halo_only=(g.mask>0)&(g.highlight_mask==0)
        self.assertTrue(halo_only.any())
        self.assertTrue(np.all(rgb[halo_only]>(40,60,80)))
        self.assertTrue(np.all(g._presentation_mask[g.mask>0]>0))
        self.assertTrue(np.all(rgb[g.highlight_mask>0]>(40,60,80)))
    def test_duplicate_query_completion_is_idempotent(self):
        q=ObjectGlowQuery(N(shot_id=1,peak_ts=1.,game_regions=[],camera_regions=[]),lambda:2.)
        q.finish(reason='one');q.finish(reason='two');self.assertEqual(q.result['reason'],'one')
        self.assertEqual(q.compact_record()['shot_id'],1)

class ShrinkAndClockTests(unittest.TestCase):
    def test_short_reference_capture_retries_then_fails_boundedly(self):
        clock,parent,g=LifecycleTests().make_glow()
        parent.cfg=replace(parent.cfg,reference_max_attempts=2)
        g.is_reference=True;g.state='MEASURING'
        g._finish(dict(valid=False,reason='insufficient_causal_plateau',frames=4))
        self.assertEqual(g.state,'REFERENCE_WAIT');self.assertFalse(g.ready)
        self.assertEqual(g.results[-1]['reference_retry']['next_attempt'],2)
        g.state='MEASURING';g.is_reference=True
        g._finish(dict(valid=False,reason='insufficient_causal_plateau',frames=4))
        self.assertEqual(g.state,'FAILED')
        self.assertNotIn('reference_retry',g.results[-1])

    def test_active_subset_uses_last_reference_without_a_second_intro(self):
        mask=glow_mask([region(),region('two',x=140)],SIZE,12)
        cfg=GlowConfig();fc=FlashConfig(edge_margin_px=3)
        ref=measure_glow(frames(10,10.4,mask),10,10.4,None,mask,(0,0,*SIZE),projection,cfg,fc,is_reference=True)
        smaller=glow_mask([region()],SIZE,12)&mask
        post=measure_glow(frames(20,20.18,smaller,[(85,60)]),20,20.18,ref['image'],smaller,(0,0,*SIZE),projection,cfg,fc)
        self.assertTrue(post['valid'],post);self.assertEqual(post['rect'],ref['rect']);self.assertEqual(len(post['holes']),1)
    def test_wall_clock_jump_does_not_bypass_rest_interval(self):
        clocks=N(wall=1000.,mono=10.)
        p=PrecisionFlash(clock=lambda:clocks.wall,monotonic=lambda:clocks.mono,log=lambda *_:None)
        g=ObjectGlow(p);g.enabled=True;g.state='REFERENCE_WAIT';g.last_off_mono=10.;g.context_since=0.
        clocks.wall+=100.;clocks.mono+=.5;g.tick();self.assertEqual(g.state,'REFERENCE_WAIT')
        clocks.mono+=2;g.tick();self.assertEqual(g.state,'SHOWING')

class IntegrationTests(unittest.TestCase):
    def test_registry_to_object_manager_never_uses_dense_or_xy_fallback(self):
        from src.engine import precision_flash as pf
        from src.engine.input.object_hit_v2223 import ObjectHitRegistryV2223,ObjectShotSnapshotV2223
        from src.engine.input.hit_regions import object_hit_regions
        from src.engine.game_objects import ObjectManager,ObjectGeometry,make_breakable_object
        from src.engine.events import event_bus
        manager=ObjectManager();obj=make_breakable_object('one',ObjectGeometry(70,45,30,35));manager.add(obj)
        requested=object_hit_regions(manager.get_hit_regions(),evidence_method='object_glow')
        snap=ObjectShotSnapshotV2223(1,19.8,19.8,(),game_regions=requested,viewport_xywh=(0,0,*SIZE),transform_available=True,object_camera_to_game=lambda x,y:(x,y))
        registry=ObjectHitRegistryV2223();registry._snapshots[1]=snap
        clock=N(t=20.0);parent=PrecisionFlash(clock=lambda:clock.t,monotonic=lambda:clock.t,log=lambda *_:None)
        parent.enabled=True;parent.mode='object_glow';g=ObjectGlow(parent);parent._glow=g;g.enabled=True
        ref,post=evidence([(85,60)]);g.reference=ref['image'];g.camera_valid=ref['camera_valid']
        g.projection=projection;g.viewport=(0,0,*SIZE);g.scene=N();g.on=20.;g.is_reference=False
        camera=N(get_ring_snapshot=lambda:[])
        with patch.object(pf,'_CONTROLLER',parent),patch('src.engine.camera.object_physical_query.ObjectPhysicalQuery',side_effect=AssertionError('dense must not run')):
            registry.begin_object_query(1,N(),camera)
        q=snap.object_query;self.assertFalse(q.final)
        with patch.object(event_bus,'emit') as emit:
            registry.advance_object_queries(N(),camera,now=21.3)
            self.assertFalse(q.final);emit.assert_not_called()  # ordinary deadline already passed
            clock.t=21.4;g._finish(post)
            registry.advance_object_queries(N(),camera,now=clock.t)
            self.assertEqual(emit.call_args.args[0],'physical.object_result')
        result=q.result;self.assertEqual(result['object_decision']['state'],'OBJECT_HIT')
        with patch.object(manager,'_snapshot_for_shot',return_value=snap):
            resolution,h=manager.resolve_object_query(result)
            self.assertTrue(resolution.interactions)
            self.assertEqual(manager.resolve_object_query(result),(None,None))

    def test_co2_follower_can_never_award_a_second_object(self):
        clock=N(t=21.0);parent=PrecisionFlash(clock=lambda:clock.t,monotonic=lambda:clock.t,log=lambda *_:None)
        g=ObjectGlow(parent);g.enabled=True;g.is_reference=False;g.on=20.;g.scene=N();g.projection=projection;g.viewport=(0,0,*SIZE)
        for sid,peak in ((1,19.7),(2,19.8)):
            g.queries.append(ObjectGlowQuery(N(shot_id=sid,peak_ts=peak,game_regions=[region()],camera_regions=[]),lambda:clock.t))
        _,post=evidence([(85,60)]);g._finish(post)
        self.assertEqual(g.queries[0].result['object_decision']['state'],'OBJECT_HIT')
        follower=g.queries[1].result
        self.assertEqual(follower['object_decision']['state'],'UNKNOWN');self.assertFalse(follower['gameplay_owner'])

if __name__=='__main__':unittest.main()
