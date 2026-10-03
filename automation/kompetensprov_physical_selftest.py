"""Camera-free game contract tests. Synthetic geometry is not physical validation."""
from __future__ import annotations
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import copy, json, tempfile, unittest, importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import pygame, numpy as np, cv2
from src.engine.input.hit_input import HitEvent
from src.engine.input.object_hit_v2223 import ObjectHitRegistryV2223, object_hit_registry_v2223, GameHitRegionV240
from src.engine.range_projection import RangeProjectionGeometry
from src.engine.scenes.game import GameScene
from automation.physical_collection import prepare_kompetensprov_launch, verify_capture_launch, launch_kompetensprov

ROOT=Path(__file__).resolve().parents[1]
GAME_PATH=Path(os.environ.get('KOMPETENSPROV_TEST_GAME_PATH',ROOT/'content/games/kompetensprov_ak/game.py'))
spec=importlib.util.spec_from_file_location('kompetensprov_contract_test_game',GAME_PATH)
import sys
game_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=game_module;spec.loader.exec_module(game_module)

class GameContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  pygame.init();pygame.display.set_mode((900,700))
 @classmethod
 def tearDownClass(cls):pygame.quit()
 def setUp(self):
  self.g=game_module.KompetensprovAKGame(str(ROOT/'assets/games/kompetensprov_ak'),pygame.Rect(37,53,760,580))
  self.g._geometry=lambda:RangeProjectionGeometry(6.3,90.,70.,105.)
  self.g.state='firing';self.g.fire_started=game_module.time.monotonic()-2.;self.g._update_target_rect()
  self.mask=pygame.surfarray.array3d(self.g.zone_mask)[:,:,0]
  self.no_snapshot=patch.object(game_module,'hit_context_snapshot_for_shot',return_value=None,create=True);self.no_snapshot.start();self.addCleanup(self.no_snapshot.stop)
 def point(self,z,rect=None):
  image=(self.mask==z).astype(np.uint8);dist=cv2.distanceTransform(image,cv2.DIST_L2,3);x,y=np.unravel_index(np.argmax(dist),dist.shape)
  r=rect or self.g._target_local_rect;return (r.x+(x+.5)*r.w/self.mask.shape[0],r.y+(y+.5)*r.h/self.mask.shape[1])
 def hit(self,point,sid=19):
  d={k:0. for k in HitEvent.__dataclass_fields__};d.update(source='camera',game_x=point[0],game_y=point[1],camera_x=1734.759787,camera_y=1221.345586,is_simulated=False,timestamp=123.456)
  h=HitEvent(**d);h.shot_id=sid;return h
 def test_all_six_zones_both_distances(self):
  for profile in range(2):
   self.g.profile_index=profile;self.g._update_target_rect()
   for z,letter in game_module.ZONE_ID_TO_LETTER.items():
    with self.subTest(profile=profile,zone=letter):
     h=self.hit(self.point(z),profile*10+z);self.g._on_hit(h);s=self.g.shots[-1]
     self.assertEqual((s.zone,s.points),(letter,game_module.ZONE_POINTS[letter]));self.assertTrue(s.is_hit)
 def test_outside_point_cannot_be_replaced_by_in_target_candidate(self):
  r=self.g._target_local_rect;h=self.hit((r.left-20.,r.centery));before=copy.deepcopy(vars(h))
  assist=SimpleNamespace(hit=True,confidence=1.,candidate_rank=1,camera_x=111.,camera_y=222.)
  with patch.object(object_hit_registry_v2223,'result',return_value=assist),patch.object(object_hit_registry_v2223,'camera_to_game_point',return_value=self.point(2)):
   self.g._on_hit(h)
  s=self.g.shots[-1];self.assertFalse(s.is_hit);self.assertEqual((s.raw_x,s.raw_y),(h.game_x,h.game_y));self.assertEqual(vars(h),before)
 def test_mask_background_inside_aabb_stays_miss(self):
  r=self.g._target_local_rect
  self.g._on_hit(self.hit((r.left+.01*r.w,r.top+.1*r.h)))
  self.assertFalse(self.g.shots[-1].is_hit)
 def test_outside_silhouette_not_expanded_by_bullet_radius(self):
  # A source-mask zero directly adjacent to a nonzero pixel remains outside.
  near=(cv2.dilate((self.mask>0).astype(np.uint8),np.ones((3,3),np.uint8))>0)&(self.mask==0)
  x,y=np.argwhere(near)[0];r=self.g._target_local_rect
  point=(r.x+(x+.5)*r.w/self.mask.shape[0],r.y+(y+.5)*r.h/self.mask.shape[1])
  self.assertEqual(self.g._mask_zone_at(*point),(None,0))
 def test_camera_game_xy_and_shot_id_preserved_no_ballistic_shift(self):
  h=self.hit(self.point(3));before=copy.deepcopy(vars(h));self.g._on_hit(h);s=self.g.shots[-1]
  self.assertEqual(vars(h),before);self.assertEqual(s.shot_id,h.shot_id)
  self.assertEqual((s.camera_x,s.camera_y),(h.camera_x,h.camera_y))
  self.assertEqual((s.raw_x,s.raw_y,s.scored_x,s.scored_y),(h.game_x,h.game_y,h.game_x,h.game_y))
  self.assertEqual(s.ballistic_cm,0.);self.assertEqual(s.detection_mode,'DIRECT')
  self.g._on_hit(h);self.assertEqual(len(self.g.shots),1)
 def test_region_provider_reaches_frozen_camera_context(self):
  scene=GameScene('', '');scene.game=self.g;registry=ObjectHitRegistryV2223()
  with patch.object(registry,'_load_viewport_xywh',return_value=tuple(self.g.viewport)),patch.object(registry,'_camera_transformers',return_value=('fixture',lambda x,y:(2*x+100,3*y+200),None)):
   snap=registry.snapshot(19,123.,scene)
  self.assertEqual(len(snap.game_regions),1);self.assertEqual(len(snap.camera_regions),1)
  region=snap.game_regions[0];self.assertEqual(region.metadata['zone_mask_sha256'],self.g.zone_mask_sha256)
  self.assertEqual(region.rect,tuple(self.g._target_local_rect));self.assertEqual(region.owner,'kompetensprov_ak')
  # Transform must add viewport offset once, through existing engine API.
  self.assertEqual(snap.camera_regions[0].x,2*(region.x+self.g.viewport.x)+100)
  self.assertEqual(snap.camera_regions[0].y,3*(region.y+self.g.viewport.y)+200)
 def test_delayed_hit_uses_frozen_shape_not_current_render_rect(self):
  region=self.g.get_hit_regions()[0];h=self.hit(self.point(2));snapshot=SimpleNamespace(game_regions=(region,))
  self.g._update_target_rect=lambda:None;self.g._target_local_rect.move_ip(500,0)
  with patch.object(game_module,'hit_context_snapshot_for_shot',return_value=snapshot):self.g._on_hit(h)
  self.assertEqual(self.g.shots[-1].zone,'A');self.assertEqual(self.g.shots[-1].target_rect,tuple(region.rect))
  self.assertEqual(self.g.shots[-1].camera_x,h.camera_x)
 def test_translation_scaling_exact_mask_mapping(self):
  for rect in (pygame.Rect(80,90,100,240),pygame.Rect(-10,30,250,600),pygame.Rect(200,10,500,1200)):
   for z,letter in game_module.ZONE_ID_TO_LETTER.items():self.assertEqual(self.g._mask_zone_at(*self.point(z,rect),rect=rect)[0],letter)
 def test_half_open_bounds_and_nonfinite(self):
  r=self.g._target_local_rect
  for point in ((r.right,r.centery),(r.centerx,r.bottom),(float('nan'),1)):
   self.assertEqual(self.g._mask_zone_at(*point),(None,0))
 def test_diagnostic_static_silent_and_operator_ended(self):
  self.g.physical_test=True;self.g.owns_hit_presentation=True;self.g.sounds={'beep':Mock(),'eld':Mock()}
  self.g._start_competition();self.assertEqual(self.g.state,'firing');self.assertEqual(self.g.eld_overlay_until,0)
  self.g.fire_started=game_module.time.monotonic()-100.;self.g.update(0.)
  self.assertEqual(self.g.state,'firing');self.g.sounds['eld'].play.assert_not_called();self.g.sounds['beep'].play.assert_not_called()
  self.g._on_hit(self.hit(self.point(2)));self.assertEqual(len(self.g.shots),1)
  with patch.object(self.g,'_capture_review_snapshot'):
   self.g.handle_event(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_RETURN))
  self.assertEqual(self.g.state,'ceasefire')
 def test_diagnostic_render_unchanged_after_hit(self):
  self.g.physical_test=True;self.g.owns_hit_presentation=True;self.g.eld_overlay_until=0
  screen=pygame.Surface((900,700));self.g.render(screen);before=pygame.surfarray.array3d(screen)
  self.g._on_hit(self.hit(self.point(2)));self.g.render(screen)
  self.assertTrue(np.array_equal(before,pygame.surfarray.array3d(screen)))
  scene=GameScene('', '');scene.game=self.g;self.assertTrue(scene.owns_hit_presentation)
 def test_non_test_mode_retains_45_second_window(self):
  self.g.physical_test=False;self.g.fire_started=game_module.time.monotonic()-46.
  self.g._on_hit(self.hit(self.point(2)));self.assertEqual(self.g.shots,[])
 def test_existing_full_rescue_can_select_outside_region(self):
  import src.engine.shot_region_freshness_v252 as freshness
  from src.engine.camera.hit_scanner import HitScanner
  outside=SimpleNamespace(camera_x=-100.,camera_y=-100.);base=Mock(return_value=outside)
  with patch.object(HitScanner,'_v252_registered_authority_selector',False,create=True),patch.object(HitScanner,'_best_track_for_event',base),patch.object(HitScanner,'_v251_previous_best_track',base,create=True),patch.object(freshness,'_enabled',return_value=True),patch.object(object_hit_registry_v2223,'snapshot_for_shot',return_value=SimpleNamespace(camera_regions=(1,))),patch.object(freshness.rescue_router_v2225,'was_consumed',return_value=True):
   freshness._install_authority_selector_patch()
   self.assertIs(HitScanner._best_track_for_event(SimpleNamespace(),SimpleNamespace(shot_id=19)),outside)
  base.assert_called_once()

class CaptureTests(unittest.TestCase):
 def test_trace_copies_exact_frozen_context_without_selection_effect(self):
  from src.engine.physical_trace import PhysicalTraceRecorder
  from automation.physical_trace_selftest import TraceTests
  context=dict(shot_id=1,peak_ts=2.,game_hit_regions=[dict(object_id='kompetensprov_hf2020',metadata=dict(target_rect=[1,2,3,4],zone_mask_sha256='fixture'))],camera_hit_regions=[])
  with tempfile.TemporaryDirectory() as tmp:
   scanner=TraceTests().scanner();event=SimpleNamespace(shot_id=1,peak_ts=2.,state='pending',emitted=False)
   recorder=PhysicalTraceRecorder(tmp,enabled=True)
   with patch.object(object_hit_registry_v2223,'game_context',return_value=context):recorder.start_from_scanner(scanner,event)
   context['game_hit_regions'][0]['metadata']['target_rect'][0]=999
   self.assertTrue(recorder.flush());trace=json.loads((Path(tmp)/'shots/shot_00000001/trace.json').read_text())
   self.assertEqual(trace['context']['shot_game_context']['game_hit_regions'][0]['metadata']['target_rect'],[1,2,3,4])
   self.assertFalse(trace['outcome']['emitted']);self.assertEqual(event.state,'pending')
  with tempfile.TemporaryDirectory() as tmp:
   recorder=PhysicalTraceRecorder(tmp,enabled=True)
   with patch.object(object_hit_registry_v2223,'game_context',return_value={**context,'peak_ts':1.}):recorder.start_from_scanner(scanner,event)
   self.assertIsNone(recorder._active[1]['trace']['context']['shot_game_context']);self.assertTrue(recorder.flush())
 def test_process_only_fresh_roots_and_startup(self):
  settings=ROOT/'content/ai/settings.json';before=settings.read_bytes();link=os.readlink(ROOT/'content/ai/physical_traces')
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);b1=prepare_kompetensprov_launch(p/'one',p/'traces');b2=prepare_kompetensprov_launch(p/'two',p/'traces')
   first=verify_capture_launch(b1);second=verify_capture_launch(b2)
   self.assertNotEqual(first['trace_root'],second['trace_root']);self.assertEqual(first['planned_session_id'],'KOMPETENSPROV_AK')
   self.assertEqual(len(first['rows']),6);self.assertEqual(first['capture_history']['post_seconds'],5.)
   with patch.dict(os.environ,SKJUTBANA_PHYSICAL_CAPTURE_BINDING=str(b1)):
    exec(compile((ROOT/'main.py').read_text().split('def _restore_terminal')[0],'startup','exec'),{})
    self.assertEqual(os.environ['SKJUTBANA_KOMPETENSPROV_TEST'],'1')
   with self.assertRaises(ValueError):verify_capture_launch(b1)
   with patch('automation.physical_collection._run_capture',return_value=0) as runner:
    launch_kompetensprov(p/'three',p/'traces')
   self.assertEqual(runner.call_args.args[1]['SKJUTBANA_KOMPETENSPROV_TEST'],'1')
  self.assertEqual(settings.read_bytes(),before);self.assertEqual(os.readlink(ROOT/'content/ai/physical_traces'),link)

class AuditDomainTests(unittest.TestCase):
 def test_coordinates_are_recorded_owned_and_causal(self):
  from automation.d02_pre_retention_onset import candidate_coordinates
  pipeline=dict(source_frame_ts=8.,upstream=dict(crop_origin_camera=(10,20),pipeline=dict(pre_limit_candidates=[dict(camera_x=2,camera_y=3)])),cleanup_stages={},retained_candidates=[])
  trace=dict(shot_id=1,stages=[dict(timestamp=9.,candidate_pool_shot_id=1,pipeline=pipeline),dict(timestamp=20.,candidate_pool_shot_id=1,pipeline={**pipeline,'source_frame_ts':19.})],decision_input=dict(complete_track_audit=dict(associations=[],tracks=[])))
  before=copy.deepcopy(trace);self.assertEqual(set(candidate_coordinates(trace,10.)),{(12.,23.)});self.assertEqual(trace,before)
  trace['stages'][0]['candidate_pool_shot_id']=2;self.assertEqual(candidate_coordinates(trace,10.),{})

if __name__=='__main__':unittest.main()
