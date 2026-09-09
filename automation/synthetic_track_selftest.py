"""Offline image funnel and timing episodes; never opens camera/projector."""
import contextlib, io, unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import cv2,numpy as np
from automation.synthetic_track_research import prepare,execute
from src.engine.synthetic.synthetic_hole_overlay import SyntheticHoleOverlay
from src.engine.shot_fast_v2225 import LocalConfirmManagerV2225,local_confirm_candidates_v2225
from src.engine.camera.hit_scanner import HitScanner,AudioShotEvent
from src.engine.shot_track_v2226 import update_tracks_frame_unique_v2226 as update
from src.engine.track_audit import capture
from src.engine.offline.track_replay import current_exact_replay

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  with contextlib.redirect_stdout(io.StringIO()):prepare()
  cls.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='shot-cv-v2224')
 @classmethod
 def tearDownClass(cls):cls.worker.shutdown()
 def image(self,holes=()):
  base=np.full((160,220),190,np.uint8);o=SyntheticHoleOverlay(220,160,rng_seed=31)
  for i,p in enumerate(holes):o.add_hole(*p,radius_px=3,hole_id=str(i))
  return cv2.cvtColor(o.composite_on(cv2.cvtColor(base,cv2.COLOR_GRAY2BGR)),cv2.COLOR_BGR2GRAY)
 def run_shot(self,pre,post,sid,**kw):
  with contextlib.redirect_stdout(io.StringIO()):return execute(pre,post,sid,self.worker,**kw)
 def test_unchanged_scene_no_confirmation(self):
  pre=self.image();r=self.run_shot(pre,pre,1)
  self.assertFalse(r['confirmed']);self.assertFalse(r['ready']);current_exact_replay(r['snapshot'])
 def test_crop_full_frame_new_hole_and_tracing_parity(self):
  pre=self.image();post=self.image([(100,80)]);coords=[]
  for crop,tracing in ((False,True),(True,True),(True,False)):
   r=self.run_shot(pre,post,2,crop=crop,tracing=tracing)
   self.assertTrue(r['ready']);coords.append((r['selected'].camera_x,r['selected'].camera_y))
   self.assertLess(np.hypot(coords[-1][0]-100,coords[-1][1]-80),3)
  np.testing.assert_allclose(coords[0],coords[1],atol=1e-5);np.testing.assert_allclose(coords[1],coords[2],atol=1e-5)
 def test_two_sequential_nearby_impacts_not_blacklisted(self):
  base=self.image();first=self.image([(100,80)]);second=self.image([(100,80),(108,80)])
  r=self.run_shot(base,first,3);s=r['scanner'];e=r['event']
  from src.engine.input.hit_input import hit_input
  with patch.object(hit_input,'push_camera_hit'):
   s._emit_track_result(r['selected'],e)
  # Model actual idle negative frames so old tracks age out naturally.
  for i in range(30):update(s,[],1012.7+i*.04)
  r2=self.run_shot(first,second,4,scanner=s)
  self.assertTrue(any(np.hypot(c['camera_x']-108,c['camera_y']-80)<5 for c in r2['proposal']))
  self.assertTrue(any(np.hypot(c['camera_x']-108,c['camera_y']-80)<5 for c in r2['confirmed']))
 def test_delayed_processing_keeps_frame_identity(self):
  r=self.run_shot(self.image(),self.image([(100,80)]),5,delivery_delay=1.2)
  self.assertTrue(r['ready']);self.assertAlmostEqual(r['selected'].first_seen_ts,1020.08)
  self.assertAlmostEqual(r['selected'].last_seen_ts,1021.22)
 def test_false_audio_boundary_between_impacts(self):
  for sid,peak,boundary in ((6,1788885626.4964387,1788885627.9803684),(14,1788885752.6620774,1788885754.1525483)):
   s=HitScanner();e1=AudioShotEvent(sid,peak,peak);false=AudioShotEvent(sid+1,boundary,boundary);e2=AudioShotEvent(sid+2,boundary+3,boundary+3)
   s.audio_events.extend([e1,false]);m=LocalConfirmManagerV2225();c={'camera_x':100,'camera_y':80,'score':10,'timestamp':peak+.1}
   m.start(sid,peak+.1,[c],self.image())
   self.assertIsNotNone(m.active_waiting(s,peak+.3));self.assertIsNone(m.active_waiting(s,boundary+.01))
   e1.state='missed';false.state='missed';s.audio_events.append(e2)
   m.start(sid+2,e2.peak_ts+.1,[c],self.image([(100,80)]))
   self.assertEqual(m.active_waiting(s,e2.peak_ts+.4).shot_id,sid+2)
 def test_zero_darkening_can_pass_local_confirmation(self):
  pre=np.full((80,80),100,np.uint8);post=pre.copy();post[36:43,36:43]=150
  c={'camera_x':39.,'camera_y':39.,'score':10.}
  confirmed,_=local_confirm_candidates_v2225(pre,post,[c],frame_ts=2)
  self.assertEqual(len(confirmed),1);self.assertEqual(confirmed[0]['v2225_confirm_darkening'],0)

 def test_saved_pool_ablation_distinguishes_readiness_from_eligibility(self):
  from automation.track_survival_replay import choice_record
  s=HitScanner();s.physical_trace_capture_enabled=True;s.last_trace_pipeline_shot_id=1
  e=AudioShotEvent(1,100,100);s.audio_events.append(e)
  c={'camera_x':100.,'camera_y':80.,'score':10.,'detector_v1':1}
  update(s,[c],100.08);chosen=s._best_track_for_event(e)
  a=capture(s,e,chosen);r=choice_record(a,'EXCLUDE_FAST',c,100.6)
  self.assertEqual(r['status'],'NOT_READY_AT_RECORDED_DECISION');self.assertIsNone(r['error'])
  update(s,[c],100.6);chosen=s._best_track_for_event(e)
  r=choice_record(capture(s,e,chosen),'EXCLUDE_FAST',c,100.6)
  self.assertTrue(r['ready']);self.assertEqual(r['error'],0)

if __name__=='__main__':unittest.main()
