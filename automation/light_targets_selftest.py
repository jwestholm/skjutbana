"""Ljusjakten software contracts. Generated holes do not establish physical accuracy."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ['SKJUTBANA_PHYSICAL_TRACE_DISABLE']='1'
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch,Mock
import numpy as np
import pygame
from content.games.light_targets_game import LightTargetsGame,BACKGROUND,PALETTE
from src.engine.input.hit_regions import object_hit_regions
from src.engine.input.object_hit_v2223 import object_hit_registry_v2223,CameraHitRegionV240
from src.engine.game_objects import ObjectGeometry,make_breakable_object
from src.engine.camera.object_physical_query import ObjectPhysicalQuery
from automation import object_physical_query_selftest as fx

class GameTests(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.cancel=patch.object(object_hit_registry_v2223,'cancel_object_queries');self.cancel.start();self.addCleanup(self.cancel.stop)
        self.g=LightTargetsGame('',pygame.Rect(20,10,760,580));self.g.on_enter();self.addCleanup(self.g.on_exit)
        self.screen=pygame.Surface((900,700))
    def live(self):
        self.g.start();self.g.render(self.screen);self.g.on_frame_presented(10.)
        self.g.presented_mono-=2;self.g.update(0);self.assertEqual(self.g.phase,'live')
    def test_static_pending_and_presented_settle(self):
        self.g.start();self.g.phase_at-=10;self.g.update(0);self.assertEqual(self.g.phase,'settling')
        self.g.render(self.screen);self.g.on_frame_presented(10);self.g.update(0);self.assertEqual(self.g.phase,'settling')
        self.g.presented_mono-=2;self.g.update(0);self.g.render(self.screen)
        before=pygame.surfarray.array3d(self.screen).copy()
        self.g.on_object_shot_frozen(NS(shot_id=7,peak_ts=11))
        self.g.update(0);self.g.render(self.screen)
        self.assertTrue(np.array_equal(before,pygame.surfarray.array3d(self.screen)))
        self.assertEqual(self.g.active_shot,7)
        self.g.on_object_shot_frozen(NS(shot_id=8,peak_ts=11.08));self.assertEqual(self.g.active_shot,7)
    def test_spread_brightness_regions_and_edge(self):
        centers=[]
        for _ in range(12):
            self.g._next_target();g=self.g.target;centers.append(g.center)
            self.assertGreater(g.x,60);self.assertGreater(g.y,70)
            self.assertLess(g.right,760-60);self.assertLess(g.bottom,580-30)
        self.assertEqual(len(set(centers)),12)
        self.assertGreaterEqual(BACKGROUND[1],190)
        self.assertTrue(all(c[1]>=190 for c in PALETTE))
        self.live();r=self.g.get_hit_regions()[0]
        self.assertEqual(r.metadata['physical_query']['evidence_method'],'dense')
        self.assertTrue(r.metadata['physical_query']['surface_after_ts'])
        self.assertGreater(r.metadata['v250_object_snapshot']['shape']['touch_radius'],0)
    def test_owned_results_and_duplicate_unknown(self):
        self.live();regions=self.g.get_hit_regions();self.g.on_object_shot_frozen(NS(shot_id=42,peak_ts=12))
        snap=NS(game_regions=regions,object_gameplay={})
        result=dict(final=True,shot_id=42,object_decision=dict(state='UNKNOWN'),reason='uncertain')
        with patch('content.games.light_targets_game.hit_context_snapshot_for_shot',return_value=snap):
            self.g._on_object_event(dict(event='physical.object_result',data=dict(result,shot_id=43)))
            self.assertEqual(self.g.used[0],0)
            self.g._on_object_event(dict(event='physical.object_result',data=result))
            self.g._on_object_event(dict(event='physical.object_result',data=result))
        self.assertEqual(self.g.used[0],1);self.assertEqual(self.g.scores[0],0)
        self.assertIn('OSÄKERT',self.g.feedback)
    def test_hit_scores_once_physical_xy_never_scores(self):
        self.live();self.g._on_hit(NS(is_simulated=False,game_x=10,game_y=10));self.assertEqual(self.g.used[0],0)
        regions=self.g.get_hit_regions();self.g.on_object_shot_frozen(NS(shot_id=1,peak_ts=1))
        snap=NS(game_regions=regions,object_gameplay={})
        with patch('content.games.light_targets_game.hit_context_snapshot_for_shot',return_value=snap),patch.object(self.g.manager,'resolve_object_query',return_value=(NS(interactions=[1]),dict(camera_xy=[100,100]))):
            msg=dict(event='physical.object_result',data=dict(final=True,shot_id=1,object_decision=dict(state='OBJECT_HIT')))
            self.g._on_object_event(msg);self.g._on_object_event(msg)
        self.assertEqual(self.g.scores[0],100);self.assertEqual(self.g.used[0],1)
    def test_stale_round_and_two_player_finish(self):
        self.live();self.g.on_object_shot_frozen(NS(shot_id=10,peak_ts=1))
        with patch('content.games.light_targets_game.hit_context_snapshot_for_shot',return_value=NS(game_regions=(),object_gameplay={})):
            self.g._on_object_event(dict(event='physical.object_result',data=dict(final=True,shot_id=10)))
        self.assertEqual(self.g.used[0],0)
        self.g.players=2;self.g.shots_per_player=5
        for _ in range(5):self.g._finish('UNKNOWN',None,None,None,simulated=True);self.g.phase_at-=2;self.g.update(0)
        self.assertEqual(self.g.player,1)
        for _ in range(5):self.g._finish('MISS',None,None,None,simulated=True);self.g.phase_at-=2;self.g.update(0)
        self.assertEqual(self.g.phase,'finished')
    def test_menu_api_factory_and_handler(self):
        from src.engine.scene_factory import game_scene_from_menu_id
        scene=game_scene_from_menu_id('light_targets');self.assertTrue(scene.inner.script_path.endswith('light_targets_game.py'))
        with self.assertRaises(ValueError):game_scene_from_menu_id('../../etc/passwd')
        from src.engine.app import App
        app=NS(_switch_to=Mock());cmd=NS(args={'id':'light_targets'},reply_success=Mock(),reply_error=Mock())
        App._handle_start_game(app,cmd);cmd.reply_success.assert_called_once();cmd.reply_error.assert_not_called()
        cmd.args={'id':'not_a_game'};App._handle_start_game(app,cmd);cmd.reply_error.assert_called_once()

class DenseOptInTests(unittest.TestCase):
    def test_per_query_dense_without_global_flag(self):
        with patch.dict(os.environ,{'SKJUTBANA_OBJECT_DENSE':'0'}):
            from src.engine.game_objects import ObjectManager
            manager=ObjectManager();manager.add(make_breakable_object('lamp',ObjectGeometry(100,60,80,80)))
            game=object_hit_regions(manager.get_hit_regions(),evidence_method='dense')
            snap=fx.snapshot([CameraHitRegionV240('lamp',100,60,80,80)],game)
            pre=np.full((240,480),200,np.uint8)
            q=ObjectPhysicalQuery(snap,[fx.frame(pre,t) for t in (1.87,1.90,1.94)],camera_to_game=lambda x,y:(x,y))
            self.assertTrue(q.dense_enabled);self.assertEqual(q.deadline_s,1.2)
            post=pre.copy();post[96:103,136:143]=30
            for t in (2.36,2.40,2.44):r=q.observe(fx.frame(post,t),delivered_ts=t+.001)
            self.assertTrue(r['final']);self.assertEqual(r['object_decision']['state'],'OBJECT_HIT')
            plain=fx.snapshot([CameraHitRegionV240('lamp',100,60,80,80)],object_hit_regions(manager.get_hit_regions()))
            legacy=ObjectPhysicalQuery(plain,[fx.frame(pre,t) for t in (1.87,1.90,1.94)],camera_to_game=lambda x,y:(x,y))
            self.assertFalse(legacy.dense_enabled)
    def test_bad_method_refused(self):
        with self.assertRaises(ValueError):object_hit_regions([],evidence_method='magic')

class ScoringTests(unittest.TestCase):
    def test_absent_unknown_and_false_award_are_not_success(self):
        from automation.light_targets_score import score
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);log=p/'log';truth=p/'truth'
            rows=[dict(owner='a',shot_id=1,source='OBJECT',outcome='HIT'),dict(owner='a',shot_id=2,source='OBJECT',outcome='UNKNOWN'),dict(owner='a',shot_id=3,source='OBJECT',outcome='HIT')]
            log.write_text('\n'.join('[LAMP-RESULT] '+json.dumps(r) for r in rows))
            truth.write_text('\n'.join(json.dumps(dict(owner='a',shot_id=i,expected=e)) for i,e in [(1,'HIT'),(2,'HIT'),(3,'MISS'),(4,'HIT')]))
            with truth.open('a') as f: f.write('\n'+json.dumps(dict(owner='a',shot_id=None,attempt_id='unheard',expected='HIT')))
            r=score(log,truth);self.assertEqual(r['cases'],5);self.assertEqual(r['correct'],1)
            self.assertEqual(r['false_awards'],1);self.assertEqual(r['abstentions'],3);self.assertFalse(r['point_target_met'])

if __name__=='__main__':unittest.main()
