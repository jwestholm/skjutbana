"""Per-game highlight opt-in, original hues, rings and restoration; no hardware."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ['SKJUTBANA_SOURCE_RECORD'] = '0'
import colorsys
import unittest
from types import SimpleNamespace as N
from unittest.mock import patch
import numpy as np
import pygame
from src.engine.precision_flash import PrecisionFlash, resolve_policy
from src.engine.precision_object_glow import (
    ObjectGlow, GlowConfig, glow_mask, object_highlight_mask, measurement_circles, pastel_highlight,
)


def region(oid, x=60, entity=None):
    return N(enabled=True, object_id=oid, metadata={'v250_object_snapshot': {
        'entity_id':entity or oid,
        'shape':dict(kind='rect', x=x, y=45, width=30, height=35),
    }})


class FlagTests(unittest.TestCase):
    def test_explicit_boolean_opt_in_and_default_off(self):
        scene = N(hit_precision='OBJECT')
        self.assertFalse(resolve_policy(scene, env={})[0])
        scene.object_highlight = True
        self.assertEqual(resolve_policy(scene, env={}), (True, 'OBJECT', 'object_glow'))
        scene.object_highlight = 'false'
        self.assertFalse(resolve_policy(scene, env={})[0])

    def test_explicit_scene_or_environment_off_wins(self):
        scene = N(hit_precision='OBJECT', object_highlight=True, precision_flash='off')
        self.assertFalse(resolve_policy(scene, env={'SKJUTBANA_PRECISION_FLASH':'on'})[0])
        scene.precision_flash = 'auto'
        self.assertFalse(resolve_policy(scene, env={'SKJUTBANA_PRECISION_FLASH':'off'})[0])

    def test_game_and_overlay_forward_the_flag(self):
        from src.engine.scenes.game import GameScene
        from src.engine.scenes.overlay_scene import OverlayScene
        scene = object.__new__(GameScene)
        scene.game = N(hit_precision='OBJECT', object_highlight=True)
        overlay = OverlayScene(scene)
        self.assertEqual(resolve_policy(overlay, env={}), (True, 'OBJECT', 'object_glow'))
        scene.game.object_highlight = False
        self.assertFalse(resolve_policy(overlay, env={})[0])

    def test_pang_boolean_flag_and_legacy_alias_stay_consistent(self):
        from pathlib import Path
        from content.games.light_targets_game import LightTargetsGame
        with patch.dict(os.environ, {'SKJUTBANA_PANG_OBJECT_HIGHLIGHT':'0', 'SKJUTBANA_PANG_OBJECT_GLOW':'1'}):
            game = LightTargetsGame(Path('content/games'), pygame.Rect(0,0,800,600))
        self.assertFalse(game.object_highlight)
        self.assertEqual(game.precision_glow['reference_frames'],3)
        self.assertEqual(GlowConfig.from_scene(game).measurement_s,.80)
        game.object_highlight = True
        self.assertEqual(game.precision_flash, 'object_glow')
        self.assertTrue(resolve_policy(game, env={})[0])
        game.object_highlight = False
        self.assertEqual(game.precision_flash, 'off')
        with self.assertRaises(ValueError):
            game.object_highlight = 'false'

    def test_g_cannot_disable_or_replace_an_active_settling_round(self):
        from pathlib import Path
        from content.games.light_targets_game import LightTargetsGame
        with patch.dict(os.environ, {'SKJUTBANA_PANG_OBJECT_HIGHLIGHT':'1'}):
            game = LightTargetsGame(Path('content/games'), pygame.Rect(0,0,800,600))
        game.phase='settling'
        game.layout_id=7
        with patch.object(game, '_next_target') as next_target:
            game.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g))
        self.assertTrue(game.object_highlight)
        self.assertEqual(game.layout_id,7)
        next_target.assert_not_called()


class PaletteTests(unittest.TestCase):
    def test_red_green_blue_and_wood_keep_their_hues(self):
        rgb = np.array([[(140,20,20),(20,140,20),(20,20,140),(170,90,30),(240,240,240)]], np.uint8)
        mask = np.zeros(rgb.shape[:2],np.uint8);mask[:,:4]=1
        out, _ = pastel_highlight(rgb, mask, [])
        self.assertEqual(out[0,0].tolist(), [174,90,90])
        self.assertTrue(np.all(out >= rgb))
        self.assertEqual(out[0,-1].tolist(), rgb[0,-1].tolist())
        for before, after in zip(rgb[0,:4], out[0,:4]):
            h0 = colorsys.rgb_to_hsv(*(before/255.))[0]
            h1 = colorsys.rgb_to_hsv(*(after/255.))[0]
            error = min(abs(h1-h0), 1-abs(h1-h0))
            self.assertLess(error, .005)

    def test_each_entity_gets_its_own_thin_yellow_ring(self):
        regions = [region('body'),region('handle',x=85,entity='body'),region('other',x=170)]
        circles = measurement_circles(regions,12)
        self.assertEqual(len(circles),2)
        mask = glow_mask(regions,(260,160),12)
        rgb = np.full((*mask.shape,3),(140,20,20),np.uint8)
        out, overlay = pastel_highlight(rgb,mask,circles,ring_width=1)
        yellow = np.all(out == (230,220,150),axis=2)
        self.assertLess(int(yellow.sum()), int(mask.sum())*.25)
        for (cx,cy), radius in circles:
            self.assertTrue(yellow[cy,cx+radius-1])
            self.assertFalse(yellow[cy,cx])
        self.assertTrue(np.all(overlay[mask==0] <= 1))
        np.testing.assert_array_equal(out[mask==0],rgb[mask==0])

    def test_invalid_palette_input_is_rejected(self):
        mask = np.ones((3,4),np.uint8)
        with self.assertRaises(ValueError):
            pastel_highlight(np.zeros((3,4,3),np.float32),mask,[])
        with self.assertRaises(ValueError):
            pastel_highlight(np.zeros((3,4,3),np.uint8),mask,[],mix=.6)


class RenderTests(unittest.TestCase):
    def make(self):
        regions = [region('one'),region('two',x=170)]
        parent = PrecisionFlash(clock=lambda:100.,monotonic=lambda:100.,log=lambda *_:None)
        glow = ObjectGlow(parent)
        glow.enabled = True
        glow.mask = glow_mask(regions,(260,160),12)
        glow.highlight_mask = object_highlight_mask(regions,(260,160))
        glow.circles = measurement_circles(regions,0)
        glow.viewport = (17,11,260,160)
        glow.state = 'SHOWING'
        return glow

    def test_reference_and_post_reuse_identical_palette_after_scene_damage(self):
        glow = self.make()
        screen = pygame.Surface((300,190))
        screen.fill((140,20,20))
        glow.render(screen)
        first = pygame.surfarray.array3d(screen).copy()
        fingerprint = glow.highlight_rgb_sha256
        self.assertEqual(tuple(first[92,71]),(174,90,90))
        screen.fill((20,140,20))
        glow.render(screen)
        after = pygame.surfarray.array3d(screen)
        scope = glow.highlight_mask.T > 0
        np.testing.assert_array_equal(first[17:277,11:171][scope],after[17:277,11:171][scope])
        self.assertEqual(tuple(after[0,0]),(20,140,20))
        self.assertEqual(glow.highlight_rgb_sha256,fingerprint)

    def test_live_context_populates_rings_and_damage_preserves_remaining_pixels(self):
        regions = [region('one'),region('two',x=170)]
        context = dict(key='lane',viewport=(17,11,260,160),regions=regions,active_regions=regions)
        scene = N(get_precision_glow_context=lambda:context)
        parent = PrecisionFlash(clock=lambda:100.,monotonic=lambda:100.,log=lambda *_:None)
        glow = ObjectGlow(parent)
        with patch('src.engine.precision_object_glow.freeze_projection',return_value=(lambda p:np.asarray(p),'fixture')):
            glow.sync(scene)
            self.assertEqual(len(glow.circles),2)
            screen = pygame.Surface((300,190));screen.fill((140,20,20))
            glow.state='SHOWING';glow.render(screen)
            before = pygame.surfarray.array3d(screen).copy();old_mask=glow.mask.copy()
            context['active_regions'] = [regions[0]]
            glow.state='IDLE';glow.sync(scene)
            self.assertFalse(np.any((glow.mask>0)&(old_mask==0)))
            self.assertLess(glow.mask.sum(),old_mask.sum())
            screen.fill((20,140,20));glow.state='SHOWING';glow.render(screen)
            after=pygame.surfarray.array3d(screen);scope=glow.mask.T>0
            np.testing.assert_array_equal(before[17:277,11:171][glow.highlight_mask.T>0],after[17:277,11:171][glow.highlight_mask.T>0])
            context['key']='next_lane';context['regions']=[regions[0]]
            glow.sync(scene);screen.fill((20,140,20));glow.state='SHOWING';glow.render(screen)
            self.assertEqual(tuple(pygame.surfarray.array3d(screen)[92,71]),(90,174,90))

    def test_unrenderable_viewport_abstains_without_changing_screen(self):
        glow = self.make();glow.viewport=(200,100,260,160);glow.scene=N()
        screen=pygame.Surface((300,190));screen.fill((140,20,20))
        glow.render(screen)
        self.assertEqual(glow.state,'FAILED');self.assertFalse(glow.ready)
        self.assertTrue(np.all(pygame.surfarray.array3d(screen)==(140,20,20)))

    def test_normal_colors_return_before_decision_computation_finishes(self):
        glow = self.make()
        screen = pygame.Surface((300,190))
        screen.fill((140,20,20))
        glow.render(screen)
        for state in ('REMOVING','CAPTURING','MEASURING','IDLE'):
            glow.state = state
            screen.fill((140,20,20))
            glow.render(screen)
            self.assertTrue(np.all(pygame.surfarray.array3d(screen)==(140,20,20)),state)


if __name__ == '__main__':
    unittest.main()
