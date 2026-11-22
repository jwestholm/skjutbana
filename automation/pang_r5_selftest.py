"""Living-room parts and display contracts. Software only, no physical accuracy claim."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ['SKJUTBANA_PHYSICAL_TRACE_DISABLE']='1'
import contextlib,hashlib,io,unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
import pygame
from PIL import Image
from automation.light_room_palette import protect_render,camera_level
from automation.light_room_selftest import part_point
from content.games.light_targets_game import LightTargetsGame
from src.engine.input.hit_regions import object_hit_registry_v2223

class MaterialTests(unittest.TestCase):
    def test_exact_color_lookup_matches_pre_optimization_fixture(self):
        a=np.random.default_rng(71).integers(0,256,(64,71,4),dtype=np.uint8)
        out=np.asarray(protect_render(Image.fromarray(a)))
        self.assertEqual(hashlib.sha256(out.tobytes()).hexdigest(),'4ebd278585bb51eb30ef6e3280411894814051ed83227114fe19182353f3f624')

class LivingRoomTests(unittest.TestCase):
    def setUp(self):
        pygame.init();self.redirect=contextlib.redirect_stdout(io.StringIO());self.redirect.__enter__();self.addCleanup(self.redirect.__exit__,None,None,None)
        p=patch.object(object_hit_registry_v2223,'cancel_object_queries');p.start();self.addCleanup(p.stop)
        self.g=LightTargetsGame('',pygame.Rect(0,0,760,580));self.g.on_enter();self.addCleanup(self.g.on_exit)
    def ready(self,kind):
        for seed in range(50):
            self.g.rng.seed(seed);self.g.start()
            if any(i['kind']==kind for i in self.g.items):break
        else:self.fail('No safe layout for '+kind)
        self.g.phase='live';self.g.ready_wall_ts=0.
    def hit(self,kind,part):
        obj,xy=part_point(self.g,kind,part)
        self.g._on_hit(NS(is_simulated=True,game_x=xy[0],game_y=xy[1],shot_id=None))
        self.assertEqual(self.g.last_outcome,'HIT')
        return obj
    def test_library_and_tv_are_declared(self):
        self.assertEqual(self.g.art.manifest['scene'],'living_room')
        self.assertTrue({'tv','books','plant','clock','mug'}<=set(self.g.art.objects))
        for seed in range(5):
            self.g.rng.seed(seed);self.g.start()
            self.assertEqual(len(self.g.items),8);self.assertTrue(any(i['kind']=='tv' for i in self.g.items))
            for i in self.g.items:
                if i['kind']=='books':self.assertEqual(i['anchor']['support'],'shelf')
    def test_contact_shadows_do_not_add_black_alpha_fringes(self):
        self.ready('tv')
        self.assertTrue(self.g.art._shadows)
        for shadow in self.g.art._shadows.values():
            rgb=pygame.surfarray.array3d(shadow)
            self.assertGreaterEqual(float(camera_level(rgb).min()),120)
    def test_tv_screen_cracks_without_breaking_its_stand(self):
        self.ready('tv');obj=self.hit('tv','screen')
        item=next(i for i in self.g.items if i['entity_id']==obj.entity_id)
        self.assertIn(obj.object_id,self.g.broken_ids);self.assertNotIn(item['parts']['stand'],self.g.broken_ids)
        self.assertEqual(self.g.art.objects['tv']['parts']['screen']['motion'],'crack')
    def test_one_book_falls_without_awarding_the_others(self):
        self.ready('books');obj=self.hit('books','middle')
        item=next(i for i in self.g.items if i['entity_id']==obj.entity_id)
        self.assertEqual(set(item['parts'].values())&self.g.broken_ids,{obj.object_id})
    def test_foliage_hit_preserves_pot(self):
        self.ready('plant');obj=self.hit('plant','foliage')
        item=next(i for i in self.g.items if i['entity_id']==obj.entity_id)
        self.assertNotIn(item['parts']['body'],self.g.broken_ids)
    def test_pot_and_cup_body_detach_their_structural_part(self):
        for kind,child in [('plant','foliage'),('mug','handle')]:
            self.ready(kind);obj=self.hit(kind,'body')
            item=next(i for i in self.g.items if i['entity_id']==obj.entity_id)
            self.assertIn(item['parts'][child],self.g.broken_ids)

if __name__=='__main__':unittest.main()
