"""Room variation, local display safety and persisted placement contracts; no hardware."""
import os,contextlib,io,json,tempfile,hashlib,unittest
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ['SKJUTBANA_PHYSICAL_TRACE_DISABLE']='1';os.environ['SKJUTBANA_SOURCE_RECORD']='0'
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pygame
from content.games.light_room_safety import local_background,camera_proxy,TARGET_MARGIN
from content.games.light_room_art import ASSET_ROOT
from content.games.light_targets_game import LightTargetsGame
from src.engine.game_objects import ObjectGeometry

class SafetyTests(unittest.TestCase):
    def test_only_target_neighbourhood_is_changed(self):
        a=np.random.default_rng(17).integers(20,180,(220,180,3),dtype=np.uint8)
        source=pygame.surfarray.make_surface(a);items=[dict(geometry=ObjectGeometry(60,65,35,40))]
        result,weight=local_background(source,items);rgb=pygame.surfarray.array3d(result)
        self.assertTrue(np.array_equal(a,pygame.surfarray.array3d(source)))
        self.assertTrue(np.array_equal(rgb[weight==0],a[weight==0]))
        self.assertGreaterEqual(float(camera_proxy(rgb[weight==1]).min()),124)
        self.assertLess(float((weight>0).mean()),.45)

class RoomTests(unittest.TestCase):
    def setUp(self):
        pygame.font.init();self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        p=patch.dict(os.environ,SKJUTBANA_PANG_ROOM_HISTORY=str(Path(self.t.name)/'history.json'));p.start();self.addCleanup(p.stop)
        self.stdout=contextlib.redirect_stdout(io.StringIO());self.stdout.__enter__();self.addCleanup(self.stdout.__exit__,None,None,None)
    def game(self,size=(760,580)):
        g=LightTargetsGame('',pygame.Rect(0,0,*size));g.on_enter();self.addCleanup(g.on_exit);return g
    def test_six_distinct_natural_room_plates(self):
        m=json.loads((ASSET_ROOT/'manifest.json').read_text());rooms=m['rooms']
        self.assertEqual(len(rooms),6);self.assertEqual({r['template'] for r in rooms},{'living','kitchen','study'})
        self.assertEqual(len({r['raw_sha256']['room.png'] for r in rooms}),6)
        for r in rooms:
            self.assertEqual(hashlib.sha256((ASSET_ROOT/r['image']).read_bytes()).hexdigest(),r['raw_sha256']['room.png'])
    def test_templates_and_positions_survive_restart(self):
        g=self.game();g.rng.seed(2);g.start();first=g.art.room_id;before=[i['geometry'].center for i in g.items]
        other=self.game();other.rng.seed(2);other.start()
        self.assertNotEqual(first.rsplit('_',1)[0],other.art.room_id.rsplit('_',1)[0])
        self.assertGreaterEqual(len(other._recent_positions),20)
        data=json.loads(other._history_path.read_text());self.assertEqual(data['rooms'],[first,other.art.room_id])
    def test_all_templates_fit_supported_viewports_and_local_floor(self):
        for size in ((640,480),(760,580),(1280,720)):
            g=self.game(size)
            for room in list(g.art.manifest['rooms']):
                g.art.rooms=[room];g.rng.seed(13);g.start()
                self.assertNotEqual(g.phase,'unsupported',(size,room['id']))
                self.assertEqual(len(g.items),10)
                tv=next(i for i in g.items if i['kind']=='tv');self.assertEqual(tv['anchor']['support'],'tv_bench')
                self.assertGreater(tv['geometry'].height,60)
                self.assertGreaterEqual(float(camera_proxy(pygame.surfarray.array3d(g.art.background)[g.art.safety_weight==1]).min()),124)
                out=pygame.Surface(size);g.art.render(out,g.items);rgb=pygame.surfarray.array3d(out)
                self.assertGreaterEqual(float(camera_proxy(rgb[g.art.safety_weight==1]).min()),120,(size,room['id']))
                self.assertTrue(np.array_equal(pygame.surfarray.array3d(g.art.background)[g.art.safety_weight==0],
                                               pygame.surfarray.array3d(g.art._plate)[g.art.safety_weight==0]))
                for item in g.items:
                    if item['kind']=='bowl':self.assertLess(item['geometry'].height,tv['geometry'].height*.8)
    def test_room_plate_is_not_updated_during_shot_render(self):
        g=self.game();g.start();before=pygame.image.tostring(g.art.background,'RGB')
        surface=pygame.Surface(g.viewport.size)
        for frame in range(3):g.art.render(surface,g.items,frame=frame)
        self.assertEqual(before,pygame.image.tostring(g.art.background,'RGB'))

if __name__=='__main__':unittest.main()
