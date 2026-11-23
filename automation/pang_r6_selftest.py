"""CC0/PBR provenance and exact imported silhouettes; software-only contracts."""
import hashlib,json,unittest,os,contextlib,io
from unittest.mock import patch
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ['SKJUTBANA_PHYSICAL_TRACE_DISABLE']='1'
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from automation.build_light_room_parts import polygons
from automation.light_room_palette import expose_pbr
from src.engine.game_objects import HitShapeSpec,ObjectGeometry
from src.engine.game_objects.geometry import shape_contains
ROOT=Path('content/games/light_room_parts_assets')
SOURCES=Path('content/games/pang_polyhaven_sources')


def snapshot(parts,w,h):
    shapes=[HitShapeSpec.polygon(p) for p in parts]
    return (shapes[0] if len(shapes)==1 else HitShapeSpec.union(*shapes)).snapshot(ObjectGeometry(0,0,w,h))

class ExactTopologyTests(unittest.TestCase):
    def test_display_exposure_preserves_alpha_and_channel_order(self):
        a=np.random.default_rng(23).integers(0,256,(20,30,4),dtype=np.uint8);copy=a.copy()
        b=np.asarray(expose_pbr(Image.fromarray(a)))
        self.assertTrue(np.array_equal(a,copy));self.assertTrue(np.array_equal(a[:,:,3],b[:,:,3]))
        self.assertTrue(np.all(b[:,:,:3]>=a[:,:,:3]))
        order=np.argsort(a[:,:,:3],axis=2)
        ordered=np.take_along_axis(b[:,:,:3].astype(int),order,axis=2)
        self.assertTrue(np.all(np.diff(ordered,axis=2)>=0))
    def test_imported_holes_are_preserved_in_both_raster_and_engine(self):
        mask=np.zeros((80,100),np.uint8);cv2.rectangle(mask,(5,5),(94,74),1,-1)
        cv2.circle(mask,(30,40),9,0,-1);cv2.circle(mask,(70,40),8,0,-1)
        with self.assertRaises(ValueError):polygons(mask)
        clean,p=polygons(mask,partition_holes=True);s=snapshot(p,100,80)
        self.assertTrue(np.array_equal(clean>0,mask>0))
        for y in range(80):
            for x in range(100):self.assertEqual(shape_contains(s,x,y),bool(mask[y,x]),(x,y))
    def test_imported_contours_never_have_zero_length_edges(self):
        m=json.loads((ROOT/'manifest.json').read_text())
        for spec in m['objects'].values():
            for part in spec['parts'].values():
                for poly in part['polygons']:
                    self.assertGreaterEqual(len(poly),3)
                    for a,b in zip(poly,poly[1:]+poly[:1]):self.assertNotEqual(a,b)
    def test_real_openings_do_not_become_solid_hit_regions(self):
        m=json.loads((ROOT/'manifest.json').read_text());tested=0
        for kind,spec in m['objects'].items():
            for name,part in spec['parts'].items():
                mask=(np.asarray(Image.open(ROOT/part['mask']))>0).astype(np.uint8)
                cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                full=np.zeros_like(mask);cv2.drawContours(full,cs,-1,1,-1)
                n,cc,_,_=cv2.connectedComponentsWithStats(((full>0)&(mask==0)).astype(np.uint8))
                s=snapshot(part['polygons'],mask.shape[1],mask.shape[0])
                for index in range(1,n):
                    _,distance,_,point=cv2.minMaxLoc(cv2.distanceTransform((cc==index).astype(np.uint8),cv2.DIST_L2,5))
                    if distance>=1:
                        self.assertFalse(shape_contains(s,*point),(kind,name,point));tested+=1
        self.assertGreater(tested,0,'imported PBR models must exercise real openings')

class ProvenanceTests(unittest.TestCase):
    def test_scanned_textures_are_not_mirrored_when_hit_from_left(self):
        import pygame
        from content.games.light_targets_game import LightTargetsGame
        pygame.font.init()
        with contextlib.redirect_stdout(io.StringIO()):
            g=LightTargetsGame('',pygame.Rect(0,0,760,580));g.on_enter()
            try:
                g.rng.seed(11);g.start();item=next(i for i in g.items if i['kind']=='vase')
                oid=item['parts']['neck'];surface=pygame.Surface((760,580))
                with patch('pygame.transform.flip',side_effect=AssertionError('PBR photo texture must not be mirrored')):
                    g.art.render(surface,g.items,{oid},{oid},11,{oid:-1})
            finally:g.on_exit()

    def test_only_selected_original_dependencies_are_shipped_and_hashed(self):
        data=json.loads((SOURCES/'manifest.json').read_text());self.assertEqual(len(data['assets']),15)
        self.assertEqual({p.name for p in SOURCES.iterdir() if p.is_dir()},{a['id'] for a in data['assets']})
        for asset in data['assets']:
            self.assertEqual(asset['license'],'CC0-1.0')
            self.assertEqual(asset['url'],'https://polyhaven.com/a/'+asset['id'])
            for name,digest in asset['files_sha256'].items():
                path=SOURCES/asset['id']/name
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
    def test_original_gltf_pbr_maps_are_available(self):
        for asset in json.loads((SOURCES/'manifest.json').read_text())['assets']:
            root=SOURCES/asset['id'];g=json.loads(next(root.glob('*.gltf')).read_text())
            self.assertTrue(any('normalTexture' in m for m in g['materials']),asset['id'])
            self.assertTrue(any('metallicRoughnessTexture' in m.get('pbrMetallicRoughness',{}) for m in g['materials']),asset['id'])
    def test_render_manifest_names_real_source_provider(self):
        m=json.loads((ROOT/'manifest.json').read_text())
        self.assertEqual(m['asset_provider'],'polyhaven');self.assertEqual(m['scene'],'living_room')
        self.assertEqual(len(m['objects']),10)
        self.assertEqual({a['id'] for a in m['asset_provenance']['assets']},
                         {a['id'] for a in json.loads((SOURCES/'manifest.json').read_text())['assets']})
        self.assertIn('UV', (ROOT/'SOURCES.md').read_text())

if __name__=='__main__':unittest.main()
