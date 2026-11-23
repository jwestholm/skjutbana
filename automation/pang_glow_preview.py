"""Natural PANG + local measurement circles, software preview ONLY. No real reference is claimed.

python3 -m automation.pang_glow_preview --room living_0 --out NEW_DIR
"""
import argparse,os,json,hashlib,contextlib,io
from pathlib import Path
from types import SimpleNamespace as N

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--room',required=True);p.add_argument('--out',required=True);p.add_argument('--assets',type=Path);p.add_argument('--seed',type=int,default=11);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    os.environ.update(SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',SKJUTBANA_SOURCE_RECORD='0',
        SKJUTBANA_PHYSICAL_TRACE_DISABLE='1',SKJUTBANA_PANG_OBJECT_HIGHLIGHT='1',SKJUTBANA_PANG_ROOM_HISTORY=str(out/'history.json'))
    import numpy as np,cv2,pygame
    from PIL import Image
    from content.games import light_room_art
    if a.assets is not None:light_room_art.NATURAL_ASSET_ROOT=a.assets.resolve()
    from content.games.light_targets_game import LightTargetsGame
    from src.engine.precision_flash import PrecisionFlash
    from src.engine.precision_object_glow import ObjectGlow,glow_mask,GlowConfig,measurement_circles,object_highlight_mask
    pygame.font.init();g=LightTargetsGame('',pygame.Rect(0,0,760,580));g.on_enter();g.sound_enabled=False
    try:
        g.art.rooms=[r for r in g.art.rooms if r['id']==a.room]
        if not g.art.rooms:raise ValueError('Unknown room '+a.room)
        g.rng.seed(a.seed);g.start()
        if g.phase=='unsupported':raise ValueError('No safe layout '+a.room)
        surface=pygame.Surface(g.viewport.size);g.render(surface);g.on_frame_presented(1.)
        # The tool is explicitly a presentation simulation, never acquisition.
        g.on_precision_glow_state(dict(ready=True,state='ready'));g.presented_mono-=2.;g.update(0);g.render(surface)
        pygame.image.save(surface,str(out/'normal.png'))
        parent=PrecisionFlash(log=lambda *_:None);glow=ObjectGlow(parent);glow.enabled=True;glow.state='SHOWING'
        glow.circles=measurement_circles(g._glow_regions,0);glow.mask=glow_mask(g._glow_regions,g.viewport.size,12)
        glow.highlight_mask=object_highlight_mask(g._glow_regions,g.viewport.size);glow.viewport=list(g.viewport);glow.cfg=GlowConfig.from_scene(g)
        g.phase='settling';g.on_precision_glow_state(dict(ready=False,state='show_targets'));g.render(surface);glow.render(surface)
        pygame.image.save(surface,str(out/'show_targets.png'))
        g.phase='live';g.on_precision_glow_state(dict(ready=True,state='ready'));g.render(surface)
        # Real game reaction API, simulated click at an actual silhouette interior.
        obj=next(o for o in g.manager.objects if o.metadata['sprite_kind']=='vase' and o.part_id=='body')
        spec=g.art.objects['vase']['parts']['body'];mask=np.asarray(Image.open(g.art.root/spec['mask']))
        _,_,_,point=cv2.minMaxLoc(cv2.distanceTransform(mask,cv2.DIST_L2,5));h,w=mask.shape
        x=obj.geometry.x+point[0]/w*obj.geometry.width;y=obj.geometry.y+point[1]/h*obj.geometry.height
        g._on_hit(N(is_simulated=True,game_x=x,game_y=y,shot_id=None))
        g.render(surface);g._reaction_start_mono-=.4;g.render(surface);pygame.image.save(surface,str(out/'breaking.png'))
        g._reaction_start_mono-=2.;g.update(0);g.render(surface);pygame.image.save(surface,str(out/'after.png'))
        (out/'preview.json').write_text(json.dumps(dict(evidence='SIMULATED_RENDER_AND_CLICK_ONLY_NOT_CAMERA_REFERENCE_OR_PHYSICAL_ACCURACY',
            room=a.room,seed=a.seed,objects=len(g.items),mask_fraction=float(glow.mask.mean()),
            asset_manifest_sha256=g.art.manifest_sha256,points=g.scores[0],
            objects_geometry=[dict(kind=i['kind'],anchor=i['anchor']['id'],aabb=list(i['geometry'].aabb)) for i in g.items]),indent=2))
    finally:g.on_exit()
    print(out)

if __name__=='__main__':main()
