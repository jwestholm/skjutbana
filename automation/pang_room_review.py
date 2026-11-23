"""Software-only R7 room/local-color review; never starts App or capture.

python3 -m automation.pang_room_review NEW_OUTPUT.json --layouts 30
"""
import argparse,contextlib,hashlib,io,json,os,tempfile
from pathlib import Path
import numpy as np


def measure(layouts):
    os.environ['SDL_VIDEODRIVER']='dummy';os.environ['SDL_AUDIODRIVER']='dummy'
    os.environ['SKJUTBANA_PHYSICAL_TRACE_DISABLE']='1';os.environ['SKJUTBANA_SOURCE_RECORD']='0'
    import pygame
    from content.games.light_targets_game import LightTargetsGame
    from content.games.light_room_safety import camera_proxy
    pygame.font.init();rows=[];centers=[]
    with tempfile.TemporaryDirectory() as td:
        previous=os.environ.get('SKJUTBANA_PANG_ROOM_HISTORY');os.environ['SKJUTBANA_PANG_ROOM_HISTORY']=str(Path(td)/'history.json')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                game=LightTargetsGame('',pygame.Rect(0,0,760,580));game.on_enter()
                try:
                    for seed in range(layouts):
                        game.rng.seed(seed);game.start()
                        if game.phase=='unsupported':raise ValueError('Unsafe/unsupported layout '+str(seed))
                        screen=pygame.Surface(game.viewport.size);game.art.render(screen,game.items)
                        rgb=pygame.surfarray.array3d(screen);plate=pygame.surfarray.array3d(game.art._plate)
                        bg=pygame.surfarray.array3d(game.art.background);weight=game.art.safety_weight
                        if not np.array_equal(bg[weight==0],plate[weight==0]):raise ValueError('Global room lift')
                        core=weight==1;minimum=float(camera_proxy(rgb[core]).min())
                        if minimum<120:raise ValueError('Local color floor failed: '+str(minimum))
                        points=[(i['geometry'].center[0]/760,i['geometry'].center[1]/580) for i in game.items];centers+=points
                        rows.append(dict(seed=seed,room_id=game.art.room_id,objects=len(game.items),
                            min_target_neighbourhood_dn=minimum,background_changed_fraction=float(np.any(bg!=plate,axis=2).mean()),
                            outside_unchanged=True,centers=points,
                            sizes=[dict(kind=i['kind'],anchor=i['anchor']['id'],width=i['geometry'].width,height=i['geometry'].height) for i in game.items]))
                    digest=game.art.manifest_sha256
                finally:game.on_exit()
        finally:
            if previous is None:os.environ.pop('SKJUTBANA_PANG_ROOM_HISTORY',None)
            else:os.environ['SKJUTBANA_PANG_ROOM_HISTORY']=previous
    xy=np.array(centers);grid=np.histogram2d(xy[:,1],xy[:,0],bins=(6,8),range=((0,1),(0,1)))[0].astype(int)
    return dict(kind='SOFTWARE_LAYOUT_AND_DISPLAY_PROXY_NOT_PHYSICAL_ACCURACY',asset_manifest_sha256=digest,
        game_recipes_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('content/games/light_targets_game.py','content/games/light_room_art.py','content/games/light_room_safety.py')},
        layouts=layouts,rooms=sorted({r['room_id'] for r in rows}),min_target_neighbourhood_dn=min(r['min_target_neighbourhood_dn'] for r in rows),
        max_background_changed_fraction=max(r['background_changed_fraction'] for r in rows),
        target_center_grid_8x6=grid.tolist(),rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--layouts',type=int,default=30);a=p.parse_args()
    if a.layouts<1:raise ValueError('layouts must be positive')
    if a.output.exists():raise FileExistsError(a.output)
    result=measure(a.layouts)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print('Rooms:',len(result['rooms']),'minimum target neighbourhood DN:',result['min_target_neighbourhood_dn'])
