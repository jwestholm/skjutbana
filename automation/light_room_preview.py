"""Render software-only game previews and WAV cues. Never starts App, camera or projector.
python3 -m automation.light_room_preview --out /tmp/light_room_preview_NEW
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import wave


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=11);p.add_argument('--assets',type=Path,help='Preview an unpublished bundle');p.add_argument('--room',help='Pin a room id for a reproducible software preview');p.add_argument('--all-parts',action='store_true');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    os.environ['SDL_VIDEODRIVER']='dummy';os.environ['SDL_AUDIODRIVER']='dummy'
    os.environ['SKJUTBANA_PANG_ROOM_HISTORY']=str(out/'room_history.json')
    os.environ['SKJUTBANA_PHYSICAL_TRACE_DISABLE']='1';os.environ['SKJUTBANA_SOURCE_RECORD']='0'
    import cv2
    import numpy as np
    from PIL import Image
    import pygame
    from types import SimpleNamespace
    from content.games import light_room_art
    if a.assets is not None:light_room_art.ASSET_ROOT=a.assets.resolve()
    from content.games.light_targets_game import LightTargetsGame
    from content.games.light_room_art import ASSET_ROOT
    from content.games.light_room_audio import cue_wave
    pygame.font.init()  # no display, mixer, camera, network or App initialization
    summaries=[]
    cases=[('vase',p) for p in ('flower','neck','body')]+[(None,p) for p in ('MISS','UNKNOWN')]
    if a.all_parts:
        import json as _json
        manifest=_json.loads((ASSET_ROOT/'manifest.json').read_text())
        cases += [(k,p) for k,spec in manifest['objects'].items() if k!='vase' for p in spec['parts']]
    for kind,part in cases:
        label=part if kind in (None,'vase') else kind+'_'+part
        g=LightTargetsGame('',pygame.Rect(0,0,760,580));g.on_enter()
        if a.room:
            g.art.rooms=[r for r in g.art.rooms if r['id']==a.room]
            if not g.art.rooms:raise ValueError('Unknown room '+a.room)
        for offset in range(80):
            g.rng.seed(a.seed+offset);g.start()
            if g.phase!='unsupported' and (kind is None or any(i['kind']==kind for i in g.items)):break
        else:raise ValueError('No safe layout containing '+str(kind))
        try:
            surface=pygame.Surface((760,580));g.render(surface);g.on_frame_presented(1.)
            g.presented_mono-=2;g.update(0);g.render(surface)
            if part=='flower':pygame.image.save(surface,str(out/'room.png'))
            if part in ('MISS','UNKNOWN'):
                g._finish(part,None,None,None,simulated=True)
            else:
                obj=next(o for o in g.manager.objects if o.metadata['sprite_kind']==kind and o.part_id==part)
                mask=np.asarray(Image.open(ASSET_ROOT/g.art.objects[kind]['parts'][part]['mask']))
                _,_,_,point=cv2.minMaxLoc(cv2.distanceTransform(mask,cv2.DIST_L2,5));h,w=mask.shape
                x=obj.geometry.x+point[0]/w*obj.geometry.width;y=obj.geometry.y+point[1]/h*obj.geometry.height
                g._on_hit(SimpleNamespace(is_simulated=True,game_x=x,game_y=y,shot_id=None))
            g.render(surface)
            g._reaction_start_mono-=.55;g.render(surface);pygame.image.save(surface,str(out/(label+'_breaking.png')))
            g._reaction_start_mono-=2;g.update(0);g.render(surface);pygame.image.save(surface,str(out/(label+'_after.png')))
            summaries.append(dict(kind=kind,part=part,layout_seed=a.seed+offset,points=g.scores[0],broken=sorted(g.broken_ids)))
        finally:g.on_exit()
    for cue in ('shatter','topple','petal','miss','unknown'):
        data=cue_wave(cue,22050)
        with wave.open(str(out/(cue+'.wav')),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(22050);f.writeframes(data.tobytes())
    (out/'preview.json').write_text(json.dumps(dict(evidence='RENDERED_SIMULATED_ONLY_NOT_PHYSICAL',seed=a.seed,assets_manifest_sha256=hashlib.sha256((ASSET_ROOT/'manifest.json').read_bytes()).hexdigest(),cases=summaries),indent=2))
    (out/'LICENSE.txt').write_text('Original generated preview images and synthesized WAV cues: CC0-1.0.\nhttps://creativecommons.org/publicdomain/zero/1.0/\n')
    print('Software previews:',out)

if __name__=='__main__':main()
