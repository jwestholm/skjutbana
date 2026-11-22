"""Render software-only game previews and WAV cues. Never starts App, camera or projector.
python3 -m automation.light_room_preview --out /tmp/light_room_preview_NEW
"""
import argparse
import json
import os
from pathlib import Path
import wave


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=11);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    os.environ['SDL_VIDEODRIVER']='dummy';os.environ['SDL_AUDIODRIVER']='dummy'
    os.environ['SKJUTBANA_PHYSICAL_TRACE_DISABLE']='1';os.environ['SKJUTBANA_SOURCE_RECORD']='0'
    import cv2
    import numpy as np
    from PIL import Image
    import pygame
    from types import SimpleNamespace
    from content.games.light_targets_game import LightTargetsGame
    from content.games.light_room_art import ASSET_ROOT
    from content.games.light_room_audio import cue_wave
    pygame.font.init()  # no display, mixer, camera, network or App initialization
    summaries=[]
    for part in ('flower','neck','body'):
        g=LightTargetsGame('',pygame.Rect(0,0,760,580));g.on_enter();g.rng.seed(a.seed);g.start()
        try:
            surface=pygame.Surface((760,580));g.render(surface);g.on_frame_presented(1.)
            g.presented_mono-=2;g.update(0);g.render(surface)
            if part=='flower':pygame.image.save(surface,str(out/'room.png'))
            obj=next(o for o in g.manager.objects if o.metadata['sprite_kind']=='vase' and o.part_id==part)
            mask=np.asarray(Image.open(ASSET_ROOT/g.art.objects['vase']['parts'][part]['mask']))
            _,_,_,point=cv2.minMaxLoc(cv2.distanceTransform(mask,cv2.DIST_L2,5));h,w=mask.shape
            x=obj.geometry.x+point[0]/w*obj.geometry.width;y=obj.geometry.y+point[1]/h*obj.geometry.height
            g._on_hit(SimpleNamespace(is_simulated=True,game_x=x,game_y=y,shot_id=None));g.render(surface)
            g._reaction_start_mono-=.55;g.render(surface);pygame.image.save(surface,str(out/(part+'_breaking.png')))
            g._reaction_start_mono-=2;g.update(0);g.render(surface);pygame.image.save(surface,str(out/(part+'_after.png')))
            summaries.append(dict(part=part,points=g.scores[0],broken=sorted(g.broken_ids)))
        finally:g.on_exit()
    for cue in ('shatter','topple','petal'):
        data=cue_wave(cue,22050)
        with wave.open(str(out/(cue+'.wav')),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(22050);f.writeframes(data.tobytes())
    (out/'preview.json').write_text(json.dumps(dict(evidence='RENDERED_SIMULATED_ONLY_NOT_PHYSICAL',seed=a.seed,cases=summaries),indent=2))
    (out/'LICENSE.txt').write_text('Original generated preview images and synthesized WAV cues: CC0-1.0.\nhttps://creativecommons.org/publicdomain/zero/1.0/\n')
    print('Software previews:',out)

if __name__=='__main__':main()
