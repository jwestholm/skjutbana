"""Frozen-seed digital local-glow evaluation. Pixels/timing are simulated, not a physical pilot.

python3 -m automation.object_glow_bench NEW_DIR --seeds 20
GT is consumed only after measurement/OBJECT decision; no detector tuning here.
"""
import argparse,json,os,hashlib
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
from dataclasses import replace
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ['SKJUTBANA_SOURCE_RECORD']='0'
from src.engine.precision_flash import FlashConfig
from src.engine.precision_object_glow import GlowConfig,glow_mask,measure_glow,ObjectGlowQuery,decide_glow
from src.engine.input.object_hit_v2223 import GameHitRegionV240


def run(out,seeds):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    size=(240,160);camera_scale=3;r=GameHitRegionV240('target',75,50,42,42,metadata=dict(v250_object_snapshot=dict(entity_id='target',shape=dict(kind='rect',x=75,y=50,width=42,height=42))))
    regions=(r,);mask=glow_mask(regions,size,14);projection=lambda p:np.asarray(p,float).reshape(-1,2)/camera_scale
    cfg=GlowConfig();fcfg=FlashConfig();rows=[]
    for seed in range(seeds):
        for bg in ('dark','checker','moving','bright'):
            for case in ('hit','weak_hit','near_miss','no_hole','old_and_new','outside_mask','brightening'):
                rng=np.random.default_rng(seed);yy,xx=np.mgrid[:size[1]*camera_scale,:size[0]*camera_scale]
                import cv2
                camera_scope=cv2.resize(mask,(size[0]*camera_scale,size[1]*camera_scale),interpolation=cv2.INTER_NEAREST)
                normal=np.full(camera_scope.shape,30 if bg=='dark' else 95 if bg in ('checker','moving') else 230.,float)
                if bg in ('checker','moving'):normal+=45*((xx//7+yy//9)%2)
                point=(96,71) if case in ('hit','weak_hit','old_and_new','brightening') else (123,71) if case=='near_miss' else (8,8) if case=='outside_mask' else None
                old=(88,58) if case=='old_and_new' else None
                def stamp(xy):return np.exp(-((xx-xy[0]*camera_scale)**2+(yy-xy[1]*camera_scale)**2)/(2*(1.35*camera_scale)**2)) if xy is not None else None
                old_stamp=stamp(old);new_stamp=stamp(point)
                def make(on,off,post):
                    frames=[]
                    for ts in np.arange(on-.4,off+.35,.04):
                        lit=ts-.066-.0167>=on+.06 and ts-.066<=off+.02
                        image=normal.copy()
                        if post and bg=='moving':image=np.roll(image,8,axis=1)
                        if lit:image[camera_scope>0]=155.+15.*xx[camera_scope>0]/(size[0]*camera_scale)
                        if old_stamp is not None:image*=1-.32*old_stamp
                        if post and new_stamp is not None:image*=1-(-.25 if case=='brightening' else .05 if case=='weak_hit' else .16+seed%3*.07)*new_stamp
                        image=np.clip(image+rng.normal(0,.50,image.shape),0,255).astype(np.uint8)
                        frames.append(N(timestamp=float(ts),frame_bgr=image))
                    return frames
                ref=measure_glow(make(10,10.4,False),10,10.4,None,mask,(0,0,*size),projection,cfg,fcfg,is_reference=True)
                if not ref['valid']:raise ValueError(('reference',ref))
                post=measure_glow(make(20,20.18,True),20,20.18,ref['image'],mask,(0,0,*size),projection,cfg,fcfg,camera_valid=ref['camera_valid'])
                q=ObjectGlowQuery(N(shot_id=1,peak_ts=19.,game_regions=regions,camera_regions=[]),lambda:21.)
                decision,hyp=decide_glow(q,post.get('holes',[]),projection,(0,0,*size),(1,1),post) if post.get('valid') else (dict(state='UNKNOWN'),[])
                expected='OBJECT_HIT' if case in ('hit','weak_hit','old_and_new') else 'NO_OBJECT_HIT' if case=='near_miss' else 'UNKNOWN'
                row=dict(seed=seed,background=bg,case=case,expected=expected,actual=decision['state'],
                    correct=decision['state']==expected,valid=bool(post.get('valid')),holes=len(post.get('holes',[])),distance_mm=decision.get('distance_mm'))
                if hyp and point is not None:row['xy_error_px']=float(np.linalg.norm(np.asarray(hyp[0]['camera_xy'])-np.asarray(point)*camera_scale))
                rows.append(row)
    counts={}
    for row in rows:
        c=counts.setdefault(row['case'],dict(total=0,correct=0));c['total']+=1;c['correct']+=int(row['correct'])
    report=dict(kind='DIGITAL_PIXELS_AND_TIMING_ONLY_NOT_PHYSICAL_OR_SOURCE_REPLAY',seeds=seeds,cases=len(rows),
        correct=sum(r['correct'] for r in rows),by_case=counts,
        unexpected_awards=sum(r['actual']=='OBJECT_HIT' and r['expected']!='OBJECT_HIT' for r in rows),
        fixture='240x160 projector, 720x480 camera, exact scale3 mapping; bounded gray noise and Gaussian darkening, not real-hole stamps',
        glow_config=cfg.__dict__,flash_config=fcfg.__dict__,
        recipes_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('src/engine/precision_object_glow.py','src/engine/precision_flash.py','automation/object_glow_bench.py')})
    (out/'rows.json').write_text(json.dumps(rows,indent=2));(out/'metrics.json').write_text(json.dumps(report,indent=2))
    (out/'REPORT.md').write_text('# Digital object glow\n\n'+json.dumps(report,indent=2)+'\n\nNo camera/projector or physical accuracy claim.\n')
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');p.add_argument('--seeds',type=int,default=20);a=p.parse_args()
    if a.seeds<1:raise ValueError('positive seeds required')
    run(a.output,a.seeds)
