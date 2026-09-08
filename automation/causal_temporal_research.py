"""RESEARCH_ONLY spatial-reference reproduction and fixed temporal rankings.

Isolated offline process; no GT in scoring, no weight search. Saved causal
proposal coordinates are frozen. This is offline candidate evaluation, not
live-path-equivalent replay or physical validation.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2, DEFAULT_CONFIG


def temporal_score(delta, x, y):
    x,y=int(round(x)),int(round(y))
    if x<7 or y<7 or x>=delta.shape[1]-7 or y>=delta.shape[0]-7:
        return -float('inf')
    p=delta[y-7:y+8,x-7:x+8]
    yy,xx=np.ogrid[-7:8,-7:8];r=xx*xx+yy*yy
    return float(np.mean(p[r<=4])-np.mean(p[(r>=16)&(r<=49)]))


def run(root, output):
    output.mkdir(parents=True,exist_ok=False)
    cfg=dict(DEFAULT_CONFIG)
    config_path=Path('content/ai/detector_v2.json')
    if config_path.exists():
        cfg.update(json.loads(config_path.read_text()))
    (output/'reference_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    engine=CandidateGeneratorV2.__new__(CandidateGeneratorV2)
    rows=[]
    from PIL import Image, ImageDraw
    panels=[]
    for path in sorted((root/'shots').glob('*/trace.json')):
        t=json.loads(path.read_text());gtp=path.parent/'ground_truth.json'
        gt=json.loads(gtp.read_text()) if gtp.exists() else None
        decision=t['decision_input'];pool=decision['retained_candidates'];winner=decision['deterministic_selection']['last_candidate']
        stage=next(s for s in t['stages'] if s.get('candidate_pool_shot_id')==t['shot_id'] and s.get('candidates'))
        dbg=stage['window_debug']
        has_v2 = 'v2_bbox_x0' in dbg
        x0,y0,x1,y1=([int(dbg['v2_bbox_'+k]) for k in ('x0','y0','x1','y1')] if has_v2 else
                     [0,0,int(dbg['v2221_crop_w']),int(dbg['v2221_crop_h'])])
        ox,oy=int(pool[0]['analysis_crop_x0']),int(pool[0]['analysis_crop_y0'])
        pre_entries=[f for f in t['frames'] if f['kind']=='pre_history' and t['peak_ts']-cfg.get('pre_stack_window_s',.32)<=f['timestamp']<=t['peak_ts']-cfg.get('pre_stack_min_gap_s',.006)]
        pre_entries=sorted(pre_entries,key=lambda f:f['timestamp'])[-int(cfg.get('pre_stack_frames',3)):]
        first_ts=min(c['timestamp'] for c in pool)
        post_entry=min((f for f in t['frames'] if f['kind'] in ('post','pre_history')),key=lambda f:abs(f['timestamp']-first_ts))
        if abs(post_entry['timestamp']-first_ts)>1e-6 or not pre_entries:
            rows.append({'event':t['shot_id'],'reason':'exact source frames unavailable','physical':gt is not None});continue
        history=[np.load(path.parent/f['path'],mmap_mode='r') for f in pre_entries]
        post=np.load(path.parent/post_entry['path'],mmap_mode='r')[oy+y0:oy+y1,ox+x0:ox+x1]
        roi_entry=stage.get('evidence_maps',{}).get('roi_polygon')
        if roi_entry is None:
            roi_entry=next(s['evidence_maps']['roi_polygon'] for s in t['stages'] if 'roi_polygon' in s.get('evidence_maps',{}))
        roi=np.load(path.parent/roi_entry['path'])[y0:y1,x0:x1]
        arrays={};reg={};refstats={}
        for name, dx,dy in [('runtime_wrong_plane',0,0),('correct_camera_plane',ox,oy)]:
            pre=[im[dy+y0:dy+y1,dx+x0:dx+x1] for im in history]
            ref,noise,stats=engine._build_reference_and_noise(pre,roi=roi,cfg=cfg)
            blur=lambda im:cv2.GaussianBlur(im,(int(cfg.get('blur_kernel',3)),)*2,float(cfg.get('blur_sigma',.55)))
            ref=blur(ref);cur=blur(post)
            _,bias=engine._register_current(ref,ref,roi=roi,cfg=cfg)
            aligned,info=engine._register_current(ref,cur,roi=roi,cfg=cfg,registration_bias=(bias['raw_dx'],bias['raw_dy']))
            norm,offset=engine._normalise_photometry(ref,aligned,roi=roi)
            residual=ref.astype(np.float32)-norm.astype(np.float32)
            arrays[name]=(ref,norm,np.abs(residual),np.maximum(residual,0))
            reg[name]={**info,'offset':offset,'pre_mode':stats['mode']}
        fast=[c for c in pool if c.get('v2225_fast_extract')]
        comparison=[]
        for c in fast:
            x=int(round(c['camera_x']-ox-x0));y=int(round(c['camera_y']-oy-y0))
            yy,xx=np.ogrid[-2:3,-2:3];mask=xx*xx+yy*yy<=4
            if x<2 or y<2 or x>=post.shape[1]-2 or y>=post.shape[0]-2:continue
            values={k:float(np.mean(v[2][y-2:y+3,x-2:x+3][mask])) for k,v in arrays.items()}
            comparison.append({'xy':[c['camera_x'],c['camera_y']],'saved_psc':c['pre_shot_change'],**values})
        # Rank on the exact synchronous confirmation frame consumed before
        # the decision, retaining the proposal-frame reproduction separately.
        confirm_entry = next(f for f in t['frames'] if abs(f['timestamp']-winner['timestamp']) < 1e-6 and f['kind']=='post')
        if confirm_entry['timestamp'] > decision['timestamp']:
            raise ValueError('Confirmation frame follows live decision')
        confirm = np.load(path.parent/confirm_entry['path'], mmap_mode='r')[oy+y0:oy+y1,ox+x0:ox+x1]
        ref = arrays['correct_camera_plane'][0]
        _, bias = engine._register_current(ref,ref,roi=roi,cfg=cfg)
        aligned, confirm_reg = engine._register_current(ref,blur(confirm),roi=roi,cfg=cfg,registration_bias=(bias['raw_dx'],bias['raw_dy']))
        norm, _ = engine._normalise_photometry(ref,aligned,roi=roi)
        residual = ref.astype(np.float32)-norm.astype(np.float32)
        rankings={}
        for name,delta in [('REGISTERED_ABS_COMPACT',np.abs(residual)),('REGISTERED_DARK_COMPACT',np.maximum(residual,0))]:
            scored=[(temporal_score(delta,c['camera_x']-ox-x0,c['camera_y']-oy-y0),i,c) for i,c in enumerate(pool)]
            score,i,c=max(scored,key=lambda v:(v[0],-v[1]))
            rankings[name]={'score':score,'candidate':c,'error_px':math.hypot(c['camera_x']-gt['camera_x'],c['camera_y']-gt['camera_y']) if gt else None}
        rows.append({'event':t['shot_id'],'physical':gt is not None,'pre_timestamps':[f['timestamp'] for f in pre_entries],
                     'post_ts':post_entry['timestamp'], 'ranking_confirmation_ts':confirm_entry['timestamp'], 'ranking_registration':confirm_reg,'decision_ts':decision['timestamp'],'bbox_local':[x0,y0,x1,y1], 'crop_origin':[ox,oy],
                     'registration':reg,'fast_comparison':comparison,'rankings':rankings})
        # Recorded PRE/POST patches around every winner and GT; same display
        # scale, absolute residual x8 for visibility. No image editing model.
        pre=np.load(path.parent/next(f['path'] for f in t['frames'] if f['kind']=='pre_snapshot'),mmap_mode='r')
        fullpost=np.load(path.parent/confirm_entry['path'],mmap_mode='r')
        panel=Image.new('RGB',(600,180),(30,30,30));draw=ImageDraw.Draw(panel)
        draw.text((5,2),f"Event {t['shot_id']}  winner snapshot PRE / confirm POST / abs x8 | GT",fill='white')
        for j,c in enumerate([winner,gt]):
            if c is None:continue
            x,y=int(round(c['camera_x'])),int(round(c['camera_y']))
            a=np.asarray(pre[y-24:y+25,x-24:x+25]);b=np.asarray(fullpost[y-24:y+25,x-24:x+25])
            for k,im in enumerate([a,b,np.clip(np.abs(a.astype(float)-b)*8,0,255).astype('uint8')]):
                panel.paste(Image.fromarray(im).convert('RGB').resize((98,98)),(j*300+k*100,30))
            draw.text((j*300+3,135),f"XY {x},{y} PRE mean {a.mean():.1f} POST {b.mean():.1f}",fill='white')
        panels.append(panel)
    # Report evaluated and unavailable separately. No replacement with a later frame.
    metrics={}
    for name in ('REGISTERED_ABS_COMPACT','REGISTERED_DARK_COMPACT'):
        errors=[r['rankings'][name]['error_px'] for r in rows if r['physical'] and 'rankings' in r]
        metrics[name]={'evaluated':len(errors),'unavailable':20-len(errors),'hits':{str(k):sum(e<=k for e in errors) for k in (5,10,20,42)},
                       'mean_px':float(np.mean(errors)),'median_px':float(np.median(errors)), 'p95_px':float(np.percentile(errors,95))}
    (output/'research.json').write_text(json.dumps({'status':'RESEARCH_ONLY','evidence':'offline candidate evaluation; partial source-frame reconstruction', 'metrics':metrics,'rows':rows},indent=2)+'\n')
    if panels:
        image=Image.new('RGB',(1200,180*math.ceil(len(panels)/2)))
        for i,p in enumerate(panels):image.paste(p,((i%2)*600,(i//2)*180))
        image.save(output/'winner_gt_patches.png')
    print(json.dumps(metrics,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
if __name__=='__main__':main()
