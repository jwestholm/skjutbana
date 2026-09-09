"""Causal full-pool temporal feature replay over recorded event frames."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
from src.engine.offline.registered_impact import sample

def local(pre,img,xy):
    p=sample(pre,xy,(0,0),radius=16).astype(float); q=sample(img,xy,(0,0),radius=16).astype(float)
    yy,xx=np.mgrid[:33,:33];r=np.hypot(xx-16,yy-16); c=r<=4; ring=(r>=8)&(r<=12); d=p-q
    ad=np.abs(d); dark=np.maximum(d,0); return dict(center_dark=float(dark[c].mean()),center_abs=float(ad[c].mean()),ring_abs=float(ad[ring].mean()),compact=float(ad[c].mean()-ad[ring].mean()),dark_contrast=float(dark[c].mean()-dark[ring].mean()),concentration=float(ad[c].sum()/max(ad[r<=12].sum(),1e-6)))

def run(root,features,out):
    d=json.loads(features.read_text()); by={}
    for x in d['tracks']:
        if x['track'].get('eligible'): by.setdefault(int(x['shot']),[]).append(x)
    rows=[]
    for tp in sorted((root/'shots').glob('*/trace.json')):
        t=json.loads(tp.read_text()); sid=int(t['shot_id']); pool=by.get(sid,[])
        if not pool: continue
        fs=sorted(t['frames'],key=lambda f:f['timestamp']); pre_f=next(f for f in fs if f['kind']=='pre_snapshot'); pre=np.load(tp.parent/pre_f['path'],mmap_mode='r'); decision=float(t.get('decision_input',{}).get('timestamp',t['outcome'].get('decision_ts',t.get('peak_ts',0))))
        post=[f for f in fs if f['kind']=='post']; causal=[f for f in post if f['timestamp']<=decision+1e-6]; allpost=post
        for x in pool:
            tr=x['track'];xy=(float(tr['camera_x']),float(tr['camera_y']))
            def summarize(frames):
                vals=[local(pre,np.load(tp.parent/f['path'],mmap_mode='r'),xy) for f in frames]
                if not vals:return dict(available=False)
                return dict(available=True,frames=len(vals),first_center_dark=vals[0]['center_dark'],peak_center_dark=max(v['center_dark'] for v in vals),median_center_dark=float(np.median([v['center_dark'] for v in vals])),peak_compact=max(v['compact'] for v in vals),median_compact=float(np.median([v['compact'] for v in vals])),persistence=sum(v['center_dark']>1.0 for v in vals),first_dark_contrast=vals[0]['dark_contrast'])
            before=[f for f in fs if f['kind']=='pre_history' and f['timestamp']<t['peak_ts']-.05]
            b=[local(pre,np.load(tp.parent/f['path'],mmap_mode='r'),xy)['center_dark'] for f in before]
            rows.append(dict(shot=sid,track_id=tr['track_id'],source=tr.get('source'),live_rank=tr.get('final_rank'),best_score=tr.get('best_score'),gt_distance=x.get('evaluation_gt_distance'),before_stability=float(np.std(b)) if b else None,causal=summarize(causal),posthoc=summarize(allpost)))
    out.write_text(json.dumps(dict(status='RESEARCH_ONLY',rows=rows),indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),events=len(set(r['shot'] for r in rows)),causal_frames=sum(r['causal'].get('frames',0) for r in rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);run(a.root,a.features,a.output)
