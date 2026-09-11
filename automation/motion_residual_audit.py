"""Offline local-motion/edge residual audit; never live authority."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
from src.engine.offline.registered_impact import sample

def measure(pre,post,xy):
    p=sample(pre,xy,(0,0),radius=16).astype(np.float32); q=sample(post,xy,(0,0),radius=16).astype(np.float32)
    shift,_=cv2.phaseCorrelate(p,q); dx,dy=float(np.clip(shift[0],-2,2)),float(np.clip(shift[1],-2,2))
    aligned=cv2.warpAffine(q,np.float32([[1,0,-dx],[0,1,-dy]]),(33,33),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    residual=p-q; ar=p-aligned; gy,gx=np.gradient(p); pred=gx*dx+gy*dy
    yy,xx=np.mgrid[:33,:33];r=np.hypot(xx-16,yy-16);c=r<=4;ring=(r>=8)&(r<=12);mag=np.hypot(gx,gy)
    def energy(x):return float(np.mean(np.abs(x))),float(np.mean(np.abs(x)[c])),float(np.mean(np.abs(x)[ring]))
    raw,rc,rr=energy(residual); al,ac,arng=energy(ar)
    rv=residual.ravel()-residual.mean();pv=pred.ravel()-pred.mean();corr=float(np.dot(rv,pv)/(np.linalg.norm(rv)*np.linalg.norm(pv)+1e-6))
    return dict(local_dx=dx,local_dy=dy,local_shift=float(np.hypot(dx,dy)),raw_energy=raw,aligned_energy=al,motion_explained=float(1-al/max(raw,1e-6)),raw_center=rc,aligned_center=ac,raw_ring=rr,aligned_ring=arng,gradient_energy=float(np.mean(mag)),gradient_fit=corr,dipole_balance=float(abs(np.sum(np.maximum(residual,0))-np.sum(np.maximum(-residual,0)))/max(np.sum(np.abs(residual)),1e-6)) )

def run(root,features,out):
    d=json.loads(features.read_text()); by={}
    for x in d['tracks']:
        if x['track'].get('eligible'):by.setdefault(int(x['shot']),[]).append(x)
    rows=[]
    for tp in sorted((root/'shots').glob('*/trace.json')):
        t=json.loads(tp.read_text());sid=int(t['shot_id']);fs=sorted(t['frames'],key=lambda f:f['timestamp']);pre_f=next(f for f in fs if f['kind']=='pre_snapshot');post_f=max((f for f in fs if f['kind']=='post' and f['timestamp']<=float(t['decision_input']['timestamp'])),key=lambda f:f['timestamp']);pre=np.load(tp.parent/pre_f['path'],mmap_mode='r');post=np.load(tp.parent/post_f['path'],mmap_mode='r')
        for x in by.get(sid,[]):
            tr=x['track'];m=measure(pre,post,(float(tr['camera_x']),float(tr['camera_y'])));rows.append(dict(shot=sid,track_id=tr['track_id'],source=tr.get('source'),live_rank=tr.get('final_rank'),best_score=tr.get('best_score'),gt_distance=x.get('evaluation_gt_distance'),**m))
    out.write_text(json.dumps(dict(status='RESEARCH_ONLY',rows=rows),indent=2)+'\n');print(json.dumps(dict(rows=len(rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);run(a.root,a.features,a.output)
