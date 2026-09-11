"""Future-aware offline permanence diagnostics; never live authority."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from src.engine.offline.registered_impact import sample

def feat(pre,img,xy):
 p=sample(pre,xy,(0,0),radius=16).astype(float);q=sample(img,xy,(0,0),radius=16).astype(float);d=p-q
 yy,xx=np.mgrid[:33,:33];r=np.hypot(xx-16,yy-16);c=r<=4;ring=(r>=8)&(r<=12);a=np.abs(d);dark=np.maximum(d,0)
 return dict(center_abs=float(a[c].mean()),center_dark=float(dark[c].mean()),compact=float(a[c].mean()-a[ring].mean()),dark_contrast=float(dark[c].mean()-dark[ring].mean()),concentration=float(a[c].sum()/max(a[r<=12].sum(),1e-6)))

def run(root,features,out):
 d=json.loads(features.read_text()); shots=sorted((root/'shots').glob('*/trace.json')); traces=[json.loads(p.read_text()) for p in shots]; rows=[]
 for i,(tp,t) in enumerate(zip(shots,traces)):
  fs=sorted(t['frames'],key=lambda f:f['timestamp']);pre_f=next(f for f in fs if f['kind']=='pre_snapshot');pre=np.load(tp.parent/pre_f['path'],mmap_mode='r');post=[f for f in fs if f['kind']=='post'];nextpre=None
  if i+1<len(traces):
   ntp=shots[i+1];nt=traces[i+1];nf=next(f for f in nt['frames'] if f['kind']=='pre_snapshot');nextpre=np.load(ntp.parent/nf['path'],mmap_mode='r')
  pool=[x for x in d['tracks'] if x['shot']==t['shot_id'] and x['track'].get('eligible')]
  for x in pool:
   tr=x['track'];xy=(float(tr['camera_x']),float(tr['camera_y'])); immediate=feat(pre,np.load(tp.parent/post[-1]['path'],mmap_mode='r'),xy) if post else None; later=feat(pre,nextpre,xy) if nextpre is not None else None
   stable=[feat(pre,np.load(tp.parent/f['path'],mmap_mode='r'),xy) for f in post if f['timestamp']-t['peak_ts']>=1.0]
   rows.append(dict(shot=t['shot_id'],track_id=tr['track_id'],live_rank=tr.get('final_rank'),best_score=tr.get('best_score'),gt_distance=x.get('evaluation_gt_distance'),immediate=immediate,later_next_pre=later,stable_1s=stable[-1] if stable else None,stable_count=len(stable)))
 out.write_text(json.dumps(dict(status='OFFLINE_LONG_HORIZON_DIAGNOSTIC',rows=rows),indent=2)+'\n');print(json.dumps(dict(rows=len(rows),events=len(traces),next_pre_events=sum(i+1<len(traces) for i in range(len(traces))),long_post=sum(r['stable_1s'] is not None for r in rows))))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);run(a.root,a.features,a.output)
