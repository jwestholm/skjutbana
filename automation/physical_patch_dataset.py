"""Build a compact unified real physical patch dataset; labels are offline only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from src.engine.offline.common_verifier import CommonFrameContext,extract_patch_context,patch_representation

def run(root,out):
 rows=[]
 for s in sorted(root.glob('session_*')):
  for tp in sorted(s.glob('shots/*/trace.json')):
   t=json.loads(tp.read_text());gp=tp.parent/'ground_truth.json'
   if not gp.exists():continue
   g=json.loads(gp.read_text());fs=sorted(t.get('frames',[]),key=lambda f:f['timestamp']);
   try:pf=next(f for f in fs if f['kind']=='pre_snapshot');posts=[f for f in fs if f['kind']=='post'];pre=np.load(tp.parent/pf['path'],mmap_mode='r');imgs=[np.load(tp.parent/f['path'],mmap_mode='r') for f in posts]
   except (StopIteration,KeyError,FileNotFoundError):continue
   if not posts:continue
   ctx=CommonFrameContext(pre,tuple(imgs),posts[-1]['timestamp'],tuple(f['timestamp'] for f in posts));base=dict(session=s.name,event=int(t.get('shot_id',g.get('shot_id',0))),timestamps=[pf['timestamp'],posts[-1]['timestamp']],coordinate_space=g.get('space'),gt_distance=None)
   pos=(float(g['camera_x']),float(g['camera_y']));b=extract_patch_context(ctx,pos,causal_only=False);rows.append(dict(base,role='positive',label=1,origin='manual_gt',xy=pos,vectors={k:patch_representation(b,k).tolist() for k in ('difference','gradient','pca','texture')}))
   outxy=t.get('outcome',{}).get('final_camera_xy',{})
   if isinstance(outxy,dict) and outxy.get('camera_x') is not None:
    neg=(float(outxy['camera_x']),float(outxy['camera_y']));dist=float(np.hypot(neg[0]-pos[0],neg[1]-pos[1]));
    if dist>42:
     b=extract_patch_context(ctx,neg,causal_only=False);rows.append(dict(base,role='selected_false',label=0,origin='recorded_outcome',gt_distance=dist,xy=neg,vectors={k:patch_representation(b,k).tolist() for k in ('difference','gradient','pca','texture')}))
 out.write_text(json.dumps(dict(status='OFFLINE_LABELS_ONLY',rows=rows),indent=2)+'\n');print(json.dumps(dict(rows=len(rows),positives=sum(r['label']==1 for r in rows),negatives=sum(r['label']==0 for r in rows),sessions=sorted(set(r['session'] for r in rows)))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('content/ai/physical_traces'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);run(a.root,a.output)
