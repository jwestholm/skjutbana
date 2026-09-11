"""Four lightweight source-independent patch information experiments."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from src.engine.offline.common_verifier import CommonFrameContext,extract_patch_context,patch_representation

KINDS=('difference','gradient','pca','texture')
def run(root,features,out):
 d=json.loads(features.read_text()); by={}
 for x in d['tracks']:
  if x['track'].get('eligible'):by.setdefault(int(x['shot']),[]).append(x)
 rows=[]; traces=sorted((root/'shots').glob('*/trace.json'))
 for tp in traces:
  t=json.loads(tp.read_text());sid=int(t['shot_id']);fs=sorted(t['frames'],key=lambda f:f['timestamp']);pf=next(f for f in fs if f['kind']=='pre_snapshot');pre=np.load(tp.parent/pf['path'],mmap_mode='r');post=[f for f in fs if f['kind']=='post'];imgs=[np.load(tp.parent/f['path'],mmap_mode='r') for f in post];ctx=CommonFrameContext(pre,tuple(imgs),float(t['decision_input']['timestamp']),tuple(f['timestamp'] for f in post))
  for x in by.get(sid,[]):
   tr=x['track'];xy=(float(tr['camera_x']),float(tr['camera_y']));b=extract_patch_context(ctx,xy);label=1 if x.get('evaluation_gt_distance') is not None and x['evaluation_gt_distance']<=42 else 0
   rows.append(dict(shot=sid,track_id=tr['track_id'],label=label,gt_distance=x.get('evaluation_gt_distance'),live_rank=tr.get('final_rank'),vectors={k:patch_representation(b,k).tolist() for k in KINDS}))
 results={}
 for kind in KINDS:
  ranks=[]
  for hold in sorted(set(r['shot'] for r in rows)):
   train=[np.asarray(r['vectors'][kind]) for r in rows if r['shot']!=hold and r['label']==1]
   if not train:continue
   mu=np.mean(train,0); allv=[np.asarray(r['vectors'][kind]) for r in rows if r['shot']!=hold];sd=np.std(allv,0)+1e-5;pool=[r for r in rows if r['shot']==hold];order=sorted(pool,key=lambda r:float(np.linalg.norm((np.asarray(r['vectors'][kind])-mu)/sd)))
   pos=next((r['track_id'] for r in pool if r['label']==1),None);rnk=next((i+1 for i,r in enumerate(order) if r['track_id']==pos),None);ranks.append(rnk)
  results[kind]=dict(ranks=ranks,top={str(k):sum(r is not None and r<=k for r in ranks) for k in (1,3,5,10,20,50)})
 out.write_text(json.dumps(dict(status='INFORMATION_TEST_ONLY',rows=rows,results=results),indent=2)+'\n');print(json.dumps(results))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);run(a.root,a.features,a.output)
