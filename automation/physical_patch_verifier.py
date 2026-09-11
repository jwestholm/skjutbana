"""Tiny source-independent information test using saved common patch features."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

FEATURES=('ring_affine_compact','ring_affine_dark_contrast','ring_affine_center_dark','ring_affine_concentration','ring_affine_entropy')
def run(inp,out):
 d=json.loads(inp.read_text()); rows=[]
 for x in d['tracks']:
  f=x.get('image_features',{});v=[f.get(k) for k in FEATURES]
  if all(z is not None and np.isfinite(z) for z in v):
   t=dict(x['track']);t['evaluation_gt_distance']=x.get('evaluation_gt_distance'); rows.append((int(x['shot']),t,np.asarray(v,float)))
 results=[]
 for hold in sorted(set(s for s,_,_ in rows)):
  train=[v for s,t,v in rows if s!=hold and t.get('evaluation_gt_distance') is not None and t['evaluation_gt_distance']<=42]
  if not train:continue
  mu=np.mean(train,0); sd=np.std([v for s,t,v in rows if s!=hold],0)+1e-6
  pool=[(t,np.linalg.norm((v-mu)/sd)) for s,t,v in rows if s==hold]
  order=sorted(pool,key=lambda z:z[1]);results.append(dict(shot=hold,ranks=[z[0]['track_id'] for z in order[:10]],positive_count=len(train)))
 out.write_text(json.dumps(dict(status='INFORMATION_TEST_ONLY',features=FEATURES,results=results),indent=2)+'\n')
 print(json.dumps(dict(rows=len(rows),holds=len(results))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);run(a.input,a.output)
