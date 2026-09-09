"""Session-held-out centroid tests on the unified physical patch dataset."""
import argparse,json
from pathlib import Path
import numpy as np
KINDS=('difference','gradient','pca','texture')
def run(inp,out):
 d=json.loads(inp.read_text());rows=d['rows'];res={}
 for k in KINDS:
  scores=[]
  for hold in sorted(set(r['session'] for r in rows)):
   train=[np.asarray(r['vectors'][k]) for r in rows if r['session']!=hold and r['label']==1]
   test=[r for r in rows if r['session']==hold]
   if not train or not any(r['label'] for r in test):continue
   mu=np.mean(train,0);sd=np.std([np.asarray(r['vectors'][k]) for r in rows if r['session']!=hold],0)+1e-5;order=sorted(test,key=lambda r:float(np.linalg.norm((np.asarray(r['vectors'][k])-mu)/sd)))
   scores.append(dict(session=hold,positive_ranks=[i+1 for i,r in enumerate(order) if r['label']==1]))
  res[k]=scores
 out.write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);run(a.input,a.output)
