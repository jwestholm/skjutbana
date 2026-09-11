"""Compute coarse measurable session domain descriptors from PRE imagery."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
def run(root,out):
 rows=[]
 for s in sorted(root.glob('session_*')):
  vals=[]
  for p in s.glob('shots/*/trace.json'):
   try:t=json.loads(p.read_text());f=next(f for f in t.get('frames',[]) if f.get('kind')=='pre_snapshot');a=np.load(p.parent/f['path'],mmap_mode='r');small=cv2.resize(a,(320,180),interpolation=cv2.INTER_AREA).astype(float);gx,gy=np.gradient(small);vals.append(dict(brightness=float(small.mean()),contrast=float(small.std()),gradient=float(np.hypot(gx,gy).mean()),edge_fraction=float((np.hypot(gx,gy)>10).mean())))
   except (StopIteration,FileNotFoundError):continue
  if vals: rows.append(dict(session=s.name,shots=len(vals),brightness=float(np.mean([v['brightness'] for v in vals])),contrast=float(np.mean([v['contrast'] for v in vals])),gradient=float(np.mean([v['gradient'] for v in vals])),edge_fraction=float(np.mean([v['edge_fraction'] for v in vals]))))
 out.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('content/ai/physical_traces'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);run(a.root,a.output)
