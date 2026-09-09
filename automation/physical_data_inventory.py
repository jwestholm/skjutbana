"""Inventory physical traces without modifying source data."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def run(root,out):
 rows=[]
 for s in sorted(root.glob('session_*')):
  ps=sorted(s.glob('shots/*/trace.json')); labels=0; frames=0; complete=0; coords=[]
  parsed=[]
  for p in ps:
   try:t=json.loads(p.read_text());parsed.append(t)
   except Exception:continue
   labels+=int((p.parent/'ground_truth.json').exists());frames+=len(t.get('frames',[]));complete+=int('decision_input' in t and 'selectors' in t)
   coords.append(t.get('coordinate_space',{}))
  patch_ok=labels>0 and all(any(f.get('kind')=='pre_snapshot' for f in t.get('frames',[])) and any(f.get('kind')=='post' for f in t.get('frames',[])) for t in parsed)
  rows.append(dict(session=s.name,shots=len(ps),labels=labels,frames=frames,complete_rank_replay=complete==len(ps),patch_dataset_compatible=patch_ok,limitations=[] if complete==len(ps) else ['legacy trace lacks decision_input/selectors']))
 out.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('content/ai/physical_traces'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);run(a.root,a.output)
