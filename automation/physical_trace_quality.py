"""Diagnostic physical-session quality scoring; never changes detector state."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def inspect(root):
 out=[]
 for s in sorted(root.glob('session_*')):
  events=[]; reasons=[]
  for p in sorted(s.glob('shots/*/trace.json')):
   try:t=json.loads(p.read_text())
   except: reasons.append(f'{p}:invalid_json');continue
   fs=t.get('frames',[]); haspre=any(f.get('kind')=='pre_snapshot' for f in fs); haspost=any(f.get('kind')=='post' for f in fs); complete=all(k in t for k in ('decision_input','selectors'))
   if not haspre:reasons.append(f'{p.parent.name}:missing_pre')
   if not haspost:reasons.append(f'{p.parent.name}:missing_post')
   events.append(dict(id=t.get('shot_id'),frames=len(fs),pre=haspre,post=haspost,complete=complete,ownership=bool(t.get('decision_input',{}))))
  def state(ok,warn=False):return 'PASS' if ok else ('WARN' if warn else 'FAIL')
  out.append(dict(session=s.name,events=len(events),trace_completeness=state(bool(events)),label_completeness=state(sum((p.parent/'ground_truth.json').exists() for p in s.glob('shots/*/trace.json'))==len(events),True),frame_completeness=state(all(e['pre'] and e['post'] for e in events)),full_replay_ready=state(all(e['complete'] for e in events)),patch_dataset_ready=state(all(e['pre'] and e['post'] for e in events)),temporal_ready=state(all(e['post'] for e in events)),reasons=reasons))
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('content/ai/physical_traces'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(inspect(a.root),indent=2)+'\n');print(a.output)
