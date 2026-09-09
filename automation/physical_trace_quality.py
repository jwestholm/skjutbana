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
   if not t.get('completeness',{}).get('trace_complete',True): reasons.append(f'{p.parent.name}:producer_incomplete:{t.get("completeness",{}).get("completion_reason","unknown")}')
   events.append(dict(id=t.get('shot_id'),frames=len(fs),pre=haspre,post=haspost,complete=complete,ownership=bool(t.get('decision_input',{}))))
  def state(ok,warn=False):return 'PASS' if ok else ('WARN' if warn else 'FAIL')
  assignments={}
  ap=s/'physical_assignments.json'
  if ap.exists():
   try: assignments=json.loads(ap.read_text())
   except Exception: reasons.append('physical_assignments.json:invalid_json')
  resolved=sum(1 for p in s.glob('shots/*/trace.json') if (p.parent/'ground_truth.json').exists() or str(int(p.parent.name.rsplit('_',1)[-1])) in assignments)
  out.append(dict(session=s.name,events=len(events),trace_completeness=state(bool(events)),label_completeness=state(resolved==len(events),True),frame_completeness=state(all(e['pre'] and e['post'] for e in events)),full_replay_ready=state(all(e['complete'] for e in events)),patch_dataset_ready=state(all(e['pre'] and e['post'] for e in events)),temporal_ready=state(all(e['post'] for e in events)),reasons=reasons))
 return out
def inspect_one(s):
 return inspect(s.parent) if s.name.startswith('session_') else inspect(s)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('content/ai/physical_traces'));p.add_argument('--session-root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();
 if a.session_root:
  s=a.session_root; shots=list(s.glob('shots/*/trace.json')); payload=[]
  tmp=s.parent
  # inspect() is intentionally reused through a narrow temporary view only when the
  # caller supplies an actual session; avoid ambiguous latest-session discovery.
  events=[]; reasons=[]
  for pth in sorted(shots):
   try:t=json.loads(pth.read_text())
   except Exception: reasons.append(f'{pth}:invalid_json'); continue
   fs=t.get('frames',[]); pre=any(f.get('kind')=='pre_snapshot' for f in fs); post=any(f.get('kind') in ('post_snapshot','post') for f in fs); complete=all(k in t for k in ('decision_input','selectors'))
   if not pre: reasons.append(f'{pth.parent.name}:missing_pre')
   if not post: reasons.append(f'{pth.parent.name}:missing_post')
   if not t.get('completeness',{}).get('trace_complete',True): reasons.append(f'{pth.parent.name}:producer_incomplete:{t.get("completeness",{}).get("completion_reason","unknown")}')
   events.append(dict(id=t.get('shot_id'),frames=len(fs),pre=pre,post=post,complete=complete,ownership=bool(t.get('decision_input',{}))))
  assignments={}; ap=s/'physical_assignments.json'
  if ap.exists():
   try: assignments=json.loads(ap.read_text())
   except Exception: reasons.append('physical_assignments.json:invalid_json')
  resolved=sum(1 for p in shots if (p.parent/'ground_truth.json').exists() or str(int(p.parent.name.rsplit('_',1)[-1])) in assignments)
  payload=[dict(session=s.name,events=len(events),trace_completeness='PASS' if events else 'FAIL',label_completeness='PASS' if resolved==len(events) else 'WARN',frame_completeness='PASS' if all(e['pre'] and e['post'] for e in events) else 'FAIL',full_replay_ready='PASS' if all(e['complete'] for e in events) else 'WARN',patch_dataset_ready='PASS' if all(e['pre'] and e['post'] for e in events) else 'FAIL',temporal_ready='PASS' if all(e['post'] for e in events) else 'FAIL',reasons=reasons)]
 else: payload=inspect(a.root)
 a.output.write_text(json.dumps(payload,indent=2)+'\n');print(a.output)
