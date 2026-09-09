"""Safe research collection orchestration and validation helpers."""
from __future__ import annotations
import argparse,json,uuid,subprocess,os,hashlib,shutil
from datetime import datetime,timezone
from pathlib import Path

def load_plan(path):
 d=json.loads(Path(path).read_text());rows=d.get('rows',[])
 if not rows:raise ValueError('empty capture plan')
 sessions=sorted(set(r['session'] for r in rows));
 if len(sessions)!=d.get('sessions'):raise ValueError('session count mismatch')
 return d
def bind(plan,session,trace_root,source_commit='unknown'):
 rows=[r for r in plan['rows'] if r['session']==session]
 if not rows:raise ValueError('unknown session')
 existing=Path(trace_root)/f'planned_{session}.json'
 if existing.exists():raise FileExistsError(f'{session} already has a binding at {existing}; choose a new plan/session')
 if session=='S03' and any(r.get('session_class')!='VALIDATION_UNTOUCHED' for r in rows):raise ValueError('validation binding mismatch')
 meta=dict(collection_plan_id=plan.get('collection_plan_id',plan.get('plan_id','local-plan')),planned_session_id=session,session_class=rows[0]['session_class'],created_at=datetime.now(timezone.utc).isoformat(),source_commit=source_commit,trace_root=str(trace_root),rows=rows)
 return meta
def guard_training(session_class,operation):
 if session_class=='VALIDATION_UNTOUCHED' and operation in ('train','tune','fit','select_config','model_select','threshold_select','research_calibrate'):
  raise PermissionError('validation data is evaluation-only; training/tuning refused')
 return True
def _settings(path):
 p=Path(path); return json.loads(p.read_text())
def _set_trace(settings_path,root,backup):
 p=Path(settings_path); original=p.read_bytes(); Path(backup).write_bytes(original)
 data=json.loads(original); data['physical_trace_capture_enabled']=True; data['physical_trace_root']=str(root)
 p.write_text(json.dumps(data,indent=2)+'\n'); return hashlib.sha256(original).hexdigest()
def preflight(plan_path,session,trace_root,binding=None,settings_path='content/ai/settings.json'):
 reasons=[]
 try: plan=load_plan(plan_path);rows=[r for r in plan['rows'] if r['session']==session]
 except Exception as e:return False,[f'plan unreadable: {e}']
 if not rows:reasons.append(f'plan has no session {session}')
 if (Path(trace_root)/f'planned_{session}.json').exists():reasons.append('session binding already exists; refusing overwrite')
 if not os.access(Path(trace_root),os.W_OK):reasons.append(f'trace root is not writable: {trace_root}')
 if binding:
  try:
   b=json.loads(Path(binding).read_text()); configured=_settings(settings_path)
   expected=str(Path(b['trace_root']))
   if not configured.get('physical_trace_capture_enabled'): reasons.append('physical trace capture is disabled; run start again')
   if str(configured.get('physical_trace_root','')) != expected: reasons.append(f'physical trace root is {configured.get("physical_trace_root")!r}, expected {expected!r}; run start or restore binding')
   if Path(expected).name.startswith('session_20260907_'): reasons.append('refusing historical trace root; use the unique root in the binding')
  except Exception as e: reasons.append(f'binding/settings unreadable: {e}')
 try: subprocess.run(['git','rev-parse','--abbrev-ref','HEAD'],check=True,capture_output=True,text=True)
 except Exception:reasons.append('git branch cannot be verified')
 return not reasons,reasons
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 s=sub.add_parser('start');s.add_argument('--plan',type=Path,required=True);s.add_argument('--session',required=True);s.add_argument('--trace-root',type=Path,default=Path('content/ai/physical_traces'));s.add_argument('--source-commit',default='unknown');s.add_argument('--settings',type=Path,default=Path('content/ai/settings.json'));s.add_argument('--output',type=Path,required=True)
 f=sub.add_parser('preflight');f.add_argument('--plan',type=Path,required=True);f.add_argument('--session',required=True);f.add_argument('--trace-root',type=Path,default=Path('content/ai/physical_traces'));f.add_argument('--binding',type=Path);f.add_argument('--settings',type=Path,default=Path('content/ai/settings.json'))
 g=sub.add_parser('guard');g.add_argument('--class',dest='cls',required=True);g.add_argument('--operation',required=True)
 z=sub.add_parser('finalize');z.add_argument('--plan',type=Path,required=True);z.add_argument('--session',required=True);z.add_argument('--labels',type=Path,required=True);z.add_argument('--quality',type=Path,required=True);z.add_argument('--output',type=Path,required=True)
 r=sub.add_parser('restore');r.add_argument('--binding',type=Path,required=True)
 a=p.parse_args()
 if a.cmd=='preflight':
  ok,reasons=preflight(a.plan,a.session,a.trace_root,a.binding,a.settings);print(('READY TO SHOOT' if ok else 'NOT READY')+f' SESSION {a.session}');[print('- '+r) for r in reasons];raise SystemExit(0 if ok else 2)
 elif a.cmd=='start':
  plan=load_plan(a.plan); root=a.trace_root/f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{a.session}_{uuid.uuid4().hex[:8]}'; root.mkdir(parents=True)
  backup=a.output.with_suffix('.settings.backup.json'); meta=bind(plan,a.session,root,a.source_commit); meta['settings_path']=str(a.settings); meta['settings_backup']=str(backup); meta['settings_sha256']=_set_trace(a.settings,root,backup); meta['status']='READY'; meta['trace_root']=str(root)
  (root/'collection_binding.json').write_text(json.dumps(meta,indent=2)+'\n'); a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(meta,indent=2)+'\n');print(f'STATUS: READY\nSESSION: {a.session}\nTRACE_ROOT: {root}\nPHYSICAL_TRACE_CAPTURE: ENABLED')
 elif a.cmd=='restore':
  b=json.loads(a.binding.read_text()); backup=Path(b['settings_backup']); target=Path(b['settings_path'])
  if not backup.exists(): raise FileNotFoundError(f'settings backup missing: {backup}; do not continue, recover it before restoring')
  shutil.copyfile(backup,target); print(f'RESTORED SETTINGS: {target}')
 elif a.cmd=='guard': guard_training(a.cls,a.operation);print('ALLOWED')
 else:
  from automation.physical_finalize import validate
  if a.output.exists(): raise FileExistsError(f'already finalized at {a.output}; preserve existing result and do not overwrite')
  res=validate(load_plan(a.plan),a.session,a.labels,a.quality);a.output.write_text(json.dumps(res,indent=2)+'\n');print('FINALIZED',a.session)
if __name__=='__main__':main()
