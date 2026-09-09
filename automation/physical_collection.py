"""Safe research collection orchestration and validation helpers."""
from __future__ import annotations
import argparse,json,uuid
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
 if session=='S03' and any(r.get('session_class')!='VALIDATION_UNTOUCHED' for r in rows):raise ValueError('validation binding mismatch')
 meta=dict(collection_plan_id=plan.get('plan_id','local-plan'),planned_session_id=session,session_class=rows[0]['session_class'],created_at=datetime.now(timezone.utc).isoformat(),source_commit=source_commit,trace_root=str(trace_root),rows=rows)
 return meta
def guard_training(session_class,operation):
 if session_class=='VALIDATION_UNTOUCHED' and operation in ('train','tune','fit','select_config'):
  raise PermissionError('validation data is evaluation-only; training/tuning refused')
 return True
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 s=sub.add_parser('start');s.add_argument('--plan',type=Path,required=True);s.add_argument('--session',required=True);s.add_argument('--trace-root',type=Path,default=Path('content/ai/physical_traces'));s.add_argument('--source-commit',default='unknown');s.add_argument('--output',type=Path,required=True)
 g=sub.add_parser('guard');g.add_argument('--class',dest='cls',required=True);g.add_argument('--operation',required=True)
 a=p.parse_args()
 if a.cmd=='start':
  plan=load_plan(a.plan);meta=bind(plan,a.session,a.trace_root,a.source_commit);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(dict(status='PREPARED',session=a.session,shots=len(meta['rows']),session_class=meta['session_class'])))
 else: guard_training(a.cls,a.operation);print('ALLOWED')
if __name__=='__main__':main()
