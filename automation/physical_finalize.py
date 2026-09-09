"""Postflight/label consistency gate for a planned research session."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def validate(plan,session,labels,quality):
 rows=[r for r in plan['rows'] if r['session']==session]; ls=json.loads(Path(labels).read_text()); qs=json.loads(Path(quality).read_text())
 if len({r.get('planned_physical_shot') for r in rows})!=len(rows):raise ValueError('duplicate planned shots')
 ids=[x.get('event_id') for x in ls.get('labels',[])];
 if len(ids)!=len(set(ids)):raise ValueError('duplicate label assignment')
 if any(x.get('status') in (None,'UNLABELED','AMBIGUOUS') for x in ls.get('labels',[])):raise ValueError('unresolved labels remain')
 if not qs.get('frame_completeness',qs.get('status')=='PASS'):raise ValueError('trace quality not complete')
 return dict(status='FINALIZED',session=session,labels=len(ls.get('labels',[])),session_class=rows[0]['session_class'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--session',required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--quality',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();res=validate(json.loads(a.plan.read_text()),a.session,a.labels,a.quality);a.output.write_text(json.dumps(res,indent=2)+'\n');print(res)
