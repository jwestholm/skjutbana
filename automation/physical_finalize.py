"""Postflight/label consistency gate for a planned research session."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def validate(plan,session,labels,quality):
 rows=[r for r in plan['rows'] if r['session']==session]; ls=json.loads(Path(labels).read_text()); qs=json.loads(Path(quality).read_text())
 if isinstance(qs,list): qs=next((q for q in qs if q.get('session')==session or q.get('session','').endswith(session)),qs[0] if qs else {})
 if ls.get('collection_plan_id') and ls['collection_plan_id']!=plan.get('collection_plan_id'):raise ValueError(f'collection plan mismatch: labels={ls["collection_plan_id"]} plan={plan.get("collection_plan_id")}; data is safe, use the original plan')
 if len({r.get('planned_physical_shot') for r in rows})!=len(rows):raise ValueError('duplicate planned shots')
 labels_list=ls.get('labels',[]);ids=[x.get('event_id') for x in labels_list];
 if len(ids)!=len(set(ids)):raise ValueError('duplicate label assignment')
 if any(x.get('status') in (None,'UNLABELED','AMBIGUOUS') for x in labels_list):raise ValueError('unresolved or AMBIGUOUS labels remain; correct labels before finalization')
 planned=[x.get('planned_physical_shot') for x in labels_list if x.get('status')=='PHYSICAL']
 if len(planned)!=len(set(planned)):raise ValueError('duplicate planned physical shot mapping; event ids must map one-to-one')
 if not qs.get('frame_completeness',qs.get('status')=='PASS'):raise ValueError('trace quality not complete')
 return dict(status='FINALIZED',session=session,labels=len(ls.get('labels',[])),physical_shots=sum(1 for x in labels_list if x.get('status')=='PHYSICAL'),non_physical_events=sum(1 for x in labels_list if x.get('status')=='NO_PHYSICAL_SHOT'),session_class=rows[0]['session_class'],trace_root=ls.get('trace_root'),recovered=bool(ls.get('recovered')),mapping=[{'planned_physical_shot':x.get('planned_physical_shot'),'event_id':x.get('event_id'),'status':x.get('status')} for x in labels_list])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--session',required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--quality',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();res=validate(json.loads(a.plan.read_text()),a.session,a.labels,a.quality);a.output.write_text(json.dumps(res,indent=2)+'\n');print(res)
