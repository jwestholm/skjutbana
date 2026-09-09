"""Generate a deliberately balanced future physical-capture manifest."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path

DEFAULT_CATEGORIES=('light_flat','dark_flat','low_contrast','printed_line','edge_or_boundary','old_hole_nearby','texture','repeat_grouping')
def build(sessions,shots):
 rows=[]; n=0
 for s in range(1,sessions+1):
  for i in range(shots):
   cat=DEFAULT_CATEGORIES[(n)%len(DEFAULT_CATEGORIES)];n+=1
   cls='VALIDATION_UNTOUCHED' if s==sessions else 'DEVELOPMENT'
   rows.append(dict(session=f'S{ s:02d}',planned_physical_shot=i+1,category=cat,session_class=cls,notes='label immediately after trace finalization',actual_event_id=None,gt_status='UNLABELED',trace_health=None))
 base=dict(semantics='10 physical shots PLUS separately labelled no-impact audio events',categories=DEFAULT_CATEGORIES,sessions=sessions,shots_per_session=shots,rows=rows)
 base['collection_plan_id']=hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest()[:16];base['status']='PLANNED_ONLY';return base
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sessions',type=int,default=3);p.add_argument('--shots-per-session',type=int,default=10);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(build(a.sessions,a.shots_per_session),indent=2)+'\n');print(a.output)
