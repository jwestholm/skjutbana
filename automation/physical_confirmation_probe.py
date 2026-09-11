"""Oracle component probe; not a reconstruction of the complete live detector."""
import argparse,dataclasses,json
from pathlib import Path
import numpy as np
from automation.physical_session_audit import audit,distance
from src.engine.shot_fast_v2225 import local_confirm_candidates_v2225,FastConfigV2225


def run(root,mapping,output):
    output.mkdir(parents=True,exist_ok=False);rows=[];config=FastConfigV2225()
    for shot in audit(root,mapping)['shots']:
        gt=shot['ground_truth']
        if not gt:continue
        directory=root/'shots'/f"shot_{int(shot['shot_id']):08d}"
        trace=json.loads((directory/'trace.json').read_text())
        observed=next((s for s in trace['stages'] if s.get('candidates')),None)
        if observed is None:continue
        candidate=min(observed['candidates'],key=lambda c:distance(c,gt))
        frame_ts=candidate.get('timestamp',trace['peak_ts'])
        posts=[f for f in trace['frames'] if f['kind']=='post' and f['timestamp']>frame_ts+.018]
        history=[f for f in trace['frames'] if f['kind']=='pre_history']
        snapshot=next((f for f in trace['frames'] if f['kind']=='pre_snapshot'),None)
        choices=[('snapshot',snapshot),('earliest_history',min(history,key=lambda f:f['timestamp']) if history else None)]
        for label,pre in choices:
            if pre is None:continue
            pre_image=np.load(directory/pre['path'],allow_pickle=False);observations=[]
            for post in posts[:2]:
                confirmed,_=local_confirm_candidates_v2225(pre_image,np.load(directory/post['path'],allow_pickle=False),[candidate],frame_ts=post['timestamp'],config=config)
                observations.append({'post_path':post['path'],'post_ts':post['timestamp'],'confirmed':bool(confirmed),'candidate':confirmed[0] if confirmed else None})
            rows.append({'shot_id':shot['shot_id'],'pre_choice':label,'pre_path':pre['path'],'pre_ts':pre['timestamp'],
                         'candidate':candidate,'distance_px':distance(candidate,gt),'results':observations})
    result={'semantics':'Oracle nearest-retained candidate; default FastConfigV2225; first two eligible saved POSTs and two PRE choices. Not exact runtime input, scheduling, registration, or live-path-equivalent replay.',
            'config':dataclasses.asdict(config),'rows':rows}
    (output/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:print(row['shot_id'],row['pre_choice'],[r['confirmed'] for r in row['results']])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--mapping',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,json.loads(a.mapping.read_text()) if a.mapping else None,a.output)
