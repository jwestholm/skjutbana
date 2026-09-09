"""Recreate saved synthetic imagery and compare exact winners with local residuals.

Reused historical holdout cases are retrospective DEVELOPMENT for this analysis.
No new ranking or source policy is installed or fitted.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from automation.synthetic_track_research import make_images,FROZEN_POLICY
from automation.track_survival_replay import distance
from automation.registered_impact_research import montage
from src.engine.offline.track_replay import current_exact_replay
from src.engine.offline.registered_impact import prepare_pair,extract


def backgrounds(root):
    result=[]
    for p in sorted((root/'shots').glob('*/trace.json')):
        t=json.loads(p.read_text());fs=sorted([f for f in t['frames'] if f['kind']=='pre_history' and t['peak_ts']-.32<=f['timestamp']<t['peak_ts']],key=lambda f:f['timestamp'])
        result.append(tuple(np.array(np.load(p.parent/f['path'],mmap_mode='r')[1030:1510,1800:2440]) for f in (fs[0],fs[-1])))
    return result


def run(root,runs,output):
    output.mkdir(parents=True,exist_ok=False);bg=backgrounds(root);rows=[];examples=[]
    for run_path in runs:
        saved=json.loads((run_path/'results.json').read_text());seed=FROZEN_POLICY[saved['manifest']['split']+'_seed']
        for r in saved['rows']:
            # All 102 development, all 170 old holdout, all 60 stress negatives.
            if 'stress' in run_path.name and r['truth']:continue
            i=r['index'];pre,post,gt=make_images(*[bg[i%len(bg)][0],r['scene'],seed+i],post_background=bg[i%len(bg)][1])
            if gt!=r['truth']:raise AssertionError('Synthetic truth reconstruction mismatch')
            snap=json.loads((run_path/f'track_{i:04}.json').read_text());winner=current_exact_replay(snap)
            expected=r['choices']['CURRENT']['track_id']
            if (winner['track_id'] if winner else None)!=expected:raise AssertionError('Saved winner mismatch')
            pair=prepare_pair(pre,post);eligible=[t for t in snap['tracks'] if t['eligible']]
            near=min(eligible,key=lambda t:distance(t,gt)) if eligible and gt else None
            records=[];panels=[]
            roles=[('nearest_eligible',near),('current_winner',winner)]
            if near and winner and near['track_id']==winner['track_id']:roles=roles[:1]
            for role,t in roles:
                if t is None:continue
                try:
                    f,patches=extract(pair,(t['camera_x'],t['camera_y']),return_patches=True)
                    unavailable=None
                except ValueError as exc:
                    if str(exc)!='Candidate patch crosses recorded crop boundary':raise
                    f=None;patches=None;unavailable=str(exc)
                label='synthetic_true' if role=='nearest_eligible' and distance(t,gt)<=42 else ('synthetic_false' if role=='current_winner' and (gt is None or distance(t,gt)>42) else 'no_oracle_nearest')
                rec=dict(run=run_path.name,index=i,scene=r['scene'],role=role,label=label,gt_distance=distance(t,gt) if gt else None,features=f,unavailable_reason=unavailable,track=t)
                if label=='synthetic_false' and not r['ready']:rec['label']='synthetic_nonready_winner'
                rec['event_ready']=r['ready'];rec['has_impact']=gt is not None
                records.append(rec);examples.append(rec)
                if patches is not None:panels.append((f'{role}\n{t["source"]} rank {t["final_rank"]}',patches))
            if gt:
                gf,gp=extract(pair,(gt['camera_x'],gt['camera_y']),return_patches=True);panels.append(('GT diagnostic ONLY',gp))
            else:gf=None
            signature=gt is not None and r['oracle'] is not None and r['oracle']<=42 and r['choices']['CURRENT']['error'] is not None and r['choices']['CURRENT']['error']>42
            if signature or (gt is None and r['ready'] and i<170):montage(panels,output/f'{run_path.name}_{i:04}.png')
            rows.append(dict(run=run_path.name,index=i,scene=r['scene'],truth=gt,ready=r['ready'],signature=signature,physical_signature=signature and near is not None and near['final_rank']>8 and winner['source']=='FAST',registration=pair.registration,global_offset=pair.global_offset,examples=records,gt_diagnostic_features=gf))
    (output/'synthetic_features.json').write_text(json.dumps(dict(status='RETROSPECTIVE_DEVELOPMENT_FORENSICS',examples=examples,rows=rows),indent=2)+'\n')
    print(json.dumps(dict(events=len(rows),examples=len(examples),signature=sum(r['signature'] for r in rows),no_impact=sum(r['truth'] is None for r in rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.runs,a.output)
