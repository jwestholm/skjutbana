"""One read-only temporal-reference intervention; no ranker is selected or fitted.

Oldest recorded PRE is chosen identically for every event without labels. It is
not asserted clean, safe for live ownership, or equivalent to immediate PRE.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from src.engine.offline.registered_impact import prepare_pair,extract
from automation.registered_impact_research import montage


def run(features,output):
    output.mkdir(parents=True,exist_ok=False);report=json.loads(features.read_text());root=Path(report['input_session']);events=[]
    for e in report['events']:
        directory=root/'shots'/f'shot_{e["shot"]:08d}';t=json.loads((directory/'trace.json').read_text())
        ref=min((f for f in t['frames'] if f['kind']=='pre_history'),key=lambda f:f['timestamp'])
        pre=np.load(directory/ref['path'],mmap_mode='r');post=np.load(directory/e['post']['path'],mmap_mode='r')
        active=[r for r in report['tracks'] if r['shot']==e['shot']];x0,y0=e['origin'];x1=min(pre.shape[1],int(max(r['track']['camera_x'] for r in active))+41);y1=min(pre.shape[0],int(max(r['track']['camera_y'] for r in active))+41)
        pair=prepare_pair(np.array(pre[y0:y1,x0:x1]),np.array(post[y0:y1,x0:x1]),origin=(x0,y0))
        tracks=[dict(track_id=r['track']['track_id'],features=extract(pair,(r['track']['camera_x'],r['track']['camera_y']))) for r in active]
        panels=[];pairs=[]
        for r in e['pairs']:
            f,p=extract(pair,(r['track']['camera_x'],r['track']['camera_y']),return_patches=True)
            pairs.append(dict(role=r['role'],label=r['label'],track_id=r['track']['track_id'],features=f,original_features=r['features']))
            panels.append((r['role']+'\noldest recorded PRE',p))
        g=json.loads((directory/'ground_truth.json').read_text());gf,gp=extract(pair,(g['camera_x'],g['camera_y']),return_patches=True);panels.append(('GT diagnostic ONLY',gp))
        montage(panels,output/f'shot_{e["shot"]:02}.png')
        events.append(dict(shot=e['shot'],reference=ref,reference_relative_to_peak=ref['timestamp']-t['peak_ts'],pairs=pairs,tracks=tracks,gt_diagnostic_features=gf,registration=pair.registration))
    (output/'reference_counterfactual.json').write_text(json.dumps(dict(status='RESEARCH_ONLY_FEATURE_INTERVENTION_NO_RANKER',events=events),indent=2)+'\n')
    print(json.dumps(dict(events=len(events),tracks=sum(len(e['tracks']) for e in events))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.features,a.output)
