"""Frozen synthetic intervention: only the confirmation PRE image changes.

Detector proposal generation, track update, ranking and readiness are the real
project implementations. The two arms use the exact same generated proposals.
The contaminated arm's PRE already includes the new synthetic hole. This is a
controlled reference test, not a new ranker or a full audio/camera simulation.
"""
from __future__ import annotations
import argparse,contextlib,hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter
from automation.synthetic_track_research import prepare,execute,make_images,SCENES
from automation.registered_synthetic_forensics import backgrounds
from automation.track_survival_replay import metrics,distance
from src.engine.offline.registered_impact import prepare_pair,extract

DESIGN=dict(status='RESEARCH_ONLY_REFERENCE_INTERVENTION',development_seed=20260911,holdout_seed=20260912,stress_seed=20260913,
    arms=['CLEAN_CONFIRMATION_PRE','IMPACT_ALREADY_IN_CONFIRMATION_PRE'],scenes=list(SCENES),ranker='UNCHANGED_CURRENT',
    invariants=['identical clean generation PRE','same scene seed and proposal list','only local confirmation PRE differs'],
    limitation='controlled confirmation reference loss; detector PRE remains clean to isolate this single mechanism')
DESIGN_HASH=hashlib.sha256(json.dumps(DESIGN,sort_keys=True).encode()).hexdigest()


def run(root,output,count,split,frozen):
    if split!='development':
        if not frozen or json.loads(frozen.read_text())['design_hash']!=DESIGN_HASH:raise ValueError('Matching frozen development design required')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(design=DESIGN,design_hash=DESIGN_HASH,count=count,split=split,input_session=str(root),source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path('automation/synthetic_track_research.py'),Path('src/engine/offline/registered_impact.py'))})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');bg=backgrounds(root);prepare();rows=[]
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='shot-cv-v2224') as worker,open(output/'runtime.log','w') as log,contextlib.redirect_stdout(log):
        for i in range(count):
            scene=SCENES[i%len(SCENES)];seed=DESIGN[split+'_seed']+i;a,b=bg[i%len(bg)]
            pre,post,gt=make_images(a,scene,seed,b)
            _,contaminated,_=make_images(a,scene,seed,a)
            results=[];previous=None
            for arm,reference in zip(DESIGN['arms'],(pre,contaminated)):
                result=execute(pre,post,i+1,worker,crop=i%2==0,confirmation_pre=reference)
                proposals=[(c['camera_x'],c['camera_y'],c['score']) for c in result['proposal']]
                if previous is not None and proposals!=previous:raise AssertionError('Reference intervention changed generation inputs')
                previous=proposals;snap=result['snapshot'];tracks=snap['tracks']
                def near(cs):return min((distance(c,gt) for c in cs),default=None) if gt else None
                winner=next((r for r in tracks if r['selected']),None)
                pair=prepare_pair(reference,post)
                f=extract(pair,(gt['camera_x'],gt['camera_y'])) if gt else None
                # Truth-centered features are diagnostics; never ranker inputs.
                record=dict(arm=arm,ready=result['ready'],oracle=near(result['proposal']),tracked=near(tracks),locally_confirmed=near([r for r in tracks if r['local_confirmed']]),eligible=near([r for r in tracks if r['eligible']]),error=distance(winner,gt) if gt and winner and result['ready'] else None,source=winner['source'] if winner else None,track_count=len(tracks),gt_diagnostic_features=f)
                results.append(record)
                # Complete ledgers for reconstructability; generated and ignored.
                (output/f'{i:04}_{arm}.json').write_text(json.dumps(snap,separators=(',',':'))+'\n')
            rows.append(dict(index=i,scene=scene,truth=gt,arms=results))
    summary={}
    for arm in DESIGN['arms']:
        impact=[r for r in rows if r['truth']];no=[r for r in rows if not r['truth']]
        def get(r):return next(a for a in r['arms'] if a['arm']==arm)
        summary[arm]=dict(impact_count=len(impact),oracle=metrics([get(r)['oracle'] for r in impact],len(impact)),selected=metrics([get(r)['error'] for r in impact],len(impact)),funnel_at42={k:sum(get(r)[k] is not None and get(r)[k]<=42 for r in impact) for k in ('oracle','tracked','locally_confirmed','eligible')},no_impact_count=len(no),false_ready=sum(get(r)['ready'] for r in no),source_counts=dict(Counter(get(r)['source'] for r in impact)),by_scene={s:metrics([get(r)['error'] for r in impact if r['scene']==s],sum(r['scene']==s for r in impact)) for s in SCENES if s not in ('unchanged','false_audio')})
    (output/'results.json').write_text(json.dumps(dict(manifest=manifest,summary=summary,rows=rows),indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=34);p.add_argument('--split',choices=['development','holdout','stress'],default='development');p.add_argument('--frozen-manifest',type=Path);a=p.parse_args();run(a.root,a.output,a.count,a.split,a.frozen_manifest)
