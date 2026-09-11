"""RESEARCH_ONLY source ablations over the recorded physical track snapshots."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from automation.physical_score_audit import source

def dist(a,b): return math.hypot(float(a.get('camera_x',0))-b['camera_x'],float(a.get('camera_y',0))-b['camera_y'])
def tracks_for(trace):
    by={}
    for stage in trace.get('stages',[]):
        for t in stage.get('tracks',[]) or []:
            if isinstance(t,dict) and t.get('track_id') is not None: by[int(t['track_id'])]=t
    return list(by.values())
def eligible(trace, gt):
    peak=float(trace['peak_ts']); rows=[]
    for t in tracks_for(trace):
        c=t.get('last_candidate') or {}
        onset=float(t.get('first_seen_ts',peak))-peak
        if onset < -0.08 or onset > 1.5: continue
        if not c: continue
        rows.append({'track':t,'candidate':c,'source':source(c),'distance':dist(t,gt),'onset_abs':abs(onset)})
    return rows
def choose(rows, mode):
    non=[r for r in rows if r['source']!='FAST_V2225']; fast=[r for r in rows if r['source']=='FAST_V2225']
    if mode=='exclude_fast': pool=non
    elif mode=='prefer_nonfast': pool=non or fast
    else: pool=rows
    return min(pool,key=lambda r:(r['onset_abs'],-float(r['track'].get('best_score',0)))) if pool else None
def run(root, out):
    out.mkdir(parents=True,exist_ok=False); rows=[]
    for path in sorted((root/'shots').glob('*/trace.json')):
        t=json.loads(path.read_text()); gp=path.parent/'ground_truth.json'
        if not gp.exists(): continue
        gt=json.loads(gp.read_text()); choices={}
        candidates=eligible(t,gt)
        for mode in ('current_track_semantics','exclude_fast','prefer_nonfast'):
            x=choose(candidates,'current' if mode=='current_track_semantics' else mode)
            choices[mode]=None if x is None else {'distance_px':x['distance'],'source':x['source'],'track_id':x['track']['track_id'],'score':x['track'].get('best_score')}
        rows.append({'shot_id':t['shot_id'],'choices':choices,'track_count':len(candidates)})
    metrics={}
    for mode in ('current_track_semantics','exclude_fast','prefer_nonfast'):
        vals=[r['choices'][mode]['distance_px'] for r in rows if r['choices'][mode]]
        metrics[mode]={'evaluated':len(vals),'hits':{str(k):sum(v<=k for v in vals) for k in (5,10,20,42)},'mean_error_px':sum(vals)/len(vals),'median_error_px':sorted(vals)[len(vals)//2],'p95_error_px':sorted(vals)[max(0,int(math.ceil(.95*len(vals)))-1)],'errors_gt_100_px':sum(v>100 for v in vals),'source_counts':{s:sum(bool(r['choices'][mode]) and r['choices'][mode]['source']==s for r in rows) for s in ('FAST_V2225','V26_VAULT','V2_OTHER','V1')}}
    (out/'ablation.json').write_text(json.dumps({'status':'RESEARCH_ONLY','semantics':'recorded track snapshots; no live policy change','metrics':metrics,'rows':rows},indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output)
if __name__=='__main__': main()
