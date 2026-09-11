"""Read-only score/source/rank audit for a reconstructed physical pool."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def pct(v,a):
    a=np.asarray(a,float); return float((np.sum(a<v)+.5*np.sum(a==v))/max(len(a),1))

def run(inp:Path,out:Path):
    d=json.loads(inp.read_text()); tracks=d['tracks']; by={}
    for x in tracks: by.setdefault(int(x['shot']),[]).append(x)
    rows=[]
    for e in d['events']:
        sid=int(e['shot']); pool=[x for x in by.get(sid,[]) if x['track'].get('eligible')]
        for p in e['pairs']:
            if p['role'] not in ('nearest_eligible','current_winner'): continue
            t=p['track']; score=float(t['best_score']); src=t.get('source','UNKNOWN')
            same=[q['track']['best_score'] for q in pool if q['track'].get('source','UNKNOWN')==src]
            allsrc=[q['track']['best_score'] for q in tracks if q['track'].get('source','UNKNOWN')==src]
            f=p.get('features',{})
            rows.append(dict(shot=sid,role=p['role'],track_id=t['track_id'],rank=t.get('final_rank'),source=src,best_score=score,current_score=t.get('current_score'),score_first=(p.get('association_history') or [{}])[0].get('candidate',{}).get('score'),score_last=(p.get('association_history') or [{}])[-1].get('candidate',{}).get('score'),source_history=t.get('source_history'),same_source_percentile=pct(score,same),session_source_percentile=pct(score,allsrc),features={k:f.get(k) for k in ('ring_affine_center_dark','ring_affine_compact','ring_affine_concentration','ring_affine_signed_contrast')}))
    (out/'score_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    # Counterfactual ranks over the exact eligible pool.
    methods=['raw','source_percentile','source_rank','common_compact','common_dark']
    result={m:[] for m in methods}
    for sid,pool in by.items():
        pool=[x for x in pool if x['track'].get('eligible')]
        for x in pool:
            t=x['track'];src=t.get('source','UNKNOWN'); same=[q['track']['best_score'] for q in pool if q['track'].get('source','UNKNOWN')==src]
            f=x.get('image_features',{}); x['_keys']={'raw':t['best_score'],'source_percentile':pct(t['best_score'],same),'source_rank':-sum(v>t['best_score'] for v in same),'common_compact':f.get('ring_affine_compact',-1e9),'common_dark':f.get('ring_affine_dark_contrast',-1e9)}
        for m in methods:
            order=sorted(pool,key=lambda x:x['_keys'][m],reverse=True); result[m].append(dict(shot=sid,top=[x['track']['track_id'] for x in order[:100]]))
    (out/'counterfactual_pools.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),eligible=sum(len([x for x in v if x['track'].get('eligible')]) for v in by.values()),sources={s:sum(1 for x in tracks if x['track'].get('source','UNKNOWN')==s) for s in sorted(set(x['track'].get('source','UNKNOWN') for x in tracks))})))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);run(a.input,a.output)
