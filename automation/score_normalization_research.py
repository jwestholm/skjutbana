"""Deterministic, replay-only score-scale normalization experiments."""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path
from automation.physical_score_audit import source, distance

METHODS = ("WITHIN_SOURCE_PERCENTILE", "SOURCE_ROBUST_Z", "CONFIRMATION_FIRST_PERCENTILE")

def gt_for(path):
    p=path.parent/'ground_truth.json'
    return json.loads(p.read_text()) if p.exists() else None

def load_session(root):
    rows=[]
    for path in sorted((root/'shots').glob('shot_*/trace.json')):
        trace=json.loads(path.read_text()); gt=gt_for(path)
        pool=trace.get('decision_input',{}).get('retained_candidates') or []
        if not pool: pool=next((s.get('candidates') for s in reversed(trace.get('stages',[])) if s.get('candidates')),[])
        deterministic=trace.get('decision_input',{}).get('deterministic_selection')
        rows.append({'shot_id':str(trace['shot_id']),'trace':trace,'gt':gt,'pool':pool,'deterministic':deterministic})
    return rows

def percentile(value, values):
    if len(values)<=1:return 1.0
    ordered=sorted(values); rank=sum(v<=value for v in ordered)-1
    return rank/(len(ordered)-1)

def robust_z(value, values):
    med=statistics.median(values); mad=statistics.median([abs(v-med) for v in values])
    return (value-med)/(1.4826*mad) if mad>1e-9 else (0.0 if value==med else (1.0 if value>med else -1.0))

def normalized(pool, method):
    groups={}
    for c in pool:
        groups.setdefault(source(c),[]).append(float(c.get('score',0.0) or 0.0))
    def key(item):
        i,c=item; vals=groups[source(c)]; raw=float(c.get('score',0.0) or 0.0)
        if method=='WITHIN_SOURCE_PERCENTILE': primary=percentile(raw,vals)
        elif method=='SOURCE_ROBUST_Z': primary=robust_z(raw,vals)
        else:
            confirmed=bool(c.get('v2225_local_confirm')) or any(k in c for k in ('v2225_confirm_center_abs','v2225_confirm_darkening','v2225_confirm_compact'))
            primary=(2.0 if confirmed else 0.0)+percentile(raw,vals)
        return (primary,raw,-i)
    return max(enumerate(pool),key=key)

def role_distance(row, candidate): return distance(candidate,row['gt']) if candidate and row['gt'] else None

def evaluate(rows):
    selectors={m:[] for m in METHODS}; baseline=[]; mismatch=[]
    for row in rows:
        pool=row['pool']; gt=row['gt'];
        if not gt: continue
        winner=min(pool,key=lambda c: distance(c,gt)) if pool else None
        deterministic=min(pool,key=lambda c: distance(c,row['deterministic'])) if pool and row['deterministic'] else None
        gt_d=distance(winner,gt); det_d=distance(deterministic,gt)
        if gt_d is not None and gt_d<=42 and (det_d is None or det_d>42):
            gs=float((winner or {}).get('score',0) or 0); ds=float((deterministic or {}).get('score',0) or 0)
            strong=source(winner)!=source(deterministic) and ds>=30 and gs<10
            mismatch.append({'shot_id':row['shot_id'],'classification':'STRONG_CROSS_SOURCE' if strong else 'AMBIGUOUS_OTHER',
                'gt_source':source(winner),'winner_source':source(deterministic),'gt_score':gs,'winner_score':ds,
                'gt_distance_px':gt_d,'winner_distance_px':det_d})
        baseline.append({'shot_id':row['shot_id'],'distance_px':det_d})
        for method in METHODS:
            i,c=normalized(pool,method) if pool else (None,None)
            selectors[method].append({'shot_id':row['shot_id'],'distance_px':role_distance(row,c),'candidate':c,'index':None if i is None else i+1})
    return selectors,baseline,mismatch

def metrics(values):
    errors=[v['distance_px'] for v in values if v['distance_px'] is not None]; s=sorted(errors)
    p95=s[max(0,math.ceil(.95*len(s))-1)] if s else None
    return {'shots':len(values),'evaluated':len(errors),'accuracy':{str(r):sum(e<=r for e in errors)/len(errors) if errors else None for r in (5,10,20,42)},
      'mean_error_px':statistics.mean(errors) if errors else None,'median_error_px':statistics.median(errors) if errors else None,'p95_error_px':p95,'errors_gt_100_px':sum(e>100 for e in errors)}

def run(roots):
    sessions={}; aggregate=[]
    for root in roots:
        rows=load_session(root); selectors,baseline,mismatch=evaluate(rows)
        sessions[str(root)]={'shot_count':len(rows),'methods':{m:metrics(v) for m,v in selectors.items()},'mismatch_analysis':mismatch,
          'per_shot':{m:[{'shot_id':x['shot_id'],'distance_px':x['distance_px'],'retained_position':x['index'],'source':source(x['candidate']) if x['candidate'] else None,'score':x['candidate'].get('score') if x['candidate'] else None} for x in vals] for m,vals in selectors.items()}}
        aggregate.extend(mismatch)
    return {'schema':'score-normalization-research-1','status':'RESEARCH_ONLY','methods':METHODS,'sessions':sessions,
      'selection_loss_mismatch_summary':{'losses_with_gt_in_retained_42px':len(aggregate),'strong_cross_source':sum(x['classification']=='STRONG_CROSS_SOURCE' for x in aggregate),'ambiguous_other':sum(x['classification']=='AMBIGUOUS_OTHER' for x in aggregate)},
      'limitations':['Replay uses frozen retained pools and labels only for evaluation.','No method changes live behavior or the frozen confirmation selector.','Small physical samples are development evidence, not validation.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(); report=run(a.root);a.output.mkdir(parents=True,exist_ok=False);(a.output/'score_normalization_research.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Score normalization research (RESEARCH_ONLY)','', '| Session | Method | Accuracy @42 | Mean px | Median px | P95 px | >100 px |','|---|---|---:|---:|---:|---:|---:|']
    for session,data in report['sessions'].items():
      for method,m in data['methods'].items(): lines.append(f"| {session} | {method} | {m['accuracy']['42']} | {m['mean_error_px']} | {m['median_error_px']} | {m['p95_error_px']} | {m['errors_gt_100_px']} |")
    lines += ['',f"Mismatch summary: `{report['selection_loss_mismatch_summary']}`",'','These are fixed replay selectors only; no live authority was changed.'];(a.output/'score_normalization_research.md').write_text('\n'.join(lines)+'\n');print(a.output)
if __name__=='__main__':main()
