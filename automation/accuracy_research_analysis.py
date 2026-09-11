"""Physical cohort metrics, spatial null controls and session bootstrap; no detector tuning."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from automation.accuracy_physical_dataset import json_read, load_context
from automation.accuracy_verifier_research import PRIMARY, event_predictions, predict, summarize
from src.engine.offline.accuracy_verifier import features, proposals
from src.engine.offline.physical_event_truth import score_event


def baseline_rows(dataset):
    rows=[]
    for event in dataset['events']:
        candidates=[dataset['records'][i] for i in event['indices'] if dataset['records'][i]['pool']=='current']
        distances=[c['gt_distance'] for c in candidates if c['gt_distance'] is not None]
        rows.append(dict(event=event['event'],session=event['session'],truth=event['truth'],
                         result=score_event(event['truth'],event['current']),selected=event['current'],
                         oracle={str(r):any(d<=r for d in distances) for r in (5,10,20,42)},
                         ranks={str(r):None for r in (5,10,20,42)},candidate_count=len(candidates),
                         unready_at_recorded_decision=False,recorded_emission=event['recorded_emission']))
    return rows


def spatial_null(dataset, repetitions=500):
    """Random spatial coverage is a null diagnostic on real labels, never validation."""
    rng=np.random.default_rng(41019);counts={s:np.zeros((repetitions,4),int) for s in dataset['sessions']}
    observed={s:np.zeros(4,int) for s in dataset['sessions']};radii=np.array([5,10,20,42])
    for event in dataset['events']:
        if event['truth']['state']!='SINGLE_IMPACT':continue
        p=Path(event['trace_path']);trace=json_read(p);context,_=load_context(p,trace,dataset['reference'])
        valid=context.roi>0;valid[:25]=False;valid[-25:]=False;valid[:,:25]=False;valid[:,-25:]=False
        ys,xs=np.where(valid);gt=event['truth']['impacts'][0]
        candidates=[dataset['records'][i] for i in event['indices'] if dataset['records'][i]['pool']=='expanded']
        observed[event['session']]+=np.any(np.array([c['gt_distance'] for c in candidates])[:,None]<=radii,axis=0)
        sample=rng.integers(0,len(xs),size=(repetitions,len(candidates)))
        distances=np.hypot(xs[sample]+context.origin[0]-gt['camera_x'],ys[sample]+context.origin[1]-gt['camera_y'])
        counts[event['session']]+=np.any(distances[:,:,None]<=radii,axis=1)
    return {s:dict(observed_expansion_hits=observed[s].tolist(),spatial_null_mean_hits=counts[s].mean(axis=0).tolist(),
                   spatial_null_95_interval=np.quantile(counts[s],(.025,.975),axis=0).tolist(),
                   repetitions=repetitions,meaning='Same count of spatially random ROI points; diagnostic only, not a detector result') for s in counts}


def bootstrap(before,after,sessions,repetitions=10000):
    paired=[]
    for session in sessions:
        a={r['event']:r for r in before if r['session']==session and r['truth']['state']=='SINGLE_IMPACT'}
        b={r['event']:r for r in after if r['session']==session and r['truth']['state']=='SINGLE_IMPACT'}
        keys=sorted(a.keys()&b.keys())
        paired.append((len(keys),sum(int(b[k]['result']['hits']['42'])-int(a[k]['result']['hits']['42']) for k in keys)))
    rng=np.random.default_rng(41020);draw=rng.integers(0,len(sessions),(repetitions,len(sessions)))
    pairs=np.array(paired);improvement=pairs[draw,1].sum(axis=1)/pairs[draw,0].sum(axis=1)
    return dict(sessions=sessions,paired_session_count_and_hit_gain=paired,
                observed_accuracy_gain=sum(v for n,v in paired)/sum(n for n,v in paired),
                percentile95=np.quantile(improvement,(.025,.975)).tolist(),repetitions=repetitions,
                caveat='Few sessions, overlapping training folds and development model selection; descriptive uncertainty, not independent validation')


def run(root,output):
    output.mkdir(parents=True,exist_ok=False)
    dataset=json_read(root/'dataset_history_early/dataset.json')
    matrix=np.load(root/'dataset_history_early/features.npy',allow_pickle=False)
    result=json_read(root/'verifier_history_early/results.json')['families'][0]
    before=baseline_rows(dataset);after=[r for fold in result['folds'] for r in fold['rows']]+result['challenge'][0]['rows']
    baseline={s:summarize([r for r in before if r['session']==s]) for s in dataset['sessions']}
    selected={s:summarize([r for r in after if r['session']==s]) for s in dataset['sessions']}
    for summary in baseline.values():summary.pop('topk42')
    # Benchmark one existing development event, selected by session/id, never GT.
    event=next(e for e in dataset['events'] if e['event']=='S01:1');p=Path(event['trace_path'])
    start=time.perf_counter();context,_=load_context(p,json_read(p),dataset['reference']);context_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter();positions=proposals(context,256);proposal_ms=(time.perf_counter()-start)*1000
    rows=[dataset['records'][i] for i in event['indices'] if dataset['records'][i]['pool']=='current']
    start=time.perf_counter();vectors=np.stack([features(context,(r['camera_x'],r['camera_y']))[1] for r in rows]);feature_ms=(time.perf_counter()-start)*1000
    params=np.load(root/'verifier_history_early/logistic_model/parameters.npz',allow_pickle=False)
    model=dict(family='logistic',**{k:params[k] for k in params.files})
    start=time.perf_counter();predict(model,vectors);predict_ms=(time.perf_counter()-start)*1000
    runtime=dict(event=event['event'],eligible_candidates=len(rows),extra_candidates=len(positions),context_ms=context_ms,
                 proposals_ms=proposal_ms,features_ms=feature_ms,predict_ms=predict_ms,
                 process_max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                 limitation='Offline Python extraction with disk IO and registration; not optimized live latency')
    names=dataset['feature_names'];coef=model['beta'][1:]
    coefficients=[dict(feature=names[i],standardized_coefficient=float(coef[i])) for i in np.argsort(-abs(coef))[:15]]
    report=dict(baseline_selected=baseline,research_selected=selected,bootstrap_primary=bootstrap(before,after,list(PRIMARY)),
                bootstrap_including_development_challenge=bootstrap(before,after,[*PRIMARY,'S02']),
                spatial_null=spatial_null(dataset),runtime_benchmark=runtime,top_coefficients=coefficients,
                timestamp_and_scope='Research scores one decision at its recorded cutoff; no full async/known-hole emission replay',
                actual_emission_unavailable=[e['event'] for e in dataset['events'] if e['recorded_emission'] is None],
                extraction_runtime_by_session={s:{key:dict(mean=float(np.mean([e['timing_ms'][key] for e in dataset['events'] if e['session']==s])),
                                                           max=float(max(e['timing_ms'][key] for e in dataset['events'] if e['session']==s)))
                                                   for key in ('context','proposals','features')} for s in dataset['sessions']})
    (output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('bootstrap_primary','spatial_null','runtime_benchmark')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.output)
