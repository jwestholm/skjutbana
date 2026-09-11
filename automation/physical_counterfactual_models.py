"""Prespecified four-session coordinate/retention and pairwise ranking ablations.

No live policy. New fits exclude the entire evaluated session and use only the
four allowed development sessions. GT is a training/evaluation target only.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import time

import numpy as np

from automation.accuracy_physical_dataset import json_read
from automation.accuracy_verifier_research import fit_model, predict, event_predictions, summarize, save_model
from automation.physical_counterfactual_research import SESSIONS, RADII, write_json, xy


def universe(event, records):
    ids = list(dict.fromkeys(i for name in ('current', 'history', 'legacy_precap', 'raw_plausible')
                            for i in event['pools'].get(name, [])))
    return [i for i in ids if records[i]['selectable']]


def frame_medoid_pool(event, kind):
    return list(dict.fromkeys(t[kind] for t in event['representations']))


def bounded_hypotheses(event, records, budget=32):
    current = list(event['pools']['current']); seen = set(current); extra = []
    # Track IDs give stable ordering; neither GT nor source score selects extras.
    tracks = sorted(event['representations'], key=lambda t:t['track_id'])
    for kind in ('medoid', 'confirmed', 'physical'):
        for t in tracks:
            index = t[kind]
            if index not in seen:
                seen.add(index); extra.append(index)
                if len(extra) == budget:
                    return current+extra
    return current+extra


def spatial_order(ids, records, event, score):
    x0, y0, width, height = event['context']['crop']
    cells = defaultdict(list)
    for index in ids:
        c = records[index]
        key = (min(3, max(0, int(4*(c['camera_y']-y0)/height))),
               min(3, max(0, int(4*(c['camera_x']-x0)/width))))
        cells[key].append(index)
    for values in cells.values():
        values.sort(key=lambda i:(-score(i), records[i]['camera_y'], records[i]['camera_x']))
    result = []; depth = 0
    while True:
        layer = [values[depth] for key,values in sorted(cells.items()) if len(values) > depth]
        if not layer:
            break
        # Every cell participates before any cell's next-depth candidate.
        result.extend(sorted(layer, key=lambda i:(-score(i), i))); depth += 1
    return result


def late_reserve(event, records, matrix, names, mode, budget=48, cap=200, audit=None):
    """Preserve existing eligible XY; spatial/physical late reserve within 200."""
    if budget < 1 or cap < 1:
        raise ValueError('Positive budgets required')
    current = list(event['pools']['current'])
    if audit is not None:
        audit.update(mode=mode, cap=cap, requested_budget=budget, decisions=[])
    if len(current) > cap:
        raise ValueError('Cannot preserve existing pool within cap')
    budget = min(budget, cap-len(current)); selected = list(current)
    if not budget:
        return selected
    candidates = [i for i in universe(event, records) if i not in set(current)]
    physical = lambda i: records[i]['physical_priority']
    if mode == 'physical':
        queues = [spatial_order(candidates, records, event, physical)]
    elif mode == 'weak_strata':
        # Fixed thirds protect weak/middle/strong change without an absolute
        # contrast cutoff or knowledge of the held-out impact coordinates.
        ordered = sorted(candidates, key=lambda i:(physical(i), i))
        queues = [spatial_order(part.tolist(), records, event, physical) for part in np.array_split(ordered, 3)]
    elif mode == 'clean_families':
        queues = []
        for family in ('persistent', 'pre_centered', 'piecewise'):
            cols = [names.index(family+'_clean_r4_'+key) for key in ('mean','inner_ring','noise')]
            score = lambda i: float(matrix[i, cols[0]]+matrix[i, cols[1]]-matrix[i, cols[2]])
            queues.append(spatial_order(candidates, records, event, score))
    else:
        raise ValueError('Unknown late reserve')
    cursor = [0]*len(queues)
    # Interleave weak-strength/evidence families, with global 8px deduplication.
    while len(selected)-len(current) < budget:
        changed = False
        for q, values in enumerate(queues):
            while cursor[q] < len(values):
                i = values[cursor[q]]; cursor[q] += 1
                neighbor = next((j for j in selected if math.dist(xy(records[i]), xy(records[j])) < 8), None)
                if neighbor is not None:
                    if audit is not None:
                        audit['decisions'].append(dict(index=i, queue=q, operation='spatial_suppression',
                            representative=neighbor, distance=math.dist(xy(records[i]),xy(records[neighbor])), threshold=8))
                    continue
                if audit is not None:
                    audit['decisions'].append(dict(index=i, queue=q, operation='late_reserve_retained'))
                selected.append(i); changed = True; break
            if len(selected)-len(current) == budget:
                break
        if not changed:
            break
    if audit is not None:
        examined={r['index'] for r in audit['decisions']}
        audit.update(selected=selected, actual_budget=budget, unexamined=[i for i in candidates if i not in examined],
                     stopped_by='budget_exhausted' if len(selected)-len(current)==budget else 'queues_exhausted')
    return selected


def training_groups(dataset, sessions):
    if not sessions or set(sessions)-set(SESSIONS):
        raise ValueError('Only four explicitly allowed development sessions')
    records = dataset['records']; groups = []
    for event in dataset['events']:
        if event['session'] not in sessions:
            continue
        ids = [i for i in universe(event, records) if records[i]['ready']]
        if event['truth']['state'] == 'NO_PHYSICAL_SHOT':
            positive, negative = [], ids
        elif event['truth']['state'] == 'SINGLE_IMPACT':
            positive = [i for i in ids if records[i]['gt_distance'] <= 10]
            # A fixed near-miss fallback trains ranking only when an actually
            # observed point is within 20px. No manually injected GT coordinate.
            if not positive and ids:
                nearest = min(ids, key=lambda i:records[i]['gt_distance'])
                if records[nearest]['gt_distance'] <= 20:
                    positive = [nearest]
            negative = [i for i in ids if records[i]['gt_distance'] > 42]
        else:
            continue
        if positive or negative:
            groups.append(dict(event=event['event'], session=event['session'], positive=positive, negative=negative))
    return groups


def train_pointwise(dataset, matrix, sessions):
    groups = training_groups(dataset, sessions); ids=[]; labels=[]; weights=[]
    session_counts = {s:sum(g['session']==s for g in groups) for s in sessions}
    for g in groups:
        for label, key in ((0, 'negative'), (1, 'positive')):
            for index in g[key]:
                ids.append(index); labels.append(label); weights.append(1/(session_counts[g['session']]*len(g[key])))
    labels=np.array(labels); weights=np.array(weights)
    for label in (0,1):
        if not np.any(labels==label):
            raise ValueError('Both training classes required')
        weights[labels==label] /= 2*weights[labels==label].sum()
    return fit_model(matrix[ids], labels, weights, 'logistic')


def train_pairwise(dataset, matrix, sessions):
    """Same regularized logistic optimizer family, observed within-event pairs."""
    start=time.perf_counter()
    groups=[g for g in training_groups(dataset,sessions) if g['positive'] and g['negative']]
    if not groups:
        raise ValueError('No observed positive/negative training pairs')
    counts={s:sum(g['session']==s for g in groups) for s in sessions}
    ids=[]; weight=[]
    for g in groups:
        for key in ('positive','negative'):
            ids.extend(g[key]); weight.extend([1/(counts[g['session']]*len(g[key]))]*len(g[key]))
    weight=np.array(weight); weight/=weight.sum()
    mu=np.average(matrix[ids],axis=0,weights=weight)
    sd=np.maximum(.05,np.sqrt(np.average((matrix[ids]-mu)**2,axis=0,weights=weight)))
    z=np.clip((matrix-mu)/sd,-8,8).astype(np.float64)
    differences=[]; weights=[]
    for g in groups:
        # Average actual near-impact representations gives each physical event
        # one target; distant candidate density does not give it more weight.
        positive=z[g['positive']].mean(axis=0)
        differences.extend(positive-z[g['negative']])
        weights.extend([1/(counts[g['session']]*len(g['negative']))]*len(g['negative']))
    design=np.array(differences); weights=np.array(weights); weights/=weights.sum()
    beta=np.zeros(matrix.shape[1]); regularization=.05
    lipschitz=.25*np.linalg.eigvalsh(design.T@(weights[:,None]*design))[-1]+regularization
    for _ in range(400):
        margin=np.clip(design@beta,-35,35)
        gradient=design.T@(weights*(-1/(1+np.exp(margin))))+regularization*beta
        beta-=gradient/lipschitz
    return dict(family='logistic',mu=mu,sd=sd,beta=np.r_[0.,beta],train_ms=(time.perf_counter()-start)*1000)


def model_columns(names, method):
    core=[i for i,n in enumerate(names) if not n.startswith(('persistent_','pre_rms_','pre_centered_','piecewise_','history_','representative_'))]
    clean=[i for i,n in enumerate(names) if n.startswith(('persistent_','pre_rms_','pre_centered_','piecewise_'))]
    history=[i for i,n in enumerate(names) if n.startswith(('history_','representative_'))]
    if method in ('POINT_CORE','PAIR_CORE'):
        return core
    if method=='PAIR_CLEAN':
        return core+clean
    if method=='PAIR_HISTORY':
        return core+clean+history
    raise ValueError(method)


def experiment_pools(event, records, matrix, names):
    start=time.perf_counter()
    pools=dict(CURRENT=event['pools']['current'])
    times=dict(CURRENT=0.)
    for kind in ('medoid','confirmed','physical'):
        t0=time.perf_counter(); pools['REP_'+kind.upper()]=frame_medoid_pool(event,kind)
        times['REP_'+kind.upper()]=(time.perf_counter()-t0)*1000
    t0=time.perf_counter(); pools['MULTI32']=bounded_hypotheses(event,records)
    times['MULTI32']=(time.perf_counter()-t0)*1000
    for mode in ('physical','weak_strata','clean_families'):
        t0=time.perf_counter(); key='LATE_'+mode.upper()
        pools[key]=late_reserve(event,records,matrix,names,mode)
        times[key]=(time.perf_counter()-t0)*1000
    return pools,times,(time.perf_counter()-start)*1000


def run(cache, output):
    output.mkdir(parents=True,exist_ok=False)
    dataset=json_read(cache/'dataset.json'); matrix=np.load(cache/'features.npy',allow_pickle=False)
    if tuple(dataset['sessions'])!=SESSIONS or any(e['session'] not in SESSIONS for e in dataset['events']):
        raise ValueError('Unexpected session')
    names=dataset['feature_names']; records=dataset['records']
    write_json(output/'protocol.json',dict(status='OFFLINE_ONLY',sessions=SESSIONS,
        training='Leave each whole development session out; only observed coordinates from other three sessions',
        label_rule='All observed <=10px positives; nearest <=20px fallback if no <=10px; >42px negatives',
        controls='CURRENT identity order, coordinate-only identity order, POINT_CORE, PAIR_CORE, then clean and history additions',
        retention='Current coordinates preserved, <=48 late reserve, <=200 total; no source-score or GT sorting',
        interpretation='Single recorded-cutoff evaluation, not live asynchronous end-to-end validation',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_hashes={n:hashlib.sha256((cache/n).read_bytes()).hexdigest() for n in ('dataset.json','features.npy')}))
    pool_cache={}; rows=[]; model_rows=[]
    for event in dataset['events']:
        pools,timings,total=experiment_pools(event,records,matrix,names)
        pool_cache[event['event']]=dict(pools=pools,timing_ms=timings,total_ms=total)
        # Hold track identity/rank fixed to isolate coordinate representation.
        for policy in ('CURRENT','REP_MEDOID','REP_CONFIRMED','REP_PHYSICAL'):
            ids=pools[policy]; scores=np.full(len(records),-1e9)
            for rank,index in enumerate(ids): scores[index]=-rank-1
            temp=dict(dataset,records=[dict(r,pool='current') for r in records],events=[dict(event,indices=ids)])
            row=event_predictions(temp,scores,[event['session']],threshold=-math.inf)[0]
            if policy=='CURRENT' and row['selected']['index']!=event['current_id']:
                raise AssertionError('CURRENT mismatch')
            row.update(method='IDENTITY',policy=policy,selection_ms=0.,retention_ms=timings[policy]); rows.append(row)
    model_cache={}
    for hold in SESSIONS:
        train=tuple(s for s in SESSIONS if s!=hold)
        for method in ('POINT_CORE','PAIR_CORE','PAIR_CLEAN','PAIR_HISTORY'):
            cols=model_columns(names,method); x=matrix[:,cols]
            model=(train_pointwise if method=='POINT_CORE' else train_pairwise)(dataset,x,train)
            hashes=save_model(model,output/(method+'_'+hold)); t0=time.perf_counter()
            scores=predict(model,x); all_score_ms=(time.perf_counter()-t0)*1000
            model_cache[hold,method]=(model,cols)
            model_rows.append(dict(held_out=hold,train=train,method=method,feature_names=[names[i] for i in cols],
                feature_indices=cols,train_ms=model['train_ms'],hashes=hashes,all_cache_score_ms=all_score_ms,
                status='OFFLINE_RESEARCH_ONLY'))
            for event in dataset['events']:
                if event['session']!=hold:continue
                cached=pool_cache[event['event']]
                for policy,ids in cached['pools'].items():
                    temp=dict(dataset,records=[dict(r,pool='current') for r in records],events=[dict(event,indices=ids)])
                    t0=time.perf_counter(); row=event_predictions(temp,scores,[hold])[0]
                    row.update(method=method,policy=policy,selection_ms=(time.perf_counter()-t0)*1000,
                               retention_ms=cached['timing_ms'][policy])
                    rows.append(row)
            print(hold,method,'trained',round(model['train_ms'],1),flush=True)
    summaries=[]
    pairs=sorted({(r['method'],r['policy']) for r in rows})
    for method,policy in pairs:
        for session in SESSIONS:
            selected=[r for r in rows if r['session']==session and r['method']==method and r['policy']==policy]
            summary=summarize(selected)
            summary.update(session=session,method=method,policy=policy,
                mean_retention_ms=float(np.mean([r['retention_ms'] for r in selected])),
                mean_selection_ms=float(np.mean([r['selection_ms'] for r in selected])))
            summaries.append(summary)
    baselines={e['event']: next(r for r in rows if r['event']==e['event'] and r['method']=='IDENTITY' and r['policy']=='CURRENT')
               for e in dataset['events']}
    for r in rows:
        b=baselines[r['event']]
        r['regressed_radii_vs_current']=[str(radius) for radius in RADII
            if b['result'].get('hits',{}).get(str(radius)) and not r['result'].get('hits',{}).get(str(radius))]
    write_json(output/'models.json',model_rows)
    write_json(output/'pools.json',pool_cache)
    write_json(output/'results.json',dict(summaries=summaries,rows=rows))
    for method,policy in pairs:
        scores=[s for s in summaries if s['method']==method and s['policy']==policy]
        print(method,policy,'selected',[s['hits'] for s in scores],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.cache,args.output)
