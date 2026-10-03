"""Prespecified bounded robust ranking structures over frozen four-session caches.

No GT, source score or absolute target location enters inference. Supervised
comparisons exclude whole sessions. Event-level acceptance is deliberately absent.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from automation.physical_ranking_families import (install_read_guard, load_inputs,
    family_totals, frozen_contributions, session_rows, tied_ranks, family)
from automation.physical_hypothesis_research import digest, row
from automation.physical_counterfactual_models import train_pairwise, universe
from automation.physical_counterfactual_research import SESSIONS, RADII, write_json
from automation.accuracy_verifier_research import summarize, predict, save_model


METHODS = {
    'FROZEN_PAIR_CLEAN':'Existing whole-session model control on the identical pool',
    'FAMILY_NORMALIZED':'Equal family mass after training-only scale normalization',
    'FAMILY_CAPPED':'Clip normalized family contributions to [-1,1] before equal combination',
    'RANK_MEAN':'Equal mean within-event family percentile ranks',
    'RANK_MEDIAN':'Median family rank resists one misleading family',
    'CONSENSUS':'Count top-quartile family support; mean rank breaks ties',
    'PARETO':'Deprioritize family-dominated candidates; preserve the entire pool',
    'TWO_STAGE':'Prioritize candidates top-quartile in two physical families; frozen physical score within tiers; no candidates discarded',
    'PAIR_FAMILY':'Whole-session pairwise refit on six family ranks rather than 132 columns',
    'TEMPORAL_SHAPE':'Replace temporal amplitude vote by localized persistent-step shape; onset may precede audio',
    'MOTION_RESIDUAL':'Add one bounded crop-half residual-survival context vote; no motion locator',
}


def rank_columns(values):
    return np.column_stack([tied_ranks(values[:,j]) for j in range(values.shape[1])])


def pareto_counts(values):
    """Number of dominating candidates. O(N^2 F), N<=200, no candidate removal."""
    return np.sum(np.all(values[None,:,:]>=values[:,None,:],axis=2) &
                  np.any(values[None,:,:]>values[:,None,:],axis=2),axis=1)


def robust_scores(values, families, method, *, raw_scores=None):
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty family evidence required')
    ranks=rank_columns(values)
    if method=='FAMILY_NORMALIZED':return values.mean(axis=1)
    if method=='FAMILY_CAPPED':return np.clip(values,-1,1).mean(axis=1)
    if method=='RANK_MEAN':return ranks.mean(axis=1)
    if method=='RANK_MEDIAN':return np.median(ranks,axis=1)
    if method=='CONSENSUS':return (ranks>=.75).sum(axis=1)+ranks.mean(axis=1)/(len(families)+1)
    if method=='PARETO':return -pareto_counts(values)+ranks.mean(axis=1)/(len(families)+1)
    if method=='TWO_STAGE':
        cols=[families.index(g) for g in ('new_change','temporal','morphology','localization')]
        priority=(ranks[:,cols]>=.75).sum(axis=1)>=2
        return priority.astype(float)+tied_ranks(raw_scores)/2
    raise ValueError('Unknown robust structure')


def load_steps(base,data,matrix):
    """Reuse accepted derived step cache; prove its frozen CURRENT outputs repeat.

    This is consistency with the old fitted artifacts, not a recovered raw-image
    hash. The original driver checked each original trace hash during extraction.
    """
    path=base/'change_step/step_features.npy'
    steps=np.load(path,allow_pickle=False)
    protocol=json.loads((base/'change_step/protocol.json').read_text())
    names=protocol['feature_names']
    if steps.shape!=(len(data['records']),len(names)) or not np.isfinite(steps).all():
        raise ValueError('Invalid derived step cache')
    prior=json.loads((base/'change_step/results.json').read_text())
    expected={r['event']:r['selected'] for r in prior['rows'] if r['method']=='PAIR_CHANGE_STEP' and r['policy']=='CURRENT'}
    merged=np.column_stack([matrix,steps]);verified=[];hashes={str(path):digest(path)}
    for meta in json.loads((base/'change_step/models.json').read_text()):
        if meta['method']!='PAIR_CHANGE_STEP':continue
        p=base/'change_step'/('PAIR_CHANGE_STEP_'+meta['held_out'])/'parameters.npz'
        if digest(p)!=meta['hashes']['parameters.npz']:raise ValueError('Step model hash mismatch')
        with np.load(p,allow_pickle=False) as m:
            model=dict(family='logistic',**{k:m[k] for k in ('mu','sd','beta')})
            score=predict(model,merged[:,meta['feature_indices']])
        for e in data['events']:
            if e['session']!=meta['held_out']:continue
            best=min(e['pools']['current'],key=lambda i:(-score[i],i))
            actual=best if data['records'][best]['ready'] else None
            wanted=None if expected[e['event']] is None else expected[e['event']]['index']
            if actual!=wanted:raise ValueError('Derived temporal cache does not repeat: '+e['event'])
            verified.append(e['event'])
        hashes[str(p)]=digest(p)
    if len(verified)!=len(data['events']):raise ValueError('Missing temporal validation event')
    return steps.astype(float),names,dict(input_hashes=hashes,matched_frozen_outputs=verified,
        source='Accepted September 10 derived step cache; no raw trace read',
        limitation='Original step array lacks a separate historical hash; frozen-model decision consistency verified and current hash recorded')


def shape_evidence(steps,names):
    """Step stability after onset, independent of onset sign relative to audio.

    The split itself was chosen from all causal PRE/POST samples in the accepted
    cache. Do not demand low PRE variation, after-audio onset or a shot interval.
    """
    groups=[]
    for suffix in ('step_explained','step_persistent_fraction','step_snr'):
        cols=[names.index('r'+str(r)+'_'+suffix) for r in (2,4,8,16)]
        groups.append(np.median(steps[:,cols],axis=1))
    return np.column_stack(groups)


def motion_residual_evidence(matrix,names):
    """Relative survival under cached crop-half compensation, not panel physics."""
    cols=[]
    for r in (2,4,8):
        for statistic in ('mean','inner_ring'):
            p=matrix[:,names.index(f'piecewise_clean_r{r}_{statistic}')]
            raw=matrix[:,names.index(f'pre_rms_clean_r{r}_{statistic}')]
            cols.append(p-raw)  # Difference of signed-log evidence, not calibrated displacement.
    return np.median(np.column_stack(cols),axis=1)


def fixed_pools(data,prior):
    hypotheses=json.loads((prior/'final/hypotheses.json').read_text())
    coverage=json.loads((prior/'final/retention_ledger.json').read_text())
    result={}
    for e in data['events']:
        pools=dict(CURRENT=list(e['pools']['current']),OBS32=hypotheses[e['event']]['available_pool'],
            COVERAGE48=coverage[e['event']]['coverage']['pool'])
        for ids in pools.values():
            if len(ids)>200 or len(set(ids))!=len(ids):raise ValueError('Fixed pool bound/uniqueness violation')
            if not set(e['pools']['current']).issubset(ids):raise ValueError('Existing coordinates lost')
            if any(not data['records'][i]['selectable'] for i in ids):raise ValueError('GT-only point in pool')
        result[e['event']]=pools
    return result


def summarize_rows(rows):
    baseline={r['event']:r for r in rows if r['method']=='CURRENT' and r['pool']=='CURRENT'}
    for r in rows:
        r['regressed_radii']=[str(t) for t in RADII if baseline[r['event']]['result'].get('hits',{}).get(str(t))
            and not r['result'].get('hits',{}).get(str(t))]
    result=[]
    for method,pool in sorted({(r['method'],r['pool']) for r in rows}):
        for session in SESSIONS:
            group=[r for r in rows if (r['method'],r['pool'],r['session'])==(method,pool,session)]
            if not group:continue
            result.append(dict(method=method,pool=pool,session=session,**summarize(group),
                errors_gt1000=sum((r['result'].get('error_px') or 0)>1000 for r in group),
                available_count=dict(mean=float(np.mean([r['available_count'] for r in group])),max=max(r['available_count'] for r in group)),
                mean_score_ms=float(np.mean([r['ranking_ms'] for r in group])),
                regressions=[dict(event=r['event'],radii=r['regressed_radii']) for r in group if r['regressed_radii']]))
    return result


def run(base,prior,output):
    output.mkdir(parents=True,exist_ok=False);accessed=install_read_guard(output)
    data,matrix,details,models,hashes=load_inputs(base,prior);records=data['records'];names=data['feature_names']
    steps,step_names,step_check=load_steps(base,data,matrix)
    pools=fixed_pools(data,prior)
    shape=shape_evidence(steps,step_names);motion=motion_residual_evidence(matrix,names)
    write_json(output/'protocol.json',dict(status='OFFLINE_RESEARCH_ONLY',command=sys.argv,sessions=SESSIONS,
        input_hashes=hashes,temporal_cache=step_check,methods=METHODS,pools='Exactly CURRENT, accepted OBS32, accepted COVERAGE48; no new retention',
        source_hashes={str(p):digest(p) for p in [Path(__file__),Path('automation/physical_ranking_families.py')]},
        fixed_constants=dict(family_clip=1.,consensus_percentile=.75,shortlist_families=2,pair_regularization=.05),
        promotion='No protected loss, no >1000px, per-session selected nonregression, P95 nonworsening, bounded cost; physical validation still required',
        limitations=['Camera-time causal cached features; worker-delivery equivalence unavailable',
                    'No new history feature used because POST_FIX reconstruction delivery is unreliable',
                    'Visual families are not independent sensors','No acceptance threshold; four negatives remain separate']))
    rows=[];score_tables={};model_meta=[]
    for e in data['events']:
        ids=pools[e['event']]['CURRENT'];order=[e['current_id']]+[i for i in ids if i!=e['current_id']]
        r=row(e,records,order,ids,'CURRENT',0.,0.);r.update(pool='CURRENT',available_count=len(ids));rows.append(r)
    for hold in SESSIONS:
        train=[s for s in SESSIONS if s!=hold];model=models[hold]
        started=time.perf_counter();contrib=frozen_contributions(matrix,model);raw=contrib.sum(axis=1)
        families,totals=family_totals(contrib,[names[j] for j in model['meta']['feature_indices']])
        ids,weights=session_rows(data,train);weights/=weights.sum()
        mu=np.average(totals[ids],axis=0,weights=weights)
        sd=np.maximum(.05,np.sqrt(np.average((totals[ids]-mu)**2,axis=0,weights=weights)))
        normalized=(totals-mu)/sd
        # Family training inputs use within-event ranks over the fixed observed
        # universe. GT determines training pairs only inside train_pairwise.
        family_train=np.zeros_like(totals)
        for e in data['events']:
            ids=[i for i in universe(e,records) if records[i]['ready']]
            family_train[ids]=rank_columns(totals[ids])
        pair=train_pairwise(data,family_train,train)
        hashes_model=save_model(pair,output/('PAIR_FAMILY_'+hold))
        setup_ms=(time.perf_counter()-started)*1000
        model_meta.append(dict(held_out=hold,train=train,families=families,mu=mu.tolist(),sd=sd.tolist(),
            pair_hashes=hashes_model,setup_ms=setup_ms,coefficients=pair['beta'].tolist()))
        for e in data['events']:
            if e['session']!=hold:continue
            for pool,available in pools[e['event']].items():
                ids=[i for i in available if records[i]['ready']]
                score_tables[e['event']+'_'+pool]={}
                for method in METHODS:
                    start=time.perf_counter();v=normalized[ids]
                    if method=='FROZEN_PAIR_CLEAN':scores=raw[ids]
                    elif method=='PAIR_FAMILY':scores=predict(pair,rank_columns(totals[ids]))
                    elif method=='TEMPORAL_SHAPE':
                        ranks=rank_columns(v);ranks[:,families.index('temporal')]=np.median(rank_columns(shape[ids]),axis=1)
                        scores=np.mean(ranks,axis=1)
                    elif method=='MOTION_RESIDUAL':
                        ranks=rank_columns(v)
                        scores=np.mean(np.column_stack([ranks,tied_ranks(motion[ids])]),axis=1)
                    else:scores=robust_scores(v,families,method,raw_scores=raw[ids])
                    order=[ids[j] for j in sorted(range(len(ids)),key=lambda j:(-scores[j],ids[j]))]
                    score_ms=(time.perf_counter()-start)*1000
                    r=row(e,records,order,ids,method,0.,score_ms)
                    r.update(pool=pool,available_count=len(available),available_oracle={str(t):any(records[i]['gt_distance'] is not None and records[i]['gt_distance']<=t for i in available) for t in RADII})
                    rows.append(r)
                    score_tables[e['event']+'_'+pool][method]=dict(indices=ids,scores=scores.tolist(),ordered=order)
            print(hold,e['event'],'ranked fixed pools',flush=True)
    summaries=summarize_rows(rows)
    write_json(output/'results.json',dict(rows=rows,summaries=summaries))
    write_json(output/'models.json',model_meta);write_json(output/'scores.json',score_tables)
    write_json(output/'input_accesses.json',sorted(accessed))
    for name,expected in hashes.items():
        if digest(Path(name))!=expected:raise ValueError('Input changed during run')
    for method in METHODS:
        group=[s for s in summaries if s['pool']=='CURRENT' and s['method']==method]
        print(method,'@42',sum(s['hits']['42'] for s in group),'tails1000',sum(s['errors_gt1000'] for s in group),
            'regressions',[r['event'] for s in group for r in s['regressions']],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('base','prior','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();run(args.base,args.prior,args.output)
