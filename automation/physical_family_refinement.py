"""One justified refinement: fit after duplicate/correlation-family collapse.

The first family experiment inherited a frozen model's redundant-column weights.
This intervention changes that implicit weighting before fitting. Fixed grouping
criteria; no parameter sweep. Ablations/jackknife are stability diagnostics only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from automation.physical_ranking_families import (load_inputs,install_read_guard,
    family,session_rows,tied_ranks,weighted_correlation)
from automation.physical_family_rank_research import fixed_pools,summarize_rows,rank_columns
from automation.physical_counterfactual_models import train_pairwise,model_columns
from automation.physical_counterfactual_research import SESSIONS,write_json
from automation.physical_hypothesis_research import row,digest
from automation.accuracy_verifier_research import predict,save_model


def correlation_groups(x,names,weights,mode):
    """Complete-link signed correlation groups within a semantic family.

    Every pair must exceed the fixed criterion; a chain cannot collapse unrelated
    endpoints. Grouping is fitted on training sessions only, without labels.
    """
    if mode not in ('unique','grouped'):raise ValueError('Unknown grouping')
    rank=np.column_stack([tied_ranks(x[:,j]) for j in range(x.shape[1])])
    corr=weighted_correlation(rank,weights);groups=[]
    for j,n in enumerate(names):
        assigned=False
        for group in groups:
            if family(n)!=family(names[group[0]]):continue
            compatible=all(np.array_equal(x[:,j],x[:,k]) for k in group) if mode=='unique' else all(abs(corr[j,k])>=.95 for k in group)
            if compatible:group.append(j);assigned=True;break
        if not assigned:groups.append([j])
    return groups,corr


def grouped_design(data,matrix,names,train,mode):
    ids,weight=session_rows(data,train);weight/=weight.sum()
    groups,corr=correlation_groups(matrix[ids],names,weight,mode)
    mu=np.average(matrix[ids],axis=0,weights=weight)
    sd=np.maximum(.05,np.sqrt(np.average((matrix[ids]-mu)**2,axis=0,weights=weight)))
    z=np.clip((matrix-mu)/sd,-8,8)
    signs=[[1 if corr[g[0],j]>=0 else -1 for j in g] for g in groups]
    x=np.column_stack([np.mean(z[:,g]*sign,axis=1) for g,sign in zip(groups,signs)])
    meta=dict(groups=[[names[j] for j in g] for g in groups],indices=groups,signs=signs,
        mu=mu.tolist(),sd=sd.tolist(),families=[family(names[g[0]]) for g in groups],
        training_sessions=train,mode=mode,threshold=.95 if mode=='grouped' else 'exact duplicate')
    return x,meta


def direct_families(matrix,names,pixel_persistence=False):
    """Four explicit physical measurements; no fitted texture/noise coefficient.

    PRE normalization already lives in the cached residual features. It is context,
    never a standalone negative vote. Derivative columns share one family vote.
    """
    specifications={
        'change_strength':[f'r{r}_abs_late' for r in (2,4,8)],
        'temporal_consistency':[f'r{r}_{key}' for r in (2,4,8) for key in ('early_late_agreement','spatial_consistency')],
        'localization':[f'r{r}_center_ring_ratio' for r in (2,4,8)],
        'registered_residual':[f'{channel}_clean_r{r}_inner_ring' for channel in ('persistent','pre_centered','piecewise') for r in (2,4,8)],
    }
    if pixel_persistence:
        # Mean signed inner/ring contrast can cancel a mixed bright/dark change.
        # This existing feature measures temporal polarity consistency per pixel
        # before spatial averaging; it never requires onset after recorded audio.
        specifications['temporal_consistency']=[f'r{r}_spatial_consistency' for r in (2,4,8)]
    return list(specifications),np.column_stack([np.median(rank_columns(matrix[:,[names.index(n) for n in group]]),axis=1)
        for group in specifications.values()])


def run(base,prior,output,direct_only=False,pixel_persistence=False):
    output.mkdir(parents=True,exist_ok=False);accessed=install_read_guard(output)
    data,matrix,details,models,hashes=load_inputs(base,prior);records=data['records']
    cols=model_columns(data['feature_names'],'PAIR_CLEAN');matrix=matrix[:,cols];names=[data['feature_names'][j] for j in cols]
    pools=fixed_pools(data,prior);rows=[];fitted=[];jack=[]
    write_json(output/'protocol.json',dict(status='ONE_OFFLINE_REFINEMENT_NO_PROMOTION',command=sys.argv,
        input_hashes=hashes,source_sha256=digest(Path(__file__)),
        hypothesis='Duplicate visual derivatives change regularization weight; collapse before fitting, then test whole-session stability',
        fixed_constants=dict(correlation=.95,regularization=.05),
        fit='Whole-session exclusion; source identity/GT/absolute XY absent from input columns',
        jackknife='Omit each remaining training session once, retain the same held-out test session; descriptive only',
        direct_only=direct_only,pixel_persistence=pixel_persistence,
        direct_hypothesis='Fixed equal four physical-family ranks; optional one temporal refinement uses per-pixel consistency to avoid signed spatial cancellation'))
    for e in data['events']:
        ids=pools[e['event']]['CURRENT'];order=[e['current_id']]+[i for i in ids if i!=e['current_id']]
        r=row(e,records,order,ids,'CURRENT',0.,0.);r.update(pool='CURRENT',available_count=len(ids));rows.append(r)
    if direct_only:
        method='DIRECT_PIXEL_PERSISTENCE' if pixel_persistence else 'DIRECT_FOUR'
        pairs=[]
        for e in data['events']:
            for pool,available in pools[e['event']].items():
                ids=[i for i in available if records[i]['ready']];start=time.perf_counter()
                groups,values=direct_families(matrix[ids],names,pixel_persistence);score=values.mean(axis=1)
                order=[ids[j] for j in sorted(range(len(ids)),key=lambda j:(-score[j],ids[j]))]
                r=row(e,records,order,ids,method,0.,(time.perf_counter()-start)*1000)
                r.update(pool=pool,available_count=len(available));rows.append(r)
                if e['truth']['state']=='SINGLE_IMPACT':
                    good=min(ids,key=lambda i:records[i]['gt_distance']);a=ids.index(good);b=ids.index(e['current_id'])
                    pairs.append(dict(event=e['event'],pool=pool,good_id=good,good_error=records[good]['gt_distance'],
                        current_error=records[e['current_id']]['gt_distance'],family_good_minus_current=dict(zip(groups,(values[a]-values[b]).tolist()))))
        write_json(output/'results.json',dict(rows=rows,summaries=summarize_rows(rows)))
        write_json(output/'direct_pairs.json',pairs);write_json(output/'input_accesses.json',sorted(accessed))
        print(method,[(s['session'],s['hits']) for s in summarize_rows(rows) if s['method']==method and s['pool']=='CURRENT'])
        return
    for hold in SESSIONS:
        train=[s for s in SESSIONS if s!=hold]
        for mode in ('unique','grouped'):
            x,meta=grouped_design(data,matrix,names,train,mode)
            variants=[('PAIR_'+mode.upper(),list(range(x.shape[1])))]
            if mode=='grouped':
                variants += [('ABLATE_'+g,[j for j,f in enumerate(meta['families']) if f!=g]) for g in sorted(set(meta['families']))]
            for method,keep in variants:
                model=train_pairwise(data,x[:,keep],train)
                path=output/(method+'_'+hold);h=save_model(model,path)
                fitted.append(dict(held_out=hold,method=method,grouping=meta,kept_group_indices=keep,hashes=h,
                    train_ms=model['train_ms'],beta=model['beta'].tolist()))
                for e in data['events']:
                    if e['session']!=hold:continue
                    chosen_pools=pools[e['event']] if method.startswith('PAIR_') else {'CURRENT':pools[e['event']]['CURRENT']}
                    for pool,available in chosen_pools.items():
                        ids=[i for i in available if records[i]['ready']];start=time.perf_counter()
                        scores=predict(model,x[ids][:,keep]);order=[ids[j] for j in sorted(range(len(ids)),key=lambda j:(-scores[j],ids[j]))]
                        r=row(e,records,order,ids,method,0.,(time.perf_counter()-start)*1000)
                        r.update(pool=pool,available_count=len(available));rows.append(r)
                print(hold,method,'dimensions',len(keep),flush=True)
            if mode=='grouped':
                for omitted in train:
                    small=[s for s in train if s!=omitted]
                    design,small_meta=grouped_design(data,matrix,names,small,'grouped')
                    model=train_pairwise(data,design,small);scores=predict(model,design)
                    hashes_jack=save_model(model,output/('JACK_'+hold+'_omit_'+omitted))
                    selected=[]
                    for e in data['events']:
                        if e['session']!=hold:continue
                        ids=[i for i in pools[e['event']]['CURRENT'] if records[i]['ready']]
                        best=min(ids,key=lambda i:(-scores[i],i))
                        selected.append(dict(event=e['event'],index=best,error_px=records[best]['gt_distance']))
                    jack.append(dict(held_out=hold,omitted_training_session=omitted,train=small,
                        selected=selected,grouping=small_meta,hashes=hashes_jack))
    summaries=summarize_rows(rows)
    write_json(output/'results.json',dict(rows=rows,summaries=summaries));write_json(output/'models.json',fitted)
    write_json(output/'session_jackknife.json',jack);write_json(output/'input_accesses.json',sorted(accessed))
    for method in sorted({r['method'] for r in rows}):
        s=[r for r in summaries if r['method']==method and r['pool']=='CURRENT']
        print(method,'@42',sum(r['hits']['42'] for r in s),'tails1000',sum(r['errors_gt1000'] for r in s),
            'regressions',[v['event'] for r in s for v in r['regressions']],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base','prior','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--direct-only',action='store_true')
    p.add_argument('--pixel-persistence',action='store_true')
    a=p.parse_args()
    if a.pixel_persistence and not a.direct_only:p.error('--pixel-persistence requires --direct-only')
    run(a.base,a.prior,a.output,a.direct_only,a.pixel_persistence)
