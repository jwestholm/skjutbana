"""Measured follow-up: static-appearance ablation and causal persistent-step evidence.

Motivated by winner-pair texture contributions and a step visible in late PRE.
No change to audio gating, no timing veto, no live classifier or proposal path.
"""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import time

import cv2
import numpy as np

from automation.accuracy_physical_dataset import json_read, load_context
from automation.accuracy_verifier_research import predict, event_predictions, summarize, save_model
from automation.physical_counterfactual_research import SESSIONS, RADII, write_json, xy
from automation.physical_counterfactual_models import train_pairwise, model_columns, universe, spatial_order
from src.engine.offline.accuracy_verifier import patch


def step_statistics(signal, timestamps, peak):
    """Best robust two-level step, allowing early PRE onset; no truth input.

    Minimum two samples before and three after. Timestamp is reported, never
    used for unconditional rejection or to demand a minimum inter-shot interval.
    """
    signal=np.asarray(signal,dtype=np.float64);timestamps=np.asarray(timestamps)
    if len(signal)!=len(timestamps) or len(signal)<5 or not np.isfinite(signal).all():
        raise ValueError('Finite signal with at least five matching timestamps required')
    if np.any(np.diff(timestamps)<=0):raise ValueError('Strictly ordered timestamps required')
    total=float(np.mean((signal-np.median(signal))**2))+.5625
    choices=[]
    for split in range(2,len(signal)-2):
        before,after=signal[:split],signal[split:]
        a,b=np.median(before),np.median(after)
        error=float((np.sum((before-a)**2)+np.sum((after-b)**2))/len(signal))
        choices.append((error,split,float(a),float(b)))
    error,split,a,b=min(choices)
    before,after=signal[:split],signal[split:]
    pre_mad=float(np.median(np.abs(before-a)));post_mad=float(np.median(np.abs(after-b)))
    amplitude=b-a
    return dict(step_amplitude=amplitude,step_abs=abs(amplitude),
        step_snr=abs(amplitude)/(.75+1.4826*(pre_mad+post_mad)),
        step_explained=max(0.,1-error/total),step_pre_mad=pre_mad,step_post_mad=post_mad,
        step_persistent_fraction=float(np.mean(np.sign(amplitude)*(after-a)>.5)),
        step_onset_ms=float((timestamps[split]-peak)*1000),
        step_span_ms=float((timestamps[-1]-timestamps[split])*1000))


def step_features(context, point):
    yy,xx=np.mgrid[-24:25,-24:25];radius=np.hypot(xx,yy)
    pre=patch(context.pre,point,context.origin)
    # Context history can share a snapshot timestamp; unique timestamp samples
    # are merged once, preserving camera-time causality.
    samples={t:im for t,im in zip(context.pre_times,context.history)}
    samples.update({t:im for t,im in zip(context.post_times,context.post)})
    times=sorted(samples)
    if times[-1]>context.cutoff:raise ValueError('Future frame in step evidence')
    residual=np.stack([pre-patch(samples[t],point,context.origin) for t in times])
    values={}
    for r in (2,4,8,16):
        inner=radius<=r;ring=(radius>=1.5*r)&(radius<=min(2.5*r,24))
        contrast=residual[:,inner].mean(axis=1)-residual[:,ring].mean(axis=1)
        values.update({f'r{r}_{k}':v for k,v in step_statistics(contrast,times,context.peak).items()})
    raw=np.array(list(values.values()),np.float32)
    return list(values),np.sign(raw)*np.log1p(np.abs(raw))


def step_reserve(event, records, matrix, names, budget=48, cap=200):
    current=list(event['pools']['current']);selected=list(current)
    if len(current)>cap:raise ValueError('Current pool exceeds cap')
    cols=[names.index(f'r{r}_step_snr') for r in (2,4)]
    score=lambda i:float(np.mean(matrix[i,cols]))
    candidates=[i for i in universe(event,records) if i not in set(current)]
    for index in spatial_order(candidates,records,event,score):
        if len(selected)>=min(cap,len(current)+budget):break
        if any(math.dist(xy(records[index]),xy(records[j]))<8 for j in selected):continue
        selected.append(index)
    return selected


def run(cache,baseline,output):
    output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    dataset=json_read(cache/'dataset.json')
    if tuple(dataset['sessions'])!=SESSIONS:raise ValueError('Unexpected sessions')
    base=np.load(cache/'features.npy',allow_pickle=False);records=dataset['records'];names=dataset['feature_names']
    extra=None;extra_names=None;times=[]
    for event in dataset['events']:
        start=time.perf_counter();path=Path(event['trace_path']);trace=json_read(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=event['trace_sha256']:raise ValueError('Changed trace')
        context,_=load_context(path,trace,'history_early');context_ms=(time.perf_counter()-start)*1000;t0=time.perf_counter()
        for index in event['indices']:
            n,v=step_features(context,xy(records[index]))
            if extra is None:extra=np.zeros((len(records),len(v)),np.float32);extra_names=n
            if extra_names!=n:raise ValueError('Feature schema drift')
            extra[index]=v
        times.append(dict(event=event['event'],context_ms=context_ms,step_ms=(time.perf_counter()-t0)*1000,
                          evaluated_count=len(event['indices'])))
        print(event['event'],'step features',round(times[-1]['step_ms'],1),flush=True)
    np.save(output/'step_features.npy',extra)
    matrix=np.column_stack([base,extra]);all_names=names+extra_names
    pools=json_read(baseline/'pools.json')
    for event in dataset['events']:
        t0=time.perf_counter();indices=step_reserve(event,records,matrix,all_names)
        pools[event['event']]['pools']['LATE_STEP']=indices
        pools[event['event']]['timing_ms']['LATE_STEP']=(time.perf_counter()-t0)*1000
    # Remove static appearance as an authority feature; this does not blacklist
    # image edges or discard candidates near old damage.
    change_cols=[i for i in model_columns(names,'PAIR_CLEAN') if not names[i].endswith(('_pre_texture','_edge'))]
    rows=[];models=[]
    for hold in SESSIONS:
        train=[s for s in SESSIONS if s!=hold]
        for method,cols in [('PAIR_CHANGE_ONLY',change_cols),('PAIR_CHANGE_STEP',change_cols+list(range(len(names),len(all_names))))]:
            model=train_pairwise(dataset,matrix[:,cols],train);scores=predict(model,matrix[:,cols])
            models.append(dict(method=method,held_out=hold,train=train,feature_names=[all_names[i] for i in cols],
                feature_indices=cols,hashes=save_model(model,output/(method+'_'+hold)),train_ms=model['train_ms']))
            for event in dataset['events']:
                if event['session']!=hold:continue
                for policy,ids in pools[event['event']]['pools'].items():
                    temp=dict(dataset,records=[dict(r,pool='current') for r in records],events=[dict(event,indices=ids)])
                    t0=time.perf_counter();row=event_predictions(temp,scores,[hold])[0]
                    row.update(method=method,policy=policy,selection_ms=(time.perf_counter()-t0)*1000,
                        retention_ms=pools[event['event']]['timing_ms'][policy]);rows.append(row)
            print(hold,method,'trained',flush=True)
    summaries=[]
    for method,policy in sorted({(r['method'],r['policy']) for r in rows}):
        for session in SESSIONS:
            selected=[r for r in rows if (r['method'],r['policy'],r['session'])==(method,policy,session)]
            summaries.append(dict(method=method,policy=policy,session=session,**summarize(selected)))
    write_json(output/'results.json',dict(summaries=summaries,rows=rows))
    write_json(output/'models.json',models);write_json(output/'pools.json',pools)
    write_json(output/'protocol.json',dict(status='OFFLINE_ONLY',feature_names=extra_names,timing=times,
        motivation='Pairwise static PRE-texture domination; persistent change already visible in late PRE on a protected regression',
        caveats=['Exploratory follow-up after development inspection; no independent validation',
                 'Step timing is a feature, never unconditional rejection','GT-only coordinates excluded from fitting and selection'],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    for s in summaries:print(s['session'],s['method'],s['policy'],s['hits'],s['oracle'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.cache,a.baseline,a.output)
