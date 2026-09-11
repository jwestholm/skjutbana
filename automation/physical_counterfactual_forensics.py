"""Candidate survival, winner-pair evidence and four-negative offline forensics."""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import re

import cv2
import numpy as np

from automation.accuracy_physical_dataset import json_read, load_context
from automation.accuracy_verifier_research import fit_model, predict, save_model
from automation.physical_counterfactual_research import SESSIONS, RADII, write_json, xy
from automation.physical_counterfactual_models import universe
from src.engine.offline.accuracy_verifier import patch


def auc(positive, negative):
    a,b=np.asarray(positive),np.asarray(negative)
    return float(np.mean((a[:,None]>b[None])+.5*(a[:,None]==b[None]))) if len(a) and len(b) else None


def distribution(values):
    return dict(n=len(values),quantiles=np.percentile(values,[0,25,50,75,100]).tolist()) if values else dict(n=0,quantiles=None)


def decode(vector):
    return np.sign(vector)*np.expm1(np.abs(vector))


def temporal_profile(context, point):
    yy,xx=np.mgrid[-24:25,-24:25];radius=np.hypot(xx,yy)
    inner=radius<=4;ring=(radius>=8)&(radius<=12)
    pre=patch(context.pre,point,context.origin)
    before=np.stack([pre-patch(im,point,context.origin) for im in context.history])
    after=np.stack([pre-patch(im,point,context.origin) for im in context.post])
    contrast=lambda a:a[:,inner].mean(axis=1)-a[:,ring].mean(axis=1)
    a,b=contrast(before),contrast(after);support=np.abs(b)>.5
    onset=next((t-context.peak for t,ok in zip(context.post_times,support) if ok),None)
    return dict(pre_contrast=a.tolist(),post_contrast=b.tolist(),
        post_relative_ms=[1000*(t-context.peak) for t in context.post_times],
        onset_ms=1000*onset if onset is not None else None,
        support_fraction=float(np.mean(support)),pre_std=float(np.std(a)),
        pre_last=float(a[-1]),post_first=float(b[0]),post_last=float(b[-1]),
        late_peak_ratio=float(np.abs(np.median(b[len(b)//2:]))/(.75+np.max(np.abs(b)))),
        semantics='Uniform registered-image inner/ring contrast; 0.5 gray support probe, not a live decision threshold')


def summarize_pool(values, gt, available=True):
    if not available:
        return dict(status='UNAVAILABLE',count=None,nearest_px=None,oracle={str(r):None for r in RADII})
    distances=[math.dist(xy(c),xy(gt)) for c in values]
    return dict(status='OBSERVED',count=len(values),nearest_px=min(distances,default=None),
                oracle={str(r):any(d<=r for d in distances) for r in RADII})


def loss_reason(before, after, radius, event, details, records):
    """Only assign exact explanations supported by the recorded operation."""
    if before=='raw_mask_geometry' and after=='legacy_precap':
        gt=event['truth']['impacts'][0]
        good=[c for c in details['stage_positions'].get('raw_contours_reconstructed',[]) if math.dist(xy(c),xy(gt))<=radius]
        bounds=set()
        for stage in details['stage_details']:
            for rejected in stage['upstream'].get('rejected_blobs',[]):
                match=re.search(r'^area .*vs ([0-9.]+)-([0-9.]+)\)',rejected.get('reason',''))
                if match:bounds.add(tuple(map(float,match.groups())))
        if good and len(bounds)==1:
            minimum,maximum=next(iter(bounds))
            if all(c['area']<minimum or c['area']>maximum for c in good):
                return dict(rule='Legacy contour area filter',certainty='RECONSTRUCTED_RECORDED_THRESHOLD_PREDICATE',
                    min_area=minimum,max_area=maximum,areas=[c['area'] for c in good],
                    explanation='Saved mask geometry violates bounds recorded by the same-frame rejection ledger; area<1 blobs were omitted from individual diagnostics')
    if after=='track_representative':
        gt=event['truth']['impacts'][0]
        candidates=[(batch,row) for batch in details['snapshot']['associations'] for row in batch['records']
                    if math.dist(xy(row['candidate']),xy(gt))<=radius]
        return dict(rule='Track coordinate representation',certainty='RECORDED_ASSOCIATION',
            observations=[dict(track_id=r['track_id'],action=r['action'],reason=r['reason'],
                distance=r.get('association_distance'),threshold=r.get('association_threshold'),
                before=r.get('before'),after=r.get('after')) for b,r in candidates])
    if after=='final_selected':
        good=[t for t in event['representations'] if records[t['current']]['gt_distance']<=radius]
        if good:
            t=min(good,key=lambda t:t['final_rank'])
            winner=event['representations'][0]
            return dict(rule='CURRENT deterministic rank key',certainty='EXACT_RECORDED_INPUT_REPLAY',
                good_track=t['track_id'],good_rank=t['final_rank'],good_rank_key=t['rank_key'],
                winner_track=winner['track_id'],winner_rank_key=winner['rank_key'])
    if after=='legacy_retained':
        return dict(rule='Legacy candidate_limit=200 after source score sorting',certainty='RECORDED_CAP_BOUNDARY')
    if after=='hybrid_output' and event['event']=='D01:3':
        return dict(rule='hybrid_capacity_exhausted',certainty='PRIOR_EXACT_STAGE_RECONSTRUCTION',
            evidence='legacy:170, merged rank 307, legacy pool rank 174; 50 V2 +150 additional legacy fill 200',
            scope='Exact explanation for the 7.327634px point; no assumption about other lost coordinates')
    return dict(rule='UNAVAILABLE',certainty='MISSING_PER_CANDIDATE_OPERATION',
                interval=[before,after],explanation='Saved endpoints do not identify the intervening operation; do not infer from aggregate counters')


def survival(event, details, records):
    gt=event['truth']['impacts'][0];positions=details['stage_positions'];snap=details['snapshot']
    stages={}
    for label,key in [('raw_recorded','raw_candidates'),('raw_mask_geometry','raw_contours_reconstructed'),
                      ('legacy_precap','legacy_precap'),('legacy_retained','legacy_retained'),
                      ('hybrid_output','cleanup_input'),('retained_proposal','retained')]:
        stages[label]=summarize_pool(positions.get(key,[]),gt,key in positions)
    history=[r['candidate'] for b in snap['associations'] if b['frame_ts']<=event['context']['cutoff'] for r in b['records']
             if r.get('producer_shot_id') in (None,snap['event']['shot_id'])]
    for name,values in [('track_history',history),('track_representative',snap['tracks']),
                        ('eligible_pool',[t for t in snap['tracks'] if t['eligible']]),
                        ('common_verifier_pool',[records[i] for i in event['pools']['current']]),
                        ('final_selected',[records[event['current_id']]])]:
        stages[name]=summarize_pool(values,gt)
    lost={}
    for radius in RADII:
        previous=None;first=None;unknown=[]
        for name,stage in stages.items():
            present=stage['oracle'][str(radius)]
            if present is None:
                unknown.append(name);continue
            if previous and previous[1] and not present:
                first=dict(from_stage=previous[0],to_stage=name,unavailable_between=unknown,
                           reason=loss_reason(previous[0],name,radius,event,details,records));break
            previous=(name,present);unknown=[]
        lost[str(radius)]=first
    return dict(event=event['event'],stages=stages,first_observed_loss=lost,
                caveat='Raw mask geometry is reconstructed, not a full raw V1/V2 proposal ledger; availability is not physical validation')


def load_model(path):
    with np.load(path/'parameters.npz',allow_pickle=False) as p:
        return dict(family='logistic',**{k:p[k] for k in p.files})


def contributions(model, columns, matrix, names, good, wrong):
    z=np.clip((matrix[[good,wrong]][:,columns]-model['mu'])/model['sd'],-8,8)
    parts=(z[0]-z[1])*model['beta'][1:]
    order=np.argsort(parts)
    return dict(correct_minus_wrong_logit=float(parts.sum()),
        helps_correct=[dict(feature=names[columns[i]],contribution=float(parts[i])) for i in order[-8:][::-1]],
        helps_wrong=[dict(feature=names[columns[i]],contribution=float(parts[i])) for i in order[:8]])


def event_descriptors(event, records, matrix, names):
    """GT-free event evidence at a fixed physical-priority winner and whole ROI."""
    pool=event['pools']['current'];winner=max(pool,key=lambda i:records[i]['physical_priority'])
    keep=['r4_abs_late','r4_center_ring_ratio','r4_persistence','r4_pre_std','r4_onset_gain',
          'r4_spatial_consistency','pre_centered_clean_r4_mean','pre_centered_clean_r4_inner_ring',
          'piecewise_clean_r4_mean','piecewise_clean_r4_inner_ring']
    values=[float(matrix[winner,names.index(k)]) for k in keep]
    for channel in ('persistent','pre_rms','pre_centered','piecewise'):
        keep.append('event_'+channel+'_retention');values.append(event['clean_metrics'][channel]['retained_mass'])
    keep.append('compensated_to_centered_local_log_ratio')
    values.append(float(matrix[winner,names.index('piecewise_clean_r4_mean')]-matrix[winner,names.index('pre_centered_clean_r4_mean')]))
    return keep,values,winner


def no_impact(dataset,matrix,output):
    records=dataset['records'];names=dataset['feature_names'];events=dataset['events']
    negatives=[e for e in events if e['truth']['state']=='NO_PHYSICAL_SHOT']
    positive_candidates=[];positive_winners=[]
    for e in events:
        if e['truth']['state']!='SINGLE_IMPACT':continue
        candidates=[i for i in universe(e,records) if records[i]['ready'] and records[i]['gt_distance']<=20]
        if candidates:positive_candidates.append(min(candidates,key=lambda i:records[i]['gt_distance']))
        if records[e['current_id']]['gt_distance']<=10:positive_winners.append(e['current_id'])
    neg_ids=[e['current_id'] for e in negatives];distributions=[]
    for column,name in enumerate(names):
        positive=matrix[positive_candidates,column].tolist();negative=matrix[neg_ids,column].tolist()
        distributions.append(dict(feature=name,nearest_ready_physical=distribution(positive),
            correct_current_winners=distribution(matrix[positive_winners,column].tolist()),
            false_current_winners=distribution(negative),auc_positive_higher=auc(positive,negative),
            interpretation='Descriptive GT-referenced comparison; choosing direction/threshold from these four negatives is not validation'))
    descriptors=[event_descriptors(e,records,matrix,names) for e in events]
    event_names=descriptors[0][0];x=np.array([d[1] for d in descriptors]);y=np.array([int(e['truth']['state']=='SINGLE_IMPACT') for e in events])
    gate=[];models=[]
    for hold in SESSIONS:
        train=np.array([e['session']!=hold for e in events]);test=~train
        weights=np.ones(train.sum())
        for label in (0,1):weights[y[train]==label]/=2*sum(y[train]==label)
        model=fit_model(x[train],y[train],weights,'logistic')
        score=predict(model,x)
        threshold=float(np.max(score[train & (y==0)])+1e-12)
        models.append(dict(held_out=hold,train=[s for s in SESSIONS if s!=hold],
            hashes=save_model(model,output/('NO_IMPACT_'+hold)),threshold_reject_training_negatives=threshold))
        for i in np.where(test)[0]:
            gate.append(dict(event=events[i]['event'],session=hold,physical=bool(y[i]),score=float(score[i]),
                emits_at_fixed_half=bool(score[i]>=.5),emits_at_training_negative_bound=bool(score[i]>=threshold),
                training_negative_bound=threshold,descriptor_candidate=descriptors[i][2]))
    summaries=[]
    for field in ('emits_at_fixed_half','emits_at_training_negative_bound'):
        for session in SESSIONS:
            rows=[r for r in gate if r['session']==session]
            summaries.append(dict(session=session,gate=field,
                physical=sum(r['physical'] for r in rows),negative=sum(not r['physical'] for r in rows),
                false_rejections=sum(r['physical'] and not r[field] for r in rows),
                true_negatives=sum(not r['physical'] and not r[field] for r in rows),
                false_accepts=sum(not r['physical'] and r[field] for r in rows)))
    return dict(distributions=distributions,event_descriptor_names=event_names,gate_rows=gate,gate_summaries=summaries,
        gate_models=models,negative_event_ids=[e['event'] for e in negatives],
        limitation='Four negatives in two sessions; models/thresholds are forensic stress tests, not calibrated live NO_IMPACT decisions')


def run(cache,models,output):
    output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    dataset=json_read(cache/'dataset.json');matrix=np.load(cache/'features.npy',allow_pickle=False)
    if tuple(dataset['sessions'])!=SESSIONS:raise ValueError('Unexpected sessions')
    records=dataset['records'];names=dataset['feature_names'];model_info=json_read(models/'models.json')
    methods={(r['held_out'],r['method']):(load_model(models/(r['method']+'_'+r['held_out'])),r['feature_indices']) for r in model_info}
    # Inspect the previously reported COMMON regressions using frozen parameters
    # only. No historical training-session images/rows are loaded or fitted.
    old_root=Path('/data/skjutbana/evaluation_runs/D01_evidence_20260910_081143/retention_final')
    old={s:load_model(old_root/('model_'+s)) for s in SESSIONS}
    core=list(range(72));ledger=[];pairs=[];common_matches=[]
    prior=json_read(old_root/'results.json')
    prior_rows={r['event']:r for r in prior['rows'] if r['method']=='COMMON_FIXED'}
    for event in dataset['events']:
        if event['truth']['state']!='SINGLE_IMPACT':continue
        details=json_read(cache/(event['event'].replace(':','_')+'_trace_details.json'))
        ledger.append(survival(event,details,records))
        pool=event['pools']['current'];old_model=old[event['session']]
        scores=predict(old_model,matrix[pool][:,core]);common_id=pool[int(np.argmax(scores))]
        prior_xy=xy(prior_rows[event['event']]['selected'])
        common_matches.append(dict(event=event['event'],match=xy(records[common_id])==prior_xy))
        if not common_matches[-1]['match']:raise AssertionError('Historical common winner differs')
        all_candidates=[i for i in universe(event,records) if records[i]['ready']]
        nearest_current=min(pool,key=lambda i:records[i]['gt_distance'])
        nearest_observed=min(all_candidates,key=lambda i:records[i]['gt_distance'])
        targets=[('nearest_eligible',nearest_current,event['current_id']),
                 ('nearest_observed',nearest_observed,event['current_id'])]
        if records[event['current_id']]['gt_distance']<=42 and records[common_id]['gt_distance']>42:
            targets.append(('protected_current_vs_historical_common',event['current_id'],common_id))
        path=Path(event['trace_path']);trace=json_read(path);context,_=load_context(path,trace,'history_early')
        descriptor_cache={}
        def describe(index):
            if index not in descriptor_cache:
                c=records[index]
                descriptor_cache[index]=dict(record=c,features=dict(zip(names,decode(matrix[index]).tolist())),
                    temporal=temporal_profile(context,xy(c)))
            return descriptor_cache[index]
        for kind,good,wrong in targets:
            row=dict(event=event['event'],kind=kind,
                target_is_within20=records[good]['gt_distance']<=20,
                target_is_gt_referenced_diagnostic=True,good=describe(good),winner=describe(wrong),
                historical_common=contributions(old_model,core,matrix,names,good,wrong),models={})
            for method in ('POINT_CORE','PAIR_CORE','PAIR_CLEAN','PAIR_HISTORY'):
                model,cols=methods[event['session'],method]
                row['models'][method]=contributions(model,cols,matrix,names,good,wrong)
            pairs.append(row)
        print(event['event'],'ledger and pairs',flush=True)
    write_json(output/'survival.json',dict(events=ledger,unavailable=dataset['unavailable']))
    write_json(output/'pairs.json',dict(pairs=pairs,historical_common_matches=common_matches,
        scope='Historical common weights reused only to explain existing four-session regressions; new fits use four sessions only'))
    write_json(output/'no_impact.json',no_impact(dataset,matrix,output))
    write_json(output/'manifest.json',dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_hashes={n:hashlib.sha256((cache/n).read_bytes()).hexdigest() for n in ('dataset.json','features.npy')},
        model_manifest_sha256=hashlib.sha256((models/'models.json').read_bytes()).hexdigest(),status='OFFLINE_ONLY'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.cache,a.models,a.output)
