"""Session-held-out supervised physical patch experiments; no live installation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from src.engine.offline.physical_event_truth import score_event

PRIMARY = ('H10', 'H20', 'POST_FIX', 'S01')
RADII = (5, 10, 20, 42)


def training_data(dataset, matrix, sessions, pool='current'):
    if 'S02' in sessions or 'S03' in sessions:
        raise ValueError('Challenge/validation training forbidden')
    events = {e['event']:e for e in dataset['events']}
    chosen = [r for r in dataset['records'] if r['session'] in sessions and r['training_label'] is not None
              and events[r['event']]['truth']['state'] in ('SINGLE_IMPACT', 'NO_PHYSICAL_SHOT')
              and (r['pool'] == 'training_gt_only' or
                   (r['pool'] in (('current','expanded') if pool == 'union' else (pool,)) and r['ready']))]
    indices = np.array([r['index'] for r in chosen])
    y = np.array([r['training_label'] for r in chosen], np.int32)
    counts = {}
    for r in chosen:
        key = (r['event'], r['training_label'])
        counts[key] = counts.get(key, 0) + 1
    session_events = {s:len({r['event'] for r in chosen if r['session'] == s}) for s in sessions}
    weights = np.array([1/(counts[r['event'],r['training_label']]*session_events[r['session']]) for r in chosen], np.float64)
    for label in (0, 1):
        if not np.any(y == label):
            raise ValueError('Both physical classes required')
        weights[y == label] /= 2*weights[y == label].sum()
    return matrix[indices], y, weights


def fit_model(x, y, weights, family):
    start = time.perf_counter()
    mu = np.average(x, axis=0, weights=weights)
    sd = np.sqrt(np.average((x-mu)**2, axis=0, weights=weights))
    sd = np.maximum(sd, .05)
    z = np.clip((x-mu)/sd, -8, 8).astype(np.float32)
    if family == 'logistic':
        design = np.column_stack((np.ones(len(z)), z)).astype(np.float64)
        beta = np.zeros(design.shape[1])
        # Fixed regularization/optimizer; session data never selects per-shot knobs.
        regularization = np.full(len(beta), .05)
        regularization[0] = 0
        lipschitz = .25*np.linalg.eigvalsh(design.T@(weights[:,None]*design))[-1]+.05
        for _ in range(400):
            pred = 1/(1+np.exp(-np.clip(design@beta, -35, 35)))
            gradient = design.T@(weights*(pred-y))+regularization*beta
            beta -= gradient/lipschitz
        model = dict(family=family, mu=mu, sd=sd, beta=beta)
    elif family == 'forest':
        cv2.setRNGSeed(41017)
        forest = cv2.ml.RTrees_create()
        forest.setMaxDepth(6)
        forest.setMinSampleCount(4)
        forest.setActiveVarCount(max(1, int(np.sqrt(z.shape[1]))))
        forest.setCalculateVarImportance(True)
        forest.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 96, 0))
        var_type = np.array([cv2.ml.VAR_ORDERED]*z.shape[1]+[cv2.ml.VAR_CATEGORICAL], np.uint8)
        train = cv2.ml.TrainData_create(z, cv2.ml.ROW_SAMPLE, y,
                                      sampleWeights=(weights/weights.max()).astype(np.float32), varType=var_type)
        if not forest.train(train):
            raise ValueError('Forest training failed')
        model = dict(family=family, mu=mu, sd=sd, forest=forest)
    else:
        raise ValueError(family)
    model['train_ms'] = (time.perf_counter()-start)*1000
    return model


def predict(model, x):
    z = np.clip((x-model['mu'])/model['sd'], -8, 8).astype(np.float32)
    if model['family'] == 'logistic':
        beta = model['beta']
        return 1/(1+np.exp(-np.clip(beta[0]+z@beta[1:], -35, 35)))
    votes = model['forest'].getVotes(z, 0)
    positive = int(np.where(votes[0] == 1)[0][0])
    return votes[1:,positive]/votes[1:].sum(axis=1)


def event_predictions(dataset, scores, sessions, pool='current', threshold=0., exclude_caveats=False):
    records = dataset['records']
    rows = []
    for event in dataset['events']:
        if event['session'] not in sessions:
            continue
        if pool == 'expanded' and event.get('proposal_unavailable'):
            continue
        truth = event['truth']
        if exclude_caveats and truth['caveats']:
            continue
        allowed = ('current',) if pool == 'current' else ('expanded',) if pool == 'expanded' else ('current','expanded')
        candidates = [records[i] for i in event['indices'] if records[i]['pool'] in allowed]
        ordered = sorted(candidates, key=lambda r:(-float(scores[r['index']]), r['index']))
        best = ordered[0] if ordered else None
        selected = best if best and best['ready'] and scores[best['index']] >= threshold else None
        result = score_event(truth, selected)
        gt = truth['impacts'][0] if truth['state'] == 'SINGLE_IMPACT' else None
        distances = [r['gt_distance'] for r in ordered] if gt else []
        rows.append(dict(event=event['event'], session=event['session'], truth=truth, result=result,
                         selected=selected, best_score=float(scores[best['index']]) if best else None,
                         rejected_by_threshold=bool(best and scores[best['index']] < threshold),
                         unready_at_recorded_decision=bool(best and not best['ready']),
                         oracle={str(r):any(d is not None and d <= r for d in distances) for r in RADII},
                         ranks={str(r):next((i+1 for i,d in enumerate(distances) if d is not None and d <= r), None) for r in RADII},
                         candidate_count=len(candidates)))
    return rows


def summarize(rows):
    physical = [r for r in rows if r['result'].get('physical') and r['result']['evaluable']]
    nonphysical = [r for r in rows if r['result'].get('physical') is False and r['result']['evaluable']]
    errors = sorted(r['result']['error_px'] for r in physical if r['result']['error_px'] is not None)
    hits = {str(r):sum(row['result']['hits'][str(r)] for row in physical) for r in RADII}
    oracle = {str(r):sum(row['oracle'][str(r)] for row in physical) for r in RADII}
    tn = sum(row['result']['true_negative'] for row in nonphysical)
    den = len(physical)+len(nonphysical)
    return dict(physical=len(physical), no_physical=len(nonphysical), hits=hits, oracle=oracle,
                accuracy={str(r):hits[str(r)]/len(physical) if physical else None for r in RADII},
                conditional_accuracy={str(r):hits[str(r)]/oracle[str(r)] if oracle[str(r)] else None for r in RADII},
                mean_error_px=float(np.mean(errors)) if errors else None,
                median_error_px=float(np.median(errors)) if errors else None,
                p95_error_px=errors[math.ceil(.95*len(errors))-1] if errors else None,
                errors_gt100=sum(e>100 for e in errors), true_negatives=tn, false_emissions=len(nonphysical)-tn,
                false_rejections=sum(row['result']['false_rejection'] for row in physical),
                true_impact_emission_rate=sum(row['result']['emitted'] for row in physical)/len(physical) if physical else None,
                no_impact_rejection_rate=tn/len(nonphysical) if nonphysical else None,
                combined_event_accuracy={str(r):(hits[str(r)]+tn)/den if den else None for r in RADII},
                topk42={str(k):sum(row['ranks']['42'] is not None and row['ranks']['42']<=k for row in physical) for k in (1,3,5,10,20,50)},
                candidate_count=dict(mean=float(np.mean([r['candidate_count'] for r in rows])) if rows else 0,
                                     max=max((r['candidate_count'] for r in rows),default=0)),
                unready_at_recorded_decision=sum(r['unready_at_recorded_decision'] for r in rows))


def save_model(model, path):
    path.mkdir()
    np.savez(path/'parameters.npz', **{k:v for k,v in model.items() if k in ('mu','sd','beta')})
    if model['family'] == 'forest':
        model['forest'].save(str(path/'forest.xml'))
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}


def run(dataset_dir, output, train_pool='current'):
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    dataset = json.loads((dataset_dir/'dataset.json').read_text())
    matrix = np.load(dataset_dir/'features.npy', allow_pickle=False)
    if any(s not in (*PRIMARY, 'S02') for s in dataset['sessions']):
        raise PermissionError('Unexpected session in research data')
    families = []
    for family in ('logistic', 'forest'):
        folds, oof = [], []
        for hold in PRIMARY:
            train = [s for s in PRIMARY if s != hold]
            x,y,w = training_data(dataset,matrix,train,train_pool)
            model = fit_model(x,y,w,family)
            scores = predict(model,matrix)
            rows = event_predictions(dataset,scores,[hold])
            oof.extend(rows)
            folds.append(dict(hold=hold, train=train, metrics=summarize(rows), rows=rows, train_ms=model['train_ms']))
        # H4: predeclared 5th percentile of primary OOF physical event scores.
        # No S02 value is consumed in this calibration.
        calibration = sorted(r['best_score'] for r in oof if r['truth']['state']=='SINGLE_IMPACT' and r['best_score'] is not None)
        threshold = calibration[max(0, int(np.floor(.05*len(calibration))))]
        x,y,w = training_data(dataset,matrix,PRIMARY,train_pool)
        model = fit_model(x,y,w,family)
        t0=time.perf_counter();scores=predict(model,matrix);predict_ms=(time.perf_counter()-t0)*1000
        model_path=output/f'{family}_model';hashes=save_model(model,model_path)
        experiments=[]
        for pool in ('current','expanded','union'):
            for reject in (False,True):
                rows=event_predictions(dataset,scores,['S02'],pool,threshold if reject else 0.)
                sensitivity=event_predictions(dataset,scores,['S02'],pool,threshold if reject else 0.,True)
                experiments.append(dict(pool=pool,rejection=reject,threshold=threshold if reject else None,
                    S02=summarize(rows),S02_excluding_possible_ambiguous_event1=summarize(sensitivity),rows=rows))
        # Expansion OOF is assessed with each primary session excluded from fit.
        expansion_folds=[]
        for hold in PRIMARY:
            train=[s for s in PRIMARY if s != hold]
            x,y,w=training_data(dataset,matrix,train,train_pool)
            fold_model=fit_model(x,y,w,family);fold_scores=predict(fold_model,matrix)
            expansion_folds.append(dict(hold=hold,union=summarize(event_predictions(dataset,fold_scores,[hold],'union')),
                                       expanded=summarize(event_predictions(dataset,fold_scores,[hold],'expanded'))))
        item=dict(family=family,status='RESEARCH_ONLY_NOT_PROMOTED',primary_cv=summarize(oof),folds=folds,
                  expansion_folds=expansion_folds,training_sessions=list(PRIMARY),no_impact_threshold=threshold,
                  challenge=experiments,model_hashes=hashes,predict_all_rows_ms=predict_ms,train_ms=model['train_ms'])
        families.append(item)
        print(family,'CV',json.dumps(item['primary_cv']),flush=True)
        for trial in experiments:print('S02',family,trial['pool'],'reject',trial['rejection'],json.dumps(trial['S02']),flush=True)
    result=dict(status='OFFLINE_CAUSAL_FRAME_RESEARCH_NOT_PHYSICAL_VALIDATION',dataset=str(dataset_dir),training_pool=train_pool,
                reference=dataset['reference'],feature_contract=dataset['feature_contract'],families=families,
                limitations=['S02 is development-used; S03 prohibited',
                             'Expanded coordinates have multi-frame support; full async resolver/known-hole state is not replayed',
                             'Unready alternate means no selection at recorded cutoff, not proof of eventual timeout',
                             'Single stored label on possibly ambiguous S02 event 1 preserved; exclusion sensitivity provided'])
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-pool',choices=('current','union'),default='current')
    args=parser.parse_args();run(args.dataset,args.output,args.train_pool)
