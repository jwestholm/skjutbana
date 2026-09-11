"""Within-event physical-vs-background ranking with broad negatives, held out by session."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from automation.accuracy_verifier_research import PRIMARY, event_predictions, predict, save_model, summarize, training_data


def fit_pairwise(dataset, matrix, sessions, hard_rounds=0):
    # This call also enforces the no-challenge-training rule.
    x,y,w=training_data(dataset,matrix,sessions,'union')
    mu=np.average(x,axis=0,weights=w)
    sd=np.maximum(.05,np.sqrt(np.average((x-mu)**2,axis=0,weights=w)))
    z=np.clip((matrix-mu)/sd,-8,8).astype(np.float64)
    records=dataset['records'];groups=[]
    for event in dataset['events']:
        if event['session'] not in sessions or event['truth']['state']!='SINGLE_IMPACT':continue
        positives=[i for i in event['indices'] if records[i]['pool']=='training_gt_only']
        negatives=[i for i in event['indices'] if records[i]['pool'] in ('current','expanded') and
                   records[i]['ready'] and records[i]['training_label']==0]
        if positives and negatives:groups.append((event['session'],np.mean(z[positives],axis=0),negatives))
    counts={s:sum(g[0]==s for g in groups) for s in sessions}
    beta=np.zeros(matrix.shape[1])
    for iteration in range(hard_rounds+1):
        differences=[];weights=[]
        for session,positive,negatives in groups:
            if iteration:
                # Fixed top-32 hard negatives per event, derived on training sessions only.
                negatives=sorted(negatives,key=lambda i:(-float(z[i]@beta),i))[:32]
            differences.extend(positive-z[negatives])
            weights.extend([1/(counts[session]*len(negatives))]*len(negatives))
        design=np.asarray(differences);weights=np.asarray(weights);weights/=weights.sum()
        bound=.25*np.linalg.eigvalsh(design.T@(weights[:,None]*design))[-1]+.05
        for _ in range(400):
            margins=np.clip(design@beta,-35,35)
            gradient=design.T@(weights*(-1/(1+np.exp(margins))))+.05*beta
            beta-=gradient/bound
    return dict(family='logistic',mu=mu,sd=sd,beta=np.r_[0.,beta])


def run(dataset_path,output):
    output.mkdir(parents=True,exist_ok=False)
    dataset=json.loads((dataset_path/'dataset.json').read_text());matrix=np.load(dataset_path/'features.npy',allow_pickle=False)
    if any(s not in (*PRIMARY,'S02') for s in dataset['sessions']):raise PermissionError('Unexpected session')
    result=dict(dataset=str(dataset_path),status='OFFLINE_RESEARCH_ONLY',experiments=[])
    for hard in (0,2):
        start=time.perf_counter();folds=[];oof=[]
        for hold in PRIMARY:
            model=fit_pairwise(dataset,matrix,[s for s in PRIMARY if s!=hold],hard)
            scores=predict(model,matrix);rows=event_predictions(dataset,scores,[hold],'union');oof.extend(rows)
            folds.append(dict(hold=hold,metrics=summarize(rows),rows=rows,
                              current=summarize(event_predictions(dataset,scores,[hold],'current'))))
        calibration=sorted(r['best_score'] for r in oof if r['truth']['state']=='SINGLE_IMPACT')
        threshold=calibration[int(np.floor(.05*len(calibration)))]
        model=fit_pairwise(dataset,matrix,PRIMARY,hard);scores=predict(model,matrix)
        trials=[]
        for pool in ('current','expanded','union'):
            rows=event_predictions(dataset,scores,['S02'],pool)
            rejected=event_predictions(dataset,scores,['S02'],pool,threshold)
            trials.append(dict(pool=pool,metrics=summarize(rows),rejection_metrics=summarize(rejected),rows=rows,
                               excluding_possible_ambiguous_event1=summarize([r for r in rows if r['event']!='S02:1'])))
        item=dict(hard_rounds=hard,folds=folds,primary_cv=summarize(oof),S02=trials,threshold=threshold,
                  hashes=save_model(model,output/f'model_hard{hard}'),runtime_seconds=time.perf_counter()-start)
        result['experiments'].append(item)
        print('hard',hard,'CV',item['primary_cv']['hits'],flush=True)
        for t in trials:print('S02',t['pool'],t['metrics']['hits'],t['rejection_metrics']['true_negatives'],flush=True)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.dataset,args.output)
