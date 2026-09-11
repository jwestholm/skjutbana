"""Visual-only event rejection, calibrated on primary sessions; S02 development challenge."""
import argparse
import json
from pathlib import Path

import numpy as np

from automation.accuracy_verifier_research import PRIMARY, event_predictions, fit_model, predict, save_model, summarize, training_data
from src.engine.offline.physical_event_truth import score_event

EVIDENCE_NAMES=('r4_pre_peak','r4_onset_gain','r4_abs_late','r4_persistence',
                'morph_offset','morphology_mass','jitter_min_abs','r8_signed_late','r8_center_ring_ratio')


def event_matrix(dataset,matrix):
    columns=[dataset['feature_names'].index(name) for name in EVIDENCE_NAMES]
    vectors=[]
    for event in dataset['events']:
        indices=[i for i in event['indices'] if dataset['records'][i]['pool']=='expanded']
        if not indices:raise ValueError('Visual event evidence unavailable')
        # Every location comes from the same bounded image-only proposal channel.
        vectors.append(np.quantile(matrix[np.ix_(indices,columns)],(.5,.9,.99),axis=0).ravel())
    return np.asarray(vectors,np.float32)


def train_event(dataset,vectors,sessions):
    if any(s not in PRIMARY for s in sessions):raise ValueError('Only primary training sessions allowed')
    indices=[i for i,e in enumerate(dataset['events']) if e['session'] in sessions and e['truth']['state'] in ('SINGLE_IMPACT','NO_PHYSICAL_SHOT')]
    y=np.array([int(dataset['events'][i]['truth']['state']=='SINGLE_IMPACT') for i in indices],np.int32)
    if len(set(y))!=2:raise ValueError('Both event classes required')
    weights=np.zeros(len(indices))
    for value in (0,1):weights[y==value]=.5/np.sum(y==value)
    return fit_model(vectors[indices],y,weights,'logistic')


def gate(rows,event_scores,threshold):
    result=[]
    for row in rows:
        row=dict(row)
        probability=event_scores[row['event']]
        row['visual_event_score']=probability
        if probability<threshold:
            row['selected']=None;row['result']=score_event(row['truth'],None)
        result.append(row)
    return result


def run(dataset_path,output):
    output.mkdir(parents=True,exist_ok=False)
    dataset=json.loads((dataset_path/'dataset.json').read_text());matrix=np.load(dataset_path/'features.npy',allow_pickle=False)
    if any(s not in (*PRIMARY,'S02') for s in dataset['sessions']):raise PermissionError('Unexpected session')
    vectors=event_matrix(dataset,matrix);oof=[];folds=[]
    for hold in PRIMARY:
        train=[s for s in PRIMARY if s!=hold]
        model=train_event(dataset,vectors,train);event_scores=predict(model,vectors)
        mapping={e['event']:float(score) for e,score in zip(dataset['events'],event_scores)}
        patch_model=fit_model(*training_data(dataset,matrix,train,'union'),'logistic')
        rows=event_predictions(dataset,predict(patch_model,matrix),[hold],'union')
        rows=gate(rows,mapping,0);oof.extend(rows)
        gated=gate(rows,mapping,.5)
        folds.append(dict(hold=hold,ungated=summarize(rows),balanced_gate=summarize(gated),rows=gated))
    conservative=min(r['visual_event_score'] for r in oof if r['truth']['state']=='SINGLE_IMPACT')
    event_model=train_event(dataset,vectors,PRIMARY)
    scores=predict(event_model,vectors);mapping={e['event']:float(score) for e,score in zip(dataset['events'],scores)}
    patch_model=fit_model(*training_data(dataset,matrix,PRIMARY,'union'),'logistic')
    patch_scores=predict(patch_model,matrix);trials=[]
    for pool in ('current','union'):
        rows=event_predictions(dataset,patch_scores,['S02'],pool)
        for name,threshold in (('balanced',.5),('preserve_all_primary_oof_physical',conservative)):
            gated=gate(rows,mapping,threshold)
            trials.append(dict(pool=pool,policy=name,threshold=threshold,metrics=summarize(gated),rows=gated))
    report=dict(status='OFFLINE_RESEARCH_ONLY',dataset=str(dataset_path),feature_names=EVIDENCE_NAMES,
                feature_contract='Per-event .5/.9/.99 quantiles of common causal image features on bounded expanded proposals; no audio interval, amplitude, source identity or GT coordinates as inputs',
                primary_folds=folds,primary_balanced=summarize([r for f in folds for r in f['rows']]),
                S02=trials,hashes=save_model(event_model,output/'event_model'),
                caveat='Only three primary no-physical events; calibration cannot establish robust rejection')
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('primary',json.dumps(report['primary_balanced']),flush=True)
    for t in trials:print(t['pool'],t['policy'],json.dumps(t['metrics']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.dataset,args.output)
