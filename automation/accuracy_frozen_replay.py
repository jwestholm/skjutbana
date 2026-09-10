"""Hash-checked S02 replay of a frozen offline reference. Never installs a live model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from automation.accuracy_physical_dataset import guard_path, json_read
from automation.accuracy_verifier_research import event_predictions, predict, summarize


def checked(path,expected):
    guard_path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError(f'Frozen artifact hash mismatch: {path.name}')
    return path


def replay(manifest_path,output):
    manifest=json_read(manifest_path)
    if manifest['status']!='FROZEN_OFFLINE_REFERENCE_NOT_LIVE_AUTHORITY':raise ValueError('Invalid research status')
    for item in manifest['artifacts']:checked(Path(item['path']),item['sha256'])
    dataset_path=Path(manifest['dataset']);dataset=json_read(dataset_path/'dataset.json')
    if any(s not in ('H10','H20','POST_FIX','S01','S02') for s in dataset['sessions']):raise PermissionError('Unexpected session')
    matrix=np.load(dataset_path/'features.npy',allow_pickle=False)
    params=np.load(manifest['parameters'],allow_pickle=False)
    model=dict(family='logistic',**{name:params[name] for name in params.files})
    scores=predict(model,matrix)
    rows=event_predictions(dataset,scores,['S02'],'current',manifest['rejection_threshold'])
    compact=[dict(event=row['event'],selected_xy=None if row['selected'] is None else
                  [row['selected']['camera_x'],row['selected']['camera_y']],result=row['result']) for row in rows]
    digest=hashlib.sha256(json.dumps(compact,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=manifest['expected_decisions_sha256']:raise ValueError('Frozen decisions changed')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(status=manifest['status'],name=manifest['name'],metrics=summarize(rows),decisions=compact,
                decisions_sha256=digest,limitations=manifest['limitations'])
    (output/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(manifest['name'],len(rows),'deterministic decisions',digest,flush=True)
    return result


def evaluate_external(manifest_path, dataset_path, session, output):
    """Verify the frozen reference first, then score an explicit new development dataset."""
    if session not in ('S01', 'S02', 'POST_FIX', 'D01'):
        raise PermissionError('Only explicit physical development sessions are permitted')
    dataset = json_read(dataset_path/'dataset.json')
    if any(s not in ('H10', 'H20', 'POST_FIX', 'S01', 'S02', 'D01') for s in dataset['sessions']):
        raise PermissionError('Unexpected evaluation session')
    manifest = json_read(manifest_path)
    original = json_read(Path(manifest['dataset'])/'dataset.json')
    if dataset['feature_names'] != original['feature_names'] or dataset['reference'] != original['reference']:
        raise ValueError('Frozen feature/reference schema differs from evaluation dataset')
    if session not in dataset['sessions']:
        raise ValueError('Requested session is absent from dataset')
    output.mkdir(parents=True, exist_ok=False)
    baseline = replay(manifest_path, output/'frozen_reference_recheck')
    matrix = np.load(dataset_path/'features.npy', allow_pickle=False)
    params = np.load(manifest['parameters'], allow_pickle=False)
    model = dict(family='logistic', **{name: params[name] for name in params.files})
    scores = predict(model, matrix)
    results = {}
    for pool in ('current', 'expanded', 'union'):
        rows = event_predictions(dataset, scores, [session], pool, manifest['rejection_threshold'])
        results[pool] = dict(metrics=summarize(rows), rows=rows)
    result = dict(status='FROZEN_PARAMETERS_OFFLINE_DEVELOPMENT_EVALUATION', session=session,
                  model=manifest['name'], training_sessions=manifest['training_sessions'],
                  dataset=str(dataset_path), reference_decisions_sha256=baseline['decisions_sha256'],
                  dataset_sha256=hashlib.sha256((dataset_path/'dataset.json').read_bytes()).hexdigest(),
                  features_sha256=hashlib.sha256((dataset_path/'features.npy').read_bytes()).hexdigest(),
                  results=results, limitations=manifest['limitations'])
    (output/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(session, {pool: value['metrics']['hits'] for pool, value in results.items()}, flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evaluation-dataset', type=Path, help='Optional new physical dataset; first verifies the original frozen reference')
    parser.add_argument('--session', choices=('S01','S02','POST_FIX','D01'), help='Required with --evaluation-dataset; no fitting')
    args=parser.parse_args()
    if bool(args.evaluation_dataset) != bool(args.session):
        parser.error('--evaluation-dataset and --session must be supplied together')
    if args.evaluation_dataset:
        evaluate_external(args.manifest, args.evaluation_dataset, args.session, args.output)
    else:
        replay(args.manifest,args.output)
