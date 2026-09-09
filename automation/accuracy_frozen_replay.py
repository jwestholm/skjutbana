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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();replay(args.manifest,args.output)
