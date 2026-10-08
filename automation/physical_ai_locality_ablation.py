"""One frozen locality-midrank ablation; no new evidence or search space."""
import argparse
from pathlib import Path
import signal
import time

import numpy as np

from automation import physical_ai_dataset as ds, physical_ai_fusion as fusion
from automation import physical_ai_selector_gap as gap, physical_ai_event_selector as selector

PRIOR = Path('/data/skjutbana/evaluation_runs/core_ring_locality_20261008_01/feature_test')
BASELINE = Path('/data/skjutbana/evaluation_runs/selector_gap_20261007_01/linear/result.json')
PHYSICAL = gap.PHYSICAL + ('core_ring_concentration',)
WITH_NAMES = PHYSICAL + tuple(n+'__missing' for n in PHYSICAL) + tuple(n+'__event_midrank' for n in PHYSICAL)
NAMES = WITH_NAMES[:-1]


def project(physical):
    """Keep all baseline ranks; never compute a locality rank for inference."""
    if physical.ndim != 2 or physical.shape[1] != 2*len(PHYSICAL) or not np.isfinite(physical).all():
        raise ValueError('Frozen finite locality schema required')
    n = len(gap.PHYSICAL)
    baseline = np.column_stack((physical[:,:n], physical[:,n+1:2*n+1]))
    baseline_ranks = selector.relative_features(baseline)[:,2*n:]
    return np.column_stack((physical, baseline_ranks)).astype(np.float32)


def predict(model, physical):
    if tuple(model.feature_names) != NAMES:
        raise ValueError('Midrank-ablated schema required')
    scores = model.score_matrix(project(physical))
    best = int(np.argmax(scores)) if len(scores) else None
    return (best if best is not None and scores[best] > 0 else None), scores


def load_inputs():
    data, raw = fusion.load(gap.crossfit.DEFAULT)
    old = ds.read(PRIOR/'seed_1729/result.json')
    if old['identity']['dataset_hash'] != data['dataset_hash'] or old['identity']['features'] != list(WITH_NAMES):
        raise ValueError('Original locality experiment identity differs')
    if ds.digest(PRIOR/'features.npy') != old['identity']['feature_hash']:
        raise ValueError('Frozen locality features changed')
    physical = np.load(PRIOR/'features.npy', allow_pickle=False)
    n = len(gap.PHYSICAL)
    baseline = np.column_stack((physical[:,:n], physical[:,n+1:2*n+1]))
    if not np.array_equal(baseline, gap.physical_matrix(raw)):
        raise ValueError('Original safe physical inputs changed')
    for e in data['events'] + [e for f in data['folds'] for e in gap.train_events(data,f)]:
        original = selector.relative_features(physical[e['indices']], PHYSICAL)
        if not np.array_equal(project(physical[e['indices']]), original[:,:-1]):
            raise ValueError('Ablation changed more than one column')
    return data, physical, old


def run(output, stop=lambda:False):
    output = ds.output_root(output); output.mkdir(exist_ok=True, parents=True)
    data, physical, old = load_inputs(); baseline = ds.read(BASELINE)
    config = baseline['config']
    if config != old['config'] or config['kind'] != 'linear':
        raise ValueError('Comparator training configurations differ')
    identity = dict(dataset=data['dataset_hash'], input_sha256=old['identity']['feature_hash'],
        prior_result_sha256=ds.digest(PRIOR/'seed_1729/result.json'), baseline_sha256=ds.digest(BASELINE),
        config=config, seeds=[1729,1730,1731], features=list(NAMES), removed=WITH_NAMES[-1],
        code={str(p):ds.digest(p) for p in (Path(__file__),Path(selector.__file__),Path(gap.__file__))})
    intent = output/'intent.json'
    if intent.exists() and ds.read(intent) != identity: raise ValueError('Run identity changed')
    if not intent.exists(): ds.atomic_json(intent, identity)
    results = []
    for seed in identity['seeds']:
        root=output/f'seed_{seed}'; root.mkdir(exist_ok=True); final=root/'result.json'
        if final.exists():
            result=ds.read(final)
            if result['identity'] != identity: raise ValueError('Completed trial identity changed')
            results.append(result); continue
        start=time.monotonic(); predictions={}; reports=[]
        for i,fold in enumerate(data['folds']):
            path=root/f'fold_{i:02d}'; path.mkdir(exist_ok=True); done=path/'complete.json'
            if done.exists():
                saved=ds.read(done)
                if saved['identity'] != identity or saved['fold'] != fold: raise ValueError('Fold changed')
                for name,h in saved['hashes'].items():
                    if ds.digest(path/name) != h: raise ValueError('Fold model changed')
            else:
                if stop(): raise InterruptedError('At fold boundary')
                train=gap.train_events(data,fold); cfg=dict(config,seed=seed+i)
                samples=[(project(physical[e['indices']]),gap.positive_mask(e)) for e in train]
                model,history=selector.fit(samples,cfg,path,dict(run=identity,fold=fold,training_events=[e['event'] for e in train]),stop,feature_names=NAMES)
                # The other 50 TRAIN statistics must match the old 51-column fit.
                previous=selector.RankModelV223.load(PRIOR/f'seed_{seed}/fold_{i:02d}')
                if not np.array_equal(model.mean,previous.mean[:-1]) or not np.array_equal(model.scale,previous.scale[:-1]):
                    raise ValueError('TRAIN scaler differs beyond removed column')
                model.save(path); pp={}; detail=[]
                for e in data['events']:
                    if e['session'] not in fold['held']: continue
                    j,scores=predict(model,physical[e['indices']]); xy=None if j is None else e['candidates'][j]['xy']
                    pp[e['event']]=xy
                    detail.append(dict(event=e['event'],candidate_ids=[c['id'] for c in e['candidates']],scores=scores.tolist(),selected_xy=xy))
                saved=dict(identity=identity,fold=fold,training_events=[e['event'] for e in train],predictions=pp,detail=detail,history=history,
                    retained_train_scaler_exact=True,hashes={n:ds.digest(path/n) for n in ('model.json','model.npz')})
                ds.atomic_json(done,saved)
            predictions.update(saved['predictions']); reports.append(saved)
        metrics=selector.extra_metrics(ds.evaluate(data['events'],predictions))
        success=lambda r:r['error'] is not None and r['error']<=5
        before={r['event']:r for r in baseline['metrics']['per_event']}
        result=dict(identity=identity,seed=seed,config=dict(config,seed=seed),metrics=metrics,predictions=predictions,folds=reports,
            elapsed_seconds=time.monotonic()-start,comparison=fusion.compare(metrics,baseline['metrics']),
            gains=[r['event'] for r in metrics['per_event'] if success(r) and not success(before[r['event']])],
            losses=[r['event'] for r in metrics['per_event'] if not success(r) and success(before[r['event']])])
        ds.atomic_json(final,result); results.append(result)
        print(seed,metrics['selected'],'controls',metrics['control_false_emissions'],'unknown',metrics['unknown'],'>100',metrics['catastrophic_gt100'],'gains',result['gains'],'losses',result['losses'],flush=True)
    ds.atomic_json(output/'comparison.json',dict(baseline=baseline['metrics'],results=results))
    return results


def main():
    ds.install_guard(); p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    stopped=[False]
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,lambda *_:stopped.__setitem__(0,True))
    try:run(a.output,lambda:stopped[0])
    except InterruptedError:print('Paused; repeat exact command to resume.')


if __name__=='__main__':main()
