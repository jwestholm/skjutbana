"""Explicit historical parity experiment (not an accuracy/promotion gate).

Run each mode in its own process for independent ru_maxrss measurements.
Legacy loading is limited to 0.6 GB JSON input; it is never used by nightly.
"""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from . import train as t
from .candidate_cache import Export


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True); p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=['legacy', 'cached', 'reference'], required=True)
    p.add_argument('--subset', type=int, default=3); p.add_argument('--cache-root', required=True)
    a = p.parse_args(argv); m = json.loads(Path(a.manifest).read_text()); output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    start = time.monotonic(); tr = [e['export'] for e in m['train'][:a.subset]]; ho = [e['export'] for e in m['holdout']]
    if a.mode == 'cached':
        result = t.train(tr, ho, output, cache_root=a.cache_root)['metrics']
        print(json.dumps(result, indent=2)); return 0
    if a.mode == 'legacy':
        if sum((Path(d)/'candidates.jsonl').stat().st_size for d in tr+ho) > 0.6e9:
            raise ValueError('Legacy verification input exceeds 0.6 GB safety cap')
        _, rows, _ = t._load(tr); shots, candidates, _ = t._load(ho)
        rows = [r for r in rows if r['producer_has_impact']]; names = t._feature_names(rows)
        X, med = t._matrix(rows, names); mu = X.mean(0); sd = X.std(0); sd = np.where(sd>1e-9,sd,1)
        w, b = t._fit_logistic((X-mu)/sd, np.array([float(r['label']) for r in rows]))
        H, _ = t._matrix(candidates, names, med); scores = (H-mu)/sd@w+b
        live = np.array([float(r['candidate'].get('score',0) or 0) for r in candidates])
        mask = np.array([r['producer_has_impact'] for r in candidates]); y=np.array([r['label'] for r in candidates])
        result = dict(legacy_global_group_top1=dict(model=t._top1(shots,candidates,scores),live_score=t._top1(shots,candidates,live)), auc=t._auc(scores[mask],y[mask]))
    else:
        model = json.loads(Path(m['reference_model']).read_text()); names=model['features']
        exports = [Export(d,a.cache_root) for d in ho]; scores=[]; live=[]
        for e in exports:
            ix={k:j for j,k in enumerate(e.manifest['features'])}
            for lo in range(0,len(e.meta),8192):
                n=min(8192,len(e.meta)-lo); X=np.full((n,len(names)),np.nan)
                for j,k in enumerate(names):
                    if k in ix: X[:,j]=e.X[lo:lo+n,ix[k]]
                X=np.where(np.isfinite(X),X,model['impute_medians'])
                scores.extend(((X-model['mean'])/model['std']@np.array(model['weights'])+model['bias']).tolist())
            live.extend(e.meta['live_score'].tolist())
        scores=np.array(scores); live=np.array(live)
        result=dict(source_scoped_top1=dict(model=t._cached_top1(exports,scores),live_score=t._cached_top1(exports,live)),
                    legacy_global_group_top1=dict(model=t._cached_top1(exports,scores,source_scoped=False),live_score=t._cached_top1(exports,live,source_scoped=False)))
    result['resources']=dict(peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,elapsed_s=time.monotonic()-start)
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0

if __name__=='__main__': raise SystemExit(main())
