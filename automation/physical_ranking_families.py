"""Cache-only evidence-family audit. No physical trace discovery or live policy.

Family labels describe shared measurements, not independent sensors. In particular
all visual families ultimately reuse the same camera frames.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

from automation.physical_hypothesis_research import digest, frozen_cache
from automation.physical_counterfactual_models import universe
from automation.physical_counterfactual_research import SESSIONS, RADII, write_json


def install_read_guard(output):
    """Deny source captures and validation paths before open/list operations."""
    accessed = set()
    output = Path(output).resolve()
    def audit(event, args):
        if event not in ('open', 'os.listdir', 'os.scandir') or not args:
            return
        value = args[0]
        if not isinstance(value, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(value)).absolute()
        # No source capture may be opened, even through the project symlink.
        if 'physical_traces' in path.parts or any(re.search(r'(^|_)S03(_|$)', p, re.I) for p in path.parts):
            raise PermissionError('Research forbids source capture access: '+str(path))
        resolved = path.resolve()
        if 'physical_traces' in resolved.parts:
            raise PermissionError('Research forbids resolved capture access: '+str(path))
        if '/data/skjutbana/' in str(resolved) and not resolved.is_relative_to(output):
            if event == 'open' and len(args) > 2 and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                raise PermissionError('Research input is read-only: '+str(path))
            accessed.add(str(resolved))
    sys.addaudithook(audit)
    return accessed


def family(name):
    if name.startswith(('history_', 'representative_')):
        return 'history'
    if name.endswith(('_pre_std', '_pre_peak', '_pre_texture', '_noise', '_edge')):
        return 'pre_context'
    if '_clean_' in name:
        return 'registered_residual'
    if name.startswith(('morph_', 'jitter_')):
        return 'morphology'
    if name.endswith(('center_ring_ratio', 'spatial_consistency')):
        return 'localization'
    if name.endswith(('onset_gain', 'persistence', 'polarity_agreement', 'early_late_agreement', 'contrast_std')):
        return 'temporal'
    return 'new_change'


def tied_ranks(values):
    """Midrank percentile; constants are neutral, ties never gain array-order votes."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Ranks require a nonempty finite vector')
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends-counts/2)/len(a))[inverse]


def weighted_correlation(matrix, weights):
    weights = np.asarray(weights, dtype=float); weights = weights/weights.sum()
    centered = matrix-np.average(matrix, axis=0, weights=weights)
    cov = centered.T@(centered*weights[:, None])
    sd = np.sqrt(np.maximum(0, np.diag(cov)))
    return np.divide(cov, sd[:, None]*sd[None], out=np.zeros_like(cov), where=sd[:, None]*sd[None] > 1e-12)


def session_rows(data, sessions):
    events = [e for e in data['events'] if e['session'] in sessions]
    counts = Counter(e['session'] for e in events)
    ids, weights = [], []
    for e in events:
        pool = [i for i in universe(e, data['records']) if data['records'][i]['ready']]
        ids.extend(pool); weights.extend([1/(counts[e['session']]*len(pool))]*len(pool))
    return ids, np.array(weights)


def load_inputs(base, prior):
    data, matrix, hashes = frozen_cache(base)
    paths = [base/'cache/dataset.json', base/'cache/features.npy', base/'models/models.json',
             base/'forensics/pairs.json', base/'forensics/survival.json',
             prior/'final/hypotheses.json', prior/'final/retention_ledger.json',
             prior/'final/results.json', prior/'review.json']
    details = {}
    expected = json.loads((prior/'final/cached_detail_hashes.json').read_text())
    for e in data['events']:
        p = base/'cache'/(e['event'].replace(':', '_')+'_trace_details.json')
        if digest(p) != expected[str(p)]:
            raise ValueError('Cached detail changed: '+e['event'])
        details[e['event']] = json.loads(p.read_text()); paths.append(p)
        if max(e['context']['pre_times']+e['context']['post_times']) > e['context']['cutoff']:
            raise ValueError('Future frame in cached context')
    models = {}
    for meta in json.loads((base/'models/models.json').read_text()):
        if meta['method'] != 'PAIR_CLEAN':
            continue
        if set(meta['train']) != set(SESSIONS)-{meta['held_out']}:
            raise ValueError('Whole-session exclusion mismatch')
        p = base/'models'/('PAIR_CLEAN_'+meta['held_out'])/'parameters.npz'
        if digest(p) != meta['hashes']['parameters.npz']:
            raise ValueError('Frozen model changed')
        with np.load(p, allow_pickle=False) as m:
            models[meta['held_out']] = dict(meta=meta, **{k:m[k].copy() for k in ('mu','sd','beta')})
        paths.append(p)
    return data, matrix.astype(float), details, models, {str(p):digest(p) for p in paths}


def frozen_contributions(matrix, model):
    cols = model['meta']['feature_indices']
    z = np.clip((matrix[:, cols]-model['mu'])/model['sd'], -8, 8)
    return z*model['beta'][1:]


def family_totals(contributions, names):
    groups = sorted(set(map(family, names)))
    return groups, np.column_stack([contributions[:, [i for i,n in enumerate(names) if family(n)==g]].sum(axis=1) for g in groups])


def correlation_audit(data, matrix, output):
    names = data['feature_names']; rows = []
    scopes = [('all', SESSIONS)]+[(s,(s,)) for s in SESSIONS]+[('train_without_'+s, tuple(t for t in SESSIONS if t!=s)) for s in SESSIONS]
    for scope, sessions in scopes:
        ids, weights = session_rows(data, sessions)
        x = matrix[ids]
        corr = weighted_correlation(x, weights)
        ranked = np.column_stack([tied_ranks(x[:, j]) for j in range(x.shape[1])])
        spearman = weighted_correlation(ranked, weights)
        np.savez_compressed(output/(scope+'_correlations.npz'), pearson=corr, spearman=spearman)
        pairs = [(i,j) for i in range(len(names)) for j in range(i) if abs(spearman[i,j])>=.95]
        exact = [(i,j) for i,j in pairs if np.array_equal(x[:,i],x[:,j])]
        families = {}
        for g in sorted(set(map(family,names))):
            cols = [i for i,n in enumerate(names) if family(n)==g]
            eig = np.maximum(0, np.linalg.eigvalsh(spearman[np.ix_(cols,cols)]))
            families[g] = dict(columns=len(cols),effective_rank=float(eig.sum()**2/max(1e-12,np.sum(eig**2))))
        rows.append(dict(scope=scope,sessions=sessions,candidates=len(ids),weighting='equal session and event mass',
            high_correlation_count=len(pairs),exact_duplicate_count=len(exact),families=families,
            high_pairs=[dict(a=names[i],b=names[j],rho=float(spearman[i,j]),same_family=family(names[i])==family(names[j]))
                for i,j in sorted(pairs,key=lambda p:-abs(spearman[p]))],
            exact_duplicates=[[names[i],names[j]] for i,j in exact]))
    write_json(output/'correlations.json', rows)
    return rows


def atlas(data, matrix, details, models, base, output):
    names = data['feature_names']; records = data['records']; result = []
    ledger = {r['event']:r for r in json.loads((base/'forensics/survival.json').read_text())['events']}
    for e in data['events']:
        if e['truth']['state'] != 'SINGLE_IMPACT':
            continue
        model = models[e['session']]; cols = model['meta']['feature_indices']
        contrib = frozen_contributions(matrix,model); scores = contrib.sum(axis=1)
        families, f = family_totals(contrib,[names[j] for j in cols])
        ids = [i for i in universe(e,records) if records[i]['ready']]
        current = e['current_id']; pool = e['pools']['current']
        experimental = max([i for i in pool if records[i]['ready']],key=lambda i:(scores[i],-i))
        good = min(ids,key=lambda i:records[i]['gt_distance'])
        def describe(index):
            c = records[index]
            return dict(record=c,features=dict(zip(names,matrix[index].tolist())),
                family_contributions=dict(zip(families,f[index].tolist())),
                rank_evidence=[r for r in e['representations'] if index in r['observations']],
                associations=[dict(frame_ts=b['frame_ts'],observed_at=b.get('observed_at'),record=r)
                    for b in details[e['event']]['snapshot']['associations'] for r in b['records']
                    if (r['candidate']['camera_x'],r['candidate']['camera_y'])==(c['camera_x'],c['camera_y'])])
        comparisons=[]
        for winner_kind, winner in [('CURRENT',current),('PAIR_CLEAN',experimental)]:
            delta=f[good]-f[winner]
            comparisons.append(dict(winner_kind=winner_kind,winner_id=winner,
                good_minus_winner_score=float(scores[good]-scores[winner]),
                family_delta=dict(zip(families,delta.tolist())),
                supportive_families=[g for g,d in zip(families,delta) if d>1e-9],
                opposing_families=[g for g,d in zip(families,delta) if d < -1e-9],
                strongest_opposing_family=families[int(np.argmin(delta))] if np.min(delta)<0 else None,
                single_family_overturn=bool(np.sum(delta)<0 and np.sum(delta)-np.min(delta)>0)))
        within={}
        for radius in RADII:
            allowed=[i for i in ids if records[i]['gt_distance']<=radius]
            within[str(radius)] = None if not allowed else dict(nearest=min(allowed,key=lambda i:records[i]['gt_distance']),
                best_frozen_physical_score=max(allowed,key=lambda i:(scores[i],-i)),count=len(allowed))
        result.append(dict(event=e['event'],session=e['session'],truth=e['truth'],
            current=describe(current),experimental=describe(experimental),nearest_observed=describe(good),
            correct_at_tolerances=within,comparisons=comparisons,survival=ledger[e['event']],
            eligible_oracle42=any(records[i]['gt_distance']<=42 for i in pool),
            current_hit42=records[current]['gt_distance']<=42,
            observed_oracle42=records[good]['gt_distance']<=42,
            semantics='GT-referenced forensic comparison only; CURRENT authority is its recorded rank key, not this model'))
    write_json(output/'failure_atlas.json',dict(events=result,feature_encoding='signed log1p',
        source_metadata='Retained for diagnostics only; no source/GT/absolute-coordinate columns in ranker',
        context_limitations='No trusted HoleMap/projected-edge/board-seam identity; historical delivery caveats preserved'))
    counters=Counter()
    for r in result:
        state='CURRENT_correct42' if r['current_hit42'] else 'eligible_coordinate_missing42' if not r['eligible_oracle42'] else 'available_coordinate_loses'
        counters[state]+=1
        for p in r['comparisons']:
            if records[p['winner_id']]['gt_distance']>42 and r['observed_oracle42']:
                counters[p['winner_kind']+'_strongest_opposition_'+str(p['strongest_opposing_family'])]+=1
                counters[p['winner_kind']+'_single_family_overturn']+=int(p['single_family_overturn'])
    write_json(output/'atlas_counts.json',dict(counters))
    return counters


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    accessed=install_read_guard(a.output)
    data,matrix,details,models,hashes=load_inputs(a.base,a.prior)
    write_json(a.output/'protocol.json',dict(status='OFFLINE_ONLY',sessions=SESSIONS,input_hashes=hashes,
        command=sys.argv,source_sha256=digest(Path(__file__)),
        datasets=[dict(event=e['event'],source_session=Path(e['trace_path']).parts[-4],trace_hash=e['trace_sha256'],
            label_hash=e['label_sha256'],cutoff=e['context']['cutoff'],timestamp_semantics=e['context']['timestamp_semantics']) for e in data['events']],
        hypothesis='Audit shared evidence and model contributions before choosing robust structures'))
    corr=correlation_audit(data,matrix,a.output)
    counts=atlas(data,matrix,details,models,a.base,a.output)
    write_json(a.output/'input_accesses.json',sorted(accessed))
    assert all(digest(Path(name))==expected for name,expected in hashes.items())
    print('ATLAS',dict(counts),flush=True)
    print('CORRELATIONS',[(r['scope'],r['high_correlation_count'],r['exact_duplicate_count']) for r in corr],flush=True)


if __name__=='__main__':main()
