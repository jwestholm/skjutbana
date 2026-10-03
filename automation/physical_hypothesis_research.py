"""Cache-only track/XY grouping and bounded retention follow-up. No live policy."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from automation.physical_counterfactual_models import late_reserve, spatial_order, universe
from automation.physical_counterfactual_research import SESSIONS, RADII, write_json, xy
from automation.accuracy_verifier_research import summarize
from src.engine.offline.observation_hypotheses import observed_hypotheses
from src.engine.offline.physical_event_truth import score_event


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def frozen_cache(base):
    """Bind to the completed pass's recorded feature hashes, never current traces."""
    protocol = json.loads((base/'models/protocol.json').read_text())
    for name, expected in protocol['cache_hashes'].items():
        if digest(base/'cache'/name) != expected:
            raise ValueError('Frozen cache hash mismatch: '+name)
    data = json.loads((base/'cache/dataset.json').read_text())
    if tuple(data['sessions']) != SESSIONS or any(e['session'] not in SESSIONS for e in data['events']):
        raise ValueError('Unexpected session; no trace opened')
    return data, np.load(base/'cache/features.npy', allow_pickle=False), protocol['cache_hashes']


def coverage_reserve(event, records, budget=48, cap=200):
    """Preserve current; maximin raw-contour coverage without source/GT scores.

    This tests whether coarse-cell score queues crowd out weak spatial evidence.
    It is a coverage baseline, not proof of physical plausibility or a live rule.
    Distances are normalized by crop size; only actually observed plausible raw
    contours participate. O(budget * raw_count), <=48 extra coordinates.
    """
    current = list(event['pools']['current'])
    if budget < 0 or len(current) > cap:
        raise ValueError('Invalid bounded retention')
    candidates = [i for i in event['pools'].get('raw_plausible', [])
                  if i not in set(current) and records[i]['selectable']]
    if not candidates:
        return current, []
    scale = np.array(event['context']['crop'][2:])
    positions = np.array([xy(records[i]) for i in candidates])/scale
    anchors = np.array([xy(records[i]) for i in current])/scale
    distances = np.min(np.sum((positions[:,None]-anchors[None])**2, axis=2),axis=1)
    selected = list(current); audit = []
    for _ in range(min(budget, cap-len(current))):
        # Numerical ties must not depend on a translation of the camera origin.
        j = int(np.argmax(np.round(distances, 14)))
        if not math.isfinite(float(distances[j])) or distances[j] <= 0:
            break
        index = candidates[j]
        audit.append(dict(index=index, operation='maximin_raw_coverage', normalized_distance=math.sqrt(float(distances[j]))))
        selected.append(index)
        distances = np.minimum(distances, np.sum((positions-positions[j])**2, axis=1))
        distances[j] = -math.inf
    return selected, audit


def reserve_forensics(event, records, matrix, names):
    """Audit all candidates, then join GT only for reporting nearest witnesses."""
    current = set(event['pools']['current'])
    candidates = [i for i in universe(event, records) if i not in current]
    result = {}
    physical = lambda i: records[i]['physical_priority']
    for mode in ('physical','weak_strata','clean_families'):
        audit = {}; pool = late_reserve(event, records, matrix, names, mode, audit=audit)
        if mode == 'physical':
            queues = [spatial_order(candidates, records, event, physical)]
        elif mode == 'weak_strata':
            ordered = sorted(candidates, key=lambda i:(physical(i),i))
            queues = [spatial_order(part.tolist(),records,event,physical) for part in np.array_split(ordered,3)]
        else:
            queues=[]
            for family in ('persistent','pre_centered','piecewise'):
                cols=[names.index(family+'_clean_r4_'+key) for key in ('mean','inner_ring','noise')]
                score=lambda i: float(matrix[i,cols[0]]+matrix[i,cols[1]]-matrix[i,cols[2]])
                queues.append(spatial_order(candidates,records,event,score))
        audit['queues'] = queues
        audit['queue_consumed'] = [sum(d['queue']==q for d in audit['decisions']) for q in range(len(queues))]
        audit['pool'] = pool
        result[mode] = audit
    return result


def group_order(groups, scores, mode, current_track_id):
    """Separate track evidence aggregation from within-track coordinate choice."""
    ranked = []
    for track_id, ids in groups.items():
        unique = sorted(set(ids), key=lambda i:(-scores[i],i))
        if not unique:
            continue
        values = scores[unique]
        if mode == 'mean':
            value = float(np.mean(values))
        elif mode == 'max':
            value = float(np.max(values))
        elif mode == 'logmeanexp':
            maximum = float(np.max(values))
            value = maximum + math.log(float(np.mean(np.exp(values-maximum))))
        elif mode == 'current_identity':
            value = 1. if track_id == current_track_id else 0.
        else:
            raise ValueError(mode)
        ranked.append((value, track_id, unique))
    ranked.sort(key=lambda row:(-row[0], row[1]))
    return list(dict.fromkeys(i for _,_,ids in ranked for i in ids)), ranked


def row(event, records, ordered, pool, method, retention_ms, ranking_ms):
    selected = records[ordered[0]] if ordered else None
    result = score_event(event['truth'], selected)
    physical = event['truth']['state'] == 'SINGLE_IMPACT'
    distances = [records[i]['gt_distance'] for i in ordered] if physical else []
    oracle = {str(r):any(records[i]['gt_distance'] <= r for i in pool) if physical else False for r in RADII}
    return dict(event=event['event'],session=event['session'],method=method,truth=event['truth'],result=result,
        selected=None if selected is None else {k:selected[k] for k in ('index','camera_x','camera_y','gt_distance')},
        oracle=oracle,ranks={str(r):next((j+1 for j,d in enumerate(distances) if d<=r),None) for r in RADII},
        candidate_count=len(pool),unready_at_recorded_decision=False,
        retention_ms=retention_ms,ranking_ms=ranking_ms,
        semantics='Conditional physical coordinate ranking; event-level NO_IMPACT gate intentionally absent')


def run(base, output):
    data, matrix, hashes = frozen_cache(base)
    output.mkdir(parents=True,exist_ok=False)
    records=data['records']; names=data['feature_names']; rows=[]; ledgers={}; hypotheses={}; pairs=[]
    detail_hashes={}; model_hashes={}
    models=json.loads((base/'models/models.json').read_text())
    for meta in models:
        if meta['method'] not in ('PAIR_CORE','PAIR_CLEAN'):
            continue
        if set(meta['train']) != set(SESSIONS)-{meta['held_out']}:
            raise ValueError('Model whole-session exclusion mismatch')
        path=base/'models'/(meta['method']+'_'+meta['held_out'])/'parameters.npz'
        if digest(path) != meta['hashes']['parameters.npz']:
            raise ValueError('Frozen model hash mismatch')
        model_hashes[str(path)]=digest(path)
    write_json(output/'protocol.json',dict(status='OFFLINE_ONLY_NO_PROMOTION',sessions=SESSIONS,
        data_version='September 10 frozen counterfactual cache, original September 9 D01',cache_hashes=hashes,
        live_D01_paths='INVALID for original labels: September 11 runtime overwrote shots 1-6; not used',
        hypotheses='current + first/best locally confirmed/latest actual alternative; 32 extras event-wide',
        retention='one fixed maximin normalized raw-contour coverage baseline; <=48 extra <=200 total',
        ranking='reuse frozen whole-session PAIR_CORE/CLEAN; grouped max/mean/logmeanexp and fixed CURRENT identity',
        no_impact='ranking is conditional; do not derive an acceptance gate',models=model_hashes,
        source_hashes={str(p):digest(p) for p in [Path(__file__),Path('src/engine/offline/observation_hypotheses.py')]}))
    for event in data['events']:
        detail_path=base/'cache'/(event['event'].replace(':','_')+'_trace_details.json')
        detail_hashes[str(detail_path)]=digest(detail_path)
        details=json.loads(detail_path.read_text())
        if details['snapshot']['event']['peak_ts'] != event['context']['peak']:
            raise ValueError('Cached snapshot event mismatch')
        started=time.perf_counter()
        hyp=observed_hypotheses(details['snapshot'],event['context']['cutoff'])
        lookup={xy(records[i]):i for i in event['indices'] if records[i]['selectable']}
        groups=defaultdict(list); unavailable=[]
        for h in hyp['hypotheses']:
            index=lookup.get(xy(h)); h['cached_index']=index
            if index is None:
                unavailable.append(h);continue
            # Cached uniform same-XY support remains a research probe. Explicit
            # local proof is retained separately on h; never inherited from parent.
            if records[index]['ready']:
                groups[h['track_id']].append(index)
        ids=list(dict.fromkeys(i for g in groups.values() for i in g))
        hypothesis_ms=(time.perf_counter()-started)*1000
        hyp.update(unavailable_patches=unavailable,ready_pool=ids,build_ms=hypothesis_ms,
            available_pool=list(dict.fromkeys(h['cached_index'] for h in hyp['hypotheses'] if h['cached_index'] is not None)))
        hypotheses[event['event']]=hyp
        started=time.perf_counter(); coverage,coverage_audit=coverage_reserve(event,records)
        coverage_ms=(time.perf_counter()-started)*1000
        audit=reserve_forensics(event,records,matrix,names)
        audit['coverage']=dict(pool=coverage,decisions=coverage_audit)
        # Diagnostic-only GT join after every retention decision is fixed.
        if event['truth']['state']=='SINGLE_IMPACT':
            nearest=min(universe(event,records),key=lambda i:records[i]['gt_distance'])
            audit['nearest_raw_witness']=dict(index=nearest,xy=xy(records[nearest]),distance=records[nearest]['gt_distance'],
                legacy_rank=next((j+1 for j,c in enumerate(details['stage_positions'].get('legacy_precap',[])) if xy(c)==xy(records[nearest])),None),
                queues={mode:dict(ranks=[q.index(nearest)+1 if nearest in q else None for q in item['queues']],
                    consumed=item['queue_consumed'],decisions=[d for d in item['decisions'] if d['index']==nearest])
                    for mode,item in audit.items() if mode!='coverage'})
        ledgers[event['event']]=audit
        current=event['pools']['current']
        order=[event['current_id']]+[i for i in current if i!=event['current_id']]
        rows.append(row(event,records,order,current,'CURRENT',0.,0.))
        for family in ('PAIR_CORE','PAIR_CLEAN'):
            meta=next(m for m in models if m['method']==family and m['held_out']==event['session'])
            with np.load(base/'models'/(family+'_'+event['session'])/'parameters.npz',allow_pickle=False) as model:
                start=time.perf_counter()
                z=np.clip((matrix[:,meta['feature_indices']]-model['mu'])/model['sd'],-8,8)
                scores=model['beta'][0]+z@model['beta'][1:]
                score_all_ms=(time.perf_counter()-start)*1000
            for mode in ('max','mean','logmeanexp','current_identity'):
                start=time.perf_counter();order,ranked=group_order(groups,scores,mode,event['current_track_id'])
                ranking_ms=(time.perf_counter()-start)*1000
                r=row(event,records,order,ids,family+'_'+mode,hypothesis_ms,ranking_ms)
                r['feature_score_all_cache_ms']=score_all_ms
                r['available_hypothesis_count']=len(hyp['available_pool'])
                r['top_track_margin']=ranked[0][0]-ranked[1][0] if len(ranked)>1 else None
                r['winner_coordinate_logit_range']=float(np.ptp(scores[ranked[0][2]])) if ranked else None
                rows.append(r)
                if event['truth']['state']=='SINGLE_IMPACT':
                    good=min(ids,key=lambda i:records[i]['gt_distance'])
                    owner=next((t for _,t,g in ranked if good in g),None)
                    pairs.append(dict(event=event['event'],method=r['method'],good_id=good,good_distance=records[good]['gt_distance'],
                        good_track=owner,good_track_rank=next((j+1 for j,(_,t,_) in enumerate(ranked) if t==owner),None),
                        selected=r['selected'],good_coordinate_logit=float(scores[good]),
                        selected_logit=float(scores[order[0]]) if order else None))
            start=time.perf_counter();order=sorted([i for i in coverage if records[i]['ready']],key=lambda i:(-scores[i],i))
            rows.append(row(event,records,order,coverage,family+'_coverage48',coverage_ms,(time.perf_counter()-start)*1000))
        print(event['event'],'hypotheses',len(ids),'extra',hyp['extra_count'],'coverage',len(coverage),flush=True)
    summaries=[]
    baseline={r['event']:r for r in rows if r['method']=='CURRENT'}
    for r in rows:
        r['regressed_radii']=[str(n) for n in RADII if baseline[r['event']]['result'].get('hits',{}).get(str(n))
                              and not r['result'].get('hits',{}).get(str(n))]
    for method in sorted({r['method'] for r in rows}):
        for session in SESSIONS:
            group=[r for r in rows if r['method']==method and r['session']==session]
            s=summarize(group)
            s.update(method=method,session=session,errors_gt1000=sum((r['result'].get('error_px') or 0)>1000 for r in group),
                mean_retention_ms=float(np.mean([r['retention_ms'] for r in group])),
                mean_ranking_ms=float(np.mean([r['ranking_ms'] for r in group])),
                available_hypothesis_count=dict(mean=float(np.mean([r.get('available_hypothesis_count',r['candidate_count']) for r in group])),
                    max=max(r.get('available_hypothesis_count',r['candidate_count']) for r in group)),
                regressions=[dict(event=r['event'],radii=r['regressed_radii']) for r in group if r['regressed_radii']])
            summaries.append(s)
    write_json(output/'results.json',dict(summaries=summaries,rows=rows))
    write_json(output/'hypotheses.json',hypotheses)
    write_json(output/'retention_ledger.json',ledgers)
    write_json(output/'joint_pairs.json',pairs)
    write_json(output/'cached_detail_hashes.json',detail_hashes)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.base,args.output)
