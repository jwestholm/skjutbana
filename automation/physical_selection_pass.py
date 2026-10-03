"""Bounded camera-space selector audit using immutable development caches only.

No trace discovery, candidate generation, fitting, game geometry or live install.
Example: python3 -m automation.physical_selection_pass --base BASE --prior PRIOR
         --protocol PRESPECIFIED_JSON --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from automation.accuracy_verifier_research import summarize
from automation.physical_counterfactual_research import SESSIONS, RADII, write_json, xy
from automation.physical_hypothesis_research import digest, row
from automation.physical_ranking_families import (install_read_guard, load_inputs,
    frozen_contributions, family_totals)
from src.engine.camera.hit_scanner import HitScanner
from src.engine.offline.common_verifier import coherent_change_evidence, coherent_event_decision
from src.engine.offline.physical_event_truth import score_event
from src.engine.offline.track_replay import current_exact_replay


METHODS = ('CURRENT', 'COMMON_EARLY', 'FROZEN_PAIR_CLEAN', 'COHERENT', 'COHERENT_PRE',
           'COHERENT_PRE_UNCERTAIN')


def pool_measurement(candidates, truth, *, available=True):
    """GT joins only here, after upstream observations/ordering are fixed."""
    physical = truth['state'] == 'SINGLE_IMPACT'
    gt = truth['impacts'][0] if physical else None
    distances = [math.dist(xy(c), xy(gt)) for c in candidates] if gt else []
    nearest = int(np.argmin(distances)) if distances else None
    return dict(available=available, count=len(candidates) if available else None,
                nearest=None if nearest is None else dict(camera_xy=xy(candidates[nearest]),
                    error_px=distances[nearest], track_id=candidates[nearest].get('track_id'),
                    current_rank=candidates[nearest].get('final_rank')),
                oracle={str(r):any(d <= r for d in distances) if physical and available else None
                        for r in RADII})


def comparison(current, challenger):
    result = {}
    for radius in RADII:
        key = str(radius)
        a = current['result'].get('hits', {}).get(key)
        b = challenger['result'].get('hits', {}).get(key)
        result[key] = None if a is None or b is None else (
            'CORRECT_TO_CORRECT' if a and b else 'REGRESSION' if a else 'GAIN' if b else 'WRONG_TO_WRONG')
    return result


def causal_observations(detail, context):
    """Already-consumed association witnesses, including legacy reconstruction.

    A missing exported proposal stage is not an empty pool. Associations record
    the inputs that produced the verified snapshot. Legacy reconstruction wall
    times are not delivery timestamps and are never used as physical evidence.
    """
    sid = detail['snapshot']['event']['shot_id']
    candidates = []
    for key in ('causal_candidates', 'raw_candidates', 'filtered_candidates', 'retained'):
        candidates.extend(detail['stage_positions'].get(key, []))
    for batch in detail['snapshot'].get('associations', []):
        if batch['frame_ts'] > context['cutoff']:
            continue
        for item in batch['records']:
            c = item['candidate']
            owner = c.get('v2224_producer_shot_id', item.get('producer_shot_id'))
            if owner is None or int(owner) == int(sid):
                candidates.append(c)
    return list({xy(c):c for c in candidates
                 if c.get('timestamp', context['cutoff']) <= context['cutoff']
                 and c.get('v2224_producer_shot_id', sid) in (None, sid)}.values())


def verify_event(event, snapshot, records):
    chosen = current_exact_replay(snapshot, dict(track_id=event['current_track_id'],
                                               **{k:records[event['current_id']][k]
                                                  for k in ('camera_x', 'camera_y')}))
    if snapshot['policy'] != 'BASE':
        raise ValueError('This audit requires the recorded general BASE selector')
    eligible = sorted((t for t in snapshot['tracks'] if t['eligible']), key=lambda t:t['final_rank'])
    by_xy = {xy(records[i]):i for i in event['pools']['current']}
    if set(by_xy) != {xy(t) for t in eligible}:
        raise ValueError('Cached verifier pool does not cover the complete eligible coordinates')
    if [xy(records[i]) for i in event['pools']['current']] != list(dict.fromkeys(xy(t) for t in eligible)):
        raise ValueError('Cached CURRENT order differs from the full snapshot ranks')
    # Compare the current working-tree selector, not just its audit formula.
    scanner = SimpleNamespace(_active_tracks={t['track_id']:SimpleNamespace(**t) for t in snapshot['tracks']},
        association_lead_s=snapshot['association_lead_s'], association_lag_s=snapshot['association_lag_s'])
    selected = HitScanner._best_track_for_event(scanner, SimpleNamespace(**snapshot['event']))
    if selected.track_id != chosen['track_id'] or xy(vars(selected)) != xy(chosen):
        raise AssertionError('Current working-tree BASE selection differs from frozen input replay')
    for t in eligible:
        inputs = t['readiness_inputs']
        probe = SimpleNamespace(track_confirm_frames=inputs['required_hits'],
                                track_confirm_span_s=inputs['required_span_s'])
        ready = HitScanner._track_is_ready(probe, SimpleNamespace(**t), event['context']['cutoff'],
                                          SimpleNamespace(**snapshot['event']))
        if ready != records[by_xy[xy(t)]]['ready']:
            raise AssertionError('Current readiness differs from cached evidence')
    if not records[event['current_id']]['ready']:
        raise AssertionError('Recorded CURRENT winner is not emission-ready')
    return chosen, eligible


def make_ledger(event, detail, records, matrix, names):
    snapshot = detail['snapshot']
    chosen, eligible = verify_event(event, snapshot, records)
    stages = {k:pool_measurement(v, event['truth']) for k,v in detail['stage_positions'].items()}
    stages['raw_recorded'] = pool_measurement(detail['stage_positions'].get('raw_candidates', []),
        event['truth'], available='raw_candidates' in detail['stage_positions'])
    stages['causal_consumed_observations'] = pool_measurement(causal_observations(detail, event['context']), event['truth'])
    tracks = snapshot['tracks']
    # Local proof is the recorded local_confirmed flag, never emitted state=confirmed.
    confirmed = [t for t in tracks if t.get('local_confirmed')]
    stages.update(tracking=pool_measurement(tracks, event['truth']),
                  local_confirmation=pool_measurement(confirmed, event['truth']),
                  eligible=pool_measurement(eligible, event['truth']))
    physical = event['truth']['state'] == 'SINGLE_IMPACT'
    gt = event['truth']['impacts'][0] if physical else None
    best_ranks = {str(r):min((t['final_rank'] for t in eligible if math.dist(xy(t), xy(gt)) <= r), default=None)
                  if gt else None for r in RADII}
    observations = []
    for batch in snapshot.get('associations', []):
        for item in batch['records']:
            observations.append(dict(track_id=item['track_id'], camera_xy=xy(item['candidate']),
                frame_ts=batch['frame_ts'], observed_at=batch.get('observed_at'),
                producer_shot_id=item.get('producer_shot_id'), action=item.get('action'),
                reason=item.get('reason'), local_confirmed=item['candidate'].get('v2225_local_confirm')))
    candidates = []
    for index in event['pools']['current']:
        c = records[index]
        owners = [t for t in eligible if xy(t) == xy(c)]
        candidates.append(dict(index=index, camera_xy=xy(c), ready=c['ready'],
            tracks=[{k:t.get(k) for k in ('track_id','final_rank','rank_key','local_confirmed',
                'producer_shot_id','causal_ownership','first_seen_ts','last_seen_ts','readiness_inputs')}
                    for t in owners],
            evidence=dict(zip(names, matrix[index].tolist())),
            known_hole_context=c['known_hole_context'], projected_edge_context=c['projected_edge_context'],
            recorded_novelty=c.get('recorded_novelty', [])))
    return dict(event=event['event'], session=event['session'], truth=event['truth'],
        status='COMPLETE', current_exact_replay='MATCH_WORKING_TREE_BASE_AND_READINESS',
        original_trace_sha256=event['trace_sha256'], original_label_sha256=event['label_sha256'],
        reconstruction=snapshot.get('reconstruction'), stages=stages,
        observed_proposals={k:[dict(camera_xy=xy(c), timestamp=c.get('timestamp')) for c in v]
                            for k,v in detail['stage_positions'].items()},
        tracking=observations, local_confirmation_semantics='Recorded local proof, not track state; neighborhood confirmation is not exact-XY proof',
        eligible=candidates, current_selected={k:chosen[k] for k in ('track_id','camera_x','camera_y','final_rank','rank_key')},
        current_emission=dict(camera_xy=xy(chosen),
            provenance='Historical measured selected/emitted equality preserved in canonical reports; source emission payload not in this cache',
            limitation='Replayed selector/readiness only; known-hole emission remapping not re-executed'),
        nearest_correct_eligible_by_radius={str(r):stages['eligible']['nearest']
            if stages['eligible']['oracle'][str(r)] else None for r in RADII},
        first_correct_current_rank=best_ranks, evidence_context=event['context'],
        selector_evidence_availability=dict(recorded_track_inputs='AVAILABLE_AT_DECISION',
            physical_features='CAMERA_TIMESTAMP_CAUSAL; extraction/delivery before decision UNPROVEN',
            observed_delivery_after_cutoff=sum(o['observed_at'] is not None and
                o['observed_at'] > event['context']['cutoff'] for o in observations),
            surface='No trusted capture-time hole/repair/tape/projected-edge/seam state; unavailable, no veto'))


def summarize_group(rows):
    result = summarize(rows)
    # Existing score_event treats any non-emission as TN. Keep that detection
    # statistic, but never report UNCERTAIN as a correct NO_PHYSICAL_HIT decision.
    negatives = [r for r in rows if r['truth']['state'] == 'NO_PHYSICAL_SHOT']
    result.update(no_hit_correct_decisions=sum(r['decision']['state']=='NO_PHYSICAL_HIT' for r in negatives),
        no_hit_uncertain=sum(r['decision']['state']=='UNCERTAIN' for r in negatives),
        physical_uncertain=sum(r['truth']['state']=='SINGLE_IMPACT' and r['decision']['state']=='UNCERTAIN' for r in rows),
        errors_gt1000=sum((r['result'].get('error_px') or 0)>1000 for r in rows),
        comparison={str(t):dict(Counter(r['comparison'][str(t)] for r in rows if r['comparison'][str(t)])) for t in RADII},
        correct_candidate_rank_distribution={str(t):dict(Counter(str(r['ranks'][str(t)]) for r in rows
            if r['truth']['state']=='SINGLE_IMPACT')) for t in RADII})
    result['true_negatives'] = dict(nonemissions=result['true_negatives'],
        note='Detection-only count; UNCERTAIN is not a correct NO_PHYSICAL_HIT classification')
    result.pop('combined_event_accuracy')
    return result


def run(base, prior, common_baseline, protocol_path, output):
    if output.exists():
        raise FileExistsError('Use a fresh evaluation directory')
    output.mkdir(parents=True)
    accesses = install_read_guard(output)
    data, matrix, details, models, hashes = load_inputs(base, prior)
    records, names = data['records'], data['feature_names']
    # This older common-logistic control has the strongest already measured
    # selected result on these sessions. Reuse its parameters, never refit.
    common_manifest = json.loads((common_baseline/'review_manifest.json').read_text())
    common_models = {}
    for meta in common_manifest['models']:
        session = meta['evaluated_session']
        if session not in SESSIONS or session in meta['training_sessions']:
            raise ValueError('Common baseline session exclusion mismatch')
        if set(meta['training_sessions']) - {'H10','H20','POST_FIX','S01'}:
            raise ValueError('Unexpected historic development training session')
        p = common_baseline/meta['name']/'parameters.npz'
        if digest(p) != meta['parameters_sha256']:
            raise ValueError('Common baseline weights changed')
        with np.load(p, allow_pickle=False) as m:
            common_models[session] = dict(meta=meta, **{k:m[k].copy() for k in ('mu','sd','beta')})
        hashes[str(p)] = digest(p)
    if set(common_models) != set(SESSIONS):
        raise ValueError('Missing common baseline session')
    # Cached first 72 columns are the unchanged accuracy_verifier features.
    if any(len(m['mu']) != 72 for m in common_models.values()) or names[72] != 'persistent_clean_r2_mean':
        raise ValueError('Common physical feature schema mismatch')
    if digest(Path('src/engine/offline/accuracy_verifier.py')) != common_manifest['source_hashes']['src/engine/offline/accuracy_verifier.py']:
        raise ValueError('Frozen physical extractor changed')
    for p in (common_baseline/'review_manifest.json', common_baseline/'results.json'):
        hashes[str(p)] = digest(p)
    expected_common = {r['event']:r for r in json.loads((common_baseline/'results.json').read_text())['rows']
                       if r['method']=='COMMON_FIXED'}
    protocol = json.loads(protocol_path.read_text())
    hashes[str(protocol_path)] = digest(protocol_path)
    # Include manifests that authenticate the accepted cache, not just current hashes.
    for p in (base/'models/protocol.json', prior/'final/cached_detail_hashes.json'):
        hashes[str(p)] = digest(p)
    write_json(output/'configuration.json', dict(prespecified=protocol, input_hashes=hashes,
        methods=METHODS, command=sys.argv, model_folds=[m['meta'] for m in models.values()],
        common_model_folds=[m['meta'] for m in common_models.values()],
        baseline_inventory_correction='Older COMMON_EARLY is 12/35 @42, stronger than PAIR_CLEAN 10/35; both are existing controls. No new fits or variants.',
        source_hashes={str(p):digest(p) for p in (Path(__file__),
            Path('src/engine/offline/common_verifier.py'), Path('src/engine/offline/accuracy_verifier.py'),
            Path('src/engine/camera/hit_scanner.py'), Path('src/engine/offline/track_replay.py'))}))
    rows, ledger, evidence, latency = [], [], {}, []
    for event in data['events']:
        context = event['context']
        if (max(context['reference_times']+context['pre_times']) >= context['peak'] or
                min(context['post_times']) < context['peak'] or max(context['post_times']) > context['cutoff']):
            raise ValueError('Noncausal cached frame timestamps')
        entry = make_ledger(event, details[event['event']], records, matrix, names)
        ledger.append(entry)
        ids = event['pools']['current']
        order = ids  # Exact CURRENT rank order, checked against every snapshot above.
        current = row(event, records, order, ids, 'CURRENT', 0., 0.)
        if current['selected']['index'] != event['current_id']:
            raise AssertionError('CURRENT pool order mismatch')
        current['decision'] = dict(state='PHYSICAL_HIT', selected=0, reason='RECORDED_EMISSION')
        current['comparison'] = comparison(current, current)
        rows.append(current)
        ready = [i for i in ids if records[i]['ready']]
        # Frozen historical tie convention is record index; no CURRENT score fallback.
        ready.sort()
        coordinates = [xy(records[i]) for i in ready]
        scores_by_method = {}
        for method in METHODS[1:]:
            scored_ids = list(ids) if method=='COMMON_EARLY' else ready
            start = time.perf_counter()
            if method == 'COMMON_EARLY':
                model = common_models[event['session']]
                z = np.clip((matrix[scored_ids,:72]-model['mu'])/model['sd'], -8, 8)
                contrib = z*model['beta'][1:]
                scores = contrib.sum(axis=1) + model['beta'][0]
                groups, totals = family_totals(contrib, names[:72])
                why = dict(families=groups, contributions=totals.tolist(), intercept=float(model['beta'][0]))
                ordered = sorted(range(len(scored_ids)), key=lambda j:(-scores[j],scored_ids[j]))
                winner = ordered[0] if ordered and records[scored_ids[ordered[0]]]['ready'] else None
                decision = dict(state='PHYSICAL_HIT' if winner is not None else 'UNCERTAIN', selected=winner,
                                reason='FROZEN_COMMON_ORDER_THEN_READINESS')
            elif method == 'FROZEN_PAIR_CLEAN':
                model = models[event['session']]
                contrib = frozen_contributions(matrix[ready], model)
                scores = contrib.sum(axis=1)
                groups, totals = family_totals(contrib, [names[j] for j in model['meta']['feature_indices']])
                why = dict(families=groups, contributions=totals.tolist())
                ordered = sorted(range(len(ready)), key=lambda j:(-scores[j],ready[j]))
                decision = dict(state='PHYSICAL_HIT' if ordered else 'UNCERTAIN',
                    selected=ordered[0] if ordered else None, reason='FROZEN_CONDITIONAL_RANKER')
            else:
                scores, reasons = coherent_change_evidence(matrix[ready], names, pre_context=method!='COHERENT')
                ordered = sorted(range(len(ready)), key=lambda j:(-scores[j],ready[j]))
                decision = coherent_event_decision(scores, coordinates, abstain=method.endswith('UNCERTAIN'))
                why = {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in reasons.items()}
            elapsed = 1000*(time.perf_counter()-start)
            order = [scored_ids[j] for j in ordered]
            item = row(event, records, order, scored_ids, method, 0., elapsed)
            item['decision'] = decision
            if decision['selected'] is None:
                item['selected'] = None
                item['result'] = score_event(event['truth'], None)
            item['comparison'] = comparison(current, item)
            if method == 'COMMON_EARLY':
                expected = expected_common[event['event']]
                if item['result'] != expected['result'] or (
                    None if item['selected'] is None else xy(item['selected'])) != (
                    None if expected['selected'] is None else xy(expected['selected'])):
                    raise AssertionError('Existing common baseline no longer reproduces: '+event['event'])
            rows.append(item)
            scores_by_method[method] = dict(indices=scored_ids, scores=scores.tolist(), reasons=why,
                                             ordered=order, decision=decision)
            latency.append(dict(event=event['event'], method=method, score_and_order_ms=elapsed))
        evidence[event['event']] = scores_by_method
        print(event['event'], 'exact MATCH', 'pool',len(ids), 'ready',len(ready), flush=True)
    for missing in data['unavailable']:
        ledger.append(dict(**missing, session=missing['event'].split(':')[0], status='UNRESOLVED_RECORDING',
            current_selected=None, current_emission=None,
            note='Physical label exists; incomplete terminal recording is not a measured non-emission'))
    summaries = []
    for session in (*SESSIONS, 'POOLED'):
        for method in METHODS:
            group = [r for r in rows if r['method']==method and (session=='POOLED' or r['session']==session)]
            summary = dict(session=session, method=method, **summarize_group(group))
            summary['unresolved_recordings'] = sum(session=='POOLED' or u['event'].startswith(session+':') for u in data['unavailable'])
            if method == 'CURRENT':
                events = [e for e in ledger if e['status']=='COMPLETE' and e['truth']['state']=='SINGLE_IMPACT'
                          and (session=='POOLED' or e['session']==session)]
                summary.update(causal_proposal_oracle={str(t):sum(bool(e['stages']['causal_consumed_observations']['oracle'][str(t)]) for e in events) for t in RADII},
                    ranking_losses={str(t):sum(bool(e['stages']['eligible']['oracle'][str(t)]) for e in events)-summary['hits'][str(t)] for t in RADII},
                    no_usable_eligible={str(t):sum(not e['stages']['eligible']['oracle'][str(t)] for e in events) for t in RADII})
            summaries.append(summary)
    with (output/'event_ledger.jsonl').open('x') as f:
        for item in ledger:
            f.write(json.dumps(item)+'\n')
    write_json(output/'results.json', dict(rows=rows, summaries=summaries))
    write_json(output/'candidate_reasons.json', evidence)
    write_json(output/'latency.json', dict(rows=latency, scope='Cached features -> evidence + ordering only; no image I/O/extraction',
        historical_extraction=[dict(event=e['event'], context_ms=e['context_ms'], map_ms=e['map_ms'],
                                    feature_ms=e['feature_ms']) for e in data['events']]))
    with (output/'per_event.csv').open('x', newline='') as f:
        fields=['event','method','state','error_px','selected_x','selected_y']+[f'comparison_{r}' for r in RADII]+[f'correct_rank_{r}' for r in RADII]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            writer.writerow(dict(event=r['event'],method=r['method'],state=r['decision']['state'],
                error_px=r['result'].get('error_px'), selected_x=(r['selected'] or {}).get('camera_x'),
                selected_y=(r['selected'] or {}).get('camera_y'),
                **{f'comparison_{t}':r['comparison'][str(t)] for t in RADII},
                **{f'correct_rank_{t}':r['ranks'][str(t)] for t in RADII}))
    with (output/'per_session.csv').open('x',newline='') as f:
        fields=['session','method','physical','no_physical']+[f'hits_{r}' for r in RADII]+['mean_error_px','median_error_px','p95_error_px','errors_gt100','false_emissions','false_rejections','no_hit_uncertain','unresolved_recordings']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for s in summaries:
            writer.writerow({k:s[k] if k in s else s['hits'][k[5:]] for k in fields})
    write_json(output/'input_accesses.json', sorted(accesses))
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise AssertionError('Input changed during evaluation: '+path)
    write_json(output/'verification.json',dict(exact_matches=len(data['events']),
        input_hashes_unchanged=True, source_trace_access=False, live_changes=False,
        counts=dict(complete_physical=sum(e['truth']['state']=='SINGLE_IMPACT' for e in data['events']),
                    known_no_physical=sum(e['truth']['state']=='NO_PHYSICAL_SHOT' for e in data['events']),
                    unresolved_physical=sum(e['truth']['state']=='SINGLE_IMPACT' for e in data['unavailable']))))
    for s in summaries:
        if s['session']=='POOLED':
            print(s['method'],s['hits'],'>100',s['errors_gt100'],'false accepts',s['false_emissions'],
                  'false nonemissions',s['false_rejections'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('base','prior','common-baseline','protocol','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    run(args.base,args.prior,args.common_baseline,args.protocol,args.output)
