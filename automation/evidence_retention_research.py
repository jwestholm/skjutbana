"""Bounded physical-evidence experiments, never imported by the live detector.

Explicit development allowlist; immutable outputs. No GT enters retention,
coordinate choice, feature extraction or model fitting on D01/S02.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import cv2
import numpy as np

from automation.accuracy_physical_dataset import causal_stages, json_read, load_context
from automation.accuracy_contour_rescue import contour_rescue
from automation.accuracy_verifier_research import (PRIMARY, training_data, fit_model, predict,
    event_predictions, summarize, save_model)
from src.engine.offline.accuracy_verifier import features, patch
from src.engine.offline.track_replay import current_exact_replay, reconstruct_legacy

SESSIONS = ('D01', 'S01', 'S02', 'POST_FIX')
RADII = (5, 10, 20, 42)


def bounded_diverse(candidates, origin, shape, budget, head=0):
    """Keep a score head, then round-robin 4x4 spatial/size strata; no GT input.

    Stable score order within strata. Caller controls pool; this is not a new
    authority score and does not reject old holes, edges or repaired areas.
    """
    if type(budget) is not int or type(head) is not int or budget < 1 or not 0 <= head <= budget:
        raise ValueError('Invalid bounded retention budget')
    ordered = sorted(enumerate(candidates), key=lambda pair: -float(pair[1].get('score', 0)))
    selected = ordered[:head]
    cells = {}
    for index, c in ordered[head:]:
        x, y = c['camera_x']-origin[0], c['camera_y']-origin[1]
        key = (min(3, max(0, int(4*y/shape[0]))), min(3, max(0, int(4*x/shape[1]))),
               int(np.searchsorted([2, 4, 8, 16], c.get('radius', 4))))
        cells.setdefault(key, []).append((index, c))
    depth = 0
    while len(selected) < budget:
        layer = [cells[key][depth] for key in sorted(cells) if len(cells[key]) > depth]
        if not layer:
            break
        selected.extend(layer[:budget-len(selected)])
        depth += 1
    # Preserve score order when returning to an existing downstream consumer.
    return [dict(c) for _, c in sorted(selected, key=lambda pair: (-float(pair[1].get('score', 0)), pair[0]))]


def observation_coordinates(snapshot, budget=32, *, cutoff=None):
    """Additional observed coordinates only; eligibility belongs to the parent.

    Readiness is deliberately NOT inherited at a different coordinate. The
    caller must recompute it from causal images. Current XY is always preserved.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError('Positive integer observation budget required')
    tracks = {t['track_id']: t for t in snapshot['tracks'] if t['eligible']}
    seen = {(t['camera_x'], t['camera_y']) for t in tracks.values()}
    groups = {}
    for batch in snapshot['associations']:
        if cutoff is not None and batch['frame_ts'] > cutoff:
            continue
        for row in batch['records']:
            c = row['candidate']; tid = row['track_id']
            if tid not in tracks:
                continue
            producer = c.get('v2224_producer_shot_id')
            if producer is not None and int(producer) != int(snapshot['event']['shot_id']):
                continue
            xy = (c['camera_x'], c['camera_y'])
            if xy in seen:
                continue
            seen.add(xy)
            groups.setdefault(tid, []).append(dict(c, parent_track_id=tid,
                observation_id=row['observation_id'], ready=False))
    # A maximum of three additional distinct observations per parent, 32/event.
    # Farthest-first preserves spatial alternatives instead of source scores.
    for tid, values in groups.items():
        t = tracks[tid]
        values.sort(key=lambda c: (-math.hypot(c['camera_x']-t['camera_x'], c['camera_y']-t['camera_y']),
                                  c['camera_x'], c['camera_y']))
    out = []
    for depth in range(3):
        for tid in sorted(groups):
            if depth < len(groups[tid]):
                out.append(groups[tid][depth])
                if len(out) == budget:
                    return out
    return out


def causal_ready(context, xy):
    """Existing offline three-frame/90ms support rule, not live confirmation."""
    yy, xx = np.mgrid[-24:25, -24:25]; radius = np.hypot(xx, yy)
    pre = patch(context.pre, xy, context.origin)
    delta = np.stack([pre-patch(frame, xy, context.origin) for frame in context.post])
    contrast = delta[:, radius <= 4].mean(axis=1)-delta[:, (radius >= 8) & (radius <= 12)].mean(axis=1)
    support = [t for t, v in zip(context.post_times, contrast) if abs(v) > .5]
    return len(support) >= 3 and max(support)-min(support) >= .09


def hybrid_reconstruction(path, trace, stage, retained=None):
    """Regenerate FAST -> hybrid -> bank/vault for one recorded first frame.

    Configuration is the hashed checkout config, not claimed capture metadata.
    Exact equality checks determine which historical stage is reconstructable.
    """
    from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
    from src.engine.shot_fast_v2225 import fast_extract_candidates_v2225
    debug = stage['window_debug']; upstream = stage['pipeline']['upstream']
    ox, oy = map(int, upstream['crop_origin_camera'])
    width, height = int(debug['v2221_crop_w']), int(debug['v2221_crop_h'])
    stamp = stage['pipeline']['source_frame_ts']
    roi = np.load(path.parent/stage['evidence_maps']['roi_polygon']['path'], allow_pickle=False)
    history = [SimpleNamespace(timestamp=f['timestamp'], gray=np.load(path.parent/f['path'], mmap_mode='r', allow_pickle=False))
               for f in trace['frames'] if f['kind'] == 'pre_history']
    frame = next(f for f in trace['frames'] if f['timestamp'] == stamp and f['kind'] == 'post')
    gray = np.load(path.parent/frame['path'], allow_pickle=False)[oy:oy+height, ox:ox+width]
    scanner = SimpleNamespace(_frame_roi_mask=lambda shape: roi,
        audio_events=[SimpleNamespace(shot_id=trace['shot_id'], peak_ts=trace['peak_ts'], state='pending')],
        frame_history=history, _v2221_active_geometry=SimpleNamespace(crop_x0=ox, crop_y0=oy),
        debug_frames={}, last_window_debug={}, last_trace_pipeline={},
        physical_trace_capture_enabled=True, candidate_limit=trace['context']['scanner_config']['candidate_limit'])
    engine = CandidateGeneratorV2()
    engine._extract_candidates = MethodType(fast_extract_candidates_v2225, engine)
    result = engine.generate(scanner=scanner, gray=gray, frame_ts=stamp,
        legacy_candidates=upstream['pipeline']['retained_candidates'] if retained is None else retained)
    output = [dict(c, camera_x=c['camera_x']+ox, camera_y=c['camera_y']+oy) for c in result.candidates]
    recorded = stage['pipeline']['cleanup_stages']['input']
    keys = ('camera_x', 'camera_y', 'score')
    signature = lambda values: [tuple(c[k] for k in keys) for c in values]
    return dict(output=output, ledger=scanner.last_trace_pipeline.get('hybrid_merge'),
        ordered_xy_score_match=signature(output) == signature(recorded),
        matching_xy_scores=len(set(signature(output)) & set(signature(recorded))),
        recorded_count=len(recorded), config=engine.config.snapshot(), telemetry=result.telemetry)


def run(sources, output):
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    protocol = dict(status='RESEARCH_ONLY', sessions=SESSIONS,
        sources=[str(s) for s in sources],
        hypotheses=['OBSERVATIONS32: add bounded observed XY; frozen verifier and independent readiness',
                    'CONTOURS32: add bounded raw contours; frozen verifier and independent readiness',
                    'LEGACY150_PLUS50: same 200 cap, spatial/size reserve; stage recall only'],
        training='Existing primary sessions only; whole-session exclusions, no D01/S02 fitting',
        caveats=['Recorded-input CURRENT replay; alternatives are single-cutoff offline selection',
                 'No coordinate inherits another coordinate local confirmation',
                 'Legacy cap ablation is not regenerated full asynchronous detector/selection',
                 'No S03 access, new labels, no-impact fabrication, or live promotion'])
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    records, vectors, events = [], [], []
    feature_names = None
    hashes = {}
    for source in sources:
        dataset = json_read(source/'dataset.json')
        if dataset['reference'] != 'history_early' or any(s not in (*PRIMARY, 'S02', 'D01') for s in dataset['sessions']):
            raise ValueError('Expected allowlisted historical early-reference datasets')
        if feature_names is not None and dataset['feature_names'] != feature_names:
            raise ValueError('Mixed feature schemas')
        feature_names = dataset['feature_names']
        for name in ('dataset.json', 'features.npy'):
            hashes[str(source/name)] = hashlib.sha256((source/name).read_bytes()).hexdigest()
        matrix = np.load(source/'features.npy', allow_pickle=False)
        for event in dataset['events']:
            if any(e['event'] == event['event'] for e in events):
                raise ValueError('Duplicate event')
            event = copy.deepcopy(event); old = event['indices']; event['indices'] = []
            for i in old:
                if dataset['records'][i]['pool'] == 'expanded':
                    continue
                record = dict(dataset['records'][i], index=len(records))
                records.append(record); vectors.append(matrix[i]); event['indices'].append(record['index'])
            events.append(event)
    base = dict(dataset, records=records, events=events, sessions=sorted({e['session'] for e in events}))
    (output/'source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    matrix = np.array(vectors)
    models = {}
    for session in SESSIONS:
        train = tuple(s for s in PRIMARY if s != session)
        x, y, weights = training_data(base, matrix, train)
        models[session] = fit_model(x, y, weights, 'logistic')
        save_model(models[session], output/('model_'+session))
    results, forensic, cap_rows = [], [], []
    for event in events:
        alias = event['session']
        if alias not in SESSIONS:
            continue
        start = time.perf_counter(); path = Path(event['trace_path']); trace = json_read(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != event['trace_sha256']:
            raise ValueError('Changed source trace')
        gp = path.parent/'ground_truth.json'
        if event.get('native_label_sha256') and hashlib.sha256(gp.read_bytes()).hexdigest() != event['native_label_sha256']:
            raise ValueError('Changed ground truth')
        snapshot = trace['decision_input'].get('complete_track_audit') or reconstruct_legacy(trace)
        winner = current_exact_replay(snapshot, trace['decision_input']['deterministic_selection'])
        context, _ = load_context(path, trace, 'history_early')
        context_ms = (time.perf_counter()-start)*1000
        current = [records[i] for i in event['indices'] if records[i]['pool'] == 'current']
        current_vectors = [matrix[r['index']] for r in current]
        stages = [s for s in causal_stages(trace) if 'candidate_mask' in s.get('evidence_maps', {})
                  and trace['peak_ts'] <= s.get('window_debug', {}).get('frame_ts', -math.inf) <= context.cutoff]
        gt = event['truth']['impacts'][0] if event['truth']['state'] == 'SINGLE_IMPACT' else None
        dist = lambda c: math.hypot(c['camera_x']-gt['camera_x'], c['camera_y']-gt['camera_y']) if gt else None
        fr = dict(event=event['event'], exact_replay='MATCH', current_error=dist(winner),
                  eligible_count=len(current), context_ms=context_ms, tracks=[])
        for t in snapshot['tracks']:
            if t['eligible'] and (t['selected'] or (gt and dist(t) <= 42)):
                history = [dict(r, batch_frame_ts=b['frame_ts']) for b in snapshot['associations']
                           for r in b['records'] if r['track_id'] == t['track_id']]
                fr['tracks'].append(dict(track=t, distance=dist(t), history=history))
        forensic.append(fr)
        for method in ('CURRENT', 'COMMON_FIXED', 'OBSERVATIONS32', 'CONTOURS32'):
            t0 = time.perf_counter(); extra = []
            if method == 'OBSERVATIONS32':
                extra = observation_coordinates(snapshot, cutoff=context.cutoff)
                for c in extra:
                    c['ready'] = causal_ready(context, (c['camera_x'], c['camera_y']))
            elif method == 'CONTOURS32' and stages:
                mask = np.load(path.parent/stages[-1]['evidence_maps']['candidate_mask']['path'], allow_pickle=False)
                if mask.shape != context.pre.shape:
                    raise ValueError('Contour plane mismatch')
                extra = contour_rescue(mask, context, budget=32)
            proposal_ms = (time.perf_counter()-t0)*1000
            seen = {(r['camera_x'], r['camera_y']) for r in current}
            extra = [c for c in extra if (c['camera_x'], c['camera_y']) not in seen]
            test_records = [dict(c, index=i) for i, c in enumerate(current)]
            test_vectors = list(current_vectors)
            for c in extra:
                _, vector = features(context, (c['camera_x'], c['camera_y']))
                test_vectors.append(vector)
                test_records.append(dict(c, index=len(test_records), event=event['event'], session=alias,
                    pool='expanded', gt_distance=dist(c), track_id=c.get('parent_track_id'), final_rank=None))
            feature_ms = (time.perf_counter()-t0)*1000-proposal_ms
            test_dataset = dict(base, records=test_records, events=[dict(event, indices=list(range(len(test_records))))])
            if method == 'CURRENT':
                scores = np.array([-c['final_rank'] for c in test_records])
            else:
                scores = predict(models[alias], np.array(test_vectors))
            row = event_predictions(test_dataset, scores, [alias], pool='union',
                                    threshold=-math.inf if method == 'CURRENT' else 0.)[0]
            if method == 'CURRENT':
                if row['selected'] is None or any(row['selected'][k] != winner[k] for k in ('camera_x', 'camera_y', 'track_id')):
                    raise AssertionError('CURRENT measurement must match exact selector replay')
            row.update(method=method, proposal_ms=proposal_ms, extra_feature_ms=feature_ms,
                       nearest_candidate_px=min((r['gt_distance'] for r in test_records), default=None) if gt else None,
                       extra_count=len(extra), context_ms=context_ms)
            results.append(row)
        if alias == 'D01':
            for stage in stages:
                up = (stage.get('pipeline') or {}).get('upstream', {}).get('pipeline')
                if not isinstance(up, dict) or not up.get('pre_limit_candidates'):
                    continue
                before = hybrid_reconstruction(path, trace, stage)
                prelimit = up['pre_limit_candidates']; shape = context.pre.shape
                t0 = time.perf_counter()
                bounded = bounded_diverse(prelimit, (0, 0), shape, budget=200, head=150)
                retention_ms = (time.perf_counter()-t0)*1000
                after = hybrid_reconstruction(path, trace, stage, bounded)
                origin = stage['pipeline']['upstream']['crop_origin_camera']
                camera = lambda values: [dict(c, camera_x=c['camera_x']+origin[0], camera_y=c['camera_y']+origin[1]) for c in values]
                pools = dict(legacy_before=camera(up['retained_candidates']), legacy_after=camera(bounded),
                             hybrid_before=before['output'], hybrid_after=after['output'])
                cap_rows.append(dict(event=event['event'], raw_count=len(prelimit), before=before, after=after,
                    retention_ms=retention_ms,
                    pools={name:dict(count=len(pool), nearest=min(map(dist, pool), default=None),
                                    oracle={str(r): any(dist(c) <= r for c in pool) for r in RADII}) for name, pool in pools.items()}))
        print(event['event'], 'done', round(time.perf_counter()-start, 2), flush=True)
    summaries = [dict(session=s, method=m, **summarize([r for r in results if r['session'] == s and r['method'] == m]))
                 for s in SESSIONS for m in ('CURRENT', 'COMMON_FIXED', 'OBSERVATIONS32', 'CONTOURS32')]
    for name, value in [('results', dict(summaries=summaries, rows=results)), ('forensics', forensic), ('cap_reconstruction', cap_rows)]:
        (output/(name+'.json')).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')
    for r in summaries:
        print(r['session'], r['method'], r['hits'], r['oracle'], 'false', r['false_emissions'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.source, args.output)
