"""Isolated centered-PRE, local-confirmation and motion-classifier diagnostics."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from automation.accuracy_physical_dataset import json_read, load_context
from automation.accuracy_verifier_research import PRIMARY, training_data, fit_model, predict, event_predictions, summarize, save_model
from automation.clean_change_analysis import motion_features, roi_target, cells
from src.engine.offline.clean_physical_change import centered_pre_variability, clean_features, map_metrics
from src.engine.offline.track_replay import reconstruct_legacy, current_exact_replay

SESSIONS = ('D01', 'S01', 'S02', 'POST_FIX')


def centroid_predictions(train_x, train_y, test_x):
    mean, scale = train_x.mean(axis=0), np.maximum(train_x.std(axis=0), 1e-6)
    x, z = (train_x-mean)/scale, (test_x-mean)/scale
    classes = np.unique(train_y)
    centers = np.array([x[train_y == c].mean(axis=0) for c in classes])
    return classes[np.argmin(np.sum((z[:, None]-centers[None])**2, axis=2), axis=1)]


def motion_diagnostic(measurements):
    rows = [r for r in measurements if r['truth']['state'] == 'SINGLE_IMPACT']
    x = np.array([motion_features(r) for r in rows])
    y = cells(np.array([roi_target(r) for r in rows]))
    results = []
    for session in SESSIONS:
        train = np.array([r['session'] in PRIMARY and r['session'] != session for r in rows])
        test = np.array([r['session'] == session for r in rows])
        predicted = centroid_predictions(x[train], y[train], x[test])
        hits = int(np.sum(predicted == y[test]))
        majority = np.bincount(y[train], minlength=9).argmax()
        rng = np.random.default_rng(41017)
        null = [int(np.sum(centroid_predictions(x[train], rng.permutation(y[train]), x[test]) == y[test])) for _ in range(200)]
        results.append(dict(session=session, n=int(test.sum()), hits=hits,
            majority_hits=int(np.sum(y[test] == majority)), uniform_expected=float(test.sum()/9),
            permutation_mean=float(np.mean(null)), permutation_tail_p=(1+sum(v >= hits for v in null))/201,
            predicted=predicted.tolist(), truth=y[test].tolist(),
            interpretation='Normalized crop proxy, not physical Board Space; no calibrated seam'))
    return results


def run(source, output):
    output.mkdir(parents=True, exist_ok=False); cv2.setNumThreads(1)
    dataset = json_read(source/'pre_variability/dataset.json')
    if any(s not in (*PRIMARY, 'S02', 'D01') for s in dataset['sessions']):
        raise ValueError('Unexpected session')
    hashes = {str(source/name): hashlib.sha256((source/name).read_bytes()).hexdigest()
              for name in ('pre_variability/dataset.json', 'pre_variability/features.npy', 'measurements.json')}
    (output/'source_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
    (output/'protocol.json').write_text(json.dumps(dict(
        status='RESEARCH_ONLY', source=str(source), training_sessions=PRIMARY, evaluation_sessions=SESSIONS,
        experiments=['Centered PRE variability, fixed current pools and same logistic method',
                     'Recorded confirmation compactness/darkening rankings, fixed current pool',
                     'Nearest centroid 3x3 crop classifier, standardized motion only, 200 label permutations'],
        session_exclusion='Every evaluated primary session excluded from fitting; D01/S02 never fitted'), indent=2)+'\n')
    old = np.load(source/'pre_variability/features.npy', allow_pickle=False)
    ncommon = dataset['common_feature_count']
    records, events, original, centered, measurements = [], [], [], [], []
    for event in dataset['events']:
        t0 = time.perf_counter(); event = copy.deepcopy(event)
        path = Path(event['trace_path']); trace = json_read(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != event['trace_sha256']:
            raise ValueError('Changed source trace')
        gp = path.parent/'ground_truth.json'
        if event.get('native_label_sha256') and hashlib.sha256(gp.read_bytes()).hexdigest() != event['native_label_sha256']:
            raise ValueError('Changed ground truth')
        context, metadata = load_context(path, trace, 'snapshot')
        context_ms = (time.perf_counter()-t0)*1000
        t0 = time.perf_counter(); magnitude, aux = centered_pre_variability(context)
        map_ms = (time.perf_counter()-t0)*1000
        raw = np.mean(np.abs(context.pre[None]-context.post), axis=0)*(context.roi > 0)
        gt = event['truth']['impacts'][0] if event['truth']['state'] == 'SINGLE_IMPACT' else None
        old_indices = event['indices']; event['indices'] = []; t0 = time.perf_counter()
        for i in old_indices:
            c = dataset['records'][i]
            if c['pool'] == 'expanded':
                continue
            names, extra = clean_features(context, magnitude, aux, (c['camera_x'], c['camera_y']))
            if dataset['feature_names'][ncommon:] != names:
                raise ValueError('Feature schema mismatch')
            record = dict(c, index=len(records)); event['indices'].append(record['index'])
            records.append(record); original.append(old[i]); centered.append(np.concatenate([old[i, :ncommon], extra]))
        events.append(event)
        measurements.append(dict(event=event['event'], session=event['session'], truth=event['truth'], context=metadata,
            context_ms=context_ms, map_ms=map_ms, feature_ms=(time.perf_counter()-t0)*1000,
            metrics=map_metrics(context, magnitude, raw, gt)))
        print(event['event'], 'centered map', round(map_ms, 1), flush=True)
    data = dict(dataset, records=records, events=events)
    results = []
    for method, matrix in [('PRE_RMS_BASELINE', np.array(original)), ('PRE_CENTERED', np.array(centered))]:
        for session in SESSIONS:
            train = tuple(s for s in PRIMARY if s != session)
            x, y, weights = training_data(data, matrix, train)
            model = fit_model(x, y, weights, 'logistic')
            save_model(model, output/(method+'_'+session))
            rows = event_predictions(data, predict(model, matrix), [session], pool='current')
            results.append(dict(method=method, session=session, summary=summarize(rows), rows=rows))
    # A different, fixed hypothesis: rank recorded local physical confirmation
    # metrics directly. No source score enters these keys and no threshold is fit.
    confirmation_rows = []
    for event in events:
        if event['session'] not in SESSIONS:
            continue
        trace = json_read(event['trace_path']); decision = trace['decision_input']
        snap = decision.get('complete_track_audit') or reconstruct_legacy(trace)
        current_exact_replay(snap, decision['deterministic_selection'])
        tracks = {t['track_id']: t for t in snap['tracks']}
        cs = [dict(records[i], index=j) for j, i in enumerate(event['indices']) if records[i]['pool'] == 'current']
        for j, c in enumerate(cs):
            c['index'] = j
        temp = dict(data, records=cs, events=[dict(event, indices=list(range(len(cs))))])
        for method in ('CONFIRM_COMPACT', 'CONFIRM_DARKENING'):
            key = 'v2225_confirm_compact' if method == 'CONFIRM_COMPACT' else 'v2225_confirm_darkening'
            scores = np.array([tracks[c['track_id']]['last_candidate'].get(key, -1e9) for c in cs])
            row = event_predictions(temp, scores, [event['session']], threshold=-math.inf)[0]
            row['method'] = method; confirmation_rows.append(row)
    confirmation = [dict(method=m, session=s, summary=summarize([r for r in confirmation_rows if r['method'] == m and r['session'] == s]))
                    for s in SESSIONS for m in ('CONFIRM_COMPACT', 'CONFIRM_DARKENING')]
    motion = motion_diagnostic(json_read(source/'measurements.json'))
    report = dict(status='RESEARCH_ONLY', clean=results, measurements=measurements,
        confirmation=confirmation, confirmation_rows=confirmation_rows, motion=motion,
        limitations=['Fixed pools and readiness; single-cutoff offline rankings, no live integration',
                     'No D01/S02 fitting; primary whole-session exclusions',
                     'Motion classes normalize captured crop, never called physical Board Space',
                     'No new no-impact labels; no timing rejection; S03 untouched'])
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for row in results+confirmation:
        print(row['session'], row['method'], row['summary']['hits'], 'false', row['summary']['false_emissions'])
    print('motion', json.dumps(motion))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.source, a.output)
