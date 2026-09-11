"""Session-separated physical clean-change experiments; all outputs are offline.

Use frozen snapshot datasets for historical sessions plus an explicit D01 dataset.
Train only on the historical primary cohort, holding each primary session out.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from automation.accuracy_physical_dataset import json_read, load_context
from automation.accuracy_verifier_research import (PRIMARY, training_data, fit_model,
    predict, event_predictions, summarize)
from src.engine.offline.accuracy_verifier import features
from src.engine.offline.clean_physical_change import (VARIANTS, CONTRACT, change_maps,
    map_proposals, clean_features, map_metrics, response_map)

EXPANSIONS = ('raw', 'pre_variability', 'piecewise', 'flow')
COMPARATORS = ('POST_FIX', 'S01', 'S02', 'D01')


def combine(directories):
    records, vectors, events = [], [], []
    template, seen = None, set()
    for directory in directories:
        dataset = json_read(directory / 'dataset.json')
        if dataset['reference'] != 'snapshot' or any(s not in (*PRIMARY, 'S02', 'D01') for s in dataset['sessions']):
            raise ValueError('Expected explicitly allowed snapshot datasets')
        matrix = np.load(directory / 'features.npy', allow_pickle=False)
        if template and dataset['feature_names'] != template['feature_names']:
            raise ValueError('Feature schema mismatch')
        template = dataset
        for event in dataset['events']:
            if event['event'] in seen:
                raise ValueError('Duplicate dataset event')
            seen.add(event['event'])
            event = copy.deepcopy(event)
            indices = event['indices']
            event['indices'] = []
            for index in indices:
                record = dict(dataset['records'][index], index=len(records))
                records.append(record)
                vectors.append(matrix[index])
                event['indices'].append(record['index'])
            events.append(event)
    return dict(template, events=events, records=records, sessions=sorted({e['session'] for e in events})), np.array(vectors)


def diagnostic(output, event, context, maps):
    gt = event['truth']['impacts'][0]
    x, y = gt['camera_x']-context.origin[0], gt['camera_y']-context.origin[1]
    vmax = float(np.percentile(maps['raw'][context.roi > 0], 99.5))
    arrays = [context.pre, context.post[-1], maps['raw'], maps['pre_variability']]
    titles = ['PRE', 'Last causal registered POST', 'Raw mean absolute change', 'PRE variability + persistence']
    panels = []
    for row in range(2):
        columns = []
        for col, (arr, title) in enumerate(zip(arrays, titles)):
            display = np.clip(arr * (255 / max(1e-6, 255 if col < 2 else vmax)), 0, 255).astype(np.uint8)
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR) if col < 2 else cv2.applyColorMap(display, cv2.COLORMAP_INFERNO)
            cv2.drawMarker(display, (round(x), round(y)), (255, 255, 0), cv2.MARKER_CROSS, 15, 1)
            if row:
                display = display[max(0, round(y)-50):round(y)+51, max(0, round(x)-50):round(x)+51]
            display = cv2.resize(display, (500, 320), interpolation=cv2.INTER_NEAREST if row else cv2.INTER_AREA)
            panel = np.zeros((360, 500, 3), np.uint8)
            panel[40:] = display
            cv2.putText(panel, title + (' / GT crop' if row else ''), (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            columns.append(panel)
        panels.append(np.hstack(columns))
    header = np.zeros((45, 2000, 3), np.uint8)
    cv2.putText(header, event['event']+' | unchanged human GT (cyan) | raw/clean share scale | diagnostic only',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
    if not cv2.imwrite(str(output / (event['event'].replace(':', '_')+'.png')), np.vstack([header, *panels])):
        raise OSError('Diagnostic image write failed')


def extract(sources, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'diagnostics').mkdir()
    cv2.setNumThreads(1)
    dataset, original = combine(sources)
    variants = {v: dict(records=[], events=[], vectors=[]) for v in VARIANTS}
    measurements = []
    protocol = dict(contract=CONTRACT, variants=list(VARIANTS), expansions=list(EXPANSIONS),
        budget=256, reference='snapshot', training_sessions=list(PRIMARY),
        selection='frozen common logistic OR same fixed logistic method with 15 new clean features',
        training='primary current coordinates + training-only GT; whole-session exclusion; no D01/S02 fit',
        caveats=['D01 development, no independent validation',
                 'camera structure proxy only: saved projector homography/physical seam unavailable',
                 'PRE noise is not labeled no-impact truth; no new negative events fabricated',
                 'expansions compared on POST_FIX/S01/S02/D01 only; map-only intermediate ablations on current pool'])
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for event in dataset['events']:
        t0 = time.perf_counter()
        path = Path(event['trace_path'])
        trace = json_read(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != event['trace_sha256']:
            raise ValueError('Recorded trace hash changed')
        gp = path.parent / 'ground_truth.json'
        if event.get('native_label_sha256') and hashlib.sha256(gp.read_bytes()).hexdigest() != event['native_label_sha256']:
            raise ValueError('Recorded GT hash changed')
        context, metadata = load_context(path, trace, 'snapshot')
        context_ms = (time.perf_counter()-t0)*1000
        maps, aux = change_maps(context)
        gt = event['truth']['impacts'][0] if event['truth']['state'] == 'SINGLE_IMPACT' else None
        original_records = [dataset['records'][i] for i in event['indices'] if dataset['records'][i]['pool'] != 'expanded']
        common_cache = {(r['camera_x'], r['camera_y']): original[r['index']] for r in original_records}
        row = dict(event=event['event'], session=event['session'], truth=event['truth'], context=metadata,
                   context_ms=context_ms, map_timing_ms=aux['timing_ms'], motion=aux['motion'], variants={})
        for variant in VARIANTS:
            state = variants[variant]
            magnitude = maps[variant]
            metrics = map_metrics(context, magnitude, maps['raw'], gt)
            start = time.perf_counter()
            proposed, counts = (map_proposals(context, magnitude, 256)
                if variant in EXPANSIONS and event['session'] in COMPARATORS else ([], None))
            proposal_ms = (time.perf_counter()-start)*1000
            positions = [dict(r) for r in original_records]
            for c in proposed:
                d = float(np.hypot(c['camera_x']-gt['camera_x'], c['camera_y']-gt['camera_y'])) if gt else None
                positions.append(dict(c, event=event['event'], session=event['session'], pool='expanded',
                    gt_distance=d, training_label=1 if gt and d <= 10 else 0 if not gt or d > 42 else None,
                    source_metadata='CLEAN_CHANGE_RESEARCH', track_id=None, final_rank=None))
            event_copy = dict(event, indices=[])
            response = response_map(magnitude)
            start = time.perf_counter()
            for record in positions:
                xy = (record['camera_x'], record['camera_y'])
                if xy not in common_cache:
                    names, vector = features(context, xy)
                    if list(names) != dataset['feature_names']:
                        raise ValueError('Common feature schema mismatch')
                    common_cache[xy] = vector
                clean_names, vector = clean_features(context, magnitude, aux, xy)
                local_x, local_y = round(xy[0]-context.origin[0]), round(xy[1]-context.origin[1])
                record.update(index=len(state['records']), map_score=float(response[local_y, local_x]))
                state['records'].append(record)
                state['vectors'].append(np.concatenate([common_cache[xy], vector]))
                event_copy['indices'].append(record['index'])
            state['events'].append(event_copy)
            row['variants'][variant] = dict(metrics=metrics, proposal_counts=counts,
                proposal_ms=proposal_ms, feature_ms=(time.perf_counter()-start)*1000)
        if gt and (event['session'] == 'D01' or event['event'] in ('S01:1', 'S02:5', 'POST_FIX:3')):
            diagnostic(output / 'diagnostics', event, context, maps)
        row['total_ms'] = (time.perf_counter()-t0)*1000
        measurements.append(row)
        print(event['event'], 'seconds', round(row['total_ms']/1000, 2),
              'retained', round(row['variants']['pre_variability']['metrics']['retained_mass'], 3), flush=True)
    for name, state in variants.items():
        target = output / name
        target.mkdir()
        np.save(target / 'features.npy', np.array(state.pop('vectors')))
        result = dict(dataset, **state, feature_names=dataset['feature_names']+clean_names,
                      clean_variant=name, common_feature_count=len(dataset['feature_names']))
        (target / 'dataset.json').write_text(json.dumps(result)+'\n')
    (output / 'measurements.json').write_text(json.dumps(measurements, indent=2)+'\n')


def evaluate(root):
    """Fixed method; no clean model is selected using held-out event labels."""
    output = root / 'results.json'
    if output.exists():
        raise FileExistsError('Preserve existing research results')
    results = []
    for variant in VARIANTS:
        dataset = json_read(root / variant / 'dataset.json')
        x = np.load(root / variant / 'features.npy', allow_pickle=False)
        ncommon = dataset['common_feature_count']
        for method in ('common', 'augmented', 'map_only'):
            model_cache = {}
            for session in (*PRIMARY, 'S02', 'D01'):
                train = tuple(s for s in PRIMARY if s != session)
                if method == 'map_only':
                    scores = np.array([r['map_score'] for r in dataset['records']])
                else:
                    matrix = x[:, :ncommon] if method == 'common' else x
                    if train not in model_cache:
                        a, b, w = training_data(dataset, matrix, train)
                        model_cache[train] = fit_model(a, b, w, 'logistic')
                    scores = predict(model_cache[train], matrix)
                pools = ('current', 'expanded', 'union') if variant in EXPANSIONS and session in COMPARATORS else ('current',)
                for pool in pools:
                    rows = event_predictions(dataset, scores, [session], pool, 0.)
                    results.append(dict(variant=variant, method=method, session=session, pool=pool,
                                        train=list(train) if method != 'map_only' else [],
                                        metrics=summarize(rows), rows=rows))
                if method == 'augmented' and session == 'D01':
                    target = root / variant / 'research_parameters.npz'
                    np.savez(target, **{k: v for k, v in model_cache[train].items() if k in ('mu', 'sd', 'beta')})
            print('evaluated', variant, method, flush=True)
    output.write_text(json.dumps(results, indent=2)+'\n')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if not args.evaluate_only:
        if not args.datasets:
            parser.error('--datasets required for extraction')
        extract(args.datasets, args.output)
    evaluate(args.output)
