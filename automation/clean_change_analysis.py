"""Isolated edge/noise feature ablations and coarse motion information tests.

Motion coordinates are normalized recorded ROI proxies, NOT calibrated Board
Space: the physical traces do not save the required calibration/seam geometry.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from automation.accuracy_physical_dataset import json_read
from automation.accuracy_verifier_research import PRIMARY, training_data, fit_model, predict, event_predictions, summarize


def motion_features(row):
    _, _, width, height = row['context']['crop']
    global_motion = row['context']['registrations']
    halves = row['motion']['piecewise']
    signals = [[m[k]/scale for m in global_motion] for k, scale in [('dx', width), ('dy', height)]]
    for half in (0, 1):
        signals.extend([[m[half][k]/scale for m in halves] for k, scale in [('dx', width), ('dy', height)]])
    signals += [[m[0][k]/scale-m[1][k]/scale for m in halves] for k, scale in [('dx', width), ('dy', height)]]
    return np.array([v for signal in signals for v in
                     (np.mean(signal), np.std(signal), signal[0], signal[-1])])


def roi_target(row):
    x0, y0, width, height = row['context']['crop']
    gt = row['truth']['impacts'][0]
    point = np.array([(gt['camera_x']-x0)/width, (gt['camera_y']-y0)/height])
    if not np.isfinite(point).all() or np.any(point < 0) or np.any(point >= 1):
        raise ValueError('GT outside normalized captured crop')
    return point


def cells(points):
    indices = np.clip(np.floor(np.asarray(points)*3).astype(int), 0, 2)
    return indices[:, 1]*3+indices[:, 0]


def ridge_predictions(train_x, train_y, test_x):
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-6)
    design = np.column_stack([np.ones(len(train_x)), (train_x-mean)/scale])
    test = np.column_stack([np.ones(len(test_x)), (test_x-mean)/scale])
    reg = np.eye(design.shape[1])
    reg[0, 0] = 0
    beta = np.linalg.solve(design.T@design+reg, design.T@train_y)
    return test@beta


def motion_test(measurements):
    rows = [r for r in measurements if r['truth']['state'] == 'SINGLE_IMPACT']
    x, y = np.array([motion_features(r) for r in rows]), np.array([roi_target(r) for r in rows])
    results = []
    for session in (*PRIMARY, 'S02', 'D01'):
        train = np.array([r['session'] in PRIMARY and r['session'] != session for r in rows])
        test = np.array([r['session'] == session for r in rows])
        predicted = ridge_predictions(x[train], y[train], x[test])
        truth_cells = cells(y[test])
        hits = int(np.sum(cells(predicted) == truth_cells))
        majority = int(np.bincount(cells(y[train]), minlength=9).argmax())
        rng = np.random.default_rng(41017)
        null = [int(np.sum(cells(ridge_predictions(x[train], rng.permutation(y[train]), x[test])) == truth_cells))
                for _ in range(200)]
        results.append(dict(session=session, n=int(test.sum()), grid_hits=hits,
            majority_hits=int(np.sum(truth_cells == majority)), permuted_training_mean=float(np.mean(null)),
            permutation_tail_p=float((1+np.sum(np.array(null) >= hits))/(1+len(null))),
            normalized_roi_mean_error=float(np.mean(np.linalg.norm(predicted-y[test], axis=1))),
            truth_cells=truth_cells.tolist(), predicted_cells=cells(predicted).tolist(),
            label='Exploratory normalized ROI proxy; no physical Board Space or impact authority'))
    return results


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    dataset = json_read(root / 'raw/dataset.json')
    x = np.load(root / 'raw/features.npy', allow_pickle=False)
    names = dataset['feature_names']
    common = dataset['common_feature_count']
    ablations = []
    for suffix in ('edge', 'noise'):
        keep = list(range(common))+[i for i, name in enumerate(names) if name.startswith('clean_') and name.endswith('_'+suffix)]
        matrix = x[:, keep]
        cache = {}
        for session in (*PRIMARY, 'S02', 'D01'):
            train = tuple(s for s in PRIMARY if s != session)
            if train not in cache:
                a, b, w = training_data(dataset, matrix, train)
                cache[train] = fit_model(a, b, w, 'logistic')
            rows = event_predictions(dataset, predict(cache[train], matrix), [session], 'current')
            ablations.append(dict(feature=suffix, session=session, metrics=summarize(rows), rows=rows))
    edge_groups = []
    edge_index = names.index('clean_r4_edge')
    for session in (*PRIMARY, 'S02', 'D01'):
        positives = [r for r in dataset['records'] if r['session'] == session and r['pool'] == 'training_gt_only']
        winners = [r for r in dataset['records'] if r['session'] == session and r['pool'] == 'current'
                   and r['final_rank'] == 1 and (r['gt_distance'] is None or r['gt_distance'] > 42)]
        edge_groups.append(dict(session=session, gt_n=len(positives), false_winner_n=len(winners),
            gt_edge_median=float(np.median([np.expm1(x[r['index'], edge_index]) for r in positives])) if positives else None,
            false_winner_edge_median=float(np.median([np.expm1(x[r['index'], edge_index]) for r in winners])) if winners else None,
            semantics='Camera PRE Sobel magnitude, not calibrated projected-image attribution'))
    motion = motion_test(json_read(root / 'measurements.json'))
    result = dict(edge_noise_ablations=ablations, edge_groups=edge_groups, motion=motion,
        limitations=['D01/S02 never used in fitting; each primary session held out',
                     'Motion proxy is a coarse information test; no selected/emitted coordinate changed',
                     'Permutation and constructed numerical tests do not constitute synthetic validation truth'])
    (output / 'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    for r in motion:
        print('MOTION', r, flush=True)
    for r in ablations:
        print(r['feature'], r['session'], r['metrics']['hits'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
