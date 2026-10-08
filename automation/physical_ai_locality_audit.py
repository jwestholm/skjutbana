"""Bounded source-independent core/ring audit. No model or feature-cache writes.

Labels select/group audit rows outside measure(); they never enter its arithmetic.
First/last windows describe observations, not certified pre-/post-damage frames.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from automation import physical_ai_dataset as ds, physical_ai_fusion as fusion
from src.engine.offline.common_verifier import camera_local_patches, query_physical_witness
from src.engine.offline.clean_physical_change import _LOCAL_MASKS, local_contrast, persistent_local_support

PREVIOUS = Path('/data/skjutbana/evaluation_runs/temporal_collision_20261007_01')
PAIR = ('S02:13', 'S02:2')
Y, X = np.mgrid[-24:25, -24:25]
R2 = X*X + Y*Y
MASKS = dict(core=_LOCAL_MASKS[0], inner=(R2 > 16) & (R2 < 64),
             ring=_LOCAL_MASKS[1], outer=(R2 > 144) & (R2 <= 256),
             far=(R2 > 256) & (R2 <= 576))
CONTRACT = dict(version='core-ring-locality-audit-1', windows='first3 / last3 of frozen <=50-frame prefix',
    masks='core r<=4; inner 4<r<8; ring 8<=r<=12; outer 12<r<=16; far 16<r<=24',
    primary='(mean_abs_pixel_delta_core-mean_abs_pixel_delta_ring)/(sum); None if sum==0',
    pixel_delta='per-pixel median(last3) - per-pixel median(first3)',
    temporal='regional means; fixed endpoint medians, MAD floor0.75; no onset-selected window',
    registration='none, same native-camera patch sampler as frozen witness',
    truth='never an argument to measurement; no fitted threshold',
    caveat='not certified clean PRE, damage onset, or physical rejection')


def describe(series):
    x = np.asarray(series, np.float64)
    before, after = x[:3], x[-3:]
    a, b = float(np.median(before)), float(np.median(after))
    mad = lambda v: float(1.4826*np.median(np.abs(v-np.median(v))))
    noise = max(.75, mad(before), mad(after))
    response = x[3:] - a
    delta = b-a
    return dict(first=list(map(float, before)), last=list(map(float, after)),
        first_median=a, first_min=float(min(before)), first_max=float(max(before)), first_mad=mad(before),
        last_median=b, last_mad=mad(after), signed_change=delta, noise=noise,
        normalized_change=delta/noise,
        tail_coherent_strength=persistent_local_support(a, after, noise)['support'],
        peak_abs_change=float(np.max(np.abs(response))),
        signed_change_fraction=float(np.mean(np.sign(delta)*response > 0)),
        reverse_fraction=float(np.mean(np.sign(delta)*response < 0)),
        sequence=list(map(float, x)))


def measure(patches):
    """Only local pixels. No GT/XY/source/score/session inputs or learned values."""
    p = np.asarray(patches, np.float64)
    if p.ndim != 3 or p.shape[1:] != (49, 49) or not np.isfinite(p).all():
        raise ValueError('Finite native 49x49 patches required')
    if not 6 <= len(p) <= 50:
        return dict(available=False, reason='REQUIRES_6_TO_50_OBSERVATIONS', concentration=None)
    delta = np.median(p[-3:], axis=0)-np.median(p[:3], axis=0)
    regions = {}
    for name, mask in MASKS.items():
        regions[name] = dict(describe(p[:, mask].mean(1)), pixels=int(mask.sum()),
            pixel_signed_change=float(np.mean(delta[mask])),
            pixel_absolute_change=float(np.mean(np.abs(delta[mask]))))
    core, ring = regions['core'], regions['ring']
    a, b = core['pixel_absolute_change'], ring['pixel_absolute_change']
    contrast = describe(local_contrast(p))
    return dict(available=True, reason='OBSERVATION_ONLY', regions=regions, contrast=contrast,
        concentration=None if a+b == 0 else (a-b)/(a+b),
        absolute_change_sum=a+b,
        stable_signed_core_ring_ratio=(core['signed_change']/ring['signed_change']
            if abs(ring['signed_change']) >= ring['noise'] else None),
        polarity_agrees=bool(core['signed_change']*ring['signed_change'] > 0),
        clean_pre_certified=False, physical_state='UNKNOWN')


def extract(data, event, candidates):
    frames = event['frames']; points = [c['xy'] for c in candidates]
    values = []; hashes = {}
    for f in frames:
        path = f['frame_id']; actual = ds.digest(path)
        if actual != data['source_hashes'][path]:
            raise ValueError('Frozen source changed')
        hashes[path] = actual
        im = np.load(path, mmap_mode='r', allow_pickle=False)
        values.append(camera_local_patches(im, points))
    patches = np.asarray(values)
    rows = []
    for j, candidate in enumerate(candidates):
        signal = local_contrast(patches[:, j])
        witness = query_physical_witness(signal, frames, event_id=event['event'],
            candidate_id=candidate['id'], coordinate_version=candidate['id'], camera_xy=candidate['xy'],
            source_provenance={'role': 'FROZEN_COORDINATE_AUDIT'}, cutoff_frame_id=frames[-1]['frame_id'],
            cutoff_ts=event['cutoff_receipt']).witnesses[0]
        rows.append(dict(candidate_id=candidate['id'], camera_xy=candidate['xy'], measurements=measure(patches[:, j]),
            availability=witness['availability'], onset=witness['measurements']))
    return dict(event=event['event'], frames=frames, source_hashes=hashes, rows=rows)


def pair(output):
    output = ds.output_root(output); output.mkdir(parents=True, exist_ok=True)
    data, _ = fusion.load(fusion.DEFAULT)
    repaired = ds.read(PREVIOUS/'repair/seed_1729/result.json')
    identity = dict(dataset_hash=data['dataset_hash'], code_hash=ds.digest(__file__), contract=CONTRACT,
                    selection_hash=ds.digest(PREVIOUS/'repair/seed_1729/result.json'))
    plan = output/'measurement_contract.json'
    if plan.exists() and ds.read(plan) != identity:
        raise ValueError('Frozen measurement definition changed')
    if not plan.exists():
        ds.atomic_json(plan, identity)
    records = []
    for name in PAIR:
        e = next(e for e in data['events'] if e['event'] == name)
        prediction = next(e for e in repaired['metrics']['per_event'] if e['event'] == name)
        candidates = [c for c in e['candidates'] if c['xy'] == prediction['selected_xy']]
        if len(candidates) != 1:
            raise ValueError('Selected frozen coordinate not unique')
        path = output/(name.replace(':', '_')+'.json')
        if path.exists():
            record = ds.read(path)
            if record['identity'] != identity:
                raise ValueError('Replay identity changed')
        else:
            record = dict(extract(data, e, candidates), identity=identity, audit_truth=e['truth'], audit_error=prediction['error'])
            ds.atomic_json(path, record)
        records.append(record)
    transitions = []
    for record in records:
        for row in record['rows']:
            m = row['measurements']
            for i, frame in enumerate(record['frames']):
                transitions.append(dict(event=record['event'],candidate_id=row['candidate_id'], **frame,
                    **{name: region['sequence'][i] for name, region in m['regions'].items()}, contrast=m['contrast']['sequence'][i]))
    with (output/'temporal_measurements.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(transitions[0])); w.writeheader(); w.writerows(transitions)
    ds.atomic_json(output/'pair.json',dict(identity=identity,events=records))
    return records


def bank(output):
    """Replay all frozen coordinates; scoring/GT remains outside measurement."""
    output = ds.output_root(output)
    previous = ds.read(output/'pair/pair.json')
    if previous['identity']['contract'] != CONTRACT:
        raise ValueError('Pair measurement must be frozen before generality check')
    data, _ = fusion.load(fusion.DEFAULT)
    identity = dict(dataset_hash=data['dataset_hash'], code_hash=ds.digest(__file__),
                    pair_hash=ds.digest(output/'pair/pair.json'),
                    plan_hash=ds.digest(output/'generality_plan.json'), contract=CONTRACT)
    directory = output/'bank'; directory.mkdir(exist_ok=True)
    manifest = directory/'identity.json'
    if manifest.exists() and ds.read(manifest) != identity:
        raise ValueError('Bank replay identity changed')
    if not manifest.exists(): ds.atomic_json(manifest, identity)
    rows = []
    for e in data['events']:
        path = directory/(e['event'].replace(':','_')+'.json')
        if path.exists():
            record = ds.read(path)
            if record['identity'] != identity: raise ValueError('Event replay changed')
        else:
            # Exact pair coordinates reuse the same sampler/measurement; all
            # other coordinates are frozen proposals, never GT-derived probes.
            record = dict(extract(data, e, e['candidates']), identity=identity)
            ds.atomic_json(path, record)
        rows.extend(dict(event=e['event'], **r) for r in record['rows'])
        print(e['event'], len(record['rows']), 'rows', flush=True)
    ds.atomic_json(directory/'measurements.json', dict(identity=identity, rows=rows))
    return rows


def main():
    ds.install_guard()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--bank',action='store_true');a=p.parse_args()
    if a.bank:
        bank(a.output);return
    for e in pair(a.output):
        row=e['rows'][0];m=row['measurements']
        print(e['event'],row['camera_xy'],'concentration',m['concentration'],flush=True)
        for k in ('core','ring','inner','outer','far'):
            q=m['regions'][k];print(k,'signed',q['signed_change'],'abs',q['pixel_absolute_change'],'noise',q['noise'],'strength',q['tail_coherent_strength'])
        print('contrast',m['contrast']['signed_change'],m['contrast']['tail_coherent_strength'])


if __name__=='__main__':main()
