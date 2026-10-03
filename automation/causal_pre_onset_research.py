"""One fixed camera-local PRE experiment. No fitting, proposals or live callers.

Run into a new directory. Explicit development inputs only; D01 source pixels
are prohibited because their historical paths were overwritten. Existing labels
are joined AFTER every local inference. No object/game fields enter inference.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

from src.engine.offline.accuracy_verifier import patch
from src.engine.offline.clean_physical_change import (
    local_contrast, causal_local_onset, persistent_local_support)

BASE = Path('/data/skjutbana/evaluation_runs')
CACHE = BASE/'D01_counterfactual_20260910_152433/cache/dataset.json'
SELECTOR = BASE/'physical_selection_20260925_001/final/event_ledger.jsonl'
CAN = BASE/'can_retry_20260915_150509'
HISTORIC = BASE/'physical_foundations_20260924_01/research_v2/event_specific_pre.json'
RADII = (5, 10, 20, 42)
PROTOCOL = dict(status='OFFLINE_RESEARCH_ONLY', hypothesis='candidate-local observed step changes PRE',
    signal='r4 inner mean minus r8..12 annulus mean at existing camera XY',
    before=3, after=3, noise_floor_gray=.75, step_noise_multiple=3,
    pre_stability_noise_multiple=2, maximum_frame_gap_s=.25,
    history='all saved pre_history/post through original decision, after previous event decision',
    pre='last ACTUAL observation immediately before unique supported local step; never blend',
    post='same last three receipt-causal frames for old and corrected evidence',
    old='recorded pre_snapshot; accepted selector cache used a different history_early median',
    evidence='minimum same-polarity POST minus PRE local contrast / shared onset noise',
    selector='ONE challenger: greatest corrected support among ready CURRENT coordinates with ONSET_FOUND; stable CURRENT order breaks ties; absent onset => UNCERTAIN',
    no_hit='No calibrated NO_PHYSICAL_HIT declaration', registration='none; raw full-camera same-XY observations',
    ground_truth='evaluation only; no GT-only probes; no thresholds/weights fit',
    source_domain='existing CURRENT eligible coordinates; Can previous unlabelled proposals for reproduction only')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def read_guard(output, allowed_roots):
    accesses = set()
    roots = [p.resolve() for p in allowed_roots]
    def audit(event, args):
        if event not in ('open', 'os.listdir', 'os.scandir') or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        if any(re.search(r'(^|_)S03(_|$)', v, re.I) for v in p.parts):
            raise PermissionError('S03 prohibited')
        p = p.resolve()
        if 'physical_traces' in p.parts and not any(p.is_relative_to(r) for r in roots):
            raise PermissionError('Only explicit development roots permitted: '+str(p))
        if str(p).startswith('/data/skjutbana/') and not p.is_relative_to(output):
            if event == 'open' and len(args) > 2 and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                raise PermissionError('Immutable input: '+str(p))
            accesses.add(str(p))
    sys.addaudithook(audit)
    return accesses


def sample(image, points):
    """Reuse canonical sampler without promoting the entire 4K frame per point."""
    values = []
    for x, y in points:
        ix, iy = int(math.floor(x)), int(math.floor(y))
        xa, ya = ix-25, iy-25
        if xa < 0 or ya < 0 or ix+27 > image.shape[1] or iy+27 > image.shape[0]:
            raise ValueError('Full patch unavailable')
        values.append(patch(image[ya:iy+27, xa:ix+27], (x, y), (xa, ya)))
    return np.stack(values)


def load_observations(path, trace, points, cutoff, lower, hashes):
    """Deduplicate repeated snapshot/history timestamps only after pixel equality.

    Trace seq/order are serialization IDs, NOT camera acquisition sequence IDs.
    No images beyond the decision/reaction cutoff are opened here.
    """
    samples, entries = {}, {}
    snapshot = None
    for f in sorted(trace['frames'], key=lambda f: (f['timestamp'], f['order'])):
        if f['timestamp'] > cutoff or f['timestamp'] <= lower:
            continue
        p = (path.parent/f['path']).resolve()
        if not p.is_relative_to(path.parent.resolve()):
            raise ValueError('Frame path escape')
        hashes[str(p)] = digest(p)
        arr = np.load(p, mmap_mode='r', allow_pickle=False)
        if list(arr.shape) != f['shape'] or str(arr.dtype) != f['dtype']:
            raise ValueError('Frame metadata mismatch')
        patches = sample(arr, points)
        if f['kind'] == 'pre_snapshot':
            snapshot = (f, patches)
        if f['kind'] not in ('pre_history', 'post'):
            continue
        t = f['timestamp']
        if t in samples and not np.array_equal(samples[t], patches):
            raise ValueError('Different candidate pixels with identical receipt timestamp')
        samples[t] = patches
        entries[t] = {k:f.get(k) for k in ('kind','path','timestamp','seq','order','acquisition')}
    times = sorted(samples)
    if snapshot and snapshot[0]['timestamp'] in samples and not np.array_equal(snapshot[1], samples[snapshot[0]['timestamp']]):
        raise ValueError('Snapshot/history pixels disagree at identical timestamp')
    return times, np.stack([samples[t] for t in times]), snapshot, [entries[t] for t in times]


def infer(times, patches, snapshot, cutoff, early_times=()):
    rows = []
    for j in range(patches.shape[1]):
        images = patches[:, j]
        signal = local_contrast(images)
        r = causal_local_onset(signal, times, cutoff=cutoff)
        r.update(signal=signal.tolist(), pre_status='UNAVAILABLE', old=None, causal=None)
        noise = r['noise'] or max(.75, 1.4826*float(np.median(np.abs(signal[:3]-np.median(signal[:3])))))
        old = float(local_contrast(snapshot[1][j:j+1])[0]) if snapshot else None
        early = [i for i, t in enumerate(times) if t in early_times]
        r['history_early'] = None
        if early and len(early) == len(early_times):
            # This is the SAME historical reference's pixels, with this pass's
            # fixed raw-coordinate local support. It does not reuse or pretend
            # to preserve the registered common-verifier feature semantics.
            reference = np.median(images[early], axis=0)
            scalar = float(local_contrast(reference[None])[0])
            r['history_early'] = persistent_local_support(scalar, signal[-3:], noise)
            r['history_early']['absolute_pixel_change'] = float(np.abs(images[-3:]-reference).mean())
        if old is not None and len(times) >= 3:
            r['old'] = persistent_local_support(old, signal[-3:], noise)
            r['old']['absolute_pixel_change'] = float(np.abs(images[-3:]-snapshot[1][j]).mean())
        if r['state'] == 'ONSET_FOUND':
            k = r['pre_index']
            r['causal'] = persistent_local_support(signal[k], signal[-3:], noise)
            r['causal']['absolute_pixel_change'] = float(np.abs(images[-3:]-images[k]).mean())
            if snapshot:
                r['pre_status'] = ('PRE_UNCHANGED_SAFE' if snapshot[0]['timestamp'] <= times[k]
                                   else 'PRE_CONTAMINATED' if snapshot[0]['timestamp'] >= times[k+1]
                                   else 'ONSET_AMBIGUOUS')
        rows.append(r)
    return rows


def summary(rows, method):
    physical = [r for r in rows if r['truth']['state']=='SINGLE_IMPACT']
    errors = sorted(r[method]['error_px'] for r in physical if r[method]['error_px'] is not None)
    neg = [r for r in rows if r['truth']['state']=='NO_PHYSICAL_SHOT']
    return dict(physical_count=len(physical), negatives=len(neg),
        hits={str(k):sum(r[method]['error_px'] is not None and r[method]['error_px'] <= k for r in physical) for k in RADII},
        false_accepts=sum(r[method]['index'] is not None for r in neg),
        physical_uncertain=sum(r[method]['index'] is None for r in physical),
        no_physical_hit_declarations=0,
        mean=float(np.mean(errors)) if errors else None, median=float(np.median(errors)) if errors else None,
        p95=errors[math.ceil(.95*len(errors))-1] if errors else None,
        maximum=max(errors) if errors else None, over100=sum(v>100 for v in errors), over1000=sum(v>1000 for v in errors))


def run(output):
    output = output.resolve(); output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    write(output/'protocol.json', PROTOCOL)
    data = json.loads(CACHE.read_text()); records = data['records']
    if tuple(data['sessions']) != ('D01', 'S01', 'S02', 'POST_FIX'):
        raise ValueError('Development cohort mismatch')
    binding = json.loads((CAN/'binding.json').read_text())
    roots = {Path(e['trace_path']).parent.parent.parent for e in data['events'] if e['session'] != 'D01'}
    roots.add(Path(binding['trace_root']))
    accesses = read_guard(output, roots)
    hashes = {str(p):digest(p) for p in (CACHE, SELECTOR, HISTORIC, CAN/'binding.json')}
    ledger = {r['event']:r for r in map(json.loads, SELECTOR.read_text().splitlines())}
    rows, unavailable, previous = [], [], {}
    candidate_count = 0
    for event in data['events']:
        eid, session = event['event'], event['session']
        if session == 'D01':
            unavailable.append(dict(event=eid,state='INSUFFICIENT_HISTORY',reason='Original pixels overwritten; source D01 path NOT opened',
                current_error=records[event['current_id']]['gt_distance']))
            continue
        start = time.perf_counter(); path = Path(event['trace_path'])
        hashes[str(path)] = digest(path)
        if hashes[str(path)] != event['trace_sha256']:
            raise ValueError('Accepted trace hash mismatch: '+eid)
        trace = json.loads(path.read_text())
        cutoff = min(event['context']['cutoff'], trace.get('next_audio_peak_ts') or math.inf)
        lower = previous.get(session, -math.inf)
        previous[session] = cutoff
        # Stable CURRENT rank order, no source score used by the challenger.
        rank = {tuple(c['camera_xy']):min(t['final_rank'] for t in c['tracks']) for c in ledger[eid]['eligible']}
        ids = sorted(event['pools']['current'], key=lambda i:rank[(records[i]['camera_x'], records[i]['camera_y'])])
        if any(not records[i]['selectable'] or records[i]['tags']==['diagnostic_gt_only'] for i in ids):
            raise ValueError('Unobserved diagnostic point in pool')
        points = [(records[i]['camera_x'], records[i]['camera_y']) for i in ids]
        times, pixels, snapshot, frames = load_observations(path, trace, points, cutoff, lower, hashes)
        extracted_ms = (time.perf_counter()-start)*1000; start = time.perf_counter()
        effects = infer(times, pixels, snapshot, cutoff, event['context']['reference_times'])
        inference_ms = (time.perf_counter()-start)*1000
        for index, r in zip(ids, effects):
            r.update(index=index, camera_xy=[records[index]['camera_x'], records[index]['camera_y']], ready=records[index]['ready'])
        # Evidence written before truth joins / selector measurement.
        payload = dict(event=eid, peak=trace['peak_ts'], cutoff=cutoff, lower_boundary=None if not math.isfinite(lower) else lower,
            frames=frames, old_snapshot=snapshot[0] if snapshot else None, candidates=effects,
            extraction_ms=extracted_ms, inference_ms=inference_ms,
            timing_semantics='receipt chronology only; exposure/delivery unavailable; trace seq is NOT camera sequence')
        write(output/(eid.replace(':','_')+'_evidence.json'), payload)
        eligible = [r for r in effects if r['ready']]
        supported = [r for r in eligible if r['state']=='ONSET_FOUND' and r['causal']['support']>0]
        best = max(supported, key=lambda r:r['causal']['support']) if supported else None
        def decision(index):
            return dict(index=index, error_px=records[index]['gt_distance'] if index is not None else None,
                        state='COORDINATE_PROPOSAL' if index is not None else 'UNCERTAIN')
        measured = dict(event=eid, session=session, truth=event['truth'], current=decision(event['current_id']),
            causal=decision(best['index'] if best else None), candidates=len(effects),
            onset_counts=dict(Counter(r['state'] for r in effects)),
            pre_counts=dict(Counter(r['pre_status'] for r in effects)),
            event_state='ONSET_FOUND' if supported else 'ONSET_AMBIGUOUS' if any(r['state']=='ONSET_AMBIGUOUS' for r in effects) else
                        'INSUFFICIENT_HISTORY' if all(r['state']=='INSUFFICIENT_HISTORY' for r in effects) else 'NO_SUPPORTED_ONSET',
            extraction_ms=extracted_ms,inference_ms=inference_ms,frame_count=len(times),
            frame_span_ms=(times[-1]-times[0])*1000,
            receipt_margin_ms=(cutoff-times[-1])*1000,
            decision_latency_ms=(cutoff-trace['peak_ts'])*1000)
        if event['truth']['state']=='SINGLE_IMPACT':
            good = min(eligible, key=lambda r:records[r['index']]['gt_distance'])
            winner = next(r for r in effects if r['index']==event['current_id'])
            old_order = sorted(eligible, key=lambda r:-(r['old']['support'] if r['old'] else -1))
            new_order = sorted(eligible, key=lambda r:-(r['causal']['support'] if r['causal'] else -1))
            def describe(r):
                return {k:r[k] for k in ('index','camera_xy','state','pre_status','old','history_early','causal','bracket','support_available_ts')}
            pair = dict(nearest=describe(good), winner=describe(winner), nearest_error=records[good['index']]['gt_distance'],
                current_rank=rank[tuple(good['camera_xy'])],
                old_support_rank=old_order.index(good)+1,
                causal_support_rank=new_order.index(good)+1 if good['causal'] else None,
                old_separation=good['old']['support']-winner['old']['support'] if good['old'] and winner['old'] else None,
                causal_separation=good['causal']['support']-winner['causal']['support'] if good['causal'] and winner['causal'] else None)
            measured['pair'] = pair
            measured['changes'] = {str(k):('GAIN' if (measured['causal']['error_px'] is not None and measured['causal']['error_px']<=k) else 'REGRESSION')
                if (measured['current']['error_px']<=k) != (measured['causal']['error_px'] is not None and measured['causal']['error_px']<=k)
                else 'UNCHANGED' for k in RADII}
        rows.append(measured); candidate_count += len(effects)
        print(eid, measured['event_state'], 'CURRENT', measured['current']['error_px'], 'CAUSAL', measured['causal']['error_px'], flush=True)
    unavailable += [dict(event=e['event'], state='INSUFFICIENT_HISTORY',reason=e['reason']) for e in data['unavailable']]
    write(output/'results.json', dict(events=rows,unavailable=unavailable, candidate_count=candidate_count,
        summaries={s:{m:summary([r for r in rows if s=='POOLED' or r['session']==s],m) for m in ('current','causal')}
                   for s in ('S01','S02','POST_FIX','POOLED')},
        baseline_35={str(k):sum(e['truth']['state']=='SINGLE_IMPACT' and records[e['current_id']]['gt_distance']<=k for e in data['events']) for k in RADII},
        full35_challenger='UNAVAILABLE: six original D01 pixel histories absent; do not silently carry CURRENT forward'))
    write(output/'input_hashes.json',hashes)
    write(output/'input_accesses.json', sorted(accesses))
    write(output/'source_hashes.json',{str(p):digest(p) for p in [Path(__file__),Path('src/engine/offline/clean_physical_change.py'),Path('src/engine/offline/accuracy_verifier.py')]})


def audit_inputs(output, development, reproduced):
    """Provenance, historical reproduction and Can local evidence, no selector."""
    output = output.resolve(); output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    data = json.loads(CACHE.read_text()); records = data['records']
    binding = json.loads((CAN/'binding.json').read_text())
    root = Path(binding['trace_root'])
    roots = {Path(e['trace_path']).parent.parent.parent for e in data['events'] if e['session'] != 'D01'} | {root}
    accesses = read_guard(output, roots)
    hashes = {}
    prior = json.loads(HISTORIC.read_text())
    repeat = json.loads((reproduced/'event_specific_pre.json').read_text())
    old_hashes = json.loads((HISTORIC.parent/'input_hashes.json').read_text())
    new_hashes = json.loads((reproduced/'input_hashes.json').read_text())
    if old_hashes != new_hashes:
        raise ValueError('Historical Can input fingerprints changed')
    can_rows = []
    for original, again in zip(prior['events'], repeat['events']):
        if original['candidates'] != again['candidates'] or original['proposed_event_pre'] != again['proposed_event_pre']:
            raise ValueError('Historical Can result does not reproduce')
        sid = original['shot_id']; path = root/'shots'/f'shot_{sid:08d}'/'trace.json'
        trace = json.loads(path.read_text()); hashes[str(path)] = digest(path)
        cutoff = trace['object_query_diagnostic']['candidate_ledger'][-1]['frame_ts']
        reaction = trace['object_gameplay_diagnostic']['reaction_presented_ts']
        if cutoff >= reaction:
            raise ValueError('Historical cutoff includes reaction')
        points = [r['camera_xy'] for r in again['candidates']]
        start = time.perf_counter()
        times, pixels, _, frames = load_observations(path, trace, points, cutoff, -math.inf, hashes)
        old_time = max(trace['object_query_diagnostic']['pre_frame_ts'])
        old_index = times.index(old_time)
        snap = ({'timestamp':old_time}, pixels[old_index])
        results = infer(times, pixels, snap, cutoff)
        strongest = points.index(again['proposed_event_pre']['camera_xy'])
        row = dict(event='CAN:'+str(sid),points=points, frames=frames, peak=trace['peak_ts'],cutoff=cutoff,
            reaction_presented_ts=reaction, old_pre_ts=old_time, effects=results,
            strongest_index=strongest, strongest=results[strongest],
            original_reproduced_exactly=True, historical_scope='frozen OBJECT context; NOT full-camera nomination',
            local_scope='full-camera same-XY patches, no membership or desired object input',
            processing_ms=(time.perf_counter()-start)*1000)
        can_rows.append(row)
    write(output/'can_local_evidence.json', can_rows)
    # Independent check that present raw pixels still reproduce the accepted
    # original 72-feature representation at each CURRENT winner and nearest
    # eligible point. GT only chooses audit witnesses AFTER local inference.
    from automation.accuracy_physical_dataset import load_context
    from src.engine.offline.accuracy_verifier import features
    matrix = np.load(CACHE.parent/'features.npy',allow_pickle=False)
    checked = []
    for event in data['events']:
        if event['session']=='D01':
            continue
        path = Path(event['trace_path'])
        if digest(path) != event['trace_sha256']:
            raise ValueError('Accepted trace changed')
        context, _ = load_context(path, json.loads(path.read_text()), 'history_early')
        ids = {event['current_id']}
        if event['truth']['state']=='SINGLE_IMPACT':
            ids.add(min((i for i in event['pools']['current'] if records[i]['ready']),key=lambda i:records[i]['gt_distance']))
        for index in sorted(ids):
            c = records[index]
            names, values = features(context, (c['camera_x'],c['camera_y']))
            if list(names) != data['feature_names'][:72]:
                raise ValueError('Original feature schema changed')
            error = float(np.max(np.abs(values-matrix[index,:72])))
            checked.append(dict(event=event['event'],index=index,max_abs_feature_error=error))
            if not np.allclose(values,matrix[index,:72],atol=2e-5,rtol=2e-5):
                raise ValueError('Raw pixels do not reproduce accepted cache: '+event['event'])
        print(event['event'],'cached physical-feature witnesses match',flush=True)
    write(output/'cached_pixel_crosscheck.json',checked)
    # Exact selection mechanics can still be checked on D01's frozen inputs.
    from src.engine.offline.track_replay import current_exact_replay
    matches = []
    for e in data['events']:
        path = CACHE.parent/(e['event'].replace(':','_')+'_trace_details.json')
        detail = json.loads(path.read_text())
        chosen = current_exact_replay(detail['snapshot'])
        c = records[e['current_id']]
        if (chosen['camera_x'],chosen['camera_y']) != (c['camera_x'],c['camera_y']):
            raise ValueError('CURRENT coordinate mismatch')
        matches.append(e['event'])
    write(output/'current_replay.json',dict(matches=matches, scope='selection over accepted snapshots; no D01 pixel access'))
    measured_hashes = json.loads((development/'input_hashes.json').read_text())
    for path, h in measured_hashes.items():
        if digest(path) != h:
            raise ValueError('Input changed during experiment: '+path)
    write(output/'integrity.json',dict(measured_inputs_unchanged=len(measured_hashes),historical_can_fingerprints_match=True))
    write(output/'input_hashes.json',hashes)
    write(output/'input_accesses.json',sorted(accesses))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit-development', type=Path)
    p.add_argument('--can-reproduction', type=Path)
    args = p.parse_args()
    if args.audit_development:
        if not args.can_reproduction:
            p.error('--can-reproduction required for audit')
        audit_inputs(args.output,args.audit_development,args.can_reproduction)
    else:
        run(args.output)
