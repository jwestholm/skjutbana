"""Four-session offline coordinate, retention and physical-ranking evidence cache.

No live imports of this module. Absolute XY, GT, detector scores and source labels
are never model features. Recorded metadata remains available for forensic use.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np

from automation.accuracy_physical_dataset import COHORT, REPO, causal_stages, json_read, load_context
from automation.accuracy_proposal_forensics import raw_contours
from automation.evidence_retention_research import causal_ready
from src.engine.offline.accuracy_verifier import align_to_reference, features
from src.engine.offline.clean_physical_change import clean_features, local_residual, motion_compensate, map_metrics
from src.engine.offline.physical_event_truth import event_truth
from src.engine.offline.track_replay import current_exact_replay, reconstruct_legacy

SESSIONS = ('D01', 'S01', 'S02', 'POST_FIX')
RADII = (5, 10, 20, 42)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v))
        stream.write('\n')


def xy(candidate):
    return float(candidate['camera_x']), float(candidate['camera_y'])


def distinct(candidates):
    seen = set(); output = []
    for c in candidates:
        if xy(c) not in seen:
            seen.add(xy(c)); output.append(c)
    return output


def owned_history(snapshot, cutoff):
    """Retain actual observations, with capture-time and producer constraints."""
    groups = defaultdict(list)
    for batch in snapshot.get('associations', []):
        stamp = batch['frame_ts']
        if stamp > cutoff:
            continue
        for row in batch['records']:
            c = row['candidate']
            producer = c.get('v2224_producer_shot_id', row.get('producer_shot_id'))
            if producer is not None and int(producer) != int(snapshot['event']['shot_id']):
                continue
            groups[row['track_id']].append(dict(row, frame_ts=stamp))
    return groups


def observed_medoid(rows, fallback):
    """An actual observed coordinate; each distinct coordinate/frame votes once."""
    if not rows:
        return xy(fallback)
    samples = sorted({(r['frame_ts'], *xy(r['candidate'])) for r in rows})
    points = np.array([p[1:] for p in samples])
    options = np.unique(points, axis=0)
    distance = np.linalg.norm(options[:, None]-points[None], axis=2)
    # Equal total mass per frame avoids counting same-frame duplicate proposals
    # as independent temporal support.
    frame_counts = {t: sum(p[0] == t for p in samples) for t, *_ in samples}
    weights = np.array([1/frame_counts[p[0]] for p in samples])
    return tuple(options[np.argmin(distance@weights)])


def clean_channels(context):
    """Existing useful channels, common reference; deliberately no optical flow."""
    start = time.perf_counter()
    local = local_residual(context.pre, context.post)
    persistent = np.abs(np.median(local, axis=0))
    history = np.stack([align_to_reference(context.pre, im)[0] for im in context.history])
    before = local_residual(context.pre, history)
    rms = .75+cv2.GaussianBlur(np.sqrt(np.mean(before**2, axis=0)), (0, 0), 1)
    centered = before-np.median(before, axis=0)
    variability = .75+cv2.GaussianBlur(np.sqrt(np.mean(centered**2, axis=0)), (0, 0), 1)
    compensate_start = time.perf_counter()
    compensated, motion = motion_compensate(context.pre, context.post, 'piecewise')
    piecewise_ms = (time.perf_counter()-compensate_start)*1000
    piecewise = np.abs(np.median(local_residual(context.pre, compensated), axis=0))
    edge = np.hypot(cv2.Sobel(context.pre, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(context.pre, cv2.CV_32F, 0, 1, ksize=3))/8
    maps = dict(persistent=persistent, pre_rms=persistent**2/(persistent+rms),
                pre_centered=persistent**2/(persistent+variability),
                piecewise=piecewise**2/(piecewise+rms))
    maps = {k: v*(context.roi > 0) for k, v in maps.items()}
    auxiliaries = {k: dict(noise=variability if k == 'pre_centered' else rms, edge=edge) for k in maps}
    return maps, auxiliaries, dict(piecewise=motion, piecewise_ms=piecewise_ms,
                                  total_ms=(time.perf_counter()-start)*1000)


def stage_positions(path, trace, context):
    pools = defaultdict(list); details = []
    for stage in causal_stages(trace):
        debug = stage.get('window_debug', {})
        stamp = debug.get('frame_ts')
        if stamp is None or not trace['peak_ts'] <= stamp <= context.cutoff:
            continue
        pipe = stage.get('pipeline', {}); up = pipe.get('upstream', {})
        if isinstance(pipe.get('retained_candidates'), list):
            pools['retained'].extend(pipe['retained_candidates'])
        if isinstance(stage.get('candidates'), list):
            pools['causal_candidates'].extend(stage['candidates'])
        for key in ('raw_candidates', 'filtered_candidates'):
            if isinstance(pipe.get(key), list):
                pools[key].extend(pipe[key])
        origin = (debug.get('v2221_crop_x0', 0), debug.get('v2221_crop_y0', 0))
        if isinstance(up, dict) and isinstance(up.get('pipeline'), dict):
            origin = up['crop_origin_camera']; ledger = up['pipeline']
            for key, name in [('pre_limit_candidates', 'legacy_precap'), ('retained_candidates', 'legacy_retained')]:
                pools[name].extend(dict(c, camera_x=c['camera_x']+origin[0], camera_y=c['camera_y']+origin[1])
                                   for c in ledger.get(key, []))
            details.append(dict(source_frame_ts=stamp, crop_origin=origin, upstream=ledger,
                                cleanup=pipe.get('cleanup_stages', {})))
        cleanup = pipe.get('cleanup_stages', {})
        for key in ('input', 'after_novelty', 'after_ridge', 'retained'):
            if isinstance(cleanup.get(key), list):
                pools['cleanup_'+key].extend(cleanup[key])
        ref = stage.get('evidence_maps', {}).get('candidate_mask')
        if ref:
            mask = np.load(path.parent/ref['path'], allow_pickle=False)
            if mask.shape != context.pre.shape or tuple(origin) != tuple(context.origin):
                raise ValueError('Mask/canonical camera plane mismatch')
            pools['raw_contours_reconstructed'].extend(raw_contours(mask, origin))
    return {k: distinct(v) for k, v in pools.items()}, details


def track_context(point, tracks, groups, peak):
    """Geometry/time-only relationship to an existing nearby track.

    This family depends on upstream observation availability; it is separately
    ablated rather than presented as independent camera confirmation.
    """
    keys = ('history_available', 'history_unique_frames', 'history_span_ms', 'history_onset_ms',
            'history_medoid_offset', 'history_spread', 'history_local_fraction', 'representative_offset')
    if not tracks:
        return dict(zip(keys, [0.]*len(keys))), None
    nearest = min(tracks, key=lambda t: math.dist(point, xy(t)))
    rows = groups.get(nearest['track_id'], [])
    if not rows or math.dist(point, xy(nearest)) > 12:
        return dict(zip(keys, [0.]*len(keys))), None
    unique = sorted({(r['frame_ts'], *xy(r['candidate'])) for r in rows})
    coords = np.array([r[1:] for r in unique]); times = sorted({r[0] for r in unique})
    medoid = observed_medoid(rows, nearest)
    values = [1., len(times), (times[-1]-times[0])*1000, (times[0]-peak)*1000,
              math.dist(point, medoid), float(np.median(np.linalg.norm(coords-np.array(medoid), axis=1))),
              float(np.mean(np.linalg.norm(coords-np.array(point), axis=1) <= 2)), math.dist(point, xy(nearest))]
    return dict(zip(keys, values)), nearest['track_id']


def scalar_physical(names, vector):
    """Fixed within-track priority, never a calibrated impact probability."""
    v = dict(zip(names, vector))
    return float(.5*(v['r2_abs_late']+v['r4_abs_late']) +
                 .5*(v['r2_center_ring_ratio']+v['r4_center_ring_ratio'])-v['r4_pre_std'])


def prepare(output):
    output.mkdir(parents=True, exist_ok=False); cv2.setNumThreads(1)
    records, vectors, events, unavailable = [], [], [], []
    feature_names = None
    for alias in SESSIONS:
        dirname, negatives = COHORT[alias]
        root = REPO/'content/ai/physical_traces'/dirname
        for path in sorted(root.glob('shots/*/trace.json')):
            start = time.perf_counter(); trace = json_read(path); sid = trace['shot_id']
            gt_path = path.parent/'ground_truth.json'
            gt = json_read(gt_path) if sid not in negatives and gt_path.exists() else None
            truth = event_truth('NO_PHYSICAL_SHOT' if sid in negatives else 'SINGLE_IMPACT' if gt else 'UNKNOWN',
                [gt] if gt else [], reason='Unmodified existing physical labels and human-supported negative mapping',
                caveats=['POSSIBLE_MULTIPLE_VISIBLE_CHANGES_UNCONFIRMED'] if alias == 'S02' and sid == 1 else [])
            event_id = f'{alias}:{sid}'
            if not trace.get('decision_input') or not trace.get('completeness', {}).get('trace_complete'):
                unavailable.append(dict(event=event_id, reason='Incomplete terminal trace/cutoff', truth=truth)); continue
            snapshot = trace['decision_input'].get('complete_track_audit') or reconstruct_legacy(trace)
            winner = current_exact_replay(snapshot, trace['decision_input']['deterministic_selection'])
            context, metadata = load_context(path, trace, 'history_early')
            context_ms = (time.perf_counter()-start)*1000
            t0 = time.perf_counter(); stages, details = stage_positions(path, trace, context)
            groups = owned_history(snapshot, context.cutoff)
            tracks = snapshot['tracks']; eligible = [t for t in tracks if t['eligible']]
            from src.engine.camera.hit_scanner import HitScanner
            current_ready = {}
            for track in eligible:
                inputs = track['readiness_inputs']
                scanner = SimpleNamespace(track_confirm_frames=inputs['required_hits'], track_confirm_span_s=inputs['required_span_s'])
                current_ready[xy(track)] = bool(HitScanner._track_is_ready(scanner, SimpleNamespace(**track),
                    trace['decision_input']['timestamp'], SimpleNamespace(**snapshot['event'])))
            tagged = defaultdict(lambda: dict(tags=set(), candidates=[], track_ids=set()))
            def add(c, tag, track_id=None):
                item = tagged[xy(c)]; item['tags'].add(tag); item['candidates'].append(c)
                if track_id is not None:
                    item['track_ids'].add(track_id)
            for name, values in stages.items():
                for c in values:
                    add(c, name)
            for t in tracks:
                add(t, 'representatives', t['track_id'])
                if t['eligible']:
                    add(t, 'current', t['track_id'])
                for r in groups.get(t['track_id'], []):
                    add(r['candidate'], 'history', t['track_id'])
            if gt:
                add(gt, 'diagnostic_gt_only')
            maps, aux, motion = clean_channels(context)
            map_ms = (time.perf_counter()-t0)*1000; t0 = time.perf_counter()
            positions = {}; rejected = []; indices = []
            for point, item in tagged.items():
                local = (point[0]-context.origin[0], point[1]-context.origin[1])
                if min(local) < 25 or local[0]+25 >= context.pre.shape[1] or local[1]+25 >= context.pre.shape[0]:
                    rejected.append(dict(xy=point, tags=sorted(item['tags']), reason='49x49 patch unavailable at crop border')); continue
                if not context.roi[round(local[1]), round(local[0])]:
                    rejected.append(dict(xy=point, tags=sorted(item['tags']), reason='Outside recorded physical analysis ROI')); continue
                names, base = features(context, point)
                all_names = list(names); value = list(base)
                for name, magnitude in maps.items():
                    extra_names, extra = clean_features(context, magnitude, aux[name], point)
                    all_names.extend(name+'_'+n for n in extra_names); value.extend(extra)
                hist, nearest_track_id = track_context(point, tracks, groups, trace['peak_ts'])
                all_names.extend(hist); value.extend(np.sign(list(hist.values()))*np.log1p(np.abs(list(hist.values()))))
                if feature_names is None:
                    feature_names = all_names
                if all_names != feature_names:
                    raise ValueError('Feature schema drift')
                vec = np.asarray(value, np.float32)
                if not np.isfinite(vec).all():
                    raise ValueError('Nonfinite features')
                confirmed = [c for c in item['candidates'] if c.get('v2225_local_confirm') == 1]
                confirmations = [{k: v for k, v in c.items() if k.startswith('v2225_confirm_')} for c in confirmed]
                index = len(records); positions[point] = index; indices.append(index)
                distance = math.dist(point, xy(gt)) if gt else None
                actual_tags = item['tags']-{'diagnostic_gt_only'}
                # GT-only coordinates are forensic references, never injected
                # into candidate pools or fitted as observed positive proposals.
                selectable = bool(actual_tags)
                record = dict(index=index, event=event_id, session=alias, camera_x=point[0], camera_y=point[1],
                    tags=sorted(item['tags']), track_ids=sorted(item['track_ids']), nearest_track_id=nearest_track_id,
                    gt_distance=distance, selectable=selectable,
                    ready=current_ready[point] if 'current' in item['tags'] else causal_ready(context, point),
                    local_confirmation=confirmations, history=hist,
                    confirmed_compact=max((c.get('v2225_confirm_compact', -math.inf) for c in confirmed), default=None),
                    physical_priority=scalar_physical(names, base),
                    morphology=[{k:c[k] for k in ('area','radius','circularity') if k in c}
                                for c in item['candidates'] if 'radius' in c],
                    known_hole_context='UNAVAILABLE: no trusted capture-time HoleMap',
                    recorded_novelty=[{k:v for k,v in c.items() if 'novelty' in k or 'known_hole' in k or 'ridge' in k}
                                      for c in item['candidates'] if any('novelty' in k or 'known_hole' in k or 'ridge' in k for k in c)],
                    projected_edge_context='UNAVAILABLE: no saved projected-frame/physical calibration')
                records.append(record); vectors.append(vec)
            def ids(values):
                return list(dict.fromkeys(positions[xy(c)] for c in values if xy(c) in positions))
            pools = {name: ids(values) for name, values in stages.items()}
            pools['current'] = ids(sorted(eligible, key=lambda t:t['final_rank']))
            pools['representatives'] = ids(tracks)
            pools['history'] = ids([r['candidate'] for rows in groups.values() for r in rows])
            pools['raw_plausible'] = [i for i in pools.get('raw_contours_reconstructed', [])
                if any(c.get('area', 0) >= 2 and .8 <= c.get('radius', 0) <= 35 for c in records[i]['morphology'])]
            representations = []
            for t in sorted(eligible, key=lambda t:t['final_rank']):
                current_id = positions.get(xy(t))
                if current_id is None:
                    raise ValueError('Lost existing eligible coordinate to feature-boundary filter')
                observations = ids([r['candidate'] for r in groups.get(t['track_id'], [])])
                candidates = list(dict.fromkeys([current_id]+observations))
                medoid = positions.get(observed_medoid(groups.get(t['track_id'], []), t), current_id)
                confirmed_ids = [i for i in candidates if records[i]['confirmed_compact'] is not None]
                best_confirmed = max(confirmed_ids, key=lambda i: records[i]['confirmed_compact']) if confirmed_ids else current_id
                physical_id = max(candidates, key=lambda i: records[i]['physical_priority'])
                representations.append(dict(track_id=t['track_id'], current=current_id, medoid=medoid,
                    confirmed=best_confirmed, physical=physical_id, observations=candidates,
                    final_rank=t['final_rank'], rank_key=t['rank_key'], first_seen_ts=t['first_seen_ts'],
                    best_score=t['best_score'], local_confirmed=t.get('local_confirmed'),
                    observation_id=t.get('observation_id'), rejection_reason=t.get('rejection_reason')))
            feature_ms = (time.perf_counter()-t0)*1000
            raw = np.mean(np.abs(context.pre[None]-context.post), axis=0)*(context.roi > 0)
            event = dict(event=event_id, session=alias, truth=truth, indices=indices, pools=pools,
                representations=representations, current_id=positions[xy(winner)],
                current_track_id=winner['track_id'], trace_path=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                label_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest() if gt_path.exists() else None,
                context=metadata, context_ms=context_ms, map_ms=map_ms, feature_ms=feature_ms,
                motion=motion, clean_metrics={k:map_metrics(context, v, raw, gt) for k,v in maps.items()},
                rejected_patches=rejected, raw_stage_counts={k:len(v) for k,v in stages.items()},
                unavailable=dict(raw_detector_proposals='UNAVAILABLE' if 'raw_candidates' not in stages else None,
                    early_filter_ledger='AVAILABLE' if details else 'UNAVAILABLE'),
                scope='Recorded-input CURRENT exact; alternative image-time causal hypotheses are not full live replay')
            write_json(output/(event_id.replace(':','_')+'_trace_details.json'), dict(snapshot=snapshot, stage_details=details,
                       stage_positions=stages))
            events.append(event)
            print(event_id, 'cache',len(indices),'current',len(pools['current']),'raw plausible',len(pools['raw_plausible']),
                  'seconds',round(time.perf_counter()-start,2),flush=True)
    np.save(output/'features.npy', np.asarray(vectors))
    write_json(output/'dataset.json', dict(schema='physical-counterfactual-1', sessions=SESSIONS,
        status='OFFLINE_ONLY_NO_PROMOTION', reference='history_early', feature_names=feature_names,
        records=records, events=events, unavailable=unavailable,
        contract=dict(training='Only observed coordinates from other whole development sessions; no GT coordinate injection',
            feature_families=['Uniform multiscale causal patches','Persistent/PRE RMS/centered/piecewise maps','Separately ablated observation geometry'],
            excluded=['source score','source ID','absolute XY','GT','current final rank','known-hole veto','projected-edge guesses'],
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); prepare(args.output)
