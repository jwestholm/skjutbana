"""Explicit physical development cohort and causal evidence extraction; S03 refused."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import cv2
import numpy as np

from src.engine.offline.accuracy_verifier import EvidenceContext, FEATURE_CONTRACT, align_to_reference, features, proposals
from src.engine.offline.physical_event_truth import event_truth
from src.engine.offline.track_replay import current_exact_replay, reconstruct_legacy

REPO = Path(__file__).resolve().parents[1]
COHORT = {
    'H10': ('session_20260908_144822_d6713dec', ()),
    'H20': ('session_20260908_163626_9b47fac6', (7, 15)),
    'POST_FIX': ('session_20260908_194746_b38de674', ()),
    'S01': ('session_20260909_S01_recovered_20260909', (4,)),
    'S02': ('session_20260909_162957_S02_b900192c', (2, 8, 10)),
}


def guard_path(path):
    # Check lexical/resolved names before opening any trace or binding.
    for part in (*Path(path).parts, *Path(path).resolve().parts):
        if re.search(r'(^|_)S03(_|$)', part, re.IGNORECASE) or 'VALIDATION_UNTOUCHED' in part:
            raise PermissionError('S03 is prohibited in accuracy research')


def json_read(path):
    guard_path(path)
    return json.loads(Path(path).read_text())


def causal_stages(trace):
    cutoff = (trace.get('decision_input') or {}).get('timestamp')
    return [s for s in trace.get('stages', []) if cutoff is not None and s['timestamp'] <= cutoff
            and s.get('candidate_pool_shot_id') == trace['shot_id']]


def load_context(path, trace, reference='snapshot'):
    decision = trace['decision_input']
    cutoff = min(decision['timestamp'], trace.get('next_audio_peak_ts') or float('inf'))
    peak = trace['peak_ts']
    stages = causal_stages(trace)
    debug = next((s['window_debug'] for s in reversed(stages) if 'v2221_crop_x0' in s.get('window_debug', {})), {})
    full_shape = trace['context']['frame_shape']
    x0, y0 = int(debug.get('v2221_crop_x0', 0)), int(debug.get('v2221_crop_y0', 0))
    w, h = int(debug.get('v2221_crop_w', full_shape[1])), int(debug.get('v2221_crop_h', full_shape[0]))
    shape = (h, w)
    def load(ref):
        target = (path.parent / ref['path']).resolve()
        if not target.is_relative_to(path.parent.resolve()):
            raise ValueError('Evidence path escapes captured event')
        arr = np.load(target, mmap_mode='r', allow_pickle=False)
        if list(arr.shape) != ref['shape'] or str(arr.dtype) != ref['dtype']:
            raise ValueError('Frame artifact metadata mismatch')
        return np.array(arr[y0:y0+h, x0:x0+w], dtype=np.float32)
    fs = sorted(trace['frames'], key=lambda f:f['timestamp'])
    snapshot = next(f for f in fs if f['kind'] == 'pre_snapshot')
    if snapshot['timestamp'] >= peak:
        raise ValueError('Snapshot is not PRE')
    history = [f for f in fs if f['kind'] == 'pre_history' and f['timestamp'] < peak]
    if not history:
        history = [snapshot]
    before = history[-6:]
    selected_post = [f for f in fs if f['kind'] == 'post' and peak <= f['timestamp'] <= cutoff
                     and f['timestamp'] < (trace.get('next_audio_peak_ts') or float('inf'))]
    # Last ten frames bound feature cost; the policy is identical for every event.
    selected_post = selected_post[-10:]
    if not selected_post:
        raise ValueError('No causal POST frames')
    reference_frames = [snapshot]
    if reference in ('history_early', 'history_guarded'):
        reference_frames = [f for f in history if -.8 <= f['timestamp']-peak <= -.4]
        if not reference_frames:
            raise ValueError('No history in fixed early reference interval')
        if reference == 'history_guarded':
            # Test whether an onset inside late PRE is incorrectly used as noise.
            before = reference_frames
    pre = np.median(np.stack([load(f) for f in reference_frames]), axis=0)
    post, registrations = [], []
    for ref in selected_post:
        aligned, reg = align_to_reference(pre, load(ref))
        post.append(aligned)
        registrations.append(dict(reg, timestamp=ref['timestamp']))
    roi = np.full(shape, 255, np.uint8)
    map_stage = next((s for s in reversed(stages) if 'roi_polygon' in s.get('evidence_maps', {})), None)
    roi_origin = 'bounding_crop_only_exact_mask_unavailable'
    if map_stage:
        saved = np.load(path.parent / map_stage['evidence_maps']['roi_polygon']['path'], allow_pickle=False)
        if saved.shape == shape:
            roi = saved
            roi_origin = 'recorded_causal_roi_polygon'
    context = EvidenceContext(pre, np.stack([load(f) for f in before]), np.stack(post), peak, cutoff,
                              tuple(f['timestamp'] for f in before), tuple(f['timestamp'] for f in selected_post),
                              (x0, y0), roi, registrations)
    metadata = dict(reference=reference, reference_times=[f['timestamp'] for f in reference_frames],
                    pre_times=context.pre_times, post_times=context.post_times, crop=[x0, y0, w, h],
                    roi_origin=roi_origin, registrations=registrations, peak=peak, cutoff=cutoff,
                    timestamp_semantics='camera capture timestamps; worker delivery parity unavailable')
    return context, metadata


def distance(a, b):
    return float(np.hypot(a['camera_x']-b['camera_x'], a['camera_y']-b['camera_y']))


def build(output, sessions, reference='snapshot', budget=256):
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    records, vectors, event_rows, unavailable, feature_names = [], [], [], [], None
    for alias in sessions:
        dirname, false_ids = COHORT[alias]
        root = REPO / 'content/ai/physical_traces' / dirname
        guard_path(root)
        for path in sorted(root.glob('shots/*/trace.json')):
            start = time.perf_counter()
            trace = json_read(path)
            sid = trace['shot_id']
            key = f'{alias}:{sid}'
            gp = path.parent / 'ground_truth.json'
            gt = json_read(gp) if gp.exists() and sid not in false_ids else None
            caveats = ['POSSIBLE_MULTIPLE_VISIBLE_CHANGES_UNCONFIRMED'] if alias == 'S02' and sid == 1 else []
            truth = event_truth('NO_PHYSICAL_SHOT' if sid in false_ids else 'SINGLE_IMPACT' if gt else 'UNKNOWN',
                                [gt] if gt else [], reason='Preserved native label and explicit human-supported event mapping', caveats=caveats)
            decision = trace.get('decision_input') or {}
            if not decision or not trace.get('completeness', {}).get('trace_complete'):
                unavailable.append(dict(event=key, truth=truth, reason='incomplete terminal trace/cutoff'))
                continue
            snapshot = decision.get('complete_track_audit')
            if not snapshot:
                snapshot = reconstruct_legacy(trace)
            winner = current_exact_replay(snapshot, decision['deterministic_selection'])
            context, metadata = load_context(path, trace, reference)
            context_ms = (time.perf_counter()-start)*1000
            t0 = time.perf_counter()
            expanded = proposals(context, budget)
            proposal_ms = (time.perf_counter()-t0)*1000
            from src.engine.camera.hit_scanner import HitScanner
            positions = []
            for track in snapshot['tracks']:
                if not track['eligible']:
                    continue
                inputs = track['readiness_inputs']
                scanner = SimpleNamespace(track_confirm_frames=inputs['required_hits'], track_confirm_span_s=inputs['required_span_s'])
                ready = HitScanner._track_is_ready(scanner, SimpleNamespace(**track), decision['timestamp'], SimpleNamespace(**snapshot['event']))
                positions.append(('current', dict(track, ready=bool(ready))))
            positions += [('expanded', c) for c in expanded]
            if gt:
                positions.append(('training_gt_only', dict(gt, ready=False)))
            event_indices, rejected_patches, cache = [], [], {}
            t0 = time.perf_counter()
            for pool, candidate in positions:
                xy = (candidate['camera_x'], candidate['camera_y'])
                try:
                    if xy not in cache:
                        cache[xy] = features(context, xy)
                    names, vector = cache[xy]
                except ValueError as exc:
                    rejected_patches.append(dict(pool=pool, xy=xy, reason=str(exc)))
                    continue
                if feature_names is None:
                    feature_names = names
                if names != feature_names:
                    raise ValueError('Feature schema changed')
                index = len(vectors)
                vectors.append(vector)
                event_indices.append(index)
                records.append(dict(index=index, event=key, session=alias, pool=pool, camera_x=xy[0], camera_y=xy[1],
                                    gt_distance=distance(candidate, gt) if gt else None,
                                    training_label=(1 if pool == 'training_gt_only' or (gt and distance(candidate, gt) <= 10) else
                                                    0 if not gt or distance(candidate, gt) > 42 else None),
                                    ready=candidate['ready'], source_metadata=candidate.get('source', 'EXPANSION'),
                                    track_id=candidate.get('track_id'), final_rank=candidate.get('final_rank'),
                                    support_frames=candidate.get('support_frames'), evidence_timestamp=candidate.get('evidence_timestamp')))
            feature_ms = (time.perf_counter()-t0)*1000
            row = dict(event=key, session=alias, event_id=sid, truth=truth, current=winner,
                       recorded_emission=trace['outcome'].get('final_camera_xy'), indices=event_indices,
                       context=metadata, rejected_patches=rejected_patches,
                       timing_ms=dict(context=context_ms, proposals=proposal_ms, features=feature_ms),
                       counts=dict(current=sum(pool=='current' for pool,c in positions), expanded=len(expanded)),
                       trace_path=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       native_label_sha256=hashlib.sha256(gp.read_bytes()).hexdigest() if gp.exists() else None)
            event_rows.append(row)
            print(key, row['counts'], 'feature_ms',round(feature_ms), 'missing_patches',len(rejected_patches),flush=True)
    np.save(output/'features.npy', np.stack(vectors))
    report = dict(schema='accuracy-physical-dataset-1', status='OFFLINE_RESEARCH_S03_PROHIBITED',
                  sessions=sessions, reference=reference, proposal_budget=budget,
                  feature_names=feature_names, feature_contract=FEATURE_CONTRACT,
                  events=event_rows, records=records, unavailable=unavailable,
                  s02_status='DEVELOPMENT_USED; never independent validation',
                  single_impact_compatibility='Native labels preserved; possible S02:1 ambiguity reported also through exclusion sensitivity',
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'dataset.json').write_text(json.dumps(report, indent=2)+'\n')
    print('DATASET',output,'events',len(event_rows),'rows',len(records),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sessions', nargs='+', choices=tuple(COHORT), default=list(COHORT))
    parser.add_argument('--reference', choices=('snapshot', 'history_early', 'history_guarded'), default='snapshot')
    parser.add_argument('--budget', type=int, default=256)
    args = parser.parse_args()
    build(args.output, args.sessions, args.reference, args.budget)


if __name__ == '__main__':
    main()
