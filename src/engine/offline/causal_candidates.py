"""Additive physical-trace causality diagnostics; never used by live authority.

Frame time is evidence time, not worker delivery time. The emission-boundary
snapshot proves delivery. Historical observations prove it only when recorded
before that boundary and owned by this event. Repeated observations are not
independent candidates. Track coordinates remain distinct from proposal XY.
"""
from __future__ import annotations

import json
import math

CLASSES = ('CAUSALLY_AVAILABLE', 'POST_DECISION', 'CROSS_EVENT_OR_FUTURE', 'UNKNOWN')
COLORS = dict(zip(CLASSES, ((0, 220, 100), (255, 180, 0), (240, 70, 220), (150, 150, 150))))


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def classify(*, cutoff, observed_at, evidence_at, owner, shot_id, next_peak=None, decision_snapshot=False):
    cutoff, observed_at, evidence_at, next_peak = map(finite, (cutoff, observed_at, evidence_at, next_peak))
    # Explicit contradictory ownership/timing must never become causal merely
    # because a producer copied a pool at the decision boundary.
    if owner is not None and str(owner) != str(shot_id):
        return 'CROSS_EVENT_OR_FUTURE'
    if next_peak is not None and evidence_at is not None and evidence_at >= next_peak:
        return 'CROSS_EVENT_OR_FUTURE'
    if cutoff is None:
        return 'UNKNOWN'
    if evidence_at is not None and evidence_at > cutoff:
        return 'POST_DECISION'
    if decision_snapshot:
        return 'CAUSALLY_AVAILABLE'
    if observed_at is None or owner is None:
        return 'UNKNOWN'
    return 'CAUSALLY_AVAILABLE' if observed_at <= cutoff else 'POST_DECISION'


def candidate_key(c):
    # Feature versions are preserved: confirmation overwrites timestamp and may
    # supply new evidence at the same XY. Do not backdate those features.
    return json.dumps(c, sort_keys=True, separators=(',', ':'))


def analyze_trace(trace, next_peak=None, ground_truth=None):
    sid = trace['shot_id']
    decision = trace.get('decision_input') or {}
    cutoff = finite(decision.get('timestamp'))
    records = {}
    observations = []

    def add(pool, origin, observed, owner, boundary=False):
        if not isinstance(pool, list):
            return
        for c in pool:
            if not isinstance(c, dict) or finite(c.get('camera_x')) is None or finite(c.get('camera_y')) is None:
                continue
            category = classify(cutoff=cutoff, observed_at=observed, evidence_at=c.get('timestamp'),
                                owner=owner, shot_id=sid, next_peak=next_peak, decision_snapshot=boundary)
            key = candidate_key(c)
            rec = records.setdefault(key, {'candidate': c, 'origins': [], 'availability': category,
                                          'first_observed_at': observed})
            rec['origins'].append(origin)
            # A prior proof of consumption takes precedence over later copies.
            if category == 'CAUSALLY_AVAILABLE':
                rec['availability'] = category
            if observed is not None and (rec['first_observed_at'] is None or observed < rec['first_observed_at']):
                rec['first_observed_at'] = observed

    add(decision.get('retained_candidates'), 'decision_retained', cutoff,
        decision.get('candidate_pool_shot_id', sid), True)
    selected = decision.get('deterministic_selection') or {}
    add([selected.get('last_candidate')], 'decision_track_last_candidate', cutoff, sid, True)
    add(decision.get('local_confirmation', {}).get('candidates') if isinstance(decision.get('local_confirmation'), dict) else None,
        'decision_local_confirmation', cutoff, sid, True)
    for i, stage in enumerate(trace.get('stages', [])):
        ts, owner = finite(stage.get('timestamp')), stage.get('candidate_pool_shot_id')
        observations.append({'index': i, 'timestamp': ts, 'candidate_pool_shot_id': owner,
                             'event_state': (stage.get('event') or {}).get('state'),
                             'candidate_count': len(stage.get('candidates') or []),
                             'track_count': len(stage.get('tracks') or []),
                             'registration': {k: v for k, v in (stage.get('window_debug') or {}).items()
                                              if 'registration' in k or 'pre_' in k or 'exposure' in k}})
        add(stage.get('candidates'), f'observation:{i}:retained', ts, owner)
        pipeline = stage.get('pipeline')
        if isinstance(pipeline, dict):
            for name in ('raw_candidates', 'filtered_candidates', 'retained_candidates', 'confirmed_candidates', 'ranked_candidates'):
                add(pipeline.get(name), f'observation:{i}:{name}', ts, owner)
        confirmation = stage.get('local_confirmation')
        if isinstance(confirmation, dict):
            # Local confirmation is synchronous, immediately precedes tracking
            # and resolution. Same event and same frame as the captured winning
            # track prove that this output already existed at the boundary.
            proven = (str(confirmation.get('shot_id')) == str(sid) and cutoff is not None
                      and finite(confirmation.get('frame_ts')) == finite(selected.get('last_seen_ts'))
                      and finite(confirmation.get('frame_ts')) is not None
                      and float(confirmation['frame_ts']) <= cutoff)
            add(confirmation.get('candidates'), f'observation:{i}:local_confirmation',
                cutoff if proven else ts, confirmation.get('shot_id'), proven)
        for track in stage.get('tracks') or []:
            if isinstance(track, dict):
                add([track.get('last_candidate')], f'observation:{i}:track_debug', ts, owner)
    items = list(records.values())
    pools = {name: [r['candidate'] for r in items if r['availability'] == name] for name in CLASSES}
    def oracle(pool):
        if ground_truth is None:
            return None
        nearest = min((math.hypot(c['camera_x']-ground_truth['camera_x'], c['camera_y']-ground_truth['camera_y'])
                       for c in pool), default=None)
        return {'nearest_px': nearest, 'hits': {str(r): nearest is not None and nearest <= r for r in (5,10,20,42)}}
    return {'schema': 'causal-candidates-1', 'shot_id': sid, 'decision_cutoff': cutoff,
            'cutoff_semantics': 'capture_decision wall clock, immediately before emission/AI hook; frame timestamps alone do not prove worker delivery',
            'next_audio_peak': next_peak, 'counts_candidate_versions': {k: len(v) for k,v in pools.items()},
            'causal_oracle': oracle(pools['CAUSALLY_AVAILABLE']),
            'post_decision_only_oracle': oracle(pools['POST_DECISION']),
            'cross_event_only_oracle': oracle(pools['CROSS_EVENT_OR_FUTURE']),
            'posthoc_all_observed_oracle': oracle([r['candidate'] for r in items]),
            'records': items, 'observations': observations}


def overlay_candidates(trace, *, include_later=False, next_peak=None):
    result = analyze_trace(trace, next_peak)
    seen = set()
    for rec in result['records']:
        category, c = rec['availability'], rec['candidate']
        if not include_later and category != 'CAUSALLY_AVAILABLE':
            continue
        key = (c['camera_x'], c['camera_y'], category)
        if key not in seen:
            seen.add(key)
            yield c, category, COLORS[category]
