"""Bounded actual-XY witnesses from an existing track audit; offline only.

No new track ownership or confirmation mechanism. Slot names are nomination
policies, not physical probabilities. Nothing here changes a live track.
"""
from __future__ import annotations

import math


def observed_hypotheses(snapshot, cutoff, *, extra_budget=32, max_tracks=200):
    """Keep representative, first alternative, best confirmed, latest alternative.

    O(observations + tracks) time, at most four XY slots/track and three witnesses
    per slot while scanning. Return all eligible representatives plus <=32 extra
    parent/XY hypotheses. Round-robin nomination prevents one track taking the
    entire event budget. Every extra XY comes directly from a recorded association.
    Same-frame records are never described as independent temporal confirmations.
    """
    if extra_budget < 0 or max_tracks < 1 or not math.isfinite(cutoff):
        raise ValueError('Invalid hypothesis bound/cutoff')
    tracks = [t for t in snapshot['tracks'] if t['eligible']]
    if len(tracks) > max_tracks:
        raise ValueError('Track budget exceeded; refusing to silently drop current tracks')
    shot = int(snapshot['event']['shot_id'])
    slots = {t['track_id']: {} for t in tracks}
    centers = {t['track_id']: (float(t['camera_x']), float(t['camera_y'])) for t in tracks}
    rejected = dict(future_frame=0, future_delivery=0, other_producer=0, nonfinite_xy=0)
    for batch in snapshot.get('associations', []):
        frame = float(batch['frame_ts'])
        delivered = batch.get('observed_at')
        if not math.isfinite(frame) or frame > cutoff:
            rejected['future_frame'] += len(batch['records']); continue
        if delivered is not None and (not math.isfinite(float(delivered)) or delivered > cutoff):
            rejected['future_delivery'] += len(batch['records']); continue
        for row in batch['records']:
            track_id = row['track_id']
            if track_id not in slots:
                continue
            c = row['candidate']
            owners = [c.get('v2224_producer_shot_id'), row.get('producer_shot_id'),
                      row.get('dispatch_owner'), batch.get('dispatch_owner')]
            if any(p is not None and int(p) != shot for p in owners):
                rejected['other_producer'] += 1; continue
            point = (float(c['camera_x']), float(c['camera_y']))
            if not all(math.isfinite(v) for v in point):
                rejected['nonfinite_xy'] += 1; continue
            confirm = c.get('v2225_local_confirm') == 1
            witness = dict(observation_id=row.get('observation_id'), frame_ts=frame,
                observed_at=delivered, producer_shot_id=owners[0] if owners[0] is not None else owners[1],
                dispatch_owner=owners[2] if owners[2] is not None else owners[3],
                source_flags={k:c[k] for k in ('detector_v1','detector_v2') if k in c},
                local_confirmation={k:v for k,v in c.items() if k.startswith('v2225_confirm_')} if confirm else None)
            values = slots[track_id]
            # Reuse a currently held XY record; no unconstrained history copy.
            record = next((v for v in values.values() if tuple(v['xy']) == point), None)
            if record is None:
                record = dict(xy=point, first=witness, last=witness, confirmation=None)
            if (frame, str(witness['observation_id'])) < (record['first']['frame_ts'], str(record['first']['observation_id'])):
                record['first'] = witness
            if frame >= record['last']['frame_ts']:
                record['last'] = witness
            if confirm:
                previous = record['confirmation']
                strength = float(c.get('v2225_confirm_compact', 0.))
                if previous is None or strength > previous['local_confirmation'].get('v2225_confirm_compact', 0.):
                    record['confirmation'] = witness
            if point == centers[track_id]:
                values['current'] = record
                continue
            if 'first' not in values or frame < values['first']['first']['frame_ts']:
                values['first'] = record
            if 'latest' not in values or frame >= values['latest']['last']['frame_ts']:
                values['latest'] = record
            if confirm:
                old = values.get('confirmed')
                if old is None or strength > old['confirmation']['local_confirmation'].get('v2225_confirm_compact', 0.):
                    values['confirmed'] = record
    output = []
    seen = set()
    def emit(track_id, kind, record):
        point = centers[track_id] if record is None else tuple(record['xy'])
        key = (track_id, point)
        if key in seen:
            return False
        seen.add(key)
        witnesses = [] if record is None else list({(w['frame_ts'], w['observation_id']):w
            for w in (record['first'], record['last'], record['confirmation']) if w is not None}.values())
        output.append(dict(track_id=track_id, camera_x=point[0], camera_y=point[1],
            nomination=kind, cutoff=cutoff, witnesses=witnesses,
            actual_observation=bool(witnesses),
            local_confirmation=None if record is None else record['confirmation'],
            representative_only=record is None))
        return True
    ordered = sorted(slots)
    for track_id in ordered:
        emit(track_id, 'current', slots[track_id].get('current'))
    used = 0
    for kind in ('first', 'confirmed', 'latest'):
        for track_id in ordered:
            if used >= extra_budget:
                break
            if kind in slots[track_id]:
                used += emit(track_id, kind, slots[track_id][kind])
    return dict(hypotheses=output, rejected=rejected, extra_count=used,
        max_retained_slots=4*len(tracks), max_retained_witnesses=12*len(tracks),
        confirmation_semantics='Only local proof recorded at exactly this XY; no parent confirmation inherited')


def retain_exact_physical_witnesses(candidates, ordinal, capacity=24):
    """Keep exact supported coordinates; revoke normally; never merge signals.

    Distinct supported XY bypass producer-score NMS, NOT the hard capacity limit.
    Caller has already associated identical XY and recomputed prefix support.
    """
    if not 1<=capacity<=24 or len(candidates)>24+8:
        raise ValueError('Exact-witness experiment is bounded to 24 retained / 32 competing')
    from src.engine.shot_region_proposal_v251 import _dedupe_candidates
    for c in candidates:
        if 'physical_witness' in c and (c['physical_witness']['state']=='SUPPORTED') != c['temporal']['supported']:
            raise ValueError('Witness/temporal support mismatch')
    supported=[c for c in candidates if c['temporal']['supported']]
    pending=_dedupe_candidates(
        [c for c in candidates if not c['temporal']['supported']],radius=3.5)
    pending=[c for c in pending if not any(
        math.hypot(c['camera_x']-s['camera_x'],c['camera_y']-s['camera_y'])<=3.5
        for s in supported)]
    deduped=supported+pending
    eligible=[c for c in deduped if c['temporal']['supported'] or ordinal-c['born']<3]
    eligible.sort(key=lambda c:(bool(c['temporal']['supported']),
        c['temporal']['support'] if c['temporal']['supported'] else c['v251_region_evidence'],c['score']),reverse=True)
    kept=eligible[:capacity]; ids={c['uid'] for c in kept}
    dedup_ids={c['uid'] for c in deduped};eligible_ids={c['uid'] for c in eligible}
    reasons={c['uid']:('spatial_representative_replaced' if c['uid'] not in dedup_ids else
        'unsupported_lifetime_expired' if c['uid'] not in eligible_ids else 'capacity')
        for c in candidates if c['uid'] not in ids}
    return kept,reasons
