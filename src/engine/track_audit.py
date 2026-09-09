"""Opt-in, observational tracking ledger. Never supplies detector authority.

Candidate ids identify consumption attempts (not detector list positions). Source
history includes same-frame support, whose candidate does not replace track XY.
"""
from __future__ import annotations
from copy import deepcopy
import math
import time

SCHEMA = 'complete-track-audit-1'


def source(c):
    if c.get('v2225_fast_extract', 0) > .5: return 'FAST'
    if c.get('detector_v1', 0) > .5: return 'V1'
    if c.get('detector_v2', 0) > .5: return 'V2'
    return 'UNKNOWN'


def enabled(scanner):
    return bool(getattr(scanner, 'physical_trace_capture_enabled', False))


def track_state(t):
    c = dict(t.last_candidate or {})
    return dict(track_id=t.track_id, camera_x=t.camera_x, camera_y=t.camera_y,
                created_at=t.created_at, first_seen_ts=t.first_seen_ts,
                last_seen_ts=t.last_seen_ts, hits=t.hits, best_score=t.best_score,
                current_score=c.get('score'), state=t.state, emitted=t.emitted,
                missed_frames=t.missed_frames, last_candidate=c, source=source(c),
                producer_shot_id=c.get('v2224_producer_shot_id'),
                dispatch_owner=getattr(t,'_audit_owner_shot_id',None),
                unique_frame_hits=getattr(t, 'v2226_unique_frame_hits', t.hits),
                observation_id=getattr(t, '_audit_observation_id', None),
                source_history=[dict(h) for h in getattr(t, '_audit_source_history', [])],
                local_confirmation={k:v for k,v in c.items() if 'confirm' in k},
                local_confirmed=c.get('v2225_local_confirm', 0) > .5)


class AssociationBatch:
    def __init__(self, scanner, frame_ts, candidates):
        self.scanner = scanner
        self.frame_ts = frame_ts
        seq = getattr(scanner, '_audit_batch_seq', 0) + 1
        scanner._audit_batch_seq = seq
        self.seq = seq
        self.rows = []
        self.before_ids = set(scanner._active_tracks)
        self.owner = getattr(scanner, 'last_trace_pipeline_shot_id', None)
        events = list(getattr(scanner, 'audio_events', []))
        if self.owner is None:
            earlier = [e for e in events if e.peak_ts <= frame_ts]
            self.owner = max(earlier, key=lambda e:e.peak_ts).shot_id if earlier else None
        self.data = dict(batch_id=seq, frame_ts=frame_ts, observed_at=time.time(),
                         dispatch_owner=self.owner, records=self.rows)

    def before(self, track):
        if track is None: return None
        return dict(xy=[track.camera_x, track.camera_y], source=source(track.last_candidate),
                    best_score=track.best_score, current_score=track.last_candidate.get('score'))

    def record(self, candidate, track, before, reason, distance=None):
        oid = f'{self.owner}:{self.seq}:{len(self.rows)}'
        hist = getattr(track, '_audit_source_history', [])
        src = source(candidate)
        if not hist or hist[-1]['source'] != src:
            hist.append(dict(source=src, first_observation_id=oid, frame_ts=self.frame_ts,
                             producer_shot_id=candidate.get('v2224_producer_shot_id'), dispatch_owner=self.owner, count=1))
        else: hist[-1]['count'] += 1
        track._audit_source_history = hist
        if reason != 'same_frame_support':
            track._audit_observation_id = oid
            track._audit_owner_shot_id = candidate.get('v2224_producer_shot_id',self.owner)
        self.rows.append(dict(observation_id=oid, candidate=dict(candidate),
            producer_shot_id=candidate.get('v2224_producer_shot_id'), dispatch_owner=self.owner,
            action='CREATED_NEW_TRACK' if before is None else 'ASSOCIATED_EXISTING_TRACK',
            reason=reason, track_id=track.track_id, association_distance=distance,
            association_threshold=self.scanner.track_merge_radius_px, before=before,
            after=dict(xy=[track.camera_x,track.camera_y], source=source(track.last_candidate),
                       best_score=track.best_score, current_score=track.last_candidate.get('score'),
                       observation_id=track._audit_observation_id)))

    def finish(self):
        self.data['dropped_track_ids'] = sorted((self.before_ids | {r['track_id'] for r in self.rows}) - set(self.scanner._active_tracks))
        self.data['drop_rule'] = 'HitScanner._drop_dead_tracks (missed frames and age/emitted gates)'
        batches = getattr(self.scanner, '_audit_batches', None)
        if batches is None: self.scanner._audit_batches = batches = []
        batches.append(self.data)


def begin_batch(scanner, candidates, frame_ts):
    return AssociationBatch(scanner, frame_ts, candidates) if enabled(scanner) else None


def eligibility(t, event, lead, lag, policy):
    c = t['last_candidate']
    sid = c.get('v2224_producer_shot_id')
    if policy != 'V252' and sid is not None and int(sid) != int(event['shot_id']): return 'producer_shot_id_mismatch'
    onset = t['first_seen_ts'] - event['peak_ts']
    if onset < -lead: return 'onset_before_association_lead'
    if onset > lag: return 'onset_after_association_lag'
    if t['emitted'] and event.get('matched_track_id') != t['track_id']: return 'already_emitted_for_other_event'
    if policy == 'V251' and c.get('v251_confirm_score', 0) <= 0: return 'v251_confirm_score_not_positive'
    if policy == 'V252':
        if c.get('v252_fresh_physical',0) <= .5: return 'v252_fresh_physical_failed'
        if c.get('v252_confirm_score',c.get('v252_authority_score',0)) <= 0: return 'v252_score_not_positive'
    if policy == 'V253':
        p = policy.lower()
        if c.get(p+'_authority_ok',0) <= .5 or c.get(p+'_confirm_score',0) <= 0: return p+'_authority_or_confirm_score_failed'
    return None


def rank_key(t, event, policy):
    c=t['last_candidate']; onset=abs(t['first_seen_ts']-event['peak_ts'])
    if policy=='BASE': return [onset,-t['best_score']]
    if policy=='V251': return [-c.get('v251_confirm_score',0),-c.get('v251_region_evidence',0),-t['best_score'],onset]
    if policy=='V253': return [-c.get('v253_confirm_score',0),-c.get('v253_distance_novelty',0),-c.get('v253_group_excess',0),onset]
    if policy=='V252': return [-c.get('v252_confirm_score',c.get('v252_authority_score',0)),-c.get('v252_group_excess',0),-c.get('v252_physical_score',0),onset]
    raise ValueError('Unsupported selector policy: '+policy)


def record_selection(scanner, event, selected, policy='BASE', gate=None):
    if not enabled(scanner): return
    observed_at=time.time()
    next_peak=min((e.peak_ts for e in getattr(scanner,'audio_events',[]) if e.peak_ts>event.peak_ts),default=math.inf)
    ev=dict(shot_id=event.shot_id, peak_ts=event.peak_ts, matched_track_id=event.matched_track_id)
    rows=[track_state(t) for t in scanner._active_tracks.values()]
    for row in rows:
        row['rejection_reason']=gate or eligibility(row,ev,scanner.association_lead_s,scanner.association_lag_s,policy)
        row['eligible']=row['rejection_reason'] is None
        owner=row['producer_shot_id'] if row['producer_shot_id'] is not None else row['dispatch_owner']
        row['causal_ownership']=('UNKNOWN' if owner is None else ('CAUSALLY_AVAILABLE' if owner==event.shot_id and row['last_seen_ts']<next_peak else 'CROSS_EVENT_OR_FUTURE'))
        row['readiness_inputs']={'hits':row['hits'], 'span_s':row['last_seen_ts']-row['first_seen_ts'], 'required_hits':scanner.track_confirm_frames, 'required_span_s':scanner.track_confirm_span_s}
        row['rank_key']=rank_key(row,ev,policy)
        row['final_rank']=None
        row['selected']=selected is not None and row['track_id']==selected.track_id
    ranked=sorted((r for r in rows if r['eligible']),key=lambda r:r['rank_key'])
    for rank,row in enumerate(ranked,1): row['final_rank']=rank
    snapshots=getattr(scanner,'_audit_selections',None)
    if snapshots is None: scanner._audit_selections=snapshots={}
    snapshots[event.shot_id]=dict(schema=SCHEMA, complete=True, policy=policy, gate=gate,
        event=ev, association_lead_s=scanner.association_lead_s, association_lag_s=scanner.association_lag_s,
        decision_observed_at=observed_at, pool_semantics='all active tracks; eligibility precedes readiness; causal ownership is separate', tracks=rows, selected_track_id=None if selected is None else selected.track_id)


def capture(scanner,event,track=None):
    """Freeze before emission; failed/missing hooks are explicit, never guessed."""
    saved=getattr(scanner,'_audit_selections',{}).get(event.shot_id)
    # Finished batches and selection rows are immutable after construction.
    # Share them until the recorder's _safe serialization instead of copying
    # hundreds of kilobytes twice on the critical emission path.
    snap=dict(saved) if saved is not None else None
    if snap is None: return dict(schema=SCHEMA,complete=False,reason='selection_hook_not_observed')
    if track is not None and snap['selected_track_id']!=track.track_id:
        return dict(schema=SCHEMA,complete=False,reason='selection_hook_does_not_match_emission')
    snap['associations']=list(getattr(scanner,'_audit_batches',[]))
    return snap


def record_not_consumed(scanner, candidates, frame_ts, owner, reason):
    if not enabled(scanner) or not candidates: return
    batch=AssociationBatch(scanner,frame_ts,candidates)
    batch.owner=owner;batch.data['dispatch_owner']=owner
    for candidate in candidates:
        batch.rows.append(dict(observation_id=f'{owner}:{batch.seq}:{len(batch.rows)}',
            candidate=dict(candidate), dispatch_owner=owner, action='NOT_CONSUMED',
            reason=reason, track_id=None))
    batches=getattr(scanner,'_audit_batches',None)
    if batches is None: scanner._audit_batches=batches=[]
    batches.append(batch.data)
