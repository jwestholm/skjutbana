"""Strict complete-pool replay. Missing inputs are unavailable; mismatches fail."""
from __future__ import annotations
from copy import deepcopy
import math
from src.engine.track_audit import eligibility, rank_key


def current_exact_replay(snapshot, expected=None):
    if not snapshot.get('complete'): raise ValueError('CURRENT_EXACT_REPLAY: incomplete snapshot')
    event=snapshot['event']; policy=snapshot['policy']; rows=snapshot['tracks']
    if len({r['track_id'] for r in rows})!=len(rows): raise AssertionError('duplicate track ids')
    ranked=[]
    for r in rows:
        reason=snapshot.get('gate') or eligibility(r,event,snapshot['association_lead_s'],snapshot['association_lag_s'],policy)
        assert r['rejection_reason']==reason, ('eligibility mismatch',r['track_id'])
        assert r['eligible']==(reason is None), ('eligible flag mismatch',r['track_id'])
        key=rank_key(r,event,policy)
        assert r['rank_key']==key, ('rank metric mismatch',r['track_id'])
        if reason is None: ranked.append(r)
    ranked.sort(key=lambda r:rank_key(r,event,policy))
    for i,r in enumerate(ranked,1): assert r['final_rank']==i, ('rank mismatch',r['track_id'])
    chosen=ranked[0] if ranked else None
    sid=None if chosen is None else chosen['track_id']
    assert sid==snapshot['selected_track_id'], ('CURRENT_EXACT_REPLAY mismatch',sid,snapshot['selected_track_id'])
    assert [r['track_id'] for r in rows if r['selected']]==([] if sid is None else [sid]), 'selected flags mismatch'
    if expected is not None:
        assert chosen is not None and sid==expected['track_id'], 'emission track mismatch'
        for k in ('camera_x','camera_y'): assert chosen[k]==expected[k], ('emission XY mismatch',k)
    return chosen


def reconstruct_legacy(trace):
    """Conditional reconstruction from two captured input batches; verify all
    pre-decision top-eight snapshots. Does not manufacture absent histories.
    Only supports the global BASE two-frame path present in the post-PRE run.
    """
    from src.engine.camera.hit_scanner import HitScanner, AudioShotEvent
    from src.engine.shot_track_v2226 import update_tracks_frame_unique_v2226
    from src.engine.track_audit import capture
    dec=trace['decision_input']; cs=deepcopy(dec['retained_candidates'])
    conf=dec.get('local_confirmation') or {}
    # Legacy synchronous confirmation snapshots may be appended after emission.
    # An exact winner confirmation frame plus matching event id proves that
    # call ran before the selected track consumed it; never take a later frame.
    if not conf:
        selected=dec.get('deterministic_selection') or {}
        stamp=selected.get('last_seen_ts')
        if (selected.get('last_candidate') or {}).get('v2225_local_confirm',0)>.5:
            conf=next((s['local_confirmation'] for s in trace.get('stages',[]) if (s.get('local_confirmation') or {}).get('shot_id')==trace['shot_id'] and (s.get('local_confirmation') or {}).get('frame_ts')==stamp),{})
    assert cs and conf.get('candidates'), 'missing proposal/confirmation inputs'
    sid=trace['shot_id']; ts=cs[0]['timestamp']; cts=conf['frame_ts']
    assert all(c['timestamp']==ts for c in cs), 'multiple proposal frames unsupported'
    assert dec.get('candidate_pool_shot_id',dec.get('shot_id'))==sid==conf.get('shot_id'), 'wrong event pool'
    assert cts < trace.get('next_audio_peak_ts',math.inf), 'cross-event confirmation'
    stages=[s for s in trace['stages'] if s['timestamp']<=dec['timestamp'] and s.get('candidate_pool_shot_id',s.get('event',{}).get('shot_id'))==sid and s.get('tracks')]
    assert stages, 'missing checkpoints'
    cfg=trace['context'].get('scanner_config',{})
    scanner=HitScanner(); scanner.physical_trace_capture_enabled=True
    for k in ('track_confirm_frames','track_confirm_span_s'):
        if cfg.get(k) is not None: setattr(scanner,k,cfg[k])
    event=AudioShotEvent(sid,trace['peak_ts'],trace['peak_ts']); scanner.audio_events.append(event)
    scanner.last_trace_pipeline_shot_id=sid
    update_tracks_frame_unique_v2226(scanner,cs,ts)
    # Infer only the id offset, from exact XY of every exported first-frame track.
    first=[s for s in stages if all(t['last_seen_ts']==ts for t in s['tracks'])]
    assert first, 'missing first-frame checkpoint'
    offsets=set()
    for t in first[0]['tracks']:
        matched=[r for r in scanner._active_tracks.values() if r.camera_x==t['camera_x'] and r.camera_y==t['camera_y']]
        assert len(matched)==1, 'ambiguous id correspondence'
        offsets.add(t['track_id']-matched[0].track_id)
    assert len(offsets)==1, 'inconsistent id offsets'
    offset=offsets.pop()
    # Re-run with proven offset, never inferred candidate-list identity.
    scanner._active_tracks={}; scanner._next_track_id=1+offset; scanner._audit_batches=[]
    update_tracks_frame_unique_v2226(scanner,cs,ts)
    keys=('camera_x','camera_y','best_score','first_seen_ts','last_seen_ts','hits','missed_frames','emitted','state')
    checked=0
    def verify(group):
        nonlocal checked
        for s in group:
            # A debug view is refreshed after resolution; emission may change
            # state only. Pending snapshots below are exact pre-emission checks.
            if s.get('event',{}).get('state')!='pending': continue
            for t in s['tracks']:
                r=scanner._active_tracks[t['track_id']]
                for k in keys: assert getattr(r,k)==t[k], ('checkpoint mismatch',sid,t['track_id'],k,getattr(r,k),t[k])
                checked+=1
    verify(first)
    confirmations={cts: conf}
    for s in stages:
        c=s.get('local_confirmation') or {}
        if c.get('shot_id')==sid and ts < c.get('frame_ts',0) <= cts:
            confirmations[c['frame_ts']]=c
    remaining=[s for s in stages if s not in first]
    for frame_ts,c in sorted(confirmations.items()):
        update_tracks_frame_unique_v2226(scanner,c['candidates'],frame_ts)
        matching=[s for s in remaining if max(t['last_seen_ts'] for t in s['tracks'])==frame_ts]
        verify(matching)
        remaining=[s for s in remaining if s not in matching]
    assert not remaining, 'unrecorded frame update'
    selected=scanner._best_track_for_event(event)
    snapshot=capture(scanner,event,selected)
    current_exact_replay(snapshot,dec['deterministic_selection'])
    # Even if there was no refreshed second-frame debug view, the winner is a
    # separate exact checkpoint for all state fields before emission.
    expected=dec['deterministic_selection']
    for k in keys: assert getattr(selected,k)==expected[k], ('winner state mismatch',sid,k)
    snapshot['reconstruction']=dict(status='VERIFIED_INPUT_RECONSTRUCTION',checked_track_snapshots=checked,id_offset=offset,
        limitation='Full session empty-frame aging was not recorded; verification covers visible checkpoints and exact winner, not invisible prehistory.')
    return snapshot
