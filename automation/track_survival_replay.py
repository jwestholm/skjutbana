"""Read-only complete-track reconstruction/audit; generated reports stay local."""
from __future__ import annotations
import argparse, json, math, statistics
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from src.engine.offline.track_replay import current_exact_replay, reconstruct_legacy
from src.engine.track_audit import source


def distance(c,g): return math.hypot(c['camera_x']-g['camera_x'],c['camera_y']-g['camera_y'])


def metrics(errors,total):
    v=sorted(x for x in errors if x is not None)
    return dict(total=total,selected=len(v),hits={str(k):sum(x<=k for x in v) for k in (5,10,20,42)},
        mean=statistics.mean(v) if v else None,median=statistics.median(v) if v else None,
        p95=v[math.ceil(.95*len(v))-1] if v else None,errors_gt100=sum(x>100 for x in v),misses=total-len(v))


def choose(snapshot,mode):
    rows=[r for r in snapshot['tracks'] if r['eligible']]
    non=[r for r in rows if r['source']!='FAST']
    if mode=='EXCLUDE_FAST': rows=non
    if mode=='PREFER_NONFAST': rows=non or rows
    if mode=='SIGNED_LOCAL_CONTRAST':
        rows=[r for r in rows if r['local_confirmed']]
        # Frozen one-feature physical contrast hypothesis. No source weights/GT.
        return min(rows,key=lambda r:(-(r['last_candidate'].get('v2225_confirm_darkening',0)-r['last_candidate'].get('v2225_confirm_ring_abs',0)),r['rank_key'])) if rows else None
    return min(rows,key=lambda r:r['rank_key']) if rows else None


def choice_record(snapshot,mode,gt,decision_ts):
    """One recorded decision point; an unready alternate is not a final timeout."""
    from src.engine.camera.hit_scanner import HitScanner
    row=choose(snapshot,mode)
    if row is None: return None
    requirements=row['readiness_inputs']
    scanner=SimpleNamespace(track_confirm_frames=requirements['required_hits'],track_confirm_span_s=requirements['required_span_s'])
    ready=HitScanner._track_is_ready(scanner,SimpleNamespace(**row),decision_ts,SimpleNamespace(**snapshot['event']))
    return dict(track_id=row['track_id'],error=distance(row,gt) if ready else None,source=row['source'],
        ready=ready,status='READY_AT_RECORDED_DECISION' if ready else 'NOT_READY_AT_RECORDED_DECISION')


def audit(trace,gt,snapshot):
    current_exact_replay(snapshot,trace['decision_input']['deterministic_selection'])
    pool=trace['decision_input']['retained_candidates']; true=min(pool,key=lambda c:distance(c,gt))
    batches=[b for b in snapshot['associations'] if b.get('dispatch_owner')==trace['shot_id']]
    matches=[r for b in batches for r in b['records'] if r['action'] in ('CREATED_NEW_TRACK','ASSOCIATED_EXISTING_TRACK') and r['candidate']['camera_x']==true['camera_x'] and r['candidate']['camera_y']==true['camera_y'] and r['candidate']['score']==true['score'] and r['candidate'].get('timestamp')==true.get('timestamp') and source(r['candidate'])==source(true)]
    assert len(matches)==1,'ambiguous true candidate observation'
    match=matches[0]; tid=match['track_id'];track=next(r for r in snapshot['tracks'] if r['track_id']==tid)
    history=[r for b in batches for r in b['records'] if r['track_id']==tid]
    eligible=[r for r in snapshot['tracks'] if r['eligible']]
    top8=sorted(snapshot['tracks'],key=lambda r:-r['best_score'])[:8]
    stats={}
    for name,seq in [('causal_retained',pool),('consumed',[r['candidate'] for r in batches[0]['records']]),('created',[r['candidate'] for b in batches for r in b['records'] if r['action']=='CREATED_NEW_TRACK'])]: stats[name]=dict(Counter(source(c) for c in seq))
    for name,seq in [('tracks',snapshot['tracks']),('local_confirmed',[r for r in snapshot['tracks'] if r['local_confirmed']]),('eligible',eligible),('top8',top8)]:stats[name]=dict(Counter(r['source'] for r in seq))
    # A cluster is defined here by actual runtime association, not visual identity.
    cluster=Counter(r['track_id'] for r in batches[0]['records'] if source(r['candidate'])=='FAST')
    stats['fast_association_clusters']=dict(proposals=sum(cluster.values()),clusters=len(cluster),multi_candidate_clusters=sum(n>1 for n in cluster.values()),max_cluster=max(cluster.values(),default=0))
    stats['same_frame_support']=sum(r['reason']=='same_frame_support' for b in batches for r in b['records'])
    stats['source_changes']=sum(r['before'] is not None and r['before']['source']!=r['after']['source'] for b in batches for r in b['records'])
    stats['historical_best_above_current']=sum(r['best_score']>r['current_score']+1e-9 for r in snapshot['tracks'])
    stats['selected_source']=next(r['source'] for r in snapshot['tracks'] if r['selected'])
    return dict(shot=trace['shot_id'],gt=gt,nearest=true,nearest_distance=distance(true,gt),true_track=track,
        true_track_distance=distance(track,gt),true_association=match,history=history,counts=stats,
        fate='NO_CAUSAL_ORACLE_AT42' if distance(true,gt)>42 else ('G_SELECTED_CORRECTLY' if track['selected'] else ('E_ELIGIBLE_OUTSIDE_TOP8' if track['eligible'] and tid not in [r['track_id'] for r in top8] else 'F_ELIGIBLE_OUTRANKED')),
        primary_loss='causal_retained_generation' if distance(true,gt)>42 else ('final_ranking' if not track['selected'] else None),
        choices={mode:choice_record(snapshot,mode,gt,trace['decision_input']['timestamp']) for mode in ('CURRENT','EXCLUDE_FAST','PREFER_NONFAST','SIGNED_LOCAL_CONTRAST')},
        top8_choices={mode:choice_record({**snapshot,'tracks':top8},mode,gt,trace['decision_input']['timestamp']) for mode in ('CURRENT','EXCLUDE_FAST','PREFER_NONFAST')},
        verification=snapshot.get('reconstruction'),snapshot=snapshot)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exclude-events',type=int,nargs='*',default=[],help='Explicit independently classified no-physical-shot event ids')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for path in sorted((a.root/'shots').glob('*/trace.json')):
        trace=json.loads(path.read_text());gp=path.parent/'ground_truth.json'
        if trace['shot_id'] in a.exclude_events: continue
        if not gp.exists(): continue
        gt=json.loads(gp.read_text())
        if 'camera_x' not in gt: continue
        # Fail loudly on mismatch. Never silently downgrade failed reconstruction.
        snap=trace.get('decision_input',{}).get('complete_track_audit')
        if not snap: snap=reconstruct_legacy(trace)
        rows.append(audit(trace,gt,snap))
    result=dict(status='RESEARCH_ONLY',input_session=str(a.root),excluded_events=a.exclude_events,evidence='COMPLETE_ELIGIBLE_REPLAY verified recorded-input reconstruction',rows=rows,
        metrics={mode:metrics([r['choices'][mode]['error'] if r['choices'][mode] else None for r in rows],len(rows)) for mode in ('CURRENT','EXCLUDE_FAST','PREFER_NONFAST','SIGNED_LOCAL_CONTRAST')})
    result['top8_metrics']={mode:metrics([r['top8_choices'][mode]['error'] if r['top8_choices'][mode] else None for r in rows],len(rows)) for mode in ('CURRENT','EXCLUDE_FAST','PREFER_NONFAST')}
    result['causal_retained_oracle']=metrics([r['nearest_distance'] for r in rows],len(rows))
    positives=[r for r in rows if r['nearest_distance']<=42]
    result['true_candidate_funnel_at42']=dict(physical_shots=len(rows),causal_retained=len(positives),
        tracked=sum(r['true_track_distance']<=42 for r in positives),
        locally_confirmed=sum(r['true_track_distance']<=42 and r['true_track']['local_confirmed'] for r in positives),
        eligible=sum(r['true_track_distance']<=42 and r['true_track']['eligible'] for r in positives),
        selected=sum(r['true_track_distance']<=42 and r['true_track']['selected'] for r in positives))
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['metrics'],indent=2))
    for r in rows: print(r['shot'],round(r['nearest_distance'],2),r['true_track']['track_id'],r['true_track']['final_rank'],r['counts'])
if __name__=='__main__':main()
