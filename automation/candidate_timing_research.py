"""Frozen-development timing audit and one serialized scheduling simulation.

No detector integration, feature/rank change, discovery of capture roots or GT
input to timing reconstruction/scheduling. Receipt time is not exposure time.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from automation import causal_pre_onset_research as reference

RUNTIME = reference.BASE/'causal_pre_runtime_20260925_01'
CAUSAL = reference.BASE/'causal_pre_onset_20260925_01'


def stamp(value, status, source, meaning):
    return dict(value=value, status='UNKNOWN' if value is None else status, source=source, meaning=meaning)


def xy(c):
    return (float(c['camera_x']), float(c['camera_y']))


def summarize(values):
    a=np.asarray([v for v in values if v is not None],float)
    if not len(a): return dict(count=0)
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),p05=float(np.percentile(a,5)),
                p95=float(np.percentile(a,95)),minimum=float(a.min()),maximum=float(a.max()))


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csv_rows(path, rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in rows)


def reconstruct(snapshot, trace, cutoff):
    """Coordinate availability witnesses, separate from first-delivery UNKNOWN.

    Native association observed_at is list-consumption start. Reconstructed
    association wall clocks are rejected. Recorded scanner snapshots give upper
    bounds for legacy lists. A new track XY made by association cannot be used
    at the input list's timestamp: wait for a post-batch witness instead.
    """
    peak=snapshot['event']['peak_ts'];sid=snapshot['event']['shot_id']
    native=bool(trace and (trace.get('decision_input') or {}).get('complete_track_audit')==snapshot)
    cached_only=trace is None
    # Original D01 cache can describe observation chronology; do not assert its
    # wall clocks native without the overwritten source to authenticate them.
    stages=sorted((s for s in (trace or {}).get('stages',[]) if s['timestamp']<=cutoff
        and (s.get('event') or {}).get('shot_id')==sid),key=lambda s:s['timestamp'])
    selector=snapshot.get('decision_observed_at') if native else None
    created=next(((s.get('event') or {}).get('created_at') for s in stages
                  if (s.get('event') or {}).get('created_at') is not None),None)
    batches=[];all_versions={};histories=defaultdict(list)
    for b in snapshot['associations']:
        if b.get('dispatch_owner')!=sid:continue
        frame=b['frame_ts'];witnesses=[]
        if native:witnesses.append((b['observed_at'],'RECORDED','native association list-consumption start'))
        for s in stages:
            pool=s.get('candidates') or []
            conf=s.get('local_confirmation') or {}
            matching=any(c.get('timestamp')==frame for c in pool) or (conf.get('frame_ts')==frame and conf.get('candidates'))
            if matching and s.get('candidate_pool_shot_id')==sid:
                witnesses.append((s['timestamp'],'RECORDED','original scanner snapshot contains this observation list'))
        if witnesses:available,status,source=min(witnesses)
        elif trace:
            available,status,source=cutoff,'DERIVED','verified final input reconstruction; available by decision, first delivery unknown'
        else:available,status,source=None,'UNKNOWN','original D01 metadata unavailable'
        ends=[s['timestamp'] for s in stages if available is not None and s['timestamp']>=available
              and any(t.get('last_seen_ts')==frame for t in s.get('tracks',[]))]
        if selector is not None:ends.append(selector)
        if trace:ends.append(cutoff)
        end=min(ends) if ends else None
        if available is not None and end is not None and end<available:raise ValueError('Association witness order')
        conf_ms=0.
        for s in [*(trace or {}).get('stages',[]),(trace or {}).get('decision_input',{})]:
            conf=s.get('local_confirmation') or {}
            if conf.get('frame_ts')==frame:conf_ms=max(conf_ms,float((conf.get('diagnostics') or {}).get('ms',0)))
        batches.append(dict(frame=frame,available_by=stamp(available,status,source,'list existed by this time; not first delivery'),
            consumed_by=stamp(end,'DERIVED','post-association witness','track updates complete by this time'),
            confirmation_cpu_ms=conf_ms,records=len(b['records'])))
        for rec in b['records']:
            if rec.get('track_id') is None or 'after' not in rec:continue
            if rec.get('producer_shot_id',sid) not in (None,sid):continue
            coord=tuple(rec['after']['xy']);input_coord=xy(rec['candidate'])
            release=available if coord==input_coord else end
            release_status=status if coord==input_coord else 'DERIVED'
            key=coord
            entry=dict(track_id=rec['track_id'],camera_xy=list(coord),observation=frame,
                available_by=stamp(release,release_status,source if coord==input_coord else 'post-association coordinate witness',
                    'exact XY existed by this time; not first delivery'),
                same_frame=rec.get('reason')=='same_frame_support',
                local_confirm=rec['candidate'].get('v2225_local_confirm',0)>.5,
                consumed_by=end,batch=len(batches)-1)
            histories[rec['track_id']].append(entry)
            if key not in all_versions or (release is not None and (all_versions[key]['available_by']['value'] is None
                    or release<all_versions[key]['available_by']['value'])):all_versions[key]=entry
    candidates=[]
    for t in snapshot['tracks']:
        if not t['eligible']:continue
        coord=xy(t);h=histories[t['track_id']];matching=[v for v in h if tuple(v['camera_xy'])==coord]
        seen=sorted({v['observation'] for v in h})
        c=all_versions.get(coord)
        ready=None;rule=t['readiness_inputs']
        for n,ts in enumerate(seen,1):
            end=max((v['consumed_by'] for v in h if v['observation']==ts and v['consumed_by'] is not None),default=None)
            if end is None:continue
            if n>=rule['required_hits'] and ts-seen[0]>=rule['required_span_s']:ready=end;break
            if n>=max(2,rule['required_hits']-1):ready=max(end,peak+.30);break
        if ready is not None and ready>cutoff:ready=None
        confirmed=[v for v in h if v['local_confirm']]
        candidates.append(dict(track_id=t['track_id'],camera_xy=list(coord),final_rank=t['final_rank'],
            first_observation=stamp(t['first_seen_ts'],'RECORDED','track first_seen_ts','image receipt observation, not delivery'),
            exact_xy_first_observation=stamp(min((v['observation'] for v in matching),default=None),'DERIVED','association coordinate versions','frame for exact final XY version'),
            exact_xy_available_by=c['available_by'] if c else stamp(None,'UNKNOWN','no witness','exact XY availability'),
            first_delivery=stamp(None,'UNKNOWN','not exported','producer completion / queue delivery instant'),
            first_eligibility_check=stamp(None,'UNKNOWN','only final selection eligibility recorded','actual first eligibility check'),
            readiness_derived=stamp(ready,'DERIVED','distinct association frames + recorded readiness rule + consumption upper bounds','logical readiness by, not actual first readiness check'),
            confirmation_observation=stamp(min((v['observation'] for v in confirmed),default=None),'RECORDED','local-confirm candidate timestamp','confirmation image receipt'),
            confirmation_available_by=stamp(min((batches[v['batch']]['available_by']['value'] for v in confirmed
                    if batches[v['batch']]['available_by']['value'] is not None),default=None),'DERIVED','confirmation list witness','confirmation values existed by'),
            observations=h,final_ready=(t['hits']>=rule['required_hits'] and t['last_seen_ts']-t['first_seen_ts']>=rule['required_span_s'])
                or (cutoff-peak>=.30 and t['hits']>=max(2,rule['required_hits']-1)),
            changed_coordinate=bool(matching and min(v['observation'] for v in matching)>t['first_seen_ts'])))
    return dict(native=native,cached_only=cached_only,peak=peak,created=created,selector=selector,
                batches=batches,candidates=candidates,versions=list(all_versions.values()))


def serialize_jobs(jobs, *, busy=()):
    """One nonpreemptive worker; stable input-order ties, no hidden parallelism.

    Jobs have release, cost_ms, name and optional dependency names. Busy windows
    are explicit modeled reservations. Never mutate jobs or begin before release.
    """
    remaining=[dict(j,order=i) for i,j in enumerate(jobs)];done={};out=[]
    if len({j['name'] for j in remaining})!=len(remaining):raise ValueError('Unique job identities required')
    for j in remaining:
        if not math.isfinite(j['release']) or not math.isfinite(j['cost_ms']) or j['cost_ms']<0:raise ValueError('Finite job times/costs required')
    cursor=-float('inf')
    while remaining:
        ready=[]
        for j in remaining:
            if all(k in done for k in j.get('depends',[])):
                release=max([j['release'],*(done[k] for k in j.get('depends',[]))])
                ready.append((max(cursor,release),j['order'],release,j))
        if not ready:raise ValueError('Missing/cyclic job dependency')
        start,_,release,j=min(ready,key=lambda v:v[:2]);duration=j['cost_ms']/1000
        for low,high in sorted(busy):
            if high>start and low<start+duration:start=high
        end=start+duration;done[j['name']]=end;cursor=end
        out.append(dict(name=j['name'],release=release,start=start,end=end,cost_ms=j['cost_ms'],kind=j.get('kind')))
        remaining.remove(j)
    return out


def simulate(frames, releases, final_xy, cutoff, costs, *, busy=()):
    """One fixed opportunity model: stream patch work, retain full final inference.

    Per patch cost is ESTIMATED from the prior measured event's batch patch cost.
    Includes all observed coordinate versions, not only future final survivors.
    Frame availability at receipt is an explicit optimistic assumption.
    """
    if not frames or any(a>=b for a,b in zip(frames,frames[1:])) or max(frames)>cutoff:
        raise ValueError('Ordered camera observations at/before cutoff required')
    if not final_xy or not final_xy<=releases.keys():raise ValueError('Required coordinates need availability witnesses')
    jobs=[];first=min(releases.values());frame_names=[];patch_names=[]
    for i,ts in enumerate(frames):
        name=f'frame:{i}';frame_names.append(name)
        jobs.append(dict(name=name,release=ts,cost_ms=costs['append_ms'],kind='continuous'))
    jobs.append(dict(name='setup',release=first,cost_ms=costs['setup_ms'],kind='setup'))
    for n,(coord,release) in enumerate(releases.items()):
        for i,ts in enumerate(frames):
            name=f'patch:{n}:{i}'
            jobs.append(dict(name=name,release=max(ts,release),cost_ms=costs['patch_ms']/(len(frames)*len(final_xy)),
                depends=['setup',frame_names[i]],kind='patch'))
            if coord in final_xy:patch_names.append(name)
    jobs.append(dict(name='final_evidence',release=max(frames),cost_ms=costs['inference_ms'],depends=patch_names,kind='finalize'))
    jobs.append(dict(name='select',release=max(frames),cost_ms=costs.get('selector_ms',0.),depends=['final_evidence'],kind='select'))
    schedule=serialize_jobs(jobs,busy=busy);last=next(x for x in schedule if x['name']=='select')
    latest=max(max(frames),max(releases[c] for c in final_xy))
    # Charge frame maintenance at final arrival too; prior 7/33 assumes already
    # maintained frames and is reported separately, not silently overwritten.
    final_data_time=max(max(frames),max(releases[c] for c in final_xy))
    final_evidence=next(x for x in schedule if x['name']=='final_evidence')
    return dict(evidence_ready=final_evidence['end'],selection_ready=last['end'],deadline=cutoff,
        slack_ms=(cutoff-last['end'])*1000,meets_deadline=last['end']<=cutoff,
        total_cpu_ms=sum(j['cost_ms'] for j in jobs),post_data_critical_ms=(last['end']-final_data_time)*1000,
        latest_required_input=latest,schedule=schedule)


def batch_baseline(frames, releases, final_xy, costs, *, busy=()):
    """Same single worker/maintenance reservations as the incremental model."""
    jobs=[dict(name=f'frame:{i}',release=ts,cost_ms=costs['append_ms']) for i,ts in enumerate(frames)]
    jobs.append(dict(name='query',release=max(max(frames),max(releases[c] for c in final_xy)),
        cost_ms=costs['total_ms'],depends=[j['name'] for j in jobs]))
    return serialize_jobs(jobs,busy=busy)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    data=json.loads(reference.CACHE.read_text());bench=json.loads((RUNTIME/'after/profiling_after.json').read_text())
    by_event={v['event']:v for v in bench['events'] if not v['event'].startswith('CAN:')}
    roots={Path(e['trace_path']).parent.parent.parent for e in data['events'] if e['session']!='D01'}
    access=reference.read_guard(output,roots);hashes={};events=[];candidates=[];feasibility=[]
    expected_trace=json.loads((RUNTIME/'inputs/trace_hashes.json').read_text())
    def read(p):
        p=Path(p);raw=p.read_bytes();hashes[str(p.resolve())]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    for e in data['events']:
        eid=e['event'];detail=read(reference.CACHE.parent/(eid.replace(':','_')+'_trace_details.json'));snap=detail['snapshot']
        tr=read(e['trace_path']) if e['session']!='D01' else None
        if tr and hashes[str(Path(e['trace_path']).resolve())]!=expected_trace[str(Path(e['trace_path']).resolve())]:raise ValueError('Trace changed')
        cutoff=e['context']['cutoff'];t=reconstruct(snap,tr,cutoff)
        spec=read(RUNTIME/'before'/eid.replace(':','_')/'event.json') if eid in by_event else None
        points=spec['points'] if spec else [[data['records'][i]['camera_x'],data['records'][i]['camera_y']] for i in e['pools']['current']]
        lookup={tuple(c['camera_xy']):c for c in t['candidates']}
        ordered=[lookup[tuple(p)] for p in points]
        frame_times=spec['times'] if spec else []
        for j,c in enumerate(ordered):
            c.update(event=eid,local_index=j)
            a=c['first_observation']['value'];ready=c['readiness_derived']['value'];available=c['exact_xy_available_by']['value'];last=frame_times[-1] if frame_times else None
            c['interval_ms']=dict(event_to_observation=(a-t['peak'])*1000,event_to_ready=None if ready is None else (ready-t['peak'])*1000,
                observation_to_ready=None if ready is None else (ready-a)*1000,ready_to_deadline=None if ready is None else (cutoff-ready)*1000,
                observation_to_last_causal_frame=None if last is None else (last-a)*1000,last_frame_to_deadline=None if last is None else (cutoff-last)*1000,
                event_to_exact_xy_known_by=None if available is None else (available-t['peak'])*1000,
                exact_xy_known_by_to_deadline=None if available is None else (cutoff-available)*1000)
            effect=spec['expected'][j] if spec else {}
            c['causal_pre']=stamp(frame_times[effect['pre_index']] if effect.get('pre_index') is not None else None,'DERIVED','frozen causal method','receipt timestamp of selected observed PRE')
            c['onset']=stamp(frame_times[effect['onset_index']] if effect.get('onset_index') is not None else None,'DERIVED','frozen causal method','receipt timestamp of observed onset')
            c['last_required_post']=stamp(last,'RECORDED','frozen full-prefix persistence and final-three support','last camera receipt; delivery unknown')
            candidates.append(c)
        settings=(tr or {}).get('provenance',{}).get('effective_detector_settings',{});cfg=(tr or {}).get('context',{}).get('scanner_config',{})
        row=dict(event=eid,peak=stamp(t['peak'],'RECORDED','audio peak','receipt/sample-estimated event timestamp'),
            event_created=stamp(t['created'],'RECORDED','scanner event.created_at','event construction wall clock'),
            dispatch=stamp(None,'UNKNOWN','not separately exported','audio dispatch instant'),
            original_selector=stamp(t['selector'],'RECORDED','native complete_track_audit','selector observation before emission'),
            cutoff=stamp(cutoff,'RECORDED','decision_input.timestamp / frozen context','original evaluation boundary'),
            timeout=stamp(t['peak']+cfg['event_timeout_s'] if 'event_timeout_s' in cfg else None,'DERIVED','recorded scanner configuration','nominal timeout, not a reserved decision budget'),
            rescue_cutoff=stamp(t['peak']+settings['full_rescue_timeout_s_v2225'] if (tr or {}).get('outcome',{}).get('rescue_used') and 'full_rescue_timeout_s_v2225' in settings else None,'DERIVED','rescue-used outcome + recorded settings','conditional extension; dispatch still unknown'),
            native_associations=t['native'],cached_only=t['cached_only'],batches=t['batches'],versions=t['versions'],
            frames=[dict(**f,read_status='RECORDED',camera_sequence=None,delivery=stamp(None,'UNKNOWN','not exported','grayscale pickup/publication'),exposure=stamp(None,'UNKNOWN','not exported','sensor exposure')) for f in spec['frames']] if spec else [],
            old_pre=spec['snapshot'] if spec else None,
            available_by_deadline_offsets={str(ms):sum(c['exact_xy_available_by']['value'] is not None and c['exact_xy_available_by']['value']<=cutoff-ms/1000 for c in ordered) for ms in (25,50,100,200)},
            coordinate_changes=sum(c['changed_coordinate'] for c in ordered),candidate_count=len(ordered))
        if spec:
            measured=by_event[eid];costs=dict(append_ms=measured['append_summary_ms']['median'],**measured['stages_ms'])
            releases={tuple(v['camera_xy']):v['available_by']['value'] for v in t['versions']}
            assert all(v is not None for v in releases.values())
            final_xy={tuple(p) for p in points};assert final_xy<=releases.keys()
            busy=[]
            # No parallel worker is assumed. Reserve startup through first known
            # result and recorded local-confirm costs immediately before its
            # list witness. Precise original CPU occupancy remains unknown.
            first=min(releases.values());busy.append((t['created'] or t['peak'],first))
            for b in t['batches']:
                end=b['available_by']['value']
                if b['confirmation_cpu_ms'] and end is not None:busy.append((end-b['confirmation_cpu_ms']/1000,end))
            sim=simulate(frame_times,releases,final_xy,cutoff,costs,busy=busy)
            baseline_ready=batch_baseline(frame_times,releases,final_xy,costs,busy=busy)[-1]['end']
            result=dict(event=eid,provenance='CONDITIONAL_SIMULATION_FRAME_DELIVERY_AND_CPU_OCCUPANCY_UNPROVEN',
                baseline_ready=baseline_ready,baseline_slack_ms=(cutoff-baseline_ready)*1000,
                baseline_meets=baseline_ready<=cutoff,prior_receipt_only_slack_ms=measured['slack_ms']-measured['stages_ms']['total_ms'],
                **{k:v for k,v in sim.items() if k!='schedule'},observed_coordinate_versions=len(releases),final_candidates=len(points),
                simulated_other_work_ms=sum((high-low)*1000 for low,high in busy),
                native_selector_slack_ms=(t['selector']-sim['selection_ready'])*1000 if t['selector'] else None)
            feasibility.append(result);row['simulation']=result
            dump(output/(eid.replace(':','_')+'_schedule.json'),dict(costs=costs,busy=busy,**sim))
        # GT joins only AFTER timing and scheduling have completed.
        records=[data['records'][i] for i in e['pools']['current']]
        if e['truth']['state']=='SINGLE_IMPACT':
            nearest=min(records,key=lambda r:r['gt_distance']);correct=lookup[(nearest['camera_x'],nearest['camera_y'])]
            winner=data['records'][e['current_id']];current=lookup[(winner['camera_x'],winner['camera_y'])]
            row['posthoc_correct']=dict(nearest_xy=correct['camera_xy'],nearest_error=nearest['gt_distance'],useful_at42=nearest['gt_distance']<=42,
                correct_available_by=correct['exact_xy_available_by'],current_available_by=current['exact_xy_available_by'],
                correct_first_observation=correct['first_observation'],current_first_observation=current['first_observation'])
        events.append(row)
        print(eid,'native',t['native'],'versions',len(t['versions']),'changed XY',row['coordinate_changes'],
              'sim slack',round(row.get('simulation',{}).get('slack_ms',0),2),flush=True)
    for e in data['unavailable']:
        events.append(dict(event=e['event'],status='PROVENANCE_INSUFFICIENT',reason=e['reason']))
    dump(output/'event_timeline.json',events);dump(output/'candidate_timeline.json',candidates)
    csv_rows(output/'event_timeline.csv',[{k:v for k,v in r.items() if k not in ('batches','versions','frames','simulation')} for r in events])
    csv_rows(output/'candidate_timeline.csv',[{**{k:v for k,v in c.items() if k not in ('observations','interval_ms')},**c['interval_ms']} for c in candidates])
    csv_rows(output/'deadline_feasibility.csv',feasibility);csv_rows(output/'scheduling_before_after.csv',feasibility)
    summary=dict(events=len(events),candidate_count=len(candidates),simulated=len(feasibility),
        before_success=sum(r['baseline_meets'] for r in feasibility),after_success=sum(r['meets_deadline'] for r in feasibility),
        before_slack=summarize([r['baseline_slack_ms'] for r in feasibility]),after_slack=summarize([r['slack_ms'] for r in feasibility]),
        total_cpu=summarize([r['total_cpu_ms'] for r in feasibility]),critical_cpu=summarize([r['post_data_critical_ms'] for r in feasibility]),
        all_status='PROVENANCE_INSUFFICIENT for certified live feasibility; conditional opportunity only')
    dump(output/'summary.json',summary);dump(output/'input_hashes.json',hashes);dump(output/'input_accesses.json',sorted(access))
    print(json.dumps(summary),flush=True)


def finish(output, audit):
    """Report attribution and ONE frozen semantic/result replay, no prototype."""
    from src.engine.offline.clean_physical_change import causal_local_evidence
    events=json.loads((audit/'event_timeline.json').read_text());candidates=json.loads((audit/'candidate_timeline.json').read_text())
    grouped=defaultdict(list)
    for c in candidates:grouped[c['event']].append(c)
    bench=json.loads((RUNTIME/'after/profiling_after.json').read_text());by_bench={r['event']:r for r in bench['events']}
    summaries={};feasibility=[];blockers=[];diagnoses=[];benchmarks=[];equivalence=[];replayed=[];selector_ms={}
    prior=json.loads((CAUSAL/'final/results.json').read_text());prior_events={r['event']:r for r in prior['events']}
    source_hashes={}
    # Snapshot numerical equivalence recheck only; cached image patches already
    # authenticated by the preceding runtime pass. No native trace discovery.
    for p in sorted((RUNTIME/'before').glob('*/event.json')):
        e=json.loads(p.read_text());patches=np.load(p.parent/'patches.npy',allow_pickle=False)
        snapshot=np.load(p.parent/'snapshot.npy',allow_pickle=False)
        expected=json.loads((p.parent/'reference.json').read_text())
        actual=causal_local_evidence(e['times'],patches,(e['snapshot'],snapshot),e['cutoff'],e['early_times'])
        if actual!=expected:raise AssertionError('Frozen evidence changed: '+e['event'])
        equivalence.append(dict(event=e['event'],candidates=len(actual),all_fields_exact=True,tolerance=0,
                                implementation='UNCHANGED causal_local_evidence; no new incremental evidence state'))
        for path in (p,p.parent/'patches.npy',p.parent/'snapshot.npy',p.parent/'reference.json'):
            source_hashes[str(path)]=reference.digest(path)
        if e['event'] not in prior_events:continue
        started=time.perf_counter()
        eligible=[j for j,r in enumerate(actual) if e['expected'][j]['ready'] and r['state']=='ONSET_FOUND' and r['causal']['support']>0]
        winner=max(eligible,key=lambda j:actual[j]['causal']['support']) if eligible else None
        selector_ms[e['event']]=(time.perf_counter()-started)*1000
        index=e['expected'][winner]['index'] if winner is not None else None
        frozen=prior_events[e['event']];assert index==frozen['causal']['index']
        replayed.append({k:frozen[k] for k in ('event','session','truth','current','causal','changes') if k in frozen})
    for e in events:
        eid=e['event'];cs=grouped[eid]
        if 'simulation' not in e:
            blockers.append(dict(event=eid,primary='PROVENANCE_INSUFFICIENT',conditional_primary='NOT_SIMULATABLE',
                reason='missing original D01 pixels/timing authentication or S01:11 terminal cutoff'))
            continue
        spec=json.loads((RUNTIME/'before'/eid.replace(':','_')/'event.json').read_text());run=json.loads((audit/(eid.replace(':','_')+'_schedule.json')).read_text())
        sim=dict(e['simulation']);cost=run['costs'];cutoff=e['cutoff']['value'];peak=e['peak']['value']
        releases={tuple(v['camera_xy']):v['available_by']['value'] for v in e['versions']};final_xy={tuple(p) for p in spec['points']}
        baseline=batch_baseline(spec['times'],releases,final_xy,cost,busy=run['busy'])
        sim['baseline_ready']=baseline[-1]['end']+selector_ms[eid]/1000
        sim['baseline_slack_ms']=(cutoff-sim['baseline_ready'])*1000;sim['baseline_meets']=sim['baseline_ready']<=cutoff
        sim['selection_ready']+=selector_ms[eid]/1000;sim['slack_ms']=(cutoff-sim['selection_ready'])*1000
        sim['meets_deadline']=sim['selection_ready']<=cutoff
        sim['post_data_critical_ms']+=selector_ms[eid];sim['total_cpu_ms']+=selector_ms[eid]
        sim['baseline_total_cpu_ms']=sum(j['cost_ms'] for j in baseline)+selector_ms[eid]
        if e['original_selector']['value'] is not None:sim['native_selector_slack_ms']=(e['original_selector']['value']-sim['selection_ready'])*1000
        latest_xy=max(releases[p] for p in final_xy);latest_post=spec['times'][-1]
        # These are scenario attributions, never identified historical causes.
        category=('FEASIBLE' if sim['meets_deadline'] else 'CANDIDATE_LATE' if latest_xy>latest_post
                  else 'PROCESSING_LATE')
        blockers.append(dict(event=eid,primary='PROVENANCE_INSUFFICIENT',conditional_primary=category,
            reason='exact frame pickup/producer completion/CPU occupancy not exported; availability-by bounds are not first delivery',
            last_xy_known_by_ms=(latest_xy-peak)*1000,last_post_receipt_ms=(latest_post-peak)*1000,
            frame_delivery_after_deadline='UNKNOWN',inherently_infeasible='UNKNOWN'))
        sim['conditional_blocker']=category;sim['certified_live_feasible']=None;feasibility.append(sim)
        e['simulation']=sim
        available=[c['exact_xy_available_by']['value'] for c in cs]
        e['first_exact_xy_known_by']=min(available);e['last_exact_xy_known_by']=max(available)
        e['candidate_arrival_steps']=[dict(available_by=ts,status='DERIVED_FROM_RECORDED_UPPER_BOUNDS',count=sum(v<=ts for v in available)) for ts in sorted(set(available))]
        e['all_candidates_known_by_counts']=e['available_by_deadline_offsets']
        useful=e.get('posthoc_correct')
        if useful:
            nearest=next(c for c in cs if c['camera_xy']==useful['nearest_xy']);delivery=nearest['exact_xy_available_by']['value']
            wrong=useful['current_available_by']['value']
            single_cost=cost['total_ms']/len(cs)
            ready=max(delivery,latest_post)+single_cost/1000
            diagnoses.append(dict(event=eid,useful_at42=useful['useful_at42'],nearest_error=useful['nearest_error'],
                exact_xy_known_by_ms=(delivery-peak)*1000,known_by_to_deadline_ms=(cutoff-delivery)*1000,
                current_known_earlier=wrong<delivery,current_observed_earlier=useful['current_first_observation']['value']<useful['correct_first_observation']['value'],
                single_candidate_amortized_cost_ms=single_cost,single_candidate_conditional_fits=ready<=cutoff,
                candidate_alone_late=delivery+single_cost/1000>cutoff,
                post_alone_late=latest_post+single_cost/1000>cutoff,
                attribution='ESTIMATED amortized batch share, not measured standalone latency',primary='PROVENANCE_INSUFFICIENT'))
        # Descriptive first/last useful timing is joined AFTER scheduling.
        cache=json.loads(reference.CACHE.read_text()) if 'cache' not in locals() else cache
        ce=next(x for x in cache['events'] if x['event']==eid)
        useful_xy={(cache['records'][i]['camera_x'],cache['records'][i]['camera_y']) for i in ce['pools']['current']
                   if cache['records'][i].get('gt_distance') is not None and cache['records'][i]['gt_distance']<=42}
        useful_times=[c['exact_xy_available_by']['value'] for c in cs if tuple(c['camera_xy']) in useful_xy]
        e['posthoc_useful_arrival']=dict(count=len(useful_times),first_known_by=min(useful_times) if useful_times else None,
                                        last_known_by=max(useful_times) if useful_times else None)
        patch_share=cost['patch_ms']/(len(cs)*len(spec['times']))
        steps=run['schedule'];post_peak_cpu=sum(j['cost_ms'] for j in steps if j['start']>=peak)
        peak_slice=max((sum(j['cost_ms'] for j in steps if st<=j['start']<st+.08) for st in [j['start'] for j in steps]),default=0.)
        benchmarks.append(dict(event=eid,continuous_per_frame_ms=cost['append_ms'],candidate_creation_backfill_upper_ms=patch_share*len(spec['times']),
            candidate_update_per_frame_ms=patch_share,creation_and_update_status='ESTIMATED_FROM_PRIOR_BATCH_COST; no prototype measurement',
            full_finalization_ms=cost['inference_ms'],selector_measured_once_ms=selector_ms[eid],setup_ms=cost['setup_ms'],
            total_cpu_ms=sim['total_cpu_ms'],post_event_cpu_ms=post_peak_cpu,peak_80ms_start_bucket_cpu_ms=peak_slice,
            critical_path_ms=sim['post_data_critical_ms'],superseded_versions=len(e['versions'])-len(cs)))
    metrics={m:{key:reference.summary([r for r in replayed if m=='POOLED' or r['session']==m],key) for key in ('current','causal')}
             for m in ('S01','S02','POST_FIX','POOLED')}
    assert metrics==prior['summaries']
    summary=dict(events=len(events),candidates=len(candidates),simulated=len(feasibility),
        certified_feasible=0,certified_infeasible=0,certified_unknown=len(events),
        before_success=sum(r['baseline_meets'] for r in feasibility),after_success=sum(r['meets_deadline'] for r in feasibility),
        before_slack=summarize([r['baseline_slack_ms'] for r in feasibility]),after_slack=summarize([r['slack_ms'] for r in feasibility]),
        conditional_blockers=dict(Counter(r['conditional_primary'] for r in blockers)),
        primary_blockers=dict(Counter(r['primary'] for r in blockers)),
        measured_selector_ms=summarize(list(selector_ms.values())),
        cpu_before=summarize([r['baseline_total_cpu_ms'] for r in feasibility]),cpu_after=summarize([r['total_cpu_ms'] for r in feasibility]),
        critical_after=summarize([r['post_data_critical_ms'] for r in feasibility]),
        posthoc_nearest=dict(count=len(diagnoses),useful_at42=sum(d['useful_at42'] for d in diagnoses),
            conditional_single_fits=sum(d['single_candidate_conditional_fits'] for d in diagnoses),
            current_known_earlier=sum(d['current_known_earlier'] for d in diagnoses),current_observed_earlier=sum(d['current_observed_earlier'] for d in diagnoses),
            conditional_candidate_alone_late=sum(d['candidate_alone_late'] for d in diagnoses),conditional_post_alone_late=sum(d['post_alone_late'] for d in diagnoses)),
        prototype='NOT IMPLEMENTED: one event gain, one session only, exact delivery/CPU provenance absent; no demonstrated robust timing opportunity',
        inference_semantics_changed=False)
    for name,rows in [('event_timeline',events),('candidate_timeline',candidates),('deadline_feasibility',feasibility),
        ('scheduling_before_after',feasibility),('blocker_classification',blockers),('correct_candidate_timing',diagnoses),('benchmark_results',benchmarks)]:
        dump(output/(name+'.json'),rows)
        csv_rows(output/(name+'.csv'),[{k:v for k,v in r.items() if k not in ('frames','observations','versions','batches','simulation')} for r in rows])
    dump(output/'semantic_equivalence.json',dict(scope='Frozen implementation recheck, no new prototype',events=equivalence,
        exact_events=len(equivalence),exact_candidates=sum(e['candidates'] for e in equivalence),mismatches=0,tolerance=0))
    dump(output/'frozen_selector_replay.json',dict(events=replayed,summaries=metrics))
    dump(output/'summary.json',summary);dump(output/'frozen_replay_input_hashes.json',source_hashes)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--finish-audit',type=Path)
    args=p.parse_args()
    if args.finish_audit:finish(args.output.resolve(),args.finish_audit.resolve())
    else:run(args.output.resolve())
