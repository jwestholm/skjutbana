"""Read-only OBJECT_LOCAL_CHANGE / mature-pool reconciliation, no detector replay.

Uses saved native ledgers, explicitly unavailable deliveries, and frozen historical
stage caches. No imagery, inference GT, current calibration or live authority.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from automation.accuracy_physical_dataset import guard_path
from automation.object_hit_replay import mapping
from automation.object_physical_replay import safe_file, mature_candidate_bridge
from src.engine.input.object_hit_v2223 import ObjectShotSnapshotV2223, GameHitRegionV240, CameraHitRegionV240
from src.engine.offline.track_replay import current_exact_replay, require
from src.engine.physical_trace import _diagnostic_source

TOLERANCES=(5,10,20,42)


def digest(path):
    guard_path(path)
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path,hashes):
    guard_path(path);hashes[str(path)]=digest(path)
    return json.loads(Path(path).read_text())


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def snapshot(trace):
    ctx=trace['context']['shot_game_context']
    def region(cls,r):return cls(**{f.name:r[f.name] for f in fields(cls) if f.name in r})
    return ObjectShotSnapshotV2223(trace['shot_id'],trace['peak_ts'],trace['peak_ts'],(),
        game_regions=tuple(region(GameHitRegionV240,r) for r in ctx['game_hit_regions']),
        camera_regions=tuple(region(CameraHitRegionV240,r) for r in ctx['camera_hit_regions']),
        viewport_xywh=ctx['viewport_xywh'],transform_available=ctx['transform_available'],
        transform_method=ctx['transform_method'])


def nearest(point,points):
    return min((math.dist(point,p) for p in points),default=None)


def overlap(a,b):
    """Asymmetric counts; <=5 is overlap diagnostic, never identity/confirmation."""
    return dict(a_count=len(a),b_count=len(b),exact_shared=len(set(map(tuple,a))&set(map(tuple,b))),
        a_without_b_within5=sum(nearest(p,b) is None or nearest(p,b)>5 for p in a),
        b_without_a_within5=sum(nearest(p,a) is None or nearest(p,a)>5 for p in b))


def metrics(errors):
    finite=[e for e in errors if e is not None]
    return dict(events=len(errors),at={str(t):sum(e is not None and e<=t for e in errors) for t in TOLERANCES},
        unknown=sum(e is None for e in errors),mean=float(np.mean(finite)) if finite else None,
        median=float(np.median(finite)) if finite else None,p95=float(np.percentile(finite,95)) if finite else None,
        max=max(finite,default=None),over100=sum(e>100 for e in finite))


def embedded_json(line,key):
    """Read an intact named value even if interleaved later console text breaks the envelope."""
    token='"'+key+'":'
    if token not in line:return None
    try:return json.JSONDecoder().raw_decode(line.split(token,1)[1].lstrip())[0]
    except ValueError:return None


def gameplay(root,hashes):
    root=Path(root);guard_path(root)
    manifest=read(safe_file(root,'play.json'),hashes)
    if manifest.get('mode')!='OBJECT_HIT_GAMEPLAY_ACCEPTANCE':raise ValueError('play-can capture required')
    truth_path=safe_file(root,'operator_ground_truth.json')
    truth=read(truth_path,hashes) if truth_path.exists() else {'shots':[]}
    log=safe_file(root,'runtime.log');hashes[str(log)]=digest(log);lines=log.read_text().splitlines()
    ids=sorted({int(m.group(1)) for l in lines if '[OBJECT-HIT] ' in l
                for m in [re.search(r'"shot_id":\s*(\d+)',l)] if m})
    labels={int(r['shot_id']):r for r in truth['shots']}
    traces={sid:read(safe_file(root,f'object_crops/shots/shot_{sid:08d}/trace.json'),hashes) for sid in ids}
    rows=[];previous=[]
    for sid in ids:
        tp=safe_file(root,f'object_crops/shots/shot_{sid:08d}/trace.json');trace=traces[sid]
        next_peak=min((t['peak_ts'] for t in traces.values() if t['peak_ts']>trace['peak_ts']),default=None)
        q=trace['object_query_diagnostic'];cutoff=q['decision_ts'];snap=snapshot(trace)
        convert,map_meta=mapping(trace)
        label=labels.get(sid,{});gt=label.get('object_id')
        state=label.get('physical_truth',label.get('truth','UNKNOWN'))
        # Labels remain audit-only. score_label/intent never enters either bridge.
        local=q['per_object'];ledger=q.get('candidate_ledger') or []
        local_xy=[c['camera_xy'] for o in local for c in o.get('candidates',[])]
        all_xy=[c['camera_xy'] for f in ledger for o in f['objects'] for c in o['candidates']]
        supported=q.get('object_decision',{}).get('supported_object_ids',[])
        candidates_inside=[o['object_id'] for o in local if any(
            a['object_id']==o['object_id'] and a.get('inside_exact_shape') is True
            for c in o.get('candidates',[]) for a in c.get('regions',[]))]
        cv=[];fresh=[];exact=None;parse_warnings=[]
        for line in lines:
            if not re.search(r'\bshot='+str(sid)+r'\b',line):continue
            if line.startswith('[V2.22.4 CV]'):
                v=dict(re.findall(r'(age|queue|worker|v2|cand)=(-?[\d.]+)',line))
                costs={k:float(v[k]) for k in v}
                costs.update(registered_stage_logged=' reg=' in line,
                    earliest_delivery_lower_bound_ms=costs.get('age',0)+costs.get('worker',0),
                    timing_semantics='rounded frame age + worker CPU lower bound; actual pickup unavailable')
                cv.append(costs)
            if line.startswith('[V2.25.2 FRESHNESS]'):
                m=re.search(r'object=(\S+) n=(\d+) fresh=(\d+).*?xy=\(([^,]+),([^\)]+)\)',line)
                if m:fresh.append(dict(group=m[1],count=int(m[2]),fresh_count=int(m[3]),
                    rounded_working_xy=[float(m[4]),float(m[5])],
                    semantics='expanded_group_best_only; rounded_working_XY; delivery_not_proven'))
        for line in lines:
            if '[OBJECT-HIT-CURRENT] ' not in line:continue
            if embedded_json(line,'shot_id')==sid:
                exact=embedded_json(line,'exact_coordinate_result')
                try:json.loads(line.split('[OBJECT-HIT-CURRENT] ',1)[1])
                except ValueError:parse_warnings.append('interleaved_console_envelope; intact_exact_result_only')
        exact_objects=[]
        if exact and exact.get('camera_xy'):
            # Mapping only. No flag is manufactured to certify this emitted XY.
            native=dict(camera_x=exact['camera_xy'][0],camera_y=exact['camera_xy'][1],timestamp=exact.get('timestamp'))
            probe=mature_candidate_bridge(snap,convert,dict(candidates=[native]),cutoff)
            exact_objects=probe['rows'][0]['inside_object_ids']
        packets=[]
        for name in [f'pool_{i:04d}.json' for i in range(32)]+['LIMIT.json']:
            path=safe_file(tp.parent,'mature_candidates/'+name)
            if path.exists():packets.append(read(path,hashes))
        capture_complete=trace.get('completeness',{}).get('trace_complete') is True
        comparisons=[dict(packet=p,shadow=mature_candidate_bridge(snap,convert,
            p if capture_complete else dict(p,complete=False),cutoff,
            next_peak=next_peak)) for p in packets]
        eligible=[r for r in comparisons if r['packet'].get('available_at',math.inf)<=cutoff
                  and r['packet'].get('shot_id')==sid]
        latest=max(eligible,key=lambda r:r['packet']['available_at'],default=None)
        mature_xy=[r['camera_xy'] for r in latest['shadow']['rows']
                   if r['availability']=='CAUSALLY_AVAILABLE'] if latest else None
        row=dict(session=root.name,shot_id=sid,truth_state=state,gt_object=gt,
            truth_source=str(truth_path) if truth_path.exists() else None,exact_hole_gt='UNAVAILABLE',
            recorded_object_decision=q.get('object_decision'),local_supported=supported,
            local_gt_available=(gt in candidates_inside) if gt else None,
            local_gt_supported=(gt in supported) if gt else None,
            local_false_supported_objects=([o for o in supported if o!=gt] if gt else
                supported if state=='NO_PHYSICAL_SHOT' else None),
            local_inside_objects=candidates_inside,local_retained_candidate_count=len(local_xy),
            local_examined=q['examined_candidate_count'],local_generated=q['candidate_count'],
            local_recurring_within5=sum(nearest(p,previous) is not None and nearest(p,previous)<=5 for p in local_xy),
            local_post_frames=q['post_frames'],object_processing_ms=sum(q['timing']['processing_ms']),
            object_prepare_ms=q['timing']['prepare_ms'],
            object_cpu_ms=q['timing']['prepare_ms']+sum(q['timing']['processing_ms']),
            object_latency_ms=(cutoff-trace['peak_ts'])*1000,cutoff=cutoff,
            image_replay='UNAVAILABLE_PREPARE_TERMINAL' if not trace.get('object_evidence_crops') else 'SAVED_CROPS',
            mature_cv_log=cv,mature_fresh_group_log=fresh,
            mature_fresh_groups=sorted({f['group'] for f in fresh if f['fresh_count']}),
            mature_pool_status='CAPTURED' if packets else 'UNAVAILABLE_NOT_RECORDED',
            mature_capture_complete=capture_complete if packets else None,
            mature_at_cutoff=latest['shadow'] if latest else None,mature_packets=comparisons,
            source_overlap=overlap(local_xy,mature_xy) if mature_xy is not None else None,
            local_vs_late_current_overlap=overlap(all_xy,[exact['camera_xy']]) if exact and exact.get('camera_xy') else None,
            current_exact_result=exact,current_xy_inside_objects=exact_objects,
            current_object_correct=((gt in exact_objects) if exact.get('emitted') else False) if gt and exact else None,
            current_is_object_authority=False,current_parse_warnings=parse_warnings,mapping=map_meta)
        rows.append(row);previous.extend(local_xy)
    return dict(capture=str(root),source_manifest=manifest,rows=rows,
        boundaries=['No exact camera GT','Mature group log is not an exact-shape pool',
                    'CURRENT emission arrives later and may include reaction pixels',
                    'Absence of candidate sidecars is unavailable, never zero candidates'])


def historical(base,hashes):
    """Re-score immutable September10 caches; never open overwritten D01 pixels."""
    base=Path(base);guard_path(base)
    protocol=read(safe_file(base,'models/protocol.json'),hashes)
    for name,expected in protocol['cache_hashes'].items():
        path=safe_file(base,'cache/'+name);actual=digest(path);hashes[str(path)]=actual
        if actual!=expected:raise ValueError('Frozen cache hash mismatch: '+name)
    data=read(safe_file(base,'cache/dataset.json'),hashes)
    allowed={'D01','S01','S02','POST_FIX'}
    if set(data['sessions'])!=allowed or any(e['session'] not in allowed for e in data['events']):
        raise ValueError('Unexpected historical cohort')
    rows=[];source_rows=[]
    for e in data['events']:
        details=read(safe_file(base,'cache/'+e['event'].replace(':','_')+'_trace_details.json'),hashes)
        chosen=current_exact_replay(details['snapshot'])
        index=e['current_id'];record=data['records'][index] if index is not None else None
        require(bool(chosen)==bool(record), 'Frozen selection presence changed: '+e['event'])
        if chosen and record:
            require(all(chosen[k]==record[k] for k in ('camera_x','camera_y')), 'Frozen selected XY changed: '+e['event'])
        gt=e['truth']['impacts'][0] if e['truth']['impacts'] else None
        def error(c):return math.hypot(c['camera_x']-gt['camera_x'],c['camera_y']-gt['camera_y']) if c and gt else None
        emission_proof=details['snapshot'].get('selected_passed_readiness_for_emission')
        emitted=bool(chosen and emission_proof) if emission_proof is not None else None
        snap=details['snapshot'];peak=e['context']['peak']
        native_clock='reconstruction' not in snap
        batches=[a for a in snap['associations'] if a.get('dispatch_owner')==snap['event']['shot_id']
                 and a.get('observed_at') is not None] if native_clock else []
        delivered=[a for a in batches if a['observed_at']<=peak+.350 and a['frame_ts']<=peak+.350]
        fast_cs=[r['candidate'] for a in delivered for r in a['records']
                 if r.get('candidate',{}).get('v2224_producer_shot_id')==snap['event']['shot_id']]
        positions=details['stage_positions'];pools={}
        for name,indices in e['pools'].items():
            pool=[data['records'][i] for i in indices]
            pools[name]=dict(count=len(pool),nearest_px=min((error(c) for c in pool),default=None) if gt else None)
        for name,pool in positions.items():
            grouped=defaultdict(list)
            for c in pool:grouped[_diagnostic_source(c)].append(c)
            for source,cs in grouped.items():
                source_rows.append(dict(event=e['event'],session=e['session'],stage=name,source=source,
                    count=len(cs),nearest_px=min((error(c) for c in cs),default=None) if gt else None))
        rows.append(dict(event=e['event'],session=e['session'],truth=e['truth']['state'],pools=pools,
            available_stages=sorted(positions),
            selected_xy=[chosen['camera_x'],chosen['camera_y']] if chosen else None,
            selected_error=error(chosen),emitted=emitted,
            snapshot_semantics=details['snapshot']['pool_semantics'],current_exact_replay='PASS',
            fast_delivery=dict(status='NATIVE_CONSUMPTION_CLOCK' if native_clock else 'UNAVAILABLE_RECONSTRUCTED_CLOCK',
                first_consumption_ms=min(((a['observed_at']-peak)*1000 for a in batches),default=None),
                candidate_versions_by_350ms=len(fast_cs) if native_clock else None,
                nearest_by_350ms=min((error(c) for c in fast_cs),default=None) if gt else None),
            ready_ranks=[dict(track_id=t['track_id'],rank=t['final_rank'],error=error(t))
                         for t in details['snapshot']['tracks'] if t['eligible']] if gt else [],
            object_geometry='UNAVAILABLE_NOT_MANUFACTURED',
            object_deadline_availability=('NATIVE_CONSUMPTION_CLOCK; no frozen object geometry' if native_clock else
                'UNAVAILABLE; reconstructed clock is not worker delivery proof')))
    summaries={}
    for session in sorted(allowed):
        all_rows=[r for r in rows if r['session']==session]
        impacts=[r for r in all_rows if r['truth']=='SINGLE_IMPACT']
        summaries[session]=dict(selected=metrics([r['selected_error'] for r in impacts]),
            oracle={p:dict(metrics([r['pools'][p]['nearest_px'] for r in impacts if p in r['pools']]),
                           unavailable_events=sum(p not in r['pools'] for r in impacts))
                    for p in sorted({p for r in impacts for p in r['pools']})},
            controls=sum(r['truth']=='NO_PHYSICAL_SHOT' for r in all_rows),
            false_control_emissions=sum(r['truth']=='NO_PHYSICAL_SHOT' and r['emitted'] is True for r in all_rows),
            emission_proof_unavailable=sum(r['emitted'] is None for r in all_rows),
            exact_replay_verified=len(all_rows),selection_changes=0)
    source_summaries=[]
    for stage in sorted({r['stage'] for r in source_rows}):
        # Only events with a saved stage participate; missing stages are not failures.
        stage_events={r['event'] for r in rows if stage in r['available_stages']}
        for session in sorted(allowed):
            impacts=[r for r in rows if r['session']==session and r['truth']=='SINGLE_IMPACT']
            available=[r for r in impacts if r['event'] in stage_events]
            for source in sorted({r['source'] for r in source_rows if r['stage']==stage and r['session']==session}):
                group={r['event']:r for r in source_rows if r['stage']==stage and r['session']==session and r['source']==source}
                source_summaries.append(dict(session=session,stage=stage,source=source,
                    oracle=metrics([group[r['event']]['nearest_px'] if r['event'] in group else None for r in available]),
                    unavailable_stage_events=len(impacts)-len(available),
                    candidate_count=sum(r['count'] for r in group.values())))
    return dict(base=str(base),rows=rows,source_rows=source_rows,source_summaries=source_summaries,summaries=summaries,
        semantics='Frozen stage/original CURRENT comparison. Not modern OBJECT pool or live350ms replay.',
        source_bucket_semantics='Existing trace diagnostic precedence; overlapping flags retained in source cache; scores not compared')


def run(captures,output,history=None):
    for p in [*captures,output]+([history] if history else []):guard_path(p)
    out=Path(output).resolve()
    if not out.is_relative_to(Path('/data/skjutbana')):raise ValueError('output must be under /data/skjutbana')
    if any(out==Path(p).resolve() or out.is_relative_to(Path(p).resolve()) for p in [*captures]+([history] if history else [])):
        raise ValueError('output must not modify capture/cache')
    out.mkdir(parents=True,exist_ok=False);hashes={};start=time.perf_counter()
    results=[gameplay(p,hashes) for p in captures]
    for r in results:write(out/(Path(r['capture']).name+'.json'),r)
    hist=historical(history,hashes) if history else None
    if hist:write(out/'historical.json',hist)
    sources=[Path(__file__),Path('automation/object_physical_replay.py'),Path('automation/object_hit_replay.py'),
        Path('src/engine/camera/object_physical_query.py'),Path('src/engine/physical_trace.py'),
        Path('src/engine/shot_async_v2224.py'),Path('src/engine/shot_fast_v2225.py'),
        Path('src/engine/offline/causal_candidates.py'),Path('src/engine/offline/track_replay.py'),
        Path('src/engine/input/object_hit_v2223.py'),Path('src/engine/input/hit_regions.py')]
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('input mutated during comparison')
    write(out/'provenance.json',dict(input_hashes=hashes,source_hashes={str(p):digest(p) for p in sources},
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope='RESEARCH_ONLY_NO_AUTHORITY',elapsed_seconds=time.perf_counter()-start))
    lines=['# Native candidate reconciliation','',
        'Object truth only for gameplay. Missing native pools stay UNAVAILABLE; group logs are not exact-shape proof.','',
        '|Capture:event|GT object|Local support count|GT supported|Mature fresh groups (expanded)|Native pool|',
        '|---|---|---|---|---|---|']
    for result in results:
        for r in result['rows']:
            lines.append(f"|{r['session']}:{r['shot_id']}|{r['gt_object'] or r['truth_state']}|{len(r['local_supported'])}|{r['local_gt_supported']}|{len(r['mature_fresh_groups'])}|{r['mature_pool_status']}|")
    if hist:
        lines+=['','Historical immutable cache, selected counts separate from oracle:',
            '|Session|Selected @5/10/20/42|Current-pool oracle|Retained oracle|False controls|',
            '|---|---|---|---|---|']
        for name,s in hist['summaries'].items():
            fmt=lambda m:'/'.join(str(m['at'][str(t)]) for t in TOLERANCES)
            lines.append(f"|{name}|{fmt(s['selected'])}|{fmt(s['oracle']['current'])}|{fmt(s['oracle']['retained'])}|{s['false_control_emissions']}/{s['controls']}|")
    lines+=['','Per-event JSON contains timings, exact native-source boundaries, current outcome, overlap availability and hashes.',
        'No scalar ranking, new physical thresholds, candidate generation, model fitting or live promotion.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(out)


def _csv(path, rows):
    import csv
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def paired_event_ids(root, lines):
    """Explicit log IDs plus named shot directories; no ordinal-to-intent binding."""
    guard_path(root)
    ids = set()
    for line in lines:
        if any(tag in line for tag in ('[V2.22.3 SHOT]', '[OBJECT-MATURE-CAPTURE]', '[V2.22.4 CV]')):
            match = re.search(r'\bshot=(\d+)\b', line)
            if match: ids.add(int(match[1]))
        if '[OBJECT-HIT' in line:
            sid = embedded_json(line, 'shot_id')
            if type(sid) is int and sid > 0: ids.add(sid)
    directory = safe_file(root, 'object_crops/shots')
    if directory.exists():
        for child in directory.iterdir():
            if re.fullmatch(r'shot_\d{8}', child.name):
                guard_path(child)
                ids.add(int(child.name[5:]))
    return sorted(ids)


def operator_truth_rows(payload, ids, capture_hash):
    """Existing operator_ground_truth shots schema. Intent is never an observation."""
    if payload.get('capture_play_sha256') not in (None, capture_hash):
        raise ValueError('operator truth belongs to another capture')
    result = {}
    for row in payload.get('shots', []):
        sid = row.get('shot_id')
        if type(sid) is not int or sid not in ids or sid in result:
            raise ValueError('operator truth has unknown/duplicate shot_id')
        state = row.get('physical_truth', row.get('truth', 'UNKNOWN'))
        if state not in ('OBJECT_HIT', 'TARGET_MISS', 'NO_PHYSICAL_SHOT', 'UNKNOWN'):
            raise ValueError('unsupported operator physical truth')
        oid = row.get('object_id')
        if (state == 'OBJECT_HIT' and not oid) or (state != 'OBJECT_HIT' and oid is not None):
            raise ValueError('actual object must be explicit and only accompanies OBJECT_HIT')
        discharge = row.get('physical_discharge', 'unknown')
        if discharge not in ('yes', 'no', 'unknown') or (
            state == 'NO_PHYSICAL_SHOT' and discharge == 'yes') or (
            state in ('OBJECT_HIT', 'TARGET_MISS') and discharge == 'no'):
            raise ValueError('contradictory physical discharge label')
        result[sid] = dict(row, physical_truth=state, object_id=oid, physical_discharge=discharge)
    return {sid: result.get(sid, dict(shot_id=sid, physical_truth='UNKNOWN', object_id=None,
                                   physical_discharge='unknown')) for sid in ids}


def saved_pool_oracle(points, gt, *, available=True):
    """Saved boundary oracle only. Native order is NOT a cross-source score rank."""
    if not available:
        return dict(status='UNAVAILABLE_POOL', nearest_distance_px=None, nearest_candidate_order=None)
    if gt is None:
        return dict(status='UNAVAILABLE_GT', nearest_distance_px=None, nearest_candidate_order=None)
    distances=[math.dist(p,gt) for p in points]
    index=min(range(len(points)),key=lambda i:distances[i]) if points else None
    d=distances[index] if index is not None else None
    return dict(status='AVAILABLE',candidate_count=len(points),nearest_distance_px=d,
        nearest_candidate_order=index+1 if index is not None else None,
        nearest_camera_xy=list(points[index]) if index is not None else None,
        order_semantics='saved pool order, not score rank',
        **{f'oracle_at{t}':int(d is not None and d<=t) for t in TOLERANCES})


def exact_label_analysis(root, out, label_path, ids, traces, stage_rows, candidates, shots, hashes):
    """Optional independent manual labels; source hashes and transform fail closed."""
    from automation.physical_trace_label_crops import load_capture, load_gt_review, REVIEW_LABEL_SCHEMA, validate_document, assert_inputs
    supplied=read(label_path,hashes)
    manifest=load_gt_review(root) if supplied.get('schema')==REVIEW_LABEL_SCHEMA else load_capture(root)
    hashes.update(manifest['input_hashes'])
    doc=validate_document(supplied,manifest)
    assert_inputs(manifest)
    labels={r['shot_id']:r for r in doc['shots']}
    valid={sid:r['camera_xy'] for sid,r in labels.items() if r['physical_truth_status']=='EXACT_CAMERA_XY'}
    status=dict(source=str(label_path),schema=doc['schema'],revision=doc['revision'],
                exact_xy='AVAILABLE' if valid else 'UNAVAILABLE',valid_shot_ids=sorted(valid),
                unknown_shot_ids=[sid for sid in ids if sid not in valid])
    for shot in shots:
        shot['exact_camera_gt']=valid.get(shot['shot_id'],'UNAVAILABLE_NOT_INFERRED')
    if not valid:return status
    rows=[];selected=[]
    for sid in ids:
        gt=valid.get(sid);t=traces.get(sid);q=(t or {}).get('object_query_diagnostic') or {}
        complete=bool(t and t.get('completeness',{}).get('trace_complete') and 'per_object' in q)
        retained=[];supported=[]
        for ob in q.get('per_object',[]):
            for c in ob.get('candidates',[]):
                xy=tuple(c['camera_xy'])
                if xy not in retained:retained.append(xy)
                if (ob.get('decision')=='supported_exact_shape' and c.get('confirmed') is True
                    and c.get('event_freshness',{}).get('state')=='SUPPORTED'
                    and any(a.get('object_id')==ob['object_id'] and a.get('inside_exact_shape') is True for a in c.get('regions',[]))
                    and xy not in supported):supported.append(xy)
        for stage,points in [('OBJECT_LOCAL_RETAINED',retained),('OBJECT_LOCAL_SUPPORTED_RETAINED',supported),('OBJECT_LOCAL_RAW',[])]:
            rows.append(dict(shot_id=sid,stage=stage,packet=None,ground_truth=gt,
                **saved_pool_oracle(points,gt,available=complete and stage!='OBJECT_LOCAL_RAW')))
        for r in stage_rows:
            if r['shot_id']!=sid:continue
            points=[c['camera_xy'] for c in candidates if c['shot_id']==sid and c['packet']==r.get('packet')]
            rows.append(dict(shot_id=sid,stage=r['stage'],packet=r.get('packet'),ground_truth=gt,
                delivery_ms=r.get('delivery_ms'),full_rescue=r.get('full_rescue'),
                delivered_before_object_decision=r.get('delivered_before_object_decision'),
                registered_worker=r.get('registered_worker'),follows_registered_worker=r.get('follows_registered_worker'),
                **saved_pool_oracle(points,gt,available=r['availability']=='COMPLETE_BOUNDARY_OUTPUT')))
        current=next(s for s in shots if s['shot_id']==sid).get('current_exact_result')
        xy=(current or {}).get('camera_xy')
        # An explicit emitted result only, never a winner/candidate treated as emission.
        emitted=(current or {}).get('emitted') is True and xy is not None
        error=math.dist(xy,gt) if emitted and gt is not None else None
        selected.append(dict(shot_id=sid,status='AVAILABLE' if gt is not None and current else 'UNAVAILABLE',
            emitted=emitted if current else None,camera_xy=xy if emitted else None,error_px=error,
            **{f'selected_at{t}':int(error is not None and error<=t) if gt is not None and current else None for t in TOLERANCES}))
    _csv(out/'candidate_nearest_gt.csv',rows)
    # Summaries use ONE named boundary per event. Never take retrospective best
    # across packets or inflate coverage by merging changing candidate versions.
    summary=[]
    for stage in sorted({r['stage'] for r in rows}):
        choices=('first_saved','last_saved') if stage in ('WORKER_DELIVERY','LOCAL_CONFIRMATION') else ('final_retained',)
        for boundary in choices:
            chosen=[]
            for sid in valid:
                rs=[r for r in rows if r['shot_id']==sid and r['stage']==stage]
                if rs:
                    r=rs[-1] if boundary=='last_saved' else rs[0]
                    if r['status']=='AVAILABLE':chosen.append(r)
            summary.append(dict(stage=stage,boundary=boundary,labelled_available_events=len(chosen),
                labelled_unavailable_events=len(valid)-len(chosen),
                **{f'oracle_at{t}':sum(r[f'oracle_at{t}'] for r in chosen) for t in TOLERANCES}))
    write(out/'oracle_summary.json',dict(label_status=status,scope='saved boundaries only, no missing raw-pool reconstruction',
        oracle=summary,selected_current_separate=selected,
        object_truth='NOT_INFERRED_FROM_XY',support_scope='retained observed support, not OBJECT_HIT emission'))
    return status


def paired_analysis(capture, output, operator_truth=None, exact_camera_gt=None):
    """Read saved pools at each handoff. Does not rerun or change any detector.

    Counts are candidate VERSIONS at named boundaries, not pooled truth. No
    cross-source ranking, reconstruction of missing stages, or exact-GT inference.
    """
    for path in (capture, output, *([operator_truth] if operator_truth else []), *([exact_camera_gt] if exact_camera_gt else [])):
        guard_path(path)
        if 's03' in str(path).lower(): raise PermissionError('validation source forbidden')
    root, out = Path(capture).resolve(), Path(output).resolve()
    if not out.is_relative_to(Path('/data/skjutbana')) or out == root or out.is_relative_to(root):
        raise ValueError('new output outside original capture and under /data/skjutbana required')
    for truth_input in (operator_truth,exact_camera_gt):
        if truth_input and (Path(truth_input).resolve() == out or out in Path(truth_input).resolve().parents):
            raise ValueError('output may not contain source truth')
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}; start = time.perf_counter()
    manifest = read(safe_file(root, 'play.json'), hashes)
    if manifest.get('mode') != 'OBJECT_HIT_GAMEPLAY_ACCEPTANCE': raise ValueError('play-can required')
    log_path = safe_file(root, 'runtime.log'); hashes[str(log_path)] = digest(log_path)
    lines = log_path.read_text().splitlines(); ids = paired_event_ids(root, lines)
    exit_path = safe_file(root, 'runtime_exit.json')
    exit_record = read(exit_path, hashes) if exit_path.exists() else None
    notes_path = safe_file(root, 'event_notes.json')
    if notes_path.exists(): read(notes_path, hashes)  # Preserve identity, never infer labels/order.
    truth_path = Path(operator_truth) if operator_truth else safe_file(root, 'operator_ground_truth.json')
    truth = operator_truth_rows(read(truth_path, hashes) if truth_path.exists() else {}, ids,
                                hashes[str(root/'play.json')])
    traces = {}
    for sid in ids:
        path = safe_file(root, f'object_crops/shots/shot_{sid:08d}/trace.json')
        if path.exists():
            t = read(path, hashes)
            if t['shot_id'] != sid: raise ValueError('trace/shot identity mismatch')
            traces[sid] = t
    stage_rows, candidate_rows, shots, timing_rows, object_rows = [], [], [], [], []
    previous_worker = []
    for sid in ids:
        t = traces.get(sid); label = truth[sid]
        current = next((embedded_json(l, 'exact_coordinate_result') for l in reversed(lines)
                        if '[OBJECT-HIT-CURRENT]' in l and embedded_json(l, 'shot_id') == sid), None)
        native_logs = [l for l in lines if re.search(r'\bshot='+str(sid)+r'\b', l)]
        export_logs = [l for l in native_logs if '[OBJECT-MATURE-CAPTURE]' in l]
        latency = next((re.search(r'\bhit=([\d.]+)ms', l) for l in native_logs
                        if '[V2.22.3 LATENCY]' in l), None)
        row = dict(shot_id=sid, physical_truth=label['physical_truth'], actual_object_id=label.get('object_id'),
            intended_object=label.get('intended_object'), exact_camera_gt='UNAVAILABLE_NOT_INFERRED',
            current_exact_result=current, current_latency_ms=float(latency[1]) if latency else None,
            maturity_export_log_count=len(export_logs), complete_mature_packets=0,
            object_state=None, object_reason=None, local_supported=None, local_observed_supported=None,
            object_latency_ms=None, local_unique_xy=None, local_exact_objects=None,
            first_worker_ms=None, first_registered_worker_ms=None, confirmation_times_ms=[],
            raw_object_pool='UNAVAILABLE_NOT_EXPORTED', region_proposal_pool='UNAVAILABLE_NOT_EXPORTED',
            pre_cleanup_pool='UNAVAILABLE_NOT_EXPORTED', classifications=['GROUND_TRUTH_UNAVAILABLE'] if label['physical_truth']=='UNKNOWN' else [])
        if t is None or not t.get('object_query_diagnostic'):
            row.update(trace_status='UNAVAILABLE' if t is None else 'INCOMPLETE_NO_QUERY_RESULT',
                       crop_status='UNAVAILABLE_NOT_ANALYZABLE', mature_pool_status='UNAVAILABLE_NOT_ANALYZABLE',
                       complete_mature_packets=None,
                       unavailable_reason='missing trace or unfinished query; no complete saved geometry/query context')
            row['classifications'].append('OTHER/UNKNOWN_MISSING_ARTIFACTS')
            for stage in ('WORKER_DELIVERY', 'LOCAL_CONFIRMATION'):
                stage_rows.append(dict(shot_id=sid, stage=stage, availability='UNAVAILABLE', candidate_count=None))
            shots.append(row); continue
        q = t['object_query_diagnostic']; peak = t['peak_ts']; cutoff = q['decision_ts']
        directory = safe_file(root, f'object_crops/shots/shot_{sid:08d}')
        complete = t.get('completeness', {}).get('trace_complete') is True
        convert, map_meta = mapping(t); snap = snapshot(t)
        next_peak = min((tr['peak_ts'] for tr in traces.values() if tr['peak_ts'] > peak), default=None)
        known_objects = {r.object_id for r in snap.game_regions}
        if label.get('object_id') and label['object_id'] not in known_objects:
            raise ValueError('operator object not in frozen geometry')
        crops = t.get('object_evidence_crops', [])
        missing_crops = []
        for c in crops:
            path = safe_file(directory, c['path']); guard_path(path)
            if not path.is_file(): missing_crops.append(c['path']); continue
            hashes[str(path)] = digest(path)
            pixels = np.load(path, allow_pickle=False)
            if list(pixels.shape) != c['shape'] or str(pixels.dtype) != c['dtype'] or hashlib.sha256(pixels.tobytes()).hexdigest() != c['sha256']:
                raise ValueError('saved crop hash/shape/dtype mismatch')
        def membership(xy):
            return mature_candidate_bridge(snap, convert, dict(candidates=[dict(camera_x=xy[0],camera_y=xy[1])]), cutoff)['rows'][0]['inside_object_ids']
        local_xy = []; observed = []
        for ob in q['per_object']:
            cs = ob['candidates']; local_xy.extend(c['camera_xy'] for c in cs)
            if ob['decision'] == 'supported_exact_shape': observed.append(ob['object_id'])
            for c in cs:
                recomputed = membership(c['camera_xy'])
                for a in c['regions']:
                    if (a['object_id'] in recomputed) != bool(a['inside_exact_shape']):
                        raise ValueError('saved exact membership differs from frozen geometry')
            object_rows.append(dict(shot_id=sid, object_id=ob['object_id'], observed_count=ob['observed_count'],
                retained_count=len(cs), decision=ob['decision'], best_camera_xy=cs[0]['camera_xy'] if cs else None,
                best_game_xy=cs[0].get('game_xy') if cs else None,
                best_evidence=cs[0].get('event_freshness') if cs else None))
        local_xy = sorted(set(map(tuple, local_xy)))
        local_objects = sorted({oid for xy in local_xy for oid in membership(xy)})
        supported = q['object_decision']['supported_object_ids']
        row.update(trace_status='COMPLETE' if complete else 'INCOMPLETE',
            crop_status='MISSING_REFERENCES' if missing_crops else 'CAPTURED' if crops else 'UNAVAILABLE',
            crop_count=len(crops), missing_crops=missing_crops, mapping=map_meta,
            pre_frame_ts=q.get('pre_frame_ts'), saved_post_frame_ts=sorted({c['timestamp'] for c in crops if c['kind']=='post'}),
            object_state=q['object_decision']['state'], object_reason=q['object_decision']['reason'],
            local_supported=supported, local_observed_supported=observed, local_unique_xy=len(local_xy),
            local_exact_objects=local_objects, local_examined=q['examined_candidate_count'],
            local_generated=q['candidate_count'], local_ledger_versions=sum(len(o['candidates']) for f in q['candidate_ledger'] for o in f['objects']),
            object_latency_ms=(cutoff-peak)*1000, object_processing_ms=q['timing']['prepare_ms']+sum(q['timing']['processing_ms']),
            local_gt_supported=label['object_id'] in supported if label.get('object_id') else None,
            local_non_gt_support=[x for x in supported if x!=label['object_id']] if label.get('object_id') else None)
        if row['object_reason']=='processing_deadline_exceeded': row['classifications'].append('TIMING/DELIVERY_FAILURE')
        elif len(supported)>1: row['classifications'].append('DISCRIMINATION_FAILURE_MULTI_SUPPORT_NOT_LABELLED_ERROR')
        packets = []
        for name in [f'pool_{i:04d}.json' for i in range(32)]+['LIMIT.json']:
            path = safe_file(directory, 'mature_candidates/'+name)
            if path.exists(): packets.append((path, read(path, hashes)))
        row['mature_pool_status'] = 'CAPTURED' if packets else 'UNAVAILABLE'
        row['complete_mature_packets'] = sum(p.get('complete') is True for _,p in packets)
        last_worker = None; last_registered = None; last_worker_objects = []; shot_worker_xy = []
        for path, p in packets:
            if p.get('shot_id')!=sid or p.get('peak_ts')!=peak or p.get('coordinate_space')!='camera':
                raise ValueError('native packet belongs to another event/coordinate system')
            deliver = p['available_at']; stage=p['stage']; dt=(deliver-peak)*1000
            shadow_packet = p if complete else dict(p, complete=False)
            at_delivery = mature_candidate_bridge(snap, convert, shadow_packet, deliver, next_peak=next_peak)
            at_cutoff = mature_candidate_bridge(snap, convert, shadow_packet, cutoff, next_peak=next_peak)
            cs = p['candidates']
            xy = [[c['camera_x'],c['camera_y']] for c in cs]
            flags = lambda c: all(c.get(k,0)>.5 for k in ('v2225_local_confirm','v252_evidence_ready','v252_fresh_physical'))
            inside_counts = {oid:sum(oid in a['inside_object_ids'] for a in at_delivery['rows']) for oid in sorted(known_objects)}
            native_objects = sorted({oid for a in at_delivery['rows'] if flags(a['candidate']) for oid in a['inside_object_ids']})
            diag=p.get('diagnostics',{}); registered=bool(diag.get('window_debug',{}).get('v252_registered_ready'))
            if stage=='WORKER_DELIVERY':
                last_registered=registered; last_worker=xy; shot_worker_xy.extend(xy)
                last_worker_objects=sorted(oid for oid,n in inside_counts.items() if n)
                if row['first_worker_ms'] is None: row['first_worker_ms']=dt
                if registered and row['first_registered_worker_ms'] is None: row['first_registered_worker_ms']=dt
            else: row['confirmation_times_ms'].append(dt)
            rec=dict(shot_id=sid, packet=path.name, stage=stage,
                availability='COMPLETE_BOUNDARY_OUTPUT' if p.get('complete') else 'INCOMPLETE_EXPORT',
                candidate_count=len(cs), native_input_count=p.get('input_count'), full_rescue=p.get('full_rescue'),
                frame_ts=p['frame_ts'], available_at=deliver, frame_ms=(p['frame_ts']-peak)*1000, delivery_ms=dt,
                delivered_before_object_decision=deliver<=cutoff, delivered_by_350ms=dt<=350,
                registered_worker=registered if stage=='WORKER_DELIVERY' else None,
                follows_registered_worker=last_registered, inside_exact_counts=inside_counts,
                candidates_inside_any_exact_shape=sum(bool(a['inside_object_ids']) for a in at_delivery['rows']),
                objects_with_candidates=sorted(oid for oid,n in inside_counts.items() if n),
                native_local_confirm_count=sum(c.get('v2225_local_confirm',0)>.5 for c in cs),
                registered_fresh_count=sum(c.get('v252_fresh_physical',0)>.5 for c in cs),
                native_flags_objects=native_objects, strict_exact_objects_at_delivery=at_delivery['supported_object_ids'],
                strict_objects_at_object_cutoff=at_cutoff['supported_object_ids'],
                strict_rejection_counts=dict(Counter(reason for a in at_delivery['rows'] for reason in a['reasons'])),
                gt_object_available=inside_counts.get(label['object_id'],0)>0 if label.get('object_id') else None,
                non_gt_objects=[oid for oid,n in inside_counts.items() if n and oid!=label['object_id']] if label.get('object_id') else None,
                source_counts=dict(Counter(a['source'] for a in at_delivery['rows'])),
                vault_carried_count=sum(c.get('v26_vault_carried',0)>.5 for c in cs),
                xy_survive_previous_worker=sum(tuple(v) in set(map(tuple,last_worker or [])) for v in xy) if stage=='LOCAL_CONFIRMATION' else None,
                worker_objects_no_longer_represented=[oid for oid in last_worker_objects if not inside_counts.get(oid)] if stage=='LOCAL_CONFIRMATION' else None,
                local_overlap=overlap(local_xy,xy), worker_ms=diag.get('worker_ms'), queue_ms=diag.get('queue_ms'),
                confirmation_core_ms=diag.get('ms'), tested_before_balance=diag.get('tested'),
                passed_before_balance=diag.get('confirmed'), stage_cpu_ms=diag.get('stages_ms'))
            stage_rows.append(rec)
            timing_rows.append({k:rec.get(k) for k in ('shot_id','packet','stage','frame_ms','delivery_ms','worker_ms','queue_ms','confirmation_core_ms','stage_cpu_ms')})
            for a in at_delivery['rows']:
                c=a['candidate'];candidate_rows.append(dict(shot_id=sid,packet=path.name,stage=stage,
                    order=a['index']+1,order_semantics='native pool order, not comparable score rank',
                    camera_xy=a['camera_xy'],game_xy=list(convert(*a['camera_xy'])),source=a['source'],
                    candidate_timestamp=c.get('timestamp'),available_at=deliver,
                    availability_at_delivery=a['availability'],pre_peak_candidate=c.get('timestamp',peak)<peak,
                    inside_object_ids=a['inside_object_ids'],full_rescue=p.get('full_rescue'),
                    native_local_confirm=c.get('v2225_local_confirm'),registered_fresh=c.get('v252_fresh_physical'),
                    evidence_offsets={k:c.get(k) for k in ('v2225_confirm_best_dx','v2225_confirm_best_dy','v252_evidence_best_dx','v252_evidence_best_dy')},
                    strict_reasons=a['reasons'],payload_ref=dict(path=str(path),candidate_index=a['index'])))
        for stage in ('WORKER_DELIVERY','LOCAL_CONFIRMATION'):
            if not any(p['stage']==stage for _,p in packets):
                stage_rows.append(dict(shot_id=sid,stage=stage,availability='UNAVAILABLE',candidate_count=None))
        row['mature_worker_unique_xy']=len(set(map(tuple,shot_worker_xy)))
        row['mature_worker_recurring_within5']=sum(nearest(x,previous_worker) is not None and nearest(x,previous_worker)<=5 for x in set(map(tuple,shot_worker_xy)))
        previous_worker.extend(shot_worker_xy)
        row['candidate_signal_loss']='UNDETERMINABLE_WITHOUT_INDEPENDENT_HOLE_GT_AND_INTERMEDIATE_POOLS'
        shots.append(row)
    exact_status=(exact_label_analysis(root,out,exact_camera_gt,ids,traces,stage_rows,candidate_rows,shots,hashes)
                  if exact_camera_gt else dict(exact_xy='UNAVAILABLE',valid_shot_ids=[]))
    worksheet=dict(capture=str(root),capture_play_sha256=hashes[str(root/'play.json')],
        truth_contract='Only independent observations; no intent, game response or detector-derived labels',
        shots=[dict(truth[sid],intended_object=truth[sid].get('intended_object'),confidence=truth[sid].get('confidence','unknown'),
                    notes=truth[sid].get('notes','No independent physical observation supplied')) for sid in ids])
    write(out/'operator_ground_truth.worksheet.json',worksheet)
    write(out/'truth_status.json',dict(shots=list(truth.values()),exact_xy=exact_status['exact_xy'],exact_camera_labels=exact_status,
        source=str(truth_path) if truth_path.exists() else None,planned_notes_not_bound_to_shots=True))
    write(out/'summary.json',dict(capture=str(root),events=len(ids),shots=shots,exit=exit_record,
        interruption='RECORDED' if exit_record and exit_record.get('interrupted') else 'NOT_RECORDED_IN_LEGACY_EXIT_METADATA',
        exact_camera_labels=exact_status,
        missing_truth='Only independently labelled subsets may be measured; object truth remains separate from exact XY',
        source_manifest=manifest))
    write(out/'stage_counts.json',stage_rows);write(out/'local_objects.json',object_rows)
    with (out/'candidates.jsonl').open('x') as f:
        for row in candidate_rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    _csv(out/'shots.csv',shots);_csv(out/'stage_counts.csv',stage_rows);_csv(out/'timing.csv',timing_rows)
    write(out/'input_manifest.json',dict(input_hashes=hashes,absence=dict(operator_truth=not truth_path.exists()),
        captured_event_ids=ids,scope='saved boundary analysis, not full detector replay'))
    source_paths=[Path(__file__),Path('automation/object_physical_replay.py'),Path('automation/object_hit_replay.py'),
                  Path('src/engine/game_objects/geometry.py'),Path('src/engine/input/object_hit_v2223.py')]
    write(out/'source_hashes.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        hashes={str(p):digest(p) for p in source_paths},elapsed_seconds=time.perf_counter()-start))
    if any(digest(p)!=h for p,h in hashes.items()):raise ValueError('input mutated during analysis')
    report=['# Paired physical candidate analysis','',
        'Practical conclusion: saved mature boundary pools can now be compared with local retained candidates.',
        'Without independent object/hole truth, superiority and actual-hole availability remain UNKNOWN.',
        'Early unregistered proposals and late registered evidence are separate; counts do not establish physical correctness.',
        'No live authority/threshold/ranking change. Next: independent annotation of existing saved evidence, not score tuning.','',
        '|Shot|Physical truth|OBJECT result/reason|Local supports|First worker ms|Registered worker ms|Confirmations ms|',
        '|---|---|---|---|---|---|---|']
    for r in shots:
        report.append(f"|{r['shot_id']}|{r['physical_truth']}|{r['object_state']} / {r['object_reason']}|{r['local_supported']}|{r['first_worker_ms']}|{r['first_registered_worker_ms']}|{r['confirmation_times_ms']}|")
    report+=['','Local ledgers retain only up to three points/object/frame, not the full generated pool.',
        'Mature worker output is post-cleanup; confirmation output is post-region-balance. Missing earlier stages remain UNAVAILABLE.',
        'Native confirmation/freshness flags allow neighborhood search; strict exact-XY bridge also requires zero offsets.',
        'Strict abstention is not proof no impact occurred. No scalar score comparison across producers.',
        ('Independent source-bound manual exact labels supplied; see candidate_nearest_gt.csv and oracle_summary.json, separate from object truth.' if exact_status['valid_shot_ids'] else 'No oracle files emitted: independent source-bound exact XY is unavailable.'),
        'CURRENT is a separate late result, never object truth. Recurrence is descriptive, not a re-hit veto.',
        'Missing shot artifacts remain explicit, including export log calls that did not persist any sidecar.',
        'physical_trace_label --paired-object-crops supports independent original-image clicks with source/crop hash binding and separate output.',
        'F1/F2 synthetic/projected labels do not retroactively supply physical GT for this capture.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,action='append',required=True)
    p.add_argument('--historical-cache',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--paired',action='store_true',help='Per-packet counts/timing and explicit partial-event/GT boundaries')
    p.add_argument('--operator-truth',type=Path,help='Separate existing operator_ground_truth shots schema; never inferred')
    p.add_argument('--exact-camera-gt',type=Path,help='Validated independent source-bound manual crop labels')
    a=p.parse_args()
    if a.paired:
        if len(a.capture)!=1 or a.historical_cache:p.error('--paired requires one capture and no historical cache')
        print(paired_analysis(a.capture[0],a.output,a.operator_truth,a.exact_camera_gt))
    else:
        if a.operator_truth or a.exact_camera_gt:p.error('truth options require --paired')
        run(a.capture,a.output,a.historical_cache)


if __name__=='__main__':main()
