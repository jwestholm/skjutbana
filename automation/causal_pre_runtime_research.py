"""Fixed causal PRE performance/equivalence measurement. No ranking research."""
from __future__ import annotations
import argparse
from collections import defaultdict
import cProfile
import csv
import json
from pathlib import Path
import pstats
import time
import platform
from unittest.mock import patch as mock_patch

import cv2
import numpy as np

from automation import causal_pre_onset_research as reference

PRIOR = reference.BASE/'causal_pre_onset_20260925_01'


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), worst=float(a.max()), count=len(a))


def write(path, value):
    reference.write(path, value)


def events():
    data = json.loads(reference.CACHE.read_text())
    paths = {e['event']:Path(e['trace_path']) for e in data['events'] if e['session']!='D01'}
    contexts = {e['event']:e['context'] for e in data['events']}
    rows = []
    for e in json.loads((PRIOR/'final/results.json').read_text())['events']:
        f = json.loads((PRIOR/'final'/(e['event'].replace(':','_')+'_evidence.json')).read_text())
        rows.append(dict(event=e['event'], path=str(paths[e['event']]), times=[v['timestamp'] for v in f['frames']],
            frames=f['frames'], points=[c['camera_xy'] for c in f['candidates']], expected=f['candidates'],
            snapshot=f['old_snapshot'], cutoff=f['cutoff'], lower=f['lower_boundary'],
            early_times=contexts[e['event']]['reference_times']))
    root = Path(json.loads((reference.CAN/'binding.json').read_text())['trace_root'])
    for e in json.loads((PRIOR/'audit/can_local_evidence.json').read_text()):
        sid=int(e['event'].split(':')[1]);path=root/'shots'/f'shot_{sid:08d}'/'trace.json'
        rows.append(dict(event=e['event'],path=str(path),times=[v['timestamp'] for v in e['frames']],frames=e['frames'],
            points=e['points'],expected=e['effects'],snapshot={'timestamp':e['old_pre_ts']},
            cutoff=e['cutoff'],lower=None,early_times=[]))
    return rows


def exact_rows(actual, expected):
    fields=set(actual[0]) if actual else set()
    return actual == [{k:c[k] for k in fields} for c in expected]


def profile_before(output):
    output.mkdir(parents=True, exist_ok=False);cv2.setNumThreads(1)
    rows=events();roots={Path(e['path']).parent.parent.parent for e in rows}
    accesses=reference.read_guard(output,roots);hashes={};results=[]
    for e in rows:
        path=Path(e['path']);trace=json.loads(path.read_text())
        counters=defaultdict(float);calls=defaultdict(int)
        def timed(name, fn):
            def call(*args,**kwargs):
                start=time.perf_counter()
                try:return fn(*args,**kwargs)
                finally:counters[name]+=(time.perf_counter()-start)*1000;calls[name]+=1
            return call
        start=time.perf_counter()
        with mock_patch.object(reference,'digest',timed('hash_io',reference.digest)), \
             mock_patch.object(reference.np,'load',timed('npy_open',np.load)), \
             mock_patch.object(reference,'sample',timed('patch_extraction',reference.sample)):
            times,pixels,snapshot,frames=reference.load_observations(path,trace,e['points'],e['cutoff'],
                e['lower'] if e['lower'] is not None else -float('inf'),hashes)
        counters['load_total']=(time.perf_counter()-start)*1000
        if e['event'].startswith('CAN:'):
            snapshot=(e['snapshot'],pixels[times.index(e['snapshot']['timestamp'])])
        assert times==e['times']
        args=(times,pixels,snapshot,e['cutoff'],e['early_times'])
        durations=[]
        for repeat in range(3):
            start=time.perf_counter();actual=reference.infer(*args);durations.append((time.perf_counter()-start)*1000)
            if not exact_rows(actual,e['expected']):raise AssertionError('Reference mismatch '+e['event'])
        profiler=cProfile.Profile()
        profiler.enable();reference.infer(*args);profiler.disable()
        raw=pstats.Stats(profiler).stats
        functions=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],exclusive_ms=v[2]*1000,cumulative_ms=v[3]*1000)
                   for k,v in raw.items()]
        # Timed wrappers give direct phase boundaries; cProfile separately
        # attributes median/allocation/reduction overhead, without claiming
        # those nested values are additive with enclosing phase durations.
        with mock_patch.object(reference,'local_contrast',timed('local_contrast',reference.local_contrast)), \
             mock_patch.object(reference,'causal_local_onset',timed('onset_search',reference.causal_local_onset)), \
             mock_patch.object(reference,'persistent_local_support',timed('support',reference.persistent_local_support)):
            start=time.perf_counter();reference.infer(*args);wrapped=(time.perf_counter()-start)*1000
        counters['reference_comparisons_and_loop']=wrapped-counters['local_contrast']-counters['onset_search']-counters['support']
        counters['inference']=float(np.median(durations))
        counters['gray_conversion']=0.;counters['global_difference_maps']=0.
        saved=output/e['event'].replace(':','_');saved.mkdir()
        np.save(saved/'patches.npy',pixels,allow_pickle=False)
        if snapshot:np.save(saved/'snapshot.npy',snapshot[1],allow_pickle=False)
        write(saved/'event.json',e)
        write(saved/'reference.json',actual)
        write(saved/'cprofile.json',sorted(functions,key=lambda f:-f['cumulative_ms']))
        results.append(dict(event=e['event'],frames=len(times),candidates=len(e['points']),stages_ms=dict(counters),
            calls=dict(calls),repeated_inference_ms=durations,reference_exact=True))
        print(e['event'],'reference exact',len(e['points']),'inference',round(counters['inference'],2),flush=True)
    expected_hashes=json.loads((PRIOR/'final/input_hashes.json').read_text())
    expected_hashes.update(json.loads((PRIOR/'audit/input_hashes.json').read_text()))
    for p,h in hashes.items():
        if expected_hashes.get(p)!=h:raise ValueError('Input differs from prior package: '+p)
    dev=[r for r in results if not r['event'].startswith('CAN:')]
    aggregate={k:stats([e['stages_ms'][k] for e in dev]) for k in dev[0]['stages_ms']}
    write(output/'profiling_before.json',dict(events=results,development_summary_ms=aggregate,
        repeats=3,summary='Median of three uninstrumented inference calls per event; phase wrappers and cProfile separate',
        gray='Inputs already grayscale; no conversion or global differencing in the prior 292 ms',
        io='np.load open excludes subsequent memory-map page faults; patch extraction includes touched-page reads',
        categories=dict(hash_io='D',npy_open='D',patch_extraction='A; grayscale input reusable B/C',
            local_contrast='A; masks reusable B',onset_search='A; shared times B; reductions batched',
            reference_comparisons_and_loop='A with research-only old/early diagnostics',support='A')))
    with (output/'profiling_before.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=['event','frames','candidates',*aggregate]);writer.writeheader()
        writer.writerows(dict(event=r['event'],frames=r['frames'],candidates=r['candidates'],**r['stages_ms']) for r in results)
    write(output/'input_hashes.json',hashes);write(output/'input_accesses.json',sorted(accesses))
    write(output/'source_hashes.json',{s:reference.digest(s) for s in (
        __file__,'automation/causal_pre_onset_research.py','src/engine/offline/clean_physical_change.py')})
    print('SUMMARY',json.dumps(aggregate),flush=True)


def profile_after(output, before):
    from src.engine.offline.common_verifier import CommonTemporalHistory
    output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    entries=[json.loads(p.read_text()) for p in sorted(before.glob('*/event.json'))]
    roots={Path(e['path']).parent.parent.parent for e in entries}
    accesses=reference.read_guard(output,roots)
    results=[];equivalence=[];maintenance=[];replayed=[];scaling=[]
    prior={r['event']:r for r in json.loads((PRIOR/'final/results.json').read_text())['events']}
    for e in entries:
        h=CommonTemporalHistory();frame_ids=[];append_ms=[];load_ms=[]
        for f in e['frames']:
            start=time.perf_counter()
            gray=np.load(Path(e['path']).parent/f['path'],mmap_mode='r',allow_pickle=False)
            opened=time.perf_counter()
            # Historical trace seq is serialization, not camera sequence. No
            # sensor acquisition timestamp or camera sequence is fabricated.
            h.append(gray,timestamp=f['timestamp'],frame_id=f['path'],stream_id=e['event'],
                     camera_sequence=None,provenance=f)
            end=time.perf_counter();frame_ids.append(f['path'])
            append_ms.append((end-opened)*1000);load_ms.append((opened-start)*1000)
        maintenance.extend(append_ms)
        snapshot_id=frame_ids[e['times'].index(e['snapshot']['timestamp'])]
        expected=json.loads((before/e['event'].replace(':','_')/'reference.json').read_text())
        def query(points):
            return h.causal_query(frame_ids,points,cutoff=e['cutoff'],snapshot_frame_id=snapshot_id,early_times=e['early_times'])
        durations=[]
        # Warm-up, then five measured queries, checking every result exactly.
        for repeat in range(6):
            actual,meta=query(e['points'])
            if actual!=expected:
                write(output/'MISMATCH.json',dict(event=e['event'],repeat=repeat,actual=actual,expected=expected))
                raise AssertionError('STOP: semantic mismatch '+e['event'])
            if repeat:durations.append({k:v for k,v in meta.items() if k.endswith('_ms')})
        def ordering(rows):
            supported=[j for j,r in enumerate(rows) if r['state']=='ONSET_FOUND' and r['causal']['support']>0
                       and e['expected'][j].get('ready',True)]
            return sorted(supported,key=lambda j:-rows[j]['causal']['support'])
        order=ordering(actual);old_order=ordering(expected)
        assert order==old_order
        chosen=order[0] if order else None
        detail=[]
        for j,r in enumerate(actual):
            detail.append(dict(candidate=j,camera_xy=e['points'][j],exact=True,state=r['state'],
                pre_frame=meta['frames'][r['pre_index']] if r['pre_index'] is not None else None,
                onset_frame=meta['frames'][r['onset_index']] if r['onset_index'] is not None else None))
        equivalence.append(dict(event=e['event'],candidate_count=len(actual),all_fields_exact=True,
            supported_order=order,selected_local_index=chosen,
            selected_camera_xy=e['points'][chosen] if chosen is not None else None,
            decision='PHYSICAL_HIT' if chosen is not None else 'UNCERTAIN',candidates=detail))
        if e['event'] in prior:
            p=prior[e['event']]
            selected=e['expected'][chosen]['index'] if chosen is not None else None
            assert selected==p['causal']['index']
            # Join frozen labels only after exact evidence/ordering verification.
            replayed.append({k:p[k] for k in ('event','session','truth','current','causal','changes') if k in p})
        totals={k:stats([r[k] for r in durations]) for k in durations[0]}
        results.append(dict(event=e['event'],frames=len(frame_ids),candidates=len(actual),
            stages_ms={k:v['median'] for k,v in totals.items()},repetitions=durations,stage_statistics_ms=totals,
            per_candidate_amortized_ms=totals['total_ms']['median']/len(actual),
            append_ms=append_ms,append_summary_ms=stats(append_ms),research_open_ms=load_ms,
            history_bytes=h.bytes_used,temporary_patch_bytes=meta['temporary_patch_bytes'],
            slack_ms=prior[e['event']]['receipt_margin_ms'] if e['event'] in prior else None))
        # One fixed real history; duplicated existing coordinates ONLY for
        # performance scaling, not a new candidate or accuracy experiment.
        if e['event']=='S01:1':
            for count in (1,8,32,64,128,200):
                points=[e['points'][j%len(e['points'])] for j in range(count)]
                query(points);measure=[]
                for _ in range(5):
                    _,timing=query(points);measure.append(timing['total_ms'])
                scaling.append(dict(candidates=count,frames=len(frame_ids),post_event_ms=stats(measure),samples_ms=measure))
        write(output/(e['event'].replace(':','_')+'_evidence.json'),actual)
        print(e['event'],'EXACT',len(actual),'total',round(totals['total_ms']['median'],2),flush=True)
    dev=[r for r in results if not r['event'].startswith('CAN:')]
    aggregate={k:stats([r['stages_ms'][k] for r in dev]) for k in dev[0]['stages_ms']}
    write(output/'profiling_after.json',dict(events=results,development_summary_ms=aggregate,
        continuous_append_all_frames_ms=stats(maintenance),
        per_candidate_amortized_ms=stats([r['per_candidate_amortized_ms'] for r in dev]),
        deadline_fit=sum(r['stages_ms']['total_ms']<=r['slack_ms'] for r in dev),
        full_query_under50=sum(r['stages_ms']['total_ms']<50 for r in dev),
        method='One warm-up; median of five exact queries per event; gray append owns one copy, includes mmap read faults',
        environment=dict(python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,opencv=cv2.__version__,opencv_threads=cv2.getNumThreads())))
    with (output/'profiling_after.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=['event','frames','candidates','slack_ms','per_candidate_amortized_ms',*aggregate]);writer.writeheader()
        writer.writerows({**{k:r[k] for k in ('event','frames','candidates','slack_ms','per_candidate_amortized_ms')},**r['stages_ms']} for r in results)
    write(output/'semantic_equivalence.json',dict(events=equivalence,exact_events=len(equivalence),
        exact_candidates=sum(r['candidate_count'] for r in equivalence),mismatches=0,tolerance=0,
        exclusions='Six D01 overwritten histories and S01:11 lacking cutoff remain unavailable; S03 excluded'))
    write(output/'performance_scaling.json',dict(event='S01:1',synthetic_counts_only=True,results=scaling))
    write(output/'selector_replay.json',dict(events=replayed,
        summaries={s:{m:reference.summary([r for r in replayed if s=='POOLED' or r['session']==s],m) for m in ('current','causal')}
                   for s in ('S01','S02','POST_FIX','POOLED')}))
    write(output/'input_accesses.json',sorted(accesses))
    write(output/'source_hashes.json',{s:reference.digest(s) for s in (__file__,
        'src/engine/offline/common_verifier.py','src/engine/offline/clean_physical_change.py',
        'src/engine/offline/accuracy_verifier.py')})
    print('SUMMARY',json.dumps(aggregate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--before',type=Path,help='Frozen reference profiling directory; enables optimized equivalence/benchmark')
    args=p.parse_args()
    if args.before:profile_after(args.output,args.before)
    else:profile_before(args.output)
