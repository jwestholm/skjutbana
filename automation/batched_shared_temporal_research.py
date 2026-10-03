"""Bounded scalar/shared versus batch measurement. No physical policy change."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from automation import chronological_pre_research as chrono
from automation.shared_witness_research import events
from automation.causal_pre_onset_research import read_guard,write
from src.engine.offline.common_verifier import query_physical_witnesses
from src.engine.offline.observation_hypotheses import retain_exact_physical_witnesses

BASE=Path('/data/skjutbana/evaluation_runs/shared_witness_donors_20260927_01')

def witness_batch(cs,event,frames):
    start=time.perf_counter()
    fs=[dict(frame_id=str(f['path']),ordinal=f['ordinal'],receipt_timestamp=f['timestamp'],camera_sequence=f['sequence']) for f in frames]
    identities=[dict(candidate_id=c['uid'],coordinate_version=c['uid'],camera_xy=c['camera_xy'],
        source_provenance=dict(producer='V251_RECONSTRUCTED',born_ordinal=c['born'],proposal_ordinal=c['source_ordinal'],reference_ordinals=c['reference_ordinals']),
        previous=c.get('physical_witness')) for c in cs]
    values=np.asarray([c['signal'] for c in cs],float).T if cs else np.empty((len(fs),0))
    setup=(time.perf_counter()-start)*1000;profile={}
    bundles=query_physical_witnesses(values,fs,event_id=event['id'],candidates=identities,
        cutoff_frame_id=fs[-1]['frame_id'],cutoff_ts=fs[-1]['receipt_timestamp'],profile=profile)
    profile['context_ms']+=setup
    return [b.witnesses[0] for b in bundles],profile


def query_gate(output):
    """Saved scalar inputs, every prefix; no producer float variability involved."""
    results=[];cv2.setNumThreads(1)
    for event in events():
        key=event['id'].replace(':','_');previous={};checked=0
        for line in (BASE/'replay'/(key+'_transitions.jsonl')).open():
            r=json.loads(line);old=r['evaluated'];cs=[]
            for c in old:
                w=c['physical_witness'];cs.append(dict(c,signal=w['signal'],physical_witness=previous.get(c['uid'])))
            computed,profile=witness_batch(cs,event,event['pre'][:r['ordinal']+1])
            for c,w in zip(old,computed):
                if w!=c['physical_witness']:
                    raise AssertionError((event['id'],r['ordinal'],c['uid'],{k:(c['physical_witness'].get(k),w.get(k)) for k in w if w[k]!=c['physical_witness'].get(k)}))
                checked+=1
            previous={c['uid']:w for c,w in zip(cs,computed) if c['uid'] in r['retained']}
        results.append(dict(event=event['id'],candidate_prefixes=checked,mismatches=0))
    write(output/'query_equivalence.json',results);print('QUERY GATE',sum(r['candidate_prefixes'] for r in results),'exact matches',flush=True)


def compare(event,folder):
    key=event.replace(':','_');oldroot=BASE/'replay';mismatches=[];numeric={};count=0
    before=[json.loads(s) for s in (oldroot/(key+'_transitions.jsonl')).read_text().splitlines()]
    after=[json.loads(s) for s in (folder/(key+'_transitions.jsonl')).read_text().splitlines()]
    if len(before)!=len(after):mismatches.append('prefix_count')
    identity=('camera_xy','source_ordinal','reference_ordinals','source_rank','association_uid','query_unavailable')
    for a,b in zip(before,after):
        for field in ('retained','retired'):
            if a[field]!=b[field]:mismatches.append([a['ordinal'],field])
        if [[c.get(k) for k in identity] for c in a['generated']] != [[c.get(k) for k in identity] for c in b['generated']]:mismatches.append([a['ordinal'],'proposal_identity_order'])
        if [c['uid'] for c in a['evaluated']] != [c['uid'] for c in b['evaluated']]:mismatches.append([a['ordinal'],'candidate_identity_order'])
        for c,d in zip(a['generated'],b['generated']):
            for k in set(c)|set(d):
                if c.get(k)!=d.get(k) and isinstance(c.get(k),(float,int)) and isinstance(d.get(k),(float,int)):
                    numeric.setdefault(k,[]).append(abs(c[k]-d[k]))
        for c,d in zip(a['evaluated'],b['evaluated']):
            count+=1
            if c['physical_witness']!=d['physical_witness']:mismatches.append([a['ordinal'],c['uid'],'witness'])
    a=json.loads((oldroot/(key+'_frozen.json')).read_text());b=json.loads((folder/(key+'_frozen.json')).read_text())
    xy=lambda r:r['selected']['camera_xy'] if r['selected'] else None
    if xy(a)!=xy(b) or a['primary_state']!=b['primary_state']:mismatches.append('final_decision')
    if [c['camera_xy'] for c in a['supported']]!=[c['camera_xy'] for c in b['supported']]:mismatches.append('final_order')
    post_a=json.loads((oldroot/(key+'_post_checks.json')).read_text());post_b=json.loads((folder/(key+'_post_checks.json')).read_text())
    if post_a!=post_b:mismatches.append('post_diagnostics')
    return dict(event=event,candidate_prefixes=count,mismatches=mismatches,selected_xy=xy(b),
        final_supported=len(b['supported']),producer_float_drift={k:dict(count=len(v),max_abs=max(v)) for k,v in numeric.items()})


def replay(output):
    es=events();access=read_guard(output,[e['root'].parent.parent for e in es]);cv2.setNumThreads(1)
    folder=output/'after_replay';folder.mkdir(exist_ok=False);rows=[];comparisons=[]
    settings=json.loads(Path('content/ai/settings.json').read_text())
    with patch.object(chrono.region,'_runtime_settings',return_value=settings):
        for e in es:
            r=chrono.run_event(e,folder,keep_fn=retain_exact_physical_witnesses,witness_batch_query=witness_batch)
            rows.append(dict(event=e['id'],timings=r['timings'],total_ms=r['total_pre_work_ms'],array_bytes=r['accounted_peak_array_bytes'],process_peak_rss_kib=r['process_peak_rss_kib']))
            c=compare(e['id'],folder);comparisons.append(c)
            # Checkpoints belong only to this fresh output package.
            (output/'profiling_after.json').write_text(json.dumps(rows,indent=2))
            (output/'semantic_equivalence.json').write_text(json.dumps(comparisons,indent=2))
            print(e['id'],round(r['total_pre_work_ms'],1),'ms; mismatches',len(c['mismatches']),flush=True)
            if c['mismatches']:raise AssertionError(c)
    write(output/'read_accesses.json',sorted(access))



def scalar_reference_modules():
    """Load preserved scalar source for measurements, not a new runtime path."""
    import importlib.util
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        import sys
        sys.modules[name]=module;spec.loader.exec_module(module)
        return module
    root=BASE/'source_snapshot'
    common=load('src.engine.offline._scalar_benchmark_reference',root/'src/engine/offline/common_verifier.py')
    oldchrono=load('automation._scalar_chrono_benchmark_reference',root/'automation/chronological_pre_research.py')
    shared=load('automation._scalar_shared_benchmark_reference',root/'automation/shared_witness_research.py')
    shared.query_physical_witness=common.query_physical_witness
    return common,oldchrono,shared


def scalar_replay(output,older_only=False):
    _,old,shared=scalar_reference_modules();cv2.setNumThreads(1)
    es=events()[12:] if older_only else events()
    read_guard(output,[e['root'].parent.parent for e in es])
    folder=output/('before_older_replay' if older_only else 'scalar_replay');folder.mkdir(exist_ok=False)
    rows=[];settings=json.loads(Path('content/ai/settings.json').read_text())
    with patch.object(old.region,'_runtime_settings',return_value=settings):
        for e in es:
            r=old.run_event(e,folder,keep_fn=retain_exact_physical_witnesses,witness_query=shared.witness)
            rows.append(dict(event=e['id'],timings=r['timings'],total_ms=r['total_pre_work_ms'],array_bytes=r['accounted_peak_array_bytes']))
            print('SCALAR',e['id'],r['total_pre_work_ms'],flush=True)
    write(output/('profiling_before_older.json' if older_only else 'profiling_scalar.json'),rows)


def scaling(output):
    scalar,_,_=scalar_reference_modules();rows=[];cv2.setNumThreads(1)
    for key in ('KOMP_2','D02_4'):
        previous={}
        for line in (BASE/'replay'/(key+'_transitions.jsonl')).open():
            r=json.loads(line);cs=r['evaluated'];n=r['ordinal']+1
            if n in (12,24,50):
                for count in (1,8,16,32):
                    if count>len(cs):continue
                    selected=cs[:count];fs=selected[0]['physical_witness']['frames']
                    candidates=[dict(candidate_id=c['uid'],coordinate_version=c['uid'],camera_xy=c['camera_xy'],source_provenance=c['physical_witness']['source_provenance'],previous=previous.get(c['uid'])) for c in selected]
                    x=np.asarray([c['physical_witness']['signal'] for c in selected]).T
                    args=dict(event_id=selected[0]['physical_witness']['event_id'],cutoff_frame_id=fs[-1]['frame_id'],cutoff_ts=fs[-1]['receipt_timestamp'])
                    baseline=lambda:[scalar.query_physical_witness(x[:,i],fs,**c,**args) for i,c in enumerate(candidates)]
                    batched=lambda:query_physical_witnesses(x,fs,candidates=candidates,**args)
                    assert [b.witnesses for b in baseline()]==[b.witnesses for b in batched()]
                    for mode,fn in [('scalar',baseline),('batch',batched)]:
                        values=[]
                        for _ in range(20):
                            t=time.perf_counter();fn();values.append((time.perf_counter()-t)*1000)
                        rows.append(dict(event=key,frames=n,candidates=count,mode=mode,ms=values))
            previous={c['uid']:c['physical_witness'] for c in cs if c['uid'] in r['retained']}
    write(output/'scaling.json',dict(scope='performance only; subsets of real recorded candidates/prefixes; repeated warm queries, no accuracy claims',rows=rows))


def operation_profile(output):
    import cProfile,pstats
    folder=output/'after_operation_profile';folder.mkdir(exist_ok=False);rows=[];cv2.setNumThreads(1)
    settings=json.loads(Path('content/ai/settings.json').read_text())
    with patch.object(chrono.region,'_runtime_settings',return_value=settings):
        for e in [e for e in events() if e['id'] in ('KOMP:2','D02:4')]:
            p=cProfile.Profile();p.enable()
            chrono.run_event(e,folder,keep_fn=retain_exact_physical_witnesses,witness_batch_query=witness_batch)
            p.disable();p.dump_stats(str(folder/(e['id'].replace(':','_')+'.prof')));calls=[]
            for (file,line,name),(pc,n,selft,total,_) in pstats.Stats(p).stats.items():
                if name in ('query_physical_witnesses','_physical_witness_context','_physical_witness_identity','_physical_witness_bundle','temporal_step_evidence_batch','_observed_step_choices_batch','causal_local_onset','median','local_contrast_batch','camera_local_contrasts','camera_local_patches','patch','transition_maps','_build_reference_and_noise','_register_current','_normalise_photometry','_multiscale_change_response','_edge_weights','_robust_threshold','_refine_peak','_candidate_features','deepcopy'):
                    calls.append(dict(file=file,line=line,name=name,calls=n,self_ms=selft*1000,total_ms=total*1000))
            rows.append(dict(event=e['id'],calls=calls));print('PROFILE',e['id'],flush=True)
    write(output/'operation_profile_after.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('query-gate','replay','scalar-replay','scalar-older','scaling','profile'),required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    actions={'query-gate':query_gate,'replay':replay,'scalar-replay':scalar_replay,
        'scalar-older':lambda out:scalar_replay(out,older_only=True),'scaling':scaling,'profile':operation_profile}
    actions[a.phase](a.output)
