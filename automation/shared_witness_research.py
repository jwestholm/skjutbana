"""Consolidation parity and stage profiling; no new physical policy."""
import argparse,json,math,time
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from automation import chronological_pre_research as chrono
from automation.causal_pre_onset_research import read_guard,write,digest
from src.engine.offline.common_verifier import query_physical_witness
from src.engine.offline.observation_hypotheses import retain_exact_physical_witnesses

BASE=Path('/data/skjutbana/evaluation_runs/exact_witness_20260927_01')

def witness(c,event,frames):
    fs=[dict(frame_id=str(f['path']),ordinal=f['ordinal'],receipt_timestamp=f['timestamp'],camera_sequence=f['sequence']) for f in frames]
    return query_physical_witness(c['signal'],fs,event_id=event['id'],candidate_id=c['uid'],coordinate_version=c['uid'],
        camera_xy=c['camera_xy'],source_provenance=dict(producer='V251_RECONSTRUCTED',born_ordinal=c['born'],
        proposal_ordinal=c['source_ordinal'],reference_ordinals=c['reference_ordinals']),cutoff_frame_id=fs[-1]['frame_id'],
        cutoff_ts=fs[-1]['receipt_timestamp'],previous=c.get('physical_witness')).witnesses[0]


def events():
    out=chrono.descriptors(d02_sids=range(1,7))
    for row in json.loads((BASE/'historical/inputs.json').read_text()):
        pre=[dict(f,path=Path(f['path'])) for f in row['pre']];root=pre[0]['path'].parent.parent
        out.append(dict(id=row['event'],sid=int(row['event'].split(':')[1]),pre=pre,post=[],root=root,trace=root/'trace.json',
            mask=Path(row['mask']),cutoff=row['cutoff'],crop=row['crop']))
    return out


def run(output):
    output.mkdir(parents=True,exist_ok=False);es=events();accesses=read_guard(output.parent,{e['root'].parent.parent for e in es});cv2.setNumThreads(1)
    settings=json.loads(Path('content/ai/settings.json').read_text());comparisons=[];profiles=[]
    before=output.parent/'profile_reference';before.mkdir(exist_ok=False)
    with patch.object(chrono.region,'_runtime_settings',return_value=settings):
        # Establish stage profile with the unwrapped rule before any optimization.
        for e in es[:12]:
            r=chrono.run_event(e,before,keep_fn=retain_exact_physical_witnesses)
            profiles.append(dict(event=e['id'],mode='reference',timings=r['timings'],total_ms=r['total_pre_work_ms']))
        write(output.parent/'profiling_before.json',profiles)
        print('Reference profile complete; no optimization applied.',flush=True)
        for e in es:
            r=chrono.run_event(e,output,keep_fn=retain_exact_physical_witnesses,witness_query=witness)
            profiles.append(dict(event=e['id'],mode='shared',timings=r['timings'],total_ms=r['total_pre_work_ms']))
            key=e['id'].replace(':','_');oldroot=BASE/'retention' if e['id'].startswith(('KOMP','D02')) else BASE/'historical/exact'
            old=json.loads((oldroot/(key+'_frozen.json')).read_text())
            a=[json.loads(l) for l in (oldroot/(key+'_transitions.jsonl')).read_text().splitlines()]
            b=[json.loads(l) for l in (output/(key+'_transitions.jsonl')).read_text().splitlines()]
            mismatches=[];revocations=[];checked=0
            for aa,bb in zip(a,b):
                for field in ('retained','retired'):
                    if aa[field]!=bb[field]:mismatches.append([aa['ordinal'],field])
                identity_fields=('camera_xy','source_ordinal','source_rank','reference_ordinals','association_uid','query_unavailable')
                if [[c.get(k) for k in identity_fields] for c in aa['generated']] != [[c.get(k) for k in identity_fields] for c in bb['generated']]:
                    mismatches.append([aa['ordinal'],'generated_coordinate_provenance_order'])
                if [c['uid'] for c in aa['evaluated']] != [c['uid'] for c in bb['evaluated']]:
                    mismatches.append([aa['ordinal'],'evaluated_identity_order'])
                oldc={c['uid']:c for c in aa['evaluated']}
                for c in bb['evaluated']:
                    checked+=1;prior=oldc.get(c['uid'])
                    if prior is None or any(prior[k]!=c[k] for k in ('camera_xy','temporal')):mismatches.append([aa['ordinal'],c['uid'],'exact_xy_or_evidence'])
                    if c['physical_witness']['revocation']['revoked_now']:revocations.append(dict(ordinal=bb['ordinal'],xy=c['camera_xy'],state=c['physical_witness']['state'],reason=c['physical_witness']['revocation']['reason']))
            oldxy=old['selected']['camera_xy'] if old['selected'] else None;newxy=r['selected']['camera_xy'] if r['selected'] else None
            if oldxy!=newxy or len(a)!=len(b):mismatches.append(['final_or_length'])
            comparisons.append(dict(event=e['id'],compared_candidate_prefixes=checked,mismatches=mismatches,old_xy=oldxy,new_xy=newxy,final_supported=len(r['supported']),revocations=revocations))
            print(e['id'],'mismatches',len(mismatches),'selected',newxy,flush=True)
            if mismatches:raise AssertionError(comparisons[-1])
    write(output.parent/'semantic_equivalence.json',comparisons);write(output.parent/'profiling_after.json',profiles)
    write(output.parent/'read_accesses.json',sorted(accesses))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)


def benchmark_existing_batch(package):
    """DIAGNOSTIC only: reuse existing batch kernel on recorded-clock witnesses.

    Does not change selection, ordinal handling, sampling, scheduling or runtime.
    """
    from src.engine.offline.clean_physical_change import causal_local_onset_batch,persistent_local_support
    rows=[];mismatches=[];checked=0
    for e in events():
        if e['pre'][0]['timestamp'] is None:continue
        path=package/'replay'/(e['id'].replace(':','_')+'_transitions.jsonl')
        for line in path.open():
            r=json.loads(line);cs=r['evaluated']
            if not cs:continue
            signal=np.asarray([c['physical_witness']['signal'] for c in cs]).T
            frames=cs[0]['physical_witness']['frames'];times=[f['receipt_timestamp'] for f in frames]
            seq=[f.get('camera_sequence') for f in frames]
            if all(s is None for s in seq):seq=None
            start=time.perf_counter();batch=causal_local_onset_batch(signal,times,cutoff=times[-1],sequences=seq)
            for i,(c,b) in enumerate(zip(cs,batch)):
                original=c['temporal'];checked+=1
                if b['state']!=original['state'] or b['candidates']!=original['choices']:mismatches.append([e['id'],r['ordinal'],c['uid'],'state_or_choices'])
                if b['state']=='ONSET_FOUND':
                    support=persistent_local_support(signal[b['pre_index'],i],signal[-3:,i],b['noise'])
                    if support['support']!=original['support'] or support['signed_change']!=original['signed_change']:mismatches.append([e['id'],r['ordinal'],c['uid'],'support'])
            rows.append(dict(event=e['id'],ordinal=r['ordinal'],candidates=len(cs),ms=(time.perf_counter()-start)*1000))
    write(package/'batch_reuse_diagnostic.json',dict(rows=rows,compared_candidates=checked,mismatches=mismatches,
        scope='already existing batch kernel, clocked history only; no streaming/runtime integration; includes equivalence checking cost'))
    if mismatches:raise AssertionError(mismatches[:3])
