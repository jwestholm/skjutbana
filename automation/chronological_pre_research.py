"""One bounded chronological PRE experiment. Offline only; no live imports here.

Camera ordinals are never passed off as acquisition/receipt timestamps. Unknown
proposal timestamp is NaN inside the legacy constructor only, and is stripped
before export; all physical computations use observed images, not that field.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import resource
import time
from types import SimpleNamespace
from unittest.mock import patch as mock

import cv2
import numpy as np

from automation.causal_pre_onset_research import digest, read_guard, sample, write
from automation.kompetensprov_selection_research import ordinal_steps, CROP, SESSION, RUN, clean
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
from src.engine.offline.clean_physical_change import local_contrast, causal_local_onset, persistent_local_support
from src.engine import shot_region_proposal_v251 as region

BASE=Path('/data/skjutbana/evaluation_runs')
D02=Path('/data/skjutbana/physical_traces/session_20260926_162528_D02_4e1d6abd')
CAPACITY=24
PENDING_FRAMES=3
MAX_HISTORY=50
RADII=(5,10,20,42)


def transition_maps(engine, previous, current, mask):
    """Existing V2 physical map operations without audio dispatch or fake clocks.

    Every reference is a distinct transition. Its first persistence contribution
    is zero, exactly as the existing generator's fresh-reference path. Tests
    require array-exact parity with that path on independent image fixtures.
    """
    cfg=engine.config.snapshot()
    bx,by,bw,bh=cv2.boundingRect(mask)
    x0,y0=max(0,bx-2),max(0,by-2)
    x1,y1=min(mask.shape[1],bx+bw+2),min(mask.shape[0],by+bh+2)
    roi=mask[y0:y1,x0:x1]
    ref,noise,stats=engine._build_reference_and_noise([p[y0:y1,x0:x1] for p in previous],roi=roi,cfg=cfg)
    kernel=max(1,int(cfg.get('blur_kernel',3)));kernel+=int(kernel%2==0)
    sigma=max(0.,float(cfg.get('blur_sigma',.55)))
    cur=current[y0:y1,x0:x1]
    if kernel>1:
        ref=cv2.GaussianBlur(ref,(kernel,kernel),sigma)
        cur=cv2.GaussianBlur(cur,(kernel,kernel),sigma)
    _,bias=engine._register_current(ref,ref,roi=roi,cfg=cfg,registration_bias=(0.,0.))
    aligned,registration=engine._register_current(ref,cur,roi=roi,cfg=cfg,
            registration_bias=(bias.get('raw_dx',0.),bias.get('raw_dy',0.)))
    norm,offset=engine._normalise_photometry(ref,aligned,roi=roi)
    delta=ref.astype(np.float32)-norm.astype(np.float32)
    absolute=np.abs(delta);dark=np.maximum(delta,0.)
    floor=max(.25,float(cfg.get('noise_floor',1.35)))
    z=absolute/np.clip(noise.astype(np.float32),floor,max(floor,float(cfg.get('noise_cap',28.))))
    dog=engine._multiscale_change_response(absolute,cfg)
    saliency=(float(cfg.get('weight_zscore',5.8))*z+float(cfg.get('weight_absdiff',.72))*absolute
        +float(cfg.get('weight_dog',1.75))*dog+float(cfg.get('weight_darkening',.32))*dark).astype(np.float32)
    margin=max(0,int(cfg.get('roi_edge_margin_px',5)))
    valid=(cv2.erode(roi,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(margin*2+1,margin*2+1))) if margin else roi)>0
    saliency[~valid]=0.
    if cfg.get('edge_penalty_enabled',True):saliency*=engine._edge_weights(ref,cfg)
    return dict(saliency=saliency,absdiff=absolute,darkening=dark,dog=dog,zscore=z,
        valid=valid,bbox=(x0,y0,x1,y1),cfg=cfg),dict(registration=registration,
        exposure_offset=float(offset),reference_stats=stats)


def temporal(signal, times=None, sequences=None):
    """Compatibility name; the shared evidence owner holds the frozen rule."""
    from src.engine.offline.clean_physical_change import temporal_step_evidence
    return temporal_step_evidence(signal,times,sequences)


def bounded_keep(candidates, ordinal, capacity=CAPACITY):
    """Pure bounded list operation; reuse existing exact-observation deduper.

    Nearby evidence is never averaged. If a stronger observed representative
    replaces another, its own same-XY signal and first observation travel with it.
    """
    deduped=region._dedupe_candidates(candidates,radius=3.5)
    eligible=[c for c in deduped if c['temporal']['supported'] or ordinal-c['born']<PENDING_FRAMES]
    eligible.sort(key=lambda c:(bool(c['temporal']['supported']),
        c['temporal']['support'] if c['temporal']['supported'] else c['v251_region_evidence'],
        c['score']),reverse=True)
    kept=eligible[:capacity];kept_ids={c['uid'] for c in kept}
    dedup_ids={c['uid'] for c in deduped};eligible_ids={c['uid'] for c in eligible}
    reasons={c['uid']:('spatial_representative_replaced' if c['uid'] not in dedup_ids else
        'unsupported_lifetime_expired' if c['uid'] not in eligible_ids else 'capacity')
        for c in candidates if c['uid'] not in kept_ids}
    return kept,reasons


def public(c):
    return {k:v for k,v in c.items() if k!='signal' and k!='timestamp' and not k.endswith('_ts')}


def descriptors(d02_sids=(2,4,5,6)):
    events=[]
    for sid in range(1,7):
        root=SESSION/'shots'/f'shot_{sid:08d}'
        paths=sorted((root/'frames').glob('pre_history_*.npy'))
        maskpath=sorted((root/'maps').glob('*_roi_polygon.npy'))[-1]
        events.append(dict(id=f'KOMP:{sid}',sid=sid,root=root,pre=[dict(path=p,ordinal=i,timestamp=None,sequence=None) for i,p in enumerate(paths)],
            mask=maskpath,post=[dict(path=p,timestamp=None) for p in sorted((root/'frames').glob('post_*.npy'))],trace=None,cutoff=None))
    for sid in d02_sids:
        root=D02/'shots'/f'shot_{sid:08d}'
        t=json.loads((root/'trace.json').read_text())
        pre=sorted((f for f in t['frames'] if f['kind']=='pre_history'),key=lambda f:f['timestamp'])
        cutoff=min(t['decision_input']['timestamp'],t['decision_input']['complete_track_audit']['decision_observed_at'],t.get('next_audio_peak_ts',math.inf))
        post=sorted((f for f in t['frames'] if f['kind']=='post' and f['timestamp']<=cutoff),key=lambda f:f['timestamp'])
        maskpath=root/next(s['evidence_maps']['roi_polygon']['path'] for s in reversed(t['stages']) if s.get('evidence_maps',{}).get('roi_polygon'))
        origins=[s['pipeline']['upstream']['crop_origin_camera'] for s in t['stages'] if isinstance(s.get('pipeline'),dict)]
        assert origins and all(tuple(o)==CROP[:2] for o in origins)
        events.append(dict(id=f'D02:{sid}',sid=sid,root=root,pre=[dict(path=root/f['path'],ordinal=i,timestamp=f['timestamp'],
            sequence=f.get('acquisition',{}).get('sequence')) for i,f in enumerate(pre)],mask=maskpath,
            post=[dict(path=root/f['path'],timestamp=f['timestamp']) for f in post],trace=root/'trace.json',cutoff=cutoff))
    return events


def crop(im):
    x,y,w,h=CROP
    return im[y:y+h,x:x+w]


def run_event(event,out,*,keep_fn=bounded_keep,witness_query=None,witness_batch_query=None):
    if witness_query is not None and witness_batch_query is not None:
        raise ValueError('Choose scalar or batch shared query')
    from src.engine.offline.common_verifier import camera_local_contrasts
    started=time.perf_counter();pre=event['pre']
    crop_box=tuple(event.get('crop',CROP))
    def event_crop(image):
        x,y,w,h=crop_box
        return image[y:y+h,x:x+w]
    if len(pre)>MAX_HISTORY:raise ValueError('History exceeds prespecified bound')
    mask=np.load(event['mask'],allow_pickle=False)
    assert mask.shape==(crop_box[3],crop_box[2])
    bx,by,bw,bh=cv2.boundingRect(mask)
    group=region.RegionGroupV251('recorded_search_region',('recorded_search_region',),(),bx,by,bx+bw-1,by+bh-1)
    engine=CandidateGeneratorV2();scanner=SimpleNamespace()
    history=[];active=[];timings=[];max_active=0;max_working=0;max_bytes=0;generated=0;witnesses=0
    times=None if pre[0]['timestamp'] is None else [f['timestamp'] for f in pre]
    seq=None if times is None else [f['sequence'] for f in pre]
    if seq is not None and all(s is None for s in seq):seq=None
    eventfile=out/(event['id'].replace(':','_')+'_transitions.jsonl')
    with eventfile.open('x') as f:
        for i,frame in enumerate(pre):
            begin=time.perf_counter();raw=np.load(frame['path'],mmap_mode='r',allow_pickle=False)
            history.append(event_crop(raw).copy());del raw
            load_ms=(time.perf_counter()-begin)*1000
            if i<3:continue
            begin=time.perf_counter();maps,diag=transition_maps(engine,history[i-3:i],history[i],mask)
            valid=region._bbox_region_mask(maps['saliency'].shape,maps['valid'],maps['bbox'],group)
            threshold,_=engine._robust_threshold(maps['saliency'],valid=valid,cfg=maps['cfg'])
            candidates,telemetry=region._region_physical_candidates(engine,scanner,group,
                **{k:maps[k] for k in ('saliency','absdiff','darkening','dog','zscore','bbox','cfg')},
                region_valid=valid,region_threshold=threshold,frame_ts=float('nan'),shot_id=event['sid'])
            candidates=region._balance_merged_candidates(candidates,(group,),shot_id=event['sid'])
            assert len(candidates)<=8
            proposal_ms=(time.perf_counter()-begin)*1000
            begin=time.perf_counter();working=[dict(c,signal=list(c['signal'])) for c in active]
            if witness_batch_query is not None:
                values=camera_local_contrasts(history[-1],[(c['camera_x'],c['camera_y']) for c in working])
                for c,value in zip(working,values):c['signal'].append(float(value))
            else:
                for c in working:
                    c['signal'].append(float(local_contrast(sample(history[-1],[(c['camera_x'],c['camera_y'])]))[0]))
            generated+=len(candidates);coord_index={(c['camera_x'],c['camera_y']):c for c in working}
            proposed=[];new_candidates=[]
            for j,original in enumerate(candidates):
                c=clean(original);x,y=c['camera_x'],c['camera_y']
                c.update(camera_xy=[x+crop_box[0],y+crop_box[1]],source_ordinal=i,source_rank=j+1,
                         reference_ordinals=[i-3,i-2,i-1])
                proposal_record=dict(c)
                if x<25 or y<25 or x+27>history[-1].shape[1] or y+27>history[-1].shape[0]:
                    proposed.append(dict(c,query_unavailable='patch_outside_saved_crop'));continue
                old=coord_index.get((x,y))
                if old is None:
                    c.update(uid=f'{i}:{j}',born=i,last_proposed=i,observations=1,
                        signal=[] if witness_batch_query is not None else [float(local_contrast(sample(im,[(x,y)]))[0]) for im in history])
                    working.append(c);coord_index[(x,y)]=c
                    new_candidates.append(c)
                else:
                    old['last_proposed']=i;old['observations']+=1
                    # Exact XY association keeps its earliest identity. Existing
                    # scores only improve when another actual proposal supplies them.
                    if c['v251_region_evidence']>old['v251_region_evidence']:
                        for key in c:
                            if key not in ('source_ordinal','reference_ordinals','source_rank'):old[key]=c[key]
                    c=old
                # Journal the CURRENT proposal, not its carried representative's
                # earlier score/reference. Association is a separate identity.
                proposed.append(dict(proposal_record,association_uid=c['uid']))
            if witness_batch_query is not None and new_candidates:
                points=[(c['camera_x'],c['camera_y']) for c in new_candidates]
                for im in history:
                    for c,value in zip(new_candidates,camera_local_contrasts(im,points)):
                        c['signal'].append(float(value))
            extraction_ms=(time.perf_counter()-begin)*1000
            query_start=time.perf_counter()
            query_profile={}
            proofs=None
            if witness_batch_query is not None:
                proofs,query_profile=witness_batch_query(working,event,pre[:i+1])
                if len(proofs)!=len(working):raise ValueError('Batch must preserve all candidate identities')
            for j,c in enumerate(working):
                if proofs is not None:
                    c['physical_witness']=proofs[j];c['temporal']=proofs[j]['measurements']
                elif witness_query is None:
                    c['temporal']=temporal(c['signal'],times[:i+1] if times is not None else None,seq[:i+1] if seq is not None else None)
                else:
                    proof=witness_query(c,event,pre[:i+1])
                    c['physical_witness']=proof
                    c['temporal']=proof['measurements']
            temporal_ms=(time.perf_counter()-query_start)*1000
            retain_start=time.perf_counter()
            active,reasons=keep_fn(working,i)
            retention_ms=(time.perf_counter()-retain_start)*1000
            journal_start=time.perf_counter()
            witnesses+=sum(c['temporal']['supported'] for c in working)
            row=dict(ordinal=i,frame=frame['path'].name,receipt_timestamp=frame['timestamp'],camera_sequence=frame['sequence'],
                reference_ordinals=[i-3,i-2,i-1],proposal_telemetry=telemetry,physical_maps=diag,
                generated=proposed,evaluated=[public(c) for c in working],retained=[c['uid'] for c in active],retired=reasons)
            f.write(json.dumps(row,allow_nan=False)+'\n')
            journal_ms=(time.perf_counter()-journal_start)*1000
            query_ms=(time.perf_counter()-begin)*1000
            livebytes=sum(im.nbytes for im in history)+sum(v.nbytes for v in maps.values() if isinstance(v,np.ndarray))
            livebytes+=sum(len(c['signal'])*8 for c in working)
            max_bytes=max(max_bytes,livebytes);max_active=max(max_active,len(active));max_working=max(max_working,len(working))
            timings.append(dict(ordinal=i,load_ms=load_ms,proposal_ms=proposal_ms,evidence_retention_and_journal_ms=query_ms,
                extraction_ms=extraction_ms,temporal_ms=temporal_ms,retention_ms=retention_ms,journal_ms=journal_ms,
                shared_query=query_profile,candidates=len(working),new_candidates=len(new_candidates),history_frames=len(history)))
    supported=sorted((c for c in active if c['temporal']['supported']),key=lambda c:-c['temporal']['support'])
    selected=public(supported[0]) if supported else None
    # Primary decision is frozen before any POST is opened.
    result=dict(event=event['id'],status='OFFLINE_ORDINAL_RECONSTRUCTION' if times is None else 'OFFLINE_RECEIPT_CAUSAL_RECONSTRUCTION',
        primary_cutoff_ordinal=len(pre)-1,primary_cutoff_receipt=None if times is None else times[-1],
        primary_state='PHYSICAL_HIT_HYPOTHESIS' if selected else 'UNCERTAIN',selected=selected,
        supported=[public(c) for c in supported],retained=[public(c) for c in active],generated=generated,
        supported_prefix_observations=witnesses,max_retained=max_active,max_transient=max_working,
        accounted_peak_array_bytes=max_bytes,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        timings=timings,total_pre_work_ms=(time.perf_counter()-started)*1000)
    write(out/(event['id'].replace(':','_')+'_frozen.json'),result)
    post_checks=[]
    for c in supported:
        values=[float(local_contrast(sample(event_crop(np.load(p['path'],mmap_mode='r',allow_pickle=False)),[(c['camera_x'],c['camera_y'])]))[0]) for p in event['post']]
        t=c['temporal'];base=c['signal'][t['pre_ordinal']];direction=np.sign(t['signed_change'])
        support=[float(direction*(v-base)/t['noise']) for v in values]
        post_checks.append(dict(uid=c['uid'],camera_xy=c['camera_xy'],frames=len(values),
            minimum_support=min(support) if support else None,all_above_frozen_3sigma=bool(support and min(support)>=3),
            values=values,post_scope='RETROSPECTIVE_TIMING_UNKNOWN' if times is None else 'AT_OR_BEFORE_RECORDED_DECISION',
            changes_primary_selection=False))
    write(out/(event['id'].replace(':','_')+'_post_checks.json'),post_checks)
    return result


def evaluation(out,events,results):
    """This is the first place labels and CURRENT results enter the experiment."""
    recovered=json.loads((BASE/'kompetensprov_recovery_20260927_01/recovered_results.json').read_text())['events']
    assignments=json.loads((D02/'physical_assignments.json').read_text())
    rows=[]
    for event,result in zip(events,results):
        gt=None
        if event['id'].startswith('KOMP'):
            label=json.loads((event['root']/'ground_truth.json').read_text());gt=[label['camera_x'],label['camera_y']]
            old=recovered[event['sid']-1];current=dict(xy=[old['emission_record']['camera_x'],old['emission_record']['camera_y']],error=old['error_px'])
        else:
            assert assignments[str(event['sid'])]['state']=='NO_PHYSICAL_SHOT'
            current=dict(false_accept=True)
        records=[json.loads(line) for line in (out/(event['id'].replace(':','_')+'_transitions.jsonl')).read_text().splitlines()]
        proposals=[dict(c,transition=r['ordinal']) for r in records for c in r['generated']]
        supported=[dict(c,discovered_at=r['ordinal'],retained_now=c['uid'] in r['retained'],retirement=r['retired'].get(c['uid']))
            for r in records for c in r['evaluated'] if c['temporal']['supported']]
        def closest(cs):
            if gt is None or not cs:return None
            c=min(cs,key=lambda c:math.dist(c['camera_xy'],gt));return dict(candidate=c,error=math.dist(c['camera_xy'],gt))
        chosen=result['selected'];error=math.dist(chosen['camera_xy'],gt) if chosen and gt else None
        ranked=[dict(rank=i+1,candidate=c,error=math.dist(c['camera_xy'],gt) if gt else None) for i,c in enumerate(result['supported'])]
        first=supported[0] if supported else None
        earliest_by_radius={str(r):next((c for c in supported if math.dist(c['camera_xy'],gt)<=r),None) if gt else None for r in RADII}
        rows.append(dict(event=event['id'],gt=gt,current=current,selected=chosen,selected_error=error,
            selected_at={str(r):error is not None and error<=r for r in RADII},generated_nearest=closest(proposals),
            supported_nearest=closest(supported),retained_nearest=closest(result['retained']),first_supported=first,
            earliest_supported_by_radius=earliest_by_radius,ranked=ranked,any_supported_prefix=bool(supported),
            supported_unique_xy=len({tuple(c['camera_xy']) for c in supported}),final_supported=len(result['supported']),
            proposal_count=len(proposals),retained_count=len(result['retained'])))
    summary=dict(selected_at={str(r):sum(row['selected_at'][str(r)] for row in rows[:6]) for r in RADII},
        oracle_at={str(r):sum(row['generated_nearest'] is not None and row['generated_nearest']['error']<=r for row in rows[:6]) for r in RADII},
        real_supported_at={str(r):sum(row['supported_nearest'] is not None and row['supported_nearest']['error']<=r for row in rows[:6]) for r in RADII},
        emissions=sum(row['selected'] is not None for row in rows[:6]),uncertain=sum(row['selected'] is None for row in rows[:6]),
        false_final=sum(row['final_supported']>0 for row in rows[6:]),false_any_prefix=sum(row['any_supported_prefix'] for row in rows[6:]),
        protected_shot6=rows[5]['selected_at']['5'],live_authority_changed=False)
    write(out/'evaluation.json',dict(summary=summary,events=rows))
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(BASE) or out.is_relative_to(RUN) or 'S03' in str(out).upper():raise ValueError('Fresh research output required')
    out.mkdir(parents=True,exist_ok=False);accesses=read_guard(out,[SESSION,D02]);cv2.setNumThreads(1)
    events=descriptors();settings=json.loads(Path('content/ai/settings.json').read_text())
    sources=[Path(__file__),Path('automation/kompetensprov_selection_research.py'),Path('automation/causal_pre_onset_research.py'),
        Path('src/engine/camera/candidate_generator_v2.py'),Path(region.__file__),Path('src/engine/offline/clean_physical_change.py'),
        Path('content/ai/detector_v2.json'),Path('content/ai/settings.json')]
    write(out/'configuration.json',dict(capacity=CAPACITY,pending_lifetime_frames=PENDING_FRAMES,max_history=MAX_HISTORY,
        detector_config=CandidateGeneratorV2().config.snapshot(),sources={str(p):digest(p) for p in sources}))
    write(out/'input_frames.json',[dict(event=e['id'],pre=[{**f,'path':str(f['path'])} for f in e['pre']],
        post=[{**f,'path':str(f['path'])} for f in e['post']],mask=str(e['mask']),decision_cutoff=e['cutoff']) for e in events])
    results=[]
    with mock.object(region,'_runtime_settings',return_value=settings):
        assert region._per_region_proposals()==8,'Prespecified current cap must be eight'
        for event in events:
            result=run_event(event,out);results.append(result)
            print(event['id'],'generated',result['generated'],'retained',len(result['retained']),
                  'supported',len(result['supported']),'ms',round(result['total_pre_work_ms'],1),flush=True)
    print(json.dumps(evaluation(out,events,results)),flush=True)
    input_paths={str(f['path']) for e in events for f in e['pre']+e['post']}
    input_paths.update(str(e['mask']) for e in events);input_paths.update(str(e['trace']) for e in events if e['trace'])
    input_paths.update(str(e['root']/'ground_truth.json') for e in events[:6]);input_paths.add(str(D02/'physical_assignments.json'))
    write(out/'input_hashes.json',{p:digest(p) for p in sorted(input_paths)})
    write(out/'read_accesses.json',sorted(accesses))


if __name__=='__main__':main()
