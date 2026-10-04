"""GT-isolated raw-information diagnostics. No training or runtime authority.

Ranks are over the frozen native crop, not all 3840x2160 camera pixels. Choosing
best transitions at GT is explicitly a retrospective diagnostic upper bound.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import time
import cv2
import numpy as np
from automation import physical_ai_dataset as ds, physical_ai_vision as vision, physical_ai_vision_peaks as peaks
from src.engine.offline import direct_proposal_v221 as direct
from src.engine.offline.common_verifier import camera_local_contrasts
from src.engine.offline.clean_physical_change import temporal_step_evidence
from src.engine.ai.training_v223.evidence_patch_v2235 import build_registered_evidence_channels, EVIDENCE_CHANNEL_NAMES
from src.engine.ai.training_v223.heatmap_model_v2236 import HeatmapModelV2236

HARD=('POST_FIX:10','S01:3','S01:5','S02:4','S02:6')
PROTECTED=('KOMP:1','KOMP:5','S02:5','S02:11','S02:12','S02:13','POST_FIX:3')
CACHE=Path('/data/skjutbana/physical_ai_lab/datasets/vision_proposals_v1_20261003')
BASE=Path('/data/skjutbana/physical_ai_lab/physical_vision_v1_1h/trials/trial_000044')


def cohort(events):
    chosen=set(HARD+PROTECTED);easy=[]
    for session in sorted({e['session'] for e in events}):
        options=sorted((e for e in events if e['session']==session and e['event'] not in chosen and e['truth']=='HIT'
                        and not e.get('caveats') and e['baseline_xy'] is not None and ds.distance(e['baseline_xy'],e['gt_xy'])<=5),
                       key=lambda e:int(e['event'].split(':')[1]))
        if options:easy.append(options[0]['event'])
    return list(HARD)+list(PROTECTED)+easy


def validate_frames(frames,cutoff):
    if not frames or len({f['frame_id'] for f in frames})!=len(frames):raise ValueError('Unique nonempty frame identities')
    for f in frames:ds.guard(f['frame_id'])
    if any(a['ordinal']>=b['ordinal'] for a,b in zip(frames,frames[1:])):raise ValueError('Ordinal order')
    seq=[f.get('camera_sequence') for f in frames]
    if all(s is not None for s in seq) and any(a>=b for a,b in zip(seq,seq[1:])):raise ValueError('Camera sequence order')
    times=[f['receipt_timestamp'] for f in frames]
    if all(t is None for t in times):
        if cutoff is not None:raise ValueError('No fabricated ordinal clock')
        return
    if any(t is None or not np.isfinite(t) for t in times) or cutoff is None:raise ValueError('Mixed/missing receipt provenance')
    if any(a>=b for a,b in zip(times,times[1:])) or times[-1]>cutoff:raise ValueError('Receipt order/cutoff violation')


def available_frames(event):
    """Explicit event only. Original primary prefix plus recorded pre-decision POST.

    No discovery/glob, guessed exposure times or unclocked KOMP POST inclusion.
    Snapshot duplicates are not independent observations; chronological originals
    are authoritative and all excluded snapshot metadata is retained.
    """
    primary=[dict(f,kind='pre_history',primary=True) for f in event['frames']]
    validate_frames(primary,event['cutoff_receipt'])
    meta=dict(primary_cutoff=event['cutoff_receipt'],primary_count=len(primary),decision_cutoff=None,
              excluded_post=0,excluded_snapshots=[],timestamp_semantics='RECEIPT_NOT_EXPOSURE',clock_status=event['status'])
    if not event['trace_available']:
        meta['extension_unavailable']='NO_RECORDED_POST_CLOCK_OR_DECISION_CUTOFF';return primary,meta
    trace_path=Path(event['source_root'])/'trace.json';trace=ds.read(trace_path)
    recorded_history={str(Path(event['source_root'])/f['path']) for f in trace['frames'] if f['kind']=='pre_history'}
    if recorded_history!={f['frame_id'] for f in primary}:raise ValueError('Frozen history omits recorded PRE observations')
    decision=trace.get('decision_input',{});audit=decision.get('complete_track_audit',{})
    cutoffs={k:v for k,v in dict(decision_timestamp=decision.get('timestamp'),decision_observed_at=audit.get('decision_observed_at'),next_audio=trace.get('next_audio_peak_ts')).items() if v is not None}
    if not cutoffs or decision.get('timestamp') is None:meta['extension_unavailable']='MISSING_DECISION_TIMESTAMP';return primary,meta
    cutoff=min(cutoffs.values());meta.update(decision_cutoff=cutoff,cutoff_components=cutoffs,trace_path=str(trace_path),trace_hash=ds.digest(trace_path),peak_timestamp=trace.get('peak_ts'))
    if primary[-1]['receipt_timestamp']>cutoff:raise ValueError('Primary prefix exceeds decision')
    extra=[]
    for f in trace['frames']:
        if f['kind']=='pre_snapshot':
            match=[p for p in primary if p['receipt_timestamp']==f.get('timestamp')]
            path=Path(event['source_root'])/f['path']
            if len(match)!=1 or ds.digest(path)!=ds.digest(match[0]['frame_id']):
                raise ValueError('Snapshot is not an identical history observation; association audit needed')
            meta['excluded_snapshots'].append(dict(f,duplicate_of=match[0]['frame_id'],sha256=ds.digest(path)));continue
        if f['kind']!='post':continue
        timestamp=f.get('timestamp')
        if timestamp is None or timestamp>cutoff:meta['excluded_post']+=1;continue
        if timestamp<=primary[-1]['receipt_timestamp']:raise ValueError('POST overlaps history: explicit association audit required')
        path=Path(event['source_root'])/f['path'];ds.guard(path)
        extra.append(dict(frame_id=str(path),receipt_timestamp=timestamp,kind='post',primary=False,
                          camera_sequence=f.get('acquisition',{}).get('sequence'),recorded_order=f.get('order')))
    extra.sort(key=lambda f:f['receipt_timestamp'])
    for i,f in enumerate(extra,start=len(primary)):f['ordinal']=i
    frames=primary+extra;validate_frames(frames,cutoff)
    if len(frames)>100:raise ValueError('Audit history bound100 exceeded')
    meta['included_post']=len(extra)
    return frames,meta


def transition_maps(before,after,registered=False):
    """Image-only measurements. Positive signed delta is brightening."""
    reg={};current=after
    if registered:current,reg=direct._register(before,after,direct.DirectProposalConfigV221())
    delta=current.astype(np.float32)-before.astype(np.float32)
    y,x=np.mgrid[-12:13,-12:13];r2=x*x+y*y
    kernel=(r2<=16).astype(np.float32);kernel/=kernel.sum()
    ring=((r2>=64)&(r2<=144)).astype(np.float32);ring/=ring.sum()
    contrast=cv2.filter2D(delta,cv2.CV_32F,kernel-ring,borderType=cv2.BORDER_REFLECT_101)
    def gradient(im):return cv2.magnitude(cv2.Sobel(im,cv2.CV_32F,1,0),cv2.Sobel(im,cv2.CV_32F,0,1))
    return dict(dark=np.maximum(-delta,0),bright=np.maximum(delta,0),absolute=np.abs(delta),
                contrast=np.abs(contrast),gradient=np.maximum(gradient(current)-gradient(before),0)),reg


def capture_direct(pre,posts,config):
    """Expose existing function locals at return; do not reimplement preprocessing."""
    captured={};previous=sys.getprofile()
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is direct.build_direct_maps_v221.__code__:
            for key in ('raw_maps','signed','reference','post_stack'):captured[key]=frame.f_locals[key]
    sys.setprofile(profile)
    try:result=direct.build_direct_maps_v221([pre],posts,config=config)
    finally:sys.setprofile(previous)
    if not captured:raise RuntimeError('Missing direct-map diagnostic capture')
    return result,captured


def stage_maps(images):
    """Frozen first3/last3 ladder, produced without coordinate/truth inputs."""
    pre=np.median(np.stack(images[:3]),axis=0).astype(np.uint8);posts=images[-3:]
    out={};metadata={}
    for name,config in [('2_group_raw',direct.DirectProposalConfigV221(pre_blur_sigma=0,registration_enabled=False,photometric_compensation=False)),
                        ('2_group_registered',direct.DirectProposalConfigV221(pre_blur_sigma=0)),
                        ('3_filtered',direct.DirectProposalConfigV221())]:
        result,raw=capture_direct(pre,posts,config)
        out[name]=raw['raw_maps'];metadata[name]=result[-1]
        if name=='3_filtered':
            normalized=dict(result[2],fused=result[3]);out['4_unit_float']=normalized
            f32={};u8={}
            from src.engine.ai.training_v223.evidence_patch_v2235 import _normalise_map
            for k,a in normalized.items():
                u8[k],stats=_normalise_map(a)
                f32[k]=np.clip((a-stats['q50'])/max(stats['q995']-stats['q50'],1e-6),0,2)*127.5
            out['4_scaled_float']=f32;out['5_uint8']=u8
    return out,metadata


def rank_map(score,gt_local,origin):
    """GT DIAGNOSTIC ONLY: never called by representation constructors.

    Stable raw-max rank can be tie-sensitive on clipped/quantized maps. Strict
    greater rank and tied range are reported to prevent false collapse claims.
    """
    a=np.asarray(score,np.float32);xy,values=peaks.raw_peaks(a)
    distance=np.linalg.norm(xy-np.asarray(gt_local),axis=1);good=np.flatnonzero(distance<=5)
    yy,xx=np.mgrid[max(0,int(gt_local[1])-6):min(a.shape[0],int(gt_local[1])+7),max(0,int(gt_local[0])-6):min(a.shape[1],int(gt_local[0])+7)]
    near=(xx-gt_local[0])**2+(yy-gt_local[1])**2<=25
    best=float(a[yy[near],xx[near]].max());j=int(good[0]) if len(good) else None
    value=None if j is None else float(values[j]);p99=float(np.percentile(a,99))
    accepted,_=peaks.nms_trace(xy,6,32);hit=[i+1 for i,k in enumerate(accepted) if distance[k]<=5]
    top=[dict(camera_xy=(xy[k]+origin).tolist(),score=float(values[k]),gt_distance=float(distance[k])) for k in accepted[:3]]
    stronger=np.flatnonzero(values>best);nearest=int(stronger[np.argmin(distance[stronger])]) if len(stronger) else None
    return dict(best_local=best,pixel_percentile=float(np.mean(a<best)*100),score_over_global_p99=best/p99 if p99>0 else None,
                raw_rank=None if j is None else j+1,strict_rank=None if j is None else int(np.sum(values>value))+1,
                tie_rank_end=None if j is None else int(np.sum(values>=value)),peak_score=value,
                peak_xy=None if j is None else (xy[j]+origin).tolist(),peak_error=None if j is None else float(distance[j]),
                nms_top32_rank=min(hit) if hit else None,top3=top,
                nearest_stronger=None if nearest is None else dict(raw_rank=nearest+1,camera_xy=(xy[nearest]+origin).tolist(),distance=float(distance[nearest]),score=float(values[nearest])),
                global_mean=float(np.mean(a)),global_p99=p99,global_nonzero_fraction=float(np.mean(a>0)),maxima_count=len(values))


def flatten(metric):
    return {k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in metric.items()}


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([flatten(r) for r in rows])


def persistent_maps(images,times=None):
    """GT-blind INFORMATION PROBE, not a new verifier.

    Max over consecutive transitions of the minimum same-sign change remaining
    for three observations against that transition's preceding observation.
    Uses no physical clean-PRE claim. Spatially different pixels may use different
    transitions. Known >250ms gaps (existing temporal guard) invalidate a window.
    """
    if len(images)<4:raise ValueError('Need four observed frames')
    if times is not None and (len(times)!=len(images) or any(a>=b for a,b in zip(times,times[1:]))):raise ValueError('Ordered receipt times required')
    dark=np.zeros(images[0].shape,np.float32);bright=dark.copy();windows=[]
    for k in range(1,len(images)-2):
        if times is not None and max(np.diff(times[k-1:k+3]))>.25:continue
        delta=np.stack([images[j].astype(np.float32)-images[k-1] for j in range(k,k+3)])
        np.maximum(dark,np.maximum(np.min(-delta,axis=0),0),out=dark)
        np.maximum(bright,np.maximum(np.min(delta,axis=0),0),out=bright)
        windows.append(dict(before=k-1,onset=k,available=k+2))
    return dict(dark=dark,bright=bright,unsigned=np.maximum(dark,bright)),windows


def probe(output,cache=CACHE):
    """One preregistered persistence probe + existing decision-cutoff grouping.

    Dataset GT is passed only to rank_map / reporting. Exact baseline coordinates
    already present in the frozen dataset are separately queried, never relabelled.
    """
    ds.install_guard();output=ds.output_root(output);cv2.setNumThreads(1)
    if not (output/'summary.json').exists():raise ValueError('Finish raw ladder first')
    destination=output/'probes'
    if (destination/'results.json').exists():raise FileExistsError('Preserve completed probe')
    destination.mkdir(exist_ok=True)
    ds.atomic_json(destination/'configuration.json',dict(pool=32,nms_radius=6,post_observations=3,max_receipt_gap=.25,
        variants=['max_persistent_signed_change','first3_decision_last3'],no_training=True,no_selected_accuracy_claim=True))
    data,cache=vision.load(cache);rows=[];witnesses={};baseline={}
    for e in data['events']:
        if e['event'] not in cohort(data['events']):continue
        eid=e['event'];r=data['vision_rows'][eid];origin=np.array(r['origin']);gt=np.array(e['gt_xy'])-origin
        f=ds.read(output/eid.replace(':','_')/'frames.json');frames=f['frames'];n=f['provenance']['primary_count']
        x,y=origin;h,w=r['shape'];images=[np.array(np.load(a['frame_id'],mmap_mode='r',allow_pickle=False)[y:y+h,x:x+w],copy=True) for a in frames]
        witnesses[eid]={}
        for scope,end in [('primary',n),('decision',len(images))]:
            ts=[a['receipt_timestamp'] for a in frames[:end]];ts=None if ts[0] is None else ts
            maps,windows=persistent_maps(images[:end],ts)
            for family,a in maps.items():rows.append(dict(event=eid,scope=scope,variant='persistent_'+family,windows=len(windows),**rank_map(a,gt,origin)))
            # Fixed positive-valued top32 unsigned peaks, never GT-centred probes.
            xy,values=peaks.raw_peaks(maps['unsigned']);accepted,_=peaks.nms_trace(xy,6,32)
            xy=xy[[i for i in accepted if values[i]>0]]
            # Preserve full-camera context at crop-border proposals. The raw
            # source is available; do not reject/refill peaks to hide boundaries.
            signal=np.array([camera_local_contrasts(np.load(f['frame_id'],mmap_mode='r',allow_pickle=False),xy+origin) for f in frames[:end]])
            records=[]
            for i,point in enumerate(xy):
                evidence=temporal_step_evidence(signal[:,i],ts)
                records.append(dict(camera_xy=(point+origin).tolist(),proposal_rank=i+1,evidence=evidence,
                                    diagnostic_gt_distance=float(np.linalg.norm(point-gt))))
            witnesses[eid][scope]=dict(windows=windows,proposals=records)
        if len(images)>n:
            ladder,_=stage_maps(images)
            for stage in ['3_filtered','5_uint8']:
                for family in ['persistent_dark','persistent_bright','persistent_abs']:
                    rows.append(dict(event=eid,scope='decision',variant='decision_group_'+stage+'_'+family,**rank_map(ladder[stage][family],gt,origin)))
        if e.get('baseline_xy') is not None:
            point=np.array(e['baseline_xy'])-origin;signal=np.array([camera_local_contrasts(im,[point])[0] for im in images])
            ts=[a['receipt_timestamp'] for a in frames];ts=None if ts[0] is None else ts
            baseline[eid]=dict(camera_xy=e['baseline_xy'],source='unchanged frozen witness dataset baseline_xy; NOT corrected GT',
                gt_xy=e['gt_xy'],distance=ds.distance(e['baseline_xy'],e['gt_xy']),signal=signal.tolist(),
                primary_evidence=temporal_step_evidence(signal[:n],None if ts is None else ts[:n]),
                decision_evidence=temporal_step_evidence(signal,ts))
        print('probe',eid,flush=True)
    ds.atomic_json(destination/'results.json',rows);write_csv(destination/'counterfactual_results.csv',rows)
    ds.atomic_json(destination/'bounded_witnesses.json',witnesses);ds.atomic_json(destination/'existing_coordinate_diagnostics.json',baseline)
    association_diagnostics(data,cache,output)


def association_diagnostics(data,cache,output):
    """Evaluate unchanged baseline points in hard events; never correct labels."""
    result={}
    for e in data['events']:
        eid=e['event']
        if eid not in HARD or e.get('baseline_xy') is None:continue
        r=data['vision_rows'][eid];origin=np.array(r['origin']);x,y=origin;h,w=r['shape']
        images=[np.array(np.load(f['frame_id'],mmap_mode='r',allow_pickle=False)[y:y+h,x:x+w],copy=True) for f in e['frames']]
        stages,_=stage_maps(images);point=np.array(e['baseline_xy'])-origin
        fold=next(i for i,f in enumerate(data['folds']) if e['session'] in f['held'])
        model=HeatmapModelV2236.load(BASE/f'fold_{fold:02d}')
        if e['session'] in model.metadata['normalization']['training_sessions']:raise ValueError('Frozen fold leakage')
        heat=model.score_map(vision.normalize(np.load(cache/r['path'],allow_pickle=False),model.metadata['normalization']))
        label=Path(e['source_root'])/'ground_truth.json'
        result[eid]=dict(baseline_xy=e['baseline_xy'],gt_xy=e['gt_xy'],
            stage_ranks={k:rank_map(stages[k]['persistent_dark'],point,origin) for k in ('2_group_raw','3_filtered','4_unit_float','5_uint8')},
            heatmap=rank_map(heat,point,origin),label_record=ds.read(label),label_sha256=ds.digest(label))
    ds.atomic_json(output/'existing_coordinate_stage_comparison.json',result)


def montage(images,frames,gt_local,path):
    size=96;cols=8;tile=128;canvas=np.full(((len(images)+cols-1)//cols*tile,cols*tile,3),235,np.uint8)
    for i,(im,f) in enumerate(zip(images,frames)):
        patch=cv2.getRectSubPix(im,(size,size),tuple(map(float,gt_local)))
        tileim=cv2.cvtColor(patch,cv2.COLOR_GRAY2BGR);tileim=cv2.resize(tileim,(tile,tile-24),interpolation=cv2.INTER_NEAREST)
        x=(i%cols)*tile;y=(i//cols)*tile;canvas[y+24:y+tile,x:x+tile]=tileim
        cv2.putText(canvas,f"{f['ordinal']} {f['kind']}",(x+3,y+17),cv2.FONT_HERSHEY_SIMPLEX,.39,(0,0,0),1)
    cv2.imwrite(str(path),canvas)


def run(output,cache=CACHE,resume=False):
    ds.install_guard();output=ds.output_root(output);output.mkdir(parents=True,exist_ok=resume);cv2.setNumThreads(1)
    data,cache=vision.load(cache);ids=cohort(data['events']);events={e['event']:e for e in data['events']}
    ds.atomic_json(output/'prespecified.json',dict(events=ids,hard=HARD,protected=PROTECTED,
        easy_rule='first numeric event/session, HIT, no caveat, baseline selected <=5, not hard/protected',
        radius=5,nms_radius=6,pool=32,primary_map='dark',maps=['dark','bright','absolute','contrast','gradient'],
        scope='frozen PRE-history plus separate recorded-decision extension; no unknown-clock POST',
        diagnostic_best_transition_is_not_inference=True,dataset_hash=data['dataset_hash'],code_revision=ds.get_git_commit()))
    old_hashes=ds.read(output/'input_hashes.json') if resume and (output/'input_hashes.json').exists() else {}
    hashes={};alltrans=[];allstages=[];summaries=[]
    for eid in ids:
        start=time.perf_counter();e=events[eid];row=data['vision_rows'][eid];origin=np.array(row['origin']);gt=np.array(e['gt_xy'])-origin
        frames,provenance=available_frames(e);images=[]
        for f in frames:
            p=Path(f['frame_id']);h=ds.digest(p);hashes[str(p)]=h
            if str(p) in old_hashes and old_hashes[str(p)]!=h:raise ValueError('Resume source pixels changed')
            if str(p) in data['source_hashes'] and data['source_hashes'][str(p)]!=h:raise ValueError('Source pixels changed')
            x,y=origin;hh,ww=row['shape'];im=np.array(np.load(p,mmap_mode='r',allow_pickle=False)[y:y+hh,x:x+ww],copy=True)
            if im.dtype!=np.uint8 or im.ndim!=2:raise ValueError('Native grayscale only')
            images.append(im)
        n=provenance['primary_count'];folder=output/eid.replace(':','_')
        if resume and (folder/'summary.json').exists():
            alltrans+=ds.read(folder/'transitions.json');allstages+=ds.read(folder/'stages.json');summaries.append(ds.read(folder/'summary.json'));continue
        folder.mkdir(exist_ok=resume)
        ds.atomic_json(folder/'frames.json',dict(frames=frames,provenance=provenance,origin=origin.tolist(),shape=row['shape']))
        signal=np.array([camera_local_contrasts(im,[gt])[0] for im in images])
        pixel=np.array([float(cv2.getRectSubPix(im,(1,1),tuple(map(float,gt)))[0,0]) for im in images])
        tr=[]
        for k in range(1,len(images)):
            scope='primary' if k<n else 'decision_extension'
            for registered in (False,True):
                maps,reg=transition_maps(images[k-1],images[k],registered)
                for family,a in maps.items():
                    r=rank_map(a,gt,origin)
                    r.update(event=eid,scope=scope,transition_from=k-1,transition_to=k,family=family,registered=registered,
                        frame_from=frames[k-1]['frame_id'],frame_to=frames[k]['frame_id'],receipt_from=frames[k-1]['receipt_timestamp'],receipt_to=frames[k]['receipt_timestamp'],
                        registration=reg,contrast_change=float(signal[k]-signal[k-1]),pixel_change=float(pixel[k]-pixel[k-1]),
                        subsequent_contrast_relative_before=(signal[k:min(len(images),k+3)]-signal[k-1]).tolist(),
                        support_available_ordinal=min(len(images)-1,k+2),complete_three_after=k+2<len(images))
                    tr.append(r)
        temporal={}
        for scope,end in [('primary',n),('decision',len(images))]:
            fs=frames[:end];ts=[f['receipt_timestamp'] for f in fs];seq=[f.get('camera_sequence') for f in fs]
            temporal[scope]=temporal_step_evidence(signal[:end],None if ts[0] is None else ts,None if any(s is None for s in seq) else seq)
        ladder,metadata=stage_maps(images[:n]);cached=np.load(cache/row['path'],mmap_mode='r',allow_pickle=False)
        rebuilt=np.stack([ladder['5_uint8'][name] for name in EVIDENCE_CHANNEL_NAMES])
        difference=rebuilt.astype(np.int16)-cached.astype(np.int16)
        metadata['cache_parity']=dict(exact=bool(np.array_equal(cached,rebuilt)),differing_values=int(np.count_nonzero(difference)),max_abs=int(np.abs(difference).max()),
            per_channel={name:int(np.count_nonzero(difference[i])) for i,name in enumerate(EVIDENCE_CHANNEL_NAMES)})
        if np.any(difference):
            # Diagnose rather than silently replacing frozen input. The original
            # builder itself reproduces the observed single-LSB fused-map drift;
            # the frozen model uses channels 2/4/5/7, all exactly unchanged.
            original,_=build_registered_evidence_channels(np.median(np.stack(images[:n][:3]),axis=0).astype(np.uint8),images[:n][-3:])
            assert np.array_equal(original,rebuilt),(eid,'diagnostic builder mismatch')
            assert not np.any(difference[[2,4,5,7]]) and np.abs(difference).max()<=1,(eid,'active input mismatch')
            metadata['cache_parity']['note']='Original builder drift, <=1 LSB on unused channels; frozen active channels exact. Stage5 and heatmap use unchanged cache.'
            ladder['5_uint8']={name:cached[i] for i,name in enumerate(EVIDENCE_CHANNEL_NAMES)}
        stages=[]
        for stage,maps in ladder.items():
            for family,a in maps.items():stages.append(dict(event=eid,stage=stage,family=family,**rank_map(a,gt,origin)))
        fold=next(i for i,f in enumerate(data['folds']) if e['session'] in f['held']);model_path=BASE/f'fold_{fold:02d}'
        model=HeatmapModelV2236.load(model_path)
        if e['session'] in model.metadata['normalization']['training_sessions']:raise ValueError('Frozen model fold leakage')
        heat=model.score_map(vision.normalize(cached,model.metadata['normalization']))
        stages.append(dict(event=eid,stage='6_frozen_heatmap',family='logit',**rank_map(heat,gt,origin)))
        # Local values/signals are diagnostic and never change any map or peak.
        ds.atomic_json(folder/'temporal.json',dict(signal=signal.tolist(),pixel=pixel.tolist(),evidence=temporal))
        ds.atomic_json(folder/'stage_metadata.json',metadata);ds.atomic_json(folder/'transitions.json',tr);ds.atomic_json(folder/'stages.json',stages)
        montage(images,frames,gt,folder/'gt_chronology.png')
        def best(scope,reg=False):
            options=[r for r in tr if r['family']=='dark' and r['registered']==reg and (scope=='all' or r['scope']==scope) and r['raw_rank'] is not None]
            return min(options,key=lambda r:r['raw_rank']) if options else None
        b=best('primary');be=best('all');paired=next(r for r in tr if r['transition_to']==b['transition_to'] and r['registered'] and r['family']=='dark') if b else None
        lookup={(r['stage'],r['family']):r for r in stages}
        summary=dict(event=eid,cohort='hard' if eid in HARD else 'protected' if eid in PROTECTED else 'easy',frames=len(frames),primary_frames=n,
            raw_primary_best_rank=b['raw_rank'] if b else None,raw_primary_best_transition=b['transition_to'] if b else None,
            paired_registered_rank=paired['raw_rank'] if paired else None,raw_decision_best_rank=be['raw_rank'] if be else None,
            raw_decision_best_transition=be['transition_to'] if be else None,
            grouped_raw_rank=lookup['2_group_raw','persistent_dark']['raw_rank'],grouped_registered_rank=lookup['2_group_registered','persistent_dark']['raw_rank'],
            filtered_rank=lookup['3_filtered','persistent_dark']['raw_rank'],unit_rank=lookup['4_unit_float','persistent_dark']['raw_rank'],
            normalized_float_rank=lookup['4_scaled_float','persistent_dark']['raw_rank'],uint8_rank=lookup['5_uint8','persistent_dark']['raw_rank'],
            frozen_heatmap_rank=lookup['6_frozen_heatmap','logit']['raw_rank'],onset_primary=temporal['primary']['state'],onset_decision=temporal['decision']['state'],elapsed_seconds=time.perf_counter()-start)
        ds.atomic_json(folder/'summary.json',summary);alltrans+=tr;allstages+=stages;summaries.append(summary)
        write_csv(output/'event_summary.csv',summaries);write_csv(output/'stage_rank_ladder.csv',allstages);write_csv(output/'transition_diagnostics.csv',alltrans)
        ds.atomic_json(output/'input_hashes.json',hashes);print(summary,flush=True)
    ds.atomic_json(output/'summary.json',summaries)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--probe',action='store_true');a=p.parse_args()
    if a.probe:probe(a.output)
    else:run(a.output,resume=a.resume)
