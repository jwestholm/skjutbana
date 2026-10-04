"""Read-only two-event annotation lineage audit; never creates/changes truth.

Adjacent runtime events (and S01's next physical event after control4) are an
explicit allowlist. No session discovery, model inference, fitting or relabeling.
"""
from pathlib import Path
import argparse
import csv
import json
import cv2
import numpy as np
from automation import physical_ai_dataset as ds, physical_ai_vision as vision
from automation.physical_trace_label import frame_entries, fit_display, display_to_camera
from automation.physical_trace_target import register_pre_post
from src.engine.offline.common_verifier import camera_local_contrasts
from src.engine.offline.clean_physical_change import temporal_step_evidence

DATASET=Path('/data/skjutbana/physical_ai_lab/datasets/witness_v1_20261003')
CHAMPION=Path('/data/skjutbana/physical_ai_lab/vision_peak_smoke_final_20261003/trials/trial_000001/result.json')
TARGETS={'S01:3':dict(adjacent=[2,3,4,5],candidate=[2287.,1321.],physical_number=3),
         'S02:6':dict(adjacent=[5,6,7],candidate=[1968.,1235.],physical_number=5)}
FINALIZATION=Path('/data/skjutbana/evaluation_runs/S02_finalization_20260909_170523')


def validate_target(event):
    ds.guard(event)
    if event not in TARGETS:raise ValueError('Only S01:3 and S02:6 may be audited')


def source_path(shot,relative):
    p=shot/relative;ds.guard(p)
    if not p.resolve().is_relative_to(shot.resolve()):raise ValueError('Frame escapes event root')
    return p


def image(shot,entry):
    a=np.load(source_path(shot,entry['path']),mmap_mode='r',allow_pickle=False)
    if a.shape!=(2160,3840) or a.dtype!=np.uint8:raise ValueError('Unexpected native camera geometry')
    return a


def default_display_entry(trace):
    """Default image candidate only, NEVER proof of the image actually clicked."""
    entries=frame_entries(trace,'post')
    if not entries:raise ValueError('Missing POST')
    return dict(entry=entries[-1],association_status='UNKNOWN',basis='historical UI default index=-1; operator could navigate')


def current_calibration_identity(path=Path('content/settings.json')):
    """Record only today's identity; never substitute its matrix for history."""
    ds.guard(path)
    if not path.is_file():
        return dict(path=str(path),status='UNAVAILABLE')
    settings=ds.read(path)
    return dict(path=str(path),sha256=ds.digest(path),
                calibrated_at=settings.get('camera_calibration',{}).get('calibrated_at'),
                use='PRESENT_DAY_IDENTITY_ONLY_NOT_HISTORICAL_TRANSFORM')


def transforms(shape,gt,recorded,origin):
    """Only existing/default display and encoded coordinate conventions; no fit."""
    scale,size,offset=fit_display(shape,(1280,900))
    screen=np.array(gt)*scale+offset
    back=display_to_camera(tuple(screen),offset,scale,shape)
    rounded=display_to_camera(tuple(np.rint(screen)),offset,scale,shape)
    rows=[dict(test='native_camera_identity',status='VERIFIED',point=gt,error_to_manual_gt=0.,distance_to_recorded_candidate=ds.distance(gt,recorded)),
          dict(test='recorded_crop_roundtrip',status='VERIFIED',point=(np.array(gt)-origin+origin).tolist(),error_to_manual_gt=0.,note='No crop offset is used by the annotation UI'),
          dict(test='default_raw_display_roundtrip',status='SUPPORTED',point=list(back),error_to_manual_gt=ds.distance(back,gt),note=f'Code path only; actual window/mode unknown. size={size}, origin={offset}, scale={scale}'),
          dict(test='default_raw_integer_click_quantization',status='SUPPORTED',point=list(rounded),error_to_manual_gt=ds.distance(rounded,gt),note=f'Implied noninteger screen XY={screen.tolist()}; labels are not exactly representable by default raw-mode integer clicks.'),
          dict(test='historical_target_homography',status='UNKNOWN',note='Capture has calibration file hash, no matrix snapshot; current calibration is later. No substitute matrix or fitted translation used.')]
    # Hypothetical missed/doubled operations are explicit, never applied to GT.
    for name,point in [('missing_crop_origin',np.array(gt)-origin),('double_crop_origin',np.array(gt)+origin),
                       ('extra_180_rotation',[shape[1]-1-gt[0],shape[0]-1-gt[1]]),
                       ('extra_horizontal_flip',[shape[1]-1-gt[0],gt[1]]),('extra_vertical_flip',[gt[0],shape[0]-1-gt[1]])]:
        rows.append(dict(test=name,status='SUPPORTED',point=list(point),distance_to_recorded_candidate=ds.distance(point,recorded),note='Known coordinate operation counterfactual only; no evidence that it occurred.'))
    return rows


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def diagnostic_image(shot,trace,gt,candidate,history,onset,path):
    """Forensic illustration only; frame choices from recorded candidate onset."""
    pre=frame_entries(trace,'pre_history');post=frame_entries(trace,'post')
    entries=[pre[0],pre[max(0,(onset or 1)-1)],pre[min(len(pre)-1,onset or 1)],pre[-1],post[-1]]
    titles=['FIRST SAVED PRE','BEFORE RECORDED STEP','AT RECORDED STEP','LAST SAVED PRE','LAST SAVED POST (UI default only)']
    images=[image(shot,f) for f in entries];delta=images[-1].astype(np.float32)-images[0]
    images.append(np.clip(127.5+3*delta,0,255).astype(np.uint8));titles.append('SIGNED LAST POST - FIRST PRE')
    center=np.rint((np.array(gt)+candidate)*.5).astype(int);r=65;canvas=np.full((670,1170,3),240,np.uint8)
    for i,(im,title) in enumerate(zip(images,titles)):
        x,y=center;tile=cv2.cvtColor(im[y-r:y+r+1,x-r:x+r+1],cv2.COLOR_GRAY2BGR)
        tile=cv2.resize(tile,(262,262),interpolation=cv2.INTER_NEAREST)
        for xy,color in [(gt,(255,170,0)),(candidate,(0,130,255))]:
            q=np.rint((np.array(xy)-[x-r,y-r]+.5)*2-.5).astype(int);cv2.circle(tile,tuple(q),10,color,1)
        xx=(i%3)*390+20;yy=(i//3)*305;canvas[yy+30:yy+292,xx:xx+262]=tile
        cv2.putText(canvas,title,(xx,yy+20),cv2.FONT_HERSHEY_SIMPLEX,.44,(0,0,0),1)
        if i<5:cv2.putText(canvas,entries[i]['path'],(xx,yy+305),cv2.FONT_HERSHEY_SIMPLEX,.34,(0,0,0),1)
    cv2.putText(canvas,'BLUE: MANUAL_GT (immutable). ORANGE: RECORDED_PHYSICAL_CANDIDATE (not replacement truth).',(15,640),cv2.FONT_HERSHEY_SIMPLEX,.51,(0,0,0),1)
    cv2.imwrite(str(path),canvas)


def sensitivity(data,outputs,assessment):
    events=ds.with_label_source_lineage(data,assessment);parts=ds.label_lineage_partition(events)
    def metric(ids):
        subset=[e for e in events if e['event'] in ids]
        if not any(e['truth']=='HIT' for e in subset):return dict(status='UNAVAILABLE_NO_VERIFIED_PHYSICAL_EVENTS',impacts=0)
        m=vision.evaluate_union(subset,{e['event']:outputs[e['event']] for e in subset})['3']
        return {k:m[k] for k in ('impacts','classic_oracle','vision_oracle','combined_oracle')}
    full=metric({e['event'] for e in events});verified=metric(set(parts['verified']))
    remainder=metric({e['event'] for e in events}-set(parts['unresolved']))
    return dict(full_frozen_benchmark=full,verified_lineage_subset=verified,partition=parts,
                excluding_two_unresolved_DIAGNOSTIC_NOT_VERIFIED=remainder,
                warning='Other events are NOT_AUDITED, not verified. Exclusion sensitivity never replaces canonical metrics.')


def run(output):
    ds.install_guard();cv2.setNumThreads(1);output=ds.output_root(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'provenance_graph.json').exists():raise FileExistsError('Preserve completed audit')
    data,_=ds.load(DATASET);outputs=ds.read(CHAMPION)['outputs'];source_hashes={};graph={};alltrans=[];sequence=[];annotation=[]
    def record(path):
        path=Path(path);ds.guard(path)
        if path.is_file():source_hashes[str(path)]=ds.digest(path)
        return path
    def read(path):record(path);return ds.read(path)
    calibration=current_calibration_identity();record(calibration['path'])
    ds.atomic_json(output/'current_calibration_identity.json',calibration)
    oldhash=read(FINALIZATION/'S02_input_sha256.json')
    oldlabels=read(FINALIZATION/'S02_labels.json')
    vmanifest=read(Path('/data/skjutbana/physical_ai_lab/datasets/vision_proposals_v1_20261003/vision.json'))
    for eid,spec in TARGETS.items():
        validate_target(eid);event=next(e for e in data['events'] if e['event']==eid);shot=Path(event['source_root']);root=shot.parent.parent
        trace=read(shot/'trace.json');label=read(shot/'ground_truth.json');gt=[label['camera_x'],label['camera_y']];candidate=spec['candidate']
        if trace['shot_id']!=label['shot_id'] or gt!=event['gt_xy']:raise ValueError('Identity/GT contradiction; stop before metric overlay')
        candidate_matches=[c['id'] for c in event['candidates'] if c['xy']==candidate]
        if event['baseline_xy']!=candidate or not candidate_matches:
            raise ValueError('Previously recorded research candidate changed')
        for f in [shot/'trace.json',shot/'ground_truth.json']:
            if ds.digest(f)!=data['source_hashes'][str(f)]:raise ValueError('Immutable benchmark source changed')
        folder=output/eid.replace(':','_');folder.mkdir(exist_ok=False)
        # Inventory only the explicitly selected event, including original maps.
        inventory=[];historical_matches=[]
        for sub in [shot,shot/'frames',shot/'maps']:
            for f in sorted(sub.iterdir()):
                ds.guard(f)
                if not f.is_file():continue
                record(f);rel=str(f.relative_to(root));inventory.append(dict(path=str(f),sha256=source_hashes[str(f)],bytes=f.stat().st_size))
                if eid.startswith('S02:') and rel in oldhash:historical_matches.append(dict(path=rel,match=source_hashes[str(f)]==oldhash[rel],historical_sha256=oldhash[rel]))
        if any(not r['match'] for r in historical_matches):raise ValueError('Historical byte identity contradicted')
        for f in event['frames']:
            if ds.digest(f['frame_id'])!=data['source_hashes'][f['frame_id']]:raise ValueError('Benchmark frame drift')
        vr=next(r for r in vmanifest['events'] if r['event']==eid)
        pre=frame_entries(trace,'pre_history');posts=frame_entries(trace,'post');default=default_display_entry(trace)
        frames=sorted(pre+posts,key=lambda f:f['timestamp']);times=[f['timestamp'] for f in frames]
        if any(a>=b for a,b in zip(times,times[1:])):raise ValueError('Nonchronological source frames')
        vals=np.array([camera_local_contrasts(image(shot,f),[gt,candidate]) for f in frames])
        onset=temporal_step_evidence(vals[:len(pre),1],times[:len(pre)])
        # Historical target-overlay registration uses latest history as PRE.
        # All selectable POSTs tested; no claim about which one was clicked.
        ref=image(shot,pre[-1]);registration=[]
        for f in posts:
            _,shift=register_pre_post(ref,image(shot,f));registration.append(dict(frame=f['path'],shift=list(shift),norm=float(np.linalg.norm(shift))))
        rows=transforms((2160,3840),gt,candidate,np.array(vr['origin']))
        rows.append(dict(test='historical_annotation_overlay_translation',status='SUPPORTED',max_shift=max(r['norm'] for r in registration),note='Latest PRE vs every selectable POST; actual UI mode/frame unknown'))
        alltrans += [dict(event=eid,**r) for r in rows]
        root_files={}
        for name in ['recovery_metadata.json','collection_binding.json','physical_assignments.json']:
            f=root/name
            if f.is_file():root_files[name]=read(f)
        for sid in spec['adjacent']:
            adjacent=root/'shots'/f'shot_{sid:08d}';t=read(adjacent/'trace.json');g=read(adjacent/'ground_truth.json') if (adjacent/'ground_truth.json').is_file() else None
            st=read(adjacent/'ground_truth_status.json') if (adjacent/'ground_truth_status.json').is_file() else None
            pp=frame_entries(t,'pre_history');qq=frame_entries(t,'post');selected=[pp[0],pp[-1],qq[-1]]
            sample=[]
            for f in selected:
                im=image(adjacent,f);record(source_path(adjacent,f['path']));sample.append(dict(frame=f,contrast_at_target_gt_and_recorded_candidate=camera_local_contrasts(im,[gt,candidate]).tolist()))
            sequence.append(dict(audit_event=eid,runtime_event=sid,peak_ts=t['peak_ts'],next_peak=t.get('next_audio_peak_ts'),
                label=g,status=st,final_emission=t.get('outcome',{}).get('final_camera_xy'),pre_start=pp[0]['timestamp'],post_end=qq[-1]['timestamp'],
                target_gt_to_adjacent_gt=None if g is None else ds.distance(gt,[g['camera_x'],g['camera_y']]),
                recorded_candidate_to_adjacent_gt=None if g is None else ds.distance(candidate,[g['camera_x'],g['camera_y']]),frames=sample))
        evidence=dict(source_root=str(root),runtime_event=trace['shot_id'],physical_number=spec['physical_number'],manual_gt=label,recorded_candidate=candidate,
            candidate_origin=dict(dataset=str(DATASET/'dataset.json'),candidate_ids=candidate_matches,
                field='baseline_xy',status='PREVIOUSLY_RECORDED_OFFLINE_WITNESS_NOT_LIVE_EMISSION'),
            capture_commit=trace['provenance']['git_commit'],capture_dirty=trace['provenance']['working_tree_dirty'],
            calibration_context=trace['context']['calibration'],captured_calibration_settings_hash=trace['provenance']['model_identifiers']['content/settings.json'],
            root_metadata=root_files,default_display=default,default_display_sha256=ds.digest(source_path(shot,default['entry']['path'])),
            annotation_source_frame=None,annotation_view_mode='UNKNOWN',historical_annotation_calibration='UNKNOWN',
            history_count=len(pre),post_count=len(posts),all_frame_refs=frames,benchmark_frame_refs=event['frames'],vision_references=vr['reference_frames'],vision_post_group=vr['post_frames'],
            candidate_onset=onset,signals=vals.tolist(),overlay_registration=registration,historical_hash_matches=historical_matches,
            native_labels_match_september_manifest=(next(r['ground_truth'] for r in oldlabels['labels'] if r['event_id']==6)==label) if eid=='S02:6' else None)
        ds.atomic_json(folder/'inventory.json',inventory);ds.atomic_json(folder/'evidence.json',evidence)
        diagnostic_image(shot,trace,gt,candidate,pre,onset['onset_ordinal'],folder/'forensic_comparison.png')
        edges=[dict(source='alias',target='runtime_event',status='VERIFIED',evidence='explicit cohort mapping, binding/recovery record, trace and label shot_id agree'),
               dict(source='manual_label_file',target='benchmark_gt',status='VERIFIED',evidence='native file hash and exact XY equal frozen dataset'),
               dict(source='trace_frame_refs',target='benchmark_pixels',status='VERIFIED',evidence='frame hashes equal immutable benchmark sources'),
               dict(source='label_annotation_action',target='displayed_source_frame',status='UNKNOWN',evidence='writer omitted frame path/hash, view mode, click pixels, display geometry and calibration identity'),
               dict(source='historical_UI_default',target=default['entry']['path'],status='SUPPORTED',evidence='last POST default, navigation allowed; no click log proves actual use'),
               dict(source='label_annotation_action',target='annotation_calibration',status='UNKNOWN',evidence='Annotation did not save calibration identity/matrix'),
               dict(source='current_calibration',target='capture_calibration_snapshot',
                    status='CONTRADICTED' if calibration.get('sha256') and calibration['sha256']!=evidence['captured_calibration_settings_hash'] else 'UNKNOWN',
                    evidence=dict(current_identity=calibration,captured_hash=evidence['captured_calibration_settings_hash'],note='Compares file identity only; not a contradiction of the manual label')),
               dict(source='label',target='adjacent_event',status='UNKNOWN',evidence='no preserved positive evidence of a swap; no labels moved')]
        if eid=='S01:3':edges.append(dict(source='stale_biathlon_root',target='recovered_S01',status='SUPPORTED',evidence='recovery metadata and historical commit; original source path currently unavailable'))
        graph[eid]=dict(outcome='D. LABEL_TO_SOURCE_LINEAGE_UNRESOLVED',edges=edges,manual_gt=gt,
                       annotation_source_frame=None,default_frame_candidate=default,detail=str(folder/'evidence.json'))
        annotation.append(dict(event=eid,status='LABEL_SOURCE_LINEAGE_UNRESOLVED',label_sha256=ds.digest(shot/'ground_truth.json'),trace_sha256=ds.digest(shot/'trace.json'),
                               reason='Event/file identity verified; source frame/view/calibration at annotation not recorded and not recoverable from examined evidence.',
                               report=str(output/'REPORT.md'),annotation_source_frame=None))
    assessment=dict(schema='physical-label-lineage-1',dataset_hash=data['dataset_hash'],events=annotation)
    for name in ['evaluation_runs/S01_recovered_binding.json','evaluation_runs/S01_finalized.json','evaluation_runs/S01_quality.json','evaluation_runs/S02_retry_binding.json',
                 str(FINALIZATION/'S02_assignments.json'),str(FINALIZATION/'S02_finalized.json'),str(DATASET/'dataset.json'),str(CHAMPION)]:record(name)
    ds.atomic_json(output/'label_lineage_assessment.json',assessment)
    ds.atomic_json(output/'benchmark_sensitivity.json',sensitivity(data,outputs,assessment))
    ds.atomic_json(output/'provenance_graph.json',graph);ds.atomic_json(output/'input_hashes.json',source_hashes)
    ds.atomic_json(output/'event_sequence_audit.json',sequence);write_csv(output/'event_sequence_audit.csv',sequence)
    ds.atomic_json(output/'coordinate_transform_audit.json',alltrans);write_csv(output/'coordinate_transform_audit.csv',alltrans)
    print(json.dumps({k:v['outcome'] for k,v in graph.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
