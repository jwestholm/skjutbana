"""Small OFFLINE donor-derived dataset builder. No model training/activation."""
from __future__ import annotations
import argparse,json,hashlib,math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from automation.causal_pre_onset_research import read_guard,digest,write
from automation.accuracy_physical_dataset import COHORT
from src.engine.offline.hole_dataset_v213 import _assign_sessions_balanced
from src.engine.offline.common_verifier import camera_local_patches


def split_sources(rows,seed):
    eligible=[r for r in rows if r['result']['status']=='QUALIFIED']
    ids=[r['source_hole_id'] for r in eligible]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate donor lineage')
    return _assign_sessions_balanced([SimpleNamespace(session_id=r['source_session']) for r in eligible],
        seed=seed,train_fraction=.7,validation_fraction=.15)


def compose(donor,background,*,method,shift=(0.,0.)):
    """Two explicitly experimental physical residual-transfer assumptions.

    Measured signed residual/mask, not an invented dark blob. Native size only;
    a quarter-pixel translation is the sole geometric augmentation in this MVP.
    """
    pre=np.asarray(background,np.float32);mask=donor['damage_mask'].astype(np.float32)
    if pre.shape!=(49,49) or donor['pre'].shape!=pre.shape:raise ValueError('49x49 donor and background required')
    matrix=np.float32([[1,0,shift[0]],[0,1,shift[1]]])
    alpha=cv2.warpAffine(mask,matrix,(49,49),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    if method=='additive_delta':layer=-donor['signed_delta']*mask
    elif method=='relative_response':layer=-donor['signed_delta']/np.maximum(donor['pre'],1.)*mask
    else:raise ValueError('Unknown composition assumption')
    moved=cv2.warpAffine(layer,matrix,(49,49),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    raw=pre+moved if method=='additive_delta' else pre*(1+moved)
    post=np.clip(raw,0,255).astype(np.float32)
    return pre.copy(),post,alpha,dict(clipped_pixel_fraction=float(np.mean((raw<0)|(raw>255))),
        masked_mae=float(np.abs(post-pre)[alpha>0].mean()) if np.any(alpha>0) else 0.,
        outside_mask_mae=float(np.abs(post-pre)[alpha==0].mean()),
        method_semantics='experimental sensor-level transfer; not a validated reflectance/rendering model')


def negatives(rows,output,assignments):
    roots={r['event'].split(':')[0]:Path(r['source_session']) for r in rows};records=[]
    known={k:COHORT[k][1] for k in ('H20','S01','S02')};known['D02']=(2,4,5,6)
    for group,ids in known.items():
        root=roots[group]
        for sid in ids:
            tp=root/'shots'/f'shot_{sid:08d}'/'trace.json';t=json.loads(tp.read_text());decision=t.get('decision_input') or {}
            cut=min(decision.get('timestamp',math.inf),t.get('next_audio_peak_ts') or math.inf)
            tracks=(decision.get('complete_track_audit') or {}).get('tracks',[])
            winner=next((c for c in tracks if c.get('final_rank')==1),None)
            r=dict(event=f'{group}:{sid}',source_session=str(root),source_event=f'{root.name}:{sid}',split=assignments[str(root)],
                labels=dict(STATIC_HOLE_APPEARANCE=None,NEW_PHYSICAL_DAMAGE=0,FINAL_SELECTION=None),
                truth='NO_PHYSICAL_SHOT from existing human-supported cohort assignments',
                context='isolated_sound' if group=='D02' and sid==2 else 'shared_right_edge_action_cough_vs_knock_UNKNOWN' if group=='D02' and sid in (4,5) else 'left_edge_control' if group=='D02' else 'control_action_unspecified',
                pixel_cause='UNKNOWN; noise/line/flicker are descriptive hypotheses, not new human labels',trace_sha256=digest(tp),available=False)
            fs=t.get('frames',[]);pre=[f for f in fs if f['kind']=='pre_snapshot'];post=[f for f in fs if f['kind']=='post' and f['timestamp']<=cut]
            if winner and pre and post:
                point=[winner['camera_x'],winner['camera_y']];a=pre[0];b=max(post,key=lambda f:f['timestamp'])
                try:
                    images=[camera_local_patches(np.load(tp.parent/f['path'],mmap_mode='r'),[point])[0] for f in (a,b)]
                    dest=output/'samples'/f'negative_{group}_{sid}.npz';np.savez_compressed(dest,pre=images[0],post=images[1])
                    r.update(available=True,path=str(dest),xy=point,source_frames=[a,b],sha256=digest(dest))
                except ValueError:r['unavailable_reason']='Recorded coordinate patch out of frame'
            else:r['unavailable_reason']='Complete recorded winner/cutoff context unavailable; not fabricated'
            records.append(r)
    return records


def build(manifest,output,seed=20260927):
    output.mkdir(parents=True,exist_ok=False);data=json.loads(manifest.read_text());rows=data['events']
    read_guard(output,[Path(r['source_session']) for r in rows]);cv2.setNumThreads(1)
    donors=[r for r in rows if r['result']['status']=='QUALIFIED'];assignment=split_sources(rows,seed)
    if len(assignment)<3:raise ValueError('Three independent source sessions required for this builder')
    split=dict(seed=seed,source_manifest=str(manifest),source_manifest_sha256=digest(manifest),session_assignment=assignment,
        rule='whole source session and its donor/background descendants assigned BEFORE augmentation',status='DEVELOPMENT_SPLIT_NOT_UNTOUCHED_PHYSICAL_VALIDATION')
    # Persist split before opening donor arrays or generating variants.
    write(output/'split.json',split);(output/'samples').mkdir();samples=[];views=[]
    for i,r in enumerate(donors):
        if digest(r['extraction_path'])!=r['extraction_sha256']:raise ValueError('Donor artifact hash mismatch')
        with np.load(r['extraction_path'],allow_pickle=False) as z:donor={k:z[k].copy() for k in z.files}
        partition=assignment[r['source_session']]
        original=output/'samples'/f"original_{r['event'].replace(':','_')}.npz";np.savez_compressed(original,**donor)
        samples.append(dict(id=original.stem,path=str(original),sha256=digest(original),split=partition,synthetic=False,
            donor_id=r['source_hole_id'],donor_session=r['source_session'],source_artifact=r['extraction_path'],
            labels=dict(STATIC_HOLE_APPEARANCE=1,NEW_PHYSICAL_DAMAGE=1,FINAL_SELECTION=None)))
        if partition!='train':continue
        backgrounds=[b for b in donors if assignment[b['source_session']]==partition and b['source_hole_id']!=r['source_hole_id']]
        if not backgrounds:raise ValueError('Separate same-partition background required')
        bg=backgrounds[i%len(backgrounds)]
        with np.load(bg['extraction_path'],allow_pickle=False) as z:background=z['pre'].copy()
        for j,method in enumerate(('additive_delta','relative_response')):
            for variant,shift in enumerate(((0.,0.),(.25,-.25))):
                pre,post,mask,metrics=compose(donor,background,method=method,shift=shift)
                name=f"synthetic_{r['event'].replace(':','_')}_{j}_{variant}";path=output/'samples'/f'{name}.npz'
                np.savez_compressed(path,pre=pre,post=post,damage_mask=mask,gt_patch_xy=np.asarray([24+shift[0],24+shift[1]]))
                samples.append(dict(id=name,path=str(path),sha256=digest(path),split=partition,synthetic=True,method=method,
                    donor_id=r['source_hole_id'],donor_session=r['source_session'],background_id=bg['source_hole_id'],background_session=bg['source_session'],
                    donor_hash=r['extraction_sha256'],background_hash=bg['extraction_sha256'],shift=shift,metrics=metrics,
                    labels=dict(STATIC_HOLE_APPEARANCE=1,NEW_PHYSICAL_DAMAGE=1,FINAL_SELECTION=None),
                    final_selection_target_unavailable='No actual candidate pool; GT patch center is supervised construction only'))
                if variant==0:
                    panel=np.hstack([cv2.resize(x.astype(np.uint8),(147,147),interpolation=cv2.INTER_NEAREST) for x in (donor['post'],background,post)])
                    panel=cv2.cvtColor(panel,cv2.COLOR_GRAY2BGR);panel=cv2.copyMakeBorder(panel,25,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
                    cv2.putText(panel,r['event']+' '+method,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1);views.append(panel)
    neg=negatives(rows,output,assignment)
    for i in range(0,len(views),8):cv2.imwrite(str(output/f'composition_{i//8}.png'),np.vstack(views[i:i+8]))
    for s in samples:
        assert assignment[s['donor_session']]==s['split']
        if s['synthetic']:assert assignment[s['background_session']]==s['split']=='train'
    write(output/'dataset.json',dict(samples=samples,negatives=neg,summary=dict(qualified_donors=len(donors),
        original_pairs=sum(not s['synthetic'] for s in samples),synthetic_pairs=sum(s['synthetic'] for s in samples),
        originals_by_split=dict(Counter(s['split'] for s in samples if not s['synthetic'])),
        negative_inventory=len(neg),negative_pairs=sum(n['available'] for n in neg)),
        training_run=False,temporal_sequences_generated=False,targets_not_collapsed=True))
    write(output/'source_hashes.json',{str(p):digest(p) for p in (Path(__file__),manifest,Path('src/engine/offline/hole_dataset_v213.py'))})
    print(json.dumps(json.loads((output/'dataset.json').read_text())['summary']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--qualified-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=20260927)
    a=p.parse_args();build(a.qualified_manifest,a.output,a.seed)
