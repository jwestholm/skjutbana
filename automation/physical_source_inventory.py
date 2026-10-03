"""Explicit DEVELOPMENT real-source inventory; no generator, augmentation or fit."""
from __future__ import annotations
import argparse,json,math
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from automation.accuracy_physical_dataset import COHORT
from automation.causal_pre_onset_research import read_guard,digest,write
from automation import chronological_pre_research as chrono


def valid_patch_footprint(image,point):
    """Extractability only: this cannot certify a new physical damage delta."""
    if len(point)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in point):return False
    x,y=point
    return image.ndim==2 and x>=25 and y>=25 and x+27<=image.shape[1] and y+27<=image.shape[0]


def run(output):
    output.mkdir(exist_ok=False)
    roots={k:Path('/data/skjutbana/physical_traces')/COHORT[k][0] for k in ('H10','H20','POST_FIX','S01','S02')}
    roots.update(D02=chrono.D02,KOMP=chrono.SESSION,CAN=Path('/data/skjutbana/physical_traces/session_20260915_150509_CAN_OBJECT_b2eb4562'))
    accesses=read_guard(output.parent,list(roots.values()));rows=[];hashes={}
    for name,root in roots.items():
        false_ids=COHORT[name][1] if name in COHORT else (2,4,5,6) if name=='D02' else ()
        for shot in sorted((root/'shots').glob('shot_*')):
            sid=int(shot.name.split('_')[-1]);gp=shot/'ground_truth.json'
            if sid in false_ids or not gp.exists():continue
            gt=json.loads(gp.read_text());point=[gt.get('camera_x'),gt.get('camera_y')]
            if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in point):continue
            hashes[str(gp)]=digest(gp);tp=shot/'trace.json';trace=json.loads(tp.read_text()) if tp.exists() else None
            if trace:
                hashes[str(tp)]=digest(tp)
                pre=sorted((f for f in trace['frames'] if f['kind']=='pre_history'),key=lambda f:f['timestamp'])
                snaps=[f for f in trace['frames'] if f['kind']=='pre_snapshot']
                post=sorted((f for f in trace['frames'] if f['kind']=='post'),key=lambda f:f['timestamp'])
                pre_paths=[shot/f['path'] for f in pre] or [shot/f['path'] for f in snaps]
                post_paths=[shot/f['path'] for f in post];cut=(trace.get('decision_input') or {}).get('timestamp')
                causal_post=[f for f in post if cut and trace['peak_ts']<=f['timestamp']<=cut and f['timestamp']<(trace.get('next_audio_peak_ts') or math.inf)]
            else:
                pre_paths=sorted((shot/'frames').glob('pre_history_*.npy'));post_paths=sorted((shot/'frames').glob('post_*.npy'));cut=None;causal_post=[]
            usable=bool(pre_paths and post_paths);reason=None;static_patch_hash=None
            if usable:
                for p in (pre_paths[0],post_paths[-1]):
                    im=np.load(p,mmap_mode='r',allow_pickle=False);x,y=point
                    if not valid_patch_footprint(im,point):usable=False;reason='incomplete patch footprint'
                    hashes[str(p)]=digest(p)
                    if p==post_paths[-1]:
                        import hashlib
                        ix,iy=round(x),round(y);static_patch_hash=hashlib.sha256(im[iy-24:iy+25,ix-24:ix+25].tobytes()).hexdigest()
            rows.append(dict(event=f'{name}:{sid}',source_session=str(root),source_hole_id=f'{root.name}:{sid}',
                gt=point,label_path=str(gp),label_source=gt.get('label_source'),label_quality=gt.get('quality'),
                pre_frames=len(pre_paths),post_frames=len(post_paths),paired_patch_available=usable,reason=reason,
                first_pre_path=str(pre_paths[0]) if pre_paths else None,last_post_path=str(post_paths[-1]) if post_paths else None,
                post_patch_sha256=static_patch_hash,recorded_causal_cutoff=cut,post_frames_by_cutoff=len(causal_post),
                causal_damage_delta_certified=False,qualification='pixels/GT extractable; clean local before/onset/after and mask still require donor qualification',
                caveat='S02:1 possible multiple visible changes unconfirmed' if name=='S02' and sid==1 else None))
    # Legacy F2 patches have manual GT but no paired PRE and no event-unique provenance guarantee.
    assets=[]
    for p in sorted(Path('content/ai/holes').glob('hole_*.json')):
        meta=json.loads(p.read_text());image=p.with_suffix('.png');im=cv2.imread(str(image),cv2.IMREAD_GRAYSCALE)
        assets.append(dict(path=str(image),metadata=str(p),session=meta.get('session_id'),run=meta.get('run_id'),round=meta.get('round_id'),
            gt=[meta.get('gt_camera_x'),meta.get('gt_camera_y')],shape=list(im.shape) if im is not None else None,
            image_sha256=digest(image) if image.exists() else None,paired_pre=False))
        hashes[str(p)]=digest(p)
    packs=[]
    # Explicit existing F2 directories, both source_kind=single_projected; not real projectile donors.
    for dirname in ('20260829_212658_ai_training_scene_67465571','20260829_212403_ai_training_scene_29603627'):
        root=Path('content/ai/training_v223/framepacks')/dirname
        for p in sorted(root.glob('shot_*.json')):
            m=json.loads(p.read_text());packs.append(dict(path=str(p),source_kind=m.get('source_kind'),npz_exists=p.with_suffix('.npz').exists()))
    counts={g:dict(labelled=len([r for r in rows if r['event'].startswith(g+':')]),
        paired=sum(r['paired_patch_available'] for r in rows if r['event'].startswith(g+':')),
        clocked_post=sum(bool(r['post_frames_by_cutoff']) for r in rows if r['event'].startswith(g+':'))) for g in roots}
    summary=dict(sessions=counts,paired_total=sum(r['paired_patch_available'] for r in rows),
        distinct_source_hole_ids=len({r['source_hole_id'] for r in rows}),distinct_sampled_post_patch_hashes=len({r['post_patch_sha256'] for r in rows if r['paired_patch_available']}),
        legacy_static_patches=len(assets),legacy_static_unique_hashes=len({a['image_sha256'] for a in assets}),legacy_static_sessions=dict(Counter(a['session'] for a in assets)),
        legacy_round_keys_unique=len({(a['session'],a['run'],a['round']) for a in assets}),
        projected_framepacks=len(packs),projected_source_kinds=dict(Counter(p['source_kind'] for p in packs)),
        ready_temporal_donor_masks=0,scope='Explicit independently labelled source events; extractability is not clean damage verification; no S03 or invalid D01 raw')
    write(output/'source_manifest.json',dict(summary=summary,events=rows,legacy_static=assets,framepacks=packs))
    write(output/'input_hashes.json',hashes);write(output/'read_accesses.json',sorted(accesses));print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
