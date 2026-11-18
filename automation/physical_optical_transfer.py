"""Read-only optical transfer falsification, never a hit detector or calibration writer.

Reuses GT frame identity and Optical Lab measurement/output contracts. WHITE/BLACK
predict held MID_GRAY; a second descriptive arm estimates one coefficient/channel
from OTHER triplets only. Same capture, not independent session validation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from automation import physical_optical_lab as optical
from automation import physical_trace_label_crops as originals
from automation.physical_gt_capture import atomic, checked

SCHEMA='physical_optical_transfer_audit_v1'
STATES=('BLACK','WHITE','MID_GRAY')
FRACTION=127/255
MIN_SPAN_DN=20  # predeclared measurement stratum, NEVER a detector threshold
CONTRACT=dict(live_authority=False,projectile_truth=False,physical_labels_used=False,
    emitted_accuracy='UNAVAILABLE',candidate_oracle='UNAVAILABLE',
    interpretation='UNEXPLAINED_OPTICAL_RESIDUAL_NOT_DAMAGE_PROOF',
    scope='POST_DECISION_NEUTRAL_IMAGES_NEVER_SAME_SHOT_LIVE_EVIDENCE',
    exposure_and_photon_timing_certified=False)


def stats(a):
    a=np.asarray(a)
    if not a.size:return dict(status='UNAVAILABLE',count=0)
    if not np.isfinite(a).all():raise ValueError('nonfinite measurement')
    return dict(status='AVAILABLE',count=int(a.size),mean=float(a.mean()),median=float(np.median(a)),
        p05=float(np.percentile(a,5)),p95=float(np.percentile(a,95)))


def error_stats(residual,mask):
    a=residual[mask]
    if not a.size:return dict(status='UNAVAILABLE',pixels=0)
    return dict(status='AVAILABLE',pixels=int(a.size),bias_dn=float(a.mean()),
        mae_dn=float(np.abs(a).mean()),median_abs_dn=float(np.median(np.abs(a))),
        p95_abs_dn=float(np.percentile(np.abs(a),95)))


def endpoint_mask(black,white):
    """Held MID_GRAY never participates in fit-mask or endpoint validity."""
    return (white-black>=MIN_SPAN_DN)&(black>0)&(black<255)&(white>0)&(white<255)


def measure_triplet(black,white,gray,coefficient=None):
    images=(black,white,gray)
    if any(a.dtype!=np.uint8 or a.ndim!=3 or a.shape[2]!=3 for a in images):
        raise ValueError('original native BGR uint8 required')
    if len({a.shape for a in images})!=1:raise ValueError('triplet dimensions differ')
    if coefficient is not None and (len(coefficient)!=3 or not np.isfinite(coefficient).all()):
        raise ValueError('finite BGR coefficients required')
    channels=[];maps={};h,w=black.shape[:2];grid=(math.ceil(w/64),math.ceil(h/64))
    for k,name in enumerate(('B','G','R')):
        b,wh,g=(a[...,k].astype(np.float32) for a in images);span=wh-b
        mask=endpoint_mask(b,wh);all_pixels=np.ones(mask.shape,bool)
        coeff=FRACTION if coefficient is None else coefficient[k]
        residual=g-(b+coeff*span)  # no clipping/registration/held fitting
        ratio=(g[mask]-b[mask])/span[mask]
        channels.append(dict(channel=name,coefficient=float(coeff),signed_span=stats(span),
            nonpositive_span_fraction=float(np.mean(span<=0)),
            endpoint_clipped_fraction=float(np.mean((b==0)|(b==255)|(wh==0)|(wh==255))),
            held_gray_clipped_fraction=float(np.mean((g==0)|(g==255))),
            responsive_endpoint_fraction=float(mask.mean()),held_ratio=stats(ratio),
            all_camera=error_stats(residual,all_pixels),responsive_endpoints=error_stats(residual,mask)))
        maps[name+'_span_mean']=cv2.resize(span,grid,interpolation=cv2.INTER_AREA)
        maps[name+'_residual_mae']=cv2.resize(np.abs(residual),grid,interpolation=cv2.INTER_AREA)
    return channels,maps


def coefficients_without(held_id,records):
    """Equal triplet weighting; refuses unknown coefficients, never reads held pixels."""
    held=next(r for r in records if r['shot_id']==held_id)
    train=[r for r in records if r['shot_id']!=held_id and r['status']=='AVAILABLE'
           and r.get('transfer_context')==held.get('transfer_context')]
    if len(train)<2:return None,[]
    values=[]
    for k in range(3):
        ratios=[r['channels'][k]['held_ratio'] for r in train]
        if any(s['status']!='AVAILABLE' for s in ratios):return None,[]
        values.append(float(np.median([s['median'] for s in ratios])))
    return values,[r['shot_id'] for r in train]


def neutral_entries(shot):
    entries={}
    for f in shot['frames']:
        for e in f['crops']:
            if e['role']!='GT_SETTLED':continue
            state=e['projector_state']
            if state in entries:raise ValueError('ambiguous duplicate neutral frame')
            entries[state]=e
    return entries


def triplet_identity(entries):
    if any(s not in entries for s in STATES):return 'MISSING_NEUTRAL_STATE'
    records=[entries[s] for s in STATES]
    if len({r['acquisition'].get('stream_id') for r in records})!=1:return 'CAMERA_STREAM_CHANGED'
    if not records[0]['acquisition'].get('stream_id'):return 'CAMERA_STREAM_UNKNOWN'
    if len({tuple(r['full_camera_size']) for r in records})!=1:return 'CAMERA_SIZE_CHANGED'
    if len({json.dumps(r.get('orientation'),sort_keys=True) for r in records})!=1:return 'CAMERA_ORIENTATION_CHANGED'
    if len({r['frame_id'] for r in records})!=3:return 'DUPLICATE_ACQUISITION'
    return None


def transfer_audit(source,output):
    source=checked(source);output=optical.output_path(output)
    if output.is_relative_to(source):raise ValueError('new analysis outside source capture required')
    if output.exists():raise ValueError('new immutable output directory required')
    manifest=originals.load_gt_review(source)  # shared hash/role/receipt checks, no candidates/labels
    output.mkdir(parents=True,exist_ok=False)
    config=dict(schema=SCHEMA,source=str(source),states=list(STATES),linear_fraction=FRACTION,
        min_endpoint_span_dn=MIN_SPAN_DN,clip_endpoints=[0,255],alignment='NONE',
        gamma_fitted=False,spatial_scope='FULL_CAMERA_NOT_SURVEYED_BOARD',
        held_gray_never_used_for_mask=True,coefficient_rule='median of OTHER triplet median ratios',
        validation_scope='SAME_CAPTURE_TRIPLET_HOLDOUT_NOT_SESSION_GENERALIZATION',**CONTRACT)
    atomic(output/'config.json',config)
    result=dict(**config,status='RUNNING',shots=[],input_hashes=manifest['input_hashes'],
        sources={p:optical.digest(p) for p in ('automation/physical_optical_transfer.py',
            'automation/physical_optical_lab.py','automation/physical_trace_label_crops.py')})
    atomic(output/'summary.json',result)
    started=time.perf_counter();rows=[]
    def load(entries):return [originals.source_image(manifest,entries[s]) for s in STATES]
    try:
        for shot in manifest['shots']:
            entries=neutral_entries(shot);reason=triplet_identity(entries)
            row=dict(shot_id=shot['shot_id'],status='UNAVAILABLE' if reason else 'AVAILABLE',reason=reason,
                sources={s:{k:e.get(k) for k in ('path','source_image_sha256','metadata_sha256','frame_id','timestamp','acquisition','orientation')} for s,e in entries.items()},
                pre_causality='NOT_USED; LEGACY_PRE_DOES_NOT_CERTIFY_BIRTH',
                physical_truth='UNKNOWN_NOT_DERIVED_FROM_NEUTRAL_RESPONSE')
            if not reason:
                ref=entries['BLACK']
                row['transfer_context']=dict(stream=ref['acquisition']['stream_id'],
                    size=ref['full_camera_size'],orientation=ref['orientation'])
                for state,e in entries.items():
                    md=json.loads((source/e['metadata_path']).read_text())
                    row['sources'][state].update({k:md.get(k) for k in
                        ('presentation_monotonic','settle_delay_s','projector_rgb','exposure_age_certified')})
                t=time.perf_counter();channels,maps=measure_triplet(*load(entries));row['channels']=channels
                row['elapsed_ms']=(time.perf_counter()-t)*1000
                row['neutral_sequence_span_ms']=(max(e['timestamp'] for e in entries.values())-min(e['timestamp'] for e in entries.values()))*1000
                row['tile_map']=f'shot_{shot["shot_id"]:03d}_linear_tiles.npz'
                np.savez_compressed(output/row['tile_map'],**maps)
                row['tile_map_sha256']=optical.digest(output/row['tile_map'])
            result['shots'].append(row);atomic(output/'summary.json',result)
        for row in result['shots']:
            coefficient,train=coefficients_without(row['shot_id'],result['shots'])
            arm=dict(status='UNAVAILABLE',reason='MISSING_TRIPLET_OR_OTHER_TWO_TRAIN_TRIPLETS')
            if row['status']=='AVAILABLE' and coefficient is not None:
                shot=next(s for s in manifest['shots'] if s['shot_id']==row['shot_id'])
                channels,_=measure_triplet(*load(neutral_entries(shot)),coefficient)
                arm=dict(status='AVAILABLE',train_shot_ids=train,held_shot_id=row['shot_id'],
                    coefficient_bgr=coefficient,channels=channels,validation='SAME_CAPTURE_ONLY')
            row['other_triplets_coefficient']=arm;atomic(output/'summary.json',result)
            for name,channels in [('two_endpoint_linear',row.get('channels',[])),('other_triplets_coefficient',arm.get('channels',[]))]:
                for c in channels:
                    for scope in ('all_camera','responsive_endpoints'):
                        rows.append(dict(shot_id=row['shot_id'],model=name,channel=c['channel'],scope=scope,
                            coefficient=c['coefficient'],**c[scope]))
        originals.assert_inputs(manifest)
        result.update(status='COMPLETED',elapsed_s=time.perf_counter()-started,
            available_triplets=sum(r['status']=='AVAILABLE' for r in result['shots']),
            limitation='No repeated within-state frames; gamma/exposure/motion/ambient causes not identifiable. No synchronized gameplay render or independent impact XY.')
    except BaseException as exc:
        result.update(status='INTERRUPTED',error=repr(exc));atomic(output/'summary.json',result);raise
    atomic(output/'summary.json',result)
    atomic(output/'input_manifest.json',dict(input_hashes=manifest['input_hashes'],source_unchanged_verified=True))
    if rows:
        keys=sorted(set().union(*(r.keys() for r in rows)))
        with (output/'measurements.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    (output/'REPORT.md').write_text('# Neutral optical transfer audit\n\n'+optical.DISCLAIMER+
        '\n\nWHITE/BLACK predict held MID_GRAY. Other-triplet coefficients never fit held gray; same-run only.\n'
        'See measurements.csv and summary.json. Unexplained residual is not material damage.\n'
        'No labels, source scores, candidates or live calibration are used. No emitted accuracy/oracle available.\n'
        'Repeated neutral recovery/within-state variability are missing: causes cannot be separated.\n')
    return result


def synthetic_sensitivity(output):
    """Fixed analytical stress cases, not a detector or synthetic physical validation."""
    from src.engine.offline.scenario_generator_v220 import OfflineScenarioGeneratorV220, _CameraStateV220, _procedural_frame
    output=optical.output_path(output);output.mkdir(parents=True,exist_ok=False)
    # Existing generator primitives; no media catalog/data discovery.
    w,h=320,240;pre=_procedural_frame('checker',w,h,0,1729)
    post=cv2.warpAffine(pre,np.float32([[1,0,3],[0,1,0]]),(w,h),borderMode=cv2.BORDER_REFLECT)
    H=np.array([[1.,.02,0],[.01,1.,0],[0.,0.,1.]])
    camera=_CameraStateV220(.72,1.7,.7,18.,(1.,.95,1.05))
    def observe(render,matrix=H):
        mapped=cv2.warpPerspective(render,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        return OfflineScenarioGeneratorV220._apply_camera_state(mapped,camera).astype(np.float32)
    a,b=observe(pre),observe(post);black,white=observe(np.zeros_like(pre)),observe(np.full_like(pre,255))
    warped=[cv2.warpPerspective(r,H,(w,h),borderMode=cv2.BORDER_REFLECT).astype(np.float32) for r in (pre,post)]
    shifted=H.copy();shifted[0,2]+=1.
    models=dict(raw_no_explanation=np.zeros_like(a),known_transfer=b-a,
        two_endpoint_affine=(white-black)/255*(warped[1]-warped[0]),
        one_pixel_alignment_error=observe(post,shifted)-observe(pre,shifted))
    yy,xx=np.mgrid[:h,:w];center=(160,120);mask=(xx-center[0])**2+(yy-center[1])**2<=3**2
    injection=np.zeros_like(a);injection[mask]=-8.  # test perturbation in camera DN, NOT a hole simulator
    rows=[]
    for name,prediction in models.items():
        without=b-a-prediction;with_local=b+injection-a-prediction
        retained=with_local-without;saliency=np.mean(np.abs(with_local),axis=2)
        y,x=np.unravel_index(np.argmax(saliency),saliency.shape)
        rows.append(dict(model=name,synthetic=True,injected_delta_dn=-8.,radius_px=3,
            perturbation_retained_l1_fraction=float(np.abs(retained).sum()/np.abs(injection).sum()),
            perturbation_identity_max_error=float(np.max(np.abs(retained-injection))),
            outside_residual_p95_dn=float(np.percentile(saliency[~mask],95)),
            inside_residual_mean_dn=float(saliency[mask].mean()),
            diagnostic_max_xy=[int(x),int(y)],diagnostic_max_distance_to_injection=float(np.hypot(x-center[0],y-center[1])),
            emitted_coordinate=None,selection_semantics='RESIDUAL_PIXEL_MAX_ONLY_NOT_ENGINE_SELECTOR'))
    # Explicit counterexample: fitting a free residual on the evaluated pair can
    # explain away ANY new change. This is a forbidden inference, never a model arm.
    result=dict(schema=SCHEMA,synthetic=True,status='COMPLETED',**CONTRACT,rows=rows,
        negative_control=dict(name='FORBIDDEN_SAME_PAIR_DENSE_EXPLANATION',perturbation_retained_l1_fraction=0.,
            reason='Explaining the evaluated POST with itself destroys identifiability; do not fit impact pixels away'),
        seed=1729,camera_fixture=dict(gain=.72,gamma=1.7,blur_sigma=.7,black_level=18.),
        geometry_fixture=H.tolist(),physical_transfer_validated=False)
    atomic(output/'summary.json',result)
    (output/'REPORT.md').write_text('# Synthetic transfer sensitivity\n\n'+optical.DISCLAIMER+
        '\n\nFrozen prediction preserves additive perturbations algebraically. Wrong transfer/geometry can still dominate their residual rank.\n'
        'These are controlled counterexamples, not candidate/selected accuracy measurements or new live features.\n')
    return result
