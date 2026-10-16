"""One frozen region: offline adapter to ObjectPhysicalQuery, NOT a detector.

No live caller, tuning, labels, fused score, emission or inferred physical cause.
Saved POST camera pixels remain the coordinate plane. Context outside the exact
shape cannot create inside support. Counterfactual crop replay is not live parity.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import cv2
import numpy as np

from automation import object_hit_replay as replay
from automation.physical_collection import _development_path
from src.engine.camera.analysis_geometry_v2221 import AnalysisGeometryV2221
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
from src.engine.camera.object_physical_query import ObjectPhysicalQuery
from src.engine.game_objects.geometry import shape_contains
from src.engine.offline.accuracy_verifier import RADII

SCHEMA = 'single_region_evidence_research_v1'
CLOCK = time.perf_counter
CONTRACT = dict(authority='OFFLINE_RESEARCH_ONLY', live_authority=False,
    labels_used=False, weights=None, coordinate_plane='ORIGINAL_POST_CAMERA_PIXELS',
    primary_secondary_cause='UNRESOLVED', exposure_time_certified=False,
    projector_component='UNAVAILABLE_WITHOUT_SHOT_RENDER_AND_TRANSFER',
    residual_semantics='unexplained_by_tested_model_NOT_proof_of_material_damage')


def subset_crop(geometry, saved):
    """Only a subset of ONE real saved tile; never pad, stitch or invent pixels."""
    if not isinstance(saved, replay.SavedCrops):
        raise TypeError('saved inputs only')
    x0,y0,x1,y1 = geometry.crop_x0,geometry.crop_y0,geometry.crop_x1,geometry.crop_y1
    matches=[]
    for (a,b,c,d),image in sorted(saved.crops.items()):
        if a<=x0<x1<=c and b<=y0<y1<=d:
            matches.append(image[y0-b:y1-b,x0-a:x1-a])
    if not matches: raise ValueError('REGION_CROP_UNAVAILABLE')
    if any(not np.array_equal(matches[0],a) for a in matches[1:]):
        raise ValueError('conflicting overlapping source crops')
    return matches[0]


def one_snapshot(snapshot, object_id):
    game=tuple(r for r in snapshot.game_regions if r.object_id==object_id and r.enabled)
    camera=tuple(r for r in snapshot.camera_regions if r.object_id==object_id)
    if len(game)!=1 or len(camera)!=1: raise ValueError('ONE_FROZEN_REGION_REQUIRED')
    return replace(snapshot, regions=(), game_regions=deepcopy(game), camera_regions=deepcopy(camera))


def inside(region, game_xy):
    shape=region.metadata.get('v250_object_snapshot',{}).get('shape')
    if shape: return shape_contains(shape,*game_xy)
    x,y=game_xy
    # Same plain HitRegion half-open semantics as ObjectPhysicalQuery.observe.
    return region.x<=x<region.x+region.width and region.y<=y<region.y+region.height


def exact_mask(region, geometry, convert):
    h,w=geometry.crop_height,geometry.crop_width
    return np.fromiter((inside(region,convert(*geometry.local_to_camera(x,y)))
        for y in range(h) for x in range(w)),dtype=bool,count=h*w).reshape(h,w)


def distribution(values, mask):
    x=np.asarray(values)[mask]
    if not x.size:return dict(status='UNAVAILABLE',pixels=0)
    return dict(status='AVAILABLE',pixels=int(x.size),signed_mean=float(np.mean(x)),
        abs_mean=float(np.mean(np.abs(x))),abs_p95=float(np.percentile(np.abs(x),95)),
        abs_max=float(np.max(np.abs(x))),abs_mass=float(np.abs(x).sum()))


def family(name, values, start, *, available=True, reason=None, scope='exact_shape_and_margin', quality=None):
    return dict(family=name,availability='AVAILABLE' if available else 'UNAVAILABLE',
        reason=reason,values=values,quality=quality,confidence=None,scope=scope,
        causal_availability='recorded_observation_cutoff_only_not_exposure_certification',
        live_inputs_possible=True,source_independent=True,
        provenance='parent.frame_sources + parent.peak_ts/recorded_cutoff_ts + frame_ts; hashed manifest inputs',
        runtime_ms=(CLOCK()-start)*1000,
        inference_authority=False)


def warp_valid(shape, registration):
    if not registration['applied']: return np.ones(shape,bool)
    m=np.float32([[1,0,-registration['dx']],[0,1,-registration['dy']]])
    return cv2.warpAffine(np.ones(shape,np.float32),m,(shape[1],shape[0]),
        flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)==1.


def residual_maps(history, post, search, exact):
    """Reuse V2 bounded translation and photometry, retaining raw + fit residual.

    Translation/brightness fit uses OUTSIDE context; never an inside-target reward.
    Old structure can move without vetoing new residual at the same coordinate.
    This joint photometric/surface model cannot separate projector from surface.
    """
    start=CLOCK()
    history=np.asarray(history,np.float32);post=np.asarray(post,np.float32)
    if history.ndim!=3 or history.shape[1:]!=post.shape or search.shape!=post.shape or exact.shape!=post.shape:
        raise ValueError('image/mask dimensions differ')
    if not np.isfinite(history).all() or not np.isfinite(post).all():raise ValueError('nonfinite pixels')
    pre=history[-1]
    context=(search & ~exact)
    raw=post-pre
    # Same bounded algorithm/defaults as V2. No new motion/damage threshold.
    if np.count_nonzero(context)<64:
        return {'raw':raw},[family('NEW_LOCAL_RESIDUAL',
            dict(raw_inside=distribution(raw,search&exact),raw_margin=distribution(raw,context)),start),
            family('SURFACE_MOTION_EXPLANATION',{},CLOCK(),available=False,reason='INSUFFICIENT_CONTEXT')]
    roi=context.astype(np.uint8)*255
    aligned,reg=CandidateGeneratorV2._register_current(None,post,pre,roi=roi,
        cfg={'registration_max_dimension':256})
    current,offset=CandidateGeneratorV2._normalise_photometry(aligned,post,roi=roi)
    residual=current.astype(np.float32)-aligned
    valid=warp_valid(post.shape,reg)&search
    low,high=aligned.copy(),aligned.copy()
    history_reg=[]
    for prior in history[:-1]:
        a,r=CandidateGeneratorV2._register_current(None,post,prior,roi=roi,
            cfg={'registration_max_dimension':256})
        # Normalize older PRE into latest registered PRE photometric plane.
        a,o=CandidateGeneratorV2._normalise_photometry(aligned,a,roi=roi)
        np.minimum(low,a,out=low);np.maximum(high,a,out=high)
        valid &= warp_valid(post.shape,r)
        history_reg.append(dict(registration=r,photometric_offset=o))
    novelty=ObjectPhysicalQuery._new_change(current.astype(np.float32)-low,
                                           (current.astype(np.float32)-high,)) if len(history)>=2 else None
    maps=dict(raw=raw,registered_residual=residual,valid=valid.astype(np.uint8),
              aligned_pre=aligned,pre_envelope_width=high-low)
    if novelty is not None:maps['observed_envelope_residual']=novelty
    elapsed=(CLOCK()-start)*1000
    f1=family('NEW_LOCAL_RESIDUAL',dict(raw_inside=distribution(raw,search&exact),
        raw_margin=distribution(raw,context),registered_inside=distribution(residual,valid&exact),
        registered_margin=distribution(residual,valid&~exact),
        observed_envelope_inside=None if novelty is None else distribution(novelty,valid&exact),
        observed_envelope_margin=None if novelty is None else distribution(novelty,valid&~exact),
        history_count=len(history),clean_pre_certified=False,physical_cause_certified=False),CLOCK(),
        reason='OBSERVED_DIFFERENCE_NOT_CERTIFIED_NEW_DAMAGE')
    f1['runtime_ms']=elapsed
    f3=family('SURFACE_MOTION_EXPLANATION',dict(registration=reg,photometric_offset=offset,
        pre_history_registration=history_reg,context_pixels=int(context.sum()),
        raw_inside=distribution(raw,valid&exact),residual_inside=distribution(residual,valid&exact),
        raw_margin=distribution(raw,valid&~exact),residual_margin=distribution(residual,valid&~exact),
        interpolation_boundary_excluded_pixels=int((search&~valid).sum()),
        motion_fit_status='APPLIED' if reg['applied'] else 'NOT_APPLIED',
        photometry_semantics='existing_clipped_median_offset_includes_uint8_quantization',
        projector_vs_surface_separated=False),CLOCK(),quality={'phase_response':reg['response']},
        reason='BOUNDED_TRANSLATION_PLUS_OFFSET_ONLY_NOT_CAUSE_IDENTIFICATION')
    # Shared extraction cost belongs to F1; do not double-count correlated work.
    f3.update(runtime_ms=0.,shared_compute_family='NEW_LOCAL_RESIDUAL')
    return maps,[f1,f3]


def residual_structure(residual, valid, exact, points, geometry):
    """Describe existing coordinates, not new proposals or a preferred center.

    Radii from accuracy_verifier. No hard size plausibility without physical scale.
    Multiple correlated summaries stay ONE family. No topology/damage assertion.
    """
    start=CLOCK();a=np.abs(residual);rows=[]
    for xy in sorted(set(tuple(c['camera_xy']) for c in points)):
        x,y=xy[0]-geometry.crop_x0,xy[1]-geometry.crop_y0
        rmax=max(RADII)*2
        x0,y0=max(0,math.floor(x-rmax)),max(0,math.floor(y-rmax))
        x1,y1=min(a.shape[1],math.ceil(x+rmax)+1),min(a.shape[0],math.ceil(y+rmax)+1)
        if x1<=x0 or y1<=y0:continue
        yy,xx=np.mgrid[y0:y1,x0:x1];distance=np.hypot(xx-x,yy-y)
        local=a[y0:y1,x0:x1];v=valid[y0:y1,x0:x1]
        levels=[]
        for r in RADII:
            core=(distance<=r)&v;ring=(distance>r)&(distance<=2*r)&v
            c=distribution(local,core);s=distribution(local,ring)
            levels.append(dict(radius_camera_px=r,core=c,surround=s,
                support_truncated=bool(x-r*2<0 or y-r*2<0 or x+r*2>=a.shape[1] or y+r*2>=a.shape[0])))
        rows.append(dict(camera_xy=list(xy),multiscale=levels))
    total=float(a[valid].sum());im=float(a[valid&exact].sum())
    return family('LOCAL_RESIDUAL_STRUCTURE',dict(existing_coordinate_descriptors=rows,
        inside_absolute_mass=im,outside_absolute_mass=total-im,
        inside_mass_fraction=im/total if total else None,
        physical_size_plausibility='UNAVAILABLE_NO_SURFACE_SCALE',
        primary_secondary_assignment='UNRESOLVED_NO_MATERIAL_DAMAGE_CLASSIFIER'),start,
        reason='CORRELATED_RESIDUAL_DESCRIPTIONS_NOT_INDEPENDENT_WITNESSES')


def region_result(query_result, region, convert):
    """Native support counterfactual stays separate from physical research truth.

    No new discriminator exists to qualify material damage. Never elevate a
    single-region legacy result just because multi-object ambiguity disappeared.
    """
    d=query_result.get('object_decision',{})
    points={tuple(c['camera_xy']):c for row in query_result.get('per_object',[]) for c in row['candidates']
            if c.get('confirmed') and inside(region,convert(*c['camera_xy']))}
    diagnostics=[]
    for xy,c in sorted(points.items()):
        game=convert(*xy)
        diagnostics.append(dict(camera_xy=list(xy),game_xy=list(game),
            object_local_xy=None,object_local_status='UNAVAILABLE_NO_FROZEN_SPRITE_ORIGIN_TRANSFORM',
            region_local_xy=[game[0]-region.x,game[1]-region.y],
            region_local_basis='frozen_HitRegion_AABB_origin_not_rotated_sprite_uv',
            evidence=deepcopy(c.get('event_freshness')),producer=c.get('producer')))
    return dict(state='UNKNOWN',reason='NEW_PHYSICAL_IMPACT_NOT_CERTIFIED',
        camera_xy=None,game_xy=None,object_local_xy=None,physical_cause_certified=False,
        native_support_state=d.get('state','UNKNOWN'),native_support_reason=d.get('reason'),
        observation_state={'OBJECT_HIT':'SUPPORTED_LOCAL_CHANGE',
            'NO_OBJECT_HIT':'NO_SUPPORTED_IMPACT'}.get(d.get('state'),'UNKNOWN'),
        native_diagnostic_hypothesis=deepcopy(d.get('hypothesis')),
        qualified_by_existing_query_only=diagnostics,
        diagnostic_xy_status='MULTIPLE_UNRESOLVED' if len(points)>1 else
            'ONE_UNCERTIFIED_COORDINATE' if points else 'NO_SUPPORTED_COORDINATE',
        no_supported_impact_is_not_miss=True,no_physical_shot_established=False)


def analyze_region(shot, object_id):
    start=CLOCK();snap=one_snapshot(shot['snapshot'],object_id);region=snap.game_regions[0]
    original=shot['trace']['object_query_diagnostic'];events=[];maps_by_frame={};candidates=[]
    post_times=[f.timestamp for f in shot['posts']]
    if any(not math.isfinite(t) or t<snap.peak_ts or t>original['decision_ts'] for t in post_times) or any(
            a>=b for a,b in zip(post_times,post_times[1:])):
        raise ValueError('POST_OUTSIDE_RECORDED_CAUSAL_WINDOW')
    with patch.object(AnalysisGeometryV2221,'crop_array',subset_crop), \
            patch('src.engine.camera.object_physical_query.time.perf_counter',new=lambda:0.):
        qstart=CLOCK()
        q=ObjectPhysicalQuery(snap,shot['pre'],camera_to_game=shot['convert'],
            top_k=original['top_k'],margin_px=original['margin_camera_px'])
        query_ms=(CLOCK()-qstart)*1000
        tiles=list(q.tiles)  # retain geometry only after query clears its own tile list
        if not tiles:return dict(object_id=object_id,status='UNAVAILABLE',reason=q.result['reason']),{}
        if len(tiles)!=1:raise ValueError('single region unexpectedly has multiple tiles')
        tile=tiles[0];geo=tile['geometry'];maskstart=CLOCK()
        mask=exact_mask(region,geo,shot['convert']);search=tile['mask']>0
        mask_ms=(CLOCK()-maskstart)*1000
        history=np.stack([subset_crop(geo,f.frame_bgr) for f in shot['pre']])
        original_select=q._select_hypotheses
        def collect(cs,production):
            candidates.append(dict(frame_ts=q.last_frame_ts,candidates=deepcopy(cs)))
            return original_select(cs,production)
        q._select_hypotheses=collect
        for frame in shot['posts']:
            # Missing late tile is a recorded partial frame, not an invented image.
            try:post=subset_crop(geo,frame.frame_bgr)
            except ValueError as exc:
                if str(exc)!='REGION_CROP_UNAVAILABLE':raise
                events.append(dict(frame_ts=frame.timestamp,status='UNAVAILABLE',reason=str(exc)));continue
            used_by_query=not q.final
            if used_by_query:
                t=CLOCK();q.observe(frame,delivered_ts=frame.timestamp);query_ms+=(CLOCK()-t)*1000
            m,families=residual_maps(history,post,search,mask)
            points=[c for obs in candidates if obs['frame_ts']==frame.timestamp for c in obs['candidates']]
            if 'registered_residual' in m:
                families.append(residual_structure(m['registered_residual'],m['valid']>0,mask,points,geo))
            else:families.append(family('LOCAL_RESIDUAL_STRUCTURE',{},CLOCK(),available=False,reason='NO_RESIDUAL_MODEL'))
            families.append(family('PROJECTOR_EXPLAINABILITY',{},CLOCK(),available=False,
                reason='SHOT_RENDER_IDENTITY_AND_CALIBRATED_RESPONSE_UNAVAILABLE'))
            idx=len(events);maps_by_frame.update({f'f{idx:02d}_{key}':value for key,value in m.items()})
            events.append(dict(frame_ts=frame.timestamp,status='AVAILABLE',families=families,
                used_by_query=used_by_query,shadow_after_query_closed=not used_by_query,
                query_decision_ts=q.result.get('decision_ts') if q.final else None,
                recorded_candidate_count=len(points)))
        if not q.final:q.finish('UNCERTAIN','saved_post_sequence_exhausted',original['decision_ts'])
    maps_by_frame.update(exact_shape_mask=mask.astype(np.uint8),analysis_mask=search.astype(np.uint8))
    crops=shot['trace']['object_evidence_crops']
    provenance=[c for c in crops if c['timestamp'] in {f.timestamp for f in shot['pre']+shot['posts']} and
        c['camera_bbox'][0]<=geo.crop_x0 and c['camera_bbox'][1]<=geo.crop_y0 and
        c['camera_bbox'][2]>=geo.crop_x1 and c['camera_bbox'][3]>=geo.crop_y1]
    row=dict(schema=SCHEMA,shot_id=shot['sid'],object_id=object_id,status='AVAILABLE',contract=CONTRACT,
        scope='ONE_FROZEN_REGION_COUNTERFACTUAL_NOT_RECORDED_LIVE_DECISION',
        frozen_region=asdict(region),camera_bbox=[geo.crop_x0,geo.crop_y0,geo.crop_x1,geo.crop_y1],
        mapping=shot['mapping'],peak_ts=snap.peak_ts,recorded_cutoff_ts=original['decision_ts'],
        pre_frame_ts=[f.timestamp for f in shot['pre']],
        pre_status='RECEIPT_ORDERED_ONLY_NOT_500MS_GT_GUARDED_OR_CAUSALLY_CERTIFIED',
        pre_delta_ms=[(f.timestamp-snap.peak_ts)*1000 for f in shot['pre']],
        projected_frame=dict(status='UNAVAILABLE',reason='no_immutable_presented_image_in_capture',
            geometry_available=True,radiometric_transfer_available=False),
        recorded_multi_object_decision=deepcopy(original.get('object_decision')),
        frame_sources=provenance,frames=events,query_result=q.result,candidate_observations=candidates,
        result=region_result(q.result,region,shot['convert']),
        timings=dict(query_ms=query_ms,exact_mask_ms=mask_ms,total_ms=(CLOCK()-start)*1000),
        physical_object_truth='UNKNOWN',exact_camera_truth='UNAVAILABLE',
        source_context='known_holes_tape_edges_never_vetoed; saved_object_metadata_audit_only')
    return row,maps_by_frame


def semantic(value):
    """Stable analysis identity excludes CPU cost, never measurement values."""
    if isinstance(value,dict):return {k:semantic(v) for k,v in value.items() if k not in ('timing','timings','runtime_ms')}
    if isinstance(value,list):return [semantic(v) for v in value]
    return value


def run(capture,output,*,object_id=None):
    source=_development_path(capture).resolve();out=_development_path(output).resolve()
    if not out.is_relative_to(Path('/data/skjutbana/evaluation_runs')) or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('new output under evaluation_runs and outside capture required')
    if out.exists():raise ValueError('immutable output already exists')
    root,shots,hashes=replay.load(source)
    # Reuse the paired inventory for log-only/interrupted events (not aim ordinals).
    from automation.object_candidate_reconciliation import paired_event_ids
    known={s['sid'] for s in shots}
    for sid in paired_event_ids(root,(root/'runtime.log').read_text().splitlines()):
        if sid not in known:
            shots.append(dict(sid=sid,trace=None,boundary='NO_SAVED_QUERY_GEOMETRY_OR_CROPS'))
    shots.sort(key=lambda s:s['sid'])
    out.mkdir(parents=True)
    code={str(Path(p).resolve()):replay.digest(p) for p in
        (__file__,replay.__file__, 'src/engine/camera/object_physical_query.py',
         'src/engine/camera/candidate_generator_v2.py','src/engine/game_objects/geometry.py',
         'src/engine/offline/accuracy_verifier.py','src/engine/input/hit_input.py',
         'src/engine/input/object_hit_v2223.py','automation/object_candidate_reconciliation.py')}
    manifest=dict(schema=SCHEMA,input_capture=str(root),input_hashes=hashes,code_hashes=code,
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        object_id=object_id,contract=CONTRACT,status='IN_PROGRESS',completed=[])
    replay.save(out/'manifest.json',manifest);summary=[];timings=[]
    try:
        for shot in shots:
            ids=([object_id] if object_id else sorted(r.object_id for r in shot['snapshot'].game_regions if r.enabled)) if shot['trace'] else [object_id]
            for oid in ids:
                if not shot['trace']:
                    row=dict(shot_id=shot['sid'],object_id=oid,status='UNAVAILABLE',reason=shot['boundary'])
                    maps={}
                else:
                    try:row,maps=analyze_region(shot,oid)
                    except ValueError as exc:
                        if str(exc) not in ('REGION_CROP_UNAVAILABLE','ONE_FROZEN_REGION_REQUIRED'):raise
                        row=dict(shot_id=shot['sid'],object_id=oid,status='UNAVAILABLE',reason=str(exc));maps={}
                # Opaque index avoids accepting arbitrary path text from object IDs.
                name=f'entry_{len(summary):03d}'
                if maps:np.savez_compressed(out/(name+'_maps.npz'),**maps)
                replay.save(out/(name+'.json'),row)
                digest=hashlib.sha256(json.dumps(semantic(row),sort_keys=True,allow_nan=False).encode()).hexdigest()
                result=row.get('result',{})
                summary.append(dict(shot_id=shot['sid'],object_id=oid,status=row['status'],
                    reason=row.get('reason'),native_support=result.get('native_support_state'),
                    research_state=result.get('state','UNKNOWN'),xy_status=result.get('diagnostic_xy_status'),
                    analysis_sha256=digest,artifact=name+'.json',**row.get('timings',{})))
                for frame in row.get('frames',[]):
                    for f in frame.get('families',[]):timings.append(dict(shot_id=shot['sid'],object_id=oid,
                        frame_ts=frame['frame_ts'],family=f['family'],runtime_ms=f['runtime_ms'],availability=f['availability']))
                manifest['completed'].append(name);replay.save(out/'manifest.json',manifest)
        if any(replay.digest(p)!=h for p,h in hashes.items()):raise ValueError('source input changed during read-only audit')
        manifest['status']='COMPLETE'
    except BaseException as exc:
        manifest.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        replay.save(out/'manifest.json',manifest);replay.save(out/'summary.json',summary)
        for filename,rows in (('regions.csv',summary),('timing.csv',timings)):
            if rows:
                keys=list(dict.fromkeys(k for row in rows for k in row))
                with (out/filename).open('w') as f:
                    w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    text=['# Single-region research foundation',
        'Existing ObjectPhysicalQuery supports one region. This is counterfactual saved-crop research, not live parity or physical validation.',
        'All physical results remain UNKNOWN. Removing other regions does not certify physical damage. No accuracy or oracle is measured.',
        'Projector explanation is UNAVAILABLE: no immutable shot-time rendered image or calibrated transfer. Camera homography alone is insufficient.',
        'Maps preserve raw change, bounded-motion/photometric residual, observed PRE-envelope residual, exact shape and outside context. No material-cause decomposition is claimed.',
        'CPU timings exclude source loading/hash checks and map writes. Shared registration cost belongs to NEW_LOCAL_RESIDUAL; motion family is the same computation, not a second witness.',
        '', '| shot | region | available | native support only | research result | XY status |',
        '|---|---|---|---|---|---|']
    text += [f"| {r['shot_id']} | {r['object_id']} | {r['status']} | {r['native_support']} | {r['research_state']} | {r['xy_status']} |" for r in summary]
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(output=str(out),status=manifest['status'],entries=len(summary),physical_acceptance='PENDING'),indent=2))
    return summary
