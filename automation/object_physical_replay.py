"""Read-only CAN_OBJECT replay and temporal forensics using the LIVE verifier.

No physical labels are inferred. Continuation is an offline counterfactual, with
recorded presentation contamination marked. Clock-neutral replay reproduces
evidence/selection, not physical delivery/processing latency. Outputs must be new.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as S
from unittest.mock import patch

import cv2
import numpy as np

from src.engine.camera import object_physical_query as live
from src.engine.camera.candidate_generator_v2 import CandidateGeneratorV2
from src.engine.game_objects.geometry import shape_contains
from src.engine.input.object_hit_v2223 import (ObjectShotSnapshotV2223,
    CameraHitRegionV240,GameHitRegionV240)


def mature_candidate_bridge(snapshot, camera_to_game, packet, cutoff, *, next_peak=None):
    """Read-only research projection of ONE delivered native pool, never live authority.

    Delivery and evidence clocks are both required. Native confirmation is kept
    distinct from exact-XY confirmation: nonzero/missing neighborhood offsets
    abstain. No score ordering, target priority, XY movement or pool union through
    time (which would incorrectly make revoked support permanent).
    """
    from src.engine.offline.causal_candidates import classify, finite
    from src.engine.physical_trace import _diagnostic_source
    sid = snapshot.shot_id
    rows = []
    available, credible = set(), set()
    requested = [r for r in snapshot.game_regions
                 if r.enabled and r.metadata.get('physical_query', {}).get('mode') == 'OBJECT']
    valid_packet = (sid is not None and sid > 0 and packet.get('shot_id') == sid
                    and packet.get('coordinate_space') == 'camera'
                    and packet.get('peak_ts') == snapshot.peak_ts
                    and finite(packet.get('frame_ts')) is not None
                    and finite(packet.get('available_at')) is not None)
    packet_class = classify(cutoff=cutoff, observed_at=packet.get('available_at'),
        evidence_at=packet.get('frame_ts'), owner=packet.get('shot_id'),
        shot_id=sid, next_peak=next_peak)
    for index, candidate in enumerate(packet.get('candidates', [])):
        c = deepcopy(candidate)  # Audit provenance stays external to physical evidence.
        x, y = finite(c.get('camera_x')), finite(c.get('camera_y'))
        owner = c.get('v2224_producer_shot_id')
        availability = classify(cutoff=cutoff, observed_at=packet.get('available_at'),
            evidence_at=c.get('timestamp'), owner=owner, shot_id=sid, next_peak=next_peak)
        if packet_class != 'CAUSALLY_AVAILABLE': availability = packet_class
        if not valid_packet or x is None or y is None or finite(c.get('timestamp')) is None:
            availability = 'UNKNOWN'
        if (c.get('v251_shot_id',sid) != sid or
                c.get('v253_peak_ts',snapshot.peak_ts) != snapshot.peak_ts):
            availability = 'CROSS_EVENT_OR_FUTURE'
        ids = []
        if x is not None and y is not None and camera_to_game is not None:
            gx, gy = camera_to_game(x, y)
            for region in requested:
                shape = region.metadata.get('v250_object_snapshot', {}).get('shape')
                inside = (shape_contains(shape, gx, gy) if shape else
                          region.x <= gx <= region.x+region.width and region.y <= gy <= region.y+region.height)
                if inside: ids.append(region.object_id)
        causal = (availability == 'CAUSALLY_AVAILABLE' and
                  c['timestamp'] >= snapshot.peak_ts)
        if causal: available.update(ids)
        offsets = ('v252_evidence_best_dx', 'v252_evidence_best_dy',
                   'v2225_confirm_best_dx', 'v2225_confirm_best_dy')
        reasons = []
        if not causal: reasons.append('not_proven_available_in_event_at_cutoff')
        if not ids: reasons.append('outside_requested_exact_shapes_or_mapping_unavailable')
        if packet.get('stage') != 'LOCAL_CONFIRMATION': reasons.append('proposal_only')
        if packet.get('full_rescue') is not False: reasons.append('global_rescue_or_scope_unknown')
        for field in ('v252_evidence_ready', 'v252_fresh_physical', 'v2225_local_confirm'):
            if (finite(c.get(field)) or 0) <= .5: reasons.append('missing_'+field)
        if any(finite(c.get(field)) != 0. for field in offsets):
            reasons.append('neighborhood_proof_is_not_exact_xy_proof')
        if not packet.get('complete'): reasons.append('incomplete_pool')
        supported = not reasons
        if supported: credible.update(ids)
        rows.append(dict(index=index, camera_xy=[c.get('camera_x'), c.get('camera_y')],
            source=_diagnostic_source(c), candidate=c, availability=availability,
            inside_object_ids=ids, supported=supported, reasons=reasons))
    unique = len(credible) == 1 and packet.get('complete') and valid_packet
    return dict(authority='RESEARCH_ONLY', shot_id=sid, rows=rows,
        available_object_ids=sorted(available), supported_object_ids=sorted(credible),
        object_id=next(iter(credible)) if unique else None,
        state='OBJECT_HIT' if unique else 'UNKNOWN',
        reason='unique_exact_xy_native_qualification' if unique else
               'multiple_supported_objects' if len(credible)>1 else 'insufficient_evidence',
        no_physical_impact_established=False, scores_compared=False)


def safe_file(root,relative):
    root=Path(root)
    p=root/relative
    if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():
        raise ValueError('input path escapes captured session')
    return p


def change_signature(reference, images, mask, surround):
    """Descriptive signed change, with surrounding median brightness removed.

    Probe/mask selection is forensic, not an inference proposal or physical GT.
    No acceptance threshold, detector gate or coordinate is produced here.
    """
    if not np.any(mask) or not np.any(surround):
        raise ValueError('nonempty probe and surrounding context required')
    reference=np.asarray(reference,np.float32)
    rows=[]
    for im in images:
        signed=np.asarray(im,np.float32)-reference
        illumination=float(np.median(signed[surround]))
        local=signed[mask]-illumination
        rows.append(dict(signed_mean=float(np.mean(local)),
            absolute_mean=float(np.mean(np.abs(local))),
            dark_fraction=float(np.mean(local<0)),illumination=illumination))
    return rows


def surface_probe(shot, bbox):
    """Timeline for a manually reviewed image feature, NEVER an inferred GT XY.

    Early reference and pre-presentation step amplitude are retrospective
    descriptions. All frame timestamps and presentation contamination survive.
    This does not feed the live query or nominate a candidate.
    """
    x,y,w,h=map(int,bbox)
    first=frame(shot,shot['frames'][0]).frame_bgr
    if min(w,h)<=0 or x<12 or y<12 or x+w+12>first.shape[1] or y+h+12>first.shape[0]:
        raise ValueError('probe outside image or insufficient context')
    images=[np.asarray(frame(shot,f).frame_bgr[y-12:y+h+12,x-12:x+w+12]) for f in shot['frames']]
    mask=np.zeros(images[0].shape,bool);mask[12:12+h,12:12+w]=True
    surround=np.ones_like(mask);surround[8:16+h,8:16+w]=False
    reference=np.median(images[:5],axis=0)
    rows=change_signature(reference,images,mask,surround)
    t=shot['trace'];presented=t['object_gameplay_diagnostic']['reaction_presented_ts']
    for f,row in zip(shot['frames'],rows):
        row.update(timestamp=f['timestamp'],dt_ms=(f['timestamp']-t['peak_ts'])*1000.,
            after_reaction=f['timestamp']>=presented,
            used_as_object_pre=f['timestamp'] in t['object_query_diagnostic']['pre_frame_ts'])
    before=[r for r in rows if not r['after_reaction']]
    amplitude=float(np.median([r['signed_mean'] for r in before[-5:]]))
    # Report an explicitly descriptive half-step crossing; never correct clocks
    # or choose PRE using this retrospective, manually selected feature.
    crossed=[r['signed_mean']*np.sign(amplitude)>=abs(amplitude)*.5 for r in before]
    onset=None
    for i in range(1,len(crossed)):
        if all(crossed[i:]) and not crossed[i-1]:
            onset=[before[i-1]['dt_ms'],before[i]['dt_ms']];break
    snap,convert,_=snapshot(shot)
    inside={g.object_id:sum(shape_contains(g.metadata['v250_object_snapshot']['shape'],*convert(px,py))
        for py in range(y,y+h) for px in range(x,x+w)) for g in snap.game_regions}
    return dict(camera_bbox_xywh=list(bbox),coordinate_status='MANUAL_IMAGE_FEATURE_NOT_GT_OR_HYPOTHESIS',
        early_reference_frame_ts=[f['timestamp'] for f in shot['frames'][:5]],
        pre_presentation_signed_step=amplitude,descriptive_half_step_bracket_ms=onset,
        bbox_pixels_inside_frozen_shapes=inside,rows=rows)


def load_capture(run):
    run=Path(run).resolve()
    if any('S03' in p for p in run.parts):raise ValueError('forbidden session')
    binding=json.loads((run/'binding.json').read_text())
    if binding.get('planned_session_id')!='CAN_OBJECT':raise ValueError('CAN_OBJECT binding required')
    root=Path(binding['trace_root'])
    if any('S03' in p for p in root.parts) or '_CAN_OBJECT_' not in root.name:
        raise ValueError('only explicit CAN_OBJECT capture roots are supported')
    if root.is_symlink():raise ValueError('capture root must be direct')
    logs={}
    for line in (run/'runtime.log').read_text().splitlines():
        if '[OBJECT-SHOT] ' in line:
            row=json.loads(line.split('[OBJECT-SHOT] ',1)[1]);logs[int(row['shot_id'])]=row
    shots=[]
    for p in sorted((root/'shots').glob('shot_*/trace.json')):
        t=json.loads(p.read_text());sid=int(t['shot_id'])
        if sid not in logs:raise ValueError('missing frozen geometry in joined shot log')
        frames={f['timestamp']:f for f in t['frames']}
        for c in t['object_evidence_crops']:
            pixels=np.load(safe_file(p.parent,c['path']),allow_pickle=False)
            if hashlib.sha256(pixels.tobytes()).hexdigest()!=c['sha256']:
                raise ValueError('OBJECT input hash mismatch')
            f=frames.get(c['timestamp'])
            if f is None:raise ValueError('OBJECT input has no matching full-frame timestamp')
            full=np.load(safe_file(p.parent,f['path']),mmap_mode='r',allow_pickle=False)
            x,y,x1,y1=map(int,c['camera_bbox'])
            if not np.array_equal(pixels,full[y:y1,x:x1]):
                raise ValueError('global/OBJECT pixels differ; cannot silently mix inputs')
        shots.append(dict(root=p.parent,trace=t,log=logs[sid],frames=sorted(frames.values(),key=lambda f:f['timestamp'])))
    return shots


def frozen_mapping(log):
    """Recover only geometric correspondence, never fit from physical labels.

    Logged observed XYs get their exact logged game XY. A projective fit over
    these calibration correspondences handles newly nominated coordinates;
    alternate-pair holdout residual must be <.001 game pixel.
    """
    pairs={}
    for f in log['candidate_ledger']:
        for o in f['objects']:
            for c in o['candidates']:
                if c.get('game_xy') is not None:pairs[tuple(c['camera_xy'])]=c['game_xy']
    if len(pairs)<8:raise ValueError('insufficient frozen mapping correspondences')
    xy=np.array(list(pairs),np.float64);uv=np.array(list(pairs.values()),np.float64)
    train=np.arange(len(xy))%2==0
    h,_=cv2.findHomography(xy[train],uv[train],0)
    projected=cv2.perspectiveTransform(xy.reshape(-1,1,2),h).reshape(-1,2)
    error=np.linalg.norm(projected-uv,axis=1)
    if not np.isfinite(error).all() or max(error)>.001:raise ValueError('frozen mapping reconstruction failed')
    full_h,_=cv2.findHomography(xy,uv,0)
    def convert(x,y):
        if (x,y) in pairs:return tuple(pairs[x,y])
        p=full_h@np.array([x,y,1.]);return float(p[0]/p[2]),float(p[1]/p[2])
    return convert,dict(method='logged_pairs_with_validated_projective_extension',pairs=len(pairs),
        holdout_max_game_px=float(max(error[~train])),matrix=full_h.tolist(),not_physical_GT=True)


def snapshot(shot):
    t,log=shot['trace'],shot['log'];convert,meta=frozen_mapping(log)
    camera=tuple(CameraHitRegionV240(r['object_id'],*r['camera_aabb']) for r in log['frozen_regions'])
    game=tuple(GameHitRegionV240(r['object_id'],*r['game_aabb'],
        metadata={'v250_object_snapshot':{'shape':r['exact_shape']}}) for r in log['frozen_shapes'])
    return ObjectShotSnapshotV2223(t['shot_id'],t['peak_ts'],t['peak_ts'],(),
        game_regions=game,camera_regions=camera,transform_available=True),convert,meta


def frame(shot,entry):
    return S(timestamp=entry['timestamp'],frame_bgr=np.load(safe_file(shot['root'],entry['path']),mmap_mode='r',allow_pickle=False))


def make_query(shot):
    snap,convert,meta=snapshot(shot);qdata=shot['trace']['object_query_diagnostic']
    pre=[frame(shot,f) for f in shot['frames'] if f['timestamp'] in qdata['pre_frame_ts']]
    return live.ObjectPhysicalQuery(snap,pre,camera_to_game=convert,
        top_k=qdata['top_k'],margin_px=qdata['margin_camera_px']),meta


def object_choice(result):
    if result.get('status')!='PHYSICAL_HIT':return None
    for c in result['hypotheses']:
        if c['confirmed']:
            ids=sorted(a['object_id'] for a in c['regions'] if a['inside_exact_shape'] is True)
            if ids:return ids[0]
    return None


def replay(shot, *, continue_frames=False, older_surface=False):
    """Same verifier and production thresholds. No wall-clock cutoff accidents.

    older_surface is a labeled diagnostic substitution of the earliest three
    causal history frames; it explicitly bypasses live's .5s PRE-age policy.
    It is NOT a production/reference-policy recommendation.
    """
    q,mapping=make_query(shot);t=shot['trace'];original=t['object_query_diagnostic']
    if older_surface:
        old=[frame(shot,f) for f in shot['frames'][:3]]
        for tile in q.tiles:
            stack=np.stack([tile['geometry'].crop_array(f.frame_bgr).astype(np.float32) for f in old])
            tile['pre']=np.median(stack,axis=0);tile['noise']=np.median(np.abs(stack-tile['pre']),axis=0)
        q.pre_times=[f.timestamp for f in old]
    records=[];observed=[];select=q._select_hypotheses
    def collect(candidates,production):
        chosen=select(candidates,production);observed[:]=deepcopy(candidates);return chosen
    q._select_hypotheses=collect
    if continue_frames:
        # Diagnostic extension only: keep existing crop arrays after each result.
        def keep(status,reason,now=None):
            q.result=q.payload(status,reason,now);return q.result
        q.finish=keep
    wanted=set(f['frame_ts'] for f in original['candidate_ledger'])
    posts=[f for f in shot['frames'] if f['kind']=='post' and (continue_frames or f['timestamp'] in wanted)]
    start=time.perf_counter()
    with patch.object(live.time,'perf_counter',return_value=0.),patch.object(live,'DEADLINE_S',3. if continue_frames else .35):
        for entry in posts:
            ts=entry['timestamp'];result=q.observe(frame(shot,entry),delivered_ts=ts)
            records.append(dict(timestamp=ts,dt_ms=(ts-t['peak_ts'])*1000.,
                after_reaction=ts>=t['object_gameplay_diagnostic']['reaction_presented_ts'],
                result=deepcopy(result),observed=deepcopy(observed),object_choice=object_choice(result)))
    cost=(time.perf_counter()-start)*1000.
    actual=records[-1]['result'];same_xy=[c['camera_xy'] for c in actual['hypotheses']]==[c['camera_xy'] for c in original['hypotheses']]
    error=max((abs(a['evidence_score']-b['evidence_score']) for a,b in zip(actual['hypotheses'],original['hypotheses'])),default=0.)
    return dict(shot_id=t['shot_id'],mode='continued_offline_evidence' if continue_frames else 'recorded_two_frame_replay',
        older_surface_diagnostic=older_surface,mapping=mapping,records=records,processing_total_ms=cost,
        matches_original_xy=same_xy,max_original_score_error=error,object_choice=object_choice(actual),
        latency='Clock-neutral CPU replay; camera timestamps are read completion, not exposure; future delivery was not recorded')


def patch_evidence(pre,post,xy, *, local_registration=False):
    """Explaining local visual change, not estimating impact position."""
    x,y=map(int,xy);radius=24
    a=np.asarray(pre[y-radius:y+radius+1,x-radius:x+radius+1],np.float32)
    b=np.asarray(post[y-radius:y+radius+1,x-radius:x+radius+1],np.float32)
    if a.shape!=(49,49) or b.shape!=a.shape:raise ValueError('candidate patch outside frame')
    mask=np.ones(a.shape,np.uint8);cv2.circle(mask,(radius,radius),8,0,-1)
    reg={};reference=a
    if local_registration:
        # Fit surrounding pixels only: no damage-center contribution to motion.
        reference,reg=CandidateGeneratorV2._register_current(None,b,a,roi=mask,
            cfg={'registration_max_dimension':256,'registration_min_shift_px':0.})
    current,_=CandidateGeneratorV2._normalise_photometry(reference,b,roi=mask)
    signed=current-reference;absolute=np.abs(signed)
    metric,ok,score=live.ObjectPhysicalQuery._evidence(absolute,np.zeros_like(a),signed,radius,radius)
    gradient=np.hypot(cv2.Sobel(a,cv2.CV_32F,1,0,ksize=3)/8.,cv2.Sobel(a,cv2.CV_32F,0,1,ksize=3)/8.)
    yy,xx=np.ogrid[:49,:49];inner=(xx-radius)**2+(yy-radius)**2<=4
    ring=((xx-radius)**2+(yy-radius)**2>=64)&((xx-radius)**2+(yy-radius)**2<=400)
    # First-order motion/illumination explanation fit on surrounding pixels.
    gx=cv2.Sobel(a,cv2.CV_32F,1,0,ksize=3)/8.;gy=cv2.Sobel(a,cv2.CV_32F,0,1,ksize=3)/8.
    design=np.stack([gx,gy,np.ones_like(a)],axis=-1)
    coef=np.linalg.lstsq(design[ring],(b-a)[ring],rcond=None)[0]
    residual=(b-a)-design@coef
    metric_res,ok_res,score_res=live.ObjectPhysicalQuery._evidence(np.abs(residual),np.zeros_like(a),residual,radius,radius)
    return dict(center_abs=metric.center_abs,ring_abs=metric.ring_abs,compact_abs=metric.compact_abs,
        score=score,localized=ok,center_signed=float(np.mean(signed[inner])),
        pre_gradient=float(np.mean(gradient[inner])),patch_change=float(np.mean(absolute)),
        registration=reg,linear_motion_xy=coef[:2].tolist(),illumination=float(coef[2]),
        ring_motion_explained_fraction=1.-float(np.sum(residual[ring]**2))/max(1.,float(np.sum((b-a)[ring]**2))),
        residual_score=score_res,residual_localized=ok_res)


def timelines(shot,replayed):
    t=shot['trace'];full=replayed['records'][-1]['observed'];probes={}
    for o in t['object_query_diagnostic']['per_object']:
        inside=[c for c in full if c['confirmed'] and any(a['object_id']==o['object_id'] and a['inside_exact_shape'] for a in c['regions'])]
        if inside:probes[o['object_id']]=max(inside,key=lambda c:c['evidence_score'])
    selected=shot['frames'];pre=frame(shot,next(f for f in selected if f['timestamp']==max(t['object_query_diagnostic']['pre_frame_ts']))).frame_bgr
    older=np.median(np.stack([frame(shot,f).frame_bgr for f in selected[:3]]),axis=0).astype(np.uint8)
    rows=[]
    for entry in selected:
        post=frame(shot,entry).frame_bgr
        for oid,c in probes.items():
            for name,reference,registered in [('current_pre',pre,False),('current_pre_local_registration',pre,True),('early_history',older,False)]:
                m=patch_evidence(reference,post,c['camera_xy'],local_registration=registered)
                rows.append(dict(shot_id=t['shot_id'],object_id=oid,camera_xy=c['camera_xy'],reference=name,
                    timestamp=entry['timestamp'],dt_ms=(entry['timestamp']-t['peak_ts'])*1000.,
                    after_reaction=entry['timestamp']>=t['object_gameplay_diagnostic']['reaction_presented_ts'],**m))
    return rows


def input_fingerprints(capture,shots):
    paths={Path(capture)/'runtime.log',Path(capture)/'binding.json'}
    for shot in shots:
        paths.add(shot['root']/'trace.json')
        paths.update(safe_file(shot['root'],f['path']) for f in shot['frames'])
        paths.update(safe_file(shot['root'],f['path']) for f in shot['trace']['object_evidence_crops'])
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run(capture,output,probe_spec=None,probes_only=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shots=load_capture(capture);summary=[]
    inputs=input_fingerprints(capture,shots)
    (output/'input_hashes.json').write_text(json.dumps(inputs,indent=2))
    repo=Path(__file__).resolve().parents[1]
    provenance=dict(command=sys.argv,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        working_tree=subprocess.check_output(['git','status','--short'],cwd=repo,text=True),
        replay_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        verifier_source_sha256=hashlib.sha256(Path(live.__file__).read_bytes()).hexdigest(),
        capture=str(Path(capture).resolve()),physical_labels_used_by_replay=False,
        probe_spec=probe_spec,probes_only=probes_only)
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    for shot in shots:
        sid=shot['trace']['shot_id']
        if probe_spec:
            probes=[surface_probe(shot,bbox) for bbox in probe_spec.get(str(sid),[])]
            (output/f'shot_{sid}_surface_probes.json').write_text(json.dumps(probes,indent=2))
        if probes_only:continue
        base=replay(shot)
        if not base['matches_original_xy'] or base['max_original_score_error']>1e-6:
            raise ValueError(f'shot {sid}: live verifier replay mismatch')
        rows=timelines(shot,base)
        variants={'baseline':base,'continued':replay(shot,continue_frames=True),
                  'older_pre':replay(shot,continue_frames=True,older_surface=True)}
        for name,value in variants.items():
            (output/f'shot_{sid}_{name}.json').write_text(json.dumps(value,indent=2))
        (output/f'shot_{sid}_timelines.json').write_text(json.dumps(rows,indent=2))
        summary.append(dict(shot_id=sid,baseline_object=base['object_choice'],matches_original=True,
            score_error=base['max_original_score_error'],cpu_ms=base['processing_total_ms'],mapping=base['mapping']))
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    if input_fingerprints(capture,shots)!=inputs:
        raise ValueError('source changed during analysis; results invalid')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'output_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe-spec',type=Path,help='JSON shot ID -> list of reviewed feature boxes [x,y,w,h]; never GT')
    p.add_argument('--probes-only',action='store_true',help='measure feature timelines without rerunning completed replay')
    a=p.parse_args()
    if a.probes_only and not a.probe_spec:p.error('--probes-only requires --probe-spec')
    run(a.capture,a.output,json.loads(a.probe_spec.read_text()) if a.probe_spec else None,a.probes_only)
