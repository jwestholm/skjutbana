"""Supervised real donor qualification/extraction; never detector inference.

Fixed existing onset and 3-sigma residual-component criteria. QUALIFIED means
bounded research patch extraction, not calibrated mechanical segmentation.
"""
from __future__ import annotations
import argparse,json,math,hashlib
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from automation.causal_pre_onset_research import read_guard,digest,write
from src.engine.offline.common_verifier import camera_local_patches
from src.engine.offline.clean_physical_change import local_contrast,temporal_step_evidence
from src.engine.offline.registered_impact import residual_features

INVENTORY=Path('/data/skjutbana/evaluation_runs/exact_witness_20260927_01/source_inventory/source_manifest.json')


def damage_layer(pre,post):
    """Reuse registered_impact's ring/noise/component convention at labelled GT.

    Raw same-camera pixels remain available. No warp of the whole source texture.
    The mask is a residual proxy, not a manually segmented true hole boundary.
    """
    pre=np.asarray(pre,np.float32);post=np.asarray(post,np.float32)
    if pre.shape!=(49,49) or post.shape!=pre.shape or not np.isfinite([pre,post]).all():raise ValueError('Finite 49x49 pairs required')
    yy,xx=np.mgrid[-24:25,-24:25];r=np.hypot(xx,yy);ring=(r>=8)&(r<=12)
    raw=pre-post;offset=float(np.median(raw[ring]));delta=raw-offset
    sigma=max(1.,1.4826*float(np.median(np.abs(delta[ring]-np.median(delta[ring])))))
    binary=((np.abs(delta)>3*sigma)&(r<=12)).astype(np.uint8)
    count,labels,_,_=cv2.connectedComponentsWithStats(binary,8)
    ids=[i for i in range(1,count) if np.any((labels==i)&(r<=4))]
    selected=max(ids,key=lambda i:float(np.abs(delta)[labels==i].sum()),default=0)
    mask=(labels==selected) if selected else np.zeros_like(binary,dtype=bool)
    localized=bool(mask.any() and not np.any(mask&(r>=11)))
    return dict(pre=pre.copy(),post=post.copy(),signed_delta=delta,raw_signed_delta=raw,
        absolute_delta=np.abs(delta),damage_mask=mask.astype(np.uint8),appearance=post.copy()),dict(
        ring_offset=offset,ring_sigma=sigma,mask_area=int(mask.sum()),localized=localized,
        touches_footprint_boundary=bool(np.any(mask&(r>=11))),metrics=residual_features(delta),
        mask_semantics='3sigma camera residual component touching labelled r4; NOT segmented mechanical damage truth')


def qualify(signal,frames,*,cutoff_known,complete,lineage_unique,caveats,patches):
    times=[f.get('timestamp') for f in frames];known=bool(times and all(t is not None for t in times))
    evidence=temporal_step_evidence(signal,times if known else None)
    reasons=[];arrays=None;layer=None
    if not complete:reasons.append('INCOMPLETE_AVAILABLE_FRAME_SET')
    if not lineage_unique:reasons.append('SOURCE_LINEAGE_AMBIGUOUS')
    if not cutoff_known:reasons.append('CAUSAL_CUTOFF_UNAVAILABLE')
    if not known:reasons.append('FRAME_TIMING_UNAVAILABLE')
    if caveats:reasons.append('SOURCE_LABEL_CAVEAT')
    if not evidence['supported']:reasons.append(evidence['state'])
    if evidence['supported']:
        k=evidence['onset_ordinal'];arrays,layer=damage_layer(patches[k-1],patches[-1])
        if not layer['localized']:reasons.append('CHANGE_COMPONENT_NOT_LOCALIZABLE')
        last=np.asarray(signal[-3:]);stable=bool(np.max(np.abs(last-np.median(last)))<=2*evidence['noise'])
        if not stable:reasons.append('LAST_POST_NOT_SETTLED_UNDER_EXISTING_2NOISE_RULE')
        if not all(f['kind']=='post' for f in frames[-3:]):reasons.append('THREE_POST_OBSERVATIONS_UNAVAILABLE')
    status='QUALIFIED' if not reasons else 'UNKNOWN' if not lineage_unique else 'PARTIAL'
    return dict(status=status,reasons=reasons,evidence=evidence,layer=layer,
        pre_demonstrably_before_local_step=bool(evidence['supported']),
        post_persistent=bool(evidence['supported']),spatially_localizable=layer['localized'] if layer else None,
        claim='research donor with observed local change; receipt is not exposure; residual mask is approximate'),arrays


def run(output,review_manifest=None):
    output.mkdir(parents=True,exist_ok=False);data=json.loads(INVENTORY.read_text());sources=data['events']
    roots=[Path(s['source_session']) for s in sources];read_guard(output.parent,roots);cv2.setNumThreads(1)
    write(output/'protocol.json',dict(criterion='frozen temporal_step_evidence; existing 3sigma ring residual component; within r12 and not touching r11 boundary; existing 2noise stability on final three posts',
        labels='supervised GT allowed only here, never inference',qualification_scope='research extraction; not true mechanical segmentation',
        cutoff='recorded decision and strictly before next peak; no-cutoff material PARTIAL',inventory_sha256=digest(INVENTORY)))
    reviews=json.loads(Path(review_manifest).read_text()) if review_manifest else {}
    rows=[];hashes={str(INVENTORY):digest(INVENTORY)};ids=Counter(s['source_hole_id'] for s in sources)
    if review_manifest:hashes[str(review_manifest)]=digest(review_manifest)
    (output/'donors').mkdir();tiles=[]
    for source in sources:
        root=Path(source['source_session'])/'shots'/f"shot_{int(source['event'].split(':')[1]):08d}";tracepath=root/'trace.json'
        trace=json.loads(tracepath.read_text()) if tracepath.exists() else None;frames=[];complete=True
        if trace:
            hashes[str(tracepath)]=digest(tracepath);cut=(trace.get('decision_input') or {}).get('timestamp')
            limit=min(cut or math.inf,trace.get('next_audio_peak_ts') or math.inf)
            for f in sorted(trace['frames'],key=lambda f:f['timestamp']):
                if f['kind'] not in ('pre_history','post') or f['timestamp']>limit or f['timestamp']>=(trace.get('next_audio_peak_ts') or math.inf):continue
                frames.append(dict(path=str(root/f['path']),kind=f['kind'],timestamp=f['timestamp'],camera_sequence=f.get('acquisition',{}).get('sequence')))
            context=trace.get('context',{});cutoff_known=cut is not None
        else:
            cutoff_known=False;context={}
            for kind,pattern in [('pre_history','pre_history_*.npy'),('post','post_*.npy')]:
                frames.extend(dict(path=str(p),kind=kind,timestamp=None,camera_sequence=None) for p in sorted((root/'frames').glob(pattern)))
        patches=[];available=[]
        for f in frames:
            path=Path(f['path'])
            if not path.exists():complete=False;continue
            try:
                im=np.load(path,mmap_mode='r',allow_pickle=False)
                patches.append(camera_local_patches(im,[source['gt']])[0]);available.append(f)
                # Full frames are immutable originals; patch hashes also bind exact queried pixels.
                f['patch_sha256']=hashlib.sha256(patches[-1].tobytes()).hexdigest()
            except (ValueError,OSError):complete=False
        if len(patches)<6:
            result=dict(status='UNUSABLE',reasons=['INSUFFICIENT_EXTRACTABLE_FRAMES']);arrays=None
        else:
            signal=local_contrast(np.stack(patches));result,arrays=qualify(signal,available,cutoff_known=cutoff_known,
                complete=complete,lineage_unique=ids[source['source_hole_id']]==1,caveats=source['caveat'],patches=patches)
            result['signal']=signal.tolist()
        name=source['event'].replace(':','_')
        row=dict(source,result=result,frame_records=available,source_frames_complete=complete,lineage_unique=ids[source['source_hole_id']]==1,
            projection_background_context={k:v for k,v in context.items() if k in ('scene','game','game_id','shot_game_context','surface_revision','camera_geometry','geometry')},
            context_identity_available_keys=sorted(context),qualification_does_not_change_human_truth=True)
        row['completeness_scope']='All declared frames in the selected causal interval readable; does not certify unavailable requested/native frames'
        row['recorded_capture_coverage']=trace.get('capture_history') if trace else None
        row['recorded_writer_errors']=trace.get('writer_errors') if trace else None
        row['automatic_status']=result['status']
        if result['status']=='QUALIFIED':
            review=reviews.get(source['event'])
            pair_hashes={k:hashlib.sha256(arrays[k].tobytes()).hexdigest() for k in ('pre','post')}
            if not review or review.get('patch_hashes')!=pair_hashes:
                result['status']='PARTIAL';result['reasons'].append('VISUAL_PRE_DAMAGE_REVIEW_PENDING_OR_STALE')
            elif review['decision']!='ACCEPT_RESEARCH_PAIR':
                result['status']='PARTIAL';result['reasons'].append(review['reason'])
            row['visual_review']=review
        if arrays is not None:
            k=result['evidence']['onset_ordinal'];row['chosen_pre']=available[k-1];row['chosen_post']=available[-1]
            # Preserve partial diagnostics too, but only qualified files can enter the builder.
            dest=output/'donors'/f'{name}.npz';np.savez_compressed(dest,**arrays)
            row['extraction_path']=str(dest);row['extraction_sha256']=digest(dest)
            panels=[]
            for key in ('pre','post','signed_delta','damage_mask'):
                im=arrays[key];im=(np.clip(im*4+128,0,255) if key=='signed_delta' else im*255 if key=='damage_mask' else im).astype(np.uint8)
                panels.append(cv2.resize(im,(147,147),interpolation=cv2.INTER_NEAREST))
            tile=cv2.cvtColor(np.hstack(panels),cv2.COLOR_GRAY2BGR);tile=cv2.copyMakeBorder(tile,28,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
            cv2.putText(tile,name+' '+result['status'],(5,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,0),1)
            tiles.append(tile)
        rows.append(row);print(source['event'],result['status'],','.join(result['reasons']),flush=True)
    for i in range(0,len(tiles),8):cv2.imwrite(str(output/f'contact_{i//8}.png'),np.vstack(tiles[i:i+8]))
    summary=dict(counts=dict(Counter(r['result']['status'] for r in rows)),total=len(rows),qualified_sessions=sorted({r['source_session'] for r in rows if r['result']['status']=='QUALIFIED'}),
        scope='supervised research donors, no physical accuracy validation, no model training')
    write(output/'qualification.json',dict(summary=summary,events=rows));write(output/'input_hashes.json',hashes)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--review-manifest',type=Path)
    args=p.parse_args();run(args.output,args.review_manifest)
