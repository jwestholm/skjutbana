"""Explicit CAN capture geometry/onset characterization; no ranking or promotion.

Onsets scan merged OBJECT context with existing residual/response functions.
Reviewed damage boxes are never read. Every estimate retains its causal cutoff.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np

from automation.object_physical_replay import load_capture, frame, make_query, input_fingerprints
from automation.physical_foundation_check import write_new
from src.engine.offline.clean_physical_change import local_residual, response_map


def temporal_steps(images, mask, timestamps):
    """One bounded two-segment least-squares comparison, without manual probes.

    Latest frame of the earlier segment is a proposed PRE per localized step,
    not event truth. Multiple distinct steps mean event ambiguity. Only the
    supplied causal prefix is inspected; no age correction or oldest-frame rule.
    """
    if len(images)<6:return []
    history=np.asarray(images,np.float32)
    local=local_residual(history[0],history)
    local=np.stack([cv2.GaussianBlur(im,(3,3),0) for im in local])
    csum=np.cumsum(local,axis=0);n=len(history)
    best=np.zeros(mask.shape,np.float32);when=np.zeros(mask.shape,np.int16)
    amplitude=np.zeros(mask.shape,np.float32)
    for k in range(3,n-2):
        delta=(csum[-1]-csum[k-1])/(n-k)-csum[k-1]/k
        improvement=(k*(n-k)/n)*delta*delta
        update=improvement>best;best[update]=improvement[update];when[update]=k
        amplitude[update]=delta[update]
    support=response_map(np.sqrt(best))*mask
    maxima=(support==cv2.dilate(support,np.ones((9,9),np.uint8)))&(mask>0)
    ys,xs=np.where(maxima)
    order=np.lexsort((xs,ys,-support[ys,xs]));result=[]
    for j in order:
        x,y=int(xs[j]),int(ys[j]);k=int(when[y,x])
        if k<3 or any((x-r['local_xy'][0])**2+(y-r['local_xy'][1])**2<144 for r in result):continue
        values=local[:,y,x];before=values[:k];after=values[k:]
        result.append(dict(local_xy=[x,y],step_support=float(support[y,x]),signed_step=float(amplitude[y,x]),
            before_std=float(np.std(before)),after_std=float(np.std(after)),
            suggested_pre_ts=timestamps[k-1],appearance_bracket=[timestamps[k-1],timestamps[k]],
            before_frames=k,after_frames=n-k,cutoff_ts=timestamps[-1],curve=values.tolist()))
        if len(result)>=3:break
    return result


def run(capture, output, observed_boxes=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shots=load_capture(capture);hashes=input_fingerprints(capture,shots)
    write_new(output/'input_hashes.json',hashes)
    rows=[]
    for shot in shots:
        query,_=make_query(shot);peak=shot['trace']['peak_ts']
        # First two recorded OBJECT post frames only. No game reaction frames.
        cutoff=shot['trace']['object_query_diagnostic']['candidate_ledger'][-1]['frame_ts']
        selected=[f for f in shot['frames'] if f['timestamp']<=cutoff]
        start=time.perf_counter();candidates=[]
        for tile in query.tiles:
            g=tile['geometry'];images=[g.crop_array(frame(shot,f).frame_bgr) for f in selected]
            steps=temporal_steps(images,tile['mask'],[f['timestamp'] for f in selected])
            for step in steps:
                step['camera_xy']=list(g.local_to_camera(*step['local_xy']))
                step['suggested_pre_relative_audio_ms']=(step['suggested_pre_ts']-peak)*1000
                step['crop_bbox']=[g.crop_x0,g.crop_y0,g.crop_x1,g.crop_y1]
            candidates.extend(steps)
        strongest=max(candidates,key=lambda c:c['step_support'],default=None)
        rows.append(dict(shot_id=shot['trace']['shot_id'],candidates=candidates,
            processing_ms=(time.perf_counter()-start)*1000,
            scope='merged_frozen_object_context_only',labels_used=False,
            delivery_provenance='Historical receipt timestamps only; delivery/acquisition not reconstructable',
            proposed_event_pre=strongest,
            reason='Strongest localized persistent step is a development PRE proposal, not live authority; scene revision/delivery constraints remain unverified'))
    write_new(output/'event_specific_pre.json',dict(status='RESEARCH_ONLY_PRE_PROPOSAL',
        no_manual_probe_input=True,live_policy_changed=False,events=rows))
    if observed_boxes:
        log=shots[0]['log'];residuals=[]
        for region in log['frozen_regions']:
            oid=region['object_id'];x,y,w,h=region['camera_aabb']
            bx,by,bw,bh=observed_boxes[oid]
            residuals.append(dict(object_id=oid,predicted_xywh=[x,y,w,h],observed_xywh=[bx,by,bw,bh],
                predicted_ltrb=[x,y,x+w,y+h],observed_ltrb=[bx,by,bx+bw,by+bh],
                predicted_center=[x+w/2,y+h/2],observed_center=[bx+bw/2,by+bh/2],
                dx=bx+bw/2-x-w/2,dy=by+bh/2-y-h/2,
                scale_x=bw/w,scale_y=bh/h,width_error=bw-w,height_error=bh-h))
        write_new(output/'geometry_residuals.json',dict(
            method='Manual visible silhouette bounds on earliest shot-1 frame; excludes labels/shadows',
            edge_uncertainty_camera_px=3,limitation='Blur and intentional hit-shape inset limit scale estimates; not calibration or damage GT',
            observed_boxes=observed_boxes,objects=residuals,
            median_translation=np.median([[r['dx'],r['dy']] for r in residuals],axis=0).tolist()))
    if input_fingerprints(capture,shots)!=hashes:raise ValueError('source changed during research')
    write_new(output/'output_hashes.json',{p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--observed-outline-boxes',type=Path)
    a=p.parse_args();run(a.capture,a.output,json.loads(a.observed_outline_boxes.read_text()) if a.observed_outline_boxes else None)
