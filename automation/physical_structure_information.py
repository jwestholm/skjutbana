"""Descriptive OFFLINE candidate geometry. Never selection features or vetoes."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2
import numpy as np
from automation.causal_pre_onset_research import write,read_guard,digest
from automation import chronological_pre_research as chrono


def spatial_values(xy,proposals,pre=None):
    """A frame-local neighborhood; repeated XY and correlated operators not votes.

    All supplied proposals arise from camera image processing and share one broad
    image-response family. Producer variety is reported separately, not confidence.
    """
    points=np.asarray([c['camera_xy'] for c in proposals],float).reshape(-1,2)
    if len(points) and not np.isfinite(points).all():raise ValueError('finite camera XY required')
    raw_dist=np.linalg.norm(points-np.asarray(xy),axis=1)
    unique=np.unique(points,axis=0);dist=np.linalg.norm(unique-np.asarray(xy),axis=1)
    local=unique[dist<=12];near=raw_dist<=12
    producers=sorted({str(c.get('source_name',c.get('source','UNKNOWN'))) for c,k in zip(proposals,near) if k})
    result={f'neighbor_count_r{r}':int(np.sum((dist<=r)&(dist>0))) for r in (3,6,12)}
    result.update(raw_proposal_count_r12=int(np.sum(near)),unique_xy_count_r12=len(local),
        unique_producer_count_r12=None if 'UNKNOWN' in producers else len(producers),
        known_producer_count_r12=len([p for p in producers if p!='UNKNOWN']),producers=producers,
        unique_source_family_count_r12=int(len(local)>0),source_family='correlated_camera_image_response',
        isolatedness=result['neighbor_count_r12']==0,cluster_radius=None,cluster_density=None,
        compactness=None,isotropy=None,linearity=None,pca_eigenvalue_ratio=None,line_edge_alignment=None,
        pre_edge_strength=None,pre_edge_coherence=None)
    axis=None
    if len(local)>=2:
        centered=local-local.mean(axis=0);cov=centered.T@centered/len(local);vals,vecs=np.linalg.eigh(cov)
        radius=float(np.max(np.linalg.norm(centered,axis=1)))
        result.update(cluster_radius=radius,cluster_density=float(len(local)/(math.pi*radius**2)),
            isotropy=float(vals[0]/vals[1]),linearity=float(1-vals[0]/vals[1]),
            pca_eigenvalue_ratio=float(vals[1]/vals[0]) if vals[0]>1e-12 else None)
        axis=vecs[:,1]
        if len(local)>=3:
            hull=cv2.convexHull(local.astype(np.float32));area=cv2.contourArea(hull);per=cv2.arcLength(hull,True)
            result['compactness']=float(4*math.pi*area/per**2) if per>0 else None
    if pre is not None:
        # Same PRE gradient family as existing soft edge evidence; no threshold.
        gx=cv2.Sobel(pre.astype(np.float32),cv2.CV_32F,1,0,ksize=3)/8
        gy=cv2.Sobel(pre.astype(np.float32),cv2.CV_32F,0,1,ksize=3)/8
        h,w=pre.shape;yy,xx=np.mgrid[:h,:w];rr=(xx-(w-1)/2)**2+(yy-(h-1)/2)**2
        result['pre_edge_strength']=float(np.hypot(gx,gy)[rr<=9].mean())
        x,y=gx[rr<=144],gy[rr<=144];tensor=np.array([[np.mean(x*x),np.mean(x*y)],[np.mean(x*y),np.mean(y*y)]])
        v,q=np.linalg.eigh(tensor);result['pre_edge_coherence']=float((v[1]-v[0])/(v.sum()+1e-12))
        if axis is not None and v[1]>0:
            tangent=np.array([-q[1,1],q[0,1]]);result['line_edge_alignment']=float(abs(np.dot(axis,tangent)))
    return result


def patch_at(image,xy):
    x,y=xy
    if min(x,y)<13 or x+14>=image.shape[1] or y+14>=image.shape[0]:return None
    ix,iy=int(x),int(y)
    return cv2.getRectSubPix(image[iy-13:iy+15,ix-13:ix+15].astype(np.float32),(27,27),(x-ix+13,y-iy+13))


def run(package,output_name='spatial'):
    if Path(output_name).name!=output_name:raise ValueError('Output name must be one directory name')
    out=package/output_name;out.mkdir(exist_ok=False);read_guard(package,[chrono.SESSION,chrono.D02]);cv2.setNumThreads(1)
    recovery=json.loads((chrono.BASE/'kompetensprov_recovery_20260927_01/recovered_results.json').read_text())['events']
    events=chrono.descriptors(d02_sids=range(1,7));summaries=[];hashes={}
    with (out/'measurements.jsonl').open('x') as dest:
        for e in events:
            eventrows=[];key=e['id'].replace(':','_')
            ts=[json.loads(l) for l in (package/'retention'/(key+'_transitions.jsonl')).read_text().splitlines()]
            if e['id'].startswith('KOMP'):
                c=recovery[e['sid']-1]['emission_record'];winner=[c['camera_x'],c['camera_y']]
            else:
                trace=json.loads(e['trace'].read_text());a=trace['decision_input']['complete_track_audit']
                c=next(c for c in a['tracks'] if c['final_rank']==1);winner=[c['camera_x'],c['camera_y']]
            batches=[dict(domain='chronological_capped8',identity=t['ordinal'],proposals=t['generated'],pre=e['pre'][t['ordinal']-1]['path'],
                motion=t['physical_maps']['registration'],scope='RECONSTRUCTED_PREFIX') for t in ts]
            if e['id'].startswith('D02'):
                seen=set()
                for s in trace['stages']:
                    p=s.get('pipeline')
                    if not isinstance(p,dict) or s['timestamp']>e['cutoff'] or s.get('candidate_pool_shot_id')!=e['sid'] or p['source_frame_ts'] in seen:continue
                    seen.add(p['source_frame_ts']);u=p['upstream'];origin=u['crop_origin_camera'];cs=[]
                    # One pool only; no double-count of its downstream copies.
                    for c in u['pipeline'].get('pre_limit_candidates',[]):cs.append(dict(c,camera_xy=[c['camera_x']+origin[0],c['camera_y']+origin[1]]))
                    if cs:batches.append(dict(domain='recorded_pre_limit',identity=p['source_frame_ts'],proposals=cs,pre=e['pre'][-1]['path'],motion=None,scope='RECORDED_PROPOSALS_PRE_REFERENCE_LATE'))
                    hybrid=[]
                    for record in u['pipeline'].get('hybrid_merge',{}).get('records',[]):
                        c=record['candidate']
                        hybrid.append(dict(c,camera_xy=[c['camera_x']+origin[0],c['camera_y']+origin[1]],source_name=record.get('family','UNKNOWN')))
                    if hybrid:batches.append(dict(domain='recorded_hybrid_input',identity=p['source_frame_ts'],proposals=hybrid,pre=e['pre'][-1]['path'],motion=None,scope='RECORDED_PROPOSALS_PRE_REFERENCE_LATE'))
            for b in batches:
                image=np.load(b['pre'],mmap_mode='r',allow_pickle=False);hashes[str(b['pre'])]=digest(b['pre'])
                points=list(dict.fromkeys(tuple(c['camera_xy']) for c in b['proposals']))
                for xy in points+([] if tuple(winner) in points else [tuple(winner)]):
                    anchor=xy==tuple(winner) and xy not in points
                    values=spatial_values(xy,b['proposals'],patch_at(image,xy))
                    row=dict(event=e['id'],domain=b['domain'],frame=b['identity'],xy=xy,diagnostic_winner_anchor=anchor,
                        current_winner=xy==tuple(winner),scope=b['scope'],motion=b['motion'],values=values)
                    eventrows.append(row)
                del image
            # Truth is joined after this event's complete measurements; never used to form neighborhoods.
            gp=e['root']/'ground_truth.json';gt=json.loads(gp.read_text()) if e['id'].startswith('KOMP') or e['sid'] in (1,3) else None
            for r in eventrows:
                r['gt_distance']=math.dist(r['xy'],[gt['camera_x'],gt['camera_y']]) if gt else None
                r['truth_group']='GT_NEAR_5' if gt and r['gt_distance']<=5 and not r['diagnostic_winner_anchor'] else 'CURRENT_WRONG_WINNER' if r['current_winner'] and (not gt or r['gt_distance']>42) else 'FALSE_CONTROL' if not gt and not r['diagnostic_winner_anchor'] else 'OTHER'
                dest.write(json.dumps(r,allow_nan=False)+'\n')
            for domain in sorted({r['domain'] for r in eventrows}):
                for group in ('GT_NEAR_5','CURRENT_WRONG_WINNER','FALSE_CONTROL'):
                    rs=[r for r in eventrows if r['domain']==domain and r['truth_group']==group]
                    if not rs:continue
                    fields={k:[r['values'][k] for r in rs if isinstance(r['values'][k],(float,int))] for k in rs[0]['values']}
                    summaries.append(dict(event=e['id'],domain=domain,group=group,count=len(rs),
                        median={k:float(np.median(v)) for k,v in fields.items() if v},
                        fraction_cluster_5_unique=float(np.mean([r['values']['unique_xy_count_r12']>=5 for r in rs])),
                        fraction_isolated=float(np.mean([r['values']['isolatedness'] for r in rs]))))
            print(e['id'],'spatial rows',len(eventrows),flush=True)
    write(out/'summary.json',summaries);write(out/'input_hashes.json',hashes)
    write(out/'contract.json',dict(selection_affected=False,neighbors='same source frame only; unique XY; excludes the queried point',
        density='N/(pi*max centroid radius^2); null for <2',compactness='convex hull 4*pi*area/perimeter^2; null for <3',
        pca='isotropy=lambda_min/lambda_max; linearity=1-isotropy; ratio null if singular',
        source_families='all camera response operators grouped together; producers separately counted, no independent-vote claim',
        limitations=['capped8 already spatially thinned','prelimit includes correlated sources; no full uncapped KOMP live ledger survived','motion is global registration where reconstructed; no per-candidate motion vector/wave field','late PRE edge affinity may include new damage','repeated observations are not independent samples']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--output-name',default='spatial')
    args=p.parse_args();run(args.package,args.output_name)
