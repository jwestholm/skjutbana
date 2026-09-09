"""Bounded rescue of pre-filter contour locations from saved causal masks; offline only."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from automation.accuracy_physical_dataset import causal_stages, json_read, load_context
from automation.accuracy_proposal_forensics import raw_contours
from src.engine.offline.accuracy_verifier import features, patch


def contour_rescue(mask,context,budget=256):
    if budget<1:raise ValueError('Positive budget required')
    candidates=raw_contours(mask,context.origin);height,width=mask.shape
    cells={}
    for c in candidates:
        x,y=c['camera_x']-context.origin[0],c['camera_y']-context.origin[1]
        if c['area']<2 or not .8<=c['radius']<=35 or min(x,y)<25 or x+25>=width or y+25>=height:continue
        if not context.roi[round(y),round(x)]:continue
        scale=int(np.searchsorted([2,4,8,16],c['radius']))
        key=(min(3,int(4*y/height)),min(3,int(4*x/width)),scale)
        cells.setdefault(key,[]).append(c)
    for group in cells.values():group.sort(key=lambda c:(-c['circularity'],c['camera_y'],c['camera_x']))
    selected=[];depth=0
    while len(selected)<budget:
        layer=[cells[key][depth] for key in sorted(cells) if len(cells[key])>depth]
        if not layer:break
        for c in layer:
            if any(np.hypot(c['camera_x']-p['camera_x'],c['camera_y']-p['camera_y'])<8 for p in selected):continue
            selected.append(c)
            if len(selected)>=budget:break
        depth+=1
    yy,xx=np.mgrid[-24:25,-24:25];radius=np.hypot(xx,yy)
    inside=radius<=4;ring=(radius>=8)&(radius<=12)
    for c in selected:
        xy=(c['camera_x'],c['camera_y']);pre=patch(context.pre,xy,context.origin)
        changes=np.stack([pre-patch(frame,xy,context.origin) for frame in context.post])
        contrast=changes[:,inside].mean(axis=1)-changes[:,ring].mean(axis=1)
        support=[t for t,v in zip(context.post_times,contrast) if abs(v)>.5]
        c.update(ready=len(support)>=3 and max(support)-min(support)>=.09,
                 support_frames=len(support),evidence_timestamp=max(support) if support else None)
    return selected


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    dataset=json_read(source/'dataset.json');original=np.load(source/'features.npy',allow_pickle=False)
    records=[];vectors=[];events=[]
    for event in dataset['events']:
        start=time.perf_counter();event=copy.deepcopy(event);old_indices=event['indices'];event['indices']=[]
        for index in old_indices:
            record=dataset['records'][index]
            if record['pool']=='expanded':continue
            record=dict(record,index=len(records));event['indices'].append(record['index']);records.append(record);vectors.append(original[index])
        path=Path(event['trace_path']);trace=json_read(path);context,_metadata=load_context(path,trace,dataset['reference'])
        stages=[s for s in causal_stages(trace) if 'candidate_mask' in s.get('evidence_maps',{}) and
                trace['peak_ts']<=s.get('window_debug',{}).get('frame_ts',float('-inf'))<=context.cutoff]
        positions=[]
        if stages:
            stage=stages[-1]
            mask=np.load(path.parent/stage['evidence_maps']['candidate_mask']['path'],allow_pickle=False)
            if mask.shape!=context.pre.shape:raise ValueError('Mask/crop mismatch')
            frame_ts=stage['window_debug']['frame_ts']
            if not trace['peak_ts']<=frame_ts<=context.cutoff:raise ValueError('Noncausal proposal mask')
            positions=contour_rescue(mask,context,dataset['proposal_budget'])
        else:
            event['proposal_unavailable']='No owned candidate mask with a causal POST source timestamp recorded'
        gt=event['truth']['impacts'][0] if event['truth']['state']=='SINGLE_IMPACT' else None
        for candidate in positions:
            xy=(candidate['camera_x'],candidate['camera_y']);names,vector=features(context,xy)
            if list(names)!=dataset['feature_names']:raise ValueError('Feature schema mismatch')
            dist=float(np.hypot(xy[0]-gt['camera_x'],xy[1]-gt['camera_y'])) if gt else None
            record=dict(index=len(records),event=event['event'],session=event['session'],pool='expanded',camera_x=xy[0],camera_y=xy[1],
                        gt_distance=dist,training_label=1 if gt and dist<=10 else 0 if not gt or dist>42 else None,
                        ready=candidate['ready'],source_metadata='CONTOUR_RESCUE',track_id=None,final_rank=None,
                        support_frames=candidate['support_frames'],evidence_timestamp=candidate['evidence_timestamp'])
            event['indices'].append(record['index']);records.append(record);vectors.append(vector)
        event['counts']['expanded']=len(positions);event['rescue_build_ms']=(time.perf_counter()-start)*1000
        events.append(event);print(event['event'],'rescue',len(positions),flush=True)
    dataset.update(events=events,records=records,proposal_channel='retrospective causal mask contour rescue; full live detection replay unavailable',parent_dataset=str(source))
    np.save(output/'features.npy',np.asarray(vectors));(output/'dataset.json').write_text(json.dumps(dataset,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.output)
