"""Physical image forensics using exact recorded confirmation frames. Read-only."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2,numpy as np
from src.engine.offline.track_replay import reconstruct_legacy,current_exact_replay
from src.engine.offline.registered_impact import prepare_pair,extract
from src.engine.shot_fast_v2225 import _candidate_metrics_local


def distance(track,gt):return math.hypot(track['camera_x']-gt['camera_x'],track['camera_y']-gt['camera_y'])


def montage(rows,path):
    names=('pre','registered_pre','post','aligned_post','abs_registered','raw','registered','global_median','ring_median','ring_affine')
    canvas=np.full((70+len(rows)*230,200+len(names)*210,3),245,np.uint8)
    cv2.putText(canvas,'PRE camera coordinates; signed residual: red darkening / blue brightening, scale +/-20; circles r4/8/12',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
    for j,name in enumerate(names):cv2.putText(canvas,name,(200+j*210,50),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
    for i,(label,patches) in enumerate(rows):
        for n,line in enumerate(label.splitlines()):cv2.putText(canvas,line,(5,105+i*230+n*20),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1)
        for j,name in enumerate(names):
            im=patches[name]
            if j<4:color=cv2.cvtColor(np.clip(im,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            elif name=='abs_registered':color=cv2.cvtColor(np.clip(im/20*255,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            else:
                v=np.clip(im/20,-1,1);color=np.full((*v.shape,3),255.,np.float32)
                color[:,:,0]-=255*np.maximum(v,0);color[:,:,1]-=255*np.abs(v);color[:,:,2]-=255*np.maximum(-v,0);color=color.astype(np.uint8)
            color=cv2.resize(color,(198,198),interpolation=cv2.INTER_NEAREST)
            for radius in (4,8,12):cv2.circle(color,(99,99),radius*6,(0,180,0) if radius==4 else (0,160,200),1)
            canvas[70+i*230:268+i*230,200+j*210:398+j*210]=color
    if not cv2.imwrite(str(path),canvas):raise OSError('Could not save patch montage')


def physical(root,output):
    output.mkdir(parents=True,exist_ok=False);events=[];all_tracks=[];examples=[]
    for path in sorted((root/'shots').glob('*/trace.json')):
        t=json.loads(path.read_text());gt=json.loads((path.parent/'ground_truth.json').read_text());sid=t['shot_id']
        snapshot=t['decision_input'].get('complete_track_audit') or reconstruct_legacy(t)
        current_exact_replay(snapshot,t['decision_input']['deterministic_selection'])
        cf=t['decision_input']['local_confirmation'];pre_ref=next(f for f in t['frames'] if f['kind']=='pre_snapshot')
        post_ref=next(f for f in t['frames'] if f['timestamp']==cf['frame_ts'])
        if post_ref['timestamp']>t['decision_input']['timestamp']:raise ValueError('Post-decision image forbidden')
        pre=np.load(path.parent/pre_ref['path'],mmap_mode='r');post=np.load(path.parent/post_ref['path'],mmap_mode='r')
        # Verify actual PRE and CURRENT pixels independently against every saved
        # confirmation value, before deriving a new feature or comparison label.
        checks=[]
        for c in cf['candidates']:
            m=_candidate_metrics_local(pre,post,c['camera_x'],c['camera_y'],patch_radius=11,search_radius=4)
            for attr in ('center_abs','ring_abs','compact','peak_abs','darkening','best_dx','best_dy'):
                error=abs(getattr(m,attr)-c['v2225_confirm_'+attr]);checks.append(error)
                if error>1e-5:raise AssertionError(('confirmation image mismatch',sid,attr,error))
        eligible=[r for r in snapshot['tracks'] if r['eligible']]
        near=min(eligible,key=lambda r:distance(r,gt));winner=current_exact_replay(snapshot)
        # Estimation window is determined by all recorded candidates, never GT.
        x0=max(0,int(min(r['camera_x'] for r in snapshot['tracks']))-40);y0=max(0,int(min(r['camera_y'] for r in snapshot['tracks']))-40)
        x1=min(pre.shape[1],int(max(r['camera_x'] for r in snapshot['tracks']))+41);y1=min(pre.shape[0],int(max(r['camera_y'] for r in snapshot['tracks']))+41)
        pair=prepare_pair(np.array(pre[y0:y1,x0:x1]),np.array(post[y0:y1,x0:x1]),origin=(x0,y0))
        features={}
        for r in snapshot['tracks']:
            f=extract(pair,(r['camera_x'],r['camera_y']));features[r['track_id']]=f
            all_tracks.append(dict(shot=sid,track=r,image_features=f,evaluation_gt_distance=distance(r,gt)))
        records=[];panels=[]
        roles=[('nearest_eligible',near),('current_winner',winner)]
        if winner['track_id']==near['track_id']:roles=roles[:1]
        for role,r in roles:
            f,patches=extract(pair,(r['camera_x'],r['camera_y']),return_patches=True)
            label='physical_true' if role=='nearest_eligible' and distance(r,gt)<=42 else ('physical_false' if role=='current_winner' and distance(r,gt)>42 else 'no_oracle_nearest')
            obs=[a for b in snapshot['associations'] for a in b['records'] if a.get('track_id')==r['track_id']]
            record=dict(shot=sid,role=role,label=label,gt_distance=distance(r,gt),track=r,association_history=obs,features=f,
                first_seen_after_peak=r['first_seen_ts']-t['peak_ts'],last_seen_after_peak=r['last_seen_ts']-t['peak_ts'])
            records.append(record);examples.append(record)
            panels.append((f'{role}\n{r["source"]}, rank {r["final_rank"]}\nGT distance {distance(r,gt):.2f}',patches))
        # GT-centered patches characterize physical rendering only. They are
        # excluded from candidate features and runtime/research ranking inputs.
        gt_features,gt_patches=extract(pair,(gt['camera_x'],gt['camera_y']),return_patches=True)
        panels.append(('GT diagnostic ONLY',gt_patches));montage(panels,output/f'shot_{sid:02}.png')
        events.append(dict(shot=sid,peak=t['peak_ts'],decision=t['decision_input']['timestamp'],pre=pre_ref,post=post_ref,
            registration=pair.registration,origin=pair.origin,global_offset=pair.global_offset,
            confirmation_checks=len(checks),confirmation_max_error=max(checks),pairs=records,gt_diagnostic_features=gt_features))
    report=dict(status='RESEARCH_ONLY_PHYSICAL_DEVELOPMENT',input_session=str(root),events=events,tracks=all_tracks,examples=examples)
    (output/'physical_features.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(shot=e['shot'],registration=e['registration'],offset=e['global_offset'],checks=e['confirmation_checks'],max_error=e['confirmation_max_error']) for e in events],indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    physical(a.root,a.output)
if __name__=='__main__':main()
