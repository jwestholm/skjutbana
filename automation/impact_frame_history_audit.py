"""Read-only image chronology. Labels locate diagnostic patches, never detector inputs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
from src.engine.offline.registered_impact import sample


def run(root,output):
    output.mkdir(parents=True,exist_ok=False)
    traces=[(p,json.loads(p.read_text())) for p in sorted((root/'shots').glob('*/trace.json'))]
    gts=[json.loads((p.parent/'ground_truth.json').read_text()) for p,t in traces]
    rows=[];patches={g['shot_id']:[] for g in gts}
    yy,xx=np.mgrid[:33,:33];radius=np.hypot(xx-16,yy-16);center=radius<=4;ring=(radius>=8)&(radius<=12)
    for p,t in traces:
        for f in sorted(t['frames'],key=lambda f:f['timestamp']):
            frame=np.load(p.parent/f['path'],mmap_mode='r')
            for g in gts:
                patch=sample(frame,(g['camera_x'],g['camera_y']),(0,0))
                row=dict(label_shot=g['shot_id'],frame_shot=t['shot_id'],kind=f['kind'],path=str(p.parent/f['path']),timestamp=f['timestamp'],relative_to_frame_event=f['timestamp']-t['peak_ts'],center_mean=float(patch[center].mean()),ring_mean=float(patch[ring].mean()),contrast=float(patch[ring].mean()-patch[center].mean()))
                rows.append(row);patches[g['shot_id']].append((row,patch))
    for g in gts:
        sid=g['shot_id'];series=patches[sid]
        # Ten earliest event frames, then all own PRE/POST frames.
        chosen=[next((r,a) for r,a in series if r['frame_shot']==n) for n in range(1,11)]
        chosen += [(r,a) for r,a in series if r['frame_shot']==sid]
        cols=10;cell=140;canvas=np.full((((len(chosen)+cols-1)//cols)*185,cols*cell,3),245,np.uint8)
        for k,(r,a) in enumerate(chosen):
            x=(k%cols)*cell;y=(k//cols)*185
            color=cv2.cvtColor(np.clip(a,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR);color=cv2.resize(color,(132,132),interpolation=cv2.INTER_NEAREST)
            cv2.circle(color,(66,66),16,(0,180,0),1)
            canvas[y+40:y+172,x:x+132]=color
            cv2.putText(canvas,f"event {r['frame_shot']} {r['relative_to_frame_event']:+.2f}s",(x,y+12),cv2.FONT_HERSHEY_SIMPLEX,.31,(0,0,0),1)
            cv2.putText(canvas,f"{r['kind']} C={r['contrast']:.1f}",(x,y+28),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)
        cv2.imwrite(str(output/f'gt_{sid:02}_chronology.png'),canvas)
    (output/'frame_history.json').write_text(json.dumps(dict(status='DIAGNOSTIC_LABEL_COORDINATES_ONLY',input_session=str(root),rows=rows),indent=2)+'\n')
    print(json.dumps(dict(frames=len(rows)//len(gts),patches=len(rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,a.output)
