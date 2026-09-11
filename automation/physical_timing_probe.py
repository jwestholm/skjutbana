"""Read-only camera patch timeline at human labels; never reassigns ground truth."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from automation.physical_session_audit import audit


def run(root,mapping,out):
    out.mkdir(parents=True,exist_ok=False);result=audit(root,mapping);rows=[];summary=[]
    for shot in result['shots']:
        gt=shot['ground_truth']
        if not gt:continue
        directory=root/'shots'/f"shot_{int(shot['shot_id']):08d}";t=json.loads((directory/'trace.json').read_text());series=[]
        x,y=int(round(gt['camera_x'])),int(round(gt['camera_y']));yy,xx=np.mgrid[-10:11,-10:11];radius=xx*xx+yy*yy
        for f in t['frames']:
            a=np.load(directory/f['path'],mmap_mode='r',allow_pickle=False);patch=a[y-10:y+11,x-10:x+11].astype(float)
            if patch.shape!=(21,21):continue
            center=float(np.mean(patch[radius<=4]));ring=float(np.mean(patch[(radius>=36)&(radius<=100)]))
            row={'shot_id':shot['shot_id'],'kind':f['kind'],'timestamp':f['timestamp'],'relative_ms':1000*(f['timestamp']-t['peak_ts']),
                 'center_mean':center,'ring_mean':ring,'ring_minus_center':ring-center,'path':f['path']}
            rows.append(row);series.append(row)
        series.sort(key=lambda r:r['timestamp'])
        summary.append({'shot_id':shot['shot_id'],'earliest':series[0] if series else None,
                        'snapshot':next((r for r in series if r['kind']=='pre_snapshot'),None),
                        'last_post':next((r for r in reversed(series) if r['kind']=='post'),None)})
    with (out/'timeline.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['shot_id']);writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'shots':summary,'limitations':['Camera timestamps are host-read timestamps; audio timestamps are chunk processing estimates, not synchronized exposure/acoustic hardware time.','Intensity contrast is diagnostic only, not a calibrated hole-onset classifier.']},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--mapping',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.root,json.loads(a.mapping.read_text()) if a.mapping else None,a.output)
