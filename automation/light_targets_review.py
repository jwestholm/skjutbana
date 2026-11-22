"""Read-only Ljusjakten follow-up; approximate crop replay, never a physical/live scheduling claim."""
from __future__ import annotations
import argparse
import json
import hashlib
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from automation.can_dense_offline import Frames, load_shots, mapper, decide
from automation import can_peak_dataset as peak_dataset
from src.engine.camera.dense_peak_model import PeakModel
from src.engine.game_objects.geometry import shape_contains


def review(session, log, truth, model_path=None):
    session = Path(session)
    meta = json.loads((session/'session.json').read_text())
    settings = meta['settings']['content/settings.json']['content']
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp)/'settings.json'; p.write_text(json.dumps(settings)); to_game=mapper(p)
    shots=load_shots(log); frames=Frames(session)
    truth_rows={int(t['shot_id']):t for t in map(json.loads,Path(truth).read_text().splitlines()) if t.get('kind')=='shot'}
    model=PeakModel(model_path) if model_path else None
    old_bright=peak_dataset.BRIGHT
    if model: peak_dataset.BRIGHT=model.uses_bright  # isolated offline export; restore below
    results=[]
    try:
        for sid, shot in sorted(shots.items()):
            nxt=min((s['peak_ts'] for s in shots.values() if s['peak_ts']>shot['peak_ts']+.15),default=math.inf)
            if nxt < shot['peak_ts'] + float(os.environ.get('SKJUTBANA_OBJECT_DENSE_DEADLINE_S',1.2)):
                raise ValueError('This crop replay requires disjoint shot windows; use live-path replay for doubles')
            # Reject substituted PREs rather than silently accepting nearest unrelated frames.
            if any(min(abs(frames.ts-t))>.002 for t in shot.get('pre_frame_ts',[])):
                raise ValueError(f'missing exact PRE for shot {sid}')
            oid,peaks,why=decide(shot,frames,to_game)
            row=dict(shot_id=sid,truth=truth_rows.get(sid,{}).get('truth_object'),live=shot.get('object_id'),offline=oid,
                     reason=why,peaks=peaks[:4],mapping='saved_session_settings',next_peak=nxt if math.isfinite(nxt) else None)
            if model:
                candidates=peak_dataset.shot_rows(shot,frames,to_game,nxt)
                shapes={p['object_id']:p.get('shape') for p in shot.get('per_object',[])}
                ps=model.p([r['f'] for r in candidates]) if candidates else []
                values=[]
                for c,p in zip(candidates,ps):
                    inside=[k for k,sh in shapes.items() if sh and shape_contains(sh,*c['game_xy'])]
                    values.append(dict(p=float(p),inside=inside,xy=c['game_xy'],s=c['s']))
                outs={v['inside'][0] if len(v['inside'])==1 else ('AMBIG' if v['inside'] else 'OUTSIDE') for v in values if v['p']>=model.ambig_p}
                row['shadow']=dict(high_p_outcomes=sorted(outs),top=sorted(values,key=lambda x:-x['p'])[:6],ambig_p=model.ambig_p)
            results.append(row)
            # Bound decoded 4K memory to one shot.
            frames.cache.clear()
    finally:
        frames.fh.close()
        peak_dataset.BRIGHT=old_bright
    labelled=[r for r in results if r['shot_id'] in truth_rows]
    negative=[r for r in labelled if r['truth'] is None]
    return dict(kind='APPROXIMATE_OFFLINE_CROP_REPLAY',session=str(session),
                inputs_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (log,truth,session/'session.json')},
                model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest() if model_path else None,
                environment={k:v for k,v in os.environ.items() if k.startswith(('SKJUTBANA_OBJECT_DENSE','SKJUTBANA_H15','SKJUTBANA_H13','SKJUTBANA_DETECTOR_LUMA','CAN_OFF_','CAN_PEAK_'))},
                n=len(results),agreement=sum(r['offline']==r['live'] for r in results),
                correct=sum(r['truth'] is not None and r['offline']==r['truth'] for r in labelled),
                wrong_object=sum(r['offline'] is not None and r['truth'] is not None and r['offline']!=r['truth'] for r in labelled),
                negative_controls=len(negative),false_awards=sum(r['offline'] is not None for r in negative),
                limitations=['No live tile pixel mask','Truth labels object/box, not exact hole XY','No negative controls means no false-positive bound','Model shadow features are recomputed, not stored live tiles'],rows=results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('session','log','truth','out'):p.add_argument(arg)
    p.add_argument('--model');a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError('Preserve baseline: choose a new output')
    result=review(a.session,a.log,a.truth,a.model)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','inputs_sha256')}))

if __name__=='__main__':main()
