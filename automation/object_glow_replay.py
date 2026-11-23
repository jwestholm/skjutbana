"""Replay recorded local-glow image evidence, never the current machine's settings.

python3 -m automation.object_glow_replay SOURCE_SESSION NEW_OUTPUT_DIR
Reads MJPG (including camera_bin_store compression) and the source events added
by object_glow. No camera, display, PCM detector, GT or gameplay side effects.
This is measurement replay, not a claim of full live-loop-equivalent replay.
"""
import argparse,bisect,hashlib,json
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
import cv2
from src.engine.precision_flash import FlashConfig
from src.engine.precision_object_glow import GlowConfig,glow_mask,measure_glow
from src.engine.input.object_hit_v2223 import GameHitRegionV240
from src.engine.game_objects.geometry import shape_bounds


def projection_from_state(state):
    from src.engine.lens_model import undistort_points
    H=np.asarray(state['H'],dtype=float);lens=state.get('lens')
    def points(p):
        p=np.asarray(p,float).reshape(-1,2)
        if lens:p=undistort_points(p,lens)
        return cv2.perspectiveTransform(p.reshape(-1,1,2),H).reshape(-1,2)
    return points


def event_windows(path):
    pending=None
    with path.open() as f:
        for line in f:
            e=json.loads(line)
            if e.get('kind')=='object_glow_presented':
                if pending is not None:yield pending,None
                pending=e
            elif e.get('kind')=='object_glow_result':
                result=e['result']
                if pending is not None and abs(result['on']-pending['presented_ts'])<1e-6:
                    yield pending,result;pending=None
        if pending is not None:yield pending,None


def replay(session,out):
    from src.engine.source_recording import SourceSessionReader
    from src.engine.camera.camera_manager import apply_frame_transform
    from src.engine.camera_bin_store import open_camera_bin
    reader=SourceSessionReader(session);session=Path(session);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows=iter(reader._rows(session/'camera_index.jsonl'));current=next(rows,None);previous_ts=-1e18
    reference=None;reference_key=None;previous_mask=None;results=[]
    with open_camera_bin(session) as handle:
        for event,live in event_windows(session/'events.jsonl'):
            if live is None:
                results.append(dict(valid=False,reason='missing_or_cancelled_live_result',on=event['presented_ts']));continue
            required=('config','flash_config','projection','regions','viewport')
            if any(event.get(k) is None for k in required):raise ValueError('Incomplete recorded glow provenance')
            cfg=GlowConfig(**event['config']);flash=FlashConfig(**event['flash_config']);on,off=live['on'],live['off']
            if event.get('reference'):
                reference=None;previous_mask=None;reference_key=event.get('key')
            elif event.get('key')!=reference_key:
                reference=None;previous_mask=None;reference_key=event.get('key')
            regions=[]
            for row in event['regions']:
                x,y,w,h=shape_bounds(row['shape'])
                regions.append(GameHitRegionV240(row['object_id'],x,y,w,h,metadata=dict(v250_object_snapshot=dict(
                    shape=row['shape'],entity_id=row.get('entity_id',row['object_id'])))))
            viewport=tuple(event['viewport']);mask=glow_mask(regions,viewport[2:],cfg.radius_px)
            if previous_mask is not None:mask &= previous_mask
            previous_mask=mask
            if hashlib.sha256(mask.tobytes()).hexdigest()!=event['mask_sha256']:raise ValueError('Recorded projector mask does not reproduce')
            frames=[];a,b=on-.6,off+.20
            while current is not None:
                acq=current.get('acquisition') or {};ts=float(acq['receive_wall_ts'])
                if ts>b:break
                if ts<previous_ts:raise ValueError('Nonmonotonic camera receive timestamps')
                previous_ts=ts
                if ts>=a:
                    handle.seek(int(current['offset']));data=handle.read(int(current['length']))
                    if len(data)!=int(current['length']):raise ValueError('Truncated camera bytes')
                    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                    if image is None:raise ValueError('Undecodable recorded MJPG')
                    image=apply_frame_transform(image,int(acq.get('rotation',0) or 0),bool(acq.get('mirror_horizontal')),bool(acq.get('mirror_vertical')))
                    frames.append(N(timestamp=ts,frame_bgr=image))
                current=next(rows,None)
            if not event.get('reference') and reference is None:
                result=dict(valid=False,reason='no_valid_recorded_reference')
            else:
                result=measure_glow(frames,on,off,reference,mask,viewport,projection_from_state(event['projection']),
                    cfg,flash,cutoff=live.get('cutoff_peak'),is_reference=bool(event.get('reference')))
            image=result.pop('image',None);result.pop('camera_valid',None)
            if result.get('valid'):reference=image
            result.update(on=on,off=off,reference=bool(event.get('reference')),live_valid=live.get('valid'),
                live_holes=live.get('holes',[]),live_reason=live.get('reason'))
            results.append(result)
    summary=dict(kind='RECORDED_LOCAL_GLOW_MEASUREMENT_REPLAY_NOT_FULL_GAMEPLAY_OR_PHYSICAL_GT',windows=len(results),
        valid=sum(bool(r.get('valid')) for r in results),
        source_manifest_sha256=hashlib.sha256((session/'session.json').read_bytes()).hexdigest(),
        recipes_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('src/engine/precision_object_glow.py','src/engine/precision_flash.py','automation/object_glow_replay.py')})
    with (out/'results.jsonl').open('x') as f:
        for r in results:f.write(json.dumps(r)+'\n')
    (out/'metrics.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('output');a=p.parse_args()
    print(json.dumps(replay(a.session,a.output),indent=2))
