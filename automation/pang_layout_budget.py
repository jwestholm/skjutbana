"""Software-only OBJECT region budget against a SAVED camera calibration; no capture.
python3 -m automation.pang_layout_budget SESSION NEW.json --seeds 30
"""
import argparse,contextlib,hashlib,io,json,os
from pathlib import Path


def measure(session,seeds):
    os.environ['SDL_VIDEODRIVER']='dummy';os.environ['SDL_AUDIODRIVER']='dummy'
    os.environ['SKJUTBANA_PHYSICAL_TRACE_DISABLE']='1'
    import numpy as np
    import pygame
    from types import SimpleNamespace as NS
    from content.games.light_targets_game import LightTargetsGame
    from src.engine import lens_model
    from src.engine.input.object_hit_v2223 import transform_game_rect_to_camera_aabb,ObjectShotSnapshotV2223
    from src.engine.camera.object_physical_query import ObjectPhysicalQuery,MAX_PIXELS
    session=Path(session);raw=(session/'session.json').read_bytes()
    settings=json.loads(raw)['settings']['content/settings.json']['content'];cal=settings['camera_calibration']
    model=lens_model.parse(cal);H=np.array(cal['homography'])
    def to_camera(x,y):
        if model is not None:return lens_model.screen_to_camera(model,x,y)
        p=H@np.array([x,y,1.]);return tuple(p[:2]/p[2])
    with (session/'camera_index.jsonl').open() as f:shape=json.loads(next(f))['acquisition']['frame_shape'][:2]
    pygame.font.init();pre=np.zeros(shape,np.uint8);rows=[]
    old_model=os.environ.pop('SKJUTBANA_OBJECT_DENSE_MODEL',None)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            game=LightTargetsGame('',pygame.Rect(0,0,*settings['viewport'][2:]));game.on_enter()
            try:
                for seed in range(seeds):
                    game.rng.seed(seed);game._next_target()
                    if game.phase=='unsupported':raise ValueError('Layout unavailable for seed '+str(seed))
                    game.phase='live';regions=game.get_hit_regions()
                    camera=tuple(transform_game_rect_to_camera_aabb(r,settings['viewport'],to_camera) for r in regions)
                    snap=ObjectShotSnapshotV2223(seed,2.,2.,(),game_regions=regions,camera_regions=camera,
                        transform_available=True,viewport_xywh=tuple(settings['viewport']))
                    query=ObjectPhysicalQuery(snap,[NS(frame_bgr=pre,timestamp=1.94)])
                    rows.append(dict(seed=seed,objects=len(game.items),regions=len(regions),read_pixels=query.read_pixels,
                                     reason=(query.result or {}).get('reason')))
                asset_sha=game.art.manifest_sha256
            finally:game.on_exit()
    finally:
        if old_model is not None:os.environ['SKJUTBANA_OBJECT_DENSE_MODEL']=old_model
    return dict(kind='SAVED_CALIBRATION_GEOMETRY_BUDGET_NOT_ACCURACY_OR_LIVE_LATENCY',
                session_sha256=hashlib.sha256(raw).hexdigest(),asset_manifest_sha256=asset_sha,
                camera_shape=shape,pixel_limit=MAX_PIXELS,max_read_pixels=max(r['read_pixels'] for r in rows),
                passed=all(r['reason'] is None and r['read_pixels']<=MAX_PIXELS for r in rows),rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('output');p.add_argument('--seeds',type=int,default=30);a=p.parse_args()
    if a.seeds<1:raise ValueError('seeds must be positive')
    out=Path(a.output)
    if out.exists():raise FileExistsError('Choose a new output')
    result=measure(a.session,a.seeds)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print('Passed:',result['passed'],'max ROI:',result['max_read_pixels'],'/',result['pixel_limit'])
