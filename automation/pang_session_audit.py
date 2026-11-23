"""Exploratory saved-run review: query-math replay + white-photo review spots.

Not live scheduling replay, automated GT or a trained detector. White-photo spot
threshold is for human review only. All measurement uses saved session calibration.
python3 -m automation.pang_session_audit SESSION LOG WHITE.png NEW_OUTPUT_DIR
"""
from pathlib import Path
import argparse,hashlib,json,os,tempfile
from types import SimpleNamespace as NS
import cv2
import numpy as np
from automation.can_dense_offline import Frames,load_shots,mapper
from src.engine import lens_model
from src.engine.camera.detector_luma import detector_gray
from src.engine.camera.object_physical_query import ObjectPhysicalQuery
from src.engine.game_objects.geometry import shape_contains
from src.engine.input.object_hit_v2223 import ObjectShotSnapshotV2223,GameHitRegionV240,CameraHitRegionV240


def camera_grid(settings):
    vx,vy,w,h=settings['viewport'];cal=settings['camera_calibration'];model=lens_model.parse(cal)
    x,y=np.meshgrid(np.arange(w),np.arange(h));grid=np.column_stack((x.ravel()+vx,y.ravel()+vy)).astype(float)
    if model is not None:
        uv=cv2.perspectiveTransform(grid.reshape(-1,1,2),model['Hinv']).reshape(-1,2)
        points=lens_model.distort_points(uv,model)
    else:points=cv2.perspectiveTransform(grid.reshape(-1,1,2),np.array(cal['homography'])).reshape(-1,2)
    return points.reshape(h,w,2)


def review_spots(white,grid,to_game):
    gray=detector_gray(white).astype(np.float32);mask=np.zeros(gray.shape,np.uint8)
    edge=np.concatenate([grid[0],grid[:,-1],grid[-1,::-1],grid[::-1,0]])
    cv2.fillPoly(mask,[np.rint(edge).astype(np.int32)],255);mask=cv2.erode(mask,np.ones((9,9),np.uint8))
    response=cv2.GaussianBlur(gray,(0,0),4)-cv2.GaussianBlur(gray,(0,0),.8)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(((response>7)&(mask>0)).astype(np.uint8))
    points=[]
    for k in range(1,count):
        if not 2<=stats[k,cv2.CC_STAT_AREA]<=200:continue
        x,y=centers[k];strength=float(response[labels==k].max())
        points.append(dict(camera_xy=[float(x),float(y)],game_xy=[float(v) for v in to_game(x,y)],strength=strength,area=int(stats[k,cv2.CC_STAT_AREA])))
    points.sort(key=lambda p:-p['strength'])
    for i,p in enumerate(points):p['review_id']=i
    return points


def spot_change(pre,post,xy):
    x,y=map(round,xy);a=pre[y-5:y+6,x-5:x+6];b=post[y-5:y+6,x-5:x+6]
    yy,xx=np.mgrid[-5:6,-5:6];r=np.hypot(xx,yy);core=r<=1.5;ring=(r>=3)&(r<=5)
    dark=float((a-b)[core].mean()-(a-b)[ring].mean())
    return dict(signed_spot_dn=dark,relative=dark/max(1.,float(a[core].mean())))


def white_spot(gray,xy):
    """Post-hoc photo review only; never passed into the detector."""
    x,y=map(round,xy);patch=gray[y-8:y+9,x-8:x+9]
    yy,xx=np.mgrid[-8:9,-8:9];r=np.hypot(xx,yy)
    return float(patch[(r>=5)&(r<=8)].mean()-patch[r<=1.5].mean())


def audit(session,log,white_path,out):
    session,out=Path(session),Path(out);out.mkdir(parents=True,exist_ok=False)
    settings=json.loads((session/'session.json').read_text())['settings']['content/settings.json']['content']
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'settings.json';p.write_text(json.dumps(settings));to_game=mapper(p)
    grid=camera_grid(settings);white=cv2.imread(str(white_path));points=review_spots(white,grid,to_game)
    white_gray=detector_gray(white).astype(float)
    for p in points:p['white_photo_spot_dn']=white_spot(white_gray,p['camera_xy'])
    picture=cv2.remap(white,grid[:,:,0].astype(np.float32),grid[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
    for p in points:
        x,y=map(round,p['game_xy']);cv2.circle(picture,(x,y),9,(0,0,255),1)
        cv2.putText(picture,str(p['review_id']),(x+8,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
    cv2.imwrite(str(out/'white_review.png'),picture)
    rows=[];frames=Frames(session);shots=load_shots(log);presented={};contexts={}
    for line in Path(log).read_text().splitlines():
        if line.startswith('[LAMP-PRESENTED] '):
            p,_=json.JSONDecoder().raw_decode(line[len('[LAMP-PRESENTED] '):]);presented[p['owner'],p['token']]=p['wall_ts']
        elif line.startswith('[LAMP-SHOT] '):
            p,_=json.JSONDecoder().raw_decode(line[len('[LAMP-SHOT] '):]);contexts[int(p['shot_id'])]=(p['owner'],p['token'])
    old_model=os.environ.pop('SKJUTBANA_OBJECT_DENSE_MODEL',None)
    try:
        for sid,shot in sorted(shots.items()):
            next_peak=min((s['peak_ts'] for s in shots.values() if s['peak_ts']>shot['peak_ts']+.15),default=float('inf'))
            if next_peak<shot['peak_ts']+1.2:raise ValueError('Audit requires disjoint shot windows, not double shots')
            pi=[int(np.argmin(abs(frames.ts-t))) for t in shot['pre_frame_ts']]
            if any(abs(frames.ts[i]-t)>.002 for i,t in zip(pi,shot['pre_frame_ts'])):raise ValueError('Missing exact PRE')
            qi=list(np.where((frames.ts>=shot['peak_ts']+.35)&(frames.ts<=shot['peak_ts']+1.2))[0][:3])
            if len(qi)<3:raise ValueError('Missing POST')
            pres=[frames.gray(i) for i in pi];posts=[frames.gray(i) for i in qi]
            pre=np.median(np.stack(pres),0);post=np.median(np.stack(posts),0)
            game=[]
            for p in shot['per_object']:
                game.append(GameHitRegionV240(p['object_id'],*p['game_aabb'],metadata=dict(
                    v250_game_object=True,v250_object_snapshot=dict(shape=p['shape']),
                    physical_query=dict(evidence_method='dense',result_semantics='OBJECT_HIT'))))
            cam=tuple(CameraHitRegionV240(p['object_id'],*p['camera_aabb']) for p in shot['frozen_regions'])
            snap=ObjectShotSnapshotV2223(sid,shot['peak_ts'],shot['peak_ts'],(),game_regions=tuple(game),camera_regions=cam,
                transform_available=True,transform_method='saved_session',viewport_xywh=tuple(settings['viewport']))
            q=ObjectPhysicalQuery(snap,[NS(frame_bgr=a,timestamp=float(frames.ts[i])) for a,i in zip(pres,pi)],camera_to_game=to_game)
            for i in np.where((frames.ts>=shot['peak_ts'])&(frames.ts<=shot['peak_ts']+1.2))[0]:
                if q.final:break
                q.observe(NS(frame_bgr=frames.gray(int(i)),timestamp=float(frames.ts[i])),delivered_ts=float(frames.ts[i])+.001)
            contrasts=[dict(review_id=p['review_id'],**spot_change(pre,post,p['camera_xy']),
                            nearest_camera_aabb_distance_px=min(np.hypot(max(r.x-p['camera_xy'][0],0,p['camera_xy'][0]-r.right),max(r.y-p['camera_xy'][1],0,p['camera_xy'][1]-r.bottom)) for r in cam),
                            inside_active_parts=[r.object_id for r in game if shape_contains(r.metadata['v250_object_snapshot']['shape'],*p['game_xy'])]) for p in points]
            stability=[]
            for region in cam:
                x,y,w,h=map(round,(region.x,region.y,region.width,region.height))
                stack=np.stack([a[y:y+h,x:x+w] for a in pres]);med=np.median(stack,0)
                stability.append(dict(object_id=region.object_id,pre_dn=float(med.mean()),
                    pre_p99_absolute_variation_dn=float(np.quantile(abs(stack-med),.99))))
            rows.append(dict(shot_id=sid,live_object_id=shot.get('object_id'),query_object_id=(q.result or {}).get('object_decision',{}).get('object_id'),
                query_reason=(q.result or {}).get('reason'),query_peaks=[dict(p,white_photo_spot_dn=white_spot(white_gray,(p['cx'],p['cy'])),
                    distance_to_nearest_white_review_spot_px=min(np.hypot(p['cx']-v['camera_xy'][0],p['cy']-v['camera_xy'][1]) for v in points) if points else None) for p in getattr(q,'dense_peaks',[])[:4]],
                seconds_since_presented=shot['peak_ts']-presented[contexts[sid]] if contexts.get(sid) in presented else None,
                stability=stability,review_spot_changes=contrasts))
            frames.cache.clear()
    finally:
        frames.fh.close()
        if old_model is not None:os.environ['SKJUTBANA_OBJECT_DENSE_MODEL']=old_model
    for p in points:
        strongest=max(rows,key=lambda r:r['review_spot_changes'][p['review_id']]['signed_spot_dn'])
        p['suggested_onset_shot']=strongest['shot_id'];p['evidence_kind']='WHITE_PHOTO_REVIEW_SPOT_NOT_AUTOMATIC_GT'
    result=dict(kind='EXPLORATORY_QUERY_MATH_REPLAY_NOT_LIVE_SCHEDULING',session=str(session),
                inputs_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (session/'session.json',session/'camera_index.jsonl',Path(log),Path(white_path))},
                query_code_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (Path(__file__),Path('src/engine/camera/object_physical_query.py'),Path('src/engine/shot_dense_h15.py'))},
                future_photo_fields='POST_HOC_REVIEW_ONLY_NEVER_TRAINING_OR_LIVE_FEATURES',agreement=sum(r['query_object_id']==r['live_object_id'] for r in rows),n=len(rows),review_spots=points,rows=rows)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('session','log','white','out'):p.add_argument(name)
    a=p.parse_args();r=audit(a.session,a.log,a.white,a.out);print('Query agreement:',r['agreement'],'/',r['n'],'white review spots:',len(r['review_spots']))
