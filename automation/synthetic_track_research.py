"""Deterministic offline shooting through project renderer and detector/tracker.

Uses synchronous worker execution with the real FAST worker thread name. Camera
and audio transport are replaced by explicit arrays/event times. No AI or GT is
passed to the scanner. This is a component-integrated synthetic test, not full
hardware/live-path validation. Historical labels are never read.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from src.engine.camera.hit_scanner import HitScanner, AudioShotEvent, ScanportFrame
from src.engine.camera.analysis_geometry_v2221 import AnalysisGeometryV2221
from src.engine.camera.hit_scanner_v2221 import install_v2221_hit_scanner_patch
from src.engine.camera.hit_scanner_v2222 import install_v2222_hit_scanner_patch
from src.engine.shot_fast_v2225 import _install_fast_extractor_patch, local_confirm_candidates_v2225
from src.engine.shot_track_v2226 import update_tracks_frame_unique_v2226
from src.engine.synthetic.synthetic_hole_overlay import SyntheticHoleOverlay
from src.engine.track_audit import capture
from src.engine.offline.track_replay import current_exact_replay
from automation.track_survival_replay import choose, distance, metrics

SCENES=('light','dark','real_dart','old_holes','near_old','sequential_nearby','target_edge','tape_lines','texture','low_contrast','strong_contrast','jitter','illumination_small','illumination_large','delayed_result','false_audio','unchanged')
FROZEN_POLICY=dict(name='SIGNED_LOCAL_CONTRAST',status='RESEARCH_ONLY',formula='confirm_darkening - confirm_ring_abs',tie_break='existing live rank key',development_seed=20260908,holdout_seed=20260909)
POLICY_HASH=hashlib.sha256(json.dumps(FROZEN_POLICY,sort_keys=True).encode()).hexdigest()


def make_images(real,scene,seed,post_background=None):
    rng=np.random.default_rng(seed); h,w=real.shape
    if scene=='light': base=np.full((h,w),200,np.uint8)
    elif scene=='dark': base=np.full((h,w),45,np.uint8)
    elif scene=='texture': base=np.clip(130+rng.normal(0,30,(h,w)),0,255).astype(np.uint8)
    else: base=real.copy()
    point=(int(rng.integers(32,w-32)),int(rng.integers(32,h-32)))
    if scene=='target_edge': point=(40, h//2); cv2.rectangle(base,(40,20),(w-40,h-20),35,2)
    if scene=='tape_lines':
        for x in range(35,w,60): cv2.line(base,(x,0),(x+50,h-1),230,5)
    old=SyntheticHoleOverlay(w,h,rng_seed=seed)
    if scene in ('old_holes','near_old','sequential_nearby'):
        for i in range(20): old.add_hole(int(rng.integers(20,w-20)),int(rng.integers(20,h-20)),radius_px=3,hole_id=str(i))
        if scene in ('near_old','sequential_nearby'): old.add_hole(point[0]-8,point[1],radius_px=3,hole_id='near')
    base=cv2.cvtColor(old.composite_on(cv2.cvtColor(base,cv2.COLOR_GRAY2BGR)),cv2.COLOR_BGR2GRAY)
    overlay=SyntheticHoleOverlay(w,h,rng_seed=seed+1)
    if scene not in ('unchanged','false_audio'):
        overlay.add_hole(*point,radius_px=3,opacity=.24 if scene=='low_contrast' else .98,hole_id='new')
    later_base=base
    if post_background is not None and scene in ('real_dart','old_holes','near_old','sequential_nearby','low_contrast','strong_contrast','delayed_result','false_audio'):
        later_base=cv2.cvtColor(old.composite_on(cv2.cvtColor(post_background,cv2.COLOR_GRAY2BGR)),cv2.COLOR_BGR2GRAY)
    post=cv2.cvtColor(overlay.composite_on(cv2.cvtColor(later_base,cv2.COLOR_GRAY2BGR)),cv2.COLOR_BGR2GRAY)
    if scene=='jitter': post=cv2.warpAffine(post,np.float32([[1,0,.8],[0,1,-.6]]),(w,h),borderMode=cv2.BORDER_REFLECT)
    if scene.startswith('illumination'): post=np.clip(post.astype(float)+(3 if scene=='illumination_small' else 18),0,255).astype(np.uint8)
    return base,post,None if scene in ('unchanged','false_audio') else dict(camera_x=point[0]+(.8 if scene=='jitter' else 0),camera_y=point[1]-(.6 if scene=='jitter' else 0))


def geometry(shape,crop):
    h,w=shape;ox,oy=(17,13) if crop else (0,0);x1,y1=(w-17,h-13) if crop else (w,h)
    poly=np.float32([[ox,oy],[x1-1,oy],[x1-1,y1-1],[ox,y1-1]])
    return AnalysisGeometryV2221(h,w,ox,oy,x1,y1,poly,poly,np.full((y1-oy,x1-ox),255,np.uint8),'synthetic_explicit_crop',0,0)


def prepare():
    install_v2221_hit_scanner_patch();install_v2222_hit_scanner_patch();_install_fast_extractor_patch()
    # Settings are read, never written. The geometry transport is deterministic.
    from src.engine.camera import hit_scanner_v2222 as cleanup
    from src.engine.camera import hit_scanner_v2221 as crop
    crop._build_geometry_from_live_settings=lambda s,shape:s._synthetic_geometry
    cleanup._screen_rect_and_homographies=lambda:((0,0,640,480),np.eye(3),np.eye(3))


def execute(pre,post,sid,executor,crop=True,tracing=True,scanner=None,delivery_delay=.15,confirmation_pre=None):
    confirmation_pre=pre if confirmation_pre is None else confirmation_pre
    s=scanner or HitScanner();s.physical_trace_capture_enabled=tracing
    peak=1000+sid*4.;event=AudioShotEvent(sid,peak,peak);s.audio_events.append(event)
    s.last_audio_event_ts=peak;s._diag_shot_id=sid;s._diag_frame_count=0
    s.scene_reference_gray=pre.copy();s.pre_shot_snapshot=pre.copy();s.pre_shot_snapshot_ts=peak-.1
    s.frame_history.clear()
    for dt in (-.45,-.3,-.2,-.1,-.02):s.frame_history.append(ScanportFrame(peak+dt,pre.copy()))
    s._synthetic_geometry=geometry(pre.shape,crop)
    s.last_trace_pipeline_shot_id=sid
    s._candidate_generator_v2_engine.reset_runtime_state()
    start=time.perf_counter()
    proposal=executor.submit(s._detect_frame_candidates,post,peak+.08).result()
    assert not s.last_window_debug.get('v2_fallback'), 'synthetic detector fell back unexpectedly'
    detector_ms=(time.perf_counter()-start)*1000
    # Use actual result candidates (apply_result's diagnostic ownership copy is
    # intentionally not substituted for runtime candidates).
    from src.engine.shot_async_v2224 import tracking_frame_timestamp, AsyncDetectorV2224, DetectorJobResultV2224
    job=DetectorJobResultV2224(sid,peak+.08,peak,start,start,time.perf_counter(),proposal,{},dict(s.last_window_debug),s.last_threshold_value,s.last_change_threshold_value,s.last_vote_threshold_value,trace_pipeline=getattr(s,"last_trace_pipeline",None))
    AsyncDetectorV2224.apply_result(s,job)
    proposal=job.candidates
    evidence_ts=tracking_frame_timestamp(s,proposal,peak+delivery_delay)
    assert evidence_ts==peak+.08
    update_tracks_frame_unique_v2226(s,proposal,evidence_ts)
    confirm_ts=peak+max(.6,delivery_delay+.02)
    confirmed,diag=local_confirm_candidates_v2225(confirmation_pre,post,proposal,frame_ts=confirm_ts)
    update_tracks_frame_unique_v2226(s,confirmed,confirm_ts)
    selected=s._best_track_for_event(event)
    decision_ts=confirm_ts
    ready=selected is not None and s._track_is_ready(selected,decision_ts,event)
    if not ready and confirmed:
        confirmed,diag=local_confirm_candidates_v2225(confirmation_pre,post,confirmed,frame_ts=confirm_ts+.05)
        update_tracks_frame_unique_v2226(s,confirmed,confirm_ts+.05)
        selected=s._best_track_for_event(event)
        decision_ts=confirm_ts+.05
        ready=selected is not None and s._track_is_ready(selected,decision_ts,event)
    snap=capture(s,event,selected) if tracing else None
    if snap: current_exact_replay(snap)
    return dict(snapshot=snap,proposal=proposal,confirmed=confirmed,detector_ms=detector_ms,
                selected=selected,ready=ready,scanner=s,event=event,decision_ts=decision_ts)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=102);p.add_argument('--split',choices=('development','holdout'),default='development');p.add_argument('--frozen-manifest',type=Path);a=p.parse_args()
    if a.split=='holdout':
        if not a.frozen_manifest or json.loads(a.frozen_manifest.read_text())['policy_hash']!=POLICY_HASH:
            raise ValueError('Matching frozen development manifest required before holdout')
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(input_session=str(a.root),source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),Path('src/engine/shot_track_v2226.py'),Path('src/engine/shot_async_v2224.py'),Path('src/engine/shot_fast_v2225.py'))},policy=FROZEN_POLICY,policy_hash=POLICY_HASH,split=a.split,count=a.count,scenes=SCENES,
        limitations=['deterministic worker completion; not hardware scheduling','dark overlay renderer is not bullet damage physics','scene crops are derived from real PRE images; no physical labels used'])
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    paths=sorted((a.root/'shots').glob('*/trace.json')); backgrounds=[]
    for path in paths:
        t=json.loads(path.read_text());fs=sorted([f for f in t['frames'] if f['kind']=='pre_history' and t['peak_ts']-.32<=f['timestamp']<t['peak_ts']],key=lambda f:f['timestamp'])
        assert len(fs)>=2
        backgrounds.append(tuple(np.array(np.load(path.parent/f['path'],mmap_mode='r')[1030:1510,1800:2440]) for f in (fs[0],fs[-1])))
    assert backgrounds and backgrounds[0][0].shape==(480,640)
    rows=[];prepare();seed=FROZEN_POLICY[a.split+'_seed']
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='shot-cv-v2224') as executor, open(a.output/'runtime.log','w') as log, contextlib.redirect_stdout(log):
        for i in range(a.count):
            scene=SCENES[i%len(SCENES)];pre,post,gt=make_images(backgrounds[i%len(backgrounds)][0],scene,seed+i,backgrounds[i%len(backgrounds)][1])
            result=execute(pre,post,i+1,executor,crop=i%2==0,delivery_delay=1.2 if scene=='delayed_result' else .15)
            snap=result['snapshot'];proposal=result['proposal'];tracks=snap['tracks'];eligible=[r for r in tracks if r['eligible']]
            def near(cs): return min((distance(c,gt) for c in cs),default=None) if gt else None
            oracle=near(proposal);td=near(tracks);cd=near([r for r in tracks if r['local_confirmed']]);ed=near(eligible)
            choices={}
            for mode in ('CURRENT','SIGNED_LOCAL_CONTRAST'):
                picked=choose(snap,mode)
                ready=picked is not None and result['scanner']._track_is_ready(result['scanner']._active_tracks[picked['track_id']],result['decision_ts'],result['event'])
                choices[mode]=dict(error=distance(picked,gt) if picked and gt and ready else None,source=picked['source'] if picked else None,track_id=picked['track_id'] if picked else None)
            loss='selected'
            if gt:
                for stage,d in [('generation',oracle),('tracking',td),('local_confirmation',cd),('eligibility',ed),('final_ranking',choices['CURRENT']['error'])]:
                    if d is None or d>42:loss=stage;break
            row=dict(index=i,scene=scene,truth=gt,oracle=oracle,tracked_distance=td,confirmed_distance=cd,eligible_distance=ed,choices=choices,loss=loss,ready=result['ready'],detector_ms=result['detector_ms'],tracks=len(tracks),associations=sum(len(b['records']) for b in snap['associations']))
            encoded=json.dumps(snap,separators=(',',':'));row['trace_bytes']=len(encoded.encode())
            (a.output/f'track_{i:04}.json').write_text(encoded+'\n');rows.append(row)
    impacts=[r for r in rows if r['truth']];false=[r for r in rows if not r['truth']]
    summary=dict(impact_count=len(impacts),false_events=len(false),false_emissions=sum(r['ready'] for r in false),
        candidate_oracle=metrics([r['oracle'] for r in impacts],len(impacts)),
        policies={mode:metrics([r['choices'][mode]['error'] for r in impacts],len(impacts)) for mode in ('CURRENT','SIGNED_LOCAL_CONTRAST')},
        funnel={s:sum(r[k] is not None and r[k]<=42 for r in impacts) for s,k in [('generated','oracle'),('tracked','tracked_distance'),('local_confirmed','confirmed_distance'),('eligible','eligible_distance')]},
        scene_results={scene:dict(total=sum(r['scene']==scene and bool(r['truth']) for r in rows),current_hits=sum(r['scene']==scene and r['truth'] is not None and r['choices']['CURRENT']['error'] is not None and r['choices']['CURRENT']['error']<=42 for r in rows)) for scene in SCENES},
        overhead=dict(mean_tracks=float(np.mean([r['tracks'] for r in rows])),max_tracks=max(r['tracks'] for r in rows),mean_associations=float(np.mean([r['associations'] for r in rows])),mean_json_bytes=float(np.mean([r['trace_bytes'] for r in rows])),mean_detector_ms=float(np.mean([r['detector_ms'] for r in rows]))))
    (a.output/'results.json').write_text(json.dumps(dict(manifest=manifest,summary=summary,rows=rows),indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
