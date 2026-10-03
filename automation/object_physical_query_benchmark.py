"""Complete OBJECT crop-path software timing; synthetic frames, NOT accuracy.

No camera, audio capture, settings writes or physical trace reads. Full-frame
fixture allocation is outside timings; all query PRE preparation, crop conversion,
registration, candidate extraction and persistence work is timed.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace as S

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
import numpy as np
import pygame

from content.games.can_shootout_game import ShootStuffGame,hit_sound_wave,GAME_PROJECTILE
from src.engine.game_objects import ObjectManager
from src.engine.events import EventBus
from src.engine.audio.audio_peak_detector import AudioPeakDetector
from src.engine.camera.object_physical_query import ObjectPhysicalQuery
from src.engine.input.object_hit_v2223 import ObjectShotSnapshotV2223,transform_game_rect_to_camera_aabb


def stats(values):
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),
                p95=float(np.percentile(values,95)),maximum=float(max(values)))


def run(output,repeats):
    output.mkdir(parents=True,exist_ok=False)
    pygame.init();pygame.display.set_mode((800,600))
    game=ShootStuffGame('',pygame.Rect(0,0,800,600));game.on_enter()
    rng=np.random.default_rng(253)
    pre=np.repeat(rng.integers(125,155,(2160,3840),dtype=np.uint8)[:,:,None],3,axis=2)
    frames=[S(timestamp=t,frame_bgr=pre) for t in (1.84,1.90,1.94)]
    results=[]
    audio=AudioPeakDetector()
    for distance,count,layout in ((20,1,'rack'),(20,7,'rack'),(20,14,'overlap'),
                                  (20,14,'grid'),(20,25,'grid'),(25,7,'rack'),
                                  (50,7,'rack'),(20,25,'grid_clutter')):
        game.distance_m=distance;game._build_rack();game.phase='live'
        originals=list(game.manager.objects)
        templates=[]
        for i in range(count):
            obj=deepcopy(originals[i%7])
            if i>=7:obj.object_id+=f'_{i}'
            if layout.startswith('grid'):
                obj.geometry.x=60+(i%5)*150
                obj.geometry.y=60+(i//5)*105
            templates.append(obj)
        def manager_for_sample():
            manager=ObjectManager()
            for obj in templates:manager.add(deepcopy(obj))
            return manager
        manager=manager_for_sample()
        regions=manager.get_hit_regions()
        camera=tuple(transform_game_rect_to_camera_aabb(r,(0,0,800,600),lambda x,y:(x*4.8,y*3.6)) for r in regions)
        snap=ObjectShotSnapshotV2223(1,2.,2.,(),game_regions=regions,camera_regions=camera,transform_available=True)
        c=camera[0];x,y=round(c.x+c.width/2),round(c.y+c.height/2)
        post=pre.copy();post[y-2:y+3,x-2:x+3]=30
        if layout=='grid_clutter':
            for region in camera:
                for py in range(int(region.y),int(region.bottom),12):
                    for px in range(int(region.x),int(region.right),12):
                        post[py:py+3,px:px+3]=30
        times=[];observations=[]
        for i in range(repeats+2):
            manager=manager_for_sample();manager._snapshot_provider=lambda sid:snap
            bus=EventBus();delivery={}
            def consume(message):
                started=time.perf_counter()
                resolution,_=manager.resolve_object_query(message['data'],GAME_PROJECTILE)
                delivery['exact_resolution_ms']=(time.perf_counter()-started)*1000.
                started=time.perf_counter()
                if resolution is not None:
                    audio.play_feedback(game._hit_sounds['soda_can'],owner='benchmark',shot_id=i+1,cue='soda_can')
                delivery['sfx_dispatch_ms']=(time.perf_counter()-started)*1000.
            bus.subscribe(consume)
            start=time.perf_counter()
            q=ObjectPhysicalQuery(snap,frames,camera_to_game=lambda x,y:(x/4.8,y/3.6))
            prepared=time.perf_counter()
            delivered_first=max(2.035,2.+prepared-start)
            q.observe(S(timestamp=2.033,frame_bgr=post),delivered_ts=delivered_first)
            first=time.perf_counter()
            delivered_second=max(2.068,delivered_first+first-prepared)
            q.observe(S(timestamp=2.066,frame_bgr=post),delivered_ts=delivered_second)
            end=time.perf_counter()
            if not q.final:q.finish_deadline(2.4)
            dispatch=time.perf_counter();bus.emit('physical.object_result',q.result,source='OBJECT')
            bus_ms=(time.perf_counter()-dispatch)*1000.
            if i>=2:
                times.append(dict(prepare_ms=(prepared-start)*1000,first_frame_ms=(first-prepared)*1000,
                    second_frame_ms=(end-first)*1000,processing_total_ms=(end-start)*1000,
                    roi_prepare_ms=q.roi_prepare_ms,pre_prepare_ms=q.pre_prepare_ms,
                    merge_dedupe_total_ms=sum(q.merge_dedupe_ms),event_bus_total_ms=bus_ms,
                    event_bus_overhead_ms=max(0,bus_ms-sum(delivery.values())),**delivery))
                observations.append(q.result)
        results.append(dict(distance_m=distance,active_regions=count,layout=layout,repeats=repeats,
            processed_crop_pixels=q.read_pixels,searched_pixels=q.search_pixels,normal_fixture_pixels=pre.shape[0]*pre.shape[1],
            scan_fraction=q.search_pixels/(pre.shape[0]*pre.shape[1]),
            normal_scan_basis='Full-camera affine viewport fixture, not measured live calibrated ROI',
            candidate_counts=[r['candidate_count'] for r in observations],
            examined_candidate_counts=[r['examined_candidate_count'] for r in observations],
            timings={key:stats([t[key] for t in times]) for key in times[0]},samples=times,
            status_counts={s:sum(r['status']==s for r in observations) for s in ('PHYSICAL_HIT','NO_PHYSICAL_HIT','UNCERTAIN')}))
    import wave
    for target in ('rubber_duck','soda_can','beans_xxl','sardines','toilet_roll'):
        with wave.open(str(output/(target+'.wav')),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(22050);f.writeframes(hit_sound_wave(target,22050).tobytes())
    game.on_exit()
    report=dict(status='SOFTWARE_BENCHMARK_NOT_PHYSICAL_VALIDATION',cases=results,
        camera_wait_and_real_audio_to_gameplay_latency='NOT_MEASURED',
        blocking='Synchronous bounded crop CPU only; no capture_fresh_frame, sleeps, disk reads or waiting on global CV',
        delivery_semantics='Real ObjectManager and EventBus with one resolver/SFX subscriber; no rendering, camera wait or microphone. SDL dummy output.',
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path('src/engine/camera/object_physical_query.py'),Path('content/games/can_shootout_game.py')]})
    (output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    for case in results:
        print(case['distance_m'],case['active_regions'],case['layout'],'pixels',case['searched_pixels'],
            'fraction',round(case['scan_fraction'],4),'timing',case['timings'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=20)
    args=parser.parse_args()
    if args.repeats<1:parser.error('repeats must be positive')
    run(args.output,args.repeats)
